"""Activity awareness: a memory-only log of what the user does during a session.

While a session is active, a small poller (~4 Hz) watches the frontmost app, the
focused window, open dialogs and the focused UI element through the Accessibility
API; a mouse listener notes clicks and scrolls (and which element was under the
pointer); a keyboard listener notes *that* the user is typing. Changes become
timestamped events in a ring buffer. No API calls are made here.

Hard rules (tests in the scratchpad enforce them):
- Never which keys: the keyboard listener only reports "typing" or "Return/Tab".
  What the user typed is read from the focused field's AXValue when they finish
  (focus leaves, Return/Tab, or a ~1.2s pause), so autofill, paste and
  backspace are all reflected correctly.
- Secure/password fields are never read. Only "typed in a password field
  (content not read)" is recorded. A field whose AXValue can't be read as text is
  treated the same way (content not read).
- Field content is redacted when captured (long digit runs, SSN-like numbers,
  secret-looking tokens become "[redacted]"; emails and ordinary words stay) and
  truncated to ~200 chars, so nothing unredacted is ever stored or rendered.
- Memory only. `stop()` clears everything. SHERPA_ACTIVITY=0 disables it all.

Sources (AX polling, mouse, keyboard) sit behind the `Sources` protocol so tests
drive the log with fake events. Probe from a trusted terminal:
    python -m sherpa.activity [--show-content] [--seconds N]
"""

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from . import config

POLL_INTERVAL_S = 0.25
TYPING_PAUSE_S = 1.2  # idle this long after typing => the field is "finished"
MAX_EVENTS = 400
FIELD_CHARS = 200
READ_CHARS = 4000  # never pull more than this out of a field before trimming
TIMELINE_MAX_CHARS = 2400
EVENT_MAX_CHARS = 240
FOCUS_FLAP_S = 1.5
SCROLL_MERGE_S = 1.5
FIELD_MERGE_S = 30.0
TYPING_MERGE_S = 5.0

SECURE_NOTE = "typed in a password field (content not read)"
KINDS = (
    "app_switched", "window_changed", "dialog_opened", "focus_changed", "clicked",
    "typing_started", "typing_stopped", "field_finished", "scrolled",
)
EDITABLE_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
DIALOG_SUBROLES = {"AXDialog", "AXSystemDialog"}


# ---------------------------------------------------------------- redaction
_DIGITS = re.compile(r"(?<!\d)\d(?:[ -]?\d){11,}(?!\d)")  # 12+ digits, optionally spaced/dashed
_SSN = re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)")
_TOKEN = re.compile(r"(?<![\w@.\-/:+=])[A-Za-z0-9_\-+/=]{24,}(?![\w@\-+/=])(?!\.\w)")
_PREFIXED = re.compile(r"(?<![\w@])(?:sk-|pk_|rk_|ghp_|gho_|github_pat_|xox[abprs]-|AKIA|AIza|ya29\.)[A-Za-z0-9_\-.]{12,}")
_JWT = re.compile(r"(?<![\w@])eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*")


def _secret_like(tok: str) -> bool:
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    if has_digit and has_alpha:
        return True
    return len(tok) >= 32 and bool(re.fullmatch(r"[0-9a-fA-F]+", tok))


def redact(text: str) -> str:
    """Mask numbers and tokens that look like secrets. Emails and plain words stay."""
    text = _JWT.sub("[redacted]", text)
    text = _PREFIXED.sub("[redacted]", text)
    text = _SSN.sub("[redacted]", text)
    text = _DIGITS.sub("[redacted]", text)
    return _TOKEN.sub(lambda m: "[redacted]" if _secret_like(m.group(0)) else m.group(0), text)


def clean_field_text(raw: str) -> str:
    """Redact, then trim to the last FIELD_CHARS chars (where typing happens)."""
    text = " ".join(raw[-READ_CHARS:].split())
    text = redact(text)
    if len(text) > FIELD_CHARS:
        text = "…" + text[-FIELD_CHARS:]
    return text


# ---------------------------------------------------------------- data types
@dataclass
class Elem:
    role: str = ""
    label: str = ""
    subrole: str = ""
    window: str = ""
    ident: object = None  # stable id of the element (e.g. CFHash), if known
    value: Optional[str] = None  # only ever set for editable, non-secure elements
    secure: bool = False
    editable: bool = False


@dataclass
class WinInfo:
    ident: object
    title: str
    role: str = "AXWindow"
    subrole: str = ""

    @property
    def dialogish(self) -> bool:
        return self.role == "AXSheet" or self.subrole in DIALOG_SUBROLES


@dataclass
class Snapshot:
    app: str
    window: str = ""
    windows: list = field(default_factory=list)
    focus: Optional[Elem] = None
    pid: int = 0


@dataclass
class Event:
    kind: str
    t: float
    data: dict
    count: int = 1


class Sources(Protocol):
    def available(self) -> bool: ...
    def snapshot(self) -> Optional[Snapshot]: ...
    def element_at(self, x: float, y: float) -> Optional[Elem]: ...
    def start_input(self, on_click: Callable, on_scroll: Callable, on_key: Callable) -> None: ...
    def stop_input(self) -> None: ...


def _role(role: str) -> str:
    return (role or "element").removeprefix("AX")


def _quoted(s: str) -> str:
    return f'"{s}"' if s else ""


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- the log
class ActivityLog:
    def __init__(self, sources: Optional[Sources] = None, enabled: Optional[bool] = None,
                 clock: Callable[[], float] = time.monotonic, poll_interval: float = POLL_INTERVAL_S,
                 pause_s: float = TYPING_PAUSE_S):
        self._sources = sources
        self.enabled = config.ACTIVITY if enabled is None else enabled
        self._clock = clock
        self._interval = poll_interval
        self._pause = pause_s
        self._lock = threading.RLock()
        self._events: deque = deque(maxlen=MAX_EVENTS)
        self._callbacks: list = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        # raw input hand-off (listener threads -> poller); never holds key identities
        self._in_lock = threading.Lock()
        self._keys = 0
        self._last_key_t = 0.0
        self._commit: Optional[str] = None
        self._clicks: deque = deque(maxlen=50)
        self._scrolls: deque = deque(maxlen=200)
        self._reset_state()

    def _reset_state(self) -> None:
        self._app = None
        self._window = None
        self._win_ids: set = set()
        self._focus_key = None
        self._field: Optional[dict] = None
        self._typing = False
        self._last_activity_t = 0.0
        self._first = True
        self._dialog_seen = ("", 0.0)
        with self._in_lock:
            self._keys = 0
            self._commit = None
            self._clicks.clear()
            self._scrolls.clear()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        """Begin watching. Returns False (and does nothing) if disabled or unavailable."""
        if not self.enabled:
            return False
        with self._lock:
            if self._running:
                return True
            if self._sources is None:
                self._sources = AXSources()
            if not self._sources.available():
                print("[Sherpa] Activity awareness is off: no Accessibility permission for this terminal.")
                return False
            self._events.clear()
            self._reset_state()
            self._running = True
            self._stop.clear()
        try:
            self._sources.start_input(self._on_click, self._on_scroll, self._on_key)
        except Exception as e:
            print(f"[Sherpa] Activity: input listeners unavailable ({type(e).__name__}); watching windows only.")
        self._thread = threading.Thread(target=self._run, name="sherpa-activity", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop watching and purge everything."""
        with self._lock:
            was_running = self._running
            self._running = False
        self._stop.set()
        self._wake.set()
        if was_running and self._sources is not None:
            try:
                self._sources.stop_input()
            except Exception:
                pass
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._events.clear()
            self._reset_state()

    @property
    def running(self) -> bool:
        return self._running

    def on_event(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Call `callback(event)` for every new or updated event (on the poller thread,
        so keep it quick). Subscriptions survive stop()/start(). Returns an unsubscribe."""
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)
        return unsubscribe

    # ------------------------------------------------------------ input hand-off (listener threads)
    def _on_click(self, x: float, y: float, t: Optional[float] = None) -> None:
        if self._running:
            self._clicks.append((x, y, self._clock() if t is None else t))
            self._wake.set()

    def _on_scroll(self, dy: float = 0.0, t: Optional[float] = None) -> None:
        if self._running:
            self._scrolls.append((dy, self._clock() if t is None else t))

    def _on_key(self, kind: str) -> None:
        """kind is "type" or "commit" (Return/Tab); never which key."""
        if not self._running:
            return
        with self._in_lock:
            if kind == "commit":
                self._commit = "commit"
                self._wake.set()
            else:
                self._keys += 1
                self._last_key_t = self._clock()

    # ------------------------------------------------------------ events
    def _append(self, kind: str, data: dict, t: Optional[float] = None, merge_key=None,
                merge_within: float = 0.0, combine: Optional[Callable] = None) -> Optional[Event]:
        with self._lock:
            if not self._running:
                return None
            t = self._clock() if t is None else t
            ev = None
            if merge_key is not None:
                for old in reversed(list(self._events)[-6:]):
                    if old.kind == kind and old.data.get("key") == merge_key and t - old.t <= merge_within:
                        ev = old
                        break
            if ev is not None:
                self._events.remove(ev)
                ev.data = combine(ev.data, data) if combine else data
                ev.t = max(ev.t, t)
                ev.count += 1
            else:
                ev = Event(kind, t, data)
            self._events.append(ev)
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(ev)
            except Exception:
                pass
        return ev

    def recent(self, kind: Optional[str] = None, within: Optional[float] = None, n: Optional[int] = None) -> list:
        """Copies of events, oldest first; filter by kind, by age (seconds), or last `n`."""
        with self._lock:
            now = self._clock()
            evs = [Event(e.kind, e.t, dict(e.data), e.count) for e in self._events
                   if (kind is None or e.kind == kind) and (within is None or now - e.t <= within)]
        return evs[-n:] if n else evs

    def last_field_value(self) -> Optional[Event]:
        """The most recent field_finished event (data: label, value, secure, ...), or None."""
        evs = self.recent(kind="field_finished", n=1)
        return evs[0] if evs else None

    def summary(self) -> str:
        """Non-sensitive: event kinds and counts only."""
        with self._lock:
            counts: dict = {}
            for e in self._events:
                counts[e.kind] = counts.get(e.kind, 0) + 1
        return ", ".join(f"{k} x{v}" for k, v in counts.items()) or "no events"

    # ------------------------------------------------------------ rendering
    def timeline(self, since: float, until: Optional[float] = None, max_events: int = 40) -> str:
        """Compact text of events with since <= t <= until (monotonic times), oldest
        first / newest last, with relative times, capped in count and total length."""
        if not self.enabled:
            return ""
        with self._lock:
            now = self._clock()
            until = now if until is None else until
            evs = [e for e in self._events if since <= e.t <= until]
            evs.sort(key=lambda e: e.t)
            finished = {e.data.get("key") for e in evs if e.kind == "field_finished"}
            evs = [e for e in evs if not (e.kind in ("typing_started", "typing_stopped") and e.data.get("key") in finished)]
            evs = evs[-max_events:]
            lines = [f"[{max(0, round(now - e.t))}s ago] {_truncate(self._describe(e), EVENT_MAX_CHARS)}" for e in evs]
        total, kept = 0, []
        for line in reversed(lines):  # keep the newest when over the cap
            if total + len(line) + 1 > TIMELINE_MAX_CHARS:
                kept.append("(earlier activity omitted)")
                break
            kept.append(line)
            total += len(line) + 1
        return "\n".join(reversed(kept))

    def recent_timeline(self, window_s: float = 60.0, max_events: int = 40) -> str:
        return self.timeline(self._clock() - window_s, None, max_events)

    @staticmethod
    def _describe(e: Event) -> str:
        d = e.data
        app = f" in {d['app']}" if d.get("app") else ""
        times = f" (x{e.count})" if e.count > 1 and e.kind == "clicked" else ""
        if e.kind == "app_switched":
            return f"working in {d['app']}" + (f" ({_quoted(d['window'])})" if d.get("window") else "") if d.get("initial") else f"switched to {d['app']}"
        if e.kind == "window_changed":
            return f"window changed to {_quoted(d.get('window', ''))}{app}"
        if e.kind == "dialog_opened":
            return (f"dialog opened: {_quoted(d['window'])}" if d.get("window") else "dialog opened") + app
        if e.kind == "focus_changed":
            where = f' of {_quoted(d["window"])}' if d.get("window") else ""
            return f"focused {_role(d.get('role', ''))} {_quoted(d.get('label', ''))}{where}".replace("  ", " ")
        if e.kind == "clicked":
            return f"clicked {_role(d.get('role', ''))} {_quoted(d.get('label', ''))}{app}{times}".replace("  ", " ")
        if e.kind == "typing_started":
            return f"started typing in {d['label'] + ' field' if d.get('label') else (d.get('app') or 'an app')}"
        if e.kind == "typing_stopped":
            return "stopped typing"
        if e.kind == "field_finished":
            if d.get("secure"):
                return SECURE_NOTE
            name = f"{d['label']} field" if d.get("label") else "a field"
            where = f' of {_quoted(d["window"])}' if d.get("window") else ""
            if d.get("value") is None:
                return f"typed in {name}{where} (content not read)"
            return f'typed in {name}{where}: "{d["value"]}" (finished)'
        if e.kind == "scrolled":
            return f"scrolled {'down' if d.get('dy', 0) < 0 else 'up'}{app}"
        return e.kind

    # ------------------------------------------------------------ the poller
    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.poll_once()
            except Exception as e:  # a flaky AX read must never kill the watcher
                print(f"[Sherpa] Activity poll error: {type(e).__name__}")

    @staticmethod
    def _field_id(snap: Snapshot, el: Elem) -> tuple:
        return (snap.app, el.window or snap.window, el.ident if el.ident is not None else (el.role, el.label))

    def poll_once(self, now: Optional[float] = None) -> None:
        """One watch cycle. Public so tests can drive it with fake sources/clock."""
        if not self._running:
            return
        now = self._clock() if now is None else now
        self._drain_inputs(now)
        snap = self._sources.snapshot()
        if snap is None:
            return
        app = snap.app

        if self._first or app != self._app:
            self._finish_field("focus lost", now)
            initial = self._first
            self._first = False
            self._app = app
            self._window = snap.window
            self._win_ids = {w.ident for w in snap.windows}
            self._typing = False
            self._field = None
            self._focus_key = None
            self._append("app_switched", {"app": app, "window": snap.window, "initial": initial}, now)
            self._adopt_focus(snap, snap.focus, now, silent=True)
            return

        # dialogs / new windows
        new = [w for w in snap.windows if w.ident not in self._win_ids]
        opened_title = None
        for w in new:
            if w.dialogish or self._win_ids:
                opened_title = w.title
                self._dialog_seen = (w.title, now)
                self._append("dialog_opened", {"app": app, "window": w.title, "sheet": w.role == "AXSheet"}, now)
        self._win_ids = {w.ident for w in snap.windows}
        if snap.window != self._window:
            seen_title, seen_t = self._dialog_seen
            if snap.window and snap.window != opened_title and not (snap.window == seen_title and now - seen_t < 3.0):
                self._append("window_changed", {"app": app, "window": snap.window}, now)
            self._window = snap.window

        el = snap.focus
        cur = self._field
        same = cur is not None and el is not None and self._field_id(snap, el) == cur["key"]
        if same:
            self._read_value(cur, el, now)

        # typing activity (timing only)
        with self._in_lock:
            keys, self._keys = self._keys, 0
            last_key = self._last_key_t
            commit, self._commit = self._commit, None
        if keys:
            self._last_activity_t = last_key or now
            if not self._typing:
                self._typing = True
                d = {"app": app, "label": cur["label"] if cur else "", "key": cur["key"] if cur else app}
                self._append("typing_started", d, now, merge_key=d["key"], merge_within=TYPING_MERGE_S,
                             combine=lambda old, new: old)
            if cur is not None:
                cur["dirty"] = True
        if commit and cur is not None:
            self._finish_field("return/tab", now)
        if self._typing and now - self._last_activity_t >= self._pause:
            self._typing = False
            self._append("typing_stopped", {"app": app, "key": cur["key"] if cur else app}, now)
        if cur is not None and cur["dirty"] and now - self._last_activity_t >= self._pause:
            self._finish_field("paused", now)

        # focus
        if not same:
            self._finish_field("focus lost", now)
            self._adopt_focus(snap, el, now, silent=False)

    # ------------------------------------------------------------ field tracking
    def _adopt_focus(self, snap: Snapshot, el: Optional[Elem], now: float, silent: bool) -> None:
        if el is None or not el.role:
            self._field = None
            self._focus_key = None
            return
        key = self._field_id(snap, el)
        prev_key = self._focus_key
        self._focus_key = key
        editable = el.editable or el.secure or el.role in EDITABLE_ROLES
        self._field = None
        if editable:
            self._field = {"key": key, "label": el.label, "role": el.role, "window": el.window or snap.window,
                           "app": snap.app, "secure": el.secure, "readable": False, "value": None,
                           "hash": None, "baseline": None, "dirty": False}
            self._read_value(self._field, el, now, baseline=True)
        if silent:
            return
        data = {"app": snap.app, "role": el.role, "label": el.label, "window": el.window or snap.window,
                "key": key, "prev_key": prev_key}
        with self._lock:
            last = self._events[-1] if self._events else None
            if last is not None and last.kind == "focus_changed" and now - last.t < FOCUS_FLAP_S:
                self._events.remove(last)  # focus flap: keep only the final focus
                if last.data.get("prev_key") == key:
                    return  # A -> B -> A: no net change
                data["prev_key"] = last.data.get("prev_key")
        self._append("focus_changed", data, now)

    def _read_value(self, f: dict, el: Elem, now: float, baseline: bool = False) -> None:
        if f["secure"] or el.secure:
            f["secure"] = True
            return  # never read
        if not isinstance(el.value, str):
            return  # unreadable: content not read
        f["readable"] = True
        h = hash(el.value[-READ_CHARS:])
        if baseline:
            f["baseline"] = h
        elif h != f["hash"]:
            f["dirty"] = True  # typed, pasted or autofilled
            self._last_activity_t = now
        f["hash"] = h
        f["value"] = clean_field_text(el.value)

    def _finish_field(self, reason: str, now: float) -> None:
        f = self._field
        if f is None or not f["dirty"]:
            return
        f["dirty"] = False
        if f["secure"]:
            value = None
        elif f["readable"]:
            if f["hash"] == f["baseline"]:
                return  # typed and deleted again: no net change
            f["baseline"] = f["hash"]
            value = f["value"]
        else:
            value = None
        self._append(
            "field_finished",
            {"app": f["app"], "label": f["label"], "role": f["role"], "window": f["window"], "secure": f["secure"],
             "value": value, "reason": reason, "key": f["key"]},
            now, merge_key=f["key"], merge_within=FIELD_MERGE_S,
        )

    # ------------------------------------------------------------ clicks / scrolls
    def _drain_inputs(self, now: float) -> None:
        while self._clicks:
            try:
                x, y, t = self._clicks.popleft()
            except IndexError:
                break
            try:
                el = self._sources.element_at(x, y)
            except Exception:
                el = None
            d = {"app": self._app or "", "role": el.role if el else "", "label": el.label if el else ""}
            if el is not None and el.secure:
                d["label"] = d["label"] or "password field"
            self._append("clicked", d, t)
        while self._scrolls:
            try:
                dy, t = self._scrolls.popleft()
            except IndexError:
                break
            d = {"app": self._app or "", "dy": dy, "key": self._app or ""}
            self._append("scrolled", d, t, merge_key=d["key"], merge_within=SCROLL_MERGE_S,
                         combine=lambda old, new: {**new, "dy": old.get("dy", 0) + new.get("dy", 0)})


# ---------------------------------------------------------------- real sources
class AXSources:
    """Accessibility polling plus pynput mouse/keyboard listeners (needs permissions)."""

    def __init__(self) -> None:
        self._mouse = None
        self._keyboard = None
        self._manual: set = set()

    def available(self) -> bool:
        from . import accessibility
        return accessibility.trusted()

    @staticmethod
    def _label(ax, el) -> str:
        for attr in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXLabel", "AXHelp"):
            t = ax._text(ax._get(el, attr))
            if t:
                return t
        titled = ax._get(el, "AXTitleUIElement")
        if titled is not None:
            t = ax._text(ax._get(titled, "AXValue")) or ax._text(ax._get(titled, "AXTitle"))
            if t:
                return t
        return ""

    def _elem(self, el, window: str = "", read_value: bool = False) -> Elem:
        from . import accessibility as ax
        role = ax._text(ax._get(el, "AXRole"))
        subrole = ax._text(ax._get(el, "AXSubrole"))
        secure = "AXSecureTextField" in (role, subrole)
        editable = role in EDITABLE_ROLES or secure
        value = None
        if read_value and editable and not secure:  # the secure check comes BEFORE any value read
            raw = ax._get(el, "AXValue")
            value = raw[-READ_CHARS:] if isinstance(raw, str) else None
        try:
            from ApplicationServices import CFHash  # type: ignore
            ident = CFHash(el)
        except Exception:
            ident = None
        return Elem(role=role, subrole=subrole, label=self._label(ax, el), window=window, ident=ident,
                    value=value, secure=secure, editable=editable)

    def snapshot(self) -> Optional[Snapshot]:
        from ApplicationServices import AXUIElementCreateApplication, AXUIElementSetAttributeValue, AXUIElementSetMessagingTimeout
        from . import accessibility as ax
        pid, name = ax.frontmost_app()
        app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, ax.MESSAGE_TIMEOUT_S)
        if pid not in self._manual:  # Chromium/Electron only expose their tree when asked, once is enough
            self._manual.add(pid)
            try:
                AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
            except Exception:
                pass
        fw = ax._get(app, "AXFocusedWindow")
        title = ax._text(ax._get(fw, "AXTitle")) if fw is not None else ""
        wins = []
        for w in ax._get(app, "AXWindows") or []:
            wins.append(self._win(ax, w))
            for c in ax._get(w, "AXChildren") or []:  # sheets hang off their window
                if ax._text(ax._get(c, "AXRole")) == "AXSheet":
                    wins.append(self._win(ax, c))
        fe = ax._get(app, "AXFocusedUIElement")
        focus = self._elem(fe, title, read_value=True) if fe is not None else None
        return Snapshot(app=name, window=title, windows=wins, focus=focus, pid=pid)

    @staticmethod
    def _win(ax, w) -> WinInfo:
        try:
            from ApplicationServices import CFHash  # type: ignore
            ident = CFHash(w)
        except Exception:
            ident = id(w)
        return WinInfo(ident, ax._text(ax._get(w, "AXTitle")), ax._text(ax._get(w, "AXRole")) or "AXWindow",
                       ax._text(ax._get(w, "AXSubrole")))

    def element_at(self, x: float, y: float) -> Optional[Elem]:
        from ApplicationServices import AXUIElementCopyElementAtPosition, AXUIElementCreateSystemWide, AXUIElementSetMessagingTimeout
        from . import accessibility as ax
        system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, ax.MESSAGE_TIMEOUT_S)
        err, el = AXUIElementCopyElementAtPosition(system, x, y, None)
        if err != 0 or el is None:
            return None
        return self._elem(el)  # never reads the value

    def start_input(self, on_click, on_scroll, on_key) -> None:
        from pynput import keyboard, mouse

        def click(x, y, button, pressed):
            if pressed:
                on_click(x, y)

        def scroll(x, y, dx, dy):
            on_scroll(dy)

        ignore = {getattr(keyboard.Key, n) for n in (
            "shift", "shift_l", "shift_r", "ctrl", "ctrl_l", "ctrl_r", "cmd", "cmd_l", "cmd_r", "alt", "alt_l",
            "alt_r", "caps_lock", "esc", "up", "down", "left", "right", "home", "end", "page_up", "page_down",
            "insert", "menu", "num_lock", "print_screen", "scroll_lock", "pause") if hasattr(keyboard.Key, n)}
        commit = {keyboard.Key.enter, keyboard.Key.tab}

        def key(k):
            # Classified here and immediately forgotten: which key it was never leaves this function.
            if k in ignore:  # includes Right Option, the Sherpa hotkey
                return
            if isinstance(k, keyboard.Key) and k.name and k.name.startswith("f") and k.name[1:].isdigit():
                return
            on_key("commit" if k in commit else "type")

        self._mouse = mouse.Listener(on_click=click, on_scroll=scroll)
        self._keyboard = keyboard.Listener(on_press=key)
        for l in (self._mouse, self._keyboard):
            l.daemon = True
            l.start()

    def stop_input(self) -> None:
        for l in (self._mouse, self._keyboard):
            if l is not None:
                l.stop()
        self._mouse = self._keyboard = None


log = ActivityLog()  # the session-wide instance (see session.py)


# ---------------------------------------------------------------- probe
def _main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Watch your own activity as Sherpa would (from a trusted terminal).")
    ap.add_argument("--show-content", action="store_true", help="print captured field contents (default: masked)")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    probe = ActivityLog(enabled=True)

    def show(e: Event) -> None:
        d = e.data
        bits = [e.kind]
        if d.get("app"):
            bits.append(d["app"])
        if d.get("role"):
            bits.append(_role(d["role"]))
        if d.get("label"):
            bits.append(f'"{d["label"]}"')
        if e.kind in ("window_changed", "dialog_opened") and d.get("window"):
            bits.append(f'"{d["window"]}"')
        if e.kind == "field_finished":
            if d.get("secure"):
                bits.append("(password field, content not read)")
            elif d.get("value") is None:
                bits.append("(content not read)")
            else:
                bits.append(f'content: "{d["value"]}"' if args.show_content else f'({len(d["value"])} chars, masked; --show-content to see)')
            bits.append(f'[{d.get("reason")}]')
        print(f"  {time.strftime('%H:%M:%S')}  " + "  ".join(bits))

    probe.on_event(show)
    if not probe.start():
        sys.exit("Couldn't start (no Accessibility permission for this terminal, or SHERPA_ACTIVITY=0 is not the cause: it is ignored here).")
    print(f"Watching for {args.seconds:.0f}s (Ctrl+C to stop). Switch apps, click, type in a field, press Tab/Return...")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print("\nSummary:", probe.summary())
    tl = probe.recent_timeline(window_s=max(args.seconds, 60) + 5)
    if not args.show_content:
        tl = re.sub(r'(: )"[^"\n]*"( \(finished\))', r'\1"<masked>"\2', tl)
    print("\nTimeline as Claude would see it:\n" + (tl or "(nothing happened)"))
    probe.stop()
    print("\nStopped; log purged.")


if __name__ == "__main__":
    _main()
