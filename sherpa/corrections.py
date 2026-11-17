"""Gentle corrections: if what the user just typed contradicts what they asked
for, Sherpa says so ("you said Gmail, but that's a hotmail address").

Trigger. Subscribes (once) to the activity log's `field_finished` events. The
activity log has already redacted and truncated the text, and never reports the
content of secure/password or unreadable fields (value is None): those are
never checked here either.

Goal first. A correction is only ever relative to something the user ASKED for.
The goal context is built from the session's recent turns (their questions and
what Sherpa told them, plus guide steps, which live in the turn) within
GOAL_WINDOW_S, plus questions still pending an answer. No recent goal => no
check, no API call: the user didn't ask for help and isn't policed.

Check. One text-only Haiku call (no screenshots) given the goal, what they were
just doing and the value typed. It must say mismatch + confidence + one short
sentence. Everything ambiguous is "not a mismatch"; she only speaks when
mismatch and confidence >= CONFIDENCE (SHERPA_CORRECTION_CONFIDENCE, 0.8).

Interjection rules (all enforced here or in conversation.deliver_correction):
- Only while the session is active and SHERPA_CORRECTIONS isn't 0.
- Only when the phase is idle or guiding: never over her own speech, never
  while working/listening, never while the user is speaking in voice mode
  (a capture in progress) or has an unanswered message pending. If she is busy
  when the check returns, wait up to BUSY_WAIT_S for a gap, else drop it.
- Never mid-typing: any typing_started or newer field_finished after the
  triggering event cancels it, both before the call (superseded) and when it
  returns. It is also dropped if the field's value is no longer the one checked.
- Rate limit: at least MIN_GAP_S between corrections, at most MAX_PER_SESSION
  per session (counters reset on start()). A rate-limited event makes no call.
- Dedupe: the same (field, value) is checked once; the same sentence is never
  spoken twice.
- Delivery uses the normal ownership model (conversation.deliver_correction): a
  fresh token, ONE sentence through _speak_events (so barge-in, interrupt and
  the echo guard work), phase released after. If a guide was running, the
  correction ends it (the user is off-script anyway; the history note tells the
  model so, and the user can ask to continue). It is committed to history as a
  synthetic turn so follow-ups ("wait, which one?") work.

Privacy. The typed text goes only into the one check call (and only when a
goal exists). Nothing but a short non-sensitive status line is printed, and
nothing here is written to the debug dump, even with SHERPA_DEBUG=1.
"""

import threading
import time
from typing import Callable, Optional

from . import activity, brain, config

GOAL_TURNS = 4  # most recent turns that make up the stated goal
GOAL_WINDOW_S = config.CORRECTION_GOAL_WINDOW_S
ANSWER_CHARS = 300
TIMELINE_S = 120.0
TIMELINE_LINES = 8
MIN_GAP_S = 30.0
MAX_PER_SESSION = 5
BUSY_WAIT_S = 4.0
MIN_VALUE_CHARS = 2
MAX_SAY_CHARS = 220

SYSTEM_PROMPT = (
    "You are a quiet proofreader watching over a user's shoulder. The user asked an assistant for "
    "something (their stated goal). They have just finished typing a value into a field. Decide "
    "whether that value CONTRADICTS or clearly does not fit what they asked for, so a friendly "
    "heads-up would save them a mistake.\n\n"
    "Flag (mismatch true) ONLY when you are confident of a real conflict with the stated goal, for example:\n"
    "- an email address at a different provider/domain than the account they said they'd use\n"
    "- a name, number, date, address or code that differs from the one they said or dictated "
    "(typo, transposed or missing digit, wrong date or month)\n"
    "- an obviously wrong format for the field (an email with no @ in an email field)\n"
    "- text clearly meant for a different field than the one it went into, given the goal\n\n"
    "Do NOT flag (mismatch false) when:\n"
    "- the field has nothing to do with the stated goal (chat boxes, search bars, notes, other forms)\n"
    "- the goal doesn't state or imply what this value should be\n"
    "- the value is plausible or an equivalent format of the intended one (3/5 vs March 5th in the "
    "same sense, phone digits with different punctuation, different capitalisation, nickname vs the "
    "same person's name)\n"
    "- the field is about someone or something else than what the goal states: a recipient, "
    "\"share with\", friend's, colleague's or other person's address, name or number is NOT the "
    "user's own account just because the goal mentions the user's account\n"
    "- the user may have simply changed their mind or the goal was about something else\n"
    "- you cannot point to a specific thing the user said that this exact field's value conflicts with\n"
    "- you are unsure for ANY reason. A false alarm is far worse than a missed mistake.\n"
    "The value may be partly redacted as [redacted]; ignore redactions, never guess hidden parts.\n"
    "Compare only against what the user actually said or asked; never invent requirements. Assistant "
    "answers are context, not requirements: only hold the user to what THEY said.\n\n"
    "Reply with ONLY a JSON object: {\"mismatch\": true|false, \"confidence\": 0.0-1.0, \"say\": \"...\"}. "
    "\"say\" is what a friendly assistant says aloud: one short sentence, specific (mention both "
    "what they asked for and what they typed, briefly), a gentle heads-up rather than a lecture, "
    "e.g. \"Heads up, you said Gmail but that's a hotmail address.\" Leave \"say\" empty when "
    "mismatch is false."
)


# ------------------------------------------------------------- goal context
def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_goal(turns: list, pending: list, now: float, window_s: float = GOAL_WINDOW_S) -> str:
    """The user's recent stated goal as compact text, or "" if there is none.
    `turns` are session turns (dicts), `pending` the texts of unanswered
    utterances. Synthetic turns (our own corrections) never count as a goal."""
    lines = []
    recent = [t for t in turns if not t.get("synthetic") and now - t.get("t", now) <= window_s]
    for t in recent[-GOAL_TURNS:]:
        lines.append(f"User asked: {_clip(t.get('question', ''), 300)}")
        if t.get("answer"):
            lines.append(f"Assistant replied: {_clip(t['answer'], ANSWER_CHARS)}")
        steps = t.get("steps") or []
        if steps:
            route = " -> ".join(_clip(str(st.get("target", "")), 60) for st in steps if isinstance(st, dict))
            lines.append(f"Assistant guided them through: {route}")
    for text in pending:
        lines.append(f"User said (not yet answered): {_clip(text, 300)}")
    return "\n".join(lines)


def _field_name(data: dict) -> str:
    label = data.get("label") or ""
    return f"the {label} field" if label else "a field"


def build_prompt(goal: str, data: dict, value: str, timeline: str) -> list:
    where = " ".join(p for p in (f"app {data['app']}" if data.get("app") else "",
                                 f'window "{data["window"]}"' if data.get("window") else "") if p)
    parts = [
        f"STATED GOAL (what the user asked for):\n{goal}",
        f"JUST TYPED into {_field_name(data)}" + (f" ({where})" if where else "") + f':\n"{value}"',
    ]
    if timeline:
        parts.append(f"RECENT ACTIVITY, newest last:\n{timeline}")
    return [{"type": "text", "text": "\n\n".join(parts)}]


def check(goal: str, data: dict, value: str, timeline: str) -> dict:
    """The Haiku call. Returns {"mismatch": bool, "confidence": float, "say": str}."""
    raw = brain.ask_json(SYSTEM_PROMPT, build_prompt(goal, data, value, timeline),
                         max_tokens=1500, model=brain.LOOKUP_MODEL)
    try:
        conf = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"mismatch": raw.get("mismatch") is True, "confidence": conf, "say": str(raw.get("say") or "").strip()}


# ----------------------------------------------------------------- the actor
class Corrector:
    """deps (all injectable for tests):
      get_turns() -> session turns; get_pending() -> texts of unanswered utterances;
      can_speak() -> bool (is now a good moment); deliver(say, question) -> bool (spoke it);
      check_fn(goal, data, value, timeline) -> verdict dict."""

    def __init__(self, log, get_turns: Callable[[], list], get_pending: Callable[[], list],
                 can_speak: Callable[[], bool], deliver: Callable[[str, str], bool],
                 check_fn: Callable = check, enabled: Optional[bool] = None,
                 clock: Callable[[], float] = time.monotonic, threshold: Optional[float] = None,
                 min_gap_s: float = MIN_GAP_S, max_per_session: int = MAX_PER_SESSION,
                 busy_wait_s: float = BUSY_WAIT_S, goal_window_s: Optional[float] = None,
                 spawn: Optional[Callable] = None):
        self.log = log
        self.get_turns, self.get_pending, self.can_speak, self.deliver = get_turns, get_pending, can_speak, deliver
        self.check_fn = check_fn
        self.enabled = config.CORRECTIONS if enabled is None else enabled
        self.clock = clock
        self.threshold = config.CORRECTION_CONFIDENCE if threshold is None else threshold
        self.min_gap_s, self.max_per_session, self.busy_wait_s = min_gap_s, max_per_session, busy_wait_s
        self.goal_window_s = GOAL_WINDOW_S if goal_window_s is None else goal_window_s
        self._spawn = spawn or (lambda fn, *a: threading.Thread(target=fn, args=a, daemon=True).start())
        self._lock = threading.Lock()
        self._unsub: Optional[Callable] = None
        self.active = False
        self._reset()

    def _reset(self) -> None:
        self._seq = 0  # bumped by every typing_started / field_finished: a newer one cancels older work
        self._checked: set = set()
        self._said: set = set()
        self._last_spoken_at: Optional[float] = None
        self._count = 0

    # --------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._reset()
            self.active = True
            if self._unsub is None:
                self._unsub = self.log.on_event(self._on_event)  # survives log stop/start

    def stop(self) -> None:
        with self._lock:
            self.active = False
            self._seq += 1  # cancels anything in flight
            self._checked.clear()
            self._said.clear()

    # --------------------------------------------------------- events (poller thread: stay quick)
    def _on_event(self, ev) -> None:
        if not self.active:
            return
        if ev.kind == "typing_started":
            with self._lock:
                self._seq += 1
        elif ev.kind == "field_finished":
            data = dict(ev.data)
            with self._lock:
                self._seq += 1
                seq = self._seq
            self._spawn(self._handle, data, seq)

    def _current(self, seq: int, data: dict, value: str) -> bool:
        """Still the latest thing that happened, and the field still holds `value`."""
        if not self.active or seq != self._seq:
            return False
        last = self.log.last_field_value()
        return bool(last and last.data.get("key") == data.get("key") and last.data.get("value") == value)

    def _rate_limited(self) -> bool:
        if self._count >= self.max_per_session:
            return True
        return self._last_spoken_at is not None and self.clock() - self._last_spoken_at < self.min_gap_s

    # --------------------------------------------------------- the work
    def _handle(self, data: dict, seq: int) -> Optional[str]:
        """Check one finished field and maybe speak. Returns what happened (for tests)."""
        try:
            return self._handle_inner(data, seq)
        except Exception as e:  # a failed check must never disturb the session
            print(f"[Sherpa] Correction check failed: {type(e).__name__}")
            return "error"

    def _handle_inner(self, data: dict, seq: int) -> str:
        value = data.get("value")
        if data.get("secure") or not isinstance(value, str):
            return "skipped: not readable"
        value = value.strip()
        if len(value) < MIN_VALUE_CHARS or value.replace("[redacted]", "").strip() == "":
            return "skipped: empty"
        ident = (data.get("key"), value)
        with self._lock:
            if ident in self._checked:
                return "skipped: already checked"
            if self._rate_limited():
                return "skipped: rate limit"
        goal = build_goal(self.get_turns(), self.get_pending(), self.clock(), self.goal_window_s)
        if not goal:
            return "skipped: no goal"
        with self._lock:
            if seq != self._seq or not self.active:
                return "dropped: superseded"
            self._checked.add(ident)
        timeline = self._timeline()
        t0 = time.monotonic()
        verdict = self.check_fn(goal, data, value, timeline)
        took = time.monotonic() - t0
        say = _clip(verdict.get("say", ""), MAX_SAY_CHARS)
        flagged = bool(verdict.get("mismatch")) and verdict.get("confidence", 0) >= self.threshold and bool(say)
        print(f"[Sherpa] Correction check: {'flagged' if flagged else 'ok'} ({took:.1f}s)")
        if not flagged:
            return "ok"
        if not self._current(seq, data, value):
            return "dropped: superseded"
        deadline = time.monotonic() + self.busy_wait_s
        while not self.can_speak():
            if time.monotonic() >= deadline:
                return "dropped: busy"
            time.sleep(0.1)
            if not self._current(seq, data, value):
                return "dropped: superseded"
        with self._lock:
            if self._rate_limited():
                return "dropped: rate limit"
            if say.lower() in self._said:
                return "dropped: repeat"
        if not self._current(seq, data, value):
            return "dropped: superseded"
        question = f'[The user typed "{value}" in {_field_name(data)}]'
        if not self.deliver(say, question):
            return "dropped: busy"
        with self._lock:
            self._said.add(say.lower())
            self._count += 1
            self._last_spoken_at = self.clock()
        return "spoke"

    def _timeline(self) -> str:
        try:
            text = self.log.recent_timeline(TIMELINE_S)
        except Exception:
            return ""
        return "\n".join(text.splitlines()[-TIMELINE_LINES:])


# ----------------------------------------------------- module-level wiring
_corrector: Optional[Corrector] = None


def bind(get_turns, get_pending, can_speak, deliver, log=None) -> Corrector:
    """Wire the singleton to the conversation (called once by conversation.py)."""
    global _corrector
    _corrector = Corrector(log if log is not None else activity.log, get_turns, get_pending, can_speak, deliver)
    return _corrector


def start() -> None:
    if _corrector is not None:
        _corrector.start()


def stop() -> None:
    if _corrector is not None:
        _corrector.stop()
