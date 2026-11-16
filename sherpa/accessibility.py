"""Find on-screen UI elements by name using the macOS Accessibility API.

This is the precise way to find a button, menu or text field: apps report
each element's label and exact on-screen rectangle, so no guessing from
pixels. It needs the Accessibility permission Sherpa already uses for hotkeys,
and it can't see inside things that don't expose themselves (some games,
canvases, remote desktops); locate.py falls back to vision for those.

Frames are global screen points, origin at the primary display's top-left:
the same space as the pointer tracker and the guide cursor.

Only what is on screen right now is returned. Items of a *closed* menu are
left out on purpose: their reported positions aren't reliable until the menu
is open, so the guide points at the menu title first, and the next lookup
(after the menu opens) finds the items for real.

Probe it from a trusted terminal:
    python -m sherpa.accessibility dump          # what Sherpa can see in the front app
    python -m sherpa.accessibility find "Save"   # best matches for a description
"""

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import Quartz
from AppKit import NSWorkspace
from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementSetAttributeValue,
    AXUIElementSetMessagingTimeout,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

MAX_NODES = 2500
MAX_DEPTH = 16
TIME_BUDGET_S = 1.5
MESSAGE_TIMEOUT_S = 0.25  # per attribute read; a hung app must not stall a lookup

# Elements that never contain anything worth pointing at inside them.
_LEAF_ROLES = {"AXStaticText", "AXImage", "AXScrollBar", "AXValueIndicator", "AXSlider", "AXCheckBox", "AXRadioButton"}
# Roles worth keeping as targets even though their label is thin.
_TARGET_ROLES = {
    "AXButton", "AXCheckBox", "AXRadioButton", "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField",
    "AXPopUpButton", "AXMenuButton", "AXMenuBarItem", "AXMenuItem", "AXLink", "AXTab", "AXTabGroup", "AXSlider",
    "AXDisclosureTriangle", "AXRow", "AXCell", "AXStaticText", "AXImage", "AXToolbar", "AXWindow", "AXSheet",
    "AXOutline", "AXList", "AXTable", "AXSegmentedControl", "AXIncrementor", "AXColorWell",
}


@dataclass
class Element:
    role: str
    label: str  # best human-readable name
    names: list  # every name-like string the element offers (title, description, value, help...)
    frame: tuple  # (x, y, w, h)
    path: str = ""  # labelled ancestors, e.g. "Save > Where"
    enabled: bool = True
    window: str = ""  # title of the containing window
    depth: int = 0

    @property
    def center(self) -> tuple:
        x, y, w, h = self.frame
        return x + w / 2, y + h / 2

    @property
    def aim(self) -> tuple:
        """Where to point: the middle, except long fields, where the left part
        is where you'd click or read."""
        x, y, w, h = self.frame
        if w > 320:
            return x + min(w / 2, 140), y + h / 2
        return x + w / 2, y + h / 2

    def describe(self) -> str:
        where = f" in {self.path}" if self.path else ""
        return f'{self.role.removeprefix("AX")} "{self.label}"{where}'


@dataclass
class Match:
    element: Element
    score: float


# ---------------------------------------------------------------- raw AX access
def trusted() -> bool:
    return bool(AXIsProcessTrusted())


def _get(el, attr):
    try:
        err, value = AXUIElementCopyAttributeValue(el, attr, None)
    except Exception:
        return None
    return value if err == 0 else None


def _point(value) -> Optional[tuple]:
    try:
        ok, pt = AXValueGetValue(value, kAXValueCGPointType, None)
        return (pt.x, pt.y) if ok else None
    except Exception:
        return None


def _size(value) -> Optional[tuple]:
    try:
        ok, sz = AXValueGetValue(value, kAXValueCGSizeType, None)
        return (sz.width, sz.height) if ok else None
    except Exception:
        return None


def _text(value) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _display_rects() -> list:
    _, ids, n = Quartz.CGGetActiveDisplayList(16, None, None)
    out = []
    for d in list(ids[:n]):
        b = Quartz.CGDisplayBounds(d)
        out.append((b.origin.x, b.origin.y, b.size.width, b.size.height))
    return out


def _on_screen(frame: tuple, displays: list) -> bool:
    x, y, w, h = frame
    cx, cy = x + w / 2, y + h / 2
    return any(dx <= cx < dx + dw and dy <= cy < dy + dh for dx, dy, dw, dh in displays)


def frontmost_app() -> tuple:
    """(pid, name) of the app the user is working in. Sherpa's overlays never take focus."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()), str(app.localizedName())


# ---------------------------------------------------------------- snapshot
def snapshot(pid: Optional[int] = None) -> list:
    """Every named, visible element of an app (default: the frontmost one)."""
    if pid is None:
        pid, _ = frontmost_app()
    app = AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(app, MESSAGE_TIMEOUT_S)
    # Chromium/Electron only build their accessibility tree when asked.
    try:
        AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
    except Exception:
        pass

    displays = _display_rects()
    deadline = time.monotonic() + TIME_BUDGET_S
    out: list = []
    counter = [0]

    def label_of(el, role: str) -> tuple:
        names = []
        for attr in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp", "AXLabel"):
            t = _text(_get(el, attr))
            if t and t not in names:
                names.append(t)
        value = _text(_get(el, "AXValue")) if role in {"AXStaticText", "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField", "AXButton", "AXPopUpButton"} else ""
        if value and len(value) <= 120 and value not in names:
            names.append(value)
        # A field's name is often the static text next to it.
        titled = _get(el, "AXTitleUIElement")
        if titled is not None:
            t = _text(_get(titled, "AXValue")) or _text(_get(titled, "AXTitle"))
            if t and t not in names:
                names.insert(0, t)
        return (names[0] if names else ""), names

    def walk(el, depth: int, trail: list, window: str) -> None:
        if counter[0] >= MAX_NODES or depth > MAX_DEPTH or time.monotonic() > deadline:
            return
        counter[0] += 1
        role = _text(_get(el, "AXRole"))
        if role == "AXWindow":
            window = _text(_get(el, "AXTitle")) or window

        pos, size = _point(_get(el, "AXPosition")), _size(_get(el, "AXSize"))
        frame = (pos[0], pos[1], size[0], size[1]) if pos and size else None
        label, names = label_of(el, role)

        if frame and frame[2] >= 4 and frame[3] >= 4 and _on_screen(frame, displays) and label and role in _TARGET_ROLES:
            enabled = _get(el, "AXEnabled")
            out.append(
                Element(
                    role=role,
                    label=label,
                    names=names,
                    frame=frame,
                    path=" > ".join(trail[-3:]),
                    enabled=enabled is not False,
                    window=window,
                    depth=depth,
                )
            )

        if role in _LEAF_ROLES:
            return
        if role == "AXMenuBarItem" and not _get(el, "AXSelected"):
            return  # closed menu: positions inside aren't trustworthy
        children = _get(el, "AXChildren") or []
        if role == "AXMenuItem" and not _get(el, "AXSelected"):
            children = []  # a submenu is closed until its item is highlighted
        next_trail = trail + ([label] if label and role in {"AXMenuBarItem", "AXMenuItem", "AXWindow", "AXSheet", "AXGroup", "AXToolbar", "AXTabGroup"} else [])
        for child in children:
            walk(child, depth + 1, next_trail, window)

    # The menu bar, open menus, windows, sheets: all of the app's top-level children.
    for child in _get(app, "AXChildren") or []:
        walk(child, 0, [], "")
    bar = _get(app, "AXMenuBar")
    if bar is not None and not any(e.role == "AXMenuBarItem" for e in out):
        walk(bar, 0, [], "")
    return out


# ---------------------------------------------------------------- matching
_ROLE_HINTS = {
    "button": {"AXButton"},
    "field": {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"},
    "box": {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"},
    "textbox": {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"},
    "input": {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"},
    "menu": {"AXMenuBarItem", "AXMenuItem", "AXPopUpButton", "AXMenuButton"},
    "dropdown": {"AXPopUpButton", "AXComboBox", "AXMenuButton"},
    "tab": {"AXRadioButton", "AXTab"},
    "checkbox": {"AXCheckBox"},
    "link": {"AXLink"},
    "icon": {"AXButton", "AXImage"},
}
_FILLER = {"the", "a", "an", "to", "on", "in", "of", "for", "menu", "item", "option", "your"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("…", "").replace("...", "")).strip()


def _tokens(s: str) -> list:
    return [t for t in _norm(s).split() if t]


def score(query: str, el: Element) -> float:
    q_tokens = _tokens(query)
    hinted = set()
    for t in q_tokens:
        hinted |= _ROLE_HINTS.get(t, set())
    core = [t for t in q_tokens if t not in _ROLE_HINTS and t not in _FILLER] or q_tokens
    q = " ".join(core)
    if not q:
        return 0.0

    best = 0.0
    for name in el.names:
        n = _norm(name)
        if not n:
            continue
        n_tokens = n.split()
        if n == q:
            s = 1.0
        elif q in n_tokens or (len(q) > 3 and q in n):
            s = 0.8 if len(n_tokens) <= len(core) + 2 else 0.6
        elif set(core) <= set(n_tokens):
            s = 0.75
        else:
            s = 0.8 * difflib.SequenceMatcher(None, q, n).ratio()
        best = max(best, s)
    if best == 0.0:
        return 0.0
    if hinted and el.role in hinted:
        best = min(1.0, best + 0.1)
    elif hinted and el.role not in hinted:
        best -= 0.1
    if not el.enabled:
        best -= 0.05
    return max(0.0, best)


def match(query: str, elements: list, limit: int = 8) -> list:
    scored = [Match(e, score(query, e)) for e in elements]
    scored = [m for m in scored if m.score >= 0.5]
    scored.sort(key=lambda m: (-m.score, m.element.depth))
    return scored[:limit]


def confident(matches: list) -> Optional[Match]:
    """The one clear winner, or None when the best is weak or there's a close rival."""
    if not matches or matches[0].score < 0.85:
        return None
    if len(matches) > 1 and matches[1].score >= matches[0].score - 0.05:
        a, b = matches[0].element, matches[1].element
        if a.frame != b.frame:
            return None  # two different things named alike
    return matches[0]


# ---------------------------------------------------------------- probe
if __name__ == "__main__":
    import sys

    if not trusted():
        sys.exit("This terminal isn't trusted for Accessibility (System Settings > Privacy & Security > Accessibility).")
    pid, name = frontmost_app()
    print(f"Front app: {name} (pid {pid}) — focus the app you want to inspect; reading in 3s...")
    time.sleep(3)
    pid, name = frontmost_app()
    t0 = time.monotonic()
    els = snapshot(pid)
    print(f"{name}: {len(els)} named elements in {time.monotonic() - t0:.2f}s")
    if len(sys.argv) >= 3 and sys.argv[1] == "find":
        for m in match(" ".join(sys.argv[2:]), els):
            x, y = m.element.aim
            print(f"  {m.score:.2f}  {m.element.describe()}  at ({x:.0f}, {y:.0f})")
        print("confident:", confident(match(" ".join(sys.argv[2:]), els)) is not None)
    else:
        for e in els[:120]:
            x, y, w, h = e.frame
            print(f"  {e.role.removeprefix('AX'):<14} {e.label[:50]:<50} ({x:.0f},{y:.0f} {w:.0f}x{h:.0f})  {e.path}")
