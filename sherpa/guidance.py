"""Walk the user through a task by pointing, one step at a time.

Claude plans the steps as targets described in words ("the File menu"); this
runs them. For each step it looks the target up on the screen *as it is right
now*, glides Sherpa's cursor there, speaks the step's line, and waits for the
user to click it. Because every step is looked up fresh, a button that isn't
where the plan assumed is found where it actually is. If the user clicks
somewhere else, the target is looked up again in case the screen changed and
the cursor follows it.

It never clicks for the user. It stops when the steps run out, when a target
can't be found (she says so instead of pointing at a guess), when the user goes
quiet for IDLE_TIMEOUT_S, or when `still_wanted()` turns false because the user
interrupted or asked something new.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from pynput import mouse

from . import locate
from .capture import get_cursor_position
from .guide import guide

IDLE_TIMEOUT_S = 60.0  # no click for this long: leave the guide
SETTLE_AFTER_CLICK_S = 0.6  # let a menu or dialog finish opening before the next lookup
RELOOK_DELAY_S = 0.7  # after a click elsewhere, give the screen time to change
MOVED_PX = 40  # re-point only if the target moved at least this far
AFTER_SPEECH_PAUSE_S = 1.0


@dataclass
class Step:
    target: str
    say: str
    label: str = ""
    advance: str = "click"  # "click" | "after_speech"


def parse_steps(raw: list) -> list:
    steps = []
    for item in raw or []:
        target = str(item.get("target", "")).strip()
        if not target:
            continue
        say = str(item.get("say", "")).strip() or f"Here's {target}."
        label = str(item.get("label", "")).strip() or target.removeprefix("the ").removeprefix("The ")[:24]
        advance = "after_speech" if item.get("advance") == "after_speech" else "click"
        steps.append(Step(target, say, label, advance))
    return steps


class Clicks:
    """Global left/right/any mouse-press watcher."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._last: Optional[tuple] = None
        self._listener: Optional[mouse.Listener] = None

    def start(self) -> None:
        self._listener = mouse.Listener(on_click=self._on_click)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def clear(self) -> None:
        self._event.clear()

    def _on_click(self, x, y, _button, pressed) -> None:
        if pressed:
            self._last = (x, y)
            self._event.set()

    def wait(self, timeout: float) -> Optional[tuple]:
        if self._event.wait(timeout):
            self._event.clear()
            return self._last
        return None


def _find(step: Step, task: str) -> Optional[locate.Located]:
    found = locate.locate(step.target, context=f"{task}. This step: {step.say}")
    if found is not None:
        found = locate.refine(step.target, found, guide)
    return found


def _wait_for_click(step: Step, found: locate.Located, task: str, clicks, still_wanted) -> bool:
    """True once the user clicks the target. False on timeout or interruption."""
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    while still_wanted() and time.monotonic() < deadline:
        pos = clicks.wait(0.25)
        if pos is None:
            continue
        if math.hypot(pos[0] - found.x, pos[1] - found.y) <= found.radius:
            time.sleep(SETTLE_AFTER_CLICK_S)
            return True
        # Clicked somewhere else: the screen may have changed under us, so look again.
        time.sleep(RELOOK_DELAY_S)
        if not still_wanted():
            return False
        again = _find(step, task)
        if again is not None and math.hypot(again.x - found.x, again.y - found.y) >= MOVED_PX:
            print(f"[Sherpa] Guide: '{step.target}' is somewhere else now; following it.")
            found = again
            guide.move_to(found.x, found.y, label=step.label)
    return False


def run(steps: list, task: str, still_wanted: Callable[[], bool], speak: Callable[[str], None], clicks=None) -> None:
    """Point through `steps`. `speak(text)` blocks while she talks. `clicks` is
    injectable for tests."""
    if not steps:
        return
    clicks = clicks or Clicks()
    clicks.start()
    x0, y0 = get_cursor_position()
    guide.show(x0, y0)  # she appears where the user's pointer is, then glides off
    try:
        for i, step in enumerate(steps):
            if not still_wanted():
                return
            print(f"[Sherpa] Guide {i + 1}/{len(steps)}: looking for {step.target!r}")
            found = _find(step, task)
            if not still_wanted():
                return
            if found is None:
                speak(f"I can't find {step.target} on your screen.")
                return
            print(f"[Sherpa] Guide: pointing at {found.label or step.target!r} ({found.how}) at ({found.x:.0f}, {found.y:.0f})")
            clicks.clear()
            guide.move_to(found.x, found.y, label=step.label)
            speak(step.say)
            if not still_wanted():
                return
            if step.advance == "after_speech":
                time.sleep(AFTER_SPEECH_PAUSE_S)
                continue
            if not _wait_for_click(step, found, task, clicks, still_wanted):
                if still_wanted():
                    print("[Sherpa] Guide: no click for a while; leaving the guide.")
                return
    finally:
        clicks.stop()
        guide.hide()
