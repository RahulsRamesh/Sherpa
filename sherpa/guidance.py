"""Walk the user through a task by pointing, one step at a time.

Claude plans the steps as targets described in words ("the File menu"); this
runs them. For each step it looks the target up on the screen *as it is right
now*, glides Sherpa's cursor there, speaks the step's line, and waits for the
user to click it. Because every step is looked up fresh, a button that isn't
where the plan assumed is found where it actually is. If the user clicks
somewhere else, the target is looked up again in case the screen changed and
the cursor follows it.

It never clicks for the user. It stops when the steps run out, when a target
can't be found twice (she asks the user to go back a step once, and never points at a guess), when the user goes
quiet for IDLE_TIMEOUT_S, or when `still_wanted()` turns false because the user
interrupted or asked something new.

Prefetch: while the user reads the step and goes to click, the next target is
looked up in the background (the `locate` stage only; refinement moves her
cursor, so it waits). A click usually changes the screen, so the prefetch is
only reused if, after the click and settle, a thumbnail of the display is still
essentially the one the prefetch started from; otherwise it is thrown away and
the lookup happens fresh. A prefetch that fails or finds nothing never aborts a
step; it just falls back to the normal lookup.
"""

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from pynput import mouse

from . import imgdiff, locate, verify
from .capture import get_cursor_position, grab_under_cursor
from .guide import guide

IDLE_TIMEOUT_S = 60.0  # no click for this long: leave the guide
SETTLE_AFTER_CLICK_S = 0.6  # let a menu or dialog finish opening before the next lookup
RELOOK_DELAY_S = 0.7  # after a click elsewhere, give the screen time to change
MOVED_PX = 40  # re-point only if the target moved at least this far
AFTER_SPEECH_PAUSE_S = 1.0
UNCHANGED_FRACTION = 0.005  # a prefetch is reusable while fewer thumbnail cells than this changed
PREFETCH_WAIT_S = 8.0  # longest we'll wait for an unchanged-screen prefetch still in flight
# Clicking one of these usually opens something, so the next target likely isn't
# on screen yet and a prefetch would be wasted.
_OPENS_SOMETHING = re.compile(r"menu|drop-?down|pop-?up|picker|dialog|\bopen\b", re.I)


@dataclass
class Step:
    target: str
    say: str
    label: str = ""
    advance: str = "click"  # "click" | "after_speech" | "verify"
    done_when: str = ""  # for "verify": what the screen looks like once the step is done


def parse_steps(raw: list) -> list:
    steps = []
    for item in raw or []:
        target = str(item.get("target", "")).strip()
        if not target:
            continue
        say = str(item.get("say", "")).strip() or f"Here's {target}."
        label = str(item.get("label", "")).strip() or target.removeprefix("the ").removeprefix("The ")[:24]
        advance = item.get("advance")
        advance = advance if advance in ("after_speech", "verify") else "click"
        done_when = str(item.get("done_when") or "").strip()
        steps.append(Step(target, say, label, advance, done_when))
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


def _locate_only(step: Step, task: str) -> Optional[locate.Located]:
    return locate.locate(step.target, context=f"{task}. This step: {step.say}")


def _find(step: Step, task: str) -> Optional[locate.Located]:
    found = _locate_only(step, task)
    if found is not None:
        found = locate.refine(step.target, found, guide)
    return found


def _screen_thumb():
    """Small grayscale thumbnail of the display under the pointer, or None."""
    try:
        shot, _ = grab_under_cursor(nominal=True)  # in memory, never on disk
        return shot.thumbnail()
    except Exception:
        return None


class _Prefetch:
    """Background lookup of a step that is probably about to be needed.

    Remembers the screen it started from so the result can be dropped if the
    screen has changed by the time it is wanted. The thread is a daemon and
    only reads; `cancel()` makes its result unusable."""

    def __init__(self, step: Step, task: str) -> None:
        self.step = step
        self.thumb = None
        self.found: Optional[locate.Located] = None
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, args=(task,), daemon=True)
        self._thread.start()

    def _run(self, task: str) -> None:
        try:
            self.thumb = _screen_thumb()
            if self.thumb is None or self._cancelled:
                return
            found = _locate_only(self.step, task)
            if not self._cancelled:
                self.found = found
        except Exception as e:
            print(f"[Sherpa] Guide prefetch failed ({type(e).__name__}: {e}); will look up normally.")

    def cancel(self) -> None:
        self._cancelled = True

    def take(self, still_wanted) -> Optional[locate.Located]:
        """The prefetched position if the screen is unchanged since it started, else None."""
        if self._cancelled or not still_wanted():
            return None
        now = _screen_thumb()
        if now is None:
            return None
        if not self._unchanged(now):
            print("[Sherpa] Guide: screen changed since the prefetch; looking up fresh.")
            return None
        # Unchanged so far: if the lookup is still in flight, it beats starting a duplicate.
        self._thread.join(PREFETCH_WAIT_S)
        if self._thread.is_alive() or self.found is None or self.thumb is None:
            return None
        if not self._unchanged(_screen_thumb()):
            return None
        return self.found

    def _unchanged(self, now) -> bool:
        return now is not None and self.thumb is not None and imgdiff.changed_fraction(self.thumb, now) < UNCHANGED_FRACTION


def _natural(target: str) -> str:
    """A target phrased for speech: trimmed, no trailing period, leading 'The' lowercased."""
    t = target.strip().rstrip(".")
    return "the " + t[4:] if t.startswith("The ") else t


def _not_found(steps: list, i: int, task: str, still_wanted, speak, clicks) -> Optional[locate.Located]:
    """Handle steps[i] not being found. Returns a fresh Located if a retry succeeds,
    else None (the guide should end). One retry per step, only if a step came before."""
    target = _natural(steps[i].target)
    fallback = f"I couldn't find {target}, but if you can find it for me, give me a shout and I'll show you from there."
    if i == 0:
        print(f"[Sherpa] Guide: couldn't find {steps[i].target!r} at the first step; ending.")
        speak(fallback)
        return None
    prev = steps[i - 1]
    print(f"[Sherpa] Guide: couldn't find {steps[i].target!r}; asking the user to go back to {prev.target!r}.")
    back = _find(prev, task)
    if back is not None and still_wanted():
        guide.move_to(back.x, back.y, label=prev.label)
    if not still_wanted():
        return None
    speak(f"Hmm, I don't see {target}. Can you go back to {_natural(prev.target)} and we'll take another look?")
    clicks.clear()
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    acted = False
    while still_wanted() and time.monotonic() < deadline:
        if clicks.wait(0.25) is not None:
            acted = True
            break
    if not acted:
        if still_wanted():
            print("[Sherpa] Guide: no click for a while; leaving the guide.")
        return None
    time.sleep(SETTLE_AFTER_CLICK_S)
    if not still_wanted():
        return None
    print(f"[Sherpa] Guide: looking again for {steps[i].target!r}")
    again = _find(steps[i], task)
    if again is None and still_wanted():
        print(f"[Sherpa] Guide: still can't find {steps[i].target!r}; ending.")
        speak(fallback)
    return again


def _wait_for_click(step: Step, found: locate.Located, task: str, clicks, still_wanted, filler=None) -> bool:
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
        if filler is not None:
            filler("recheck")
        again = _find(step, task)
        if again is not None and math.hypot(again.x - found.x, again.y - found.y) >= MOVED_PX:
            print(f"[Sherpa] Guide: '{step.target}' is somewhere else now; following it.")
            found = again
            guide.move_to(found.x, found.y, label=step.label)
    return False


def run(steps: list, task: str, still_wanted: Callable[[], bool], speak: Callable[[str], None], clicks=None,
        filler: Optional[Callable[[str], None]] = None) -> None:
    """Point through `steps`. `speak(text)` blocks while she talks. `clicks` is
    injectable for tests."""
    if not steps:
        return
    clicks = clicks or Clicks()
    clicks.start()
    x0, y0 = get_cursor_position()
    guide.show(x0, y0)  # she appears where the user's pointer is, then glides off
    prefetch: Optional[_Prefetch] = None
    try:
        for i, step in enumerate(steps):
            if not still_wanted():
                return
            found = None
            if prefetch is not None and prefetch.step is step:
                try:
                    early = prefetch.take(still_wanted)
                except Exception as e:
                    print(f"[Sherpa] Guide prefetch unusable ({type(e).__name__}: {e}); looking up fresh.")
                    early = None
                if early is not None:
                    print(f"[Sherpa] Guide {i + 1}/{len(steps)}: reusing prefetched lookup of {step.target!r}")
                    found = locate.refine(step.target, early, guide)
                prefetch.cancel()
                prefetch = None
            if found is None:
                print(f"[Sherpa] Guide {i + 1}/{len(steps)}: looking for {step.target!r}")
                found = _find(step, task)
            if not still_wanted():
                return
            if found is None:
                found = _not_found(steps, i, task, still_wanted, speak, clicks)
                if found is None:
                    return
            print(f"[Sherpa] Guide: pointing at {found.label or step.target!r} ({found.how}) at ({found.x:.0f}, {found.y:.0f})")
            clicks.clear()
            guide.move_to(found.x, found.y, label=step.label)
            if i + 1 < len(steps) and not _OPENS_SOMETHING.search(step.target):
                prefetch = _Prefetch(steps[i + 1], task)
            speak(step.say)
            if not still_wanted():
                return
            if step.advance == "after_speech":
                time.sleep(AFTER_SPEECH_PAUSE_S)
                continue
            if step.advance == "verify":
                result = verify.wait_until_done(step, found, task, still_wanted)
                if result == "done":
                    time.sleep(SETTLE_AFTER_CLICK_S)
                    continue
                if result == "timeout":
                    print("[Sherpa] Guide: step never looked done; ending the guide.")
                    speak("I'm not sure that worked. Ask me when you're ready and I'll take another look.")
                return
            if not _wait_for_click(step, found, task, clicks, still_wanted, filler):
                if still_wanted():
                    print("[Sherpa] Guide: no click for a while; leaving the guide.")
                return
    finally:
        if prefetch is not None:
            prefetch.cancel()
        clicks.stop()
        guide.hide()
