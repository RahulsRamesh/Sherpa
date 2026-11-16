"""Sherpa entry point: hotkeys and startup. Conversation logic is in conversation.py.

Double-tap Right Option  -> start a session (spoken greeting).
Shift + Right Option     -> ask a question about what your cursor is pointing at.
                            If Sherpa is thinking or talking, this interrupts her
                            and starts listening for your new question.
Ctrl + Right Option      -> toggle voice mode: continuous listening, no hotkey needed.
Single tap Right Option  -> stop Sherpa talking / working (no new question).
Triple-tap Right Option  -> end the session now (purges retained files).
"""

import signal
import threading
from typing import Optional

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from Foundation import NSTimer
from pynput import keyboard
from PyObjCTools import AppHelper

from . import conversation as convo
from .config import TMP_DIR, check_config

TAP_WINDOW_S = 0.4
TRIGGER_KEY = keyboard.Key.alt_r
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_r}
CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}

_key_down = False
_shift_down: set = set()
_ctrl_down: set = set()

_tap_count = 0
_tap_timer: Optional[threading.Timer] = None


def _run(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def _resolve_taps() -> None:
    global _tap_count
    count = _tap_count
    _tap_count = 0
    if count == 2:
        _run(convo.handle_start)
    elif count == 3:
        _run(convo.handle_end)
    elif count == 1:
        _run(convo.handle_interrupt)
    # any other count (4+) is treated as noise and ignored


def on_press(key) -> None:
    global _key_down, _tap_count, _tap_timer

    if key in SHIFT_KEYS:
        _shift_down.add(key)
        return
    if key in CTRL_KEYS:
        _ctrl_down.add(key)
        return

    if key != TRIGGER_KEY or _key_down:
        return
    _key_down = True

    if _shift_down or _ctrl_down:
        if _tap_timer is not None:
            _tap_timer.cancel()
        _tap_count = 0
        _run(convo.toggle_voice if _ctrl_down else convo.handle_question)
        return

    _tap_count += 1
    if _tap_timer is not None:
        _tap_timer.cancel()
    _tap_timer = threading.Timer(TAP_WINDOW_S, _resolve_taps)
    _tap_timer.daemon = True
    _tap_timer.start()


def on_release(key) -> None:
    global _key_down
    if key in SHIFT_KEYS:
        _shift_down.discard(key)
        return
    if key in CTRL_KEYS:
        _ctrl_down.discard(key)
        return
    if key == TRIGGER_KEY:
        _key_down = False


def _sweep_tmp_dir() -> None:
    for f in TMP_DIR.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)


def main() -> None:
    check_config()
    _sweep_tmp_dir()
    convo.tracker.start()
    print(
        "Sherpa is running.\n"
        "  Double-tap Right Option  -> start a session\n"
        "  Shift + Right Option     -> ask a question (interrupts Sherpa if she's talking)\n"
        "  Ctrl + Right Option      -> toggle voice mode (continuous listening)\n"
        "  Single tap Right Option  -> stop Sherpa talking\n"
        "  Triple-tap Right Option  -> end the session\n"
        "Ctrl+C to quit."
    )
    # AppKit (the trail overlay) must own the main thread, so the hotkey
    # listener runs in the background and Cocoa's event loop runs here.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    # Python only runs signal handlers when it gets control, which Cocoa's loop
    # never gives back on its own: a cheap repeating timer wakes it so Ctrl+C works.
    signal.signal(signal.SIGINT, lambda *_: AppHelper.stopEventLoop())
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.25, True, lambda _t: None)
    try:
        AppHelper.runEventLoop()
    finally:
        convo._stop_voice()
        listener.stop()


if __name__ == "__main__":
    main()
