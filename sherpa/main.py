"""Sherpa entry point.

Double-tap Right Option  -> start a session (spoken greeting).
Shift + Right Option     -> ask a question about what your cursor is pointing at.
                            If Sherpa is thinking or talking, this interrupts her
                            and starts listening for your new question.
Single tap Right Option  -> stop Sherpa talking / working (no new question).
Triple-tap Right Option  -> end the session now (purges retained files).
"""

import signal
import threading
import time
import traceback
from typing import Optional

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from pynput import keyboard
from Foundation import NSTimer
from PyObjCTools import AppHelper

from . import audio, debug, overlay, stt, tts
from .config import TMP_DIR, check_config
from .pointer import PointerTracker
from .sampler import ScreenSampler
from .trail import TrailOverlay
from .session import Session

TAP_WINDOW_S = 0.4
TRIGGER_KEY = keyboard.Key.alt_r
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_r}

# Who "owns" Sherpa right now. Every question/greeting takes a new token; work
# belonging to an older token is superseded: it stops speaking and its results
# are thrown away (network calls already in flight can't be cancelled, but
# their output is discarded). Phase: idle | listening | working | speaking.
_state_lock = threading.Lock()
_token = 0
_phase = "idle"
_key_down = False
_shift_down: set = set()

_tap_count = 0
_tap_timer: Optional[threading.Timer] = None

_session = Session()
_tracker = PointerTracker()
_trail = TrailOverlay(_tracker)


def _current(token: int) -> bool:
    return token == _token


def _set_phase(token: int, phase: str) -> bool:
    """Move this token's work to `phase`. False if it has been superseded."""
    global _phase
    with _state_lock:
        if token != _token:
            return False
        _phase = phase
        return True


def _release(token: int) -> None:
    global _phase
    with _state_lock:
        if token == _token:
            _phase = "idle"


def _supersede(new_phase: str) -> tuple:
    """Take ownership for new work. Returns (token, was_active); anything
    that was working or speaking is now stale and playback is cut off."""
    global _token, _phase
    with _state_lock:
        was_active = _phase in ("working", "speaking")
        _token += 1
        _phase = new_phase
        token = _token
    audio.stop_playback()
    return token, was_active


def handle_interrupt() -> None:
    """Stop talking/working without starting a new question."""
    global _token, _phase
    with _state_lock:
        if _phase not in ("working", "speaking"):
            return
        _token += 1
        _phase = "idle"
    audio.stop_playback()
    print("[Sherpa] Interrupted.")


def _speak(text: str, token: int) -> None:
    path = tts.synthesize(text)
    try:
        if _set_phase(token, "speaking"):
            audio.play_audio_file(path, still_wanted=lambda: _current(token))
    finally:
        path.unlink(missing_ok=True)


def handle_start() -> None:
    if _session.active:
        print("[Sherpa] Session already active — ignoring.")
        return
    token, _ = _supersede("working")
    try:
        greeting = _session.start()
        print(f"\n[Sherpa] Session started. {greeting}")
        _speak(greeting, token)
    finally:
        _release(token)


def handle_end() -> None:
    if not _session.active:
        return
    _supersede("idle")  # stops any speech and makes in-flight work discard its results
    _trail.stop()
    _session.end()
    print("[Sherpa] Session ended — data purged.")
    audio.play_cue("session_end")


def handle_question() -> None:
    if not _session.active:
        print("[Sherpa] No active session — double-tap Right Option to start one.")
        audio.play_cue("error")
        return

    with _state_lock:
        if _phase == "listening":
            print("[Sherpa] Already listening — ignoring trigger.")
            return
    token, interrupted = _supersede("listening")
    if interrupted:
        print("[Sherpa] Interrupted — listening for your new question.")

    sampler = ScreenSampler()
    try:
        window_start = time.monotonic()
        print("\n[Sherpa] Triggered — listening (tracking pointer + screen)...")

        # Screen frames are sampled in the background for the whole gesture
        # window; the pointer trace comes from the always-on tracker.
        sampler.start()
        _trail.start(window_start)  # live fading cursor trail while listening
        try:
            question_audio, audio_t0 = audio.record_until_silence(on_started=audio.play_listening_cue)
        except RuntimeError as e:
            print(f"[Sherpa] {e}")
            _trail.stop()
            sampler.discard()
            if _current(token):
                audio.play_cue("error")
            return
        debug.new_turn()
        debug.save_audio(question_audio)
        window_end = time.monotonic()
        _trail.stop()
        sampler.stop()  # grabs one last frame in the background while we transcribe
        if not _set_phase(token, "working"):
            question_audio.unlink(missing_ok=True)
            return

        audio.play_cue("processing", wait=False)
        print("[Sherpa] Transcribing...")
        question, words = stt.transcribe_with_words(question_audio)
        question_audio.unlink(missing_ok=True)
        if not _current(token):
            return
        if not question:
            print("[Sherpa] Couldn't make out a question.")
            audio.play_cue("error")
            return
        print(f"[Sherpa] Question: {question}")

        keyframes = sampler.finish()
        trace = _tracker.since(window_start, window_end)
        ctx = overlay.build_context(keyframes, trace, words, window_start, window_end, audio_t0)
        gestures = [l for l in ctx.text.splitlines() if l.lstrip().startswith("[")]
        print(f"[Sherpa] {len(ctx.frames)} frame(s), {len(gestures)} gesture(s).")
        for line in gestures:
            print(f"[Sherpa]   {line.strip()}")

        print("[Sherpa] Thinking...")
        answer = _session.ask(question, ctx, is_current=lambda: _current(token))
        if not _current(token):
            print("[Sherpa] (Superseded while thinking — answer discarded.)")
            return
        print(f"[Sherpa] Answer: {answer}")

        _speak(answer, token)

    except Exception:
        if _current(token):
            print("[Sherpa] Something went wrong:")
            traceback.print_exc()
            audio.play_cue("error")
    finally:
        if _current(token):
            _trail.stop()
        sampler.discard()  # removes any frame files left by an early exit
        _release(token)


def _resolve_taps() -> None:
    global _tap_count
    count = _tap_count
    _tap_count = 0
    if count == 2:
        threading.Thread(target=handle_start, daemon=True).start()
    elif count == 3:
        threading.Thread(target=handle_end, daemon=True).start()
    elif count == 1:
        threading.Thread(target=handle_interrupt, daemon=True).start()
    # any other count (4+) is treated as noise and ignored


def on_press(key) -> None:
    global _key_down, _tap_count, _tap_timer

    if key in SHIFT_KEYS:
        _shift_down.add(key)
        return

    if key != TRIGGER_KEY or _key_down:
        return
    _key_down = True

    if _shift_down:
        if _tap_timer is not None:
            _tap_timer.cancel()
        _tap_count = 0
        threading.Thread(target=handle_question, daemon=True).start()
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
    if key == TRIGGER_KEY:
        _key_down = False


def _sweep_tmp_dir() -> None:
    for f in TMP_DIR.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)


def main() -> None:
    check_config()
    _sweep_tmp_dir()
    _tracker.start()
    print(
        "Sherpa is running.\n"
        "  Double-tap Right Option  -> start a session\n"
        "  Shift + Right Option     -> ask a question (interrupts Sherpa if she's talking)\n"
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
        listener.stop()


if __name__ == "__main__":
    main()
