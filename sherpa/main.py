"""Sherpa entry point.

Double-tap Right Option  -> start a session (spoken greeting).
Shift + Right Option     -> ask a question about what your cursor is pointing at.
Triple-tap Right Option  -> end the session (purges retained files).
"""

import threading
import traceback
from typing import Optional

from pynput import keyboard

from . import audio, capture, stt, tts
from .config import TMP_DIR, check_config
from .session import Session

TAP_WINDOW_S = 0.4
TRIGGER_KEY = keyboard.Key.alt_r
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_r}

_busy = threading.Lock()
_key_down = False
_shift_down: set = set()

_tap_count = 0
_tap_timer: Optional[threading.Timer] = None

_session = Session()


def _speak(text: str) -> None:
    path = tts.synthesize(text)
    audio.play_audio_file(path)
    path.unlink(missing_ok=True)


def handle_start() -> None:
    if _session.active:
        print("[Sherpa] Session already active — ignoring.")
        return
    greeting = _session.start()
    print(f"\n[Sherpa] Session started. {greeting}")
    _speak(greeting)


def handle_end() -> None:
    if not _session.active:
        return
    if not _busy.acquire(blocking=False):
        print("[Sherpa] Busy — will end the session once the current question finishes.")
        _session.pending_end = True
        return
    try:
        _session.end()
        print("[Sherpa] Session ended — data purged.")
        audio.play_cue("session_end")
    finally:
        _busy.release()


def handle_question() -> None:
    if not _session.active:
        print("[Sherpa] No active session — double-tap Right Option to start one.")
        audio.play_cue("error")
        return

    if not _busy.acquire(blocking=False):
        print("[Sherpa] Still working on the last question — ignoring trigger.")
        return

    try:
        print("\n[Sherpa] Triggered — capturing screen...")
        full_path, crop_path, cursor_pos = capture.capture_context()
        print(f"[Sherpa] Cursor at {cursor_pos}. Listening...")

        audio.play_cue("listening")
        try:
            question_audio = audio.record_until_silence()
        except RuntimeError as e:
            print(f"[Sherpa] {e}")
            audio.play_cue("error")
            return

        audio.play_cue("processing")
        print("[Sherpa] Transcribing...")
        question = stt.transcribe(question_audio)
        question_audio.unlink(missing_ok=True)
        if not question:
            print("[Sherpa] Couldn't make out a question.")
            audio.play_cue("error")
            return
        print(f"[Sherpa] Question: {question}")

        print("[Sherpa] Thinking...")
        answer = _session.ask(question, full_path, crop_path)
        print(f"[Sherpa] Answer: {answer}")

        _speak(answer)

    except Exception:
        print("[Sherpa] Something went wrong:")
        traceback.print_exc()
        audio.play_cue("error")
    finally:
        _busy.release()
        if _session.pending_end:
            handle_end()


def _resolve_taps() -> None:
    global _tap_count
    count = _tap_count
    _tap_count = 0
    if count == 2:
        threading.Thread(target=handle_start, daemon=True).start()
    elif count == 3:
        threading.Thread(target=handle_end, daemon=True).start()
    # any other count (1, 4+) is treated as noise and ignored


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
    print(
        "Sherpa is running.\n"
        "  Double-tap Right Option  -> start a session\n"
        "  Shift + Right Option     -> ask a question\n"
        "  Triple-tap Right Option  -> end the session\n"
        "Ctrl+C to quit."
    )
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
