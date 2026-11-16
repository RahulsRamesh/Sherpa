"""Conversation control: who is speaking, interrupts, and message merging.

Ownership. Every question or greeting takes a new token. Work belonging to an
older token is superseded: it stops speaking and its results are thrown away
(network calls already in flight can't be cancelled, but their output is
dropped). Phases: idle | listening (hotkey recording) | working | speaking | guiding
(pointing the user through steps with her own cursor).

History and merging. The session's history only ever holds answers the user
actually started hearing. Messages that are still awaiting an answer live in
`_pending`, separate from history. If a new utterance arrives while an earlier
one is still unanswered, the earlier request is cancelled and the new message
is sent as ONE turn built from everything pending; that turn is committed to
history once, when its answer begins playing. Nothing is ever in both places,
so there is nothing to de-duplicate.

Voice mode. A toggle turns on continuous listening (voice.py). Each finished
utterance goes through the same pending/merge path. Talking over Sherpa stops
her (barge-in), with a guard that ignores her own voice coming back through
the mic.
"""

import re
import threading
import time
import traceback
from typing import Optional

from . import audio, debug, guidance, overlay, speechtext, stt, tts
from .config import VOICE_BARGE_IN
from .pointer import PointerTracker
from .sampler import ScreenSampler
from .session import Session
from .trail import TrailOverlay
from .voice import Utterance, VoiceStream

MAX_MERGED_FRAMES = 8  # cap on screen frames sent when several utterances are merged
BARGE_IN_MIN_WORDS = 2  # words heard over Sherpa before she stops talking
ECHO_OVERLAP = 0.6  # fraction of heard words found in what she's saying => it's her own voice
ECHO_TAIL_S = 2.5  # keep guarding this long after she stops (room echo, late transcripts)
CAPTURE_IDLE_TIMEOUT_S = 8.0  # drop a screen capture if speech was detected but never turned into words

session = Session()
tracker = PointerTracker()
trail = TrailOverlay(tracker)

_state_lock = threading.Lock()
_token = 0
_phase = "idle"

_pending: list = []  # Utterances heard but not yet answered aloud (guarded by _state_lock)
_last_spoken_is_turn = False  # is what's playing (or last played) a committed answer?
_spoken_text = ""
_spoken_end_at = 0.0

_voice: Optional[VoiceStream] = None
_capture: Optional[dict] = None  # screen capture for the utterance in progress
_capture_lock = threading.Lock()
_last_heard_log = ""


# ---------------------------------------------------------------- ownership
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
        was_active = _phase in ("working", "speaking", "guiding")
        if _phase == "speaking" and _last_spoken_is_turn:
            session.note_interrupted()
        _token += 1
        _phase = new_phase
        token = _token
    audio.stop_playback()
    return token, was_active


def handle_interrupt() -> None:
    """Stop talking/working without starting a new question."""
    global _token, _phase
    with _state_lock:
        if _phase not in ("working", "speaking", "guiding"):
            return
        if _phase == "speaking" and _last_spoken_is_turn:
            session.note_interrupted()
        _token += 1
        _phase = "idle"
    audio.stop_playback()
    print("[Sherpa] Interrupted.")


# ----------------------------------------------------------------- speaking
def _discard_pending_locked() -> None:
    for u in _pending:
        if u.sampler is not None:
            u.sampler.discard()  # also deletes its frame files
    _pending.clear()


def _commit_locked(turn: dict) -> None:
    """The answer is about to be heard: it becomes part of the conversation, and
    the messages that led to it are no longer pending."""
    global _last_spoken_is_turn
    session.commit(turn)
    _last_spoken_is_turn = True
    _discard_pending_locked()


def _speak(text: str, token: int, turn: Optional[dict] = None) -> None:
    """Synthesize and play `text`. If `turn` is given, it is committed to
    history at the moment playback starts (and only if still current)."""
    global _phase, _spoken_text, _spoken_end_at, _last_spoken_is_turn
    path = tts.synthesize(text)
    try:
        with _state_lock:
            if token != _token:
                return
            _phase = "speaking"
            _spoken_text = text
            _last_spoken_is_turn = False
            if turn is not None:
                _commit_locked(turn)
        audio.play_audio_file(path, still_wanted=lambda: _current(token))
    finally:
        _spoken_end_at = time.monotonic()
        path.unlink(missing_ok=True)


def _guide(token: int, turn: dict, task: str) -> None:
    """If the answer came with steps, point the user through them. Ends when
    done, or as soon as this token is superseded (interrupt, new question)."""
    steps = guidance.parse_steps(turn.get("steps"))
    if not steps or not _set_phase(token, "guiding"):
        return

    def speak(text: str) -> None:
        _speak(text, token)
        _set_phase(token, "guiding")  # _speak leaves the phase as "speaking"

    guidance.run(steps, task, still_wanted=lambda: _current(token), speak=speak)


def _is_echo(text: str) -> bool:
    """True if `text` is mostly what Sherpa is saying (or just said): her own
    voice picked up by the mic."""
    if not _spoken_text:
        return False
    if _phase != "speaking" and time.monotonic() - _spoken_end_at > ECHO_TAIL_S:
        return False
    return speechtext.overlap(text, _spoken_text) >= ECHO_OVERLAP


# ------------------------------------------------------------ session hotkeys
def handle_start() -> None:
    if session.active:
        print("[Sherpa] Session already active — ignoring.")
        return
    token, _ = _supersede("working")
    try:
        greeting = session.start()
        print(f"\n[Sherpa] Session started. {greeting}")
        _speak(greeting, token)
    finally:
        _release(token)


def handle_end() -> None:
    if not session.active:
        return
    _stop_voice()
    _supersede("idle")  # stops any speech and makes in-flight work discard its results
    trail.stop()
    with _state_lock:
        _discard_pending_locked()
    session.end()
    print("[Sherpa] Session ended — data purged.")
    audio.play_cue("session_end")


def handle_question() -> None:
    """Hotkey question: record until silence, then answer. In voice mode the
    mic is already live, so the hotkey only interrupts."""
    if _voice is not None:
        handle_interrupt()
        return
    if not session.active:
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
        trail.start(window_start)  # live fading cursor trail while listening
        try:
            question_audio, audio_t0 = audio.record_until_silence(on_started=audio.play_listening_cue)
        except RuntimeError as e:
            print(f"[Sherpa] {e}")
            trail.stop()
            sampler.discard()
            if _current(token):
                audio.play_cue("error")
            return
        debug.new_turn()
        debug.save_audio(question_audio)
        window_end = time.monotonic()
        trail.stop()
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
        trace = tracker.since(window_start, window_end)
        ctx = overlay.build_context(keyframes, trace, words, window_start, window_end, audio_t0)
        _log_context(ctx)

        print("[Sherpa] Thinking...")
        answer, turn = session.ask(question, ctx)
        if not _current(token):
            print("[Sherpa] (Superseded while thinking — answer discarded.)")
            return
        print(f"[Sherpa] Answer: {answer}")

        _speak(answer, token, turn)
        _guide(token, turn, question)

    except Exception:
        if _current(token):
            print("[Sherpa] Something went wrong:")
            traceback.print_exc()
            audio.play_cue("error")
    finally:
        if _current(token):
            trail.stop()
        sampler.discard()  # removes any frame files left by an early exit
        _release(token)


def _log_context(ctx) -> None:
    gestures = [l for l in ctx.text.splitlines() if l.lstrip().startswith("[")]
    print(f"[Sherpa] {len(ctx.frames)} frame(s), {len(gestures)} gesture(s).")
    for line in gestures:
        print(f"[Sherpa]   {line.strip()}")


# ---------------------------------------------------------------- voice mode
def voice_active() -> bool:
    return _voice is not None


def toggle_voice() -> None:
    global _voice
    if _voice is not None:
        _stop_voice()
        print("[Sherpa] Voice mode OFF.")
        audio.play_cue("session_end")
        return
    if not session.active:
        session.start()
        print("\n[Sherpa] Session started (voice mode).")
    stream = VoiceStream(_on_speech_started, _on_interim, _on_utterance)
    try:
        stream.start()
    except Exception as e:
        print(f"[Sherpa] Couldn't start the microphone: {e}")
        audio.play_cue("error")
        return
    _voice = stream
    print(
        "[Sherpa] Voice mode ON — just talk. Pause and she answers; keep talking and she "
        "merges it into one message; talk over her to stop her."
        + ("" if VOICE_BARGE_IN else " (barge-in is OFF: SHERPA_BARGE_IN=0)")
    )
    audio.play_listening_cue()


def _stop_voice() -> None:
    global _voice
    stream, _voice = _voice, None
    if stream is None:
        return
    stream.stop()
    _drop_capture()
    _supersede("idle")
    trail.stop()
    with _state_lock:
        _discard_pending_locked()


def _begin_capture(start_t: float) -> None:
    """Start sampling the screen and drawing the trail for an utterance."""
    global _capture
    with _capture_lock:
        if _capture is not None:
            return
        sampler = ScreenSampler()
        sampler.start()
        cap = {"sampler": sampler, "start": start_t, "last_activity": time.monotonic()}
        _capture = cap
    trail.start(start_t)
    threading.Thread(target=_capture_watchdog, args=(cap,), daemon=True).start()


def _capture_watchdog(cap: dict) -> None:
    """Speech was detected but nothing turned into words (a cough, a door):
    don't keep sampling the screen forever."""
    while True:
        time.sleep(1.0)
        with _capture_lock:
            if _capture is not cap:
                return
            idle = time.monotonic() - cap["last_activity"]
        if idle > CAPTURE_IDLE_TIMEOUT_S:
            _drop_capture(only=cap)
            return


def _take_capture() -> Optional[dict]:
    global _capture
    with _capture_lock:
        cap, _capture = _capture, None
    return cap


def _drop_capture(only: Optional[dict] = None) -> None:
    global _capture
    with _capture_lock:
        if _capture is None or (only is not None and _capture is not only):
            return
        cap, _capture = _capture, None
    cap["sampler"].discard()
    trail.stop()


def _on_speech_started(t: float) -> None:
    if _phase == "speaking":
        return  # might be her own voice; capture starts if real words follow (see _on_interim)
    _begin_capture(t)


def _on_interim(text: str) -> None:
    global _last_heard_log
    if _capture is not None:
        _capture["last_activity"] = time.monotonic()

    if _phase != "speaking":
        _begin_capture(time.monotonic())
        return

    # Something was heard while Sherpa is talking. This is the "does she hear
    # herself?" signal, so say what it was and what we decided.
    echo = _is_echo(text)
    n_words = len(text.split())
    if echo:
        verdict = "her own voice (ignored)"
    elif not VOICE_BARGE_IN:
        verdict = "you, but barge-in is off (ignored)"
    elif n_words < BARGE_IN_MIN_WORDS:
        verdict = "too short to count yet"
    else:
        verdict = "you — stopping her"
    line = f'[Sherpa] Voice: heard while speaking: "{text}" -> {verdict}'
    if line != _last_heard_log:
        _last_heard_log = line
        print(line)
    if echo or not VOICE_BARGE_IN or n_words < BARGE_IN_MIN_WORDS:
        return
    handle_interrupt()
    _begin_capture(time.monotonic())


def _on_utterance(utt: Utterance) -> None:
    global _token, _phase
    cap = _take_capture()
    text = utt.text.strip()

    def drop(reason: str) -> None:
        print(f'[Sherpa] Voice: ignored "{text}" ({reason})')
        if cap is not None:
            cap["sampler"].discard()
        trail.stop()

    if not text:
        return drop("empty")
    if _is_echo(text):
        return drop("her own voice")
    if _phase == "speaking" and not VOICE_BARGE_IN:
        return drop("barge-in off")

    if cap is not None:
        sampler, start_t = cap["sampler"], cap["start"]
    else:  # speech start was missed or ignored: at least grab the final screen state
        sampler, start_t = ScreenSampler(), utt.t_start
        sampler.start()
    sampler.stop()  # one last frame in the background while we get going
    utt.sampler = sampler
    utt.window_start = min(start_t, utt.t_start - 0.2)
    utt.window_end = utt.heard_at
    trail.stop()

    with _state_lock:
        if _phase == "speaking" and _last_spoken_is_turn:
            session.note_interrupted()  # talked over her before the interim check fired
        merged = len(_pending) > 0
        _token += 1
        token = _token
        _phase = "working"
        _pending.append(utt)
        snapshot = list(_pending)
    audio.stop_playback()

    print(f'\n[Sherpa] Voice: heard "{text}"')
    if merged:
        print(f"[Sherpa] Voice: merging with {len(snapshot) - 1} earlier unanswered message(s).")
    audio.play_cue("processing", wait=False)
    threading.Thread(target=_process_pending, args=(token, snapshot), daemon=True).start()


def _process_pending(token: int, snapshot: list) -> None:
    """Answer everything pending as one message. If a newer utterance arrives
    first, this is superseded and the newer call redoes it with both."""
    try:
        frames: list = []
        for u in snapshot:
            if u.keyframes is None:
                try:
                    u.keyframes = u.sampler.finish()
                except Exception as e:
                    print(f"[Sherpa] Voice: screen capture failed ({e})")
                    u.keyframes = []
            frames.extend(u.keyframes)
        if not _current(token):
            return

        frames.sort(key=lambda k: k.t)
        if len(frames) > MAX_MERGED_FRAMES:
            frames = frames[:1] + frames[-(MAX_MERGED_FRAMES - 1):]

        question = " ".join(u.text for u in snapshot)
        words = [w for u in snapshot for w in u.words]  # already absolute monotonic times
        w_start, w_end = snapshot[0].window_start, snapshot[-1].window_end
        trace = tracker.since(w_start, w_end)
        ctx = overlay.build_context(frames, trace, words, w_start, w_end, 0.0, consume=False)
        _log_context(ctx)

        debug.new_turn()
        print(f"[Sherpa] Question: {question}")
        print("[Sherpa] Thinking...")
        answer, turn = session.ask(question, ctx)
        if not _current(token):
            print("[Sherpa] (Superseded while thinking — a newer message will be answered instead.)")
            return
        print(f"[Sherpa] Answer: {answer}")
        _speak(answer, token, turn)
        _guide(token, turn, question)
    except Exception:
        if _current(token):
            print("[Sherpa] Something went wrong:")
            traceback.print_exc()
            audio.play_cue("error")
    finally:
        _release(token)
