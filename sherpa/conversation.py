"""Conversation control: who is speaking, interrupts, and message merging.

Ownership. Every question or greeting takes a new token. Work belonging to an
older token is superseded: it stops speaking and its results are thrown away
(network calls already in flight can't be cancelled, but their output is
dropped). Phases: idle | listening (hotkey recording) | working | speaking | guiding
(pointing the user through steps with her own cursor).

History and merging. The session's history only ever holds what the user
actually heard. Messages that are still awaiting an answer live in `_pending`,
separate from history. If a new utterance arrives while an earlier one is still
unanswered, the earlier request is cancelled and the new message is sent as ONE
turn built from everything pending; that turn is committed to history once,
when its FIRST sentence starts playing (the committed answer is then empty and
grows). Nothing is ever in both places, so there is nothing to de-duplicate.

Streaming. The answer is spoken as Claude writes it (_speak_events): a reader
thread cuts the text stream into sentences (sentences.py) and starts each
sentence's TTS fetch immediately; the speaking thread feeds the sentences, in
order, into ONE PcmPlayer, so a later sentence is already buffered while an
earlier one plays. Each sentence is appended to turn["answer"] (under
_state_lock, only if still current) when the player's output position passes
its first byte, so on an interrupt the stored answer is exactly the sentences
that had begun playing, plus INTERRUPTED_NOTE (session.note_interrupted); the
rest is dropped. The guide-steps suffix is added only once everything was
spoken. Superseding stops the player, the fetch threads and the Claude stream.

Corrections. corrections.py may speak ONE sentence on its own when what the
user typed contradicts what they asked for (deliver_correction). It takes a
token like any new work, but only when the phase is idle or guiding, nothing is
pending and no voice capture is in progress (checked atomically with taking the
token). A running guide is superseded by it. It is committed as a synthetic
turn (turn["synthetic"], empty frame) when it starts playing.

Voice mode. A toggle turns on continuous listening (voice.py). Each finished
utterance goes through the same pending/merge path. Talking over Sherpa stops
her (barge-in), with a guard that ignores her own voice coming back through
the mic.
"""

import queue
import re
import threading
import time
import traceback
from collections import deque
from typing import Optional

from . import audio, corrections, debug, fillers, guidance, overlay, speechtext, stt, tts
from .timing import TurnTimer
from .config import VOICE_BARGE_IN
from .pointer import PointerTracker
from .sampler import ScreenSampler
from .sentences import SentenceSplitter
from .session import Session
from .trail import TrailOverlay
from .voice import Utterance, VoiceStream

MAX_PARALLEL_TTS = 2  # sentences whose audio is fetched at the same time
MAX_MERGED_FRAMES = 8  # cap on screen frames sent when several utterances are merged
BARGE_IN_MIN_WORDS = 2  # words heard over Sherpa before she stops talking
ECHO_OVERLAP = 0.6  # fraction of heard words found in what she's saying => it's her own voice
ECHO_TAIL_S = 2.5  # keep guarding this long after she stops (room echo, late transcripts)
FILLER_DELAY_S = 0.6  # play a "plan" filler if Claude's first text hasn't arrived by then
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


class _Job:
    """One sentence on its way to the speakers: a fetch thread fills `chunks`
    with PCM, then None (end), or an Exception."""

    def __init__(self, text: str):
        self.text = text
        self.chunks: queue.Queue = queue.Queue()


_tts_slots = threading.Semaphore(MAX_PARALLEL_TTS)  # ElevenLabs limits concurrent requests


def _fetch_job(job: _Job, cancel: threading.Event) -> None:
    """Fetch one sentence's audio (started the moment the sentence exists, so
    it is already buffered by the time the previous sentence finishes)."""
    chunks = None
    try:
        with _tts_slots:
            if cancel.is_set():
                return
            chunks = tts.stream_pcm(job.text)
            for chunk in chunks:
                if cancel.is_set():
                    return
                job.chunks.put(chunk)
        job.chunks.put(None)
    except Exception as e:
        job.chunks.put(e)
    finally:
        if chunks is not None:
            chunks.close()


def _read_events(events, token: int, cancel: threading.Event, jobs: queue.Queue, result: dict,
                 timer: Optional[TurnTimer], got_text: threading.Event) -> None:
    """Consume Claude's stream: cut it into sentences and start fetching each
    one's audio immediately. Ends the stream early if superseded."""
    splitter = SentenceSplitter()

    def submit(sentences) -> None:
        for sentence in sentences:
            job = _Job(sentence)
            threading.Thread(target=_fetch_job, args=(job, cancel), daemon=True).start()
            jobs.put(job)

    try:
        for kind, payload in events:
            if cancel.is_set() or not _current(token):
                return
            if kind == "text":
                got_text.set()
                if timer is not None and "claude_first_token" not in timer.marks:
                    timer.mark("claude_first_token")
                submit(splitter.feed(payload))
            elif kind == "done":
                result.update(payload)
                if timer is not None:
                    timer.mark("claude_done")
                print(f"[Sherpa] Answer: {payload['text']}")
        if not cancel.is_set() and _current(token):
            submit(splitter.flush())
    except Exception as e:
        result["error"] = e
    finally:
        try:
            events.close()
        except Exception:
            pass
        jobs.put(None)  # no more sentences


def _speak_events(events, token: int, turn: Optional[dict] = None, timer: Optional[TurnTimer] = None,
                  filler: Optional[str] = None) -> dict:
    """Speak a text stream as it arrives (events as from brain.stream_answer),
    one sentence at a time with no gap: later sentences are fetched while
    earlier ones play, all fed into one PcmPlayer. Audio starts at the first
    bytes of the first sentence. Returns the stream's final result dict.

    If `turn` is given it is committed to history when the first sentence
    starts playing, and each sentence is added to turn["answer"] as it starts
    playing, so history holds exactly what the user heard. Everything stops
    (and is discarded) as soon as `token` is superseded.

    `filler` (a fillers kind, e.g. "plan"): if Claude's first text hasn't
    arrived FILLER_DELAY_S after we start, a cached filler line plays first
    through the same player; the answer queues behind it. The filler is never
    added to history. It is skipped if its audio isn't cached yet."""
    global _phase, _spoken_text, _spoken_end_at, _last_spoken_is_turn
    cancel = threading.Event()
    jobs: queue.Queue = queue.Queue()
    result: dict = {}
    got_text = threading.Event()
    reader = threading.Thread(target=_read_events, args=(events, token, cancel, jobs, result, timer, got_text),
                              daemon=True)
    reader.start()

    def wanted() -> bool:
        return not cancel.is_set() and _current(token)

    player = None
    fed = 0  # bytes given to the player so far
    starts: deque = deque()  # (byte offset, sentence) not yet heard
    first_audio_marked = timer is None
    committed = False
    started = time.monotonic()
    filler_pending = filler is not None

    def start_filler() -> None:
        """Begin the player with a cached filler line (not part of history)."""
        nonlocal player, fed, filler_pending
        global _phase, _spoken_text, _last_spoken_is_turn
        filler_pending = False
        text = fillers.pick(filler)
        pcm = fillers.peek_pcm(text)
        if pcm is None:
            return
        with _state_lock:
            if token != _token:
                return
            _phase = "speaking"
            _spoken_text = text  # echo guard
            _last_spoken_is_turn = False  # nothing committed: an interrupt notes nothing
        player = audio.PcmPlayer(still_wanted=wanted)
        player.feed(pcm)
        fed += len(pcm)
        print(f"[Sherpa] Filler: {text}")

    def tick() -> bool:
        """Housekeeping while we wait: sentences whose audio has begun
        playing become part of the heard answer. False once superseded."""
        nonlocal first_audio_marked, committed
        if not wanted():
            return False
        if filler_pending and player is None and not got_text.is_set() \
                and time.monotonic() - started >= FILLER_DELAY_S:
            start_filler()
        if player is None:
            return True
        if not first_audio_marked and player.started_at is not None:
            timer.mark("tts_first_audio", at=player.started_at)
            first_audio_marked = True
        if first_audio_marked and timer is not None and "claude_done" in timer.marks:
            timer.summary()
        consumed = player.consumed
        while starts and starts[0][0] < consumed:
            _, sentence = starts.popleft()
            if turn is None:
                continue
            with _state_lock:
                if token != _token:
                    return False
                if not committed:
                    committed = True
                    _commit_locked(turn)
                session.add_spoken(turn, sentence)
        return True

    def take(q: queue.Queue):
        """Next item from q, ticking while we wait; None if superseded."""
        while True:
            try:
                return q.get(timeout=0.02)
            except queue.Empty:
                if not tick():
                    return _STOP

    try:
        while True:
            job = take(jobs)
            if job is _STOP:
                return result
            if job is None:
                break
            first_of_job = True
            while True:
                chunk = take(job.chunks)
                if chunk is _STOP:
                    return result
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                if player is None:
                    with _state_lock:  # first audio bytes are in hand
                        if token != _token:
                            return result
                        _phase = "speaking"
                        _spoken_text = ""
                        _last_spoken_is_turn = False
                    player = audio.PcmPlayer(still_wanted=wanted)
                if first_of_job:
                    first_of_job = False
                    starts.append((fed, job.text))
                    _spoken_text = f"{_spoken_text} {job.text}".strip()  # echo guard: what she is saying now
                player.feed(chunk)
                fed += len(chunk)
                if player.stopped:
                    return result
        if player is not None:
            player.finish()
            while not player.done:
                time.sleep(0.02)
                if not tick():
                    return result
            tick()  # the last sentences began playing within the final ticks
        if wanted() and turn is not None and committed and "steps" in result:
            with _state_lock:
                if token == _token:
                    session.finish_turn(turn, result["steps"])
        if "error" in result:
            raise result["error"]
        return result
    finally:
        cancel.set()  # fetch threads and the Claude stream wind down
        if player is not None:
            player.stop()  # no-op once drained; silences it if we bailed early
        _spoken_end_at = time.monotonic()
        if timer is not None:
            timer.summary()


_STOP = object()


def _speak(text: str, token: int) -> None:
    """Speak a fixed line (greeting, guide steps) through the same machinery."""
    _speak_events(iter([("text", text)]), token)


def _guide(token: int, turn: dict, task: str) -> None:
    """If the answer came with steps, point the user through them. Ends when
    done, or as soon as this token is superseded (interrupt, new question)."""
    # TODO(streaming): the guide_steps tool call only arrives after the spoken
    # text is complete, so step 1 starts after the lead-in. To start earlier,
    # parse the partial tool JSON (input_json deltas) and begin step 1 as soon as
    # its first step object is complete, while the lead-in is still playing.
    steps = guidance.parse_steps(turn.get("steps"))
    if not steps or not _set_phase(token, "guiding"):
        return

    def speak(text: str) -> None:
        _speak(text, token)
        _set_phase(token, "guiding")  # _speak leaves the phase as "speaking"

    def filler(kind: str) -> None:
        speak(fillers.pick(kind))  # same machinery: interruptible, not in history, phase restored

    guidance.run(steps, task, still_wanted=lambda: _current(token), speak=speak, filler=filler)


def _can_correct() -> bool:
    """Is now a polite moment for a correction? Idle or guiding only, and the
    user isn't talking (voice-mode capture) or waiting on an answer."""
    return _phase in ("idle", "guiding") and not _pending and _capture is None


def deliver_correction(say: str, question: str) -> bool:
    """Speak ONE correction sentence (see corrections.py). Takes a fresh token
    only if the moment is right, checked atomically with taking it; a running
    guide is superseded (the correction ends it). The sentence is committed as a
    synthetic turn when it starts playing, so follow-ups know about it."""
    global _token, _phase
    with _state_lock:
        if not _can_correct():
            return False
        was_guiding = _phase == "guiding"
        _token += 1
        _phase = "working"
        token = _token
    turn = {"question": question, "frame_b64": "", "answer": "", "steps": [], "synthetic": True}
    try:
        _speak_events(iter([("text", say)]), token, turn)
        if was_guiding and turn["answer"]:
            turn["answer"] += " [This interrupted the step-by-step walkthrough, which stopped.]"
    except Exception as e:
        print(f"[Sherpa] Couldn't speak the correction: {type(e).__name__}")
        return False
    finally:
        _release(token)
    return True


corrections.bind(lambda: session.turns, lambda: [u.text for u in _pending], _can_correct, deliver_correction)


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
        corrections.start()
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
    corrections.stop()
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
        timer = TurnTimer(window_end)  # the clock starts when recording ended
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
        timer.mark("stt_done")

        keyframes = sampler.finish()
        trace = tracker.since(window_start, window_end)
        ctx = overlay.build_context(keyframes, trace, words, window_start, window_end, audio_t0)
        _log_context(ctx)
        timer.mark("context_done")

        print("[Sherpa] Thinking...")
        events, turn = session.ask_stream(question, ctx)
        _speak_events(events, token, turn, timer, filler="plan")  # speaks as the answer streams in
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
        corrections.start()
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
        last = snapshot[-1]
        # Clock starts when the last utterance's final word ended; the gap to
        # heard_at (endpointing wait) shows up as the "transcribe" stage.
        t0 = last.t_end if last.t_end <= last.heard_at else last.heard_at
        timer = TurnTimer(t0)
        timer.mark("stt_done", at=last.heard_at)
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
        timer.mark("context_done")

        debug.new_turn()
        print(f"[Sherpa] Question: {question}")
        print("[Sherpa] Thinking...")
        events, turn = session.ask_stream(question, ctx)
        _speak_events(events, token, turn, timer, filler="plan")  # speaks as the answer streams in
        _guide(token, turn, question)
    except Exception:
        if _current(token):
            print("[Sherpa] Something went wrong:")
            traceback.print_exc()
            audio.play_cue("error")
    finally:
        _release(token)
