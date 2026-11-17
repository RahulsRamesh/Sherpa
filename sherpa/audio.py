"""Microphone recording with silence-based auto-stop, and playback."""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import TMP_DIR

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

CUE_GUARD_MS = 250  # ignore detection while the listening cue plays (audio is still kept)
CALIBRATION_MS = 150  # ambient-noise sample taken right after the cue guard
MAX_RMS_THRESHOLD = 1500  # cap so speech starting during calibration can't inflate the threshold
SILENCE_STOP_MS = 900  # stop after this much continuous silence once speech has started
MAX_WAIT_FOR_SPEECH_S = 6  # give up if user never starts talking
MAX_RECORD_S = 20  # hard cap on the gesture window / recording length
MIN_RMS_THRESHOLD = 300  # floor so a dead-silent room doesn't trigger on noise floor

# Bundled, trimmed copy of the macOS "Pop" sound (~0.2s, no trailing silence).
# Played in-process: spawning afplay costs ~1s of startup per call.
LISTENING_CUE_PATH = Path(__file__).parent / "sounds" / "pop.wav"
_listening_cue = sf.read(str(LISTENING_CUE_PATH), dtype="float32")

SYSTEM_SOUNDS = {
    "processing": "/System/Library/Sounds/Tink.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
    "session_end": "/System/Library/Sounds/Glass.aiff",
}


def play_listening_cue() -> None:
    """Start the 'recording started' cue without waiting for it to finish."""
    data, sr = _listening_cue
    sd.play(data, sr)


def play_cue(name: str, wait: bool = True) -> None:
    global _cue_proc
    path = SYSTEM_SOUNDS.get(name)
    if not path:
        return
    if wait:
        subprocess.run(["afplay", path], check=False)
    else:
        _cue_proc = subprocess.Popen(["afplay", path])


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


def record_until_silence(on_started: Optional[Callable[[], None]] = None) -> tuple:
    """Record from the mic, auto-stopping after a pause in speech.

    The mic stream is opened first; on_started (e.g. the listening cue) is
    called once it is live. All audio is kept from the first frame, but
    speech detection ignores the first CUE_GUARD_MS so the cue can't trigger it.

    Returns (path to a 16kHz mono WAV file, time.monotonic() when the audio
    started) so word timestamps can be lined up with the pointer trace.
    Raises RuntimeError if the user never starts speaking within
    MAX_WAIT_FOR_SPEECH_S.
    """
    frames: list[np.ndarray] = []
    calibration_frames: list[np.ndarray] = []

    state = {"phase": "cue_guard", "silence_ms": 0, "elapsed_ms": 0, "threshold": MIN_RMS_THRESHOLD}
    stop_flag = {"stop": False, "error": None}

    def callback(indata, frame_count, time_info, status):
        if stop_flag["stop"]:
            raise sd.CallbackStop()

        frame = indata[:, 0].copy()
        frames.append(frame)
        state["elapsed_ms"] += FRAME_MS

        if state["phase"] == "cue_guard":
            if state["elapsed_ms"] >= CUE_GUARD_MS:
                state["phase"] = "calibrating"
                state["elapsed_ms"] = 0
            return

        if state["phase"] == "calibrating":
            calibration_frames.append(frame)
            if state["elapsed_ms"] >= CALIBRATION_MS:
                # Quietest frame, so speech starting mid-window doesn't skew it.
                ambient_rms = min(_rms(f) for f in calibration_frames)
                state["threshold"] = min(MAX_RMS_THRESHOLD, max(MIN_RMS_THRESHOLD, ambient_rms * 3))
                state["phase"] = "waiting_for_speech"
                state["elapsed_ms"] = 0
            return

        level = _rms(frame)

        if state["phase"] == "waiting_for_speech":
            if level >= state["threshold"]:
                state["phase"] = "recording"
                state["silence_ms"] = 0
            elif state["elapsed_ms"] >= MAX_WAIT_FOR_SPEECH_S * 1000:
                stop_flag["error"] = "no_speech"
                stop_flag["stop"] = True
            return

        if state["phase"] == "recording":
            if level < state["threshold"]:
                state["silence_ms"] += FRAME_MS
            else:
                state["silence_ms"] = 0

            if state["silence_ms"] >= SILENCE_STOP_MS:
                stop_flag["stop"] = True
            elif state["elapsed_ms"] >= MAX_RECORD_S * 1000:
                stop_flag["stop"] = True

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=FRAME_SIZE,
        dtype="int16",
        callback=callback,
    ):
        started_at = time.monotonic()
        if on_started is not None:
            on_started()
        while not stop_flag["stop"]:
            time.sleep(0.02)

    if stop_flag["error"] == "no_speech":
        raise RuntimeError("Didn't hear anything — no question was asked.")

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
    out_path = TMP_DIR / f"question_{int(time.time() * 1000)}.wav"
    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return out_path, started_at


PLAYBACK_RATE = 24000  # matches tts.PCM_FORMAT
# With "low" latency at 24 kHz on a 48 kHz device, PortAudio calls the Python
# callback with ~15 frames, ~1,600 times a second: any stretch where another
# thread holds the GIL then risks a dropout. 1024-frame blocks (~43 ms) cut
# that to ~23 calls a second for ~40 ms more latency. SHERPA_PLAYBACK_BLOCK=0
# restores PortAudio's own choice.
PLAYBACK_BLOCK = int(os.environ.get("SHERPA_PLAYBACK_BLOCK", "1024") or 0)
PREBUFFER_S = 0.1  # audio queued before output starts, so tiny network jitter doesn't stutter
_BYTES_PER_S = PLAYBACK_RATE * 2

_player_lock = threading.Lock()
_current_player: Optional["PcmPlayer"] = None


class PcmPlayer:
    """Plays 24 kHz mono int16 PCM incrementally through a sounddevice stream.

    feed() as chunks arrive, finish() when there are no more, then wait().
    Output starts once ~PREBUFFER_S of audio is queued (or finish() is called),
    underruns play silence, and stop() silences output immediately.
    `still_wanted` is polled in the audio callback, feed() and wait(); once it
    returns False the player stops itself."""

    def __init__(self, still_wanted: Callable[[], bool] = lambda: True):
        global _current_player
        self._still_wanted = still_wanted
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._odd = b""
        self._finished = False
        self._stopped = False
        self._playing = False
        self._stream: Optional[sd.OutputStream] = None
        self._done = threading.Event()
        self.started_at: Optional[float] = None  # monotonic time the first audio reached the output
        self.consumed = 0  # bytes handed to the output so far (how far playback has got)
        self.underruns = 0  # callbacks that had to pad with silence mid-answer
        with _player_lock:
            if not still_wanted():
                self._stopped = True
                self._done.set()
            else:
                _current_player = self

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def done(self) -> bool:
        """Everything played, or stopped."""
        return self._done.is_set()

    def feed(self, data: bytes) -> None:
        if self._stopped:
            return
        if not self._still_wanted():
            self.stop()
            return
        data = self._odd + data
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        else:
            self._odd = b""
        if not data:
            return
        with self._lock:
            self._buf += data
            ready = len(self._buf) >= PREBUFFER_S * _BYTES_PER_S
        if ready and self._stream is None:
            self._open()

    def finish(self) -> None:
        """No more data: play what's queued, then end."""
        if self._stopped:
            return
        with self._lock:
            self._finished = True
            has_data = len(self._buf) > 0
        if not has_data:
            self._done.set()
            self._release()
        elif self._stream is None:
            self._open()

    def wait(self) -> None:
        """Block until everything has played or the player was stopped."""
        while not self._done.wait(0.02):
            if not self._still_wanted():
                self.stop()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._buf.clear()
        self._done.set()
        self._release()
        # The callback now outputs silence, so cleanup can happen off-thread
        # (closing a PortAudio stream from here can block on its callback).
        self._close_async()

    def _close_async(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return

        def close() -> None:
            try:
                stream.close()
            except Exception:
                pass

        threading.Thread(target=close, daemon=True).start()

    def _release(self) -> None:
        global _current_player
        with _player_lock:
            if _current_player is self:
                _current_player = None

    def _open(self) -> None:
        with self._lock:
            if self._stream is not None or self._stopped:
                return
            try:
                self._stream = sd.OutputStream(
                    samplerate=PLAYBACK_RATE,
                    channels=1,
                    dtype="int16",
                    latency="low",
                    blocksize=PLAYBACK_BLOCK,
                    callback=self._callback,
                    finished_callback=self._on_finished,
                )
                self._stream.start()
            except Exception:
                self._stream = None
                self._stopped = True
                self._done.set()
                raise

    def _on_finished(self) -> None:
        self._done.set()
        self._release()
        self._close_async()

    def _callback(self, outdata, frames, time_info, status) -> None:
        need = frames * 2
        with self._lock:
            if self._stopped:
                chunk = b""
            else:
                chunk = bytes(self._buf[:need])
                del self._buf[:len(chunk)]
                self.consumed += len(chunk)
                if len(chunk) < need and not self._finished and self.started_at is not None:
                    self.underruns += 1
            drained = self._finished and not self._buf
        if not self._stopped and not self._still_wanted():
            self._stopped = True
            self._done.set()
            chunk, drained = b"", True
        out = np.zeros(frames, dtype=np.int16)
        if chunk:
            out[: len(chunk) // 2] = np.frombuffer(chunk, dtype=np.int16)
            if self.started_at is None:
                # Audible once the device pulls it, plus roughly one buffer of output latency.
                self.started_at = time.monotonic() + (time_info.outputBufferDacTime - time_info.currentTime
                                                      if time_info.outputBufferDacTime else 0.0)
        outdata[:, 0] = out
        if self._stopped or drained:
            raise sd.CallbackStop()


_cue_proc: Optional[subprocess.Popen] = None


def stop_playback() -> None:
    """Silence the answer that is currently playing (and any pending cue)."""
    with _player_lock:
        player = _current_player
    if player is not None:
        player.stop()
    proc = _cue_proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
