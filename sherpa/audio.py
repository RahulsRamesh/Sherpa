"""Microphone recording with silence-based auto-stop, and playback."""

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
    path = SYSTEM_SOUNDS.get(name)
    if not path:
        return
    if wait:
        subprocess.run(["afplay", path], check=False)
    else:
        subprocess.Popen(["afplay", path])


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


_player_lock = threading.Lock()
_player: Optional[subprocess.Popen] = None


def play_audio_file(path: Path, still_wanted: Callable[[], bool] = lambda: True) -> None:
    """Play a file and wait for it to finish, unless stop_playback() cuts it off.

    `still_wanted` is checked under the same lock stop_playback() uses, so a
    stop that lands just before playback starts can't be missed."""
    global _player
    with _player_lock:
        if not still_wanted():
            return
        proc = subprocess.Popen(["afplay", str(path)])
        _player = proc
    proc.wait()
    with _player_lock:
        if _player is proc:
            _player = None


def stop_playback() -> None:
    """Cut off whatever play_audio_file() is currently playing, if anything."""
    with _player_lock:
        if _player is not None and _player.poll() is None:
            _player.terminate()
