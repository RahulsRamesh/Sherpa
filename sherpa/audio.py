"""Microphone recording with silence-based auto-stop, and playback."""

import subprocess
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import TMP_DIR

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

CALIBRATION_MS = 300
SILENCE_STOP_MS = 900  # stop after this much continuous silence once speech has started
MAX_WAIT_FOR_SPEECH_S = 6  # give up if user never starts talking
MAX_RECORD_S = 30  # hard cap on recording length
MIN_RMS_THRESHOLD = 300  # floor so a dead-silent room doesn't trigger on noise floor

SYSTEM_SOUNDS = {
    "listening": "/System/Library/Sounds/Pop.aiff",
    "processing": "/System/Library/Sounds/Tink.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
    "session_end": "/System/Library/Sounds/Glass.aiff",
}


def play_cue(name: str) -> None:
    path = SYSTEM_SOUNDS.get(name)
    if path:
        subprocess.run(["afplay", path], check=False)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


def record_until_silence() -> Path:
    """Record from the mic, auto-stopping after a pause in speech.

    Returns the path to a 16kHz mono WAV file. Raises RuntimeError if the
    user never starts speaking within MAX_WAIT_FOR_SPEECH_S.
    """
    frames: list[np.ndarray] = []
    calibration_frames: list[np.ndarray] = []

    state = {"phase": "calibrating", "silence_ms": 0, "elapsed_ms": 0, "threshold": MIN_RMS_THRESHOLD}
    stop_flag = {"stop": False, "error": None}

    def callback(indata, frame_count, time_info, status):
        if stop_flag["stop"]:
            raise sd.CallbackStop()

        frame = indata[:, 0].copy()
        state["elapsed_ms"] += FRAME_MS

        if state["phase"] == "calibrating":
            calibration_frames.append(frame)
            if state["elapsed_ms"] >= CALIBRATION_MS:
                ambient_rms = _rms(np.concatenate(calibration_frames)) if calibration_frames else 0.0
                state["threshold"] = max(MIN_RMS_THRESHOLD, ambient_rms * 3)
                state["phase"] = "waiting_for_speech"
                state["elapsed_ms"] = 0
            return

        level = _rms(frame)
        frames.append(frame)

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
        while not stop_flag["stop"]:
            time.sleep(0.02)

    if stop_flag["error"] == "no_speech":
        raise RuntimeError("Didn't hear anything — no question was asked.")

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
    out_path = TMP_DIR / f"question_{int(time.time() * 1000)}.wav"
    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return out_path


def play_audio_file(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False)
