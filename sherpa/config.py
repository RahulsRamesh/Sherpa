import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"  # "Rachel", ElevenLabs stock voice

REQUIRED_KEYS = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
}


def check_config() -> None:
    missing = [name for name, value in REQUIRED_KEYS.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required API key(s) in .env: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in."
        )


# Voice mode: stop talking when the user talks over Sherpa. Set SHERPA_BARGE_IN=0
# to turn it off, e.g. to test whether she hears herself through the speakers.
VOICE_BARGE_IN = os.environ.get("SHERPA_BARGE_IN", "1").lower() not in ("0", "false", "no")

DEBUG = os.environ.get("SHERPA_DEBUG", "").lower() in ("1", "true", "yes")

# Print a one-line per-turn latency breakdown (terminal only). Set SHERPA_TIMING=0 to hide.
TIMING = os.environ.get("SHERPA_TIMING", "1").lower() not in ("0", "false", "no")

# Activity awareness: during a session Sherpa notes what you do (app/window changes,
# clicks, what you type into fields) in memory only, to keep her advice current.
# Set SHERPA_ACTIVITY=0 to turn it off completely.
ACTIVITY = os.environ.get("SHERPA_ACTIVITY", "1").lower() not in ("0", "false", "no")

# Corrections: while a session is active, if what you just typed contradicts something you
# asked for (e.g. you said Gmail, you typed a hotmail address), Sherpa gently says so.
# Needs recent stated goal; sends only that text to Claude. SHERPA_CORRECTIONS=0 turns it off.
CORRECTIONS = os.environ.get("SHERPA_CORRECTIONS", "1").lower() not in ("0", "false", "no")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


CORRECTION_CONFIDENCE = _float_env("SHERPA_CORRECTION_CONFIDENCE", 0.8)
CORRECTION_GOAL_WINDOW_S = _float_env("SHERPA_CORRECTION_GOAL_WINDOW_S", 600.0)

TMP_DIR = Path(__file__).parent / "_tmp"
DEBUG_DIR = Path(__file__).parent / "_debug"
TMP_DIR.mkdir(exist_ok=True)
