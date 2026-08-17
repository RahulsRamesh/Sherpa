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


TMP_DIR = Path(__file__).parent / "_tmp"
TMP_DIR.mkdir(exist_ok=True)
