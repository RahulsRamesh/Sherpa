"""Text-to-speech via ElevenLabs."""

import time
from pathlib import Path

import requests

from .config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, TMP_DIR

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
MODEL_ID = "eleven_turbo_v2_5"  # optimized for low latency


def synthesize(text: str) -> Path:
    response = requests.post(
        ELEVENLABS_URL.format(voice_id=ELEVENLABS_VOICE_ID),
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30,
    )
    response.raise_for_status()

    out_path = TMP_DIR / f"answer_{int(time.time() * 1000)}.mp3"
    out_path.write_bytes(response.content)
    return out_path
