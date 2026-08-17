"""Speech-to-text.

Kept behind a single `transcribe(path) -> str` function so the provider
(currently Deepgram) can be swapped later without touching the rest of
the app.
"""

from pathlib import Path

import requests

from .config import DEEPGRAM_API_KEY

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def transcribe(audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = requests.post(
        DEEPGRAM_URL,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav",
        },
        params={
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true",
            "language": "en",
        },
        data=audio_bytes,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    try:
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        transcript = ""

    return transcript.strip()
