"""Text-to-speech via ElevenLabs, streamed as raw PCM so playback can start at
the first bytes instead of after the whole clip is generated."""

from typing import Iterator

import requests

from .config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
MODEL_ID = "eleven_v4_turbo"  # v4 pronunciation (reads math well), lower latency
PCM_FORMAT = "pcm_24000"  # raw signed 16-bit little-endian mono, 24 kHz
SAMPLE_RATE = 24000
CHUNK_BYTES = 4096


def stream_pcm(text: str) -> Iterator[bytes]:
    """Yield PCM chunks (int16 LE mono 24 kHz) as ElevenLabs generates them.

    Every chunk has an even length, so each one is whole int16 samples.
    Raises RuntimeError with the server's message on an HTTP error."""
    response = requests.post(
        ELEVENLABS_STREAM_URL.format(voice_id=ELEVENLABS_VOICE_ID),
        params={"output_format": PCM_FORMAT},
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
        json={
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        stream=True,
        timeout=30,
    )
    try:
        if response.status_code >= 400:
            detail = response.text[:300].strip()
            raise RuntimeError(f"ElevenLabs TTS failed (HTTP {response.status_code}): {detail}")
        carry = b""
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            chunk = carry + chunk
            if len(chunk) % 2:
                chunk, carry = chunk[:-1], chunk[-1:]
            else:
                carry = b""
            if chunk:
                yield chunk
    finally:
        response.close()


def synthesize_pcm(text: str) -> bytes:
    """The whole clip as one PCM byte string."""
    return b"".join(stream_pcm(text))
