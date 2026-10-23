"""Reasoning over the screen + spoken question, via Claude."""

import base64
import io
import re
from pathlib import Path

from anthropic import Anthropic
from PIL import Image

from .config import ANTHROPIC_API_KEY

MODEL = "claude-sonnet-5"
MAX_DIMENSION = 1568  # Anthropic's recommended cap for image longest side
MAX_TOKENS = 1500  # generous headroom; the prompt keeps answers short anyway

# End of a sentence: terminal punctuation, optional closing quote/paren, then
# whitespace or end of string.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")

SYSTEM_PROMPT = (
    "You are Sherpa, a screen-aware assistant having an ongoing spoken "
    "conversation with the user about their screen. For each question you're "
    "given a cropped close-up of the area around their cursor at the moment "
    "they asked — almost always what they're actually asking about — and, "
    "when the screen has changed since the last question, a full screenshot "
    "for wider context. If the screen hasn't changed, you'll see a note "
    "saying so instead of a new full screenshot; rely on the full screenshot "
    "from earlier in the conversation in that case.\n\n"
    "Treat this as a multi-turn conversation: use earlier questions and your "
    "own earlier answers as context for follow-ups (\"what about the one "
    "below it\", \"try that again\").\n\n"
    "Your reply will be read aloud by text-to-speech, so:\n"
    "- Write in plain spoken sentences. No markdown, no bullet points, no "
    "headers, no code blocks.\n"
    "- Keep it short: usually two to four sentences. Long answers are tiring "
    "to listen to. If the question needs more, give the key point first and "
    "offer to go into more detail rather than covering everything at once.\n"
    "- Refer to things naturally (\"the button in the top right\", \"that "
    "error message\") rather than saying 'in the screenshot' or 'in the "
    "image'.\n"
    "- If it's genuinely unclear what they're pointing at, say so briefly "
    "and ask a short clarifying question instead of guessing."
)


def encode_image(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max(img.size) > MAX_DIMENSION:
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_answer(messages: list) -> str:
    """Send a pre-built conversation (see session.py) to Claude and return
    the assistant's reply text."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    parts = [block.text for block in message.content if block.type == "text"]
    text = " ".join(parts).strip()

    if message.stop_reason == "max_tokens":
        text = _trim_to_last_sentence(text)
    return text


def _trim_to_last_sentence(text: str) -> str:
    """Drop a trailing partial sentence so TTS doesn't stop mid-word. If
    there's no complete sentence at all, keep what we have."""
    ends = list(_SENTENCE_END.finditer(text))
    if not ends:
        return text
    return text[: ends[-1].end()].strip()
