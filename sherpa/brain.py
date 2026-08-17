"""Reasoning over the screen + spoken question, via Claude."""

import base64
import io
from pathlib import Path

from anthropic import Anthropic
from PIL import Image

from .config import ANTHROPIC_API_KEY

MODEL = "claude-sonnet-5"
MAX_DIMENSION = 1568  # Anthropic's recommended cap for image longest side

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
    "- Be concise — a few sentences, not a lecture, unless the question "
    "genuinely needs more.\n"
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
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    parts = [block.text for block in message.content if block.type == "text"]
    return " ".join(parts).strip()
