"""Reasoning over the screen + spoken question, via Claude."""

import base64
import io
import re

from anthropic import Anthropic
from PIL import Image

from .config import ANTHROPIC_API_KEY

MODEL = "claude-sonnet-5"
MAX_DIMENSION = 1568  # Anthropic's recommended cap for image longest side
# Sonnet 5 thinks by default, and thinking tokens count against max_tokens.
# At 1500, a hard question could spend the whole budget thinking and return
# no answer text at all. The prompt keeps spoken answers short; this cap is
# just room for thinking plus the reply.
MAX_TOKENS = 16000
# Low effort keeps thinking brief so spoken replies come back quickly; the
# answers are two to four sentences and rarely need deep reasoning.
EFFORT = "low"

# End of a sentence: terminal punctuation, optional closing quote/paren, then
# whitespace or end of string.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")

SYSTEM_PROMPT = (
    "You are Sherpa, a screen-aware assistant having an ongoing spoken "
    "conversation with the user about their screen. The user talks to you "
    "while using their mouse pointer the way they'd use a finger: pointing at "
    "things, circling them, hovering, moving between areas or windows, "
    "opening menus, submitting forms. For each question you get a short "
    "recording of that gesture window, not a single screenshot:\n"
    "- One to three screen frames in time order. A new frame means the screen "
    "changed (a menu opened, an error appeared, another window came up). The "
    "last frame is the final state of the screen.\n"
    "- On each frame, the pointer's path is drawn as a trail from blue "
    "(earlier) to red (later). A green dot marks where the pointer was when "
    "that frame's interval began and a red ring marks where it ended. Yellow "
    "numbered boxes and rings mark things the user circled or hovered on.\n"
    "- Full-resolution close-ups of the circled or hovered regions, and of "
    "where the pointer ended up. Read fine text from these.\n"
    "- A text timeline with the words the user said while pointing. Use it to "
    "resolve \"this\", \"that\", \"here\", \"these\" to whatever the pointer was "
    "on at that moment. If they said \"this one, then that one\", match the "
    "words to the order of the marks and the trail.\n\n"
    "Treat this as a multi-turn conversation: use earlier questions and your "
    "own earlier answers as context for follow-ups (\"what about the one "
    "below it\", \"try that again\"). Earlier turns only keep the final frame "
    "of that turn's gesture.\n\n"
    "Your reply will be read aloud by text-to-speech, so:\n"
    "- Write in plain spoken sentences. No markdown, no bullet points, no "
    "headers, no code blocks.\n"
    "- Keep it short: usually two to four sentences. Long answers are tiring "
    "to listen to. If the question needs more, give the key point first and "
    "offer to go into more detail rather than covering everything at once.\n"
    "- Refer to things naturally (\"the button in the top right\", \"that "
    "error message\") rather than saying 'in the screenshot', 'in the "
    "image', 'the trail' or 'the yellow box'.\n"
    "- If it's genuinely unclear what they're pointing at, say so briefly "
    "and ask a short clarifying question instead of guessing."
)


def encode_pil(img: Image.Image) -> str:
    img = img.convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img = img.copy()
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
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    parts = [block.text for block in message.content if block.type == "text"]
    text = " ".join(parts).strip()

    # An empty answer must never reach session history: the API rejects empty
    # text blocks, so it would break every later question in the session.
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={message.stop_reason})")

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
