"""Reasoning over the screen + spoken question, via Claude."""

import base64
import io
import json
import os
import re

from anthropic import Anthropic
from PIL import Image

from . import debug
from .config import ANTHROPIC_API_KEY

MODEL = "claude-sonnet-5"
# Cheap, fast model for small structured lookups (where is X on screen). Set
# SHERPA_LOOKUP_MODEL=claude-sonnet-5 to flip back to the main model.
LOOKUP_MODEL = os.environ.get("SHERPA_LOOKUP_MODEL") or "claude-haiku-5-5"
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
    "- Several screen frames in time order. A new frame means the screen "
    "changed (a menu opened, an error appeared, another window came up). The "
    "last frame is the final state of the screen.\n"
    "- On each frame, the pointer's path is drawn as a trail from blue "
    "(earlier) to red (later). A green dot marks where the pointer was when "
    "that frame's interval began and a red ring marks where it ended. Yellow "
    "numbered boxes and rings mark things the user circled or hovered on.\n"
    "- A text timeline with the words the user said while pointing. Use it to "
    "resolve \"this\", \"that\", \"here\", \"these\" to whatever the pointer was "
    "on at that moment. If they said \"this one, then that one\", match the "
    "words to the order of the marks and the trail.\n\n"
    "- Sometimes a \"Recent activity\" block listing what the user just did (apps "
    "and windows they switched to, buttons they clicked, text they typed into "
    "fields, newest last). Use it to answer in the context of what they're doing "
    "and to keep advice current, but don't read it back to them. What they typed "
    "is sensitive: mention it only when it's relevant to their question, and "
    "never repeat it unprompted.\n\n"
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
    "and ask a short clarifying question instead of guessing.\n\n"
    "Guide mode is the DEFAULT for anything to do on screen. Whenever the user's "
    "question involves doing something on their computer that comes down to "
    "following instructions on screen (clicking, opening, selecting, finding a "
    "setting, \"how do I save this\", \"where's the setting for X\"), call the "
    "guide_steps tool right away. Sherpa then moves her own cursor to each target "
    "in turn, speaks that step's line, and waits for the user to click it before "
    "moving on. NEVER narrate, list or read out the steps in your text reply: the "
    "reply is at most one brief lead-in sentence like \"Sure, I'll walk you "
    "through it.\", or nothing at all. Plan one click per step, in order, naming "
    "each target by its visible label and kind (\"the File menu\", \"Save As in "
    "the File menu\", \"the name field in the save dialog\", \"the Save "
    "button\"); a menu and the item inside it are separate steps. Targets are "
    "looked up on the live screen when their turn comes, so later steps may be "
    "for things not showing yet. Questions that are only about understanding "
    "something (what is this, why did this happen, what does this mean) get a "
    "plain spoken answer and no tool call. Don't invent steps for things you "
    "can't tell are there."
)

GUIDE_TOOL = {
    "name": "guide_steps",
    "description": (
        "Walk the user through a task by pointing Sherpa's own cursor at each thing to click, in order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "What to point at, as a person would describe it: label and kind, e.g. 'the File menu'.",
                        },
                        "label": {"type": "string", "description": "One to three words shown on the cursor's name tag."},
                        "say": {
                            "type": "string",
                            "description": "Short spoken instruction for this step, e.g. 'Open the File menu.'",
                        },
                        "advance": {
                            "type": "string",
                            "enum": ["click", "after_speech"],
                            "description": "'click' waits for the user to click the target (default); 'after_speech' moves on after speaking, for pointing out something that isn't clicked.",
                        },
                    },
                    "required": ["target", "say"],
                },
            }
        },
        "required": ["steps"],
    },
}


def encode_pil(img: Image.Image) -> str:
    img = img.convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img = img.copy()
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_answer(messages: list) -> tuple:
    """Send a pre-built conversation (see session.py) to Claude. Returns
    (reply text, guide steps); steps is a list of dicts, empty unless Claude
    chose to walk the user through something."""
    debug.save_request(SYSTEM_PROMPT, messages)
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[GUIDE_TOOL],
    )

    steps: list = []
    for block in message.content:
        if block.type == "tool_use" and block.name == "guide_steps":
            steps = list((block.input or {}).get("steps") or [])

    parts = [block.text for block in message.content if block.type == "text"]
    text = " ".join(parts).strip()

    # An empty answer must never reach session history: the API rejects empty
    # text blocks, so it would break every later question in the session.
    if not text and steps:
        text = "Sure, I'll walk you through it."
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={message.stop_reason})")

    if message.stop_reason == "max_tokens":
        text = _trim_to_last_sentence(text)
    debug.save_answer(text if not steps else text + "\n[guide_steps] " + json.dumps(steps))
    return text, steps


LEAD_IN = "Sure, I'll walk you through it."


def stream_answer(messages: list):
    """Like get_answer, but yields events as the reply arrives so it can be
    spoken while it is still being written:
      ("text", delta)  for each piece of reply text, then once at the end
      ("done", {"text", "steps", "stop_reason"})  with the complete result.
    A reply with steps but no text gets the LEAD_IN as a text event; no text
    and no steps raises. Closing the generator closes the HTTP stream."""
    debug.save_request(SYSTEM_PROMPT, messages)
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    streamed: list = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[GUIDE_TOOL],
    ) as stream:
        for delta in stream.text_stream:
            if delta:
                streamed.append(delta)
                yield ("text", delta)
        message = stream.get_final_message()

    steps: list = []
    for block in message.content:
        if block.type == "tool_use" and block.name == "guide_steps":
            steps = list((block.input or {}).get("steps") or [])
    text = "".join(streamed).strip()
    # Same protection as get_answer: an empty answer must never reach history.
    if not text and steps:
        text = LEAD_IN
        yield ("text", text)
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={message.stop_reason})")
    debug.save_answer(text if not steps else text + "\n[guide_steps] " + json.dumps(steps))
    yield ("done", {"text": text, "steps": steps, "stop_reason": message.stop_reason})


def ask_json(system: str, content: list, max_tokens: int = 4000, model: str = "") -> dict:
    """One-off structured question (not part of the spoken conversation): the
    reply must contain a JSON object, which is parsed and returned."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model or MODEL,
        max_tokens=max_tokens,
        output_config={"effort": EFFORT},
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    text = " ".join(b.text for b in message.content if b.type == "text")
    # Smaller models sometimes talk first ("shows '{ } Python' at...") and then
    # give the JSON, so take the last object that parses rather than the span
    # from the first brace to the last.
    decoder, found, end = json.JSONDecoder(), None, 0
    for m in re.finditer(r"\{", text):
        if m.start() < end:
            continue  # inside an object already taken
        try:
            obj, end = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and obj:
            found = obj
    if found is None:
        raise RuntimeError(f"Claude returned no JSON (stop_reason={message.stop_reason}): {text[:200]!r}")
    return found


def image_block(img: Image.Image) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_pil(img)}}


def _trim_to_last_sentence(text: str) -> str:
    """Drop a trailing partial sentence so TTS doesn't stop mid-word. If
    there's no complete sentence at all, keep what we have."""
    ends = list(_SENTENCE_END.finditer(text))
    if not ends:
        return text
    return text[: ends[-1].end()].strip()
