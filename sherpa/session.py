"""Conversation state for a Sherpa session: start greeting, per-turn message
building from a gesture window, and in-memory history.

Nothing screen-related is ever written to disk here: the annotated frames
are encoded straight into the outgoing message, and history keeps only the
final frame of each turn's gesture (as base64 in memory). `end()` clears it.
"""

import random

from . import brain
from .overlay import GestureContext

INTERRUPTED_NOTE = " [The user interrupted here, so they may not have heard the rest.]"
# Every request resends the history, and images dominate its size and latency.
# Only recent turns keep their frame; older ones are text, which is enough for "what about that one".
FRAMES_KEPT_TURNS = 2
MAX_HISTORY_TURNS = 12
GREETINGS = ["How can I help?", "What's up?", "What do you need?"]


def _image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


class Session:
    def __init__(self):
        self.active = False
        self.turns: list[dict] = []

    def start(self) -> str:
        self.active = True
        self.turns = []
        return random.choice(GREETINGS)

    def end(self) -> None:
        self.active = False
        self.turns = []

    def ask_stream(self, question: str, ctx: GestureContext) -> tuple:
        """Start asking Claude about this turn. Returns (events, turn): `events`
        is brain.stream_answer's iterator and `turn` an empty record the caller
        fills in as the answer is SPOKEN (see add_spoken / finish_turn).

        Nothing is added to history here. The caller commits the turn when the
        first sentence starts playing, and history only ever holds the
        sentences the user began to hear, so an answer that was cancelled,
        merged into a newer message, or cut off leaves no trace of what they
        never heard."""
        content = [{"type": "text", "text": ctx.text}]
        last_frame_b64 = ""
        for label, img in ctx.frames:
            last_frame_b64 = brain.encode_pil(img)
            content.append({"type": "text", "text": f"{label}:"})
            content.append(_image_block(last_frame_b64))
        content.append({"type": "text", "text": f"Question: {question}"})

        events = brain.stream_answer(self._history_messages() + [{"role": "user", "content": content}])
        return events, {"question": question, "frame_b64": last_frame_b64, "answer": "", "steps": []}

    @staticmethod
    def add_spoken(turn: dict, sentence: str) -> None:
        """A sentence has begun playing: it is now part of what the user heard."""
        turn["answer"] = f"{turn['answer']} {sentence}".strip()

    @staticmethod
    def finish_turn(turn: dict, steps: list) -> None:
        """The whole answer was spoken and Claude finished: record the steps, so
        follow-ups ("ok what's next?") know what she walked them through."""
        turn["steps"] = steps
        if steps:
            route = " -> ".join(str(st.get("target", "")) for st in steps)
            turn["answer"] += f" [I then pointed them through: {route}]"

    def commit(self, turn: dict) -> None:
        self.turns.append(turn)
        del self.turns[:-MAX_HISTORY_TURNS]
        for old in self.turns[:-FRAMES_KEPT_TURNS]:
            old["frame_b64"] = ""  # free the memory too

    def note_interrupted(self) -> None:
        """The user cut off the last answer mid-speech: tell the model, so it
        doesn't assume they heard all of it."""
        if self.turns and not self.turns[-1]["answer"].endswith(INTERRUPTED_NOTE):
            self.turns[-1]["answer"] += INTERRUPTED_NOTE

    def _history_messages(self) -> list:
        messages = []
        for turn in self.turns:
            content = []
            if turn["frame_b64"]:
                content += [
                    {"type": "text", "text": "Final screen frame from that earlier question:"},
                    _image_block(turn["frame_b64"]),
                ]
            content.append({"type": "text", "text": f"Question: {turn['question']}"})
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["answer"]}]})
        return messages
