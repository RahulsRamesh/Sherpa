"""Conversation state for a Sherpa session: start greeting, per-turn message
building from a gesture window, and in-memory history.

Nothing screen-related is ever written to disk here: the annotated frames
are encoded straight into the outgoing message, and history keeps only the
final frame of each turn's gesture (as base64 in memory). `end()` clears it.
"""

import random

from . import brain
from .overlay import GestureContext

GREETINGS = ["How can I help?", "What's up?", "What do you need?"]


def _image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


class Session:
    def __init__(self):
        self.active = False
        self.pending_end = False
        self.turns: list[dict] = []

    def start(self) -> str:
        self.active = True
        self.pending_end = False
        self.turns = []
        return random.choice(GREETINGS)

    def end(self) -> None:
        self.active = False
        self.pending_end = False
        self.turns = []

    def ask(self, question: str, ctx: GestureContext) -> str:
        content = [{"type": "text", "text": ctx.text}]
        last_frame_b64 = ""
        for label, img in ctx.frames:
            last_frame_b64 = brain.encode_pil(img)
            content.append({"type": "text", "text": f"{label}:"})
            content.append(_image_block(last_frame_b64))
        for label, img in ctx.closeups:
            content.append({"type": "text", "text": f"{label}:"})
            content.append(_image_block(brain.encode_pil(img)))
        content.append({"type": "text", "text": f"Question: {question}"})

        answer = brain.get_answer(self._history_messages() + [{"role": "user", "content": content}])
        self.turns.append({"question": question, "frame_b64": last_frame_b64, "answer": answer})
        return answer

    def _history_messages(self) -> list:
        messages = []
        for turn in self.turns:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Final screen frame from that earlier question:"},
                        _image_block(turn["frame_b64"]),
                        {"type": "text", "text": f"Question: {turn['question']}"},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["answer"]}]})
        return messages
