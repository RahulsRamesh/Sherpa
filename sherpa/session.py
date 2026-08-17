"""Conversation state for a Sherpa session: start greeting, per-turn message
building (with full-screenshot dedup against imgdiff), and file cleanup.

Note: crop and full-screenshot files are deleted as soon as they've been
read into memory (their base64 lives in `turns`/the outgoing message, not
the file), so there's nothing screenshot-related left to purge by the time
a session ends — `end()` just clears in-memory state.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np

from . import brain, imgdiff

GREETINGS = ["How can I help?", "What's up?", "What do you need?"]


class Session:
    def __init__(self):
        self.active = False
        self.pending_end = False
        self.turns: list[dict] = []
        self._last_full_thumb: Optional[np.ndarray] = None

    def start(self) -> str:
        self.active = True
        self.pending_end = False
        self.turns = []
        self._last_full_thumb = None
        return random.choice(GREETINGS)

    def end(self) -> None:
        self.active = False
        self.pending_end = False
        self.turns = []
        self._last_full_thumb = None

    def ask(self, question: str, full_path: Path, crop_path: Path) -> str:
        crop_b64 = brain.encode_image(crop_path)
        crop_path.unlink(missing_ok=True)

        full_thumb = imgdiff.thumbnail_of(full_path)
        screen_changed = self._last_full_thumb is None or not imgdiff.is_similar_thumb(
            full_thumb, self._last_full_thumb
        )

        full_b64 = None
        if screen_changed:
            full_b64 = brain.encode_image(full_path)
            self._last_full_thumb = full_thumb
        full_path.unlink(missing_ok=True)

        messages = self._build_messages(question, crop_b64, full_b64)
        answer = brain.get_answer(messages)

        self.turns.append({"question": question, "crop_b64": crop_b64, "answer": answer})
        return answer

    def _build_messages(self, question: str, crop_b64: str, full_b64: Optional[str]) -> list:
        messages = []
        for turn in self.turns:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": turn["crop_b64"]},
                        },
                        {"type": "text", "text": f"Question: {turn['question']}"},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["answer"]}]})

        new_content = []
        if full_b64 is not None:
            new_content.append({"type": "text", "text": "Full screen:"})
            new_content.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": full_b64}}
            )
        else:
            new_content.append({"type": "text", "text": "Screen unchanged since your last question."})
        new_content.append({"type": "text", "text": "Close-up of where the cursor is pointing:"})
        new_content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}}
        )
        new_content.append({"type": "text", "text": f"Question: {question}"})

        messages.append({"role": "user", "content": new_content})
        return messages
