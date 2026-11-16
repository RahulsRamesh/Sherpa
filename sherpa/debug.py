"""Opt-in debug dump of exactly what Sherpa sends to the APIs.

Off by default (Sherpa normally keeps nothing on disk). Set SHERPA_DEBUG=1
in .env to save, per question, under sherpa/_debug/<timestamp>/:

  question.wav   the audio sent to Deepgram
  request.txt    the system prompt and every message sent to Claude, in
                 order, with each image replaced by a filename
  NN_*.jpg       the exact JPEG bytes of each image sent to Claude
  answer.txt     what Claude replied (also what is sent to ElevenLabs)

These files contain your screen contents. The folder is gitignored; delete
it when you're done troubleshooting.
"""

import base64
import shutil
import time
from pathlib import Path
from typing import Optional

from .config import DEBUG, DEBUG_DIR

_current: Optional[Path] = None


def new_turn() -> None:
    """Start a fresh folder for this question."""
    global _current
    if not DEBUG:
        return
    _current = DEBUG_DIR / time.strftime("%Y%m%d-%H%M%S")
    _current.mkdir(parents=True, exist_ok=True)


def save_audio(path: Path) -> None:
    if DEBUG and _current is not None:
        shutil.copy(path, _current / "question.wav")


def save_request(system: str, messages: list) -> None:
    if not DEBUG or _current is None:
        return
    lines = ["=== SYSTEM PROMPT ===", system, ""]
    n = 0
    for m, msg in enumerate(messages):
        lines.append(f"=== MESSAGE {m + 1} ({msg['role']}) ===")
        for block in msg["content"]:
            if block["type"] == "text":
                lines.append(block["text"])
            elif block["type"] == "image":
                n += 1
                name = f"{n:02d}_msg{m + 1}.jpg"
                (_current / name).write_bytes(base64.b64decode(block["source"]["data"]))
                lines.append(f"[image: {name}]")
        lines.append("")
    (_current / "request.txt").write_text("\n".join(lines))
    print(f"[Sherpa] Debug dump: {_current}")


def save_answer(text: str) -> None:
    if DEBUG and _current is not None:
        (_current / "answer.txt").write_text(text)
