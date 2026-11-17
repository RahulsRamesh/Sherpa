"""Per-turn stopwatch: where did the time between 'you stopped talking' and
'you heard the answer' go? Printed to the terminal only, never sent anywhere.
Nothing here is allowed to raise."""

import time
from typing import Optional

from .config import TIMING


class TurnTimer:
    """Collects named timestamps (time.monotonic()) for one turn.

    `t0` is the moment the user stopped talking. Stage durations are the gaps
    between consecutive marks, so call mark() as each stage finishes:
    stt_done, context_done, claude_done (optionally claude_first_token),
    tts_first_audio (when audio was first audible; pass `at=` to override)."""

    STAGES = (
        ("stt_done", "transcribe"),
        ("context_done", "context"),
        ("claude_done", "Claude"),
        ("tts_first_audio", "TTS first audio"),
    )

    def __init__(self, t0: Optional[float] = None):
        self.t0 = time.monotonic() if t0 is None else t0
        self.marks: dict = {}
        self._last = self.t0

    def mark(self, name: str, at: Optional[float] = None) -> None:
        try:
            self.marks[name] = time.monotonic() if at is None else at
        except Exception:
            pass

    def line(self) -> str:
        prev = self.t0
        parts = []
        for key, label in self.STAGES:
            t = self.marks.get(key)
            if t is None:
                continue
            parts.append(f"{label} {max(0.0, t - prev):.1f}s")
            prev = t
        total = self.marks.get("tts_first_audio")
        if total is not None:
            parts.append(f"heard answer {max(0.0, total - self.t0):.1f}s after you stopped talking")
        return " · ".join(parts)

    def summary(self) -> None:
        """Print the one-line summary (if timing is on and the turn got as far as audio)."""
        try:
            if not TIMING or "tts_first_audio" not in self.marks:
                return
            print(f"[Sherpa] Timing: {self.line()}")
        except Exception:
            pass
