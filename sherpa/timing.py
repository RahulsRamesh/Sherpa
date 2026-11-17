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
    stt_done, context_done, claude_first_token, claude_done (these two overlap
    with speech now: the answer is spoken as it streams in),
    tts_first_audio (when audio was first audible; pass `at=` to override)."""

    STAGES = (
        ("stt_done", "transcribe"),
        ("context_done", "context"),
        ("claude_first_token", "Claude first token"),
        ("claude_done", "Claude done"),
        ("tts_first_audio", "TTS first audio"),
    )

    def __init__(self, t0: Optional[float] = None):
        self.t0 = time.monotonic() if t0 is None else t0
        self.marks: dict = {}
        self._printed = False

    def mark(self, name: str, at: Optional[float] = None) -> None:
        try:
            self.marks[name] = time.monotonic() if at is None else at
        except Exception:
            pass

    def line(self) -> str:
        m = self.marks
        parts = []
        prev = self.t0
        for key, label in (("stt_done", "transcribe"), ("context_done", "context")):
            if key in m:
                parts.append(f"{label} {max(0.0, m[key] - prev):.1f}s")
                prev = m[key]
        first, done = m.get("claude_first_token"), m.get("claude_done")
        if first is not None and done is not None:
            parts.append(f"Claude first token {max(0.0, first - prev):.1f}s, done {max(0.0, done - prev):.1f}s")
        elif first is not None:
            parts.append(f"Claude first token {max(0.0, first - prev):.1f}s")
        elif done is not None:
            parts.append(f"Claude {max(0.0, done - prev):.1f}s")
        # Audio starts after the first sentence is complete, i.e. measured from the first token.
        anchor = first if first is not None else (done if done is not None else prev)
        audio = m.get("tts_first_audio")
        if audio is not None:
            parts.append(f"first sentence + TTS first audio +{max(0.0, audio - anchor):.1f}s")
            parts.append(f"heard answer {max(0.0, audio - self.t0):.1f}s after you stopped talking")
        return " · ".join(parts)

    def summary(self) -> None:
        """Print the one-line summary once (if timing is on and the turn got as far as audio)."""
        try:
            if not TIMING or self._printed or "tts_first_audio" not in self.marks:
                return
            self._printed = True
            print(f"[Sherpa] Timing: {self.line()}")
        except Exception:
            pass
