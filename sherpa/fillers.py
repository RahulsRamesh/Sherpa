"""Short spoken filler lines ("Let me take a look at what we have here.") played
while Claude is still thinking or a guide step is being re-checked.

Every line is synthesized once, in the background at startup (warm()), and
cached as PCM so a filler can start instantly. Failing to warm is harmless:
get_pcm() synthesizes on demand, and callers that can't wait use peek_pcm().
"""

import random
import threading
from typing import Dict, Optional

from . import tts

LINES = {
    "plan": [
        "Let me take a look at what we have here.",
        "Okay, let me see what's on your screen.",
        "Give me a second to look at this.",
        "Alright, let me figure out the best way through this.",
        "One moment, I'm taking a look.",
        "Let me have a look at where we are.",
        "Okay, let me think about how to do this.",
        "Hmm, let me check what you've got open.",
        "Let me work out the steps for you.",
    ],
    "recheck": [
        "That looks different than I expected, let me take a look...",
        "Hmm, that's not quite what I expected. Let me look again.",
        "Something changed there, give me a second to check.",
        "Okay, that looks a little different. Let me see where we are.",
        "Let me re-check the screen, that's not what I was expecting.",
        "Hold on, let me take another look at this.",
    ],
}

_last: Dict[str, str] = {}
_cache: Dict[str, bytes] = {}
_lock = threading.Lock()
_warm_started = False


def pick(kind: str) -> str:
    """A random line of this kind, never the same one twice in a row."""
    lines = LINES[kind]
    with _lock:
        choices = [l for l in lines if l != _last.get(kind)] or lines
        line = random.choice(choices)
        _last[kind] = line
    return line


def peek_pcm(text: str) -> Optional[bytes]:
    """Cached PCM for this line, or None if it isn't ready (never blocks)."""
    with _lock:
        return _cache.get(text)


def get_pcm(text: str) -> bytes:
    """PCM for this line: cached, or synthesized now (blocks, may raise)."""
    pcm = peek_pcm(text)
    if pcm is None:
        pcm = tts.synthesize_pcm(text)
        with _lock:
            _cache[text] = pcm
    return pcm


def _warm_all() -> None:
    for lines in LINES.values():
        for text in lines:
            if peek_pcm(text) is not None:
                continue
            try:
                get_pcm(text)
            except Exception as e:
                print(f"[Sherpa] Filler warm-up stopped ({type(e).__name__}).")
                return  # likely no network or key; don't hammer the API


def warm() -> None:
    """Pre-synthesize every filler line in a background thread. Never raises."""
    global _warm_started
    with _lock:
        if _warm_started:
            return
        _warm_started = True
    try:
        threading.Thread(target=_warm_all, daemon=True).start()
    except Exception:
        pass
