"""Completion checks for guide steps that aren't clicks (typing, scrolling...).

Polls a small region around the target, held in memory only. A cheap local
image diff gates the vision call: the model is asked only when the region has
changed since the last poll (the first poll always asks)."""

import time
from typing import Callable

import numpy as np
from PIL import Image

from . import brain, imgdiff

VERIFY_TIMEOUT_S = 15.0
VERIFY_INTERVAL_S = 2.0
REGION_W, REGION_H = 400, 300
SEND_MAX_DIM = 512  # downscale before sending, for low token use
CHANGED_FRACTION = 0.01  # thumbnail cells that must change to count as "changed"

_SYSTEM = (
    "You check whether a step of an on-screen task is complete, from a small crop of the screen. "
    'Reply with only JSON: {"done": true} or {"done": false}. Say true only if the crop clearly shows the step finished.'
)


def _grab(x: float, y: float) -> Image.Image:
    from .capture import grab_region

    return grab_region(x, y, REGION_W, REGION_H)


def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((64, 64), Image.BOX), dtype=np.float32)


def _ask(img: Image.Image, step, task: str) -> bool:
    small = img.copy()
    small.thumbnail((SEND_MAX_DIM, SEND_MAX_DIM))
    prompt = (
        f"Overall task: {task}\nStep: {step.say}\n"
        f"Step is complete when: {step.done_when or 'the step has plainly been carried out'}\n"
        "Does this region show the step complete?"
    )
    reply = brain.ask_json(_SYSTEM, [brain.image_block(small), {"type": "text", "text": prompt}], max_tokens=50, model=brain.LOOKUP_MODEL)
    done = reply.get("done")
    if not isinstance(done, bool):
        raise ValueError(f"unexpected verify reply: {reply!r}")
    return done


def wait_until_done(
    step, found, task: str, still_wanted: Callable[[], bool], timeout_s: float = VERIFY_TIMEOUT_S, interval_s: float = VERIFY_INTERVAL_S
) -> str:
    """'done' | 'timeout' | 'interrupted'."""
    deadline = time.monotonic() + timeout_s
    last = None
    while True:
        if not still_wanted():
            return "interrupted"
        if time.monotonic() >= deadline:
            return "timeout"
        try:
            img = _grab(found.x, found.y)
            thumb = _thumb(img)
            changed = last is None or imgdiff.changed_fraction(last, thumb) >= CHANGED_FRACTION
            if changed:
                answer = _ask(img, step, task)
                last = thumb  # only after a successful ask, so an error retries the same view
                if answer:
                    return "done"
        except Exception as e:
            print(f"[Sherpa] Verify poll failed ({type(e).__name__}: {e}); treating as not done.")
        time.sleep(max(0.0, min(interval_s, deadline - time.monotonic())))
