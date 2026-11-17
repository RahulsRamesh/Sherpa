"""Keyframe sampling of the screen during a gesture window.

While the user is speaking and pointing, the screen can change (a dropdown
opens, a form submits and shows an error, they switch windows). This samples
the display under the pointer in the background and keeps a frame only when
it is the first one, the pointer moved to another display, or the screen
visibly changed since the last kept frame. At most MAX_KEYFRAMES are kept.

Frames are held in memory only (PIL images, longest side <= 1568px); nothing
is written to disk. They are freed when the context is built (the overlay
consumes them) or discard() is called.

While a sampler runs it holds the session-wide ScreenBuffer (screenbuffer.py)
paused, so the two never capture at once, and its first frame reuses the
buffer's newest capture when that is under 0.4s old.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import capture, imgdiff

MAX_KEYFRAMES = 6
SAMPLES_PER_SECOND = 3  # capture start-to-start. A capture is ~30ms now, but 3/s is plenty for gestures.
# Keep a new frame when this fraction of the 64x64 thumbnail cells changed.
# Just above the ~0.003 noise floor measured on a static screen (blinking
# cursor, clock), so identical-looking screens are dropped but any real
# change, even a small popup, keeps a frame.
CHANGE_FRACTION = 0.005


@dataclass
class Keyframe:
    image: Optional[object]  # PIL RGB image (None once consumed by the overlay)
    t: float  # monotonic time the screen was captured
    display_index: int
    rect: tuple  # display bounds in global points: (x, y, w, h); image.width / rect[2] is pixels per point
    thumb: np.ndarray


class ScreenSampler:
    def __init__(self) -> None:
        self._frames: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        self._held = False
        self._hold_lock = threading.Lock()

    def start(self) -> None:
        from .screenbuffer import buffer

        buffer.hold()
        self._held = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the end of the window. One last capture happens in the
        background so the final screen state is included; call finish() to
        collect the frames."""
        self._stop.set()

    def finish(self) -> list:
        if self._thread is not None:
            self._thread.join()
        if not self._frames and self._error is not None:
            raise self._error
        return list(self._frames)

    def discard(self) -> None:
        """Free any frames still held. Safe to call more than once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._release()
        self._frames = []

    def _release(self) -> None:
        with self._hold_lock:
            if not self._held:
                return
            self._held = False
        from .screenbuffer import buffer

        buffer.release()

    def _run(self) -> None:
        try:
            first = True
            while not self._stop.is_set():
                next_at = time.monotonic() + 1.0 / SAMPLES_PER_SECOND
                self._grab(allow_seed=first)
                first = False
                self._stop.wait(max(0.0, next_at - time.monotonic()))
            self._grab()  # final state of the screen
        except Exception as e:  # surfaced by finish() only if no frame was ever kept
            self._error = e
        finally:
            self._release()

    def _grab(self, allow_seed: bool = False) -> None:
        shot, t, index, rect = None, 0.0, 0, (0, 0, 0, 0)
        if allow_seed:
            from .screenbuffer import buffer

            seed = buffer.fresh_shot()
            if seed is not None:
                shot, s = seed
                t, index, rect = s.t, s.index, s.rect
        if shot is None:
            ref = capture.display_under_cursor()
            t = time.monotonic()
            shot = capture.grab_display(ref, nominal=True)
            index, rect = ref.index, ref.rect
        thumb = shot.thumbnail()

        last = self._frames[-1] if self._frames else None
        keep = (
            last is None
            or index != last.display_index
            or imgdiff.changed_fraction(thumb, last.thumb) >= CHANGE_FRACTION
        )
        if not keep:
            return

        frame = Keyframe(shot.to_pil(), t, index, rect, thumb)
        if len(self._frames) >= MAX_KEYFRAMES:
            # Out of slots: the newest state replaces the latest kept frame.
            self._frames.pop()
        self._frames.append(frame)
