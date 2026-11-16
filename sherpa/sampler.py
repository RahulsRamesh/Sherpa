"""Keyframe sampling of the screen during a gesture window.

While the user is speaking and pointing, the screen can change (a dropdown
opens, a form submits and shows an error, they switch windows). This samples
the display under the pointer in the background and keeps a frame only when
it is the first one, the pointer moved to another display, or the screen
visibly changed since the last kept frame. At most MAX_KEYFRAMES are kept.

Frames live on disk only until the context is built (or discard() is called).
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import capture, imgdiff
from .config import TMP_DIR

MAX_KEYFRAMES = 6
SAMPLES_PER_SECOND = 3  # capture start-to-start; a capture takes ~0.3s, so this is about the ceiling
# Keep a new frame when this fraction of the 64x64 thumbnail cells changed.
# Just above the ~0.003 noise floor measured on a static screen (blinking
# cursor, clock), so identical-looking screens are dropped but any real
# change, even a small popup, keeps a frame.
CHANGE_FRACTION = 0.005


@dataclass
class Keyframe:
    path: Path
    t: float  # monotonic time the screen was captured
    display_index: int
    rect: tuple  # display bounds in global points: (x, y, w, h)
    thumb: np.ndarray


class ScreenSampler:
    def __init__(self) -> None:
        self._frames: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._counter = 0
        self._error: Optional[Exception] = None

    def start(self) -> None:
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
        """Delete any frame files still on disk. Safe to call more than once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        for f in self._frames:
            f.path.unlink(missing_ok=True)
        self._frames = []

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                next_at = time.monotonic() + 1.0 / SAMPLES_PER_SECOND
                self._grab()
                self._stop.wait(max(0.0, next_at - time.monotonic()))
            self._grab()  # final state of the screen
        except Exception as e:  # surfaced by finish() only if no frame was ever kept
            self._error = e

    def _grab(self) -> None:
        self._counter += 1
        path = TMP_DIR / f"kf_{int(time.time() * 1000)}_{self._counter}.png"
        t = time.monotonic() + 0.1  # the capture lands shortly after we ask
        index, rect = capture.grab_display_under_cursor(path)
        thumb = imgdiff.thumbnail_of(path)

        last = self._frames[-1] if self._frames else None
        keep = (
            last is None
            or index != last.display_index
            or imgdiff.changed_fraction(thumb, last.thumb) >= CHANGE_FRACTION
        )
        if not keep:
            path.unlink(missing_ok=True)
            return

        frame = Keyframe(path, t, index, rect, thumb)
        if len(self._frames) >= MAX_KEYFRAMES:
            # Out of slots: the newest state replaces the latest kept frame.
            self._frames.pop().path.unlink(missing_ok=True)
        self._frames.append(frame)
