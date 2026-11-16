"""Screen + cursor capture.

Captures whichever display the pointer is currently on, so questions work
on external monitors as well as the main display.
"""

import subprocess
from pathlib import Path
from typing import Tuple

import Quartz
from pynput.mouse import Controller

_mouse = Controller()

_MAX_DISPLAYS = 16


def get_cursor_position() -> Tuple[int, int]:
    x, y = _mouse.position
    return int(x), int(y)


def _display_under_cursor(cursor_x: int, cursor_y: int) -> Tuple[int, Tuple[int, int, int, int]]:
    """Return (screencapture -D index, bounds in global points) for the cursor's display.

    screencapture numbers displays 1..N in CGGetActiveDisplayList order (main
    display first). Falls back to the main display if the cursor isn't on any
    active display (e.g. mid-reconfiguration).
    """
    _, active, count = Quartz.CGGetActiveDisplayList(_MAX_DISPLAYS, None, None)
    active = list(active[:count])

    point = Quartz.CGPointMake(cursor_x, cursor_y)
    _, hits, hit_count = Quartz.CGGetDisplaysWithPoint(point, 1, None, None)
    display_id = hits[0] if hit_count and hits[0] in active else Quartz.CGMainDisplayID()

    bounds = Quartz.CGDisplayBounds(display_id)
    rect = (
        int(bounds.origin.x),
        int(bounds.origin.y),
        int(bounds.size.width),
        int(bounds.size.height),
    )
    index = active.index(display_id) + 1 if display_id in active else 1
    return index, rect


def grab_display_under_cursor(path: Path) -> Tuple[int, Tuple[int, int, int, int]]:
    """Screenshot the display the pointer is on to `path`.

    Returns (display index, display bounds in global points) so callers can
    map global pointer coordinates onto the image.
    """
    index, rect = _display_under_cursor(*get_cursor_position())
    subprocess.run(["screencapture", "-x", f"-D{index}", str(path)], check=True)
    return index, rect
