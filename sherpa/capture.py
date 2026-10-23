"""Screen + cursor capture.

Captures whichever display the pointer is currently on, so questions work
on external monitors as well as the main display.
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple

import Quartz
from PIL import Image
from pynput.mouse import Controller

from .config import TMP_DIR

_mouse = Controller()

# Size (in points) of the region captured around the cursor.
CROP_WIDTH = 900
CROP_HEIGHT = 700

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


def _region_around_cursor(
    cursor_x: int, cursor_y: int, display: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Crop rect in display-local points, clamped to the display."""
    disp_x, disp_y, disp_w, disp_h = display

    w = min(CROP_WIDTH, disp_w)
    h = min(CROP_HEIGHT, disp_h)

    x = cursor_x - disp_x - w // 2
    y = cursor_y - disp_y - h // 2

    x = max(0, min(x, disp_w - w))
    y = max(0, min(y, disp_h - h))

    return x, y, w, h


def capture_context() -> Tuple[Path, Path, Tuple[int, int]]:
    """Screenshot the display under the cursor plus a crop around the cursor.

    Returns (full_screenshot_path, crop_path, cursor_position).
    """
    cursor_pos = get_cursor_position()
    ts = int(time.time() * 1000)

    full_path = TMP_DIR / f"full_{ts}.png"
    crop_path = TMP_DIR / f"crop_{ts}.png"

    index, display = _display_under_cursor(*cursor_pos)
    subprocess.run(
        ["screencapture", "-x", f"-D{index}", str(full_path)],
        check=True,
    )

    # Crop from the full capture rather than a second screencapture -R call:
    # it avoids negative global coordinates on displays left of/above the
    # main one, and guarantees both images come from the same instant.
    x, y, w, h = _region_around_cursor(*cursor_pos, display)
    with Image.open(full_path) as img:
        scale = img.width / display[2]  # pixels per point (2.0 on Retina)
        box = tuple(round(v * scale) for v in (x, y, x + w, y + h))
        img.crop(box).save(crop_path)

    return full_path, crop_path, cursor_pos
