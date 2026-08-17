"""Screen + cursor capture.

v1 limitation: assumes the pointer is on the main display. Multi-monitor
setups will clamp the crop region to the main display's bounds.
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple

import Quartz
from pynput.mouse import Controller

from .config import TMP_DIR

_mouse = Controller()

# Size (in points) of the region captured around the cursor.
CROP_WIDTH = 900
CROP_HEIGHT = 700


def get_cursor_position() -> Tuple[int, int]:
    x, y = _mouse.position
    return int(x), int(y)


def _main_display_bounds() -> Tuple[int, int, int, int]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return (
        int(bounds.origin.x),
        int(bounds.origin.y),
        int(bounds.size.width),
        int(bounds.size.height),
    )


def _region_around_cursor(cursor_x: int, cursor_y: int) -> Tuple[int, int, int, int]:
    disp_x, disp_y, disp_w, disp_h = _main_display_bounds()

    x = cursor_x - CROP_WIDTH // 2
    y = cursor_y - CROP_HEIGHT // 2

    x = max(disp_x, min(x, disp_x + disp_w - CROP_WIDTH))
    y = max(disp_y, min(y, disp_y + disp_h - CROP_HEIGHT))

    w = min(CROP_WIDTH, disp_w)
    h = min(CROP_HEIGHT, disp_h)

    return x, y, w, h


def capture_context() -> Tuple[Path, Path, Tuple[int, int]]:
    """Capture a full-screen screenshot and a cropped region around the cursor.

    Returns (full_screenshot_path, crop_path, cursor_position).
    """
    cursor_pos = get_cursor_position()
    ts = int(time.time() * 1000)

    full_path = TMP_DIR / f"full_{ts}.png"
    crop_path = TMP_DIR / f"crop_{ts}.png"

    subprocess.run(
        ["screencapture", "-x", str(full_path)],
        check=True,
    )

    x, y, w, h = _region_around_cursor(*cursor_pos)
    subprocess.run(
        ["screencapture", "-x", f"-R{x},{y},{w},{h}", str(crop_path)],
        check=True,
    )

    return full_path, crop_path, cursor_pos
