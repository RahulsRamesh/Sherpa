"""Cheap local heuristic for deciding whether two screenshots are
'meaningfully' the same, so we don't resend/keep duplicate full screenshots
across turns in a session. No API calls involved."""

from pathlib import Path

import numpy as np
from PIL import Image

THUMB_SIZE = (64, 64)
DEFAULT_TOLERANCE = 6.0  # mean absolute pixel difference (0-255 scale) below which screens count as "unchanged"


def thumbnail_of(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L").resize(THUMB_SIZE, Image.LANCZOS)
        return np.asarray(img, dtype=np.float64)


def is_similar_thumb(thumb_a: np.ndarray, thumb_b: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    return float(np.mean(np.abs(thumb_a - thumb_b))) < tolerance


def is_similar(path_a: Path, path_b: Path, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    return is_similar_thumb(thumbnail_of(path_a), thumbnail_of(path_b), tolerance)


def changed_fraction(thumb_a: np.ndarray, thumb_b: np.ndarray, level: float = 10.0) -> float:
    """Fraction of thumbnail cells that differ by more than `level` (0-255).

    More sensitive to a small local change (a dropdown opening, an error
    banner) than the mean difference, while still ignoring a ticking clock
    or blinking cursor, which touch only a cell or two."""
    return float(np.mean(np.abs(thumb_a - thumb_b) > level))
