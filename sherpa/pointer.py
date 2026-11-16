"""Pointer tracking and gesture detection.

A background mouse listener keeps a short rolling buffer of (time, x, y)
samples in memory. Each question asks for the slice covering its gesture
window, and `detect_gestures` finds the two things that matter most for
"pointing like a finger": circling something, and hovering on something.

All coordinates are global screen points (the same space as pynput and
Quartz). Times are `time.monotonic()` seconds.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from pynput import mouse

MIN_SAMPLE_INTERVAL_S = 0.008  # throttle: ~125 Hz is plenty
MAX_BUFFER_POINTS = 6000  # ~45s at the throttle rate; far more than a 20s window

# Hover: the pointer stays within DWELL_RADIUS of where it settled.
DWELL_RADIUS = 35.0
DWELL_MIN_S = 0.5

# Circle: the path turns through nearly a full revolution and comes back near
# where it started. These are untested starting values; tune against real use.
LOOP_STEP = 8.0  # resample the path at this spacing before measuring turning
LOOP_MIN_TURN = 2 * math.pi * 0.85
LOOP_MIN_SIZE = 40.0
LOOP_MAX_CLOSURE = 0.6  # end-to-start gap, as a fraction of the loop's size
LOOP_MAX_PATH_RATIO = 2.5  # path length / (pi * size); rejects long wandering spirals


@dataclass(frozen=True)
class TracePoint:
    t: float
    x: float
    y: float


@dataclass(frozen=True)
class Gesture:
    kind: str  # "circle" | "dwell"
    t0: float
    t1: float
    cx: float
    cy: float
    bbox: tuple  # (x0, y0, x1, y1)


class PointerTracker:
    """Always-on, in-memory rolling pointer buffer. Nothing is written to disk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: deque = deque(maxlen=MAX_BUFFER_POINTS)
        self._controller = mouse.Controller()
        self._listener: Optional[mouse.Listener] = None

    def start(self) -> None:
        x, y = self._controller.position
        self._append(time.monotonic(), x, y, force=True)
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.daemon = True
        self._listener.start()

    def _on_move(self, x: float, y: float) -> None:
        self._append(time.monotonic(), x, y)

    def _append(self, t: float, x: float, y: float, force: bool = False) -> None:
        with self._lock:
            if not force and self._points and t - self._points[-1].t < MIN_SAMPLE_INTERVAL_S:
                return
            self._points.append(TracePoint(t, float(x), float(y)))

    def since(self, start: float, end: Optional[float] = None) -> list:
        """Trace for [start, end], beginning and ending with a point at the
        window edges so a stationary pointer still yields a path."""
        end = time.monotonic() if end is None else end
        with self._lock:
            pts = list(self._points)
        before = [p for p in pts if p.t < start]
        inside = [p for p in pts if start <= p.t <= end]

        out: list = []
        if before:
            out.append(TracePoint(start, before[-1].x, before[-1].y))
        elif not inside:
            x, y = self._controller.position
            out.append(TracePoint(start, float(x), float(y)))
        out.extend(inside)
        if out[-1].t < end:
            out.append(TracePoint(end, out[-1].x, out[-1].y))
        return out


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _dwells(points: list) -> list:
    gestures = []
    n = len(points)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and _dist(points[j + 1].x, points[j + 1].y, points[i].x, points[i].y) <= DWELL_RADIUS:
            j += 1
        # The pointer left between points[j] and points[j+1]; until then it was resting.
        t_end = points[j + 1].t if j + 1 < n else points[j].t
        if t_end - points[i].t >= DWELL_MIN_S:
            cluster = points[i : j + 1]
            cx = sum(p.x for p in cluster) / len(cluster)
            cy = sum(p.y for p in cluster) / len(cluster)
            bbox = (
                min(p.x for p in cluster),
                min(p.y for p in cluster),
                max(p.x for p in cluster),
                max(p.y for p in cluster),
            )
            gestures.append(Gesture("dwell", points[i].t, t_end, cx, cy, bbox))
            i = j + 1
        else:
            i += 1
    return gestures


def _resample(points: list, step: float) -> list:
    out = [points[0]]
    for p in points[1:]:
        if _dist(p.x, p.y, out[-1].x, out[-1].y) >= step:
            out.append(p)
    return out


def _loops(points: list) -> list:
    if len(points) < 3:
        return []
    pts = _resample(points, LOOP_STEP)
    gestures = []
    start = 0
    cum = 0.0
    for k in range(2, len(pts)):
        a = math.atan2(pts[k - 1].y - pts[k - 2].y, pts[k - 1].x - pts[k - 2].x)
        b = math.atan2(pts[k].y - pts[k - 1].y, pts[k].x - pts[k - 1].x)
        turn = (b - a + math.pi) % (2 * math.pi) - math.pi

        # Turning in the opposite direction while barely turned: not a loop in progress.
        if cum != 0.0 and turn * cum < 0 and abs(cum) < 0.5:
            start, cum = k - 1, 0.0
        cum += turn

        if abs(cum) < LOOP_MIN_TURN:
            continue
        seg = pts[start : k + 1]
        xs = [p.x for p in seg]
        ys = [p.y for p in seg]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        size = max(w, h)
        closure = _dist(seg[0].x, seg[0].y, seg[-1].x, seg[-1].y)
        path_len = sum(_dist(seg[m].x, seg[m].y, seg[m + 1].x, seg[m + 1].y) for m in range(len(seg) - 1))
        if (
            size >= LOOP_MIN_SIZE
            and closure <= LOOP_MAX_CLOSURE * size
            and path_len <= LOOP_MAX_PATH_RATIO * math.pi * size
        ):
            gestures.append(
                Gesture(
                    "circle",
                    seg[0].t,
                    seg[-1].t,
                    (min(xs) + max(xs)) / 2,
                    (min(ys) + max(ys)) / 2,
                    (min(xs), min(ys), max(xs), max(ys)),
                )
            )
        start, cum = k, 0.0
    return gestures


def detect_gestures(points: list) -> list:
    """Circles and hovers in a trace, ordered by start time."""
    if len(points) < 2:
        return []
    return sorted(_dwells(points) + _loops(points), key=lambda g: g.t0)
