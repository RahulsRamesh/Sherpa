"""Live, fading cursor trail drawn on screen while Sherpa is listening.

A transparent, click-through window is placed over each display and redrawn
at ~60 fps from the pointer tracker's samples. Each part of the trail fades
out FADE_S seconds after it was drawn, so the screen never gets cluttered.

The windows are marked as not shareable (NSWindowSharingNone), so they do
NOT appear in Sherpa's own screenshots; Claude only sees the annotated path
that overlay.py draws. (Verified against `screencapture` on this macOS.)

AppKit needs the main thread, so main.py runs the Cocoa event loop there and
the public start()/stop() hop onto it.
"""

import time
from typing import Optional

from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSLineCapStyleRound,
    NSMakeRect,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSTimer
from PyObjCTools import AppHelper

FADE_S = 1.5  # how long a trail segment stays visible
LINE_WIDTH = 6.0  # points, at full opacity (tapers as it fades)
COLOR = (1.0, 0.18, 0.33)  # vivid red-pink; stands out on light and dark UIs
FRAME_INTERVAL_S = 1 / 60


class _TrailView(NSView):
    """Draws fading segments. Flipped so (0, 0) is the top-left, matching the
    global pointer coordinates (origin at the primary display's top-left)."""

    def isFlipped(self):
        return True

    def drawRect_(self, _rect):
        points = getattr(self, "points", None)
        if not points:
            return
        now = self.now
        ox, oy = self.origin  # this display's top-left, in global pointer coordinates
        for a, b in zip(points, points[1:]):
            age = now - b.t
            alpha = 1.0 - age / FADE_S
            if alpha <= 0:
                continue
            path = NSBezierPath.bezierPath()
            path.moveToPoint_((a.x - ox, a.y - oy))
            path.lineToPoint_((b.x - ox, b.y - oy))
            path.setLineWidth_(1.5 + (LINE_WIDTH - 1.5) * alpha)
            path.setLineCapStyle_(NSLineCapStyleRound)
            NSColor.colorWithSRGBRed_green_blue_alpha_(*COLOR, alpha * 0.9).setStroke()
            path.stroke()


class TrailOverlay:
    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self._windows: list = []
        self._timer = None
        self._since = 0.0
        self._stopped_at: Optional[float] = None

    # ---- thread-safe entry points (called from the question-handling thread)
    def start(self, since: float) -> None:
        """Begin drawing the trail from monotonic time `since`."""
        AppHelper.callAfter(self._start, since)

    def stop(self) -> None:
        """Stop adding to the trail; what's drawn fades out on its own."""
        AppHelper.callAfter(self._stop)

    # ---- main thread only
    def _start(self, since: float) -> None:
        self._close_windows()
        self._since = since
        self._stopped_at = None
        main_h = NSScreen.screens()[0].frame().size.height
        for screen in NSScreen.screens():
            f = screen.frame()
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(f.origin.x, f.origin.y, f.size.width, f.size.height),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            win.setOpaque_(False)
            win.setBackgroundColor_(NSColor.clearColor())
            win.setHasShadow_(False)
            win.setIgnoresMouseEvents_(True)
            win.setReleasedWhenClosed_(False)
            win.setLevel_(NSScreenSaverWindowLevel)
            win.setSharingType_(NSWindowSharingNone)  # keep it out of our own screenshots
            win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
                | NSWindowCollectionBehaviorIgnoresCycle
            )
            view = _TrailView.alloc().initWithFrame_(NSMakeRect(0, 0, f.size.width, f.size.height))
            view.points = []
            view.now = time.monotonic()
            # Global (Quartz) top-left of this display: x as-is, y flipped from Cocoa's bottom-left.
            view.origin = (f.origin.x, main_h - (f.origin.y + f.size.height))
            win.setContentView_(view)
            win.orderFrontRegardless()
            self._windows.append((win, view))

        if self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                FRAME_INTERVAL_S, True, lambda _t: self._tick()
            )

    def _stop(self) -> None:
        if self._stopped_at is None:
            self._stopped_at = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        if self._stopped_at is not None and now - self._stopped_at > FADE_S:
            self._finish()
            return
        end = self._stopped_at if self._stopped_at is not None else now
        points = self._tracker.recent(max(self._since, now - FADE_S), end)
        for _win, view in self._windows:
            view.points = points
            view.now = now
            view.setNeedsDisplay_(True)

    def _finish(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._close_windows()

    def _close_windows(self) -> None:
        for win, _view in self._windows:
            win.orderOut_(None)
            win.close()
        self._windows = []
