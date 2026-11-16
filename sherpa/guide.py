"""Sherpa's own on-screen cursor, for showing the user where to go.

A transparent, click-through window covers each display and draws a blue
arrow (with a "Sherpa" tag) that glides to wherever it's told, then pulses
to say "here". It never touches the user's real mouse.

Unlike the user's fading trail (trail.py), these windows ARE visible to
screen capture (NSWindowSharingReadOnly). That's deliberate: Sherpa's later
frames show her own cursor, so the model can check where she actually landed
and refer to it ("the button my cursor is on").

Coordinates are global screen points with the origin at the primary
display's top-left: the same space as the pointer tracker and capture rects.

AppKit needs the main thread (main.py runs the Cocoa loop there). The public
methods are safe to call from any other thread; `move_to` can block until the
cursor arrives.
"""

import math
import threading
import time
from typing import Optional

from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSString,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingReadOnly,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSTimer
from PyObjCTools import AppHelper

COLOR = (0.15, 0.45, 1.0)  # distinct from the user's red-pink trail
TAG_TEXT = "Sherpa"
ARROW_SCALE = 1.6  # relative to a standard macOS arrow

MIN_MOVE_S = 0.35
MAX_MOVE_S = 1.3
SPEED_PX_PER_S = 1600.0  # longer trips take longer, up to MAX_MOVE_S
PULSE_PERIOD_S = 1.3
FRAME_INTERVAL_S = 1 / 60

# Arrow outline relative to its tip (0, 0), pointing up-left like a system cursor; y grows downward.
_ARROW = [(0, 0), (0, 16.5), (4.2, 12.8), (7.0, 19.5), (9.8, 18.3), (7.0, 11.8), (12.2, 11.8)]


def _ease(p: float) -> float:
    return 4 * p**3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


class _GuideView(NSView):
    def isFlipped(self):
        return True

    def drawRect_(self, _rect):
        st = self.state
        if st is None:
            return
        ox, oy = self.origin
        x, y = st["x"] - ox, st["y"] - oy
        if not (-60 <= x <= self.frame().size.width + 60 and -60 <= y <= self.frame().size.height + 60):
            return

        # Pulse ring, only once she's resting on a target.
        if st["resting_since"] is not None:
            phase = ((st["now"] - st["resting_since"]) % PULSE_PERIOD_S) / PULSE_PERIOD_S
            r = 10 + 30 * phase
            ring = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - r, y - r, 2 * r, 2 * r))
            ring.setLineWidth_(3.0 * (1 - phase) + 0.5)
            NSColor.colorWithSRGBRed_green_blue_alpha_(*COLOR, 0.85 * (1 - phase)).setStroke()
            ring.stroke()

        # Arrow: white outline under a blue fill so it reads on any background.
        path = NSBezierPath.bezierPath()
        for i, (px, py) in enumerate(_ARROW):
            pt = (x + px * ARROW_SCALE, y + py * ARROW_SCALE)
            path.moveToPoint_(pt) if i == 0 else path.lineToPoint_(pt)
        path.closePath()
        path.setLineJoinStyle_(1)  # round
        path.setLineWidth_(3.5)
        NSColor.whiteColor().setStroke()
        path.stroke()
        NSColor.colorWithSRGBRed_green_blue_alpha_(*COLOR, 1.0).setFill()
        path.fill()

        # Name tag.
        label = st["label"] or TAG_TEXT
        attrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(12), NSForegroundColorAttributeName: NSColor.whiteColor()}
        size = NSString.stringWithString_(label).sizeWithAttributes_(attrs)
        tx, ty = x + 16 * ARROW_SCALE, y + 18 * ARROW_SCALE
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(tx, ty, size.width + 14, size.height + 6), 9, 9
        )
        NSColor.colorWithSRGBRed_green_blue_alpha_(*COLOR, 1.0).setFill()
        pill.fill()
        NSString.stringWithString_(label).drawAtPoint_withAttributes_((tx + 7, ty + 3), attrs)


class GuideCursor:
    def __init__(self) -> None:
        self._windows: list = []
        self._timer = None
        # Animation state; only touched on the main thread (except `position`).
        self._pos = (0.0, 0.0)
        self._from = (0.0, 0.0)
        self._to = (0.0, 0.0)
        self._t0 = 0.0
        self._dur = 0.0
        self._moving = False
        self._resting_since: Optional[float] = None
        self._label: Optional[str] = None
        self._arrived: list = []  # threading.Events to set on arrival
        self._visible = False

    # ---- thread-safe entry points
    @property
    def position(self) -> Optional[tuple]:
        """Where her cursor is right now, or None when hidden. Global screen points."""
        return self._pos if self._visible else None

    def show(self, x: float, y: float) -> None:
        """Appear at (x, y) without moving there."""
        AppHelper.callAfter(self._show, x, y)

    def move_to(self, x: float, y: float, label: Optional[str] = None, wait: bool = True, timeout: float = 5.0) -> None:
        """Glide to (x, y). With `wait`, returns once she's arrived. Don't call from the main thread."""
        arrived = threading.Event()
        AppHelper.callAfter(self._move_to, x, y, label, arrived)
        if wait:
            arrived.wait(timeout)

    def hide(self) -> None:
        AppHelper.callAfter(self._hide)

    # ---- main thread only
    def _show(self, x: float, y: float) -> None:
        self._ensure_windows()
        self._pos = self._from = self._to = (float(x), float(y))
        self._moving = False
        self._resting_since = time.monotonic()
        self._visible = True
        self._ensure_timer()

    def _move_to(self, x: float, y: float, label: Optional[str], arrived: threading.Event) -> None:
        if not self._visible:
            # Enter from the pointer's side of the screen would need the tracker; start in place instead.
            self._show(x, y)
            arrived.set()
            return
        for ev in self._arrived:  # an interrupted move counts as finished for whoever was waiting
            ev.set()
        self._arrived = [arrived]
        self._label = label
        self._from, self._to = self._pos, (float(x), float(y))
        dist = math.hypot(self._to[0] - self._from[0], self._to[1] - self._from[1])
        self._dur = min(MAX_MOVE_S, max(MIN_MOVE_S, dist / SPEED_PX_PER_S))
        self._t0 = time.monotonic()
        self._moving = True
        self._resting_since = None
        self._ensure_timer()

    def _hide(self) -> None:
        self._visible = False
        self._moving = False
        for ev in self._arrived:
            ev.set()
        self._arrived = []
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._close_windows()

    def _ensure_timer(self) -> None:
        if self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                FRAME_INTERVAL_S, True, lambda _t: self._tick()
            )

    def _tick(self) -> None:
        now = time.monotonic()
        if self._moving:
            p = (now - self._t0) / self._dur
            if p >= 1.0:
                self._pos = self._to
                self._moving = False
                self._resting_since = now
                for ev in self._arrived:
                    ev.set()
                self._arrived = []
            else:
                e = _ease(p)
                self._pos = (
                    self._from[0] + (self._to[0] - self._from[0]) * e,
                    self._from[1] + (self._to[1] - self._from[1]) * e,
                )
        state = {"x": self._pos[0], "y": self._pos[1], "now": now, "resting_since": self._resting_since, "label": self._label}
        for _win, view in self._windows:
            view.state = state
            view.setNeedsDisplay_(True)

    def _ensure_windows(self) -> None:
        self._close_windows()
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
            win.setIgnoresMouseEvents_(True)  # clicks pass straight through to what's underneath
            win.setReleasedWhenClosed_(False)
            win.setLevel_(NSScreenSaverWindowLevel)
            win.setSharingType_(NSWindowSharingReadOnly)  # visible to screen capture on purpose
            win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
                | NSWindowCollectionBehaviorIgnoresCycle
            )
            view = _GuideView.alloc().initWithFrame_(NSMakeRect(0, 0, f.size.width, f.size.height))
            view.state = None
            view.origin = (f.origin.x, main_h - (f.origin.y + f.size.height))
            win.setContentView_(view)
            win.orderFrontRegardless()
            self._windows.append((win, view))

    def _close_windows(self) -> None:
        for win, _view in self._windows:
            win.orderOut_(None)
            win.close()
        self._windows = []


guide = GuideCursor()


if __name__ == "__main__":
    # Demo: a scripted tour of the screen, so the look and motion can be judged.
    import signal

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

    def tour() -> None:
        frame = NSScreen.screens()[0].frame()
        w, h = frame.size.width, frame.size.height
        stops = [(0.06 * w, 0.02 * h, "File"), (0.09 * w, 0.20 * h, "Save As…"), (0.5 * w, 0.45 * h, "Name field"), (0.6 * w, 0.62 * h, "Save")]
        guide.show(0.5 * w, 0.5 * h)
        time.sleep(0.6)
        for x, y, label in stops:
            guide.move_to(x, y, label=label)
            time.sleep(1.8)
        guide.hide()
        time.sleep(0.3)
        AppHelper.callAfter(AppHelper.stopEventLoop)

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    signal.signal(signal.SIGINT, lambda *_: AppHelper.stopEventLoop())
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.25, True, lambda _t: None)
    threading.Thread(target=tour, daemon=True).start()
    AppHelper.runEventLoop()
