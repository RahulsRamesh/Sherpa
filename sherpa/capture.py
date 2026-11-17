"""Screen + cursor capture.

Captures whichever display the pointer is currently on, so questions work
on external monitors as well as the main display.

Frames are grabbed straight into memory, by display ID (never by
`screencapture -D` index, so numbering can't mismatch), with no subprocess
and nothing written to disk. Backends, tried in order; the first that works
is used and logged once:

  1. Quartz CGDisplayCreateImage      ~28 ms wall, ~2 ms CPU on this Mac
  2. ScreenCaptureKit (only if pyobjc-framework-ScreenCaptureKit is
     installed)                       ~75 ms wall, ~2 ms CPU
  3. `screencapture` subprocess       ~185 ms wall, ~83 ms CPU. Writes one PNG
     to sherpa/_tmp that is read back and deleted immediately.

CGDisplayCreateImage is deprecated in macOS 15+ but still works on macOS 26
(verified); if it ever starts failing we fall through automatically.
A grab returns a lazy `Shot`: the pixels stay in the capture's own buffer
until you ask for a thumbnail (cheap, ~2 ms) or a PIL image.
"""

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import Quartz
from PIL import Image
from pynput.mouse import Controller

from .config import TMP_DIR

_mouse = Controller()

_MAX_DISPLAYS = 16
THUMB_SIZE = 64
DEFAULT_MAX_DIM = 1568  # longest side of an in-memory full frame (what Claude is sent anyway)

Rect = Tuple[int, int, int, int]


def get_cursor_position() -> Tuple[int, int]:
    x, y = _mouse.position
    return int(x), int(y)


# ---------------------------------------------------------------- displays
@dataclass(frozen=True)
class DisplayRef:
    display_id: int  # CGDirectDisplayID
    index: int  # 1-based position in the active display list (what `screencapture -D` uses)
    rect: Rect  # bounds in global points: (x, y, w, h)


def display_under_cursor(cursor_x: Optional[int] = None, cursor_y: Optional[int] = None) -> DisplayRef:
    """The display the pointer is on. Falls back to the main display if the
    pointer isn't on any active display (e.g. mid-reconfiguration)."""
    if cursor_x is None or cursor_y is None:
        cursor_x, cursor_y = get_cursor_position()
    _, active, count = Quartz.CGGetActiveDisplayList(_MAX_DISPLAYS, None, None)
    active = list(active[:count])

    point = Quartz.CGPointMake(cursor_x, cursor_y)
    _, hits, hit_count = Quartz.CGGetDisplaysWithPoint(point, 1, None, None)
    display_id = hits[0] if hit_count and hits[0] in active else Quartz.CGMainDisplayID()

    b = Quartz.CGDisplayBounds(display_id)
    rect = (int(b.origin.x), int(b.origin.y), int(b.size.width), int(b.size.height))
    index = active.index(display_id) + 1 if display_id in active else 1
    return DisplayRef(display_id, index, rect)


def _display_under_cursor(cursor_x: int, cursor_y: int) -> Tuple[int, Rect]:
    """Compatibility: (screencapture -D index, bounds in global points)."""
    d = display_under_cursor(cursor_x, cursor_y)
    return d.index, d.rect


# ---------------------------------------------------------------- Shot
class Shot:
    """One captured display image, held in memory. Wraps either a CGImage
    (lazy pixels) or a PIL image (fallback path)."""

    def __init__(self, cg=None, pil: Optional[Image.Image] = None) -> None:
        self._cg = cg
        self._pil = pil
        self._pixels: Optional[Image.Image] = None
        self._data = None
        if cg is not None:
            self.width = int(Quartz.CGImageGetWidth(cg))
            self.height = int(Quartz.CGImageGetHeight(cg))
        else:
            self.width, self.height = pil.size

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def _rgbx(self) -> Image.Image:
        """The pixels as a PIL RGBX image (zero-copy view of the capture's
        buffer; the first read of a lazy capture is where its real cost is)."""
        if self._pil is not None:
            return self._pil if self._pil.mode == "RGBX" else self._pil.convert("RGBX")
        if self._pixels is None:
            data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(self._cg))
            bpr = int(Quartz.CGImageGetBytesPerRow(self._cg))
            self._pixels = Image.frombuffer("RGBX", (self.width, self.height), data, "raw", "BGRX", bpr, 1)
            self._data = data  # keep the buffer alive under the image
        return self._pixels

    def thumbnail(self, n: int = THUMB_SIZE) -> np.ndarray:
        """n x n float32 grayscale, area-averaged (same scale as imgdiff's)."""
        img = self._rgbx().convert("L").resize((n, n), Image.BOX, reducing_gap=None)
        return np.asarray(img, dtype=np.float32)

    def to_pil(self, max_dim: Optional[int] = DEFAULT_MAX_DIM) -> Image.Image:
        """RGB PIL image, scaled so its longest side is at most `max_dim`
        (None = full pixel size). Only the size changes: image_w /
        display_rect_w is still the pixels-per-point scale."""
        w, h = self.width, self.height
        img = self._rgbx()
        if max_dim and max(w, h) > max_dim:
            s = max_dim / max(w, h)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC, reducing_gap=2.0)
        return img.convert("RGB")

    def save_png(self, path: Path) -> None:
        self.to_pil(None).save(path, "PNG")


# ---------------------------------------------------------------- backends
_state_lock = threading.Lock()
_backend_order = ["coregraphics", "screencapturekit", "screencapture"]
_failures = {"coregraphics": 0, "screencapturekit": 0}
_disabled: set = set()
_announced: Optional[str] = None
_FAILS_BEFORE_DISABLE = 3


def _grab_coregraphics(ref: DisplayRef, nominal: bool = False) -> Shot:
    cg = None
    if nominal:
        # Point-sized image (1x): ~4x fewer pixels on Retina, so ~3x less CPU.
        b = Quartz.CGDisplayBounds(ref.display_id)
        cg = Quartz.CGWindowListCreateImage(
            b, Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID, Quartz.kCGWindowImageNominalResolution
        )
    if cg is None:
        cg = Quartz.CGDisplayCreateImage(ref.display_id)
    if cg is None:
        raise RuntimeError("CGDisplayCreateImage returned no image")
    return Shot(cg=cg)


_sck = {"mod": None, "tried": False, "filters": {}}


def _grab_screencapturekit(ref: DisplayRef, nominal: bool = False) -> Shot:
    if not _sck["tried"]:
        _sck["tried"] = True
        try:
            import ScreenCaptureKit  # type: ignore

            _sck["mod"] = ScreenCaptureKit
        except Exception:
            _sck["mod"] = None
    SC = _sck["mod"]
    if SC is None:
        raise RuntimeError("ScreenCaptureKit not installed")

    def wait(call):
        ev, out = threading.Event(), {}

        def handler(value, err):
            out["v"], out["e"] = value, err
            ev.set()

        call(handler)
        if not ev.wait(5.0) or out.get("v") is None:
            raise RuntimeError(f"ScreenCaptureKit call failed: {out.get('e')}")
        return out["v"]

    entry = _sck["filters"].get(ref.display_id)
    if entry is None:
        content = wait(lambda h: SC.SCShareableContent.getShareableContentWithCompletionHandler_(h))
        disp = next((d for d in content.displays() if d.displayID() == ref.display_id), None)
        if disp is None:
            raise RuntimeError("display not in ScreenCaptureKit content")
        mode = Quartz.CGDisplayCopyDisplayMode(ref.display_id)
        cfg = SC.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(int(Quartz.CGDisplayModeGetPixelWidth(mode)))
        cfg.setHeight_(int(Quartz.CGDisplayModeGetPixelHeight(mode)))
        cfg.setShowsCursor_(False)
        filt = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(disp, [])
        entry = _sck["filters"][ref.display_id] = (filt, cfg)
    filt, cfg = entry
    try:
        cg = wait(lambda h: SC.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(filt, cfg, h))
    except Exception:
        _sck["filters"].pop(ref.display_id, None)  # display layout may have changed
        raise
    return Shot(cg=cg)


def _grab_subprocess(ref: DisplayRef, nominal: bool = False) -> Shot:
    path = TMP_DIR / f"cap_{time.monotonic_ns()}.png"
    try:
        subprocess.run(["screencapture", "-x", f"-D{ref.index}", str(path)], check=True)
        img = Image.open(path)
        img.load()
    finally:
        path.unlink(missing_ok=True)
    return Shot(pil=img)


_GRABBERS = {
    "coregraphics": _grab_coregraphics,
    "screencapturekit": _grab_screencapturekit,
    "screencapture": _grab_subprocess,
}


def capture_path() -> Optional[str]:
    """Name of the backend that served the last grab (None before any grab)."""
    return _announced


def grab_display(ref: DisplayRef, nominal: bool = False) -> Shot:
    """Capture one display by ID, in memory. Tries the fast backends first and
    falls through on error; a backend that fails repeatedly is skipped from
    then on. Raises only if every backend fails.

    nominal=True asks for a point-sized (1x) image where the backend can do
    that cheaply: plenty for change detection and for frames that are sent to
    Claude downscaled anyway. Default is the display's full pixel size."""
    global _announced
    last_err: Optional[Exception] = None
    for name in _backend_order:
        if name in _disabled:
            continue
        try:
            shot = _GRABBERS[name](ref, nominal)
        except Exception as e:
            last_err = e
            if name in _failures:
                with _state_lock:
                    _failures[name] += 1
                    if _failures[name] >= _FAILS_BEFORE_DISABLE or (name == "screencapturekit" and _sck["mod"] is None):
                        _disabled.add(name)
                        print(f"[Sherpa] Screen capture: {name} unavailable ({type(e).__name__}: {e}); not using it.")
            continue
        if name in _failures:
            _failures[name] = 0
        if _announced != name:
            _announced = name
            print(f"[Sherpa] Screen capture: using {name}" + (" (slow path: writes a temp file, deleted at once)" if name == "screencapture" else " (in memory)"))
        return shot
    raise last_err or RuntimeError("no screen capture backend available")


def grab_under_cursor(nominal: bool = False) -> Tuple[Shot, DisplayRef]:
    """Capture the display the pointer is on. Returns (shot, display)."""
    ref = display_under_cursor()
    return grab_display(ref, nominal), ref


def grab_display_under_cursor(path: Path) -> Tuple[int, Rect]:
    """Compatibility wrapper: save the pointer's display to `path` as PNG.

    Prefer `grab_under_cursor()`, which never touches the disk. Returns
    (display index, display bounds in global points)."""
    shot, ref = grab_under_cursor()
    shot.save_png(path)
    return ref.index, ref.rect
