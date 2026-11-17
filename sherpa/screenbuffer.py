"""Session-long, low-rate, memory-only record of what the screen has been doing.

While a Sherpa session is active, `buffer` samples the pointer's display about
once a second and keeps the last ~30 seconds: a 64x64 thumbnail for every
sample (for change detection) and a full-size frame only for the few samples
where the screen visibly changed. Nothing is ever written to disk, and
`stop()` purges everything.

Cost control:
  * Adaptive rate. After several unchanged samples the interval backs off
    (1s -> up to 3s); any change snaps it back to the base rate.
  * Paused while the screen is locked, the display is asleep or the screen
    saver is running (nothing worth looking at, and no business capturing it).
  * Self-limiting: the interval is never shorter than 20x the time a sample
    took, so the buffer cannot use more than ~5% of one core.
  * Full frames are downscaled to <=1568px (~5 MB each), at most MAX_FULL (6)
    kept, plus the single newest frame.

Hand-off with ScreenSampler (sampler.py), which samples at ~3 fps while the
user speaks: ScreenSampler.start() calls `buffer.hold()` and releases it when
its thread ends (or on discard), so the two never capture at the same time.
The sampler also seeds its first frame from `buffer.fresh_shot()` when the
buffer's newest sample is under 0.4s old, saving one capture. Nothing needs to
be wired in conversation.py for that; it only needs `buffer.start()` when a
session begins and `buffer.stop()` when it ends.

Kill switch: SHERPA_SCREEN_BUFFER=0 makes start() a no-op. SHERPA_IDLE_FPS
sets the base rate (default 1.0).
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import Quartz

from . import capture, imgdiff
from .sampler import Keyframe

ENABLED = os.environ.get("SHERPA_SCREEN_BUFFER", "1").lower() not in ("0", "false", "no")


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except ValueError:
        return default


IDLE_FPS = _env_float("SHERPA_IDLE_FPS", 1.0)
HISTORY_S = 30.0  # how long samples are kept
MAX_FULL = 6  # full-size frames kept in the ring
MAX_INTERVAL_S = 3.0  # slowest the adaptive rate gets
BACKOFF_AFTER = 4  # unchanged samples before the interval starts growing
BACKOFF_FACTOR = 1.5
CPU_BUDGET = 0.05  # interval >= work_time / CPU_BUDGET
CHANGE_FRACTION = 0.005  # same noise-floor reasoning as sampler.CHANGE_FRACTION
ON_CHANGE_MIN_GAP_S = 1.0  # rate limit for on_change callbacks
FRESH_S = 0.4  # how old a sample may be to stand in for a new capture


@dataclass
class _Sample:
    t: float  # time.monotonic() of the capture
    display_id: int
    index: int
    rect: tuple
    thumb: np.ndarray  # 64x64 float32
    frame: Optional[object] = None  # PIL image, only for kept samples


@dataclass
class ChangeEvent:
    t: float
    fraction: float  # fraction of thumbnail cells that changed since the previous sample
    display_changed: bool  # the pointer moved to another display (fraction is then meaningless)


def screen_inactive() -> bool:
    """True when the screen is locked, the display is asleep, or the screen
    saver is up. Cheap (~0.1 ms) except the screen saver check."""
    try:
        d = Quartz.CGSessionCopyCurrentDictionary()
        if d is not None:
            if d.get("CGSSessionScreenIsLocked"):
                return True
            if d.get("kCGSSessionOnConsoleKey") is False or d.get("kCGSessionOnConsoleKey") == 0:
                return True  # another user's session (fast user switching)
        if Quartz.CGDisplayIsAsleep(Quartz.CGMainDisplayID()):
            return True
    except Exception:
        return False
    return _screensaver_running()


_saver_cache = {"t": 0.0, "v": False}


def _screensaver_running() -> bool:
    now = time.monotonic()
    if now - _saver_cache["t"] < 5.0:  # asking NSWorkspace every tick is wasteful
        return _saver_cache["v"]
    v = False
    try:
        from AppKit import NSWorkspace

        v = any(
            (a.bundleIdentifier() or "") in ("com.apple.ScreenSaver.Engine", "com.apple.loginwindow.ScreenSaver")
            for a in NSWorkspace.sharedWorkspace().runningApplications()
        )
    except Exception:
        pass
    _saver_cache.update(t=now, v=v)
    return v


class ScreenBuffer:
    def __init__(self, idle_fps: float = IDLE_FPS, history_s: float = HISTORY_S, max_full: int = MAX_FULL) -> None:
        self.base_interval = 1.0 / idle_fps
        self.history_s = history_s
        self.max_full = max_full
        self._lock = threading.RLock()
        self._samples: deque = deque()
        self._latest_shot: Optional[capture.Shot] = None
        self._latest_sample: Optional[_Sample] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._holds = 0
        self._callbacks: list = []
        self._last_callback_t = 0.0
        self._settle_pending = False
        self.interval = self.base_interval
        self.sample_count = 0  # total captures since start (for tests/diagnostics)
        self.stats = {"work_s": 0.0}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Begin sampling (idempotent). No-op when SHERPA_SCREEN_BUFFER=0."""
        if not ENABLED:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="screenbuffer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop sampling and purge every frame, thumbnail and callback."""
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5.0)
        with self._lock:
            self._thread = None
            self._samples.clear()
            self._latest_shot = None
            self._latest_sample = None
            self._callbacks.clear()
            self._settle_pending = False
            self.interval = self.base_interval

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ hand-off
    def hold(self) -> None:
        """Pause sampling until a matching release() (reference counted)."""
        with self._lock:
            self._holds += 1

    def release(self) -> None:
        with self._lock:
            self._holds = max(0, self._holds - 1)

    # ------------------------------------------------------------ consumers
    def on_change(self, callback: Callable[[ChangeEvent], None]) -> Callable[[], None]:
        """Call `callback(ChangeEvent)` from the buffer thread when the screen
        changes significantly (rate-limited to one call per second). Keep it
        quick or hand off to another thread. Returns an unsubscribe function."""
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def latest(self) -> Optional[Keyframe]:
        """Newest sample as a Keyframe (full frame converted on demand), or None."""
        with self._lock:
            s, shot = self._latest_sample, self._latest_shot
        if s is None or shot is None:
            return None
        return Keyframe(shot.to_pil(), s.t, s.index, s.rect, s.thumb)

    def fresh_shot(self, max_age: float = FRESH_S):
        """(Shot, DisplayRef-like sample) of the newest sample if it is younger
        than `max_age` seconds, else None. Lets ScreenSampler skip a capture."""
        with self._lock:
            s, shot = self._latest_sample, self._latest_shot
        if s is None or shot is None or time.monotonic() - s.t > max_age:
            return None
        return shot, s

    def recent(self, seconds: float = HISTORY_S, max_frames: int = MAX_FULL) -> list:
        """Change-triggered keyframes from the last `seconds`, oldest first,
        consecutive near-duplicates removed, newest `max_frames` kept."""
        cutoff = time.monotonic() - seconds
        with self._lock:
            kept = [s for s in self._samples if s.frame is not None and s.t >= cutoff]
        out: list = []
        for s in kept:
            if out and out[-1].display_index == s.index and imgdiff.changed_fraction(out[-1].thumb, s.thumb) < CHANGE_FRACTION:
                continue
            out.append(Keyframe(s.frame, s.t, s.index, s.rect, s.thumb))
        return out[-max_frames:]

    def thumbnail_changed_since(self, t: float, fraction: float = CHANGE_FRACTION) -> bool:
        """Has the screen changed between the sample at/just before time `t`
        (time.monotonic()) and the newest sample? Uses thumbnails only. False
        when there's no data; True if the pointer's display differs."""
        with self._lock:
            if self._latest_sample is None:
                return False
            base = None
            for s in self._samples:
                if s.t <= t:
                    base = s
                else:
                    break
            if base is None:  # nothing that old: compare against the oldest we have
                base = self._samples[0]
            now = self._latest_sample
        if base is now:
            return False
        if base.display_id != now.display_id:
            return True
        return imgdiff.changed_fraction(base.thumb, now.thumb) >= fraction

    def memory_bytes(self) -> int:
        """Approximate bytes held: thumbnails + full PIL frames + the latest capture."""
        with self._lock:
            total = sum(s.thumb.nbytes for s in self._samples)
            for s in self._samples:
                if s.frame is not None:
                    total += s.frame.width * s.frame.height * 3
            if self._latest_shot is not None:
                total += self._latest_shot.width * self._latest_shot.height * 4
        return total

    # ------------------------------------------------------------ the loop
    def _run(self) -> None:
        unchanged = 0
        while not self._stop.is_set():
            if self._holds > 0 or screen_inactive():
                self._stop.wait(0.25 if self._holds > 0 else 1.0)
                continue
            began = time.monotonic()
            changed = False
            try:
                changed = self._sample()
            except Exception as e:
                if not getattr(self, "_warned", False):
                    self._warned = True
                    print(f"[Sherpa] Screen buffer: capture failed ({type(e).__name__}: {e}); will keep retrying quietly.")
                changed = False
            work = time.monotonic() - began
            self.stats["work_s"] += work
            if changed:
                unchanged = 0
                self.interval = self.base_interval
            else:
                unchanged += 1
                if unchanged > BACKOFF_AFTER:
                    self.interval = min(MAX_INTERVAL_S, self.interval * BACKOFF_FACTOR)
            wait = max(self.interval - work, work / CPU_BUDGET - work, 0.0)
            self._stop.wait(wait)

    def _sample(self) -> bool:
        """Capture once; returns True if the screen changed since the previous sample."""
        ref = capture.display_under_cursor()
        shot = capture.grab_display(ref, nominal=True)
        t = time.monotonic()
        thumb = shot.thumbnail()
        sample = _Sample(t, ref.display_id, ref.index, ref.rect, thumb)
        self.sample_count += 1

        with self._lock:
            if self._stop.is_set():
                return False
            prev = self._latest_sample
            display_changed = prev is not None and prev.display_id != sample.display_id
            frac = 1.0 if prev is None or display_changed else imgdiff.changed_fraction(prev.thumb, thumb)
            changed = prev is not None and frac >= CHANGE_FRACTION

            # Keep a full frame for the first sample, on change, and once the
            # screen settles after a change (the frame that matters is usually
            # the one after the animation).
            keep = prev is None or changed
            if not keep and self._settle_pending:
                keep = True
            self._settle_pending = changed
            if keep:
                sample.frame = shot.to_pil()
            self._samples.append(sample)
            self._latest_sample, self._latest_shot = sample, shot
            self._trim(t)

        if changed:
            self._fire(ChangeEvent(t, frac, display_changed))
        return changed

    def _trim(self, now: float) -> None:
        while self._samples and now - self._samples[0].t > self.history_s:
            self._samples.popleft()
        full = [s for s in self._samples if s.frame is not None]
        for s in full[: max(0, len(full) - self.max_full)]:
            s.frame = None

    def _fire(self, ev: ChangeEvent) -> None:
        with self._lock:
            if ev.t - self._last_callback_t < ON_CHANGE_MIN_GAP_S:
                return
            self._last_callback_t = ev.t
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(ev)
            except Exception as e:
                print(f"[Sherpa] Screen buffer: on_change callback failed ({type(e).__name__}: {e})")


buffer = ScreenBuffer()
