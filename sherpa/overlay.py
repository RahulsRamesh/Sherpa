"""Turn a gesture window (keyframes + pointer trace + timed words) into what
Claude sees: trail-annotated screen frames and a text timeline tying speech
to pointer movement.

Everything is built in memory; the keyframe files are deleted once loaded.
"""

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .pointer import Gesture, TracePoint, detect_gestures
from .sampler import Keyframe

TRAIL_START = (40, 120, 255)  # earlier = blue
TRAIL_END = (255, 50, 50)  # later = red
MARK_COLOR = (255, 214, 10)  # circled / hovered places
START_COLOR = (40, 200, 90)

EVENT_PAD = 0.3  # seconds of slack when matching spoken words to an event


@dataclass
class GestureContext:
    text: str
    frames: list  # [(label, PIL.Image)] annotated keyframes, chronological


def _lerp(c0: tuple, c1: tuple, p: float) -> tuple:
    return tuple(int(a + (b - a) * p) for a, b in zip(c0, c1))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: fixed-size default font
        return ImageFont.load_default()


def _in_rect(x: float, y: float, rect: tuple) -> bool:
    return rect[0] <= x < rect[0] + rect[2] and rect[1] <= y < rect[1] + rect[3]


def _runs(points: list, rect: tuple) -> list:
    """Split a trace into runs of consecutive points on this display, so the
    trail isn't drawn across stretches spent on another display."""
    runs, cur = [], []
    for p in points:
        if _in_rect(p.x, p.y, rect):
            cur.append(p)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _render_frame(raw: Image.Image, rect: tuple, points: list, marks: list) -> Image.Image:
    """Draw the pointer trail and numbered circle/hover marks onto a copy of `raw`."""
    scale = raw.width / rect[2]

    def px(x: float, y: float) -> tuple:
        return ((x - rect[0]) * scale, (y - rect[1]) * scale)

    layer = Image.new("RGBA", raw.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = max(3, round(4 * scale))
    font = _font(max(14, round(15 * scale)))

    runs = _runs(points, rect)
    if points:
        span = max(points[-1].t - points[0].t, 1e-6)
        for run in runs:
            for a, b in zip(run, run[1:]):
                p = (a.t - points[0].t) / span
                d.line([px(a.x, a.y), px(b.x, b.y)], fill=_lerp(TRAIL_START, TRAIL_END, p) + (215,), width=width)

        # Start marker: only if the interval actually began on this display.
        if runs and runs[0][0] is points[0]:
            sx, sy = px(points[0].x, points[0].y)
            r = 7 * scale
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=START_COLOR + (255,), outline=(255, 255, 255, 255), width=max(2, round(2 * scale)))
        # End marker: only if the interval ended on this display.
        if runs and runs[-1][-1] is points[-1]:
            ex, ey = px(points[-1].x, points[-1].y)
            r = 12 * scale
            d.ellipse([ex - r, ey - r, ex + r, ey + r], outline=TRAIL_END + (255,), width=max(3, round(3 * scale)))

    for num, g in marks:
        cx, cy = px(g.cx, g.cy)
        if g.kind == "dwell":
            r = 26 * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=MARK_COLOR + (255,), width=max(3, round(3 * scale)))
            lx, ly = cx + r * 0.7, cy - r * 0.7
        else:
            x0, y0 = px(g.bbox[0], g.bbox[1])
            x1, y1 = px(g.bbox[2], g.bbox[3])
            d.rectangle([x0, y0, x1, y1], outline=MARK_COLOR + (255,), width=max(2, round(2 * scale)))
            lx, ly = x0, y0
        lr = 11 * scale
        d.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=MARK_COLOR + (255,))
        d.text((lx, ly), str(num), fill=(0, 0, 0, 255), font=font, anchor="mm")

    return Image.alpha_composite(raw.convert("RGBA"), layer).convert("RGB")


def _where(x: float, y: float, rect: tuple) -> str:
    fx, fy = (x - rect[0]) / rect[2], (y - rect[1]) / rect[3]
    horiz = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else "center"
    vert = "top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else "middle"
    area = "center" if (horiz, vert) == ("center", "middle") else f"{vert} {horiz}" if horiz != "center" else f"{vert} center"
    return f"{area} ({round(fx * 100)}% across, {round(fy * 100)}% down)"


def _said(words: list, t0: float, t1: float, origin: float) -> str:
    """Words spoken during [t0, t1] (monotonic), given each word's start/end
    relative to `origin` (when the audio started)."""
    out = []
    for w in words:
        mid = origin + (w["start"] + w["end"]) / 2
        if t0 - EVENT_PAD <= mid <= t1 + EVENT_PAD:
            out.append(w["word"])
    return " ".join(out)


def build_context(
    keyframes: list,
    trace: list,
    words: list,
    window_start: float,
    window_end: float,
    audio_t0: float,
    consume: bool = True,
) -> GestureContext:
    """Assemble the annotated frames and timeline for one question.

    By default consumes (frees) the in-memory keyframe images; pass consume=False
    when the same frames may be needed again (e.g. a message that gets merged).
    """
    if not keyframes:
        raise RuntimeError("No screen frames were captured.")

    def rel(t: float) -> float:
        return t - window_start

    # Frame k covers [its capture time, next frame's); first starts at the window start.
    bounds = [window_start] + [kf.t for kf in keyframes[1:]] + [window_end + 1.0]

    gestures: list = detect_gestures(trace)
    numbered = list(enumerate(gestures, start=1))

    def frame_of(g: Gesture) -> int:
        mid = (g.t0 + g.t1) / 2
        for k in range(len(keyframes)):
            if bounds[k] <= mid < bounds[k + 1] and _in_rect(g.cx, g.cy, keyframes[k].rect):
                return k
        # Same display as the frame covering that time, even if just outside; else last frame.
        for k in range(len(keyframes)):
            if bounds[k] <= mid < bounds[k + 1]:
                return k
        return len(keyframes) - 1

    owner = {num: frame_of(g) for num, g in numbered}

    frames: list = []
    lines = [
        f"Gesture window: {rel(window_end):.1f}s, {len(keyframes)} screen frame(s). On each frame the pointer's path "
        "is drawn as a trail from blue (earlier) to red (later); the green dot is where the pointer was when that "
        "frame's interval began and the red ring is where it was when the interval ended. Yellow numbered marks are "
        "places the user circled (boxes) or hovered on (rings). Times below are seconds since the user triggered.",
        "",
    ]

    for k, kf in enumerate(keyframes):
        raw = kf.image
        if raw is None:
            raise RuntimeError("Screen frame was already consumed.")
        if consume:
            kf.image = None  # drop our reference; `raw` is freed once rendered

        seg = [p for p in trace if bounds[k] <= p.t < bounds[k + 1]]
        if k > 0 and trace:  # continue the path from the previous interval's last point
            prev = [p for p in trace if p.t < bounds[k]]
            if prev:
                seg.insert(0, TracePoint(bounds[k], prev[-1].x, prev[-1].y))
        marks = [(num, g) for num, g in numbered if owner[num] == k]
        frames.append((f"Frame {k + 1} of {len(keyframes)}", _render_frame(raw, kf.rect, seg, marks)))
        del raw

        span = f"{rel(bounds[k]):.1f}s to {rel(min(bounds[k + 1], window_end)):.1f}s"
        head = f"Frame {k + 1} ({span})"
        if k > 0:
            head += " — the screen changed at the start of this interval"
        lines.append(head + ":")
        on_screen = [p for p in seg if _in_rect(p.x, p.y, kf.rect)]
        if on_screen:
            lines.append(
                f"  Pointer moved from {_where(on_screen[0].x, on_screen[0].y, kf.rect)} "
                f"to {_where(on_screen[-1].x, on_screen[-1].y, kf.rect)}."
            )
        else:
            lines.append("  The pointer was on a different display during this interval.")
        for num, g in marks:
            verb = "Circled" if g.kind == "circle" else "Hovered on"
            said = _said(words, g.t0, g.t1, audio_t0)
            lines.append(
                f"  [{num}] {rel(g.t0):.1f}s–{rel(g.t1):.1f}s: {verb} something at "
                f"{_where(g.cx, g.cy, kf.rect)}" + (f' while saying "{said}".' if said else ".")
            )
        frame_said = _said(words, bounds[k], min(bounds[k + 1], window_end), audio_t0)
        if frame_said:
            lines.append(f'  Spoken during this interval: "{frame_said}"')

    return GestureContext("\n".join(lines), frames)
