"""Work out where on screen a described thing is, so the guide cursor can point at it.

Each lookup starts from the screen as it is *now*, so a button that isn't
where anyone expected is simply found where it is. Three tiers, cheapest first:

1. Accessibility: an exact name match among the elements the OS reports gives
   the precise rectangle, with no model call.
2. Claude picks: if the match is weak or ambiguous (or the app exposes
   nothing), Claude gets a screenshot plus the numbered element list and
   either picks an element or, for something not in the list, gives a pixel
   position.
3. Refinement: a pixel position from vision is only approximate, so the guide
   cursor goes there first and Claude checks a zoomed crop where her own cursor
   is visible, nudging it onto the target if it's off.

Both Claude calls use brain.LOOKUP_MODEL (Haiku by default; set the
SHERPA_LOOKUP_MODEL env var to override), which matched Sonnet's accuracy on
these small screen lookups at about 60% of the latency.

If nothing is found, `locate` returns None and the caller should say so
rather than point somewhere arbitrary.

Try it (needs a trusted terminal; points the cursor at the result):
    python -m sherpa.locate "the Save button"
"""

import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from . import accessibility, brain
from .capture import grab_display_under_cursor
from .config import TMP_DIR

MAX_LISTED = 250
CROP_W, CROP_H = 480, 320  # points around the cursor shown to the refinement check
MAX_REFINEMENTS = 2
SETTLE_S = 0.15  # let the overlay paint her cursor before capturing it

_CHOOSE_SYSTEM = (
    "You help a guide cursor point at things on a computer screen. You get one "
    "screenshot and a numbered list of UI elements the operating system reports "
    "(it may be incomplete or missing). Find the single thing the user "
    "described. Reply with JSON only:\n"
    '{"found": true, "element": <number>}  when a listed element is it (prefer this);\n'
    '{"found": true, "x": <px>, "y": <px>}  when it is visible but not listed: the pixel '
    "position of its center in the screenshot, measured from the top-left;\n"
    '{"found": false, "reason": "<short>"}  when it is not on screen.\n'
    "Be literal about which thing: if several match, use the surrounding context "
    "(window, dialog, nearby labels) to choose the one the task needs."
)

_REFINE_SYSTEM = (
    "A guide cursor (a blue arrow with a 'Sherpa' tag) is pointing at the screen. "
    "You get a zoomed crop of the screen around it. The arrow's TIP, which is the "
    "exact spot it points at, is at the pixel position given. Decide whether the tip "
    "is on the thing the user described. Reply with JSON only:\n"
    '{"on_target": true}  if the tip is on it (anywhere within it counts);\n'
    '{"on_target": false, "x": <px>, "y": <px>}  if not: the pixel position, in this crop, '
    "of the center of the described thing;\n"
    '{"found": false}  if the described thing is not in this crop.'
)


@dataclass
class Located:
    x: float  # global screen points
    y: float
    how: str  # "accessibility" | "claude-element" | "vision"
    label: str = ""
    approximate: bool = False  # vision position that should be checked with refine()
    radius: float = 80.0  # a click this close counts as "on it"


# ---------------------------------------------------------------- screenshots
def _screenshot() -> tuple:
    """(PIL image, display rect in points) of the display the pointer is on."""
    path = TMP_DIR / f"locate_{time.monotonic_ns()}.png"
    try:
        _, rect = grab_display_under_cursor(path)
        img = Image.open(path)
        img.load()
    finally:
        path.unlink(missing_ok=True)
    return img, rect


def _scaled(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > brain.MAX_DIMENSION:
        img = img.copy()
        img.thumbnail((brain.MAX_DIMENSION, brain.MAX_DIMENSION), Image.LANCZOS)
    return img


# ---------------------------------------------------------------- locate
def _radius(e) -> float:
    return max(40.0, max(e.frame[2], e.frame[3]) / 2 + 25)


def locate(query: str, context: str = "") -> Optional[Located]:
    """Find `query` (e.g. "the Save button") on the current screen.

    `context` is the larger task ("the user is saving a file in TextEdit"),
    which helps pick between look-alikes.
    """
    elements = []
    if accessibility.trusted():
        try:
            elements = accessibility.snapshot()
        except Exception as e:
            print(f"[Sherpa] Accessibility lookup failed ({type(e).__name__}: {e}); using vision.")
    else:
        print("[Sherpa] Accessibility isn't granted to this app, so lookups will rely on vision only.")

    matches = accessibility.match(query, elements)
    best = accessibility.confident(matches)
    if best is not None:
        x, y = best.element.aim
        return Located(x, y, "accessibility", best.element.label, radius=_radius(best.element))

    return _ask_claude(query, context, elements, matches)


def _ask_claude(query: str, context: str, elements: list, matches: list) -> Optional[Located]:
    img, rect = _screenshot()
    shown = _scaled(img)

    # Likely candidates first, then the rest of the tree, capped to keep the request small.
    ordered = [m.element for m in matches]
    seen = {id(e) for e in ordered}
    ordered += [e for e in elements if id(e) not in seen and (e.role != "AXStaticText" or len(e.label) < 60)]
    ordered = ordered[:MAX_LISTED]

    lines = []
    for i, e in enumerate(ordered):
        x, y, w, h = e.frame
        px = (x - rect[0]) / rect[2] * 100
        py = (y - rect[1]) / rect[3] * 100
        loc = f"{px:.0f}%,{py:.0f}% across/down" if 0 <= px <= 100 and 0 <= py <= 100 else "other display"
        ctx = f" [{e.path}]" if e.path else ""
        win = f" (window: {e.window})" if e.window else ""
        lines.append(f'{i}. {e.role.removeprefix("AX")} "{e.label}"{ctx}{win} at {loc}, {w:.0f}x{h:.0f}')
    listing = "\n".join(lines) if lines else "(the operating system reported no elements)"

    prompt = (
        (f"Task context: {context}\n" if context else "")
        + f"Find: {query}\n\nElements reported by the OS:\n{listing}\n\n"
        f"The screenshot is {shown.size[0]}x{shown.size[1]} pixels."
    )
    try:
        reply = brain.ask_json(_CHOOSE_SYSTEM, [brain.image_block(shown), {"type": "text", "text": prompt}], model=brain.LOOKUP_MODEL)
    except Exception as e:
        print(f"[Sherpa] Couldn't ask Claude where '{query}' is: {type(e).__name__}: {e}")
        return None

    if not reply.get("found"):
        print(f"[Sherpa] Couldn't find '{query}': {reply.get('reason', 'not on screen')}")
        return None
    if "element" in reply:
        try:
            e = ordered[int(reply["element"])]
        except (ValueError, IndexError, TypeError):
            return None
        x, y = e.aim
        return Located(x, y, "claude-element", e.label, radius=_radius(e))
    if "x" in reply and "y" in reply:
        scale = rect[2] / shown.size[0]
        return Located(rect[0] + float(reply["x"]) * scale, rect[1] + float(reply["y"]) * scale, "vision", approximate=True)
    return None


# ---------------------------------------------------------------- refinement
def refine(query: str, at: Located, guide) -> Located:
    """Check a vision-based position with her cursor visible, and nudge it onto
    the target. `guide` is the GuideCursor; it's moved to `at` first. Call from
    a worker thread (not the main/Cocoa thread). Exact (Accessibility) results
    are returned unchanged."""
    if not at.approximate:
        return at
    cur = at
    for _ in range(MAX_REFINEMENTS):
        guide.move_to(cur.x, cur.y)
        time.sleep(SETTLE_S)
        img, rect = _screenshot()
        px_per_pt = img.size[0] / rect[2]
        # Crop centered on the cursor tip, shifted to stay inside the display.
        cx_pt, cy_pt = cur.x - rect[0], cur.y - rect[1]
        left = min(max(cx_pt - CROP_W / 2, 0), max(rect[2] - CROP_W, 0))
        top = min(max(cy_pt - CROP_H / 2, 0), max(rect[3] - CROP_H, 0))
        crop = img.crop((int(left * px_per_pt), int(top * px_per_pt), int((left + CROP_W) * px_per_pt), int((top + CROP_H) * px_per_pt)))
        crop = _scaled(crop)
        k = crop.size[0] / CROP_W  # crop pixels per point
        tip = ((cx_pt - left) * k, (cy_pt - top) * k)
        prompt = f"Find: {query}\nThe arrow tip is at pixel ({tip[0]:.0f}, {tip[1]:.0f}). The crop is {crop.size[0]}x{crop.size[1]} pixels."
        try:
            reply = brain.ask_json(_REFINE_SYSTEM, [brain.image_block(crop), {"type": "text", "text": prompt}], model=brain.LOOKUP_MODEL)
        except Exception as e:
            print(f"[Sherpa] Refinement check failed ({type(e).__name__}); keeping the current position.")
            return cur
        if reply.get("on_target") or reply.get("found") is False or "x" not in reply:
            return cur
        cur = Located(
            rect[0] + left + float(reply["x"]) / k,
            rect[1] + top + float(reply["y"]) / k,
            "vision",
            approximate=True,
        )
    return cur


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import signal
    import sys
    import threading

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    from Foundation import NSTimer
    from PyObjCTools import AppHelper

    from .config import check_config
    from .guide import guide

    check_config()
    query = " ".join(sys.argv[1:]) or "the Save button"

    def work() -> None:
        print("Switch to the app to inspect; looking in 3s...")
        time.sleep(3)
        t0 = time.monotonic()
        found = locate(query)
        print(f"locate took {time.monotonic() - t0:.1f}s ->", found)
        if found is not None:
            guide.show(found.x, found.y - 200)
            time.sleep(0.4)
            found = refine(query, found, guide)
            guide.move_to(found.x, found.y, label=query[:24])
            print("pointing at", found)
            time.sleep(4)
        guide.hide()
        time.sleep(0.3)
        AppHelper.callAfter(AppHelper.stopEventLoop)

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    signal.signal(signal.SIGINT, lambda *_: AppHelper.stopEventLoop())
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.25, True, lambda _t: None)
    threading.Thread(target=work, daemon=True).start()
    AppHelper.runEventLoop()
