"""Compose the social-media demo video from captured application states.

The storyboard places captured states of the real interface on a timeline
with a virtual camera, captions, a workflow stepper and a simulated cursor.
Frames are drawn with Pillow and encoded to H.264 by the ffmpeg binary that
imageio-ffmpeg bundles (a dev dependency). Used by ``capture_demo.py``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

if TYPE_CHECKING:
    from capture_demo import Shot

FPS = 30
MARGIN = 32
ACCENT = (74, 158, 255)
TEXT = (232, 234, 237)
TEXT_DIM = (157, 163, 171)
TEXT_MUTED = (90, 97, 105)
BACKGROUND_TOP = (16, 22, 30)
BACKGROUND_BOTTOM = (8, 11, 15)
VIEWPORT_FILL = (11, 15, 20)
BORDER = (45, 49, 57)
STAGES = ("Configure", "Decompose", "Edit", "Visualise")
CAPTION_FADE_S = 0.45
# Multiplies every storyboard time and motion; raise it to slow the video down.
PACE = 1.1
FONT_DIR = Path(__file__).resolve().parents[1] / "src/scd_app/gui/style/fonts"

Rect = tuple[float, float, float, float]
Point = tuple[float, float]
Camera = tuple[float, float, float]  # centre x, centre y, width (window px)


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    header: int
    footer: int
    headline_size: int

    @property
    def viewport(self) -> tuple[int, int, int, int]:
        return (
            MARGIN,
            self.header,
            self.width - 2 * MARGIN,
            self.height - self.header - self.footer,
        )


LAYOUTS = {
    "square": Layout(1080, 1080, 200, 104, 50),
    "portrait": Layout(1080, 1350, 250, 124, 54),
}


@lru_cache
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR / f"Lexend-{name}.ttf", size=size)


def _ease(progress: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    u = min(max(progress, 0.0), 1.0)
    return 4 * u**3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def _centre(rect: Rect) -> Point:
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)


def _union(*rects: Rect) -> Rect:
    left = min(r[0] for r in rects)
    top = min(r[1] for r in rects)
    right = max(r[0] + r[2] for r in rects)
    bottom = max(r[1] + r[3] for r in rects)
    return (left, top, right - left, bottom - top)


class _Track:
    """Keyed values; each key eases in from the previous one over its duration."""

    def __init__(self) -> None:
        self._keys: list[tuple[float, float, object]] = []

    def add(self, time: float, value: object, duration: float = 0.0) -> None:
        self._keys.append((time, duration, value))
        self._keys.sort(key=lambda key: key[0])

    def at(self, time: float) -> tuple[object, object, float]:
        """Return the previous value, the current value and eased progress."""
        active = [key for key in self._keys if key[0] <= time]
        if not active:
            return None, None, 1.0
        start, duration, value = active[-1]
        previous = active[-2][2] if len(active) > 1 else None
        progress = 1.0 if duration <= 0 else _ease((time - start) / duration)
        return previous, value, progress


class Timeline:
    def __init__(self, layout: Layout, window: tuple[int, int]) -> None:
        _x, _y, width, height = layout.viewport
        self.aspect = width / height
        self.window = window
        self.screens = _Track()
        self.cameras = _Track()
        self.captions = _Track()
        self.stages = _Track()
        self.cursors = _Track()
        self.clicks: list[float] = []
        self.drags: list[tuple[float, float, Point]] = []
        self.spotlights: list[tuple[float, float, Rect]] = []
        self.keys: list[tuple[float, float, str, str]] = []
        self.end_start = 0.0
        self.duration = 0.0

    def fit(self, rect: Rect, pad: float = 0.08) -> Camera:
        """Frame *rect* with padding, kept inside the window where possible."""
        window_width, window_height = self.window
        width = max(rect[2] * (1 + pad), rect[3] * (1 + pad) * self.aspect)
        height = width / self.aspect
        cx, cy = _centre(rect)
        if width <= window_width:
            cx = min(max(cx, width / 2), window_width - width / 2)
        else:
            cx = window_width / 2
        if height <= window_height:
            cy = min(max(cy, height / 2), window_height - height / 2)
        else:
            cy = window_height / 2
        return (cx, cy, width)

    def band(
        self,
        top: float,
        bottom: float,
        cx: float,
        left: float = 0.0,
        pad: float = 0.02,
    ) -> Camera:
        """Frame the rows *top* to *bottom*, centred on *cx*, right of *left*.

        Keeping the frame right of a panel edge avoids showing a sliver of
        the neighbouring panel.
        """
        window_width = self.window[0]
        width = min((bottom - top) * (1 + pad) * self.aspect, window_width - left)
        cx = min(max(cx, left + width / 2), window_width - width / 2)
        return (cx, (top + bottom) / 2, width)

    def overview(self, cx: float) -> Camera:
        """The whole window, or its full height around *cx* in tall layouts."""
        whole = self.fit((0.0, 0.0, *self.window), pad=0.0)
        if whole[2] / self.aspect <= self.window[1] * 1.15:
            return whole
        return self.band(0.0, self.window[1], cx, pad=0.0)

    # Storyboard times below are in paced seconds: each is multiplied by PACE.

    def screen(self, time: float, name: str, fade: float = 0.0) -> None:
        self.screens.add(time * PACE, name, fade * PACE)

    def camera(self, time: float, camera: Camera, duration: float = 0.0) -> None:
        self.cameras.add(time * PACE, camera, duration * PACE)

    def caption(self, time: float, headline: str, kicker: str | None = None) -> None:
        fade = CAPTION_FADE_S if time else 0.0
        self.captions.add(time * PACE, (headline, kicker), fade)

    def stage(self, time: float, index: int | None) -> None:
        self.stages.add(time * PACE, index, 0.4 if time else 0.0)

    def cursor(self, time: float, point: Point | None, duration: float = 0.25) -> None:
        self.cursors.add(time * PACE, point, duration * PACE)

    def click(self, time: float) -> None:
        self.clicks.append(time * PACE)

    def drag(self, start: float, end: float, anchor: Point) -> None:
        self.drags.append((start * PACE, end * PACE, anchor))

    def spotlight(self, start: float, end: float, rect: Rect) -> None:
        self.spotlights.append((start * PACE, end * PACE, rect))

    def key(self, start: float, end: float, key: str, label: str) -> None:
        """Show a keyboard shortcut whose button is outside the frame."""
        self.keys.append((start * PACE, end * PACE, key, label))

    def end_card(self, start: float, duration: float) -> None:
        self.end_start = start * PACE
        self.duration = (start + duration) * PACE


def build_storyboard(shots: dict[str, Shot], layout: Layout, scale: int) -> Timeline:
    first = next(iter(shots.values()))
    window = (first.image.width // scale, first.image.height // scale)
    width, height = window
    tl = Timeline(layout, window)

    def mark(shot: str, name: str):
        return shots[shot].marks[name]

    source = mark("edit_review", "source")
    rate = mark("edit_review", "rate")
    plots_left = source[0] + 2
    plots = tl.band(source[1], rate[1] + rate[3], _centre(source)[0], plots_left)
    # The MUAP grid, kept left of the plots so no sliver of them shows.
    cx, cy, muap_width = tl.fit(mark("edit_review", "muap"), pad=0.02)
    muap = (min(cx, plots_left - 4 - muap_width / 2), cy, muap_width)
    split = mark("edit_merged", "split")
    toolbar_and_source = tl.band(
        split[1] - 10, source[1] + source[3], _centre(source)[0], plots_left
    )
    zoom_steps = sum(name.startswith("edit_zoom_") for name in shots)

    # Each scene starts where the previous one ends; ``t`` is its start.
    # Configure, opening on the hook headline.
    tl.screen(0.0, "config")
    tl.camera(0.0, tl.overview(width / 2 + 150))
    tl.caption(0.0, "From raw HD-EMG to clean motor units", "Free & open source")
    tl.stage(0.0, None)
    tl.camera(
        0.6, tl.fit((width * 0.42, height * 0.42, width * 0.58, height * 0.58)), 3.0
    )
    tl.caption(2.8, "Load a recording and map your grids")
    tl.stage(2.8, 0)
    tl.cursor(3.0, (width * 0.6, height * 0.62), 0.25)
    tl.cursor(3.3, _centre(mark("config", "apply")), 0.8)
    tl.click(4.3)
    t = 5.2

    # Decompose: the source converges iteration by iteration.
    tl.screen(t, "decomp_ready", 0.35)
    tl.camera(t, tl.overview(0.0), 0.6)
    tl.caption(t, "Watch each source converge, live")
    tl.stage(t, 1)
    tl.cursor(t + 0.3, _centre(mark("decomp_ready", "start")), 0.7)
    tl.click(t + 1.1)
    for index in range(4):
        tl.screen(t + 1.2 + 0.7 * index, f"decomp_{index}", 0.2)
    plot = mark("decomp_ready", "plot")
    axes = (
        plot[0] + 0.06 * plot[2],
        plot[1] + 0.1 * plot[3],
        0.88 * plot[2],
        0.84 * plot[3],
    )
    tl.camera(t + 1.2, tl.fit(axes, pad=0.0), 0.9)
    tl.cursor(t + 1.4, None)
    t += 4.6

    # Review the first unit over the whole contraction, then its MUAPs.
    tl.screen(t, "edit_review", 0.4)
    tl.caption(t, "Review each unit against the force")
    tl.stage(t, 2)
    tl.camera(t, plots, 0.9)
    tl.caption(t + 2.0, "…and its MUAPs across the grid")
    tl.camera(t + 2.0, muap, 0.9)
    t += 4.0

    # Zoom the application in, then inspect a genuine and a false discharge.
    tl.camera(t, plots, 0.8)
    for step in range(1, zoom_steps + 1):
        tl.screen(t + 0.3 + 0.06 * (step - 1), f"edit_zoom_{step}")
    tl.caption(t + 0.3, "Click a discharge to overlay its MUAP")
    spike = mark("edit_inspect_true", "spike")
    tl.cursor(t + 1.5, (spike[0] - 120, spike[1] + 140), 0.25)
    tl.cursor(t + 1.7, spike, 0.6)
    tl.click(t + 2.4)
    tl.screen(t + 2.45, "edit_inspect_true", 0.2)
    tl.cursor(t + 2.6, None)
    tl.camera(t + 2.6, muap, 0.8)
    tl.caption(t + 4.2, "A false discharge doesn't match")
    tl.screen(t + 4.2, "edit_inspect_false", 0.3)
    tl.camera(t + 5.8, plots, 0.8)
    t += 6.8

    # Delete the false discharge and add the missed one.
    start, end = (
        mark("edit_delete_armed", "box_start"),
        mark("edit_delete_armed", "box_end"),
    )
    tl.caption(t, "Delete it with a box…")
    tl.key(t, t + 1.4, "D", "Delete spikes")
    tl.screen(t + 0.3, "edit_delete_armed", 0.25)
    tl.cursor(t + 0.4, (start[0] - 60, start[1] - 60), 0.25)
    tl.cursor(t + 0.7, start, 0.5)
    tl.drag(t + 1.3, t + 2.0, start)
    tl.cursor(t + 1.3, end, 0.7)
    tl.screen(t + 2.05, "edit_deleted", 0.25)
    start, end = mark("edit_add_armed", "box_start"), mark("edit_add_armed", "box_end")
    tl.caption(t + 2.6, "…and add the ones it missed")
    tl.key(t + 2.6, t + 4.0, "A", "Add spikes")
    tl.screen(t + 2.9, "edit_add_armed", 0.25)
    tl.cursor(t + 3.0, start, 0.7)
    tl.drag(t + 3.9, t + 4.6, start)
    tl.cursor(t + 3.9, end, 0.7)
    tl.screen(t + 4.65, "edit_added", 0.25)
    tl.cursor(t + 5.0, None)
    t += 6.0

    # Recalculate the filter from the edited discharges, in the same framing
    # so the pulses visibly grow against the fixed axis.
    tl.caption(t, "Recalculate the filter: stronger pulses")
    tl.key(t, t + 1.6, "F", "Recalc filter")
    tl.screen(t + 0.6, "edit_recalculated", 0.6)
    t += 3.8

    # The next unit merges two: split it in seconds.
    tl.caption(t, "Merged units? Split them in seconds")
    tl.camera(t, toolbar_and_source, 0.8)
    tl.key(t, t + 1.3, "N", "Next unreviewed")
    tl.screen(t + 0.3, "edit_merged", 0.35)
    tl.cursor(t + 0.9, _centre(source), 0.25)
    tl.cursor(t + 1.1, _centre(split), 0.6)
    tl.click(t + 1.8)
    tl.screen(t + 1.85, "edit_split_preview", 0.25)
    tl.cursor(t + 2.6, _centre(mark("edit_split_preview", "confirm")), 0.5)
    tl.click(t + 3.2)
    tl.screen(t + 3.25, "edit_split_done", 0.25)
    tl.cursor(t + 3.5, None)
    t += 4.6

    # Visualise the whole population against the force.
    idr_tab = mark("vis_raster", "idr_tab")
    tl.screen(t, "vis_raster", 0.4)
    tl.camera(t, tl.overview(700.0), 0.8)
    tl.caption(t, "See the whole motor-unit population")
    tl.stage(t, 3)
    tl.cursor(t + 0.8, (width * 0.55, height * 0.5), 0.25)
    tl.cursor(t + 1.1, _centre(idr_tab), 0.6)
    tl.click(t + 1.9)
    tl.screen(t + 1.95, "vis_idr", 0.35)
    tl.cursor(t + 2.2, None)
    t += 4.4

    tl.end_card(t, 3.8)
    return tl


class _Renderer:
    def __init__(
        self,
        shots: dict[str, Shot],
        layout: Layout,
        timeline: Timeline,
        scale: int,
    ):
        self.shots = shots
        self.scale = scale
        self.layout = layout
        self.tl = timeline
        _x, _y, vw, vh = layout.viewport
        self.viewport_size = (vw, vh)
        self.background = self._gradient()
        self.viewport_mask = self._rounded_mask((vw, vh), 18)
        self.cursor_sprite, self.cursor_hotspot = self._cursor_sprite(34)
        self.headline_font = _font("SemiBold", layout.headline_size)
        self.kicker_font = _font("Medium", 24)
        self.stage_font = _font("Medium", 22)
        self.wordmark_font = _font("SemiBold", 26)
        self._view_cache: dict[tuple, Image.Image] = {}
        self._caption_cache: dict[tuple, Image.Image] = {}
        self._key_cache: dict[tuple[str, str], Image.Image] = {}
        self._end_background: Image.Image | None = None

    # ── static pieces ────────────────────────────────────────────────────

    def _gradient(self) -> Image.Image:
        width, height = self.layout.width, self.layout.height
        column = Image.new("RGB", (1, height))
        for y in range(height):
            u = y / (height - 1)
            column.putpixel(
                (0, y),
                tuple(
                    round(a + (b - a) * u)
                    for a, b in zip(BACKGROUND_TOP, BACKGROUND_BOTTOM, strict=True)
                ),
            )
        return column.resize((width, height))

    def _rounded_mask(self, size: tuple[int, int], radius: int) -> Image.Image:
        big = Image.new("L", (size[0] * 4, size[1] * 4), 0)
        ImageDraw.Draw(big).rounded_rectangle(
            (0, 0, big.width - 1, big.height - 1), radius=radius * 4, fill=255
        )
        return big.resize(size, Image.Resampling.LANCZOS)

    def _cursor_sprite(self, height: int) -> tuple[Image.Image, Point]:
        """An anti-aliased arrow pointer with a soft shadow."""
        factor = 4
        outline = [
            (0.0, 0.0),
            (0.0, 16.5),
            (4.2, 12.6),
            (7.0, 19.0),
            (9.6, 17.9),
            (6.9, 11.7),
            (12.4, 11.7),
        ]
        unit = height * factor / 19.0
        pad = 8 * factor
        size = (int(13 * unit) + 2 * pad, int(20 * unit) + 2 * pad)
        points = [(pad + x * unit, pad + y * unit) for x, y in outline]
        shadow = Image.new("L", size, 0)
        ImageDraw.Draw(shadow).polygon(
            [(x + 2 * factor, y + 3 * factor) for x, y in points], fill=120
        )
        sprite = Image.new("RGBA", size, (0, 0, 0, 0))
        sprite.putalpha(shadow.filter(ImageFilter.GaussianBlur(3 * factor)))
        draw = ImageDraw.Draw(sprite)
        draw.polygon(points, fill=(255, 255, 255, 255))
        draw.line(
            [*points, points[0]],
            fill=(18, 18, 18, 255),
            width=int(1.7 * factor),
            joint="curve",
        )
        sprite = sprite.resize(
            (size[0] // factor, size[1] // factor), Image.Resampling.LANCZOS
        )
        return sprite, (pad / factor, pad / factor)

    # ── camera and screen ────────────────────────────────────────────────

    def _camera_rect(self, time: float) -> Rect:
        previous, current, progress = self.tl.cameras.at(time)
        if previous is None:
            cx, cy, width = current
        else:
            cx = previous[0] + (current[0] - previous[0]) * progress
            cy = previous[1] + (current[1] - previous[1]) * progress
            width = previous[2] * (current[2] / previous[2]) ** progress
        height = width / self.tl.aspect
        return (cx - width / 2, cy - height / 2, width, height)

    def _render_screen(self, name: str, rect: Rect) -> Image.Image:
        key = (name, *(round(value, 2) for value in rect))
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        image = self.shots[name].image
        x, y, width, height = rect
        out_w, out_h = self.viewport_size
        s = out_w / width
        window_w, window_h = self.tl.window
        left, top = max(x, 0.0), max(y, 0.0)
        right, bottom = min(x + width, window_w), min(y + height, window_h)
        k = self.scale
        view = Image.new("RGB", (out_w, out_h), VIEWPORT_FILL)
        if right > left and bottom > top:
            px0, py0 = round((left - x) * s), round((top - y) * s)
            px1, py1 = round((right - x) * s), round((bottom - y) * s)
            box = (
                max(x + px0 / s, 0.0) * k,
                max(y + py0 / s, 0.0) * k,
                min(x + px1 / s, window_w) * k,
                min(y + py1 / s, window_h) * k,
            )
            tile = image.resize(
                (px1 - px0, py1 - py0),
                Image.Resampling.BICUBIC,
                box=box,
                reducing_gap=2.0,
            )
            view.paste(tile, (px0, py0))
            if (px0, py0, px1, py1) != (0, 0, out_w, out_h):
                ImageDraw.Draw(view).rectangle(
                    (px0 - 1, py0 - 1, px1, py1), outline=BORDER, width=2
                )
        if len(self._view_cache) > 12:
            self._view_cache.clear()
        self._view_cache[key] = view
        return view

    def _to_view(self, rect: Rect, point: Point) -> Point:
        s = self.viewport_size[0] / rect[2]
        return ((point[0] - rect[0]) * s, (point[1] - rect[1]) * s)

    # ── overlays ─────────────────────────────────────────────────────────

    def _spotlight(self, view: Image.Image, camera: Rect, time: float) -> Image.Image:
        for start, end, target in self.tl.spotlights:
            if not start <= time <= end + 0.3:
                continue
            alpha = min(_ease((time - start) / 0.3), 1 - _ease((time - end) / 0.3))
            if alpha <= 0:
                continue
            x0, y0 = self._to_view(camera, (target[0], target[1]))
            x1, y1 = self._to_view(
                camera, (target[0] + target[2], target[1] + target[3])
            )
            pad = 10
            hole = [2 * v for v in (x0 - pad, y0 - pad, x1 + pad, y1 + pad)]
            mask = Image.new("L", (view.width * 2, view.height * 2), round(255 * alpha))
            ImageDraw.Draw(mask).rounded_rectangle(hole, radius=24, fill=0)
            mask = mask.resize(view.size, Image.Resampling.BILINEAR)
            dim = ImageEnhance.Brightness(view).enhance(0.35)
            view = Image.composite(dim, view, mask)
            big = Image.new("RGBA", (view.width * 2, view.height * 2), (0, 0, 0, 0))
            ImageDraw.Draw(big).rounded_rectangle(
                hole, radius=24, outline=(*ACCENT, round(255 * alpha)), width=6
            )
            ring = big.resize(view.size, Image.Resampling.BILINEAR)
            view = Image.alpha_composite(view.convert("RGBA"), ring).convert("RGB")
        return view

    def _cursor_point(self, time: float) -> tuple[Point | None, float]:
        previous, current, progress = self.tl.cursors.at(time)
        if current is None and previous is None:
            return None, 0.0
        if current is None:
            return previous, 1.0 - progress
        if previous is None:
            return current, progress
        point = (
            previous[0] + (current[0] - previous[0]) * progress,
            previous[1] + (current[1] - previous[1]) * progress,
        )
        return point, 1.0

    def _overlay_pointer(
        self, view: Image.Image, camera: Rect, time: float
    ) -> Image.Image:
        point, alpha = self._cursor_point(time)
        if point is None or alpha <= 0:
            return view
        overlay = Image.new("RGBA", (view.width * 2, view.height * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        vx, vy = self._to_view(camera, point)
        for start, end, anchor in self.tl.drags:
            if start <= time <= end + 0.25:
                fade = 1.0 - _ease((time - end) / 0.25) if time > end else 1.0
                ax, ay = self._to_view(camera, anchor)
                box = [
                    2 * min(ax, vx),
                    2 * min(ay, vy),
                    2 * max(ax, vx),
                    2 * max(ay, vy),
                ]
                draw.rectangle(
                    box,
                    fill=(*ACCENT, round(50 * fade)),
                    outline=(*ACCENT, round(230 * fade)),
                    width=4,
                )
        for click in self.tl.clicks:
            if click <= time <= click + 0.5:
                u = (time - click) / 0.5
                radius = 2 * (10 + 30 * _ease(u))
                draw.ellipse(
                    (
                        2 * vx - radius,
                        2 * vy - radius,
                        2 * vx + radius,
                        2 * vy + radius,
                    ),
                    outline=(*ACCENT, round(255 * (1 - u))),
                    width=5,
                )
        view = view.convert("RGBA")
        view = Image.alpha_composite(
            view, overlay.resize(view.size, Image.Resampling.BILINEAR)
        )
        sprite = self.cursor_sprite
        if alpha < 1:
            sprite = sprite.copy()
            sprite.putalpha(sprite.getchannel("A").point(lambda v: round(v * alpha)))
        view.alpha_composite(
            sprite,
            (round(vx - self.cursor_hotspot[0]), round(vy - self.cursor_hotspot[1])),
        )
        return view.convert("RGB")

    def _view(self, time: float) -> Image.Image:
        camera = self._camera_rect(time)
        previous, current, progress = self.tl.screens.at(time)
        view = self._render_screen(current, camera)
        if previous is not None and progress < 1:
            view = Image.blend(self._render_screen(previous, camera), view, progress)
        view = self._spotlight(view, camera, time)
        return self._overlay_pointer(view, camera, time)

    # ── header and footer ────────────────────────────────────────────────

    def _wrap(self, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
        """Wrap to *width*, balancing two lines so no word is left orphaned."""
        if font.getlength(text) <= width:
            return [text]
        words = text.split()
        splits = [
            (" ".join(words[:index]), " ".join(words[index:]))
            for index in range(1, len(words))
        ]
        fitting = [pair for pair in splits if max(map(font.getlength, pair)) <= width]
        if fitting:
            return list(min(fitting, key=lambda pair: max(map(font.getlength, pair))))
        lines, line = [], ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if font.getlength(candidate) <= width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        return [*lines, line]

    def _caption_sprite(self, caption: tuple[str, str | None]) -> Image.Image:
        cached = self._caption_cache.get(caption)
        if cached is not None:
            return cached
        headline, kicker = caption
        width = self.layout.width - 2 * MARGIN
        lines = self._wrap(headline, self.headline_font, width)
        line_height = round(self.layout.headline_size * 1.18)
        kicker_height = 42 if kicker else 0
        sprite = Image.new(
            "RGBA", (width, kicker_height + line_height * len(lines) + 12), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(sprite)
        if kicker:
            draw.text(
                (0, 0), kicker.upper(), font=self.kicker_font, fill=(*ACCENT, 255)
            )
        for index, line in enumerate(lines):
            draw.text(
                (0, kicker_height + index * line_height),
                line,
                font=self.headline_font,
                fill=(*TEXT, 255),
            )
        self._caption_cache[caption] = sprite
        return sprite

    def _paste_faded(self, canvas: Image.Image, sprite: Image.Image, xy, alpha: float):
        if alpha <= 0:
            return
        if alpha < 1:
            sprite = sprite.copy()
            sprite.putalpha(sprite.getchannel("A").point(lambda v: round(v * alpha)))
        canvas.alpha_composite(sprite, (round(xy[0]), round(xy[1])))

    def _draw_caption(self, canvas: Image.Image, time: float) -> None:
        previous, current, progress = self.tl.captions.at(time)
        header = self.layout.header

        def place(caption, alpha: float, lift: float) -> None:
            sprite = self._caption_sprite(caption)
            y = (header - sprite.height) / 2 + 4 + lift
            self._paste_faded(canvas, sprite, (MARGIN, y), alpha)

        if previous is not None and progress < 1:
            place(previous, 1 - min(progress / 0.4, 1.0), 0.0)
        place(current, progress, 18 * (1 - progress))

    def _draw_footer(self, canvas: Image.Image, time: float) -> None:
        layout = self.layout
        draw = ImageDraw.Draw(canvas)
        y = layout.height - layout.footer / 2
        draw.text(
            (MARGIN, y), "SCD-Edition", font=self.wordmark_font, fill=TEXT, anchor="lm"
        )
        previous, current, progress = self.tl.stages.at(time)
        gap = 30
        widths = [self.stage_font.getlength(stage) for stage in STAGES]
        x = layout.width - MARGIN - sum(widths) - gap * (len(STAGES) - 1)
        centres = []
        for index, stage in enumerate(STAGES):
            active = current == index
            colour = TEXT if active else TEXT_MUTED
            draw.text((x, y), stage, font=self.stage_font, fill=colour, anchor="lm")
            centres.append((x, widths[index]))
            x += widths[index] + gap
        if current is None and previous is None:
            return

        def bar(index: int | None) -> tuple[float, float] | None:
            return None if index is None else centres[index]

        start, end = bar(previous), bar(current)
        if end is None:
            return
        if start is None:
            left, width = end
            alpha = progress
        else:
            left = start[0] + (end[0] - start[0]) * progress
            width = start[1] + (end[1] - start[1]) * progress
            alpha = 1.0
        colour = tuple(
            round(c * alpha + b * (1 - alpha))
            for c, b in zip(ACCENT, BACKGROUND_BOTTOM, strict=True)
        )
        draw.rounded_rectangle(
            (left, y + 18, left + width, y + 22), radius=2, fill=colour
        )

    # ── end card ─────────────────────────────────────────────────────────

    def _end_card(self, time: float) -> Image.Image:
        if self._end_background is None:
            # The interface alone, filling the frame, so no caption blurs in.
            view = self._view(self.tl.end_start)
            width, height = self.layout.width, self.layout.height
            scale = max(width / view.width, height / view.height)
            cover = view.resize(
                (round(view.width * scale), round(view.height * scale)),
                Image.Resampling.BICUBIC,
            )
            left, top = (cover.width - width) // 2, (cover.height - height) // 2
            cover = cover.crop((left, top, left + width, top + height))
            blurred = cover.filter(ImageFilter.GaussianBlur(18))
            self._end_background = ImageEnhance.Brightness(blurred).enhance(0.3)
        local = time - self.tl.end_start
        fade = _ease(local / 0.6)
        canvas = Image.blend(self._compose(time), self._end_background, fade)
        canvas = canvas.convert("RGBA")
        width, height = self.layout.width, self.layout.height
        items = [
            ("SCD-Edition", _font("Bold", 96), TEXT),
            (
                "Open-source GUI for HD-EMG decomposition",
                _font("Regular", 32),
                TEXT_DIM,
            ),
            ("and motor-unit editing", _font("Regular", 32), TEXT_DIM),
            ("pip install scd-edition", _font("Medium", 32), TEXT),
            ("github.com/AgneGris/scd-edition", _font("Medium", 30), ACCENT),
        ]
        offsets = [0, 128, 172, 262, 352]
        top = height / 2 - 210
        for index, ((text, font, colour), offset) in enumerate(
            zip(items, offsets, strict=True)
        ):
            appear = _ease((local - 0.3 - 0.12 * index) / 0.5)
            if appear <= 0:
                continue
            layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            y = top + offset + 16 * (1 - appear)
            if text.startswith("pip"):
                text_width = font.getlength(text)
                draw.rounded_rectangle(
                    (
                        width / 2 - text_width / 2 - 28,
                        y - 30,
                        width / 2 + text_width / 2 + 28,
                        y + 30,
                    ),
                    radius=14,
                    fill=(*BACKGROUND_TOP, 230),
                    outline=(*ACCENT, 255),
                    width=2,
                )
            draw.text((width / 2, y), text, font=font, fill=(*colour, 255), anchor="mm")
            self._paste_faded(canvas, layer, (0, 0), appear)
        return canvas.convert("RGB")

    # ── keyboard shortcut hints ──────────────────────────────────────────

    def _key_sprite(self, key: str, label: str) -> Image.Image:
        cached = self._key_cache.get((key, label))
        if cached is not None:
            return cached
        k = 3  # drawn oversized, then reduced, for anti-aliased edges
        key_font = _font("SemiBold", 26 * k)
        label_font = _font("Medium", 25 * k)
        pad, cap = 12 * k, 44 * k
        width = int(pad + cap + 16 * k + label_font.getlength(label) + 2 * pad)
        height = cap + 2 * pad
        sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        draw.rounded_rectangle(
            (0, 0, width - 1, height - 1),
            radius=height // 2,
            fill=(*BACKGROUND_TOP, 235),
            outline=(*ACCENT, 255),
            width=2 * k,
        )
        draw.rounded_rectangle(
            (pad, pad, pad + cap, pad + cap), radius=9 * k, fill=(*TEXT, 255)
        )
        draw.text(
            (pad + cap / 2, pad + cap / 2),
            key,
            font=key_font,
            fill=(*BACKGROUND_BOTTOM, 255),
            anchor="mm",
        )
        draw.text(
            (pad + cap + 16 * k, height / 2),
            label,
            font=label_font,
            fill=(*TEXT, 255),
            anchor="lm",
        )
        sprite = sprite.resize((width // k, height // k), Image.Resampling.LANCZOS)
        self._key_cache[(key, label)] = sprite
        return sprite

    def _draw_keys(self, canvas: Image.Image, time: float) -> None:
        vx, vy, _vw, _vh = self.layout.viewport
        for start, end, key, label in self.tl.keys:
            alpha = min(_ease((time - start) / 0.25), 1 - _ease((time - end) / 0.25))
            if alpha > 0:
                lift = 10 * (1 - alpha)
                sprite = self._key_sprite(key, label)
                self._paste_faded(canvas, sprite, (vx + 20, vy + 20 + lift), alpha)

    # ── frames ───────────────────────────────────────────────────────────

    def _compose(self, time: float) -> Image.Image:
        canvas = self.background.copy().convert("RGBA")
        vx, vy, _vw, _vh = self.layout.viewport
        canvas.paste(self._view(time), (vx, vy), self.viewport_mask)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            (vx - 1, vy - 1, vx + self.viewport_size[0], vy + self.viewport_size[1]),
            radius=18,
            outline=BORDER,
            width=2,
        )
        self._draw_keys(canvas, time)
        self._draw_caption(canvas, time)
        self._draw_footer(canvas, time)
        return canvas.convert("RGB")

    def frame(self, time: float) -> Image.Image:
        if time >= self.tl.end_start:
            return self._end_card(time)
        return self._compose(time)


def render_video(shots: dict[str, Shot], output: Path, aspect: str, scale: int) -> None:
    """Render one aspect ratio; *scale* is the capture's pixels per window px."""
    import imageio_ffmpeg

    layout = LAYOUTS[aspect]
    timeline = build_storyboard(shots, layout, scale)
    renderer = _Renderer(shots, layout, timeline, scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{layout.width}x{layout.height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    frames = round(timeline.duration * FPS)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for index in range(frames):
            encoder.stdin.write(renderer.frame(index / FPS).tobytes())
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode:
        raise RuntimeError(f"ffmpeg failed with exit code {encoder.returncode}")
    print(
        f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB, "
        f"{frames / FPS:.1f} s)",
        flush=True,
    )
    # The finished title card, to upload as the post's custom thumbnail so the
    # video itself can open straight on the interface.
    thumbnail = output.with_name(f"{output.stem}-thumbnail.png")
    renderer.frame(timeline.duration - 1 / FPS).save(thumbnail, optimize=True)
    print(f"Wrote {thumbnail}", flush=True)
