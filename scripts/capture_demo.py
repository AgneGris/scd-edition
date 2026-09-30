"""Capture the real Qt interface for the README demo, screenshots and video.

The window is rendered by the platform's native Qt plugin at twice its
logical resolution but is never shown on screen. The "offscreen" plugin must
not be used: it has no system font fallback and draws every symbol glyph in
the interface as an empty box.

The decomposition frames use deterministic illustrative sources so the
capture does not run a costly scientific decomposition. Edition and
Visualisation are populated through the same data path used for a saved
session, from a synthetic 64-channel recording of a trapezoidal contraction:
motor units are recruited in order of size with physiological discharge
variability, the first source merges two units to demonstrate the split
preview, and one of its discharges is left unmarked to demonstrate adding a
missed spike.

    uv run python scripts/capture_demo.py                  # GIF + screenshots
    uv run python scripts/capture_demo.py --video-dir export/linkedin
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("QT_SCALE_FACTOR", str(2))

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSplitter, QWidget
from scipy import signal as sp_signal

from scd_app._vendor.motor_unit_toolbox import props as tb_props
from scd_app.core.filter_recalculation import snap_to_local_peak
from scd_app.examples import bundled_example_config
from scd_app.gui.main_window import MainWindow, _configure_example_on_startup
from scd_app.gui.style.styling import COLORS, set_style_sheet

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1080
RENDER_SCALE = 2
GIF_WIDTH = 1120
GIF_SCREEN_HEIGHT = 756
GIF_CAPTION_HEIGHT = 72
STEP_DURATION_MS = 2600
FINAL_STEP_DURATION_MS = 3400
TRANSITION_DURATION_MS = 140
ACCENT = "#4a9eff"
FONT_DIR = Path(__file__).resolve().parents[1] / "src/scd_app/gui/style/fonts"

# Synthetic recording: a 10 s trapezoidal contraction at 30 % MVC.
DURATION_S = 10.0
FORCE_KNOTS_S = (1.0, 3.5, 6.5, 9.0)  # rise start, plateau start and end, rest
MVC_N = 400.0
PLATEAU_MVC = 0.3
# Recruitment threshold of each true unit as a fraction of the plateau force.
THRESHOLDS = (0.04, 0.12, 0.20, 0.28, 0.36, 0.46, 0.56, 0.66, 0.76)
# Territory centre of each true unit on the 8 x 8 grid (row, column).
CENTRES = (
    (1.5, 1.5),
    (5.5, 2.0),
    (2.0, 5.5),
    (4.5, 4.5),
    (6.5, 6.0),
    (3.0, 3.0),
    (1.0, 6.5),
    (6.0, 0.8),
    (3.8, 7.0),
)
# Decomposed sources in extraction order; the first one merges two units.
SOURCE_UNITS = ((2, 4), (0,), (5,), (1,), (7,), (3,), (8,), (6,))
MISSED_DISCHARGE_S = 5.1
SPLIT_VIEW_S = (3.8, 6.4)
ADD_VIEW_HALF_WIDTH_S = 0.75
INSPECT_DISCHARGE_S = 4.6

README_STEPS = (
    ("config", "Load the bundled 64-channel example"),
    ("decomp_3", "Decompose with live source feedback"),
    ("edit_merged", "Review every motor unit, its quality and MUAP"),
    ("edit_inspect", "Check any discharge against the unit's MUAP"),
    ("edit_split_preview", "Preview a split: high-amplitude A, low-amplitude B"),
    ("edit_split_done", "Confirm two units with recomputed quality"),
    ("edit_add_done", "Box-select a missed discharge to add it"),
    ("vis_idr", "Compare discharge rates with the force"),
)
SCREENSHOTS = {
    "configuration": "config",
    "decomposition": "decomp_3",
    "edition": "edit_split_preview",
    "visualisation": "vis_idr",
}

Rect = tuple[float, float, float, float]


@dataclass
class Shot:
    """One application state and its named regions in logical window pixels."""

    image: Image.Image
    marks: dict[str, tuple[float, ...]] = field(default_factory=dict)


@dataclass
class DemoSession:
    data: dict
    sampling_rate: int
    missed_peak: int
    inspect_peak: int


# ── Synthetic recording ──────────────────────────────────────────────────────


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    window = np.hanning(width)
    return np.convolve(values, window / window.sum(), mode="same")


def _force_profile(n_samples: int, sampling_rate: int) -> np.ndarray:
    """Normalised force (plateau = 1) with rounded corners and slight tremor."""
    rise, plateau_start, plateau_end, rest = FORCE_KNOTS_S
    t = np.arange(n_samples) / sampling_rate
    force = np.interp(
        t,
        [0.0, rise, plateau_start, plateau_end, rest, DURATION_S],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    )
    force = _smooth(force, int(0.25 * sampling_rate))
    tremor = _smooth(
        np.random.default_rng(3).normal(0.0, 1.0, n_samples),
        int(0.1 * sampling_rate),
    )
    return force + 0.006 * tremor / tremor.std()


def _discharge_times(
    force: np.ndarray, threshold: float, sampling_rate: int, rng
) -> np.ndarray:
    """Discharges whose rate rises with force above the recruitment threshold."""
    recruited = np.flatnonzero(force >= threshold)
    if len(recruited) == 0:
        return np.array([], dtype=np.int64)
    sample = int(recruited[0])
    times = []
    while sample < len(force) and force[sample] >= threshold - 0.03:
        times.append(sample)
        rate = 8.0 + 10.0 * max(float(force[sample]) - threshold, 0.0)
        interval = float(np.clip(rng.normal(1.0, 0.1), 0.75, 1.3)) / rate
        sample += int(round(interval * sampling_rate))
    return np.asarray(times, dtype=np.int64)


def _pulse_train(n_samples: int, events: np.ndarray, heights: np.ndarray) -> np.ndarray:
    offsets = np.arange(-18, 19)
    kernel = np.exp(-0.5 * (offsets / 5.0) ** 2)
    train = np.zeros(n_samples, dtype=np.float32)
    for event, height in zip(events, heights, strict=True):
        if 18 <= event < n_samples - 18:
            train[event - 18 : event + 19] += height * kernel
    return train


def _muap_templates(unit: int, half_window: int, sampling_rate: int) -> np.ndarray:
    """Propagating, spatially localised 64-channel MUAP of one unit (µV)."""
    rows, cols = np.divmod(np.arange(64), 8)
    centre_row, centre_col = CENTRES[unit]
    spatial = np.exp(
        -((rows - centre_row) ** 2) / (2 * 2.2**2)
        - (cols - centre_col) ** 2 / (2 * 1.2**2)
    )
    # 0.5 ms conduction delay per 10 mm row along the fibres.
    delay = (rows - centre_row) * 0.0005 * sampling_rate / half_window
    width = 0.9 + 0.3 * unit / len(THRESHOLDS)
    phase = np.linspace(-1.0, 1.0, 2 * half_window, endpoint=False)
    phase = (phase[np.newaxis, :] - delay[:, np.newaxis]) / width
    shape = (
        -0.55 * np.exp(-(((phase + 0.24) / 0.16) ** 2))
        + np.exp(-((phase / 0.11) ** 2))
        - 0.42 * np.exp(-(((phase - 0.27) / 0.18) ** 2))
    )
    templates = spatial[:, np.newaxis] * shape
    peak_to_peak = 60.0 + 260.0 * THRESHOLDS[unit]  # larger units recruit later
    return templates * peak_to_peak / np.ptp(templates, axis=1).max()


def _illustrative_emg(
    trains: list[np.ndarray], n_samples: int, sampling_rate: int
) -> np.ndarray:
    rng = np.random.default_rng(29)
    emg = rng.normal(0.0, 4.0, (64, n_samples)).astype(np.float32)
    half_window = int(round(0.0125 * sampling_rate))
    for unit, events in enumerate(trains):
        templates = _muap_templates(unit, half_window, sampling_rate)
        for event in events:
            left, right = int(event) - half_window, int(event) + half_window
            if left >= 0 and right <= n_samples:
                emg[:, left:right] += rng.normal(1.0, 0.04) * templates
    return emg


def _choose_missed(a_train: np.ndarray, b_train: np.ndarray, sampling_rate: int):
    """A plateau discharge of unit A, clear of unit B, in its steadiest stretch.

    A regular neighbourhood makes the missing discharge the only visible
    anomaly in the firing-rate plot.
    """
    intervals = np.diff(a_train)
    candidates = []
    for index in range(4, len(a_train) - 4):
        event = a_train[index]
        if abs(event / sampling_rate - MISSED_DISCHARGE_S) > 0.5:
            continue
        if np.min(np.abs(b_train - event)) <= 0.012 * sampling_rate:
            continue
        nearby = intervals[index - 4 : index + 4]
        irregularity = np.max(np.abs(nearby / np.median(nearby) - 1.0))
        candidates.append((irregularity, int(event)))
    return min(candidates)[1]


def _demo_session() -> DemoSession:
    config = bundled_example_config()
    sampling_rate = int(config["sampling_rate"])
    n_samples = int(DURATION_S * sampling_rate)
    rng = np.random.default_rng(17)
    force = _force_profile(n_samples, sampling_rate)
    trains = [
        _discharge_times(force, threshold, sampling_rate, rng)
        for threshold in THRESHOLDS
    ]
    unit_a, unit_b = SOURCE_UNITS[0]
    # Keep the merged pair free of superimposed discharges, which would snap
    # to one shared peak and appear as a duplicate marker: nudge B by 5 ms.
    offsets = trains[unit_b][:, None] - trains[unit_a][None, :]
    nearest = offsets[np.arange(len(offsets)), np.abs(offsets).argmin(axis=1)]
    clash = np.abs(nearest) <= 0.004 * sampling_rate
    nudge = np.where(nearest[clash] < 0, -1, 1) * int(0.005 * sampling_rate)
    trains[unit_b][clash] += nudge - nearest[clash]
    missed = _choose_missed(trains[unit_a], trains[unit_b], sampling_rate)

    sources, timestamps = [], []
    missed_peak = -1
    for index, units in enumerate(SOURCE_UNITS):
        source = rng.normal(0.0, 0.035, n_samples).astype(np.float32)
        marked = []
        for rank, unit in enumerate(units):
            events = trains[unit]
            if len(units) > 1:
                level = 1.45 if rank == 0 else 0.62
            else:
                level = 1.2 + 0.04 * index
            heights = level * rng.normal(1.0, 0.04, len(events))
            if index == 0 and rank == 0:
                heights[events == missed] = 1.32
                events_marked = events[events != missed]
            else:
                events_marked = events
            source += _pulse_train(n_samples, events, heights)
            marked.append(events_marked)
        crosstalk = trains[SOURCE_UNITS[(index + 3) % len(SOURCE_UNITS)][0]]
        source += _pulse_train(n_samples, crosstalk, np.full(len(crosstalk), 0.16))
        nominal = np.sort(np.concatenate(marked))
        timestamps.append(
            np.unique(
                snap_to_local_peak(source, nominal, max_shift=6, square_source=False)
            )
        )
        if index == 0:
            missed_peak = int(
                snap_to_local_peak(
                    source, np.asarray([missed]), max_shift=6, square_source=False
                )[0]
            )
        sources.append(source)

    inspect_target = INSPECT_DISCHARGE_S * sampling_rate
    tall = timestamps[0][sources[0][timestamps[0]] > 1.0]
    inspect_peak = int(min(tall, key=lambda sample: abs(sample - inspect_target)))
    emg = _illustrative_emg(trains, n_samples, sampling_rate)
    data = {
        "ports": ["Example_Grid"],
        "sampling_rate": sampling_rate,
        "discharge_times": [timestamps],
        "pulse_trains": [sources],
        "mu_filters": [[None for _source in sources]],
        "peel_off_sequence": [
            [
                {"accepted_unit_idx": unit_index, "timestamps": unit_ts.copy()}
                for unit_index, unit_ts in enumerate(timestamps)
            ]
        ],
        "preprocessing_config": [
            {
                "sampling_frequency": sampling_rate,
                "extension_factor": 1,
                "peel_off_window_size": 256,
                "min_peak_separation": 20,
                "square_sources_spike_det": True,
            }
        ],
        "skip_filter_recalc": True,
        "plateau_coords": [0, n_samples],
        "data": emg,
        "chans_per_electrode": [64],
        "channel_indices": [np.arange(64)],
        "emg_mask": [np.zeros(64, dtype=np.int8)],
        "electrodes": ["GR10MM0808"],
        "aux_channels": [
            {
                "name": "Force",
                "type": "force",
                "unit": "N",
                "data": force * PLATEAU_MVC * MVC_N,
                "mvc": MVC_N,
            }
        ],
    }
    return DemoSession(data, sampling_rate, missed_peak, inspect_peak)


def _decomposition_iterations(sampling_rate: int):
    """Illustrative convergence of one source, with its true silhouette."""
    rng = np.random.default_rng(5)
    n_samples = int(DURATION_S * sampling_rate)
    force = _force_profile(n_samples, sampling_rate)
    trains = [
        _discharge_times(force, threshold, sampling_rate, rng)
        for threshold in THRESHOLDS
    ]
    start = int(FORCE_KNOTS_S[1] * sampling_rate)
    segment = slice(start, start + 25600)

    def pulses(unit: int, level: float) -> np.ndarray:
        events = trains[unit]
        heights = level * rng.normal(1.0, 0.04, len(events))
        return _pulse_train(n_samples, events, heights)[segment]

    target = pulses(SOURCE_UNITS[1][0], 1.4)
    confound = pulses(SOURCE_UNITS[2][0], 1.1) + pulses(SOURCE_UNITS[3][0], 0.9)
    noise = rng.normal(0.0, 1.0, (2, len(target)))
    for iteration, weight in ((1, 0.3), (5, 0.55), (11, 0.8), (18, 1.0)):
        source = (
            weight * target
            + (1.0 - weight) * (confound + 0.25 * noise[0])
            + 0.035 * noise[1]
        )
        peaks, _ = sp_signal.find_peaks(
            source, height=0.5 * source.max(), distance=int(0.02 * sampling_rate)
        )
        spike_train = np.zeros((len(source), 1), dtype=bool)
        spike_train[peaks, 0] = True
        silhouette = float(
            np.ravel(tb_props.get_silhouette_measure(spike_train, source[:, None]))[0]
        )
        yield iteration, source, peaks, silhouette


# ── Capture ──────────────────────────────────────────────────────────────────


def _grab(window: MainWindow) -> Image.Image:
    """Capture the window at RENDER_SCALE times its logical size."""
    QApplication.processEvents()
    QTest.qWait(250)
    qimage = window.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    image = Image.frombuffer(
        "RGB",
        (qimage.width(), qimage.height()),
        bytes(qimage.constBits()),
        "raw",
        "RGB",
        qimage.bytesPerLine(),
        1,
    )
    size = (WINDOW_WIDTH * RENDER_SCALE, WINDOW_HEIGHT * RENDER_SCALE)
    return image.resize(size, Image.Resampling.LANCZOS)


def _rect(window: QWidget, widget: QWidget) -> Rect:
    origin = widget.mapTo(window, QPoint(0, 0))
    return (origin.x(), origin.y(), widget.width(), widget.height())


def _plot_point(window: QWidget, plot, x: float, y: float) -> tuple[float, float]:
    scene_point = plot.getViewBox().mapViewToScene(QPointF(x, y))
    local = plot.mapFromScene(scene_point)
    origin = plot.viewport().mapTo(window, local)
    return (origin.x(), origin.y())


def _tab_rect(window: QWidget, tab_bar, index: int) -> Rect:
    rect = tab_bar.tabRect(index)
    origin = tab_bar.mapTo(window, rect.topLeft())
    return (origin.x(), origin.y(), rect.width(), rect.height())


def _set_sidebar_width(tab: QWidget, width: int) -> None:
    for splitter in tab.findChildren(QSplitter):
        if splitter.orientation() == Qt.Orientation.Horizontal:
            splitter.setSizes([width, WINDOW_WIDTH - width])
            return


def _edition_marks(window: MainWindow) -> dict[str, tuple[float, ...]]:
    edition = window.edition_tab
    quality = edition.quality_bar
    marks = {
        "source": _rect(window, edition.source_plot),
        "rate": _rect(window, edition.fr_plot),
        "muap": _rect(window, edition.muap_widget),
        "quality": _rect(window, quality),
        "badge": _rect(window, quality._reliability_badge),
        "sil": _rect(window, quality._r_sil),
        "cov": _rect(window, quality._r_cov),
        "split": _rect(window, edition.btn_split_unit),
        "add": _rect(window, edition.btn_sel_add),
        "unit": _rect(window, edition.mu_combo),
    }
    if edition.btn_confirm_split.isVisible():
        marks["confirm"] = _rect(window, edition.btn_confirm_split)
    return marks


def capture_shots() -> dict[str, Shot]:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("SCD-Edition demo capture")
    set_style_sheet(app)
    window = MainWindow()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.show()
    shots: dict[str, Shot] = {}

    _configure_example_on_startup(window)
    config_tab = window.config_tab
    config_tab.path_edit.setText("examples/emg.mat")
    config_tab.output_dir_edit.setText("scd-edition-output")
    window.tabs.setCurrentWidget(config_tab)
    image = _grab(window)
    shots["config"] = Shot(image, {"apply": _rect(window, config_tab.apply_btn)})

    config_tab._apply_config()
    decomp_tab = window.decomp_tab
    window.tabs.setCurrentWidget(decomp_tab)
    _set_sidebar_width(decomp_tab, 450)
    image = _grab(window)
    decomp_marks = {
        "start": _rect(window, decomp_tab.start_btn),
        "plot": _rect(window, decomp_tab.canvas),
    }
    shots["decomp_ready"] = Shot(image, decomp_marks)
    # Mirror the running state: parameters locked, Stop offered.
    decomp_tab._set_params_enabled(False)
    decomp_tab.start_btn.setEnabled(False)
    decomp_tab.stop_btn.setVisible(True)
    session = _demo_session()
    for index, (iteration, source, peaks, silhouette) in enumerate(
        _decomposition_iterations(session.sampling_rate)
    ):
        decomp_tab._plot_source_realtime(source, peaks, iteration, silhouette)
        shots[f"decomp_{index}"] = Shot(_grab(window), decomp_marks)

    edition = window.edition_tab
    edition._loaded_path = Path("bundled_example_decomposition.pkl")
    edition._load_decomposition_data(session.data)
    edition._update_file_label()
    edition.file_loaded.emit()
    window.tabs.setCurrentWidget(edition)
    _set_sidebar_width(edition, 540)
    edition.source_plot.setXRange(*SPLIT_VIEW_S, padding=0)
    image = _grab(window)
    shots["edit_merged"] = Shot(image, _edition_marks(window))

    fs = session.sampling_rate
    source = session.data["pulse_trains"][0][0]
    edition._inspect_spike_muap(session.inspect_peak)
    image = _grab(window)
    marks = _edition_marks(window)
    marks["spike"] = _plot_point(
        window,
        edition.source_plot,
        session.inspect_peak / fs,
        float(source[session.inspect_peak]) ** 2,
    )
    shots["edit_inspect"] = Shot(image, marks)
    edition._handle_escape()

    edition.btn_split_unit.click()
    image = _grab(window)
    shots["edit_split_preview"] = Shot(image, _edition_marks(window))
    edition.btn_confirm_split.click()
    image = _grab(window)
    shots["edit_split_done"] = Shot(image, _edition_marks(window))

    gap = session.missed_peak / fs
    edition.source_plot.setXRange(
        gap - ADD_VIEW_HALF_WIDTH_S, gap + ADD_VIEW_HALF_WIDTH_S, padding=0
    )
    image = _grab(window)
    shots["edit_add_view"] = Shot(image, _edition_marks(window))
    edition.btn_sel_add.setChecked(True)
    image = _grab(window)
    height = float(source[session.missed_peak]) ** 2
    box = (gap - 0.035, gap + 0.035, 0.6 * height, 1.3 * height)
    marks = _edition_marks(window)
    marks["box_start"] = _plot_point(window, edition.source_plot, box[0], box[3])
    marks["box_end"] = _plot_point(window, edition.source_plot, box[1], box[2])
    marks["peak"] = _plot_point(window, edition.source_plot, gap, height)
    shots["edit_add_armed"] = Shot(image, marks)
    edition._apply_selection_add(*box)
    image = _grab(window)
    shots["edit_add_done"] = Shot(image, marks | _edition_marks(window))

    vis = window.vis_tab
    window.tabs.setCurrentWidget(vis)
    tab_bar = vis._inner_tabs.tabBar()
    for index, name in ((0, "vis_raster"), (1, "vis_idr")):
        vis._inner_tabs.setCurrentIndex(index)
        vis.on_tab_activated()
        image = _grab(window)
        shots[name] = Shot(
            image,
            {
                "plot": _rect(window, vis._inner_tabs.currentWidget()),
                "idr_tab": _tab_rect(window, tab_bar, 1),
            },
        )

    edition._undo_stack.clear()
    edition._set_dirty(False)
    window.close()
    return shots


# ── README GIF and screenshots ───────────────────────────────────────────────


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_DIR / name, size=size)
    except OSError:
        return ImageFont.load_default()


def _gif_frame(image: Image.Image, step: str, caption: str) -> Image.Image:
    """Screenshot above a caption band, so the caption never hides the UI."""
    screen = image.resize((GIF_WIDTH, GIF_SCREEN_HEIGHT), Image.Resampling.LANCZOS)
    frame = Image.new(
        "RGB", (GIF_WIDTH, GIF_SCREEN_HEIGHT + GIF_CAPTION_HEIGHT), (12, 17, 24)
    )
    frame.paste(screen, (0, 0))
    draw = ImageDraw.Draw(frame)
    middle = GIF_SCREEN_HEIGHT + GIF_CAPTION_HEIGHT // 2
    draw.rounded_rectangle((24, middle - 20, 96, middle + 20), radius=10, fill=ACCENT)
    draw.text(
        (60, middle),
        step,
        font=_font("Lexend-SemiBold.ttf", 19),
        fill="white",
        anchor="mm",
    )
    draw.text(
        (116, middle),
        caption,
        font=_font("Lexend-Medium.ttf", 25),
        fill="white",
        anchor="lm",
    )
    return frame


def _with_transitions(stills: list[Image.Image]) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    for index, still in enumerate(stills):
        frames.append(still)
        durations.append(
            STEP_DURATION_MS if index < len(stills) - 1 else FINAL_STEP_DURATION_MS
        )
        if index == len(stills) - 1:
            continue
        for blend in (0.25, 0.5, 0.75):
            frames.append(Image.blend(still, stills[index + 1], blend))
            durations.append(TRANSITION_DURATION_MS)
    return frames, durations


def _gif_palette(stills: list[Image.Image]) -> Image.Image:
    """One palette for every frame, with the interface's semantic colours kept.

    Per-frame adaptive palettes spend their entries on large areas, so small
    status text (green "reliable", red failing metrics) loses its hue.
    """
    reserved = [
        ImageColor.getrgb(colour)
        for colour in (
            COLORS["success"],
            COLORS["error"],
            COLORS["warning"],
            COLORS["info"],
            COLORS["foreground"],
            COLORS["text_dim"],
            "#ed8936",  # split A and spike markers
            "#22d3ee",  # split B
            "#2b6cb0",  # source signal
            "#ffd700",  # force overlay
            "#ffffff",
        )
    ]
    width, height = stills[0].size
    mosaic = Image.new("RGB", (width, height * len(stills)))
    for index, still in enumerate(stills):
        mosaic.paste(still, (0, index * height))
    base = mosaic.quantize(colors=256 - len(reserved))
    colours = base.getpalette()[: 3 * (256 - len(reserved))]
    for colour in reserved:
        colours.extend(colour)
    palette = Image.new("P", (1, 1))
    palette.putpalette(colours)
    return palette


def write_readme_assets(
    shots: dict[str, Shot], output: Path, screenshots_dir: Path
) -> None:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    for name, shot_name in SCREENSHOTS.items():
        path = screenshots_dir / f"{name}.png"
        shots[shot_name].image.resize(
            (WINDOW_WIDTH, WINDOW_HEIGHT), Image.Resampling.LANCZOS
        ).save(path, format="PNG", optimize=True, compress_level=9)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", flush=True)

    stills = [
        _gif_frame(shots[shot_name].image, f"{index:02d}", caption)
        for index, (shot_name, caption) in enumerate(README_STEPS, start=1)
    ]
    frames, durations = _with_transitions(stills)
    palette = _gif_palette(stills)
    paletted = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/demo.gif"), help="output GIF path"
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("docs/screenshots"),
        help="directory for clean README screenshots",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="also render the social-media videos (MP4) into this directory",
    )
    parser.add_argument(
        "--aspect",
        choices=("square", "portrait", "both"),
        default="both",
        help="video aspect ratio: square 1:1, portrait 4:5, or both",
    )
    args = parser.parse_args()
    shots = capture_shots()
    write_readme_assets(shots, args.output, args.screenshots_dir)
    if args.video_dir is not None:
        from demo_video import render_video

        aspects = ("square", "portrait") if args.aspect == "both" else (args.aspect,)
        for aspect in aspects:
            render_video(
                shots,
                args.video_dir / f"scd-edition-{aspect}.mp4",
                aspect,
                RENDER_SCALE,
            )


if __name__ == "__main__":
    main()
