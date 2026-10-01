"""Capture the real Qt interface for the README demo, screenshots and video.

The window is rendered by the platform's native Qt plugin at twice its
logical resolution but is never shown on screen. The "offscreen" plugin must
not be used: it has no system font fallback and draws every symbol glyph in
the interface as an empty box.

The interface is driven with a synthetic 64-channel recording of a 30 % MVC
trapezoidal contraction: motor units are recruited in order of size with
physiological discharge variability, and their MUAPs propagate from each
unit's innervation zone. The sources are what the application's own filter
recalculation produces, spike-triggered-average filters applied to the
extended and whitened EMG, except that each decomposition filter carries a
stray component, as an imperfectly converged separation vector would, so
recalculating it after editing strengthens the pulses. The first unit has a
false discharge marked on another unit's crosstalk and misses one of its own;
the second merges two units to demonstrate the split preview. No
decomposition is run: the Decomposition frames show one filter converging
from a random start onto the first unit.

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
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QPen
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSplitter, QWidget
from scd.models.timestamping import spike_triggered_average
from scipy import signal as sp_signal

from scd_app._vendor.motor_unit_toolbox import props as tb_props
from scd_app.core.filter_recalculation import preprocess_emg, snap_to_local_peak
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
EXTENSION_FACTOR = 16
# Per true unit: recruitment threshold as a fraction of the plateau force,
# territory centre and spread on the 8 x 8 grid (row, column), row of the
# innervation zone, and weight of the leading MUAP phase.
THRESHOLDS = (0.04, 0.12, 0.20, 0.28, 0.36, 0.46, 0.56, 0.66, 0.76)
CENTRES = (
    (1.5, 1.5),
    (5.5, 2.0),
    (2.0, 5.5),
    (3.5, 3.5),
    (6.5, 6.0),
    (3.0, 1.0),
    (1.0, 6.5),
    (6.0, 0.8),
    (4.0, 7.0),
)
SPREADS = ((2.0, 1.4),) * 3 + ((3.0, 2.2),) + ((2.0, 1.4),) * 5
IZ_ROWS = (3.5, 2.0, 5.0, 3.5, 1.5, 6.0, 2.5, 4.5, 3.0)
LEADING_PHASE = (0.55, 0.25, 0.8, 0.5, 0.65, 0.3, 0.7, 0.5, 0.2)
EDITED_UNIT = 3  # shown first: one false and one missed discharge
MERGED_UNITS = (2, 4)  # shown second: merged, with the first dominant
# Decomposed sources in extraction order.
SOURCE_UNITS = ((EDITED_UNIT,), MERGED_UNITS, (0,), (5,), (1,), (7,), (8,), (6,))
EDIT_VIEW_S = (4.3, 5.8)
SPLIT_VIEW_S = (3.8, 6.4)
ZOOM_STEPS = 20

README_STEPS = (
    ("config", "Load the bundled 64-channel example"),
    ("decomp_3", "Decompose with live source feedback"),
    ("edit_review", "Review each unit's pulse train against the force"),
    ("edit_inspect_true", "Click a discharge to overlay its MUAP"),
    ("edit_added", "Box-select to delete false and add missed discharges"),
    ("edit_recalculated", "Recalculate the filter from the edited discharges"),
    ("edit_split_preview", "Split merged units: high-amplitude A, low-amplitude B"),
    ("vis_idr", "Compare discharge rates with the force"),
)
SCREENSHOTS = {
    "configuration": "config",
    "decomposition": "decomp_3",
    "edition": "edit_split_preview",
    "visualisation": "vis_idr_screenshot",
}
SCREENSHOT_LINE_SCALE = 3.5  # IDR and force traces in the visualisation screenshot

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
    false_peak: int  # marked by the decomposition on another unit's crosstalk
    missed_peak: int  # a discharge the decomposition left unmarked
    true_peak: int  # a genuine discharge to inspect, before the false one
    edit_top: float  # source² axis top fitting the recalculated pulses
    iterations: list[tuple[int, np.ndarray, np.ndarray, float]]


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


def _muap_templates(unit: int, half_window: int, sampling_rate: int) -> np.ndarray:
    """Spatially localised 64-channel MUAP of one unit (µV).

    The action potential travels both ways from the innervation zone at
    5 m/s, i.e. 2 ms per 10 mm row of the grid.
    """
    rows, cols = np.divmod(np.arange(64), 8)
    centre_row, centre_col = CENTRES[unit]
    spread_row, spread_col = SPREADS[unit]
    spatial = np.exp(
        -((rows - centre_row) ** 2) / (2 * spread_row**2)
        - (cols - centre_col) ** 2 / (2 * spread_col**2)
    )
    delay = np.abs(rows - IZ_ROWS[unit]) * 0.002 * sampling_rate / half_window
    width = 0.8 + 0.4 * ((unit * 5) % len(THRESHOLDS)) / (len(THRESHOLDS) - 1)
    phase = np.linspace(-1.0, 1.0, 2 * half_window, endpoint=False)
    phase = (phase[np.newaxis, :] - delay[:, np.newaxis]) / width
    lead = LEADING_PHASE[unit]
    shape = (
        -lead * np.exp(-(((phase + 0.24) / 0.16) ** 2))
        + np.exp(-((phase / 0.11) ** 2))
        - 0.8 * (1 - lead) * np.exp(-(((phase - 0.27) / 0.18) ** 2))
    )
    templates = spatial[:, np.newaxis] * shape
    peak_to_peak = 90.0 + 230.0 * THRESHOLDS[unit]  # larger units recruit later
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


def _unit_norm(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


class _Separation:
    """The application's filter maths on the extended, whitened EMG."""

    def __init__(self, emg: np.ndarray, config: dict) -> None:
        self.whitened = preprocess_emg(
            torch.from_numpy(emg.T.copy()), config, torch.device("cpu")
        )

    def matched(self, timestamps: np.ndarray) -> torch.Tensor:
        """Spike-triggered-average filter, as filter recalculation estimates it."""
        events = torch.from_numpy(np.asarray(timestamps, dtype=np.int64))
        return _unit_norm(spike_triggered_average(self.whitened, events, 1).t())

    def source(self, filt: torch.Tensor) -> np.ndarray:
        source = (self.whitened @ filt).squeeze(-1)
        return ((source - source.mean()) / source.std()).numpy().astype(np.float64)


def _plant_edits(
    source: np.ndarray, timestamps: np.ndarray, sampling_rate: int
) -> tuple[int, int, int]:
    """Choose the false, missed and inspected discharges in the edit view.

    The false one is the tallest crosstalk peak clear of the unit's own
    discharges; the missed one is the unit's lowest peak there, as the one a
    decomposition would most plausibly miss.
    """
    lo, hi = (int(s * sampling_rate) for s in EDIT_VIEW_S)
    margin = int(0.15 * sampling_rate)
    squared = source**2
    peaks, _ = sp_signal.find_peaks(
        squared[lo + margin : hi - margin], distance=int(0.01 * sampling_rate)
    )
    peaks += lo + margin
    clear = [p for p in peaks if np.min(np.abs(timestamps - p)) > 0.025 * sampling_rate]
    false_peak = int(max(clear, key=lambda p: squared[p]))
    inside = timestamps[(timestamps > lo + margin) & (timestamps < hi - margin)]
    away = inside[np.abs(inside - false_peak) > 0.1 * sampling_rate]
    missed_peak = int(min(away, key=lambda p: squared[p]))
    # Inspect a typical discharge before the false one.
    before = inside[(inside < false_peak) & (inside != missed_peak)]
    true_peak = int(before[np.argsort(squared[before])[len(before) // 2]])
    return false_peak, missed_peak, true_peak


def _decomposition_iterations(
    separation: _Separation, target: torch.Tensor, start: torch.Tensor, fs: int
) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    """One separation vector converging from *start* onto *target*."""
    first = int(FORCE_KNOTS_S[1] * fs)
    segment = slice(first, first + 25600)
    iterations = []
    for iteration, weight in ((1, 0.25), (5, 0.5), (11, 0.75), (18, 1.0)):
        filt = _unit_norm(weight * target + (1 - weight) * start)
        source = separation.source(filt)[segment]
        squared = source**2
        peaks, _ = sp_signal.find_peaks(
            squared, height=0.3 * squared.max(), distance=int(0.02 * fs)
        )
        spike_train = np.zeros((len(source), 1), dtype=bool)
        spike_train[peaks, 0] = True
        silhouette = float(
            np.ravel(tb_props.get_silhouette_measure(spike_train, squared[:, None]))[0]
        )
        iterations.append((iteration, source, peaks, silhouette))
    return iterations


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
    unit_a, unit_b = MERGED_UNITS
    # Keep the merged pair free of superimposed discharges, which would snap
    # to one shared peak and appear as a duplicate marker: nudge B by 5 ms.
    offsets = trains[unit_b][:, None] - trains[unit_a][None, :]
    nearest = offsets[np.arange(len(offsets)), np.abs(offsets).argmin(axis=1)]
    clash = np.abs(nearest) <= 0.004 * sampling_rate
    nudge = np.where(nearest[clash] < 0, -1, 1) * int(0.005 * sampling_rate)
    trains[unit_b][clash] += nudge - nearest[clash]

    emg = _illustrative_emg(trains, n_samples, sampling_rate)
    preprocessing = {
        "sampling_frequency": sampling_rate,
        "extension_factor": EXTENSION_FACTOR,
        "peel_off_window_size": 256,
        "min_peak_separation": 20,
        "square_sources_spike_det": True,
    }
    separation = _Separation(emg, preprocessing)
    stray = torch.from_numpy(
        np.random.default_rng(7)
        .normal(size=(separation.whitened.shape[1], 3))
        .astype(np.float32)
    )
    stray = stray / stray.norm(dim=0)
    filters = [
        _unit_norm(separation.matched(trains[EDITED_UNIT]) + 0.9 * stray[:, :1]),
        _unit_norm(
            separation.matched(trains[unit_a])
            + 0.5 * separation.matched(trains[unit_b])
            + 0.3 * stray[:, 1:2]
        ),
        *(separation.matched(trains[units[0]]) for units in SOURCE_UNITS[2:]),
    ]
    sources = [separation.source(filt) for filt in filters]
    timestamps = [
        np.unique(
            snap_to_local_peak(
                source,
                np.sort(np.concatenate([trains[unit] for unit in units])),
                max_shift=6,
                square_source=True,
            )
        )
        for source, units in zip(sources, SOURCE_UNITS, strict=True)
    ]
    false_peak, missed_peak, true_peak = _plant_edits(
        sources[0], timestamps[0], sampling_rate
    )
    decomposed = timestamps[0][timestamps[0] != missed_peak]
    timestamps[0] = np.sort(np.append(decomposed, false_peak))

    # Fit the edit view's axis to the pulses the recalculation will produce.
    curated = np.sort(np.append(decomposed, missed_peak))
    recalculated = separation.source(separation.matched(curated))
    lo, hi = (int(s * sampling_rate) for s in EDIT_VIEW_S)
    edit_top = 1.12 * float(np.max(recalculated[lo:hi] ** 2))
    iterations = _decomposition_iterations(
        separation, filters[0], stray[:, 2:3], sampling_rate
    )
    del separation

    data = {
        "ports": ["Example_Grid"],
        "sampling_rate": sampling_rate,
        "discharge_times": [timestamps],
        "pulse_trains": [sources],
        "mu_filters": [[filt.numpy() for filt in filters]],
        "peel_off_sequence": [
            [
                {"accepted_unit_idx": unit_index, "timestamps": unit_ts.copy()}
                for unit_index, unit_ts in enumerate(timestamps)
            ]
        ],
        "preprocessing_config": [preprocessing],
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
    return DemoSession(
        data,
        sampling_rate,
        false_peak,
        missed_peak,
        true_peak,
        edit_top,
        iterations,
    )


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


def _thicken_curves(plot, factor: float) -> None:
    """Widen every curve of a plot, which stays so until it is re-rendered."""
    for item in plot.listDataItems():
        pen = QPen(item.opts["pen"])
        pen.setWidthF(pen.widthF() * factor)
        item.setPen(pen)


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
        "recalc": _rect(window, edition.btn_recalc_filter),
    }
    if edition.btn_confirm_split.isVisible():
        marks["confirm"] = _rect(window, edition.btn_confirm_split)
    return marks


def _pulse_mark(window: MainWindow, sample: int, fs: int) -> tuple[float, float]:
    """Window position of the tip of one pulse of the current unit."""
    edition = window.edition_tab
    source = edition._current_mu().source
    return _plot_point(
        window, edition.source_plot, sample / fs, float(source[sample]) ** 2
    )


def _box(
    window: MainWindow,
    sample: int,
    fs: int,
    low: float,
    high: float,
    half_width_s: float,
) -> tuple[tuple[float, float, float, float], dict[str, tuple[float, float]]]:
    """Selection box around one pulse tip, in plot units and window pixels.

    *low* and *high* bound the box as fractions of the pulse's squared height.
    """
    edition = window.edition_tab
    height = float(edition._current_mu().source[sample]) ** 2
    t = sample / fs
    box = (t - half_width_s, t + half_width_s, low * height, high * height)
    plot = edition.source_plot
    return box, {
        "box_start": _plot_point(window, plot, box[0], box[3]),
        "box_end": _plot_point(window, plot, box[1], box[2]),
    }


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
    for index, (iteration, source, peaks, silhouette) in enumerate(session.iterations):
        decomp_tab._plot_source_realtime(source, peaks, iteration, silhouette)
        shots[f"decomp_{index}"] = Shot(_grab(window), decomp_marks)

    edition = window.edition_tab
    edition._loaded_path = Path("bundled_example_decomposition.pkl")
    edition._load_decomposition_data(session.data)
    edition._update_file_label()
    edition.file_loaded.emit()
    window.tabs.setCurrentWidget(edition)
    _set_sidebar_width(edition, 540)
    fs = session.sampling_rate
    image = _grab(window)
    shots["edit_review"] = Shot(image, _edition_marks(window))

    # Zoom the application itself from the whole contraction to the edits.
    view = edition.source_plot.getViewBox()
    (x0, x1), (_bottom, top) = view.viewRange()
    centre, width = (x0 + x1) / 2, x1 - x0
    edit_centre = sum(EDIT_VIEW_S) / 2
    edit_width = EDIT_VIEW_S[1] - EDIT_VIEW_S[0]
    for step in range(1, ZOOM_STEPS + 1):
        u = step / ZOOM_STEPS
        u = u * u * (3 - 2 * u)
        span = width * (edit_width / width) ** u
        middle = centre + (edit_centre - centre) * u
        view.setRange(
            xRange=(middle - span / 2, middle + span / 2),
            yRange=(0.0, top + (session.edit_top - top) * u),
            padding=0,
        )
        shots[f"edit_zoom_{step}"] = Shot(_grab(window), _edition_marks(window))

    for name, sample in (
        ("edit_inspect_true", session.true_peak),
        ("edit_inspect_false", session.false_peak),
    ):
        edition._inspect_spike_muap(sample)
        image = _grab(window)
        marks = _edition_marks(window) | {"spike": _pulse_mark(window, sample, fs)}
        shots[name] = Shot(image, marks)
    edition._handle_escape()

    delete_box, marks = _box(window, session.false_peak, fs, 0.3, 1.7, 0.035)
    edition.btn_sel_delete.setChecked(True)
    shots["edit_delete_armed"] = Shot(_grab(window), _edition_marks(window) | marks)
    edition._apply_selection_delete(*delete_box)
    shots["edit_deleted"] = Shot(_grab(window), _edition_marks(window) | marks)

    add_box, marks = _box(window, session.missed_peak, fs, 0.6, 1.35, 0.02)
    edition.btn_sel_add.setChecked(True)
    shots["edit_add_armed"] = Shot(_grab(window), _edition_marks(window) | marks)
    edition._apply_selection_add(*add_box)
    shots["edit_added"] = Shot(_grab(window), _edition_marks(window) | marks)
    edition.btn_sel_add.setChecked(False)

    edition.btn_recalc_filter.click()
    shots["edit_recalculated"] = Shot(_grab(window), _edition_marks(window))

    edition.btn_reviewed.click()
    edition.btn_next_unreviewed.click()
    view.enableAutoRange(axis=view.YAxis)
    edition.source_plot.setXRange(*SPLIT_VIEW_S, padding=0)
    shots["edit_merged"] = Shot(_grab(window), _edition_marks(window))
    edition.btn_split_unit.click()
    shots["edit_split_preview"] = Shot(_grab(window), _edition_marks(window))
    edition.btn_confirm_split.click()
    shots["edit_split_done"] = Shot(_grab(window), _edition_marks(window))

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
    # The README screenshot alone gets thicker traces, legible at full width.
    _thicken_curves(vis._idr_plot, SCREENSHOT_LINE_SCALE)
    shots["vis_idr_screenshot"] = Shot(_grab(window))

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
