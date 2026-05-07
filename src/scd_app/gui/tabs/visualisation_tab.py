"""
Visualisation Tab (Tab 4) — interactive discharge-rate plots for the current file.

Inner tabs: Raster | IDR | CST
X-axes are linked across all three inner tabs.
AUX force channels can be overlaid on any plot and toggled via floating legend (top-right).
"""

import re
from typing import Dict, List, Optional, Set

import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QScrollArea,
    QFrame,
)
import pyqtgraph as pg

from motor_unit_toolbox.props import get_inst_discharge_rate

from scd_app.core.mu_model import MotorUnit
from scd_app.gui.style.styling import COLORS, FONT_SIZES

# ── Filename → active-aux helpers ────────────────────────────────────────────

_FINGER_ABBREV = {"T": "Thumb", "I": "Index", "M": "Middle", "R": "Ring", "L": "Little"}
_MOTION_ABBREV = {"ext": "Ext", "flex": "Flex"}
_TASK_PATTERN = re.compile(r"mvc-\d+(ext|flex)_fing-([TIMRL]+)", re.IGNORECASE)


def _parse_task_targets(file_stem: str) -> Optional[Set[str]]:
    """Return expected active unit strings (e.g. {'Index Ext'}) from the filename,
    or None if the filename doesn't match the expected pattern."""
    m = _TASK_PATTERN.search(file_stem)
    if not m:
        return None
    motion = _MOTION_ABBREV.get(m.group(1).lower())
    if not motion:
        return None
    targets = {
        f"{_FINGER_ABBREV[c]} {motion}"
        for c in m.group(2).upper()
        if c in _FINGER_ABBREV
    }
    return targets or None


def _default_aux_states(channels: list, file_stem: str) -> List[bool]:
    """Return initial on/off states for aux channels based on the filename.
    Falls back to all-on when the filename cannot be parsed or nothing matches."""
    targets = _parse_task_targets(file_stem)
    if not targets:
        return [True] * len(channels)
    states = []
    for ch in channels:
        meta = ch.get("meta", {})
        unit = meta.get("unit") or ch.get("unit") or ""
        states.append(unit in targets)
    # If nothing matched, show all (avoids blank plots)
    if not any(states):
        return [True] * len(channels)
    return states


# ── Colour constants ──────────────────────────────────────────────────────────

_AUX_COLORS = [(255, 215, 0), (192, 239, 255), (255, 179, 71)]
_AUX_COLORS_HEX = ["#FFD700", "#C0EFFF", "#FFB347"]

# ── Colour helpers ─────────────────────────────────────────────────────────────


def _blue_palette(n: int) -> List[tuple]:
    """Generate n colours linearly interpolated from #63B3ED to #003E74."""
    if n == 0:
        return []
    if n == 1:
        return [(0x63, 0xB3, 0xED)]
    light = np.array([0x63, 0xB3, 0xED], dtype=float)
    dark = np.array([0x00, 0x3E, 0x74], dtype=float)
    return [
        tuple(int(v) for v in light * (1 - i / (n - 1)) + dark * (i / (n - 1)))
        for i in range(n)
    ]


# ── Sort helpers ──────────────────────────────────────────────────────────────

_SORT_OPTIONS = ["Recruitment Threshold", "MU Index", "Mean Discharge Rate"]


def _sort_key_recruit(mu: MotorUnit, fsamp: float) -> float:
    if len(mu.timestamps) == 0:
        return float("inf")
    return float(mu.timestamps.min()) / fsamp


def _sort_key_index(mu: MotorUnit, _fsamp: float) -> float:
    return float(mu.id)


def _sort_key_mean_dr(mu: MotorUnit, _fsamp: float) -> float:
    if mu.props is None or np.isnan(mu.props.discharge_rate_hz):
        return float("inf")
    return float(mu.props.discharge_rate_hz)


_SORT_FNS = {
    "Recruitment Threshold": _sort_key_recruit,
    "MU Index": _sort_key_index,
    "Mean Discharge Rate": _sort_key_mean_dr,
}


# ── AUX floating legend ────────────────────────────────────────────────────────


class _VisAuxLegend(pg.LegendItem):
    """Floating click-to-toggle AUX legend anchored to the top-right of each plot.

    All three per-plot instances share the same _on_states list so toggling on
    any one plot keeps the others visually in sync after _on_toggle fires.
    """

    def __init__(self, on_states: List[bool], on_toggle_callback):
        super().__init__(offset=(-10, 10))  # top-right
        self._on_states = on_states  # shared mutable reference
        self._on_toggle = on_toggle_callback
        self._names: List[str] = []
        self._colors_hex: List[str] = []

    def clear(self):
        for _sample, label in self.items:
            self.layout.removeItem(label)
            label.close()
        self.items = []
        self.updateSize()

    def populate(self, channels: list):
        self.clear()
        self._names = []
        self._colors_hex = []
        for i, ch in enumerate(channels):
            meta = ch.get("meta", {})
            name = meta.get("name") or ch.get("name") or f"AUX {i + 1}"
            unit = meta.get("unit") or ch.get("unit") or ""
            display = f"{name} ({unit})" if unit else name
            hex_color = _AUX_COLORS_HEX[i % len(_AUX_COLORS_HEX)]
            self._names.append(display)
            self._colors_hex.append(hex_color)
            label = pg.LabelItem(f"● {display}", color=hex_color, justify="left")
            self.layout.addItem(label, i, 0)
            self.items.append((None, label))
        self.updateSize()
        self.setVisible(bool(channels))

    def sync_labels(self):
        """Refresh dot/colour from current _on_states."""
        for i, (_sample, label) in enumerate(self.items):
            if i >= len(self._on_states):
                break
            on = self._on_states[i]
            dot = "●" if on else "○"
            color = self._colors_hex[i] if on else "#555555"
            label.setText(f"{dot} {self._names[i]}", color=color)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            y = event.pos().y()
            for i, (_sample, label) in enumerate(self.items):
                rect = label.mapRectToParent(label.boundingRect())
                if rect.top() <= y <= rect.bottom():
                    if i < len(self._on_states):
                        self._on_states[i] = not self._on_states[i]
                    self._on_toggle()
                    event.accept()
                    return
        event.accept()


# ── Main tab ──────────────────────────────────────────────────────────────────


class VisualisationTab(QWidget):
    """Tab 4 — raster, IDR, and CST plots for the current edition file."""

    def __init__(self, edition_tab=None, parent=None):
        super().__init__(parent)

        self._edition_tab = edition_tab

        # Cached data snapshot (filled on tab activation)
        self._ports: Dict[str, List[MotorUnit]] = {}
        self._aux_channels: list = []
        self._fsamp: float = 2048.0
        self._start_sample: int = 0
        self._end_sample: int = 0
        self._file_stem: str = ""

        # UI state
        self._disabled_mus: Set[tuple] = set()
        self._sidebar_rows: Dict[tuple, QPushButton] = {}
        self._sidebar_port_rows: Dict[str, QPushButton] = {}
        self._aux_on_states: List[bool] = []
        self._aux_legends: List[_VisAuxLegend] = []

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(self._build_header())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        content_layout.addWidget(self._build_sidebar())
        content_layout.addWidget(self._build_plot_area(), stretch=1)

        root.addWidget(content, stretch=1)

    def _build_plot_area(self) -> QTabWidget:
        self._inner_tabs = QTabWidget()
        self._inner_tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {COLORS['border']}; }}"
            f"QTabBar::tab {{ color: {COLORS['foreground']}; "
            f"background: {COLORS['background_input']}; "
            f"border-radius: 4px 4px 0 0; padding: 4px 14px; margin-right: 2px; }}"
            f"QTabBar::tab:selected {{ background: {COLORS['accent']}; }}"
            f"QTabBar::tab:hover {{ background: {COLORS['accent_hover']}; }}"
        )

        self._raster_plot = self._make_plot_widget("Motor Unit", "Time (s)")
        self._idr_plot = self._make_plot_widget("Discharge Rate (pps)", "Time (s)")
        self._cst_plot = self._make_plot_widget("CST (pps)", "Time (s)")

        self._idr_plot.setXLink(self._raster_plot)
        self._cst_plot.setXLink(self._raster_plot)

        self._inner_tabs.addTab(self._raster_plot, "Raster")
        self._inner_tabs.addTab(self._idr_plot, "IDR")
        self._inner_tabs.addTab(self._cst_plot, "CST")

        # One floating AUX legend per plot; all share _aux_on_states
        for pw in (self._raster_plot, self._idr_plot, self._cst_plot):
            leg = _VisAuxLegend(self._aux_on_states, self._on_aux_toggled)
            leg.setParentItem(pw.plotItem.vb)
            leg.setVisible(False)
            self._aux_legends.append(leg)

        return self._inner_tabs

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(160)
        sidebar.setStyleSheet(
            f"background-color: {COLORS['background_light']}; border-radius: 4px;"
        )

        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(4, 6, 4, 6)
        outer.setSpacing(4)

        # All / None toggle row
        toggle_row = QWidget()
        toggle_row.setStyleSheet("background: transparent;")
        toggle_layout = QHBoxLayout(toggle_row)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(4)

        self._btn_all = QPushButton("All")
        self._btn_none = QPushButton("None")
        for btn in (self._btn_all, self._btn_none):
            btn.setFixedHeight(22)
            btn.setStyleSheet(self._small_btn_style())
        self._btn_all.clicked.connect(self._on_sidebar_all)
        self._btn_none.clicked.connect(self._on_sidebar_none)
        toggle_layout.addWidget(self._btn_all)
        toggle_layout.addWidget(self._btn_none)
        outer.addWidget(toggle_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLORS['border']}; background: {COLORS['border']};")
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        # Scrollable MU list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._sidebar_container = QWidget()
        self._sidebar_container.setStyleSheet("background: transparent;")
        self._sidebar_inner_layout = QVBoxLayout(self._sidebar_container)
        self._sidebar_inner_layout.setContentsMargins(0, 2, 0, 2)
        self._sidebar_inner_layout.setSpacing(1)
        self._sidebar_inner_layout.addStretch()

        scroll.setWidget(self._sidebar_container)
        outer.addWidget(scroll, stretch=1)

        return sidebar

    def _make_plot_widget(self, ylabel: str, xlabel: str) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(COLORS["background"])
        for axis in ("left", "bottom"):
            pw.getAxis(axis).setTextPen(pg.mkPen(color=COLORS["foreground"]))
            pw.getAxis(axis).setPen(pg.mkPen(color=COLORS["border"]))
        pw.setLabel(
            "left", ylabel, color=COLORS["text_secondary"], size=FONT_SIZES["small"]
        )
        pw.setLabel(
            "bottom", xlabel, color=COLORS["text_secondary"], size=FONT_SIZES["small"]
        )
        pw.showGrid(x=True, y=True, alpha=0.12)
        return pw

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {COLORS['background_light']}; border-radius: 4px;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: {FONT_SIZES['small']};"
        )
        layout.addWidget(sort_lbl)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(_SORT_OPTIONS)
        self._sort_combo.setStyleSheet(
            f"QComboBox {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['foreground']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 4px; padding: 2px 6px; "
            f"font-size: {FONT_SIZES['small']}; }}"
            f"QComboBox QAbstractItemView {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['foreground']}; }}"
        )
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        layout.addWidget(self._sort_combo)

        layout.addStretch()
        return bar

    # ── Style helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_sep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        sep.setFixedWidth(1)
        sep.setFixedHeight(20)
        return sep

    @staticmethod
    def _small_btn_style() -> str:
        return (
            f"QPushButton {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['foreground']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 4px; padding: 2px 6px; "
            f"font-size: {FONT_SIZES['small']}; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; }}"
        )

    @staticmethod
    def _sidebar_row_style(active: bool, color: tuple) -> str:
        r, g, b = color
        text_color = f"rgb({r},{g},{b})" if active else COLORS["text_muted"]
        return (
            f"QPushButton {{ background: transparent; color: {text_color}; "
            f"border: none; text-align: left; padding: 2px 6px; "
            f"font-size: {FONT_SIZES['small']}; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; "
            f"border-radius: 3px; }}"
        )

    @staticmethod
    def _port_btn_style(active: bool) -> str:
        text_color = COLORS["text_secondary"] if active else COLORS["text_muted"]
        return (
            f"QPushButton {{ background: transparent; color: {text_color}; "
            f"border: none; text-align: left; padding: 6px 4px 2px 4px; "
            f"font-size: {FONT_SIZES['normal']}; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; "
            f"border-radius: 3px; }}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_data_modified(self):
        """Called when edition data changes — no-op; refresh happens on tab activation."""
        pass

    @pyqtSlot()
    def on_tab_activated(self):
        """Called when the user switches to this tab — re-fetches and re-renders."""
        self._fetch_edition_data()
        self._render()

    # ── Sidebar rebuild ───────────────────────────────────────────────────────

    def _rebuild_sidebar_rows(self):
        while self._sidebar_inner_layout.count() > 1:
            item = self._sidebar_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sidebar_rows.clear()
        self._sidebar_port_rows.clear()

        default_color = (0x63, 0xB3, 0xED)
        insert_pos = 0

        for port_name, mus in self._ports.items():
            all_disabled = bool(mus) and all(
                (port_name, mu.id) in self._disabled_mus for mu in mus
            )
            dot = "○" if all_disabled else "●"
            port_btn = QPushButton(f"{dot}  {port_name}")
            port_btn.setFlat(True)
            port_btn.setStyleSheet(self._port_btn_style(not all_disabled))
            port_btn.clicked.connect(lambda _, p=port_name: self._toggle_port(p))
            self._sidebar_port_rows[port_name] = port_btn
            self._sidebar_inner_layout.insertWidget(insert_pos, port_btn)
            insert_pos += 1

            for mu in mus:
                key = (port_name, mu.id)
                active = key not in self._disabled_mus
                dot_mu = "●" if active else "○"
                btn = QPushButton(f"{dot_mu} MU {mu.id}")
                btn.setFlat(True)
                btn.setStyleSheet(self._sidebar_row_style(active, default_color))
                btn.clicked.connect(lambda _, k=key: self._toggle_mu(k))
                self._sidebar_rows[key] = btn
                self._sidebar_inner_layout.insertWidget(insert_pos, btn)
                insert_pos += 1

    def _update_sidebar_colours(self, sorted_active: List[tuple]):
        palette = _blue_palette(len(sorted_active))
        color_map: Dict[tuple, tuple] = {}
        for rank, (port_name, mu) in enumerate(sorted_active):
            color_map[(port_name, mu.id)] = palette[rank]

        default_color = (0x63, 0xB3, 0xED)
        for key, btn in self._sidebar_rows.items():
            port_name, mu_id = key
            active = key not in self._disabled_mus
            color = color_map.get(key, default_color)
            dot = "●" if active else "○"
            btn.setText(f"{dot} MU {mu_id}")
            btn.setStyleSheet(self._sidebar_row_style(active, color))

        for port_name, btn in self._sidebar_port_rows.items():
            mus = self._ports.get(port_name, [])
            all_disabled = bool(mus) and all(
                (port_name, mu.id) in self._disabled_mus for mu in mus
            )
            dot = "○" if all_disabled else "●"
            btn.setText(f"{dot}  {port_name}")
            btn.setStyleSheet(self._port_btn_style(not all_disabled))

    # ── AUX legend rebuild ────────────────────────────────────────────────────

    def _rebuild_aux_controls(self):
        self._aux_on_states.clear()
        self._aux_on_states.extend(_default_aux_states(self._aux_channels, self._file_stem))
        for leg in self._aux_legends:
            leg.populate(self._aux_channels)
        for leg in self._aux_legends:
            leg.sync_labels()

    # ── Fetch data from edition tab ───────────────────────────────────────────

    def _fetch_edition_data(self):
        if self._edition_tab is None:
            return
        d = self._edition_tab.get_visualisation_data()
        self._ports = d["ports"]
        self._aux_channels = d["aux_channels"]
        self._fsamp = d["fsamp"]
        self._start_sample = d["start_sample"]
        self._end_sample = d["end_sample"]
        self._file_stem = d.get("file_stem", "")
        self._rebuild_sidebar_rows()
        self._rebuild_aux_controls()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _get_active_mus(self) -> List[tuple]:
        return [
            (port_name, mu)
            for port_name, mus in self._ports.items()
            for mu in mus
            if (port_name, mu.id) not in self._disabled_mus
        ]

    def _sorted_mus(self, active: List[tuple]) -> List[tuple]:
        sort_name = self._sort_combo.currentText()
        key_fn = _SORT_FNS.get(sort_name, _sort_key_recruit)
        return sorted(active, key=lambda pair: key_fn(pair[1], self._fsamp))

    # Maximum number of points sent to pyqtgraph per trace.
    _MAX_DISPLAY_PTS = 4000

    def _render(self):
        active = self._get_active_mus()
        sorted_active = self._sorted_mus(active)
        self._render_raster(sorted_active)
        spike_matrix, t_axis, display_fs = self._build_idr_matrix(sorted_active)
        # Compute IDR once and share between the two sub-renderers.
        idr = None
        if spike_matrix is not None:
            idr = get_inst_discharge_rate(spike_matrix, int(display_fs))
        self._render_idr(sorted_active, idr, t_axis)
        self._render_cst(idr, t_axis)
        self._update_sidebar_colours(sorted_active)

    # Maximum sampling rate used internally for IDR/CST computation.
    # A 1-second Hanning window captures all meaningful firing-rate variation
    # well below 500 Hz, so there is no benefit computing at the raw rate.
    _IDR_MAX_FS = 1000  # Hz

    def _build_idr_matrix(self, sorted_mus: List[tuple]):
        if not sorted_mus:
            return None, None, self._IDR_MAX_FS

        all_ts = [mu.timestamps for _, mu in sorted_mus if len(mu.timestamps) > 0]
        if not all_ts:
            return None, None, self._IDR_MAX_FS

        # Downsample timestamps to a capped display rate so the spike matrix
        # stays tractable regardless of the acquisition rate.
        display_fs = min(float(self._fsamp), float(self._IDR_MAX_FS))
        ratio = self._fsamp / display_fs  # e.g. 10 for 10 kHz → 1 kHz

        ts_global_min = int(min(ts.min() for ts in all_ts))
        ts_global_max = int(max(ts.max() for ts in all_ts))

        min_disp = int(np.floor(ts_global_min / ratio))
        max_disp = int(np.ceil(ts_global_max / ratio))
        n_samples = max_disp - min_disp + 1

        spike_matrix = np.zeros((n_samples, len(sorted_mus)), dtype=bool)
        for col, (_, mu) in enumerate(sorted_mus):
            if len(mu.timestamps) == 0:
                continue
            ts_disp = np.round(mu.timestamps / ratio).astype(np.int64) - min_disp
            valid = ts_disp[(ts_disp >= 0) & (ts_disp < n_samples)]
            spike_matrix[valid, col] = True

        t_axis = (
            np.arange(n_samples) / display_fs + ts_global_min / self._fsamp
        )
        return spike_matrix, t_axis, display_fs

    def _render_raster(self, sorted_mus: List[tuple]):
        pw = self._raster_plot
        pw.clear()
        pw.getAxis("left").setTicks([[]])

        if not sorted_mus:
            return

        palette = _blue_palette(len(sorted_mus))
        fsamp = self._fsamp
        ticks = []
        for rank, (port_name, mu) in enumerate(sorted_mus):
            if len(mu.timestamps) == 0:
                ticks.append((rank, f"MU {mu.id}"))
                continue
            t = mu.timestamps / fsamp
            y = np.full(len(t), rank, dtype=float)
            r, g, b = palette[rank]
            scatter = pg.ScatterPlotItem(
                x=t,
                y=y,
                symbol="s",
                size=4,
                brush=pg.mkBrush(r, g, b, 200),
                pen=pg.mkPen(None),
            )
            pw.addItem(scatter)
            ticks.append((rank, f"MU {mu.id}"))

        pw.getAxis("left").setTicks([ticks])
        pw.getAxis("left").setWidth(70)
        pw.setYRange(-0.5, len(sorted_mus) - 0.5, padding=0.05)

        self._draw_aux_overlay(pw, y_min=-0.5, y_max=len(sorted_mus) - 0.5)

    @staticmethod
    def _decimate_for_display(
        t: np.ndarray, y: np.ndarray, max_pts: int
    ):
        """Return (t, y) downsampled to at most *max_pts* by peak-preserving
        block decimation.  Falls through unchanged when already small enough."""
        n = len(t)
        if n <= max_pts:
            return t, y
        step = max(1, n // max_pts)
        return t[::step], y[::step]

    def _render_idr(
        self,
        sorted_mus: List[tuple],
        idr: Optional[np.ndarray],
        t_axis: Optional[np.ndarray],
    ):
        pw = self._idr_plot
        pw.clear()

        if idr is None or t_axis is None or not sorted_mus:
            return

        palette = _blue_palette(len(sorted_mus))
        y_max = 0.0
        for rank, (port_name, mu) in enumerate(sorted_mus):
            dr_trace = idr[:, rank]
            peak = float(dr_trace.max())
            if peak > y_max:
                y_max = peak
            r, g, b = palette[rank]
            t_d, y_d = self._decimate_for_display(t_axis, dr_trace, self._MAX_DISPLAY_PTS)
            pw.plot(
                t_d,
                y_d,
                pen=pg.mkPen(color=(r, g, b, 220), width=1.5),
                name=f"MU {mu.id} ({port_name})",
            )

        self._draw_aux_overlay(pw, y_min=0.0, y_max=y_max)

    def _render_cst(
        self,
        idr: Optional[np.ndarray],
        t_axis: Optional[np.ndarray],
    ):
        pw = self._cst_plot
        pw.clear()

        if idr is None or t_axis is None:
            return

        cst = idr.sum(axis=1)
        y_max = float(cst.max())

        r_info, g_info, b_info = (
            int(COLORS["info"][1:3], 16),
            int(COLORS["info"][3:5], 16),
            int(COLORS["info"][5:7], 16),
        )
        t_d, y_d = self._decimate_for_display(t_axis, cst, self._MAX_DISPLAY_PTS)
        pw.plot(
            t_d,
            y_d,
            pen=pg.mkPen(color=(r_info, g_info, b_info), width=2),
        )

        self._draw_aux_overlay(pw, y_min=0.0, y_max=y_max)

    def _draw_aux_overlay(self, pw: pg.PlotWidget, y_min: float, y_max: float):
        y_range = max(y_max * 1.05 - y_min, 1e-9)
        for i, ch in enumerate(self._aux_channels):
            if i >= len(self._aux_on_states) or not self._aux_on_states[i]:
                continue
            raw = np.asarray(ch.get("data", [])).squeeze()
            if raw.size == 0:
                continue
            raw = np.nan_to_num(raw.astype(float), nan=0.0)
            sig = raw - float(raw.min())
            sig_range = max(float(sig.max()), 1e-9)
            sig_scaled = (sig / sig_range) * y_range + y_min

            n = len(sig_scaled)
            step = max(1, n // 4000)
            t_aux = np.arange(0, n, step) / self._fsamp

            r, g, b = _AUX_COLORS[i % len(_AUX_COLORS)]
            pw.plot(
                t_aux,
                sig_scaled[:n:step],
                pen=pg.mkPen(color=(r, g, b, 100), width=2),
            )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _toggle_mu(self, key: tuple):
        if key in self._disabled_mus:
            self._disabled_mus.discard(key)
        else:
            self._disabled_mus.add(key)
        self._render()

    def _toggle_port(self, port_name: str):
        mus = self._ports.get(port_name, [])
        all_disabled = bool(mus) and all(
            (port_name, mu.id) in self._disabled_mus for mu in mus
        )
        if all_disabled:
            for mu in mus:
                self._disabled_mus.discard((port_name, mu.id))
        else:
            for mu in mus:
                self._disabled_mus.add((port_name, mu.id))
        self._render()

    def _on_aux_toggled(self):
        for leg in self._aux_legends:
            leg.sync_labels()
        self._render()

    def _on_sidebar_all(self):
        self._disabled_mus.clear()
        self._render()

    def _on_sidebar_none(self):
        self._disabled_mus = {
            (port_name, mu.id) for port_name, mus in self._ports.items() for mu in mus
        }
        self._render()

    def _on_sort_changed(self):
        if self._ports:
            self._render()
