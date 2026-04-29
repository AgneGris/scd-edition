"""
Visualisation Tab (Tab 4) — interactive discharge-rate plots for the current file.

Inner tabs: Raster | IDR | CST
X-axes are linked across all three inner tabs.
AUX force channels can be overlaid on any plot and toggled per-channel.
"""

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
    QCheckBox,
    QComboBox,
    QDialog,
    QScrollArea,
    QDialogButtonBox,
    QFrame,
    QSizePolicy,
)
import pyqtgraph as pg

from motor_unit_toolbox.props import get_inst_discharge_rate

from scd_app.core.mu_model import MotorUnit
from scd_app.gui.style.styling import COLORS, FONT_SIZES

# ── Colour constants ──────────────────────────────────────────────────────────

_MU_PALETTE = [
    (74, 158, 255),
    (72, 187, 120),
    (246, 173, 85),
    (252, 129, 129),
    (167, 139, 250),
    (255, 107, 157),
    (129, 236, 236),
    (255, 204, 100),
    (102, 204, 153),
    (255, 159, 90),
]

_AUX_COLORS = [(255, 215, 0), (192, 239, 255), (255, 179, 71)]

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


# ── MU selection dialog ───────────────────────────────────────────────────────

class MUSelectionDialog(QDialog):
    """Lets the user toggle individual motor units on/off."""

    def __init__(
        self,
        ports: Dict[str, List[MotorUnit]],
        disabled_ids: Set[tuple],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Motor Units")
        self.setMinimumWidth(340)
        self.setStyleSheet(
            f"background-color: {COLORS['background_light']}; "
            f"color: {COLORS['foreground']};"
        )
        self._checks: Dict[tuple, QCheckBox] = {}

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        container = QWidget()
        inner = QVBoxLayout(container)
        inner.setSpacing(4)
        inner.setContentsMargins(4, 4, 4, 4)

        for port_name, mus in ports.items():
            hdr = QLabel(f"  {port_name}")
            hdr.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-weight: bold; "
                f"font-size: {FONT_SIZES['small']}; padding-top: 6px;"
            )
            inner.addWidget(hdr)
            for mu in mus:
                key = (port_name, mu.id)
                label = f"MU {mu.id}"
                if mu.props and not np.isnan(mu.props.discharge_rate_hz):
                    label += f"  — {mu.props.discharge_rate_hz:.1f} Hz"
                cb = QCheckBox(label)
                cb.setChecked(key not in disabled_ids)
                cb.setStyleSheet(f"color: {COLORS['foreground']}; padding-left: 12px;")
                self._checks[key] = cb
                inner.addWidget(cb)

        inner.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setStyleSheet(
            f"QPushButton {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['foreground']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 4px; padding: 4px 12px; }}"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_disabled_ids(self) -> Set[tuple]:
        return {key for key, cb in self._checks.items() if not cb.isChecked()}


# ── Main tab ──────────────────────────────────────────────────────────────────

class VisualisationTab(QWidget):
    """Tab 4 — raster, IDR, and CST plots for the current edition file."""

    def __init__(self, edition_tab=None, parent=None):
        super().__init__(parent)

        self._edition_tab = edition_tab

        # Cached data snapshot (filled on Refresh)
        self._ports: Dict[str, List[MotorUnit]] = {}
        self._aux_channels: list = []
        self._fsamp: float = 2048.0
        self._start_sample: int = 0
        self._end_sample: int = 0

        # UI state
        self._stale: bool = False
        self._disabled_mus: Set[tuple] = set()
        self._probe_enabled: Dict[str, bool] = {}
        self._probe_checks: Dict[str, QCheckBox] = {}
        self._aux_checks: List[QCheckBox] = []

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(self._build_header())

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

        root.addWidget(self._inner_tabs, stretch=1)

    def _make_plot_widget(self, ylabel: str, xlabel: str) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(COLORS["background"])
        for axis in ("left", "bottom"):
            pw.getAxis(axis).setTextPen(pg.mkPen(color=COLORS["foreground"]))
            pw.getAxis(axis).setPen(pg.mkPen(color=COLORS["border"]))
        pw.setLabel("left", ylabel, color=COLORS["text_secondary"],
                    size=FONT_SIZES["small"])
        pw.setLabel("bottom", xlabel, color=COLORS["text_secondary"],
                    size=FONT_SIZES["small"])
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

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedWidth(90)
        self._btn_refresh.clicked.connect(self._on_refresh_clicked)
        self._btn_refresh.setStyleSheet(self._refresh_btn_style(stale=False))
        layout.addWidget(self._btn_refresh)

        layout.addWidget(self._make_sep())

        probe_lbl = QLabel("Probes:")
        probe_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: {FONT_SIZES['small']};"
        )
        layout.addWidget(probe_lbl)

        self._probe_container = QWidget()
        self._probe_container_layout = QHBoxLayout(self._probe_container)
        self._probe_container_layout.setContentsMargins(0, 0, 0, 0)
        self._probe_container_layout.setSpacing(6)
        layout.addWidget(self._probe_container)

        self._btn_select_mus = QPushButton("Select MUs…")
        self._btn_select_mus.setStyleSheet(self._small_btn_style())
        self._btn_select_mus.clicked.connect(self._open_mu_selection)
        layout.addWidget(self._btn_select_mus)

        layout.addWidget(self._make_sep())

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
        layout.addWidget(self._sort_combo)

        layout.addWidget(self._make_sep())

        self._aux_label = QLabel("AUX:")
        self._aux_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: {FONT_SIZES['small']};"
        )
        layout.addWidget(self._aux_label)

        self._aux_container = QWidget()
        self._aux_container_layout = QHBoxLayout(self._aux_container)
        self._aux_container_layout.setContentsMargins(0, 0, 0, 0)
        self._aux_container_layout.setSpacing(6)
        layout.addWidget(self._aux_container)

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
    def _refresh_btn_style(stale: bool) -> str:
        bg = "#7c4b00" if stale else COLORS["background_input"]
        border = "#f59e0b" if stale else COLORS["border"]
        return (
            f"QPushButton {{ background: {bg}; color: {COLORS['foreground']}; "
            f"border: 1px solid {border}; border-radius: 4px; padding: 4px 8px; "
            f"font-size: {FONT_SIZES['small']}; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; }}"
        )

    @staticmethod
    def _small_btn_style() -> str:
        return (
            f"QPushButton {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['foreground']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 4px; padding: 4px 8px; "
            f"font-size: {FONT_SIZES['small']}; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; }}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_data_modified(self):
        """Mark tab as stale — called when edition data changes."""
        self._mark_stale()

    # ── Header rebuild ────────────────────────────────────────────────────────

    def _rebuild_header_controls(self):
        # --- probe checkboxes ---
        while self._probe_container_layout.count():
            item = self._probe_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._probe_checks.clear()

        for port_name in self._ports:
            cb = QCheckBox(port_name)
            cb.setChecked(self._probe_enabled.get(port_name, True))
            cb.setStyleSheet(
                f"color: {COLORS['foreground']}; font-size: {FONT_SIZES['small']};"
            )
            self._probe_container_layout.addWidget(cb)
            self._probe_checks[port_name] = cb

        # --- AUX checkboxes ---
        while self._aux_container_layout.count():
            item = self._aux_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._aux_checks.clear()

        for i, ch in enumerate(self._aux_channels):
            name = ch.get("name", ch.get("unit", f"AUX {i + 1}"))
            cb = QCheckBox(name)
            cb.setChecked(True)
            r, g, b = _AUX_COLORS[i % len(_AUX_COLORS)]
            cb.setStyleSheet(
                f"color: rgb({r},{g},{b}); font-size: {FONT_SIZES['small']};"
            )
            self._aux_container_layout.addWidget(cb)
            self._aux_checks.append(cb)

        has_aux = bool(self._aux_channels)
        self._aux_label.setVisible(has_aux)
        self._aux_container.setVisible(has_aux)

    # ── Stale tracking ────────────────────────────────────────────────────────

    def _mark_stale(self):
        self._stale = True
        self._btn_refresh.setStyleSheet(self._refresh_btn_style(stale=True))
        self._btn_refresh.setText("Refresh ●")

    def _mark_fresh(self):
        self._stale = False
        self._btn_refresh.setStyleSheet(self._refresh_btn_style(stale=False))
        self._btn_refresh.setText("Refresh")

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
        # preserve prior probe enabled state
        self._probe_enabled = {
            p: self._probe_checks.get(p, QCheckBox()).isChecked()
            if p in self._probe_checks else True
            for p in self._ports
        }
        self._rebuild_header_controls()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _on_refresh_clicked(self):
        self._fetch_edition_data()
        self._render()

    def _get_active_mus(self) -> List[tuple]:
        result = []
        for port_name, mus in self._ports.items():
            cb = self._probe_checks.get(port_name)
            if cb is not None and not cb.isChecked():
                continue
            for mu in mus:
                if (port_name, mu.id) not in self._disabled_mus:
                    result.append((port_name, mu))
        return result

    def _sorted_mus(self, active: List[tuple]) -> List[tuple]:
        sort_name = self._sort_combo.currentText()
        key_fn = _SORT_FNS.get(sort_name, _sort_key_recruit)
        return sorted(active, key=lambda pair: key_fn(pair[1], self._fsamp))

    def _render(self):
        active = self._get_active_mus()
        sorted_active = self._sorted_mus(active)
        self._render_raster(sorted_active)
        spike_matrix, t_axis = self._build_idr_matrix(sorted_active)
        self._render_idr(sorted_active, spike_matrix, t_axis)
        self._render_cst(spike_matrix, t_axis)
        self._mark_fresh()

    def _build_idr_matrix(
        self, sorted_mus: List[tuple]
    ):
        """Build the (n_samples, n_units) binary spike matrix and time axis.

        Uses the union timestamp range across all active MUs so the IDR array
        aligns with the raster x-axis.
        """
        if not sorted_mus:
            return None, None

        all_ts = [
            mu.timestamps for _, mu in sorted_mus if len(mu.timestamps) > 0
        ]
        if not all_ts:
            return None, None

        ts_global_min = int(min(ts.min() for ts in all_ts))
        ts_global_max = int(max(ts.max() for ts in all_ts))
        n_samples = ts_global_max - ts_global_min + 1

        spike_matrix = np.zeros((n_samples, len(sorted_mus)), dtype=bool)
        for col, (_, mu) in enumerate(sorted_mus):
            if len(mu.timestamps) == 0:
                continue
            ts_rel = mu.timestamps.astype(np.int64) - ts_global_min
            valid = ts_rel[(ts_rel >= 0) & (ts_rel < n_samples)]
            spike_matrix[valid, col] = True

        t_axis = np.arange(n_samples) / self._fsamp + ts_global_min / self._fsamp
        return spike_matrix, t_axis

    def _render_raster(self, sorted_mus: List[tuple]):
        pw = self._raster_plot
        pw.clear()
        pw.getAxis("left").setTicks([[]])

        if not sorted_mus:
            return

        fsamp = self._fsamp
        ticks = []
        for rank, (port_name, mu) in enumerate(sorted_mus):
            if len(mu.timestamps) == 0:
                ticks.append((rank, f"MU {mu.id}"))
                continue
            t = mu.timestamps / fsamp
            y = np.full(len(t), rank, dtype=float)
            r, g, b = _MU_PALETTE[rank % len(_MU_PALETTE)]
            scatter = pg.ScatterPlotItem(
                x=t, y=y,
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

    def _render_idr(
        self,
        sorted_mus: List[tuple],
        spike_matrix: Optional[np.ndarray],
        t_axis: Optional[np.ndarray],
    ):
        pw = self._idr_plot
        pw.clear()

        if spike_matrix is None or t_axis is None or not sorted_mus:
            return

        idr = get_inst_discharge_rate(spike_matrix, int(self._fsamp))

        y_max = 0.0
        for rank, (port_name, mu) in enumerate(sorted_mus):
            dr_trace = idr[:, rank]
            peak = float(dr_trace.max())
            if peak > y_max:
                y_max = peak
            r, g, b = _MU_PALETTE[rank % len(_MU_PALETTE)]
            label = f"MU {mu.id} ({port_name})"
            pw.plot(
                t_axis, dr_trace,
                pen=pg.mkPen(color=(r, g, b, 220), width=1.5),
                name=label,
            )

        self._draw_aux_overlay(pw, y_min=0.0, y_max=y_max)

    def _render_cst(
        self,
        spike_matrix: Optional[np.ndarray],
        t_axis: Optional[np.ndarray],
    ):
        pw = self._cst_plot
        pw.clear()

        if spike_matrix is None or t_axis is None:
            return

        idr = get_inst_discharge_rate(spike_matrix, int(self._fsamp))
        cst = idr.sum(axis=1)
        y_max = float(cst.max())

        r_info, g_info, b_info = (
            int(COLORS["info"][1:3], 16),
            int(COLORS["info"][3:5], 16),
            int(COLORS["info"][5:7], 16),
        )
        pw.plot(
            t_axis, cst,
            pen=pg.mkPen(color=(r_info, g_info, b_info), width=2),
        )

        self._draw_aux_overlay(pw, y_min=0.0, y_max=y_max)

    def _draw_aux_overlay(
        self, pw: pg.PlotWidget, y_min: float, y_max: float
    ):
        y_range = max(y_max * 1.05 - y_min, 1e-9)
        for i, ch in enumerate(self._aux_channels):
            if i >= len(self._aux_checks) or not self._aux_checks[i].isChecked():
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

    def _open_mu_selection(self):
        if not self._ports:
            return
        dlg = MUSelectionDialog(self._ports, self._disabled_mus, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._disabled_mus = dlg.get_disabled_ids()
            self._mark_stale()
