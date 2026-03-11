"""
Edition Tab - EMG spike editing interface.

On load, filters are applied via peel-off replay on the full preprocessed EMG,
and timestamps are re-detected via source_to_timestamps.  The plateau region
is shown as a shaded band on the source plot.
"""

from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pickle
import traceback

import numpy as np
from scipy import signal as sp_signal

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QToolBar,
    QAction, QActionGroup, QComboBox, QLabel, QPushButton,
    QFileDialog, QMessageBox, QShortcut, QStatusBar, QFrame,
)
from PyQt5.QtGui import QKeySequence, QFont
import pyqtgraph as pg

from scd_app.gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, get_button_style,
)
from scd_app.core.mu_model import EditMode, MotorUnit, UndoAction
from scd_app.core.mu_properties import (
    MUProperties,
    compute_port_properties,
    recompute_unit_properties,
    flat_channels_to_grid,
)
from scd_app.gui.widgets.mu_properties_panel import MUPropertiesPanel
from scd_app.core.filter_recalculation import (
    recalculate_unit_filter,
    supports_filter_recalculation,
    supports_full_source_computation,
    compute_all_full_sources,
)


GRID_POSITIONS_13x5 = {
    1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0), 6: (6, 0),
    7: (7, 0), 8: (8, 0), 9: (9, 0), 10: (10, 0), 11: (11, 0), 12: (12, 0),
    25: (0, 1), 24: (1, 1), 23: (2, 1), 22: (3, 1), 21: (4, 1), 20: (5, 1),
    19: (6, 1), 18: (7, 1), 17: (8, 1), 16: (9, 1), 15: (10, 1), 14: (11, 1), 13: (12, 1),
    26: (0, 2), 27: (1, 2), 28: (2, 2), 29: (3, 2), 30: (4, 2), 31: (5, 2),
    32: (6, 2), 33: (7, 2), 34: (8, 2), 35: (9, 2), 36: (10, 2), 37: (11, 2), 38: (12, 2),
    51: (0, 3), 50: (1, 3), 49: (2, 3), 48: (3, 3), 47: (4, 3), 46: (5, 3),
    45: (6, 3), 44: (7, 3), 43: (8, 3), 42: (9, 3), 41: (10, 3), 40: (11, 3), 39: (12, 3),
    52: (0, 4), 53: (1, 4), 54: (2, 4), 55: (3, 4), 56: (4, 4), 57: (5, 4),
    58: (6, 4), 59: (7, 4), 60: (8, 4), 61: (9, 4), 62: (10, 4), 63: (11, 4), 64: (12, 4),
}
GRID_POSITIONS_8x8 = {
    8: (0, 0), 7: (1, 0), 6: (2, 0), 5: (3, 0), 4: (4, 0), 3: (5, 0), 2: (6, 0), 1: (7, 0),
    16: (0, 1), 15: (1, 1), 14: (2, 1), 13: (3, 1), 12: (4, 1), 11: (5, 1), 10: (6, 1), 9: (7, 1),
    24: (0, 2), 23: (1, 2), 22: (2, 2), 21: (3, 2), 20: (4, 2), 19: (5, 2), 18: (6, 2), 17: (7, 2),
    32: (0, 3), 31: (1, 3), 30: (2, 3), 29: (3, 3), 28: (4, 3), 27: (5, 3), 26: (6, 3), 25: (7, 3),
    40: (0, 4), 39: (1, 4), 38: (2, 4), 37: (3, 4), 36: (4, 4), 35: (5, 4), 34: (6, 4), 33: (7, 4),
    48: (0, 5), 47: (1, 5), 46: (2, 5), 45: (3, 5), 44: (4, 5), 43: (5, 5), 42: (6, 5), 41: (7, 5),
    56: (0, 6), 55: (1, 6), 54: (2, 6), 53: (3, 6), 52: (4, 6), 51: (5, 6), 50: (6, 6), 49: (7, 6),
    64: (0, 7), 63: (1, 7), 62: (2, 7), 61: (3, 7), 60: (4, 7), 59: (5, 7), 58: (6, 7), 57: (7, 7),
}
GRID_POSITIONS_20x2 = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0),
    5: (5, 0), 6: (6, 0), 7: (7, 0), 8: (8, 0), 9: (9, 0),
    10: (10, 0), 11: (11, 0), 12: (12, 0), 13: (13, 0), 14: (14, 0),
    15: (15, 0), 16: (16, 0), 17: (17, 0), 18: (18, 0), 19: (19, 0),
    20: (0, 1), 21: (1, 1), 22: (2, 1), 23: (3, 1), 24: (4, 1),
    25: (5, 1), 26: (6, 1), 27: (7, 1), 28: (8, 1), 29: (9, 1),
    30: (10, 1), 31: (11, 1), 32: (12, 1), 33: (13, 1), 34: (14, 1),
    35: (15, 1), 36: (16, 1), 37: (17, 1), 38: (18, 1), 39: (19, 1),
}
ELECTRODE_GRIDS = {
    "GR08MM1305": {"grid_shape": (13, 5), "ied_mm": 8, "n_channels": 64,
                    "muap_mapping": {i: i+1 for i in range(64)}, "positions": GRID_POSITIONS_13x5},
    "GR10MM0808": {"grid_shape": (8, 8), "ied_mm": 10, "n_channels": 64,
                    "muap_mapping": {i: i+1 for i in range(64)}, "positions": GRID_POSITIONS_8x8},
    "Thin-film":  {"grid_shape": (20, 2), "ied_mm": 5, "n_channels": 40,
                    "muap_mapping": {i: i for i in range(40)}, "positions": GRID_POSITIONS_20x2},
}


def to_numpy(obj) -> np.ndarray:
    if obj is None:
        return np.array([])
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def get_grid_config(electrode_type: Optional[str]) -> Optional[Dict]:
    if electrode_type is None:
        return None
    key = electrode_type.upper()
    for name, cfg in ELECTRODE_GRIDS.items():
        if name.upper() in key:
            return cfg
    return None


class SourcePlotWidget(pg.PlotWidget):
    spike_add_requested    = pyqtSignal(int)
    spike_delete_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS["background"])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left",   "Amplitude", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))

        self._fsamp      = 1.0
        self._edit_mode  = EditMode.VIEW
        self._source: Optional[np.ndarray]     = None
        self._timestamps: Optional[np.ndarray] = None

        self._signal_curve  = self.plot([], pen=pg.mkPen("#2b6cb0", width=1))
        self._spike_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush("#ed8936"),
            symbol="o", hoverable=True,
        )
        self.addItem(self._spike_scatter)
        self._plateau_region: Optional[pg.LinearRegionItem] = None
        self._roi: Optional[pg.ROI] = None
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def set_edit_mode(self, mode: EditMode):
        self._edit_mode = mode
        cursors = {EditMode.VIEW: Qt.ArrowCursor, EditMode.ADD: Qt.CrossCursor,
                   EditMode.DELETE: Qt.PointingHandCursor}
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))

    def set_data(self, source: np.ndarray, timestamps: np.ndarray):
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
        self._source     = source ** 2
        self._timestamps = timestamps
        t = np.arange(len(self._source)) / self._fsamp
        self._signal_curve.setData(t, self._source)
        self._update_spike_markers()

    def set_plateau_region(self, start_sample: int, end_sample: int):
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
        t0 = start_sample / self._fsamp
        t1 = end_sample / self._fsamp
        self._plateau_region = pg.LinearRegionItem(
            values=(t0, t1), movable=False,
            brush=pg.mkBrush(137, 180, 250, 20),
            pen=pg.mkPen(color=(137, 180, 250, 60), width=1, style=Qt.DashLine),
        )
        self._plateau_region.setZValue(-10)
        self.addItem(self._plateau_region)

    def _update_spike_markers(self):
        if self._source is None or self._timestamps is None or len(self._timestamps) == 0:
            self._spike_scatter.setData([], [])
            return
        valid = self._timestamps[self._timestamps < len(self._source)]
        if len(valid) == 0:
            self._spike_scatter.setData([], [])
            return
        self._spike_scatter.setData(valid / self._fsamp, self._source[valid])

    def clear_data(self):
        self._signal_curve.setData([], [])
        self._spike_scatter.setData([], [])
        self._source = None
        self._timestamps = None
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
            self._plateau_region = None

    def place_roi(self):
        self.remove_roi()
        vb = self.getViewBox(); vr = vb.viewRange()
        x_range = vr[0][1] - vr[0][0]; y_range = vr[1][1] - vr[1][0]
        w, h = x_range * 0.3, y_range * 0.6
        cx = (vr[0][0] + vr[0][1]) / 2; cy = (vr[1][0] + vr[1][1]) / 2
        self._roi = pg.ROI(
            pos=[cx - w/2, cy - h/2], size=[w, h],
            pen=pg.mkPen(color=COLORS.get("warning", "#f9e2af"), width=2),
            movable=True, resizable=True,
        )
        for a, o in [([1,1],[0,0]),([0,0],[1,1]),([1,0],[0,1]),([0,1],[1,0])]:
            self._roi.addScaleHandle(a, o)
        self.addItem(self._roi)

    def remove_roi(self):
        if self._roi is not None:
            self.removeItem(self._roi); self._roi = None

    def get_roi_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self._roi is None: return None
        p, s = self._roi.pos(), self._roi.size()
        x1, y1, x2, y2 = p.x(), p.y(), p.x()+s.x(), p.y()+s.y()
        return (min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2))

    def has_roi(self) -> bool:
        return self._roi is not None

    def _on_mouse_clicked(self, ev):
        if self._edit_mode == EditMode.VIEW or ev.button() != Qt.LeftButton:
            return
        pos    = self.plotItem.vb.mapSceneToView(ev.scenePos())
        sample = int(pos.x() * self._fsamp)
        if self._edit_mode == EditMode.ADD:
            self.spike_add_requested.emit(sample)
        elif self._edit_mode == EditMode.DELETE:
            self.spike_delete_requested.emit(sample)


class FiringRatePlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS["background"])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left",   "IFR (Hz)", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))
        self._curve = self.plot([], pen=pg.mkPen(COLORS["warning"], width=1.5))
        self._fsamp = 1.0

    def set_fsamp(self, fsamp: float): self._fsamp = fsamp
    def link_x(self, other: pg.PlotWidget): self.setXLink(other)

    def set_data(self, timestamps: np.ndarray):
        ts = np.sort(timestamps)
        if len(ts) < 2:
            self._curve.setData([], []); return
        isi = np.diff(ts) / self._fsamp
        ifr = np.where(isi > 0.01, 1.0 / isi, 0.0)
        t_mid = (ts[:-1] + ts[1:]) / 2 / self._fsamp
        self._curve.setData(t_mid, ifr)

    def clear_data(self):
        self._curve.setData([], [])

class EditionTab(QWidget):
    data_modified = pyqtSignal()

    def __init__(self, fsamp: float = 2048.0, parent=None):
        super().__init__(parent)

        self._fsamp        = fsamp
        self._ports:       Dict[str, List[MotorUnit]] = {}
        self._emg_data:    Dict[str, np.ndarray]      = {}   # port → plateau-sliced
        self._grid_info:   Dict[str, Optional[Dict]]  = {}
        self._raw_port_channels: Dict[str, np.ndarray] = {}  # port → (active_ch, full_samples)
        self._current_port: Optional[str] = None
        self._current_mu_idx: int         = -1
        self._edit_mode   = EditMode.VIEW
        self._loaded_path: Optional[Path] = None

        self._start_sample: int = 0
        self._end_sample:   int = 0
        self._full_source_mode: bool = False

        self._undo_stack: List[UndoAction] = []
        self._redo_stack: List[UndoAction] = []
        self._original_decomp_data: Optional[dict] = None
        self._filter_recalc_available: bool = False

        self._build_ui()
        self._setup_shortcuts()

    def _ts_to_plateau_local(self, timestamps: np.ndarray) -> np.ndarray:
        return timestamps - self._start_sample

    def _ts_to_absolute(self, plateau_local: np.ndarray) -> np.ndarray:
        return plateau_local + self._start_sample

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_toolbar())
        self.quality_bar = MUPropertiesPanel(); self.quality_bar.setMinimumHeight(110)
        root.addWidget(self.quality_bar)
        splitter = QSplitter(Qt.Horizontal); splitter.setHandleWidth(2)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 1080]); splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 7)
        root.addWidget(splitter, stretch=1)
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"color: {COLORS.get('text_dim', '#6c7086')}; font-size: {FONT_SIZES.get('small', '9pt')};")
        root.addWidget(self.status_bar)
        self._update_status()

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar(); tb.setMovable(False)
        tb.setStyleSheet(f"""
            QToolBar {{ background-color: {COLORS.get('background_light','#2a2a3c')};
                        border-bottom: 1px solid {COLORS['border']}; spacing: 4px; padding: 2px; }}
            QToolBar QLabel {{ color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small','9pt')}; }}
            QToolButton {{ color: {COLORS['foreground']}; background: transparent;
                           border: 1px solid transparent; border-radius: 4px;
                           padding: 4px 8px; font-size: {FONT_SIZES.get('small','9pt')}; }}
            QToolButton:hover {{ background-color: {COLORS.get('background_input','#33334d')};
                                 border-color: {COLORS['border']}; }}
            QToolButton:checked {{ background-color: {COLORS.get('info','#89b4fa')}30;
                                   border-color: {COLORS.get('info','#89b4fa')}; }}
        """)
        self.action_load = QAction("📂 Load", self); self.action_load.triggered.connect(self._load_file_dialog); tb.addAction(self.action_load)
        self.action_save = QAction("💾 Save", self); self.action_save.setShortcut(QKeySequence.Save); self.action_save.triggered.connect(self._save_file); tb.addAction(self.action_save)
        self.action_reset = QAction("⟲ Reset View", self); self.action_reset.setShortcut(QKeySequence("Home")); self.action_reset.triggered.connect(self._reset_view); tb.addAction(self.action_reset)
        tb.addSeparator(); tb.addWidget(QLabel("  Mode: ")); self.mode_group = QActionGroup(self)
        self.action_view = QAction("👁 View", self); self.action_view.setCheckable(True); self.action_view.setChecked(True); self.action_view.setShortcut(QKeySequence("V")); self.action_view.triggered.connect(lambda: self._set_mode(EditMode.VIEW)); self.mode_group.addAction(self.action_view); tb.addAction(self.action_view)
        self.action_add = QAction("➕ Add", self); self.action_add.setCheckable(True); self.action_add.setShortcut(QKeySequence("A")); self.action_add.triggered.connect(lambda: self._set_mode(EditMode.ADD)); self.mode_group.addAction(self.action_add); tb.addAction(self.action_add)
        self.action_delete = QAction("➖ Delete", self); self.action_delete.setCheckable(True); self.action_delete.setShortcut(QKeySequence("D")); self.action_delete.triggered.connect(lambda: self._set_mode(EditMode.DELETE)); self.mode_group.addAction(self.action_delete); tb.addAction(self.action_delete)
        tb.addSeparator(); tb.addWidget(QLabel("  ROI: "))
        self.action_place_roi = QAction("⬜ Place", self); self.action_place_roi.setShortcut(QKeySequence("R")); self.action_place_roi.triggered.connect(self._place_roi); tb.addAction(self.action_place_roi)
        self.action_roi_add = QAction("✅ Add in ROI", self); self.action_roi_add.setShortcut(QKeySequence("Shift+A")); self.action_roi_add.triggered.connect(self._roi_add_spikes); self.action_roi_add.setEnabled(False); tb.addAction(self.action_roi_add)
        self.action_roi_delete = QAction("🗑 Del in ROI", self); self.action_roi_delete.setShortcut(QKeySequence("Shift+D")); self.action_roi_delete.triggered.connect(self._roi_delete_spikes); self.action_roi_delete.setEnabled(False); tb.addAction(self.action_roi_delete)
        self.action_roi_clear = QAction("✕ Clear ROI", self); self.action_roi_clear.setShortcut(QKeySequence(Qt.Key_Escape)); self.action_roi_clear.triggered.connect(self._clear_roi); self.action_roi_clear.setEnabled(False); tb.addAction(self.action_roi_clear)
        tb.addSeparator()
        self.action_undo = QAction("↩ Undo", self); self.action_undo.setShortcut(QKeySequence.Undo); self.action_undo.triggered.connect(self._undo); tb.addAction(self.action_undo)
        self.action_redo = QAction("↪ Redo", self); self.action_redo.setShortcut(QKeySequence.Redo); self.action_redo.triggered.connect(self._redo); tb.addAction(self.action_redo)
        return tb

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['background']}; border-right: 2px solid {COLORS['border']};")
        lay = QVBoxLayout(panel); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        lay.addWidget(QLabel("PORT SELECTION", styleSheet=get_section_header_style("info")))
        port_row = QHBoxLayout()
        port_lbl = QLabel("Port:"); port_lbl.setStyleSheet(f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small','9pt')};")
        port_row.addWidget(port_lbl)
        self.port_combo = QComboBox(); self.port_combo.setStyleSheet(self._combo_style()); self.port_combo.currentTextChanged.connect(self._on_port_changed)
        port_row.addWidget(self.port_combo, stretch=1); lay.addLayout(port_row)
        lay.addWidget(QLabel("MOTOR UNIT", styleSheet=get_section_header_style("info")))
        mu_row = QHBoxLayout()
        mu_lbl = QLabel("Unit:"); mu_lbl.setStyleSheet(f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small','9pt')};")
        mu_row.addWidget(mu_lbl)
        self.mu_combo = QComboBox(); self.mu_combo.setStyleSheet(self._combo_style()); self.mu_combo.currentIndexChanged.connect(self._on_mu_selected)
        mu_row.addWidget(self.mu_combo, stretch=1); lay.addLayout(mu_row)
        btn_style = f"""
            QPushButton {{ background-color: {COLORS.get('background_input','#33334d')}; color: {COLORS['foreground']};
                           border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 6px 12px;
                           font-size: {FONT_SIZES.get('small','9pt')}; }}
            QPushButton:hover {{ background-color: {COLORS.get('background_light','#2a2a3c')}; border-color: {COLORS.get('info','#89b4fa')}; }}
            QPushButton:disabled {{ color: {COLORS.get('text_dim','#6c7086')}; border-color: {COLORS['border']}; }}"""
        self.btn_flag_delete = QPushButton("🗑 Flag to Delete"); self.btn_flag_delete.setStyleSheet(btn_style)
        self.btn_flag_delete.setShortcut(QKeySequence("X")); self.btn_flag_delete.clicked.connect(self._toggle_flag_delete); lay.addWidget(self.btn_flag_delete)
        self.btn_recalc_filter = QPushButton("⟳ Recalculate Filter"); self.btn_recalc_filter.setStyleSheet(btn_style)
        self.btn_recalc_filter.setShortcut(QKeySequence("F"))
        self.btn_recalc_filter.setToolTip("Replay peel-off and recompute filter + source + timestamps [F]")
        self.btn_recalc_filter.clicked.connect(self._recalculate_filter); self.btn_recalc_filter.setEnabled(False); lay.addWidget(self.btn_recalc_filter)
        self.muap_widget = pg.GraphicsLayoutWidget(); self.muap_widget.setBackground(COLORS["background"]); self.muap_widget.setMinimumHeight(50)
        lay.addWidget(self.muap_widget, stretch=1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(); panel.setStyleSheet(f"background-color: {COLORS['background']};")
        lay = QVBoxLayout(panel); lay.setContentsMargins(0,0,0,0)
        plot_splitter = QSplitter(Qt.Vertical); plot_splitter.setHandleWidth(2)
        self.source_plot = SourcePlotWidget(); self.source_plot.set_fsamp(self._fsamp)
        self.source_plot.spike_add_requested.connect(self._handle_add_click)
        self.source_plot.spike_delete_requested.connect(self._handle_delete_click)
        plot_splitter.addWidget(self.source_plot)
        self.fr_plot = FiringRatePlotWidget(); self.fr_plot.set_fsamp(self._fsamp)
        self.fr_plot.link_x(self.source_plot); self.fr_plot.setMaximumHeight(150)
        plot_splitter.addWidget(self.fr_plot); plot_splitter.setSizes([500, 150])
        lay.addWidget(plot_splitter)
        return panel

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Up"),   self, self._select_prev_mu)
        QShortcut(QKeySequence("Down"), self, self._select_next_mu)

    @staticmethod
    def _combo_style() -> str:
        return f"""QComboBox {{ background-color: {COLORS.get('background_input','#33334d')};
                    color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};
                    border-radius: 4px; padding: 4px 8px; }}"""

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp; self.source_plot.set_fsamp(fsamp); self.fr_plot.set_fsamp(fsamp)

    def load_from_path(self, path: Path):
        path = Path(path)
        if not path.exists():
            QMessageBox.critical(self, "Load Error", f"File not found:\n{path}"); return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{e}"); return
        if "ports" not in data or "discharge_times" not in data:
            QMessageBox.warning(self, "Format Error",
                "File does not contain 'ports' and 'discharge_times'."); return
        try:
            self._load_decomposition_data(data)
            self._loaded_path = path
            self._update_status(f"Loaded: {path.name}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to parse:\n{e}")

    def _load_decomposition_data(self, decomp_data: dict):
        self._ports.clear(); self._emg_data.clear(); self._grid_info.clear()
        self._raw_port_channels.clear()
        self._undo_stack.clear(); self._redo_stack.clear()
        self._original_decomp_data = decomp_data

        # Sampling rate
        fsamp = decomp_data.get("sampling_rate", decomp_data.get("fsamp", self._fsamp))
        self.set_fsamp(float(fsamp))

        # Full EMG
        raw_data = decomp_data.get("data")
        emg_full = None
        if raw_data is not None:
            emg_full = to_numpy(raw_data)
            if emg_full.ndim == 2 and emg_full.shape[0] > emg_full.shape[1]:
                emg_full = emg_full.T

        # ── Try full-source pipeline ──────────────────────────────────────
        can_full, reason = supports_full_source_computation(decomp_data)
        full_port_results: Dict[int, list] = {}
        start_sample, end_sample = 0, (emg_full.shape[1] if emg_full is not None else 0)

        if can_full:
            try:
                self._update_status("Computing full-length sources (peel-off replay)…")
                full_port_results, start_sample, end_sample, err = \
                    compute_all_full_sources(decomp_data)
                if err:
                    print(f"  [edition] Full source warning: {err}")
                    full_port_results = {}
            except Exception as e:
                print(f"  [edition] Full source failed: {e}")
                traceback.print_exc()
                full_port_results = {}
        else:
            # Parse plateau bounds for fallback
            sel_pts = decomp_data.get("plateau_coords", decomp_data.get("selected_points"))
            if sel_pts is not None:
                try:
                    pts = to_numpy(np.asarray(sel_pts)).flatten()
                    start_sample, end_sample = int(pts[0]), int(pts[1])
                except (IndexError, TypeError, ValueError):
                    pass
            if end_sample <= start_sample and emg_full is not None:
                end_sample = emg_full.shape[1]

        self._start_sample = start_sample
        self._end_sample   = end_sample
        self._full_source_mode = bool(full_port_results)

        if self._full_source_mode:
            print("  [edition] ✓ Full-length source mode (peel-off + source_to_timestamps)")
        else:
            print(f"  [edition] Fallback: plateau-only sources"
                  f"{(' — ' + reason) if not can_full else ''}")

        # ── Per-port unpacking ────────────────────────────────────────────
        ports               = decomp_data.get("ports", [])
        chans_per_electrode = decomp_data.get("chans_per_electrode", [])
        channel_indices_all = decomp_data.get("channel_indices")   # list[list[int]] or None
        mask_list           = decomp_data.get("emg_mask", [])
        electrode_list      = decomp_data.get("electrodes", [])
        ch_offset           = 0

        for port_idx, port_name in enumerate(ports):
            n_ch = int(chans_per_electrode[port_idx]) if port_idx < len(chans_per_electrode) else 64

            # Absolute channel indices for this port: use saved indices when available,
            # fall back to sequential ch_offset for old files.
            if (channel_indices_all is not None
                    and port_idx < len(channel_indices_all)
                    and channel_indices_all[port_idx] is not None):
                port_ch_idx = np.asarray(channel_indices_all[port_idx], dtype=int)
            else:
                port_ch_idx = np.arange(ch_offset, ch_offset + n_ch, dtype=int)

            # Active channels (local indices within this port's n_ch channels)
            if port_idx < len(mask_list) and mask_list[port_idx] is not None:
                local_active = np.where(to_numpy(np.asarray(mask_list[port_idx])).flatten() == 0)[0]
            else:
                local_active = np.arange(n_ch)
            global_active = port_ch_idx[local_active[local_active < len(port_ch_idx)]]

            # Plateau-sliced EMG for MUAP computation
            emg_port = None
            if emg_full is not None:
                valid_chs = global_active[global_active < emg_full.shape[0]]
                if len(valid_chs) > 0:
                    emg_port = emg_full[valid_chs,
                               max(0, start_sample):min(end_sample, emg_full.shape[1])]
                    valid_port_chs = port_ch_idx[port_ch_idx < emg_full.shape[0]]
                    self._raw_port_channels[port_name] = emg_full[valid_port_chs, :]

            # Discharge times & sources from file (fallback data)
            port_discharge = (decomp_data["discharge_times"][port_idx]
                              if port_idx < len(decomp_data["discharge_times"]) else [])
            port_sources   = (decomp_data["pulse_trains"][port_idx]
                              if port_idx < len(decomp_data["pulse_trains"]) else [])
            port_filters_raw = decomp_data.get("mu_filters", [])
            port_filters     = (port_filters_raw[port_idx]
                                if port_idx < len(port_filters_raw) else None)

            ts_list   = self._ensure_list_of_arrays(port_discharge)
            src_list  = self._ensure_list_of_arrays(port_sources)
            filt_list = (self._ensure_list_of_arrays(port_filters)
                         if port_filters is not None else [None] * len(ts_list))

            # Full-source results for this port
            full_results = full_port_results.get(port_idx, [])

            # Build MotorUnit objects
            motor_units = []
            for mu_idx in range(len(ts_list)):
                filt = (to_numpy(filt_list[mu_idx])
                        if mu_idx < len(filt_list) and filt_list[mu_idx] is not None
                        else None)

                if (self._full_source_mode and mu_idx < len(full_results)
                        and full_results[mu_idx][0] is not None):
                    # ── Full-source mode: use recomputed source + timestamps ─
                    entry  = full_results[mu_idx]
                    source = entry[0]      # full-length
                    ts_abs = entry[1]      # absolute
                    if len(entry) > 2 and entry[2] is not None:
                        filt = entry[2]   # STA-recalculated filter
                else:
                    # ── Fallback: plateau-only ────────────────────────────
                    source = (to_numpy(src_list[mu_idx]).flatten()
                              if mu_idx < len(src_list) else np.zeros(1))
                    ts_plateau = to_numpy(ts_list[mu_idx]).flatten().astype(np.int64)
                    ts_abs = self._ts_to_absolute(ts_plateau) if self._full_source_mode else ts_plateau

                motor_units.append(MotorUnit(
                    id=mu_idx, timestamps=ts_abs, source=source,
                    port_name=port_name, mu_filter=filt,
                ))

            # Grid config
            etype    = electrode_list[port_idx] if port_idx < len(electrode_list) else None
            grid_cfg = get_grid_config(etype)

            # MU properties (use plateau-local timestamps + plateau-sliced EMG)
            if motor_units:
                if self._full_source_mode:
                    props_ts = [self._ts_to_plateau_local(mu.timestamps) for mu in motor_units]
                    props_src = [
                        mu.source[start_sample:end_sample]
                        if len(mu.source) > (end_sample - start_sample) else mu.source
                        for mu in motor_units
                    ]
                else:
                    props_ts  = [mu.timestamps for mu in motor_units]
                    props_src = [mu.source for mu in motor_units]

                props_list = compute_port_properties(
                    all_timestamps=props_ts, all_sources=props_src,
                    emg_port=emg_port,
                    grid_positions=grid_cfg["positions"] if grid_cfg else None,
                    grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
                    fsamp=self._fsamp,
                )
                for mu, p in zip(motor_units, props_list):
                    mu.props = p

            self._ports[port_name]     = motor_units
            self._grid_info[port_name] = grid_cfg
            if emg_port is not None:
                self._emg_data[port_name] = emg_port

            ch_offset += n_ch
            print(f"  Port '{port_name}': {len(motor_units)} MUs, "
                  f"{sum(len(m.timestamps) for m in motor_units)} spikes"
                  f"{' (full)' if self._full_source_mode else ''}")

        self._refresh_port_combo()
        if ports:
            self.port_combo.setCurrentText(ports[0])
            self._on_port_changed(ports[0])

        if self._full_source_mode:
            self.source_plot.set_plateau_region(start_sample, end_sample)

        ok, reason = supports_filter_recalculation(decomp_data)
        self._filter_recalc_available = ok
        self.btn_recalc_filter.setEnabled(ok)
        if not ok:
            self.btn_recalc_filter.setToolTip(f"Unavailable: {reason}")

        # ── Reliability summary ───────────────────────────────────────────────
        all_mus = [mu for mus in self._ports.values() for mu in mus]
        n_total     = len(all_mus)
        n_reliable  = sum(1 for mu in all_mus if mu.props is not None and mu.props.is_reliable)
        n_unreliable = n_total - n_reliable
        print(f"  Reliability: {n_reliable}/{n_total} reliable, "
              f"{n_unreliable}/{n_total} not reliable")
        self._update_status(
            f"Loaded {n_total} MUs — "
            f"{n_reliable} reliable  |  {n_unreliable} not reliable"
        )

    @staticmethod
    def _ensure_list_of_arrays(data) -> list:
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            return []
        if isinstance(data, list):
            if len(data) == 0: return []
            first = data[0]
            if isinstance(first, (np.ndarray, list)) or hasattr(first, "detach"):
                return [to_numpy(x) for x in data]
            return [to_numpy(data)]
        arr = to_numpy(data)
        if arr.ndim == 0 or arr.size == 0: return []
        if arr.ndim == 1: return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    def _load_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Decomposition", "", "Pickle (*.pkl);;All (*)")
        if path: self.load_from_path(Path(path))

    def _save_file(self):
        if not self._ports:
            self._update_status("Nothing to save"); return
        default = ""
        if self._loaded_path:
            default = str(self._loaded_path.with_name(self._loaded_path.stem + "_edited.pkl"))
        path, _ = QFileDialog.getSaveFileName(self, "Save Decomposition", default, "Pickle (*.pkl)")
        if not path: return
        try:
            with open(path, "wb") as f:
                pickle.dump(self._build_save_dict(), f)
            self._update_status(f"Saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _build_save_dict(self) -> dict:
        import dataclasses
        ports = list(self._ports.keys())
        discharge_times, pulse_trains, mu_filters, mu_properties = [], [], [], []

        for port_name in ports:
            kept = [mu for mu in self._ports[port_name] if not mu.flagged_duplicate]

            # Convert timestamps back to plateau-local for compatibility
            if self._full_source_mode:
                save_ts = [self._ts_to_plateau_local(mu.timestamps) for mu in kept]
                save_src = [
                    mu.source[self._start_sample:self._end_sample]
                    if len(mu.source) > (self._end_sample - self._start_sample) else mu.source
                    for mu in kept
                ]
            else:
                save_ts  = [mu.timestamps for mu in kept]
                save_src = [mu.source for mu in kept]

            discharge_times.append(save_ts)
            pulse_trains.append(save_src)
            mu_filters.append([mu.mu_filter for mu in kept] or None)

            port_props = []
            for mu in kept:
                if mu.props is not None:
                    d = dataclasses.asdict(mu.props)
                    d.pop("muap_grid", None); d.pop("duplicate_candidates", None)
                    port_props.append(d)
                else:
                    port_props.append({})
            mu_properties.append(port_props)

        save_data = {
            "ports": ports, "sampling_rate": self._fsamp,
            "discharge_times": discharge_times, "pulse_trains": pulse_trains,
            "mu_filters": mu_filters, "mu_properties": mu_properties,
        }
        if self._original_decomp_data is not None:
            for key in ["data", "plateau_coords", "chans_per_electrode", "emg_mask",
                        "electrodes", "dewhitened_filters", "version",
                        "preprocessing_config", "peel_off_sequence", "w_mat", "selected_points"]:
                val = self._original_decomp_data.get(key)
                if val is not None and key not in save_data:
                    save_data[key] = val
        if self._emg_data:
            save_data["emg_per_port"] = dict(self._emg_data)

        # Save original filters for comparison
        if self._original_decomp_data is not None:
            orig_filters = self._original_decomp_data.get("mu_filters")
            if orig_filters is not None:
                save_data["mu_filters_original"] = orig_filters

        return save_data

    def _set_mode(self, mode: EditMode):
        self._edit_mode = mode; self.source_plot.set_edit_mode(mode)
        {EditMode.VIEW: self.action_view, EditMode.ADD: self.action_add,
         EditMode.DELETE: self.action_delete}[mode].setChecked(True)
        self._update_status()

    def _reset_view(self):
        self.source_plot.plotItem.autoRange(); self.fr_plot.plotItem.autoRange()
        self._update_status("View reset")

    def _handle_add_click(self, sample: int):
        if self._edit_mode != EditMode.ADD: return
        mu = self._current_mu()
        if mu is None: return
        if sample < 0 or sample >= len(mu.source):
            self._update_status("Click outside source range"); return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end   = min(len(mu.source), int(view_x[1] * self._fsamp))
        peak = self._find_nearest_peak(mu.source, sample, view_start, view_end)
        if peak is None: self._update_status("No peak found near click"); return
        if peak in mu.timestamps: self._update_status("Spike already exists"); return
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.append(mu.timestamps, peak)).astype(np.int64)
        self._push_undo(UndoAction("Add spike", self._current_port, self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Added spike at {peak / self._fsamp:.3f}s")

    def _handle_delete_click(self, sample: int):
        if self._edit_mode != EditMode.DELETE: return
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) == 0: return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end   = min(len(mu.source), int(view_x[1] * self._fsamp))
        visible_mask = (mu.timestamps >= view_start) & (mu.timestamps < view_end)
        visible_ts   = mu.timestamps[visible_mask]
        if len(visible_ts) == 0: self._update_status("No spikes in view"); return
        tolerance = int(0.01 * self._fsamp)
        distances = np.abs(visible_ts - sample)
        nearest = np.argmin(distances)
        if distances[nearest] > tolerance: self._update_status("No spike near click"); return
        target = visible_ts[nearest]
        global_idx = np.where(mu.timestamps == target)[0][0]
        old_ts = mu.timestamps.copy()
        new_ts = np.delete(mu.timestamps, global_idx)
        self._push_undo(UndoAction("Delete spike", self._current_port, self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted spike at {target / self._fsamp:.3f}s")

    def _find_nearest_peak(self, source, click_sample, view_start, view_end):
        window = int(0.01 * self._fsamp)
        start = max(view_start, click_sample - window)
        end   = min(view_end,   click_sample + window)
        if start >= end: return None
        segment = source[start:end]
        peaks, _ = sp_signal.find_peaks(segment, distance=max(1, int(0.005 * self._fsamp)))
        if len(peaks) == 0: return int(start + np.argmax(segment))
        peaks_abs = peaks + start
        return int(peaks_abs[np.argmin(np.abs(peaks_abs - click_sample))])

    def _place_roi(self):
        if self._current_mu() is None: self._update_status("Select a motor unit first"); return
        self.source_plot.place_roi()
        self.action_roi_add.setEnabled(True); self.action_roi_delete.setEnabled(True); self.action_roi_clear.setEnabled(True)
        self._update_status("ROI placed — adjust, then Add/Delete in ROI")

    def _clear_roi(self):
        self.source_plot.remove_roi()
        self.action_roi_add.setEnabled(False); self.action_roi_delete.setEnabled(False); self.action_roi_clear.setEnabled(False)
        self._update_status("ROI cleared")

    def _roi_add_spikes(self):
        mu = self._current_mu(); bounds = self.source_plot.get_roi_bounds()
        if mu is None or bounds is None: return
        x1, x2, y1, y2 = bounds
        s1 = max(0, int(x1*self._fsamp)); s2 = min(len(mu.source), int(x2*self._fsamp))
        if s1 >= s2: return
        segment = mu.source[s1:s2]
        peaks, _ = sp_signal.find_peaks(segment, distance=max(1, int(0.005*self._fsamp)))
        peaks_abs = peaks + s1; existing = set(mu.timestamps.tolist())
        new_spikes = [int(p) for p in peaks_abs
                      if 0 <= p < len(mu.source) and y1 <= mu.source[p] <= y2 and int(p) not in existing]
        if not new_spikes: self._update_status("No new peaks in ROI"); return
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.concatenate([mu.timestamps, np.array(new_spikes, dtype=np.int64)]))
        self._push_undo(UndoAction(f"ROI add {len(new_spikes)}", self._current_port, self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Added {len(new_spikes)} spikes from ROI")

    def _roi_delete_spikes(self):
        mu = self._current_mu(); bounds = self.source_plot.get_roi_bounds()
        if mu is None or bounds is None: return
        x1, x2, y1, y2 = bounds
        s1 = int(x1*self._fsamp); s2 = int(x2*self._fsamp)
        in_box = np.array([s1 <= ts < s2 and 0 <= ts < len(mu.source) and y1 <= mu.source[ts] <= y2
                           for ts in mu.timestamps], dtype=bool)
        n_remove = int(np.sum(in_box))
        if n_remove == 0: self._update_status("No spikes in ROI"); return
        old_ts = mu.timestamps.copy(); new_ts = mu.timestamps[~in_box]
        self._push_undo(UndoAction(f"ROI delete {n_remove}", self._current_port, self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted {n_remove} spikes from ROI")

    def _push_undo(self, action: UndoAction):
        self._undo_stack.append(action); self._redo_stack.clear()
        if len(self._undo_stack) > 100: self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack: self._update_status("Nothing to undo"); return
        action = self._undo_stack.pop(); self._redo_stack.append(action)
        self._apply_undo_redo(action, is_undo=True)
        self._on_data_changed(f"Undo: {action.description}")

    def _redo(self):
        if not self._redo_stack: self._update_status("Nothing to redo"); return
        action = self._redo_stack.pop(); self._undo_stack.append(action)
        self._apply_undo_redo(action, is_undo=False)
        self._on_data_changed(f"Redo: {action.description}")

    def _apply_undo_redo(self, action: UndoAction, is_undo: bool):
        if self._current_port != action.port_name: self.port_combo.setCurrentText(action.port_name)
        if self._current_mu_idx != action.mu_idx: self.mu_combo.setCurrentIndex(action.mu_idx)
        mu = self._get_mu(action.port_name, action.mu_idx)
        if mu is None:
            return
        if is_undo:
            if action.old_timestamps is not None:
                mu.timestamps = action.old_timestamps
            if action.old_source is not None:
                mu.source = action.old_source
            if action.old_filter is not None:
                mu.mu_filter = action.old_filter
        else:
            if action.new_timestamps is not None:
                mu.timestamps = action.new_timestamps
            if action.new_source is not None:
                mu.source = action.new_source
            if action.new_filter is not None:
                mu.mu_filter = action.new_filter

    def _recalculate_filter(self):
        mu = self._current_mu()
        if mu is None: self._update_status("Select a motor unit first"); return
        if not self._filter_recalc_available:
            QMessageBox.warning(self, "Unavailable",
                "Filter recalculation requires peel_off_sequence in the file."); return
        if len(mu.timestamps) < 2:
            self._update_status("Need at least 2 spikes"); return

        raw_port = self._raw_port_channels.get(self._current_port)
        if raw_port is None:
            QMessageBox.warning(self, "Missing Data", "Raw port EMG not available."); return

        port_idx   = list(self._ports.keys()).index(self._current_port)
        global_idx = self._global_unit_idx(self._current_port, self._current_mu_idx)
        if global_idx is None:
            self._update_status("Could not determine global unit index"); return

        # Gather current filters for this port (may include previously recalculated ones)
        current_port_filters = [m.mu_filter for m in self._ports[self._current_port]]

        # Timestamps for recalculation: must be absolute
        ts_abs = mu.timestamps  # already absolute in full_source_mode
        if not self._full_source_mode:
            ts_abs = self._ts_to_absolute(mu.timestamps)

        self._update_status(f"Recalculating filter for MU {mu.id}…")
        try:
            new_filter, new_source_full = recalculate_unit_filter(
                raw_port_channels=raw_port,
                decomp_data=self._original_decomp_data,
                port_idx=port_idx,
                local_mu_idx=self._current_mu_idx,
                edited_timestamps_abs=ts_abs,
                global_unit_idx=global_idx,
                start_sample=self._start_sample,
                end_sample=self._end_sample,
                current_port_filters=current_port_filters,
            )

            # Save old state for undo
            old_source = mu.source.copy()
            old_filter = mu.mu_filter.copy() if mu.mu_filter is not None else None

            # Apply — only filter and source change, timestamps untouched
            mu.source    = new_source_full
            mu.mu_filter = new_filter

            undo_action = UndoAction(
                description=f"Recalculate filter MU {mu.id}",
                port_name=self._current_port,
                mu_idx=self._current_mu_idx,
                old_source=old_source,
                old_filter=old_filter,
                new_source=new_source_full,
                new_filter=new_filter,
            )
            self._push_undo(undo_action)

            self._on_data_changed(
                f"Filter recalculated for MU {mu.id}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Recalculation Error", str(e))
            self._update_status("Filter recalculation failed")

    def _global_unit_idx(self, port_name: str, mu_idx: int) -> Optional[int]:
        offset = 0
        for pname, mus in self._ports.items():
            if pname == port_name: return offset + mu_idx
            offset += len(mus)
        return None

    def _current_mu(self) -> Optional[MotorUnit]:
        return self._get_mu(self._current_port, self._current_mu_idx)

    def _get_mu(self, port, idx):
        if port is None or idx < 0: return None
        mus = self._ports.get(port, [])
        return mus[idx] if 0 <= idx < len(mus) else None

    def _on_mu_selected(self, index):
        if self._current_port is None: return
        mus = self._ports.get(self._current_port, [])
        if index < 0 or index >= len(mus): return
        self._current_mu_idx = index
        mu = self._current_mu()
        if mu: self.btn_flag_delete.setText("Unflag" if mu.flagged_duplicate else "🗑 Flag to Delete")
        self._update_plots(); self._update_status()

    def _select_prev_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx > 0: self.mu_combo.setCurrentIndex(idx - 1)

    def _select_next_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx < self.mu_combo.count() - 1: self.mu_combo.setCurrentIndex(idx + 1)

    def _toggle_flag_delete(self):
        mu = self._current_mu()
        if mu is None: return
        mu.flagged_duplicate = not mu.flagged_duplicate
        self._refresh_mu_combo(); self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_status(f"MU {mu.id} {'flagged' if mu.flagged_duplicate else 'unflagged'}")

    def _refresh_port_combo(self):
        cur = self.port_combo.currentText()
        self.port_combo.blockSignals(True); self.port_combo.clear()
        for name in self._ports: self.port_combo.addItem(name)
        if cur in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
            self.port_combo.setCurrentText(cur)
        self.port_combo.blockSignals(False)

    def _on_port_changed(self, port_name):
        if not port_name or port_name not in self._ports: return
        self._current_port = port_name; self._current_mu_idx = -1
        self._clear_plots(); self._refresh_mu_combo()
        if self.mu_combo.count() > 0:
            self.mu_combo.setCurrentIndex(0); self._on_mu_selected(0)

    def _refresh_mu_combo(self):
        self.mu_combo.blockSignals(True); self.mu_combo.clear()
        if self._current_port is not None:
            for mu in self._ports.get(self._current_port, []):
                label = f"MU {mu.id}  ({len(mu.timestamps)} spikes)"
                if mu.flagged_duplicate: label += "  ⚠"
                if mu.props is not None: label += "  ✓" if mu.props.is_reliable else "  ✗"
                self.mu_combo.addItem(label)
        self.mu_combo.blockSignals(False)

    def _update_plots(self):
        mu = self._current_mu()
        if mu is None: self._clear_plots(); return
        self.source_plot.set_data(mu.source, mu.timestamps)
        if self._full_source_mode:
            self.source_plot.set_plateau_region(self._start_sample, self._end_sample)
        self.fr_plot.set_data(mu.timestamps)
        self._plot_muap()
        self._update_quality_panel(mu)

    def _clear_plots(self):
        self.source_plot.clear_data(); self.fr_plot.clear_data()
        self._clear_muap_plot(); self.quality_bar.clear_properties()

    def _on_data_changed(self, msg="Modified"):
        """Recompute properties after spike edit. ★ uses plateau-local for properties ★"""
        mu = self._current_mu()
        if mu is not None:
            grid_cfg = self._grid_info.get(self._current_port)
            emg_port = self._emg_data.get(self._current_port)

            if self._full_source_mode:
                props_ts = self._ts_to_plateau_local(mu.timestamps)
                props_source = (
                    mu.source[self._start_sample:self._end_sample]
                    if len(mu.source) > (self._end_sample - self._start_sample)
                    else mu.source)
            else:
                props_ts     = mu.timestamps
                props_source = mu.source

            mu.props = recompute_unit_properties(
                mu_props=mu.props or MUProperties(),
                new_timestamps=props_ts, source=props_source,
                emg_port=emg_port,
                grid_positions=grid_cfg["positions"] if grid_cfg else None,
                grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
                fsamp=self._fsamp,
            )
        self._update_plots(); self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_status(msg); self.data_modified.emit()

    def _update_quality_panel(self, mu):
        if mu.props is not None: self.quality_bar.set_properties(mu.props)
        else: self.quality_bar.clear_properties()

    def _update_status(self, msg=None):
        if msg: self.status_bar.showMessage(msg, 4000); return
        parts = []
        if self._current_port: parts.append(f"Port: {self._current_port}")
        n = len(self._ports.get(self._current_port, []))
        if n > 0: parts.append(f"{n} MUs")
        if self._current_mu_idx >= 0: parts.append(f"MU: {self._current_mu_idx}")
        hints = {EditMode.VIEW: "View [V]", EditMode.ADD: "Add [A]", EditMode.DELETE: "Delete [D]"}
        parts.append(hints[self._edit_mode])
        if self.source_plot.has_roi(): parts.append("ROI")
        if self._undo_stack: parts.append(f"Undo: {len(self._undo_stack)}")
        if self._full_source_mode: parts.append("Full signal")
        self.status_bar.showMessage("  |  ".join(parts))

    def _plot_muap(self):
        mu = self._current_mu()
        if mu is None:
            self._clear_muap_plot("Select a Motor Unit"); return
        if mu.props is None or mu.props.muap_grid is None:
            no_emg = self._emg_data.get(self._current_port) is None
            msg = "No EMG data in file" if no_emg else "MUAP unavailable"
            self._clear_muap_plot(msg); return
        muap_grid = mu.props.muap_grid
        grid_cfg  = self._grid_info.get(self._current_port)
        if grid_cfg is not None:
            rows, cols = grid_cfg["grid_shape"]; positions = grid_cfg["positions"]
            waveforms, ch_indices = [], []
            for ch in range(rows * cols):
                pos = positions.get(ch)
                if pos is None: continue
                r, c = pos
                if r < rows and c < cols:
                    waveforms.append(muap_grid[r, c]); ch_indices.append(ch)
            self._render_muap_grid(waveforms, ch_indices, grid_cfg)
        else:
            n_ch = muap_grid.shape[0]
            waveforms = [muap_grid[i, 0] for i in range(n_ch)]
            self._render_muap_stacked(waveforms, list(range(n_ch)))

    def _render_muap_grid(self, waveforms, ch_indices, grid_cfg):
        self.muap_widget.clear()
        rows, cols = grid_cfg["grid_shape"]; positions = grid_cfg["positions"]
        valid = [w for w in waveforms if len(w) > 0]
        amp = np.max(np.abs(np.concatenate(valid))) * 1.2 if valid else 1.0
        n_samples = len(waveforms[0]) if valid else 409
        label = f"<span style='color:{COLORS['foreground']};font-size:10pt;'>MU {self._current_mu_idx}</span>"
        self.muap_widget.addLabel(label, row=0, col=0, colspan=cols)
        for r in range(rows):
            for c in range(cols):
                p = self.muap_widget.addPlot(row=r+1, col=c)
                p.hideAxis("left"); p.hideAxis("bottom"); p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0); p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
        for idx, wav in enumerate(waveforms):
            if idx >= len(ch_indices): break
            pos = positions.get(int(idx))
            if pos is None: continue
            r, c = pos
            if r >= rows or c >= cols: continue
            item = self.muap_widget.getItem(r+1, c)
            if item is not None and len(wav) > 0:
                item.plot(wav, pen=pg.mkPen(color=COLORS["info"], width=1.5))

    def _render_muap_stacked(self, waveforms, ch_indices):
        self.muap_widget.clear()
        plot = self.muap_widget.addPlot(row=0, col=0)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid: return
        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["foreground"], width=1.5))
            txt = pg.TextItem(f"Ch {ch}", color=(150,150,150), anchor=(1,0.5))
            txt.setPos(-1, offset); txt.setFont(QFont(FONT_FAMILY, 7)); plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        plot.setTitle(f"MU {self._current_mu_idx} — Stacked", color=COLORS["foreground"], size="10pt")

    def _clear_muap_plot(self, message: str = "Select a Motor Unit"):
        self.muap_widget.clear()
        p = self.muap_widget.addPlot(row=0, col=0)
        t = pg.TextItem(message, color=(120,120,120), anchor=(0.5,0.5))
        t.setFont(QFont(FONT_FAMILY, 14)); p.addItem(t)
        p.hideAxis("left"); p.hideAxis("bottom")