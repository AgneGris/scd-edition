"""
Edition Tab - Self-contained EMG spike editing interface.

Features:
- Source signal display with spike markers
- Single-click spike add/delete with clear mode indication
- Place-then-act ROI workflow (place box → adjust → add/delete)
- MUAP grid/stacked visualization
- Instantaneous firing rate plot
- Quality metrics (SIL, CoV, firing rate)
- Undo/redo stack
- Loads decomposition output directly (includes EMG data)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pickle
import traceback

import numpy as np
from scipy import signal as sp_signal
from scipy.cluster.vq import kmeans2

from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QToolBar,
    QAction, QActionGroup, QComboBox, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QFileDialog, QMessageBox,
    QShortcut, QStatusBar, QFrame, QSizePolicy
)
from PyQt5.QtGui import QKeySequence, QCursor, QFont, QColor
import pyqtgraph as pg

# Use the same styling as the rest of the app
from gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, get_button_style
)


# ============================================================
# Electrode grid configurations
# ============================================================

ELECTRODE_GRIDS = {
    "GR04MM1305": {"grid_shape": (13, 5), "ied_mm": 4, "n_channels": 64},
    "GR08MM1305": {"grid_shape": (13, 5), "ied_mm": 8, "n_channels": 64},
    "GR10MM0808": {"grid_shape": (8, 8),  "ied_mm": 10, "n_channels": 64},
    "GR10MM0804": {"grid_shape": (8, 4),  "ied_mm": 10, "n_channels": 32},
}


# ============================================================
# Helpers
# ============================================================

def to_numpy(obj) -> np.ndarray:
    """Convert any array-like (including torch tensors) to numpy."""
    if obj is None:
        return np.array([])
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, 'detach'):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def get_grid_config(electrode_type: Optional[str]) -> Optional[Dict]:
    """Return grid layout info for a known electrode type, or None."""
    if electrode_type is None:
        return None
    key = electrode_type.upper().replace("-", "").replace(" ", "")
    for name, cfg in ELECTRODE_GRIDS.items():
        if name.upper() == key:
            rows, cols = cfg["grid_shape"]
            positions = {}
            for i in range(cfg["n_channels"]):
                positions[i] = (i // cols, i % cols)
            return {
                "grid_shape": cfg["grid_shape"],
                "positions": positions,
                "ied_mm": cfg["ied_mm"],
            }
    return None


# ============================================================
# Data Structures
# ============================================================

class EditMode(Enum):
    VIEW = "view"
    ADD = "add"
    DELETE = "delete"


@dataclass
class MotorUnit:
    id: int
    timestamps: np.ndarray       # spike times in samples
    source: np.ndarray           # source signal (1D)
    port_name: str = ""
    mu_filter: Optional[np.ndarray] = None
    enabled: bool = True
    flagged_duplicate: bool = False


@dataclass
class UndoAction:
    description: str
    mu_idx: int
    old_timestamps: np.ndarray
    new_timestamps: np.ndarray


# ============================================================
# MUAP Extraction
# ============================================================

def cut_muap(
    discharge_times: np.ndarray,
    window_samples: int,
    channel_data: np.ndarray,
) -> np.ndarray:
    """Cut MUAP waveforms around discharge times."""
    waveforms = []
    n_samples = len(channel_data)
    for t in discharge_times:
        t = int(t)
        start = t - window_samples
        end = t + window_samples + 1
        if 0 <= start and end <= n_samples:
            waveforms.append(channel_data[start:end])
    if waveforms:
        return np.array(waveforms)
    return np.empty((0, 2 * window_samples + 1))


def get_muaps(
    discharge_times: List[np.ndarray],
    n_mus: int,
    data: np.ndarray,
    chans2use: np.ndarray,
    fsamp: float,
    window_ms: float = 20.0,
) -> List[List[np.ndarray]]:
    """
    Extract MUAPs using spike-triggered averaging.
    data: (channels, samples)
    Returns: muaps[mu_idx][channel_idx] = averaged waveform (1D)
    """
    window_samples = int(fsamp * window_ms / 1000)
    muaps_averaged: List[List[np.ndarray]] = []

    for mu_idx in range(n_mus):
        per_channel = []
        for ch in chans2use:
            ch = int(ch)
            if ch >= data.shape[0]:
                per_channel.append(np.zeros(2 * window_samples + 1))
                continue
            waveforms = cut_muap(discharge_times[mu_idx], window_samples, data[ch])
            if waveforms.shape[0] > 0:
                per_channel.append(np.mean(waveforms, axis=0))
            else:
                per_channel.append(np.zeros(2 * window_samples + 1))
        muaps_averaged.append(per_channel)

    return muaps_averaged


# ============================================================
# Quality Metrics
# ============================================================

def compute_sil(source: np.ndarray, timestamps: np.ndarray) -> float:
    """Silhouette score: separation of spike vs noise clusters."""
    if len(timestamps) < 3 or len(source) == 0:
        return 0.0
    try:
        valid_ts = timestamps[timestamps < len(source)]
        peaks = source[valid_ts]
        if len(peaks) < 3:
            return 0.0
        spike_mean = np.mean(np.abs(peaks))
        noise_std = np.std(source)
        if noise_std == 0:
            return 0.0
        sil = np.clip(spike_mean / (noise_std * 3) - 0.2, 0, 1)
        return float(sil)
    except Exception:
        return 0.0


def compute_cov(timestamps: np.ndarray, fsamp: float) -> float:
    """Coefficient of variation of inter-spike intervals."""
    if len(timestamps) < 3:
        return 0.0
    isi = np.diff(np.sort(timestamps)) / fsamp
    isi = isi[isi > 0.01]
    if len(isi) < 2:
        return 0.0
    return float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0.0


def compute_firing_rate(timestamps: np.ndarray, fsamp: float) -> float:
    """Mean firing rate in Hz."""
    if len(timestamps) < 2:
        return 0.0
    isi = np.diff(np.sort(timestamps)) / fsamp
    isi = isi[isi > 0.01]
    if len(isi) == 0:
        return 0.0
    return float(1.0 / np.mean(isi))


# ============================================================
# Plot Widgets (styled to match app theme)
# ============================================================

_PG_BACKGROUND = COLORS['background']
_PG_FOREGROUND = COLORS['foreground']
_PG_DIM = COLORS.get('text_dim', '#6c7086')
_PG_ACCENT = COLORS.get('info', '#89b4fa')
_PG_SUCCESS = COLORS['success']
_PG_WARNING = COLORS.get('warning', '#f9e2af')
_PG_ERROR = COLORS['error']


class SourcePlotWidget(pg.PlotWidget):
    """Source signal plot with spike markers and ROI support."""

    spike_add_requested = pyqtSignal(int)
    spike_delete_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background=_PG_BACKGROUND)

        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=_PG_DIM)
        self.setLabel("left", "Amplitude", color=_PG_DIM)
        self.getAxis("bottom").setPen(_PG_DIM)
        self.getAxis("left").setPen(_PG_DIM)
        self.getAxis("bottom").setTextPen(_PG_DIM)
        self.getAxis("left").setTextPen(_PG_DIM)

        self._fsamp = 1.0
        self._edit_mode = EditMode.VIEW
        self._source: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None

        self._signal_curve = self.plot([], pen=pg.mkPen('#2b6cb0', width=1))
        self._spike_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush('#ed8936'),
            symbol="o", hoverable=True,
        )
        self.addItem(self._spike_scatter)

        self._roi: Optional[pg.ROI] = None
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def set_edit_mode(self, mode: EditMode):
        self._edit_mode = mode
        cursors = {
            EditMode.VIEW: Qt.ArrowCursor,
            EditMode.ADD: Qt.CrossCursor,
            EditMode.DELETE: Qt.PointingHandCursor,
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))

    def set_data(self, source: np.ndarray, timestamps: np.ndarray):
        self._source = source
        self._timestamps = timestamps
        t = np.arange(len(source)) / self._fsamp
        self._signal_curve.setData(t, source)
        self._update_spike_markers()

    def _update_spike_markers(self):
        if self._source is None or self._timestamps is None or len(self._timestamps) == 0:
            self._spike_scatter.setData([], [])
            return
        valid = self._timestamps[self._timestamps < len(self._source)]
        if len(valid) == 0:
            self._spike_scatter.setData([], [])
            return
        t = valid / self._fsamp
        y = self._source[valid]
        self._spike_scatter.setData(t, y)

    def clear_data(self):
        self._signal_curve.setData([], [])
        self._spike_scatter.setData([], [])
        self._source = None
        self._timestamps = None

    def place_roi(self):
        self.remove_roi()
        vb = self.getViewBox()
        vr = vb.viewRange()
        x_range = vr[0][1] - vr[0][0]
        y_range = vr[1][1] - vr[1][0]
        w = x_range * 0.3
        h = y_range * 0.6
        cx = (vr[0][0] + vr[0][1]) / 2
        cy = (vr[1][0] + vr[1][1]) / 2

        roi_border_color = COLORS.get('warning', '#f9e2af')
        self._roi = pg.ROI(
            pos=[cx - w / 2, cy - h / 2],
            size=[w, h],
            pen=pg.mkPen(color=roi_border_color, width=2),
            movable=True,
            resizable=True,
        )
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.addScaleHandle([1, 0], [0, 1])
        self._roi.addScaleHandle([0, 1], [1, 0])
        self.addItem(self._roi)

    def remove_roi(self):
        if self._roi is not None:
            self.removeItem(self._roi)
            self._roi = None

    def get_roi_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self._roi is None:
            return None
        pos = self._roi.pos()
        size = self._roi.size()
        x1, y1 = pos.x(), pos.y()
        x2, y2 = x1 + size.x(), y1 + size.y()
        return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))

    def has_roi(self) -> bool:
        return self._roi is not None

    def _on_mouse_clicked(self, ev):
        if self._edit_mode == EditMode.VIEW:
            return
        if ev.button() != Qt.LeftButton:
            return
        pos = self.plotItem.vb.mapSceneToView(ev.scenePos())
        sample = int(pos.x() * self._fsamp)

        if self._edit_mode == EditMode.ADD:
            self.spike_add_requested.emit(sample)
        elif self._edit_mode == EditMode.DELETE:
            self.spike_delete_requested.emit(sample)


class FiringRatePlotWidget(pg.PlotWidget):
    """Instantaneous firing rate display."""

    def __init__(self, parent=None):
        super().__init__(parent, background=_PG_BACKGROUND)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=_PG_DIM)
        self.setLabel("left", "IFR (Hz)", color=_PG_DIM)
        self.getAxis("bottom").setPen(_PG_DIM)
        self.getAxis("left").setPen(_PG_DIM)
        self.getAxis("bottom").setTextPen(_PG_DIM)
        self.getAxis("left").setTextPen(_PG_DIM)
        self._curve = self.plot([], pen=pg.mkPen(_PG_WARNING, width=1.5))
        self._fsamp = 1.0

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def link_x(self, other: pg.PlotWidget):
        self.setXLink(other)

    def set_data(self, timestamps: np.ndarray):
        ts = np.sort(timestamps)
        if len(ts) < 2:
            self._curve.setData([], [])
            return
        isi = np.diff(ts) / self._fsamp
        ifr = np.where(isi > 0.01, 1.0 / isi, 0.0)
        t_mid = (ts[:-1] + ts[1:]) / 2 / self._fsamp
        self._curve.setData(t_mid, ifr)

    def clear_data(self):
        self._curve.setData([], [])


class QualityBar(QFrame):
    """Compact quality metrics display - styled to match app."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)

        label_style = f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small', '9pt')};"

        self._sil_label = QLabel("SIL: —")
        self._cov_label = QLabel("CoV: —")
        self._fr_label = QLabel("FR: —")
        self._n_label = QLabel("Spikes: —")

        for lbl in (self._sil_label, self._cov_label, self._fr_label, self._n_label):
            lbl.setStyleSheet(label_style)
            layout.addWidget(lbl)
        layout.addStretch()

    def set_metrics(self, sil: float, cov: float, fr: float, n_spikes: int):
        sil_color = _PG_SUCCESS if sil > 0.6 else (_PG_WARNING if sil > 0.3 else _PG_ERROR)
        cov_color = _PG_SUCCESS if cov < 0.3 else (_PG_WARNING if cov < 0.5 else _PG_ERROR)

        self._sil_label.setText(f"SIL: <span style='color:{sil_color}'>{sil:.2f}</span>")
        self._sil_label.setTextFormat(Qt.RichText)
        self._cov_label.setText(f"CoV: <span style='color:{cov_color}'>{cov:.2f}</span>")
        self._cov_label.setTextFormat(Qt.RichText)
        self._fr_label.setText(f"FR: {fr:.1f} Hz")
        self._n_label.setText(f"Spikes: {n_spikes}")

    def clear_metrics(self):
        for lbl in (self._sil_label, self._cov_label, self._fr_label, self._n_label):
            lbl.setText(lbl.text().split(":")[0] + ": —")


# ============================================================
# Edition Tab
# ============================================================

class EditionTab(QWidget):
    """
    Main edition interface for editing motor unit spike trains.

    Supports two loading paths:
        1. Programmatic: decomposition_complete signal → load_from_path(decomp_path)
        2. Manual: user clicks Load → picks .pkl file → load_from_path(path)
    
    The decomposition output .pkl contains everything needed (EMG + results),
    so no separate EMG file is required.
    """

    data_modified = pyqtSignal()

    def __init__(self, fsamp: float = 2048.0, parent=None):
        super().__init__(parent)

        self._fsamp = fsamp
        self._ports: Dict[str, List[MotorUnit]] = {}
        self._emg_data: Dict[str, np.ndarray] = {}          # port_name → (channels, samples)
        self._muap_data: Dict[str, Dict] = {}
        self._current_port: Optional[str] = None
        self._current_mu_idx: int = -1
        self._edit_mode = EditMode.VIEW
        self._loaded_path: Optional[Path] = None

        self._undo_stack: List[UndoAction] = []
        self._redo_stack: List[UndoAction] = []

        self._build_ui()
        self._setup_shortcuts()

    # --------------------------------------------------------
    # UI Construction
    # --------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.toolbar = self._build_toolbar()
        root.addWidget(self.toolbar)

        self.quality_bar = QualityBar()
        root.addWidget(self.quality_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 1080])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        root.addWidget(splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        root.addWidget(self.status_bar)
        self._update_status()

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar()
        tb.setMovable(False)
        tb.setStyleSheet(f"""
            QToolBar {{
                background-color: {COLORS.get('background_light', '#2a2a3c')};
                border-bottom: 1px solid {COLORS['border']};
                spacing: 4px;
                padding: 2px;
            }}
            QToolBar QLabel {{
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES.get('small', '9pt')};
            }}
            QToolButton {{
                color: {COLORS['foreground']};
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {FONT_SIZES.get('small', '9pt')};
            }}
            QToolButton:hover {{
                background-color: {COLORS.get('background_input', '#33334d')};
                border-color: {COLORS['border']};
            }}
            QToolButton:checked {{
                background-color: {COLORS.get('info', '#89b4fa')}30;
                border-color: {COLORS.get('info', '#89b4fa')};
            }}
        """)

        # File
        self.action_load = QAction("📂 Load", self)
        self.action_load.triggered.connect(self._load_file_dialog)
        tb.addAction(self.action_load)

        self.action_save = QAction("💾 Save", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self._save_file)
        tb.addAction(self.action_save)

        tb.addSeparator()

        lbl = QLabel("  Mode: ")
        tb.addWidget(lbl)

        self.mode_group = QActionGroup(self)

        self.action_view = QAction("👁 View", self)
        self.action_view.setCheckable(True)
        self.action_view.setChecked(True)
        self.action_view.setShortcut(QKeySequence("V"))
        self.action_view.triggered.connect(lambda: self._set_mode(EditMode.VIEW))
        self.mode_group.addAction(self.action_view)
        tb.addAction(self.action_view)

        self.action_add = QAction("➕ Add", self)
        self.action_add.setCheckable(True)
        self.action_add.setShortcut(QKeySequence("A"))
        self.action_add.triggered.connect(lambda: self._set_mode(EditMode.ADD))
        self.mode_group.addAction(self.action_add)
        tb.addAction(self.action_add)

        self.action_delete = QAction("➖ Delete", self)
        self.action_delete.setCheckable(True)
        self.action_delete.setShortcut(QKeySequence("D"))
        self.action_delete.triggered.connect(lambda: self._set_mode(EditMode.DELETE))
        self.mode_group.addAction(self.action_delete)
        tb.addAction(self.action_delete)

        tb.addSeparator()

        lbl2 = QLabel("  ROI: ")
        tb.addWidget(lbl2)

        self.action_place_roi = QAction("⬜ Place", self)
        self.action_place_roi.setShortcut(QKeySequence("R"))
        self.action_place_roi.triggered.connect(self._place_roi)
        tb.addAction(self.action_place_roi)

        self.action_roi_add = QAction("✅ Add in ROI", self)
        self.action_roi_add.setShortcut(QKeySequence("Shift+A"))
        self.action_roi_add.triggered.connect(self._roi_add_spikes)
        self.action_roi_add.setEnabled(False)
        tb.addAction(self.action_roi_add)

        self.action_roi_delete = QAction("🗑 Del in ROI", self)
        self.action_roi_delete.setShortcut(QKeySequence("Shift+D"))
        self.action_roi_delete.triggered.connect(self._roi_delete_spikes)
        self.action_roi_delete.setEnabled(False)
        tb.addAction(self.action_roi_delete)

        self.action_roi_clear = QAction("✕ Clear", self)
        self.action_roi_clear.setShortcut(QKeySequence(Qt.Key_Escape))
        self.action_roi_clear.triggered.connect(self._clear_roi)
        self.action_roi_clear.setEnabled(False)
        tb.addAction(self.action_roi_clear)

        tb.addSeparator()

        self.action_undo = QAction("↩ Undo", self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.triggered.connect(self._undo)
        tb.addAction(self.action_undo)

        self.action_redo = QAction("↪ Redo", self)
        self.action_redo.setShortcut(QKeySequence.Redo)
        self.action_redo.triggered.connect(self._redo)
        tb.addAction(self.action_redo)

        return tb

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {COLORS['background']}; "
            f"border-right: 2px solid {COLORS['border']};"
        )

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Section header
        lay.addWidget(QLabel("PORT SELECTION", styleSheet=get_section_header_style('info')))

        port_row = QHBoxLayout()
        port_lbl = QLabel("Port:")
        port_lbl.setStyleSheet(f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small', '9pt')};")
        port_row.addWidget(port_lbl)
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS.get('background_input', '#33334d')};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
        """)
        self.port_combo.currentTextChanged.connect(self._on_port_changed)
        port_row.addWidget(self.port_combo, stretch=1)
        lay.addLayout(port_row)

        # MU list header
        lay.addWidget(QLabel("MOTOR UNITS", styleSheet=get_section_header_style('info')))

        self.mu_list = QListWidget()
        self.mu_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS.get('background_input', '#33334d')};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                font-size: {FONT_SIZES.get('small', '9pt')};
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS.get('info', '#89b4fa')}30;
                color: {COLORS['foreground']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS.get('background_light', '#2a2a3c')};
            }}
        """)
        self.mu_list.currentRowChanged.connect(self._on_mu_selected)
        lay.addWidget(self.mu_list)

        # Unit controls
        btn_row = QHBoxLayout()
        btn_style = f"""
            QPushButton {{
                background-color: {COLORS.get('background_input', '#33334d')};
                color: {COLORS['foreground']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: {FONT_SIZES.get('small', '9pt')};
            }}
            QPushButton:hover {{
                background-color: {COLORS.get('background_light', '#2a2a3c')};
                border-color: {COLORS.get('info', '#89b4fa')};
            }}
        """

        self.btn_flag_delete = QPushButton("🗑 Flag to Delete")
        self.btn_flag_delete.setStyleSheet(btn_style)
        self.btn_flag_delete.setShortcut(QKeySequence("X"))
        self.btn_flag_delete.clicked.connect(self._toggle_flag_delete)
        btn_row.addWidget(self.btn_flag_delete)
        lay.addLayout(btn_row)

        # MUAP preview
        lay.addWidget(QLabel("MUAP PREVIEW", styleSheet=get_section_header_style('info')))

        self.muap_widget = pg.GraphicsLayoutWidget()
        self.muap_widget.setBackground(_PG_BACKGROUND)
        self.muap_widget.setMinimumHeight(50)
        lay.addWidget(self.muap_widget, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['background']};")

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        plot_splitter = QSplitter(Qt.Vertical)
        plot_splitter.setHandleWidth(2)

        self.source_plot = SourcePlotWidget()
        self.source_plot.set_fsamp(self._fsamp)
        self.source_plot.spike_add_requested.connect(self._handle_add_click)
        self.source_plot.spike_delete_requested.connect(self._handle_delete_click)
        plot_splitter.addWidget(self.source_plot)

        self.fr_plot = FiringRatePlotWidget()
        self.fr_plot.set_fsamp(self._fsamp)
        self.fr_plot.link_x(self.source_plot)
        self.fr_plot.setMaximumHeight(150)
        plot_splitter.addWidget(self.fr_plot)

        plot_splitter.setSizes([500, 150])
        lay.addWidget(plot_splitter)
        return panel

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Up"), self, self._select_prev_mu)
        QShortcut(QKeySequence("Down"), self, self._select_next_mu)

    # --------------------------------------------------------
    # Public API - Loading
    # --------------------------------------------------------

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp
        self.source_plot.set_fsamp(fsamp)
        self.fr_plot.set_fsamp(fsamp)

    def load_from_path(self, path: Path):
        """
        Load a decomposition .pkl file. This is the ONE entry point for loading.
        Called both from the Load button and from the decomposition_complete signal.
        
        The .pkl contains EMG data, so no separate EMG file is needed.
        """
        path = Path(path)
        if not path.exists():
            QMessageBox.critical(self, "Load Error", f"File not found:\n{path}")
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{e}")
            return

        # Validate minimum required keys
        if "ports" not in data or "discharge_times" not in data:
            QMessageBox.warning(
                self, "Format Error",
                "File does not contain expected decomposition keys ('ports', 'discharge_times')."
            )
            return

        try:
            self._load_decomposition_data(data)
            self._loaded_path = path
            self._update_status(f"Loaded: {path.name}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to parse decomposition data:\n{e}")

    def _load_decomposition_data(self, decomp_data: dict):
        """
        Parse a decomposition dictionary and populate the editor.
        
        Expected format (from DecompositionWorker._save_results):
            - ports: List[str]
            - discharge_times: List[per_port]  (each is list of arrays or single array)
            - pulse_trains: List[per_port]
            - mu_filters: List[per_port]
            - sampling_rate: int
            - data: np.ndarray (channels, samples) — full raw EMG
            - plateau_coords: [start_sample, end_sample]
            - chans_per_electrode: List[int]
            - emg_mask: List[per_port_mask]
            - electrodes: List[Optional[str]]
        """
        # Reset state
        self._ports.clear()
        self._emg_data.clear()
        self._muap_data.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()

        # 1. Sampling rate
        fsamp = decomp_data.get("sampling_rate", decomp_data.get("fsamp", self._fsamp))
        self.set_fsamp(float(fsamp))

        # === DEBUG: Inspect what's actually in the file ===
        print("\n" + "="*60)
        print("EDITION TAB: Loading decomposition data")
        print(f"  Keys in file: {list(decomp_data.keys())}")
        print(f"  Ports: {decomp_data.get('ports', 'MISSING')}")
        print(f"  Sampling rate: {decomp_data.get('sampling_rate', 'MISSING')}")

        raw_data = decomp_data.get("data")
        if raw_data is not None:
            emg_full = to_numpy(raw_data)
            print(f"  EMG shape raw: {emg_full.shape}")
        else:
            emg_full = None
            print("  EMG data: MISSING")
        
        # After orientation fix
        if emg_full is not None and emg_full.ndim == 2:
            if emg_full.shape[0] > emg_full.shape[1]:
                emg_full = emg_full.T
            print(f"  EMG shape final: {emg_full.shape}  (channels, samples)")
            
        if emg_full is not None and emg_full.ndim == 2:
            # Ensure (channels, samples): channels < samples
            if emg_full.shape[0] > emg_full.shape[1]:
                emg_full = emg_full.T
            print(f"  EMG full shape: {emg_full.shape}  (channels, samples)")

        # 3. Plateau / time window
        sel_pts = decomp_data.get("plateau_coords", decomp_data.get("selected_points"))
        print(f"  Plateau coords raw: {sel_pts}")
        
        start_sample, end_sample = 0, 0
        if sel_pts is not None:
            try:
                pts = to_numpy(np.asarray(sel_pts)).flatten()
                start_sample = int(pts[0])
                end_sample = int(pts[1])
            except (IndexError, TypeError, ValueError):
                start_sample, end_sample = 0, 0

        if end_sample <= start_sample and emg_full is not None:
            end_sample = emg_full.shape[1]  # Use full length

        # 4. Per-port data
        ports = decomp_data.get("ports", [])
        chans_per_electrode = decomp_data.get("chans_per_electrode", [])
        mask_list = decomp_data.get("emg_mask", [])
        electrode_list = decomp_data.get("electrodes", [])

        ch_offset = 0  # Running channel offset across ports

        for port_idx, port_name in enumerate(ports):
            n_ch = int(chans_per_electrode[port_idx]) if port_idx < len(chans_per_electrode) else 64

            print(f"\n  --- Port {port_idx}: '{port_name}' ---")
            print(f"    n_channels: {n_ch}, ch_offset: {ch_offset}")

            # Active channels mask
            if port_idx < len(mask_list) and mask_list[port_idx] is not None:
                local_mask = to_numpy(np.asarray(mask_list[port_idx])).flatten()
                local_active = np.where(local_mask == 0)[0]
            else:
                local_active = np.arange(n_ch)

            global_active = local_active + ch_offset
            print(f"    active channels: {len(local_active)}/{n_ch}")

            # Slice EMG
            emg_port = None
            if emg_full is not None:
                valid_end = min(end_sample, emg_full.shape[1])
                valid_start = max(0, start_sample)
                valid_channels = global_active[global_active < emg_full.shape[0]]
                if len(valid_channels) > 0:
                    emg_port = emg_full[valid_channels, valid_start:valid_end]
                print(f"    emg_port shape: {emg_port.shape if emg_port is not None else 'None'}")

            # Get per-port data
            port_discharge = decomp_data["discharge_times"][port_idx] if port_idx < len(decomp_data["discharge_times"]) else []
            port_sources = decomp_data["pulse_trains"][port_idx] if port_idx < len(decomp_data["pulse_trains"]) else []
            port_filters = decomp_data.get("mu_filters", [])
            port_filters = port_filters[port_idx] if port_idx < len(port_filters) else None

            # DEBUG: inspect raw data
            print(f"    discharge_times type: {type(port_discharge)}")
            if isinstance(port_discharge, list):
                print(f"    discharge_times len: {len(port_discharge)}")
                for i, dt in enumerate(port_discharge[:3]):
                    arr = to_numpy(dt)
                    print(f"      [{i}] shape={arr.shape}, dtype={arr.dtype}, first5={arr.flatten()[:5]}")
            elif isinstance(port_discharge, np.ndarray):
                print(f"    discharge_times shape: {port_discharge.shape}, ndim={port_discharge.ndim}")
            else:
                print(f"    discharge_times value: {repr(port_discharge)}")

            if isinstance(port_sources, np.ndarray):
                print(f"    pulse_trains shape: {port_sources.shape}")
            elif isinstance(port_sources, list):
                print(f"    pulse_trains len: {len(port_sources)}")
            else:
                print(f"    pulse_trains type: {type(port_sources)}, value: {repr(port_sources)}")

            # Normalize to lists
            ts_list = self._ensure_list_of_arrays(port_discharge)
            src_list = self._ensure_list_of_arrays(port_sources)

            print(f"    After normalize: {len(ts_list)} MUs from timestamps, {len(src_list)} from sources")
            for i, ts in enumerate(ts_list[:3]):
                print(f"      MU[{i}] timestamps: len={len(ts)}, first5={ts[:5]}")

            if port_filters is not None:
                filt_list = self._ensure_list_of_arrays(port_filters)
            else:
                filt_list = [None] * len(ts_list)

            # Build MotorUnit objects
            motor_units = []
            for mu_idx in range(len(ts_list)):
                ts = to_numpy(ts_list[mu_idx]).flatten().astype(np.int64)
                src = to_numpy(src_list[mu_idx]).flatten() if mu_idx < len(src_list) else np.zeros(1)
                filt = to_numpy(filt_list[mu_idx]) if mu_idx < len(filt_list) and filt_list[mu_idx] is not None else None

                motor_units.append(MotorUnit(
                    id=mu_idx,
                    timestamps=ts,
                    source=src,
                    port_name=port_name,
                    mu_filter=filt,
                ))

            print(f"    Created {len(motor_units)} MotorUnit objects")

            # Store
            self._ports[port_name] = motor_units

            if emg_port is not None:
                self._emg_data[port_name] = emg_port

            # Calculate MUAPs if we have EMG
            if emg_port is not None and len(motor_units) > 0:
                chans = np.arange(emg_port.shape[0])
                etype = electrode_list[port_idx] if port_idx < len(electrode_list) else None
                try:
                    muaps = get_muaps(
                        discharge_times=[mu.timestamps for mu in motor_units],
                        n_mus=len(motor_units),
                        data=emg_port,
                        chans2use=chans,
                        fsamp=self._fsamp,
                    )
                    self._muap_data[port_name] = {
                        "muaps": muaps,
                        "channel_indices": chans,
                        "grid_config": get_grid_config(etype),
                        "electrode_type": etype,
                    }
                except Exception as e:
                    print(f"  Warning: MUAP extraction failed for {port_name}: {e}")

            ch_offset += n_ch

            n_mus = len(motor_units)
            total_spikes = sum(len(mu.timestamps) for mu in motor_units)
            print(f"  Port {port_name}: {n_mus} MUs, {total_spikes} total spikes")

        # Update UI
        self._refresh_port_combo()
        if len(ports) > 0:
            self.port_combo.setCurrentText(ports[0])
            # Force trigger even if already selected (single-port case)
            self._on_port_changed(ports[0])

    @staticmethod
    def _ensure_list_of_arrays(data) -> list:
        """
        Normalize various formats into a list of arrays.
        Handles: list of arrays, single array (1D or 2D), empty.
        """
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            return []

        # Already a list
        if isinstance(data, list):
            if len(data) == 0:
                return []
            # Check if it's a list of arrays/lists
            first = data[0]
            if isinstance(first, (np.ndarray, list)) or hasattr(first, 'detach'):
                return [to_numpy(x) for x in data]
            # It might be a flat list of numbers → single array
            return [to_numpy(data)]

        # Numpy array
        arr = to_numpy(data)
        if arr.ndim == 0:
            return []
        if arr.ndim == 1:
            if arr.size == 0:
                return []
            return [arr]
        if arr.ndim >= 2:
            return [arr[i] for i in range(arr.shape[0])]

        return []

    # --------------------------------------------------------
    # File I/O
    # --------------------------------------------------------

    def _load_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Decomposition", "", "Pickle (*.pkl);;All (*)"
        )
        if not path:
            return
        self.load_from_path(Path(path))

    def _save_file(self):
        if not self._ports:
            self._update_status("Nothing to save")
            return

        default_name = ""
        if self._loaded_path:
            default_name = str(self._loaded_path.with_name(
                self._loaded_path.stem + "_edited.pkl"
            ))

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Decomposition", default_name, "Pickle (*.pkl)"
        )
        if not path:
            return

        try:
            save_data = self._build_save_dict()
            with open(path, "wb") as f:
                pickle.dump(save_data, f)
            self._update_status(f"Saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _build_save_dict(self) -> dict:
        ports = list(self._ports.keys())
        discharge_times = []
        pulse_trains = []
        mu_filters = []

        for port_name in ports:
            mus = self._ports[port_name]
            kept = [mu for mu in mus if not mu.flagged_duplicate]

            port_ts = [mu.timestamps for mu in kept]
            port_src = [mu.source for mu in kept]
            port_filt = [mu.mu_filter for mu in kept if mu.mu_filter is not None]

            discharge_times.append(port_ts)
            pulse_trains.append(port_src)
            mu_filters.append(port_filt if port_filt else None)

        save_data = {
            "ports": ports,
            "sampling_rate": self._fsamp,
            "discharge_times": discharge_times,
            "pulse_trains": pulse_trains,
            "mu_filters": mu_filters,
        }

        if self._emg_data:
            save_data["emg_per_port"] = {
                name: data for name, data in self._emg_data.items()
            }

        return save_data
    # --------------------------------------------------------
    # Mode switching
    # --------------------------------------------------------

    def _set_mode(self, mode: EditMode):
        self._edit_mode = mode
        self.source_plot.set_edit_mode(mode)
        {
            EditMode.VIEW: self.action_view,
            EditMode.ADD: self.action_add,
            EditMode.DELETE: self.action_delete,
        }[mode].setChecked(True)
        self._update_status()

    # --------------------------------------------------------
    # Single spike add / delete
    # --------------------------------------------------------

    def _handle_add_click(self, sample: int):
        if self._edit_mode != EditMode.ADD:
            return
        mu = self._current_mu()
        if mu is None:
            return

        # Clamp to valid range
        if sample < 0 or sample >= len(mu.source):
            self._update_status("Click outside source range")
            return

        peak = self._find_nearest_peak(mu.source, sample)
        if peak is None:
            self._update_status("No peak found near click")
            return

        if peak in mu.timestamps:
            self._update_status("Spike already exists at this location")
            return

        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.append(mu.timestamps, peak)).astype(np.int64)

        self._push_undo(UndoAction("Add spike", self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Added spike at {peak / self._fsamp:.3f}s")

    def _handle_delete_click(self, sample: int):
        if self._edit_mode != EditMode.DELETE:
            return
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) == 0:
            return

        tolerance = int(0.01 * self._fsamp)
        distances = np.abs(mu.timestamps - sample)
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] > tolerance:
            self._update_status("No spike near click (zoom in or click closer)")
            return

        old_ts = mu.timestamps.copy()
        removed_sample = mu.timestamps[nearest_idx]
        new_ts = np.delete(mu.timestamps, nearest_idx)

        self._push_undo(UndoAction("Delete spike", self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted spike at {removed_sample / self._fsamp:.3f}s")

    def _find_nearest_peak(self, source: np.ndarray, click_sample: int) -> Optional[int]:
        window = int(0.01 * self._fsamp)
        start = max(0, click_sample - window)
        end = min(len(source), click_sample + window)
        if start >= end:
            return None
        segment = source[start:end]
        peaks, _ = sp_signal.find_peaks(segment, distance=max(1, int(0.005 * self._fsamp)))
        if len(peaks) == 0:
            local_max = np.argmax(segment)
            return int(start + local_max)
        peaks_abs = peaks + start
        return int(peaks_abs[np.argmin(np.abs(peaks_abs - click_sample))])

    # --------------------------------------------------------
    # ROI workflow
    # --------------------------------------------------------

    def _place_roi(self):
        if self._current_mu() is None:
            self._update_status("Select a motor unit first")
            return
        self.source_plot.place_roi()
        self.action_roi_add.setEnabled(True)
        self.action_roi_delete.setEnabled(True)
        self.action_roi_clear.setEnabled(True)
        self._update_status("ROI placed — drag edges, then Add/Delete in ROI")

    def _clear_roi(self):
        self.source_plot.remove_roi()
        self.action_roi_add.setEnabled(False)
        self.action_roi_delete.setEnabled(False)
        self.action_roi_clear.setEnabled(False)
        self._update_status("ROI cleared")

    def _roi_add_spikes(self):
        mu = self._current_mu()
        bounds = self.source_plot.get_roi_bounds()
        if mu is None or bounds is None:
            return

        x1_sec, x2_sec, y1, y2 = bounds
        s1 = max(0, int(x1_sec * self._fsamp))
        s2 = min(len(mu.source), int(x2_sec * self._fsamp))
        if s1 >= s2:
            return

        segment = mu.source[s1:s2]
        peaks, _ = sp_signal.find_peaks(segment, distance=max(1, int(0.005 * self._fsamp)))
        peaks_abs = peaks + s1

        new_spikes = []
        existing = set(mu.timestamps.tolist())
        for p in peaks_abs:
            if 0 <= p < len(mu.source) and y1 <= mu.source[p] <= y2 and int(p) not in existing:
                new_spikes.append(int(p))

        if not new_spikes:
            self._update_status("No new peaks found in ROI")
            return

        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.concatenate([mu.timestamps, np.array(new_spikes, dtype=np.int64)]))
        self._push_undo(UndoAction(f"ROI add {len(new_spikes)} spikes", self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Added {len(new_spikes)} spikes from ROI")

    def _roi_delete_spikes(self):
        mu = self._current_mu()
        bounds = self.source_plot.get_roi_bounds()
        if mu is None or bounds is None:
            return

        x1_sec, x2_sec, y1, y2 = bounds
        s1 = int(x1_sec * self._fsamp)
        s2 = int(x2_sec * self._fsamp)

        in_box = np.zeros(len(mu.timestamps), dtype=bool)
        for i, ts in enumerate(mu.timestamps):
            if s1 <= ts < s2 and 0 <= ts < len(mu.source):
                if y1 <= mu.source[ts] <= y2:
                    in_box[i] = True

        n_remove = np.sum(in_box)
        if n_remove == 0:
            self._update_status("No spikes found in ROI")
            return

        old_ts = mu.timestamps.copy()
        new_ts = mu.timestamps[~in_box]
        self._push_undo(UndoAction(f"ROI delete {n_remove} spikes", self._current_mu_idx, old_ts, new_ts))
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted {n_remove} spikes from ROI")

    # --------------------------------------------------------
    # Undo / Redo
    # --------------------------------------------------------

    def _push_undo(self, action: UndoAction):
        self._undo_stack.append(action)
        self._redo_stack.clear()
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            self._update_status("Nothing to undo")
            return
        action = self._undo_stack.pop()
        self._redo_stack.append(action)
        mu = self._get_mu(self._current_port, action.mu_idx)
        if mu is not None:
            mu.timestamps = action.old_timestamps
        self._on_data_changed(f"Undo: {action.description}")

    def _redo(self):
        if not self._redo_stack:
            self._update_status("Nothing to redo")
            return
        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        mu = self._get_mu(self._current_port, action.mu_idx)
        if mu is not None:
            mu.timestamps = action.new_timestamps
        self._on_data_changed(f"Redo: {action.description}")

    # --------------------------------------------------------
    # MU Selection & Navigation
    # --------------------------------------------------------

    def _current_mu(self) -> Optional[MotorUnit]:
        return self._get_mu(self._current_port, self._current_mu_idx)

    def _get_mu(self, port: Optional[str], idx: int) -> Optional[MotorUnit]:
        if port is None or idx < 0:
            return None
        mus = self._ports.get(port, [])
        return mus[idx] if 0 <= idx < len(mus) else None

    def _on_port_changed(self, port_name: str):
        if not port_name or port_name not in self._ports:
            return
        self._current_port = port_name
        self._current_mu_idx = -1
        self._refresh_mu_list()
        self._clear_plots()
        if self.mu_list.count() > 0:
            self.mu_list.setCurrentRow(0)

    def _on_mu_selected(self, row: int):
        if row < 0:
            self._current_mu_idx = -1
            self._clear_plots()
            return
        self._current_mu_idx = row
        mu = self._current_mu()
        if mu is not None:
            self.btn_flag_delete.setText("Unflag" if mu.flagged_duplicate else "🗑 Flag to Delete")
        self._update_plots()
        self._update_status()

    def _select_prev_mu(self):
        r = self.mu_list.currentRow()
        if r > 0:
            self.mu_list.setCurrentRow(r - 1)

    def _select_next_mu(self):
        r = self.mu_list.currentRow()
        if r < self.mu_list.count() - 1:
            self.mu_list.setCurrentRow(r + 1)

    # --------------------------------------------------------
    # Unit controls
    # --------------------------------------------------------

    def _toggle_flag_delete(self):
        mu = self._current_mu()
        if mu is None:
            return
        mu.flagged_duplicate = not mu.flagged_duplicate
        self._refresh_mu_list()
        self.mu_list.setCurrentRow(self._current_mu_idx)
        status = "flagged for deletion" if mu.flagged_duplicate else "unflagged"
        self._update_status(f"MU {mu.id} {status}")

    # --------------------------------------------------------
    # UI Refresh
    # --------------------------------------------------------

    def _refresh_port_combo(self):
        cur = self.port_combo.currentText()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for name in self._ports:
            self.port_combo.addItem(name)
        if cur in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
            self.port_combo.setCurrentText(cur)
        self.port_combo.blockSignals(False)

    def _refresh_mu_list(self):
        self.mu_list.blockSignals(True)
        self.mu_list.clear()
        mus = self._ports.get(self._current_port, [])
        for mu in mus:
            fr = compute_firing_rate(mu.timestamps, self._fsamp)
            text = f"MU {mu.id}  ({len(mu.timestamps)} spikes, {fr:.1f} Hz)"
            if mu.flagged_duplicate:
                text = f"🗑 {text}  [DELETE]"
            item = QListWidgetItem(text)
            if mu.flagged_duplicate:
                item.setForeground(QColor(COLORS['error']))
            self.mu_list.addItem(item)
        self.mu_list.blockSignals(False)

    def _update_plots(self):
        mu = self._current_mu()
        if mu is None:
            self._clear_plots()
            return
        self.source_plot.set_data(mu.source, mu.timestamps)
        self.fr_plot.set_data(mu.timestamps)
        self._plot_muap()
        self._update_quality(mu)

    def _clear_plots(self):
        self.source_plot.clear_data()
        self.fr_plot.clear_data()
        self._clear_muap_plot()
        self.quality_bar.clear_metrics()

    def _on_data_changed(self, msg: str = "Modified"):
        self._update_plots()
        self._refresh_mu_list()
        self.mu_list.setCurrentRow(self._current_mu_idx)
        self._update_status(msg)
        self.data_modified.emit()

    def _update_quality(self, mu: MotorUnit):
        sil = compute_sil(mu.source, mu.timestamps)
        cov = compute_cov(mu.timestamps, self._fsamp)
        fr = compute_firing_rate(mu.timestamps, self._fsamp)
        self.quality_bar.set_metrics(sil, cov, fr, len(mu.timestamps))

    def _update_status(self, msg: str = None):
        if msg:
            self.status_bar.showMessage(msg, 4000)
        else:
            parts = []
            if self._current_port:
                parts.append(f"Port: {self._current_port}")
            n_mus = len(self._ports.get(self._current_port, []))
            if n_mus > 0:
                parts.append(f"{n_mus} MUs")
            if self._current_mu_idx >= 0:
                parts.append(f"MU: {self._current_mu_idx}")
            mode_hints = {
                EditMode.VIEW: "View [V]",
                EditMode.ADD: "Add Spike — click on source [A]",
                EditMode.DELETE: "Delete Spike — click near spike [D]",
            }
            parts.append(mode_hints[self._edit_mode])
            if self.source_plot.has_roi():
                parts.append("ROI active")
            if self._undo_stack:
                parts.append(f"Undo: {len(self._undo_stack)}")
            self.status_bar.showMessage("  |  ".join(parts))

    # --------------------------------------------------------
    # MUAP Visualization
    # --------------------------------------------------------

    def _plot_muap(self):
        port_muaps = self._muap_data.get(self._current_port)
        if port_muaps is None or self._current_mu_idx < 0:
            self._clear_muap_plot()
            return

        muaps = port_muaps["muaps"]
        if self._current_mu_idx >= len(muaps):
            self._clear_muap_plot()
            return

        waveforms = muaps[self._current_mu_idx]
        ch_indices = port_muaps["channel_indices"]
        grid_cfg = port_muaps.get("grid_config")

        if not waveforms:
            self._clear_muap_plot()
            return

        if grid_cfg is not None:
            self._render_muap_grid(waveforms, ch_indices, grid_cfg)
        else:
            self._render_muap_stacked(waveforms, ch_indices)

    def _render_muap_grid(self, waveforms, ch_indices, grid_cfg):
        self.muap_widget.clear()
        rows, cols = grid_cfg["grid_shape"]
        positions = grid_cfg["positions"]

        valid = [w for w in waveforms if len(w) > 0]
        amp = np.max(np.abs(np.concatenate(valid))) * 1.2 if valid else 1.0

        etype = self._muap_data[self._current_port].get("electrode_type", "Grid")
        self.muap_widget.addLabel(
            f"<span style='color:{_PG_FOREGROUND}; font-size:10pt;'>{etype} — MU {self._current_mu_idx}</span>",
            row=0, col=0, colspan=cols,
        )

        for r in range(rows):
            for c in range(cols):
                p = self.muap_widget.addPlot(row=r + 1, col=c)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.setYRange(-amp, amp)
                p.getViewBox().setBorder(pg.mkPen(color=(50, 50, 50), width=1))

        for plot_idx, wav in enumerate(waveforms):
            if plot_idx >= len(ch_indices):
                break
            local_ch = int(plot_idx)
            grid_pos = positions.get(local_ch)
            if grid_pos is None:
                continue
            r, c = grid_pos
            if r >= rows or c >= cols:
                continue
            item = self.muap_widget.getItem(r + 1, c)
            if item is not None and len(wav) > 0:
                item.plot(wav, pen=pg.mkPen(color=_PG_ACCENT, width=1.5))

    def _render_muap_stacked(self, waveforms, ch_indices):
        self.muap_widget.clear()
        plot = self.muap_widget.addPlot(row=0, col=0)

        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return

        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)

        for rank, (plot_idx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch_label = int(ch_indices[plot_idx]) if plot_idx < len(ch_indices) else plot_idx
            plot.plot(wav + offset, pen=pg.mkPen(_PG_FOREGROUND, width=1.5))
            txt = pg.TextItem(f"Ch {ch_label}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 7))
            plot.addItem(txt)

        plot.getAxis("left").setVisible(False)
        plot.setTitle(f"MU {self._current_mu_idx} — Stacked", color=_PG_FOREGROUND, size="10pt")

    def _clear_muap_plot(self):
        self.muap_widget.clear()
        p = self.muap_widget.addPlot(row=0, col=0)
        t = pg.TextItem("Select a Motor Unit", color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        p.addItem(t)
        p.hideAxis("left")
        p.hideAxis("bottom")