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

from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint, QSize, QEvent
from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QToolBar,
    QAction,
    QActionGroup,
    QComboBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QShortcut,
    QStatusBar,
    QRubberBand,
    QApplication,
)
from PyQt5.QtGui import QKeySequence, QFont, QPixmap, QIcon, QPainter, QColor
import pyqtgraph as pg

from scd_app.gui.style.styling import (
    COLORS,
    FONT_SIZES,
    SPACING,
    FONT_FAMILY,
    get_section_header_style,
    get_label_style,
    get_button_style,
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
    recalculate_unit_centroid,
    supports_filter_recalculation,
    supports_full_source_computation,
    compute_all_full_sources,
)
from scd_app.core.auto_editor import auto_edit, AutoEditResult, MIN_SPIKES


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

GRID_POSITIONS_13x5 = {
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (4, 0),
    5: (5, 0),
    6: (6, 0),
    7: (7, 0),
    8: (8, 0),
    9: (9, 0),
    10: (10, 0),
    11: (11, 0),
    12: (12, 0),
    25: (0, 1),
    24: (1, 1),
    23: (2, 1),
    22: (3, 1),
    21: (4, 1),
    20: (5, 1),
    19: (6, 1),
    18: (7, 1),
    17: (8, 1),
    16: (9, 1),
    15: (10, 1),
    14: (11, 1),
    13: (12, 1),
    26: (0, 2),
    27: (1, 2),
    28: (2, 2),
    29: (3, 2),
    30: (4, 2),
    31: (5, 2),
    32: (6, 2),
    33: (7, 2),
    34: (8, 2),
    35: (9, 2),
    36: (10, 2),
    37: (11, 2),
    38: (12, 2),
    51: (0, 3),
    50: (1, 3),
    49: (2, 3),
    48: (3, 3),
    47: (4, 3),
    46: (5, 3),
    45: (6, 3),
    44: (7, 3),
    43: (8, 3),
    42: (9, 3),
    41: (10, 3),
    40: (11, 3),
    39: (12, 3),
    52: (0, 4),
    53: (1, 4),
    54: (2, 4),
    55: (3, 4),
    56: (4, 4),
    57: (5, 4),
    58: (6, 4),
    59: (7, 4),
    60: (8, 4),
    61: (9, 4),
    62: (10, 4),
    63: (11, 4),
    64: (12, 4),
}
GRID_POSITIONS_8x8 = {
    8: (0, 0),
    7: (1, 0),
    6: (2, 0),
    5: (3, 0),
    4: (4, 0),
    3: (5, 0),
    2: (6, 0),
    1: (7, 0),
    16: (0, 1),
    15: (1, 1),
    14: (2, 1),
    13: (3, 1),
    12: (4, 1),
    11: (5, 1),
    10: (6, 1),
    9: (7, 1),
    24: (0, 2),
    23: (1, 2),
    22: (2, 2),
    21: (3, 2),
    20: (4, 2),
    19: (5, 2),
    18: (6, 2),
    17: (7, 2),
    32: (0, 3),
    31: (1, 3),
    30: (2, 3),
    29: (3, 3),
    28: (4, 3),
    27: (5, 3),
    26: (6, 3),
    25: (7, 3),
    40: (0, 4),
    39: (1, 4),
    38: (2, 4),
    37: (3, 4),
    36: (4, 4),
    35: (5, 4),
    34: (6, 4),
    33: (7, 4),
    48: (0, 5),
    47: (1, 5),
    46: (2, 5),
    45: (3, 5),
    44: (4, 5),
    43: (5, 5),
    42: (6, 5),
    41: (7, 5),
    56: (0, 6),
    55: (1, 6),
    54: (2, 6),
    53: (3, 6),
    52: (4, 6),
    51: (5, 6),
    50: (6, 6),
    49: (7, 6),
    64: (0, 7),
    63: (1, 7),
    62: (2, 7),
    61: (3, 7),
    60: (4, 7),
    59: (5, 7),
    58: (6, 7),
    57: (7, 7),
}
GRID_POSITIONS_20x2 = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (4, 0),
    5: (5, 0),
    6: (6, 0),
    7: (7, 0),
    8: (8, 0),
    9: (9, 0),
    10: (10, 0),
    11: (11, 0),
    12: (12, 0),
    13: (13, 0),
    14: (14, 0),
    15: (15, 0),
    16: (16, 0),
    17: (17, 0),
    18: (18, 0),
    19: (19, 0),
    20: (0, 1),
    21: (1, 1),
    22: (2, 1),
    23: (3, 1),
    24: (4, 1),
    25: (5, 1),
    26: (6, 1),
    27: (7, 1),
    28: (8, 1),
    29: (9, 1),
    30: (10, 1),
    31: (11, 1),
    32: (12, 1),
    33: (13, 1),
    34: (14, 1),
    35: (15, 1),
    36: (16, 1),
    37: (17, 1),
    38: (18, 1),
    39: (19, 1),
}

GRID_POSITIONS_HD02MM0808 = {
    # Col 0
    53: (0, 0),
    54: (0, 1),
    55: (0, 2),
    56: (0, 3),
    64: (0, 4),
    63: (0, 5),
    62: (0, 6),
    61: (0, 7),
    52: (1, 0),
    51: (1, 1),
    50: (1, 2),
    49: (1, 3),
    60: (1, 4),
    57: (1, 5),
    58: (1, 6),
    59: (1, 7),
    48: (2, 0),
    47: (2, 1),
    46: (2, 2),
    45: (2, 3),
    33: (2, 4),
    34: (2, 5),
    35: (2, 6),
    36: (2, 7),
    44: (3, 0),
    43: (3, 1),
    42: (3, 2),
    41: (3, 3),
    37: (3, 4),
    38: (3, 5),
    39: (3, 6),
    40: (3, 7),
    32: (4, 0),
    31: (4, 1),
    30: (4, 2),
    29: (4, 3),
    28: (4, 4),
    27: (4, 5),
    26: (4, 6),
    25: (4, 7),
    24: (5, 0),
    23: (5, 1),
    22: (5, 2),
    21: (5, 3),
    20: (5, 4),
    19: (5, 5),
    18: (5, 6),
    17: (5, 7),
    1: (6, 0),
    2: (6, 1),
    3: (6, 2),
    4: (6, 3),
    5: (6, 4),
    6: (6, 5),
    7: (6, 6),
    8: (6, 7),
    16: (7, 0),
    15: (7, 1),
    14: (7, 2),
    13: (7, 3),
    12: (7, 4),
    11: (7, 5),
    10: (7, 6),
    9: (7, 7),
}

GRID_POSITIONS_HD04MM1305 = {
    # Col 0
    52: (0, 0),
    53: (0, 1),
    54: (0, 2),
    55: (0, 3),
    56: (0, 4),
    57: (0, 5),
    58: (0, 6),
    59: (0, 7),
    60: (0, 8),
    61: (0, 9),
    62: (0, 10),
    63: (0, 11),
    64: (0, 12),
    39: (1, 0),
    40: (1, 1),
    41: (1, 2),
    42: (1, 3),
    43: (1, 4),
    44: (1, 5),
    45: (1, 6),
    46: (1, 7),
    47: (1, 8),
    48: (1, 9),
    49: (1, 10),
    50: (1, 11),
    51: (1, 12),
    26: (2, 0),
    27: (2, 1),
    28: (2, 2),
    29: (2, 3),
    30: (2, 4),
    31: (2, 5),
    32: (2, 6),
    33: (2, 7),
    34: (2, 8),
    35: (2, 9),
    36: (2, 10),
    37: (2, 11),
    38: (2, 12),
    13: (3, 0),
    14: (3, 1),
    15: (3, 2),
    16: (3, 3),
    17: (3, 4),
    18: (3, 5),
    19: (3, 6),
    20: (3, 7),
    21: (3, 8),
    22: (3, 9),
    23: (3, 10),
    24: (3, 11),
    25: (3, 12),
    1: (4, 1),
    2: (4, 2),
    3: (4, 3),
    4: (4, 4),
    5: (4, 5),
    6: (4, 6),
    7: (4, 7),
    8: (4, 8),
    9: (4, 9),
    10: (4, 10),
    11: (4, 11),
    12: (4, 12),
}

GRID_POSITIONS_HD04MM1606 = {
    # Col 0: ch 81–96 (rows 0–15)
    81: (0, 0),
    82: (1, 0),
    83: (2, 0),
    84: (3, 0),
    85: (4, 0),
    86: (5, 0),
    87: (6, 0),
    88: (7, 0),
    89: (8, 0),
    90: (9, 0),
    91: (10, 0),
    92: (11, 0),
    93: (12, 0),
    94: (13, 0),
    95: (14, 0),
    96: (15, 0),
    # Col 1: ch 65–80
    65: (0, 1),
    66: (1, 1),
    67: (2, 1),
    68: (3, 1),
    69: (4, 1),
    70: (5, 1),
    71: (6, 1),
    72: (7, 1),
    73: (8, 1),
    74: (9, 1),
    75: (10, 1),
    76: (11, 1),
    77: (12, 1),
    78: (13, 1),
    79: (14, 1),
    80: (15, 1),
    # Col 2: ch 49–64
    49: (0, 2),
    50: (1, 2),
    51: (2, 2),
    52: (3, 2),
    53: (4, 2),
    54: (5, 2),
    55: (6, 2),
    56: (7, 2),
    57: (8, 2),
    58: (9, 2),
    59: (10, 2),
    60: (11, 2),
    61: (12, 2),
    62: (13, 2),
    63: (14, 2),
    64: (15, 2),
    # Col 3: ch 33–48
    33: (0, 3),
    34: (1, 3),
    35: (2, 3),
    36: (3, 3),
    37: (4, 3),
    38: (5, 3),
    39: (6, 3),
    40: (7, 3),
    41: (8, 3),
    42: (9, 3),
    43: (10, 3),
    44: (11, 3),
    45: (12, 3),
    46: (13, 3),
    47: (14, 3),
    48: (15, 3),
    # Col 4: ch 17–32
    17: (0, 4),
    18: (1, 4),
    19: (2, 4),
    20: (3, 4),
    21: (4, 4),
    22: (5, 4),
    23: (6, 4),
    24: (7, 4),
    25: (8, 4),
    26: (9, 4),
    27: (10, 4),
    28: (11, 4),
    29: (12, 4),
    30: (13, 4),
    31: (14, 4),
    32: (15, 4),
    # Col 5: ch 1–16
    1: (0, 5),
    2: (1, 5),
    3: (2, 5),
    4: (3, 5),
    5: (4, 5),
    6: (5, 5),
    7: (6, 5),
    8: (7, 5),
    9: (8, 5),
    10: (9, 5),
    11: (10, 5),
    12: (11, 5),
    13: (12, 5),
    14: (13, 5),
    15: (14, 5),
    16: (15, 5),
}

# HD10MM0804 / HD05MM0804: 8 rows × 4 cols = 32 channels
# Sequential layout (NOT serpentine): channels increase monotonically
# top-to-bottom within each physical column, columns left-to-right.
# grid_shape=(8,4) → positions as (row, col), matching the (rows,cols) convention
# used by GR08MM1305 and GR10MM0808.
GRID_POSITIONS_8x4 = {
    1: (0, 0),
    2: (1, 0),
    3: (2, 0),
    4: (3, 0),
    5: (4, 0),
    6: (5, 0),
    7: (6, 0),
    8: (7, 0),
    9: (0, 1),
    10: (1, 1),
    11: (2, 1),
    12: (3, 1),
    13: (4, 1),
    14: (5, 1),
    15: (6, 1),
    16: (7, 1),
    17: (0, 2),
    18: (1, 2),
    19: (2, 2),
    20: (3, 2),
    21: (4, 2),
    22: (5, 2),
    23: (6, 2),
    24: (7, 2),
    25: (0, 3),
    26: (1, 3),
    27: (2, 3),
    28: (3, 3),
    29: (4, 3),
    30: (5, 3),
    31: (6, 3),
    32: (7, 3),
}


ELECTRODE_GRIDS = {
    "GR08MM1305": {
        "grid_shape": (13, 5),
        "ied_mm": 8,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_13x5,
    },
    "GR10MM0808": {
        "grid_shape": (8, 8),
        "ied_mm": 10,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_8x8,
    },
    "Thin-film": {
        "grid_shape": (20, 2),
        "ied_mm": 5,
        "n_channels": 40,
        "muap_mapping": {i: i for i in range(40)},
        "positions": GRID_POSITIONS_20x2,
    },
    "HD02MM0808": {
        "grid_shape": (8, 8),
        "ied_mm": 2,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD02MM0808,
    },
    "HD04MM1305": {
        "grid_shape": (5, 13),
        "ied_mm": 4,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD04MM1305,
    },
    "HD04MM1606": {
        "grid_shape": (16, 6),
        "ied_mm": 4,
        "n_channels": 96,
        "muap_mapping": {i: i + 1 for i in range(96)},
        "positions": GRID_POSITIONS_HD04MM1606,
    },
    "HD08MM1305": {
        "grid_shape": (5, 13),
        "ied_mm": 8,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD04MM1305,
    },
    "HD10MM0804": {
        "grid_shape": (8, 4),
        "ied_mm": 10,
        "n_channels": 32,
        "muap_mapping": {i: i + 1 for i in range(32)},
        "positions": GRID_POSITIONS_8x4,
    },
    "HD05MM0804": {
        "grid_shape": (8, 4),
        "ied_mm": 5,
        "n_channels": 32,
        "muap_mapping": {i: i + 1 for i in range(32)},
        "positions": GRID_POSITIONS_8x4,
    },
}

_MIN_RUBBERBAND_PX = 5  # drags smaller than this are ignored in selection-arm mode


# ---------------------------------------------------------------------------
# Selection arm state enum
# ---------------------------------------------------------------------------


class SelectionArm:
    """Which (if any) rubberband-selection operation is currently armed."""

    NONE = "none"
    ADD = "add"
    DELETE = "delete"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# XZoomViewBox — Shift+scroll zooms X axis only, leaving Y fixed
# ---------------------------------------------------------------------------


class XZoomViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & Qt.ShiftModifier:
            # Shift+scroll → pan horizontally
            delta = ev.delta()
            self.translateBy(x=-delta / 200.0, y=0)
            ev.accept()
        elif mods & Qt.ControlModifier:
            # Ctrl/Cmd+scroll → zoom both axes (pyqtgraph default)
            super().wheelEvent(ev, axis=None)
        else:
            # Plain scroll → zoom X axis only
            super().wheelEvent(ev, axis=0)


# ---------------------------------------------------------------------------
# AUX legend overlay
# ---------------------------------------------------------------------------

_AUX_COLORS_HEX = ["#FFD700", "#C0EFFF", "#FFB347"]
_AUX_COLORS_RGB = [(255, 215, 0), (192, 239, 255), (255, 179, 71)]


class _AuxLegend(pg.LegendItem):
    """Floating click-to-toggle legend for AUX force traces.

    Labels are added directly to the grid layout (no ItemSample swatch).
    Toggling uses pen-alpha (70 → 5) rather than setVisible to avoid the
    pyqtgraph 'hidden' eye-slash glyph.
    """

    def __init__(self):
        super().__init__(offset=(-10, 10))  # top-right corner
        self._curves: list = []
        self._names: list = []
        self._colors: list = []
        self._on_states: list = []

    def clear(self):
        for _sample, label in self.items:
            self.layout.removeItem(label)
            label.close()
        self.items = []
        self.updateSize()

    def populate(self, channels: list, curves: list):
        self.clear()
        self._curves = list(curves)
        self._names = []
        self._colors = []
        self._on_states = [True] * len(channels)
        for i, (ch, _curve) in enumerate(zip(channels, curves)):
            meta = ch.get("meta", {})
            name = meta.get("name", meta.get("unit", f"AUX {i + 1}"))
            color = _AUX_COLORS_HEX[i % len(_AUX_COLORS_HEX)]
            self._names.append(name)
            self._colors.append(color)
            label = pg.LabelItem(f"● {name}", color=color, justify="left")
            self.layout.addItem(label, i, 0)
            self.items.append((None, label))
        self.updateSize()
        self.setVisible(bool(channels))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            y = event.pos().y()
            for i, (_sample, label) in enumerate(self.items):
                rect = label.mapRectToParent(label.boundingRect())
                if rect.top() <= y <= rect.bottom():
                    self._toggle(i)
                    event.accept()
                    return
        event.accept()  # consume all clicks; legend is not draggable

    def _toggle(self, idx: int):
        self._on_states[idx] = not self._on_states[idx]
        on = self._on_states[idx]
        r, g, b = _AUX_COLORS_RGB[idx % len(_AUX_COLORS_RGB)]
        alpha = 70 if on else 5
        self._curves[idx].setPen(pg.mkPen(color=(r, g, b, alpha), width=2.5))
        _sample, label = self.items[idx]
        dot = "●" if on else "○"
        color = self._colors[idx] if on else "#555555"
        label.setText(f"{dot} {self._names[idx]}", color=color)
        self.update()


# ---------------------------------------------------------------------------
# SourcePlotWidget
# ---------------------------------------------------------------------------


class SourcePlotWidget(pg.PlotWidget):
    """
    Source-signal plot with two independent interaction modes:

    1. Point-click editing  (EditMode.ADD / DELETE via set_edit_mode)
       Click  →  spike_add_requested / spike_delete_requested signal.

    2. Rubberband-drag selection  (armed via set_selection_arm)
       Drag  →  region_selected(x1, x2, y1, y2) fires on mouse-release.
       Stays armed until explicitly disarmed.  Each drag fires immediately.

    The two modes coexist independently.
    """

    spike_add_requested = pyqtSignal(int)  # sample index
    spike_delete_requested = pyqtSignal(int)  # sample index
    region_selected = pyqtSignal(float, float, float, float)  # x1,x2,y1,y2 (data)

    def __init__(self, parent=None):
        super().__init__(
            parent, background=COLORS["background"], viewBox=XZoomViewBox()
        )
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left", "Amplitude", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))

        self._fsamp = 1.0
        self._edit_mode = EditMode.VIEW
        self._sel_arm = SelectionArm.NONE

        self._source: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None

        self._signal_curve = self.plot([], pen=pg.mkPen("#2b6cb0", width=1))
        self._spike_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("#ed8936"),
            symbol="o",
            hoverable=True,
        )
        self.addItem(self._spike_scatter)
        self._plateau_region: Optional[pg.LinearRegionItem] = None

        self._aux_curves: list = []
        self._aux_raw: list = []
        self._legend = _AuxLegend()
        self._legend.setParentItem(self.plotItem.vb)

        # Rubberband overlay (pixel space, parented to this widget)
        self._rb_widget: Optional[QRubberBand] = None
        self._rb_origin: Optional[QPoint] = None

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def set_edit_mode(self, mode: EditMode):
        """Change the point-click edit mode. Does NOT affect selection arm."""
        self._edit_mode = mode
        self._apply_cursor()

    def set_selection_arm(self, arm: str):
        """
        Arm (SelectionArm.ADD / DELETE) or disarm (SelectionArm.NONE) the
        rubberband-drag operation.
        """
        self._sel_arm = arm
        self._apply_cursor()
        if arm == SelectionArm.NONE:
            self._cancel_rubberband()

    def _apply_cursor(self):
        """Selection arm takes cursor priority over edit mode."""
        if self._sel_arm != SelectionArm.NONE:
            self.setCursor(Qt.CrossCursor)
        else:
            cursors = {
                EditMode.VIEW: Qt.ArrowCursor,
                EditMode.ADD: Qt.CrossCursor,
                EditMode.DELETE: Qt.PointingHandCursor,
            }
            self.setCursor(cursors.get(self._edit_mode, Qt.ArrowCursor))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_data(self, source: np.ndarray, timestamps: np.ndarray):
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
        self._source = source**2
        self._timestamps = timestamps
        t = np.arange(len(self._source)) / self._fsamp
        self._signal_curve.setData(t, self._source)
        self._update_spike_markers()
        self._redraw_aux()

    def set_plateau_region(self, start_sample: int, end_sample: int):
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
        t0 = start_sample / self._fsamp
        t1 = end_sample / self._fsamp
        self._plateau_region = pg.LinearRegionItem(
            values=(t0, t1),
            movable=False,
            brush=pg.mkBrush(137, 180, 250, 20),
            pen=pg.mkPen(color=(137, 180, 250, 60), width=1, style=Qt.DashLine),
        )
        self._plateau_region.setZValue(-10)
        self.addItem(self._plateau_region)

    def set_aux_data(self, channels: list, fsamp: float):
        for curve in self._aux_curves:
            self.removeItem(curve)
        self._aux_curves.clear()
        self._aux_raw.clear()
        for i, ch in enumerate(channels):
            raw = np.asarray(ch["data"]).squeeze()
            raw = np.nan_to_num(raw, nan=0.0)
            self._aux_raw.append(raw)
            r, g, b = _AUX_COLORS_RGB[i % len(_AUX_COLORS_RGB)]
            curve = pg.PlotCurveItem(pen=pg.mkPen(color=(r, g, b, 70), width=2.5))
            curve.setZValue(-20)
            self.addItem(curve)
            self._aux_curves.append(curve)
        self._legend.populate(channels, self._aux_curves)
        self._redraw_aux()

    def _redraw_aux(self):
        if self._source is not None and len(self._source) > 0:
            src_min = float(self._source.min())
            src_max = float(self._source.max())
        else:
            src_min = 0.0
            src_max = 1.0
        # Map force onto the source amplitude range:
        #   force 0   → src_min  (noise floor)
        #   force max → src_max * 1.05  (5% above the tallest spike)
        src_range = max(src_max * 1.05 - src_min, 1e-9)
        n_src = len(self._source) if self._source is not None else 0
        for raw, curve in zip(self._aux_raw, self._aux_curves):
            sig = raw - float(raw.min())
            sig_range = max(float(sig.max()), 1e-9)
            sig_scaled = (sig / sig_range) * src_range + src_min
            n_plot = min(len(sig_scaled), n_src) if n_src > 0 else len(sig_scaled)
            step = max(1, n_plot // 2000)
            t = np.arange(0, n_plot, step) / self._fsamp
            curve.setData(t, sig_scaled[:n_plot:step])

    def update_timestamps(self, timestamps: np.ndarray):
        """Refresh spike markers without touching the signal curve or view range."""
        self._timestamps = timestamps
        self._update_spike_markers()

    def clear_data(self):
        self._signal_curve.setData([], [])
        self._spike_scatter.setData([], [])
        for curve in self._aux_curves:
            curve.setData([], [])
        self._source = None
        self._timestamps = None
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
            self._plateau_region = None
        self._cancel_rubberband()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            super().mousePressEvent(ev)
            return

        if self._sel_arm != SelectionArm.NONE:
            # ── Begin rubberband drag ─────────────────────────────────
            self._rb_origin = ev.pos()
            if self._rb_widget is None:
                self._rb_widget = QRubberBand(QRubberBand.Rectangle, self)
            self._rb_widget.setGeometry(QRect(self._rb_origin, QSize()))
            self._rb_widget.show()
            ev.accept()
        elif self._edit_mode in (EditMode.ADD, EditMode.DELETE):
            # ── Track origin for point-click vs. accidental micro-drag ─
            self._rb_origin = ev.pos()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if (
            self._rb_widget is not None
            and self._rb_widget.isVisible()
            and self._rb_origin is not None
        ):
            self._rb_widget.setGeometry(QRect(self._rb_origin, ev.pos()).normalized())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._rb_origin is None:
            super().mouseReleaseEvent(ev)
            return

        origin = self._rb_origin
        self._rb_origin = None

        # ── Selection-arm mode ────────────────────────────────────────
        if self._sel_arm != SelectionArm.NONE and self._rb_widget is not None:
            rect_px = QRect(origin, ev.pos()).normalized()
            self._rb_widget.hide()

            if (
                rect_px.width() > _MIN_RUBBERBAND_PX
                and rect_px.height() > _MIN_RUBBERBAND_PX
            ):
                vb = self.getViewBox()
                tl = vb.mapSceneToView(self.mapToScene(rect_px.topLeft()))
                br = vb.mapSceneToView(self.mapToScene(rect_px.bottomRight()))
                x1, x2 = sorted([tl.x(), br.x()])
                y1, y2 = sorted([tl.y(), br.y()])
                self.region_selected.emit(x1, x2, y1, y2)
            # Small drag in arm mode → silently ignore (no point-click fallthrough)
            ev.accept()
            return

        # ── Point-click mode ──────────────────────────────────────────
        if self._edit_mode in (EditMode.ADD, EditMode.DELETE):
            rect_px = QRect(origin, ev.pos()).normalized()
            if (
                rect_px.width() <= _MIN_RUBBERBAND_PX
                and rect_px.height() <= _MIN_RUBBERBAND_PX
            ):
                vb = self.getViewBox()
                pos = vb.mapSceneToView(self.mapToScene(ev.pos()))
                sample = int(pos.x() * self._fsamp)
                if self._edit_mode == EditMode.ADD:
                    self.spike_add_requested.emit(sample)
                elif self._edit_mode == EditMode.DELETE:
                    self.spike_delete_requested.emit(sample)
            ev.accept()
            return

        super().mouseReleaseEvent(ev)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_rubberband(self):
        self._rb_origin = None
        if self._rb_widget is not None:
            self._rb_widget.hide()

    def _update_spike_markers(self):
        if (
            self._source is None
            or self._timestamps is None
            or len(self._timestamps) == 0
        ):
            self._spike_scatter.setData([], [])
            return
        valid = self._timestamps[self._timestamps < len(self._source)]
        if len(valid) == 0:
            self._spike_scatter.setData([], [])
            return
        self._spike_scatter.setData(valid / self._fsamp, self._source[valid])


# ---------------------------------------------------------------------------
# FiringRatePlotWidget
# ---------------------------------------------------------------------------


class FiringRatePlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS["background"])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left", "IFR (Hz)", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))
        self._curve = self.plot([], pen=pg.mkPen(COLORS["warning"], width=1.5))
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


# ---------------------------------------------------------------------------
# MuapPopoutDialog
# ---------------------------------------------------------------------------


class MuapPopoutDialog(QDialog):
    """Floating window that mirrors the MUAP panel and live-updates with MU selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MUAP Shapes")
        self.resize(850, 620)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self._plot = pg.GraphicsLayoutWidget()
        self._plot.setBackground(COLORS["background"])
        lay.addWidget(self._plot)

    def render_grid(self, waveforms, ch_indices, grid_cfg, mu_idx):
        self._plot.clear()
        rows, cols = grid_cfg["grid_shape"]
        positions = grid_cfg["positions"]
        valid = [w for w in waveforms if len(w) > 0]
        amp = np.max(np.abs(np.concatenate(valid))) * 1.2 if valid else 1.0
        n_samples = len(waveforms[0]) if valid else 409

        label = (
            f"<span style='color:{COLORS['foreground']};font-size:11pt;'>"
            f"MU {mu_idx}</span>"
        )
        self._plot.addLabel(label, row=0, col=0, colspan=rows + 1, justify="center")

        lbl_style = f"color:{COLORS.get('text_dim','#6c7086')}; font-size:8pt;"

        def _add_lbl(widget, text, row, col, **kw):
            lbl = widget.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        _add_lbl(self._plot, f"<span style='{lbl_style}'><b>Ch</b></span>", 1, 0, justify="center")
        for r in range(rows):
            _add_lbl(self._plot, f"<span style='{lbl_style}'><b>{r + 1}</b></span>", 1, r + 1, justify="center")
        for c in range(cols):
            _add_lbl(self._plot, f"<span style='{lbl_style}'><b>{c + 1}</b></span>", c + 2, 0, justify="center")

        gl = self._plot.ci.layout
        for c in range(cols):
            for r in range(rows):
                p = self._plot.addPlot(row=c + 2, col=r + 1)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0)
                p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
                p.setMinimumWidth(0)
                p.setMinimumHeight(0)

        gl.setSpacing(0)
        gl.setColumnMinimumWidth(0, 0)
        gl.setColumnStretchFactor(0, 0)
        for r in range(rows):
            gl.setColumnMinimumWidth(r + 1, 0)
            gl.setColumnStretchFactor(r + 1, 1)
        for r in range(2):
            gl.setRowMinimumHeight(r, 0)
            gl.setRowStretchFactor(r, 0)
        for c in range(cols):
            gl.setRowMinimumHeight(c + 2, 0)
            gl.setRowStretchFactor(c + 2, 1)

        for idx, wav in enumerate(waveforms):
            if idx >= len(ch_indices):
                break
            pos = positions.get(int(idx))
            if pos is None:
                continue
            r, c = pos
            if r >= rows or c >= cols:
                continue
            item = self._plot.getItem(c + 2, r + 1)
            if item is not None and len(wav) > 0:
                item.plot(wav, pen=pg.mkPen(color=COLORS["info"], width=1.5))

        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx}")

    def render_stacked(self, waveforms, ch_indices, mu_idx):
        self._plot.clear()
        plot = self._plot.addPlot(row=0, col=0)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return
        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["foreground"], width=1.5))
            txt = pg.TextItem(f"Ch {ch}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 8))
            plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        plot.setTitle(
            f"MU {mu_idx} — Stacked",
            color=COLORS["foreground"],
            size="11pt",
        )
        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx} (Stacked)")

    def clear(self, message="Select a Motor Unit"):
        self._plot.clear()
        p = self._plot.addPlot(row=0, col=0)
        t = pg.TextItem(message, color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        p.addItem(t)
        p.hideAxis("left")
        p.hideAxis("bottom")
        self.setWindowTitle("MUAP Shapes")


# EditionTab
# ---------------------------------------------------------------------------


class EditionTab(QWidget):
    data_modified = pyqtSignal()

    def __init__(self, fsamp: float = 2048.0, parent=None):
        super().__init__(parent)

        self._fsamp = fsamp
        self._ports: Dict[str, List[MotorUnit]] = {}
        self._emg_data: Dict[str, np.ndarray] = {}
        self._grid_info: Dict[str, Optional[Dict]] = {}
        self._raw_port_channels: Dict[str, np.ndarray] = {}

        self._current_port: Optional[str] = None
        self._current_mu_idx: int = -1
        self._edit_mode = EditMode.VIEW
        self._sel_arm = SelectionArm.NONE
        self._loaded_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._quit_after_save: bool = False

        self._start_sample: int = 0
        self._end_sample: int = 0
        self._full_source_mode: bool = False

        self._undo_stack: List[UndoAction] = []
        self._redo_stack: List[UndoAction] = []
        self._original_decomp_data: Optional[dict] = None
        self._filter_recalc_available: bool = False
        self._muap_popout: Optional[MuapPopoutDialog] = None

        self._build_ui()
        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _ts_to_plateau_local(self, timestamps: np.ndarray) -> np.ndarray:
        return timestamps - self._start_sample

    def _ts_to_absolute(self, plateau_local: np.ndarray) -> np.ndarray:
        return plateau_local + self._start_sample

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        self.quality_bar = MUPropertiesPanel()
        self.quality_bar.setMinimumHeight(110)
        root.addWidget(self.quality_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 1020])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')}; "
            f"padding-right: 6px;"
        )
        self.status_bar.addPermanentWidget(self._file_label)
        root.addWidget(self.status_bar)
        self._update_status()

    # ── Style helpers ────────────────────────────────────────────────────

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{ background-color: {COLORS.get('background_input','#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 4px; padding: 4px 8px; }}"
        )

    @staticmethod
    def _sel_btn_style(accent: str) -> str:
        """Checkable push-button style with a coloured 'on' state."""
        return (
            f"QPushButton {{"
            f"  color: {COLORS['foreground']};"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"  padding: 4px 10px;"
            f"  font-size: {FONT_SIZES.get('small','9pt')};"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {COLORS.get('background_input','#33334d')};"
            f"  border-color: {COLORS['border']};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background-color: {accent}30;"
            f"  border: 2px solid {accent};"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:disabled {{ color: {COLORS.get('text_dim','#6c7086')}; }}"
        )

    @staticmethod
    def _make_warning_icon(size: int = 14) -> QIcon:
        """Render ⚠ in warning yellow to a QIcon, independent of button text colour."""
        px = QPixmap(size, size)
        px.fill(Qt.transparent)
        p = QPainter(px)
        f = QFont()
        f.setPixelSize(size)
        p.setFont(f)
        p.setPen(QColor(COLORS["warning"]))
        p.drawText(px.rect(), Qt.AlignCenter, "⚠")
        p.end()
        return QIcon(px)

    @staticmethod
    def _warn_toolbar_btn_style() -> str:
        """Subtle toolbar button style — transparent background, default text colour."""
        fg = COLORS["foreground"]
        bg_hover = COLORS.get("background_input", "#1a1f24")
        fs = FONT_SIZES.get("small", "9pt")
        return (
            f"QPushButton {{"
            f"  color: {fg};"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"  padding: 4px 10px;"
            f"  font-size: {fs};"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {bg_hover};"
            f"  border-color: {COLORS['border']};"
            f"}}"
        )

    # ── Toolbar ──────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar()
        tb.setMovable(False)
        tb.setStyleSheet(
            f"""
            QToolBar {{
                background-color: {COLORS.get('background_light','#2a2a3c')};
                border-bottom: 1px solid {COLORS['border']};
                spacing: 4px;
                padding: 2px;
            }}
            QToolBar QLabel {{
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES.get('small','9pt')};
            }}
            QToolButton {{
                color: {COLORS['foreground']};
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {FONT_SIZES.get('small','9pt')};
            }}
            QToolButton:hover {{
                background-color: {COLORS.get('background_input','#33334d')};
                border-color: {COLORS['border']};
            }}
            QToolButton:checked {{
                background-color: {COLORS.get('info','#89b4fa')}30;
                border-color: {COLORS.get('info','#89b4fa')};
            }}
        """
        )

        # ── File ──────────────────────────────────────────────────────
        self.action_load = QAction("📂 Load", self)
        self.action_load.triggered.connect(self._load_file_dialog)
        tb.addAction(self.action_load)

        self.action_save = QAction("💾 Save", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self._save_file)
        tb.addAction(self.action_save)

        self.action_reset = QAction("⟲ Reset View", self)
        self.action_reset.setShortcut(QKeySequence("Home"))
        self.action_reset.triggered.connect(self._reset_view)
        tb.addAction(self.action_reset)

        # ── Edit mode (point-click) ───────────────────────────────────
        tb.addSeparator()
        tb.addWidget(QLabel("  Mode: "))
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

        # ── Rubberband-drag selection toggles ─────────────────────────
        tb.addSeparator()
        tb.addWidget(QLabel("  Select & "))

        success_color = COLORS.get("success", "#a6e3a1")
        error_color = COLORS.get("error", "#f38ba8")

        self.btn_sel_add = QPushButton("✅ Add in Selection")
        self.btn_sel_add.setCheckable(True)
        self.btn_sel_add.setChecked(False)
        self.btn_sel_add.setStyleSheet(self._sel_btn_style(success_color))
        self.btn_sel_add.setToolTip(
            "Toggle ON: drag a rectangle on the plot to add all peaks inside it.\n"
            "Stays armed — drag as many times as needed.\n"
            "Click again to turn off."
        )
        self.btn_sel_add.toggled.connect(self._on_sel_add_toggled)
        self.btn_sel_add.setEnabled(False)
        tb.addWidget(self.btn_sel_add)

        self.btn_sel_delete = QPushButton("🗑 Del in Selection")
        self.btn_sel_delete.setCheckable(True)
        self.btn_sel_delete.setChecked(False)
        self.btn_sel_delete.setStyleSheet(self._sel_btn_style(error_color))
        self.btn_sel_delete.setToolTip(
            "Toggle ON: drag a rectangle on the plot to delete all spikes inside it.\n"
            "Stays armed — drag as many times as needed.\n"
            "Click again to turn off."
        )
        self.btn_sel_delete.toggled.connect(self._on_sel_delete_toggled)
        self.btn_sel_delete.setEnabled(False)
        tb.addWidget(self.btn_sel_delete)

        # ── Undo / Redo ───────────────────────────────────────────────
        tb.addSeparator()
        self.action_undo = QAction("↩ Undo", self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.triggered.connect(self._undo)
        tb.addAction(self.action_undo)

        self.action_redo = QAction("↪ Redo", self)
        self.action_redo.setShortcut(QKeySequence.Redo)
        self.action_redo.triggered.connect(self._redo)
        tb.addAction(self.action_redo)

        # ── Global MU management ──────────────────────────────────────
        tb.addSeparator()
        self.btn_auto_flag = QPushButton("Auto-Flag Unreliable")
        self.btn_auto_flag.setIcon(self._make_warning_icon())
        self.btn_auto_flag.setToolTip(
            "Flag all MUs with SIL < 0.9 across all ports for deletion.\n"
            "Units with SIL ≥ 0.9 are left for manual review."
        )
        self.btn_auto_flag.clicked.connect(self._auto_flag_unreliable)
        self.btn_auto_flag.setStyleSheet(self._warn_toolbar_btn_style())
        self.btn_auto_flag.setEnabled(False)
        tb.addWidget(self.btn_auto_flag)

        return tb

    # ── Left panel ───────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {COLORS['background']};"
            f" border-right: 2px solid {COLORS['border']};"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(
            QLabel("PORT SELECTION", styleSheet=get_section_header_style("info"))
        )
        port_row = QHBoxLayout()
        port_lbl = QLabel("Port:")
        port_lbl.setStyleSheet(
            f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small','9pt')};"
        )
        port_row.addWidget(port_lbl)
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(self._combo_style())
        self.port_combo.currentTextChanged.connect(self._on_port_changed)
        port_row.addWidget(self.port_combo, stretch=1)
        lay.addLayout(port_row)

        lay.addWidget(QLabel("MOTOR UNIT", styleSheet=get_section_header_style("info")))
        mu_row = QHBoxLayout()
        mu_lbl = QLabel("Unit:")
        mu_lbl.setStyleSheet(
            f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small','9pt')};"
        )
        mu_row.addWidget(mu_lbl)
        self.mu_combo = QComboBox()
        self.mu_combo.setStyleSheet(self._combo_style())
        self.mu_combo.currentIndexChanged.connect(self._on_mu_selected)
        mu_row.addWidget(self.mu_combo, stretch=1)
        lay.addLayout(mu_row)

        # ── Button styles ────────────────────────────────────────────────
        _fs = FONT_SIZES.get("small", "9pt")
        _bg = COLORS.get("background_input", "#1a1f24")
        _fg = COLORS["foreground"]
        _border = COLORS["border"]
        _bg_hover = COLORS.get("background_hover", "#1e242b")

        primary_btn_style = f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: {_fs};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS.get('accent_light', COLORS['accent'])};
            }}
            QPushButton:disabled {{
                background-color: {COLORS.get('background_hover', '#1e242b')};
                color: {COLORS.get('text_dim', '#6c7086')};
            }}"""

        base_btn_style = f"""
            QPushButton {{
                background-color: {_bg};
                color: {_fg};
                border: 1px solid {_border};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: {_fs};
            }}
            QPushButton:hover {{
                background-color: {_bg_hover};
                border-color: {COLORS.get('info', '#4a9eff')};
            }}
            QPushButton:disabled {{
                color: {COLORS.get('text_dim', '#6c7086')};
                border-color: {_border};
            }}"""

        # ── Recalculate Filter (primary action) ──────────────────────────
        self.btn_recalc_filter = QPushButton("⟳ Recalculate Filter")
        self.btn_recalc_filter.setStyleSheet(primary_btn_style)
        self.btn_recalc_filter.setShortcut(QKeySequence("F"))
        self.btn_recalc_filter.setToolTip(
            "Replay peel-off and recompute filter + source + timestamps [F]"
        )
        self.btn_recalc_filter.clicked.connect(self._recalculate_filter)
        self.btn_recalc_filter.setEnabled(False)
        lay.addWidget(self.btn_recalc_filter)

        # ── Recalculate Centroid ─────────────────────────────────────────
        self.btn_recalc_centroid = QPushButton("⟳ Recalculate Centroid")
        self.btn_recalc_centroid.setStyleSheet(base_btn_style)
        self.btn_recalc_centroid.setShortcut(QKeySequence("C"))
        self.btn_recalc_centroid.setToolTip(
            "Re-run SCD amplitude clustering on the current source [C]\n"
            "Finds peaks in the source/IPT, then k-means (100 iter) separates\n"
            "true spikes from noise. Does not recompute the filter. Undoable."
        )
        self.btn_recalc_centroid.clicked.connect(self._recalculate_centroid)
        self.btn_recalc_centroid.setEnabled(False)
        lay.addWidget(self.btn_recalc_centroid)

        # ── Per-MU actions ───────────────────────────────────────────────
        lay.addSpacing(6)

        self.btn_remove_outliers = QPushButton("⚡ Remove Outliers")
        self.btn_remove_outliers.setStyleSheet(base_btn_style)
        self.btn_remove_outliers.setShortcut(QKeySequence("O"))
        self.btn_remove_outliers.setToolTip(
            "Remove spikes causing outlier instantaneous firing rate [O]\n"
            "Uses Tukey fence (Q75 + 1.5×IQR) on the IFR distribution.\n"
            "For each outlier pair, the lower-amplitude spike is deleted."
        )
        self.btn_remove_outliers.clicked.connect(self._remove_outliers)
        self.btn_remove_outliers.setEnabled(False)
        lay.addWidget(self.btn_remove_outliers)

        self.btn_flag_delete = QPushButton("🗑 Flag to Delete")
        self.btn_flag_delete.setStyleSheet(base_btn_style)
        self.btn_flag_delete.setToolTip("Toggle deletion flag for the current MU [X]")
        self.btn_flag_delete.clicked.connect(self._toggle_flag_delete)
        self.btn_flag_delete.setEnabled(False)
        lay.addWidget(self.btn_flag_delete)

        self.btn_auto_edit_mu = QPushButton("⚙ Auto-Edit This MU")
        self.btn_auto_edit_mu.setStyleSheet(base_btn_style)
        self.btn_auto_edit_mu.setShortcut(QKeySequence("E"))
        self.btn_auto_edit_mu.setToolTip(
            "Apply rule-based auto-editing to the current MU only [E]\n"
            "Removes low/high-IPT spikes; adds missed spikes by FR/IPT criteria.\n"
            "Based on Wen et al. (2024). Undoable with Ctrl+Z."
        )
        self.btn_auto_edit_mu.clicked.connect(self._run_auto_edit_current)
        self.btn_auto_edit_mu.setEnabled(False)
        lay.addWidget(self.btn_auto_edit_mu)

        self.muap_widget = pg.GraphicsLayoutWidget()
        self.muap_widget.setBackground(COLORS["background"])
        self.muap_widget.setMinimumHeight(50)
        self.muap_widget.setStyleSheet(
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;"
        )
        lay.addWidget(self.muap_widget, stretch=1)

        self.btn_muap_popout = QPushButton("↗", self.muap_widget)
        self.btn_muap_popout.setToolTip("Pop out MUAP view")
        self.btn_muap_popout.setFixedSize(20, 20)
        self.btn_muap_popout.setFont(QFont("Arial, Helvetica", 10))
        self.btn_muap_popout.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.get('background_hover', '#1e242b')};"
            f" color: {COLORS.get('text_dim', '#6c7086')}; padding: 0px;"
            f" border: 1px solid {COLORS['border']}; border-radius: 4px; }}"
            f"QPushButton:hover {{ color: {COLORS['foreground']};"
            f" border-color: {COLORS.get('info', '#4a9eff')}; }}"
        )
        self.btn_muap_popout.clicked.connect(self._open_muap_popout)
        self.btn_muap_popout.raise_()
        self.muap_widget.installEventFilter(self)

        return panel

    # ── Right panel ──────────────────────────────────────────────────────

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
        self.source_plot.region_selected.connect(self._on_region_selected)
        plot_splitter.addWidget(self.source_plot)

        self.fr_plot = FiringRatePlotWidget()
        self.fr_plot.set_fsamp(self._fsamp)
        self.fr_plot.link_x(self.source_plot)
        self.fr_plot.setMaximumHeight(150)
        plot_splitter.addWidget(self.fr_plot)
        plot_splitter.setSizes([500, 150])

        lay.addWidget(plot_splitter)
        return panel

    def eventFilter(self, obj, event):
        if obj is self.muap_widget and event.type() == QEvent.Resize:
            self._reposition_muap_popout_btn()
        return super().eventFilter(obj, event)

    def _reposition_muap_popout_btn(self):
        btn = self.btn_muap_popout
        btn.move(self.muap_widget.width() - btn.width() - 6, 6)
        btn.raise_()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Up"), self, self._select_prev_mu)
        QShortcut(QKeySequence("Down"), self, self._select_next_mu)
        QShortcut(
            QKeySequence("Ctrl+A"),
            self,
            lambda: self.btn_sel_add.setChecked(not self.btn_sel_add.isChecked()),
        )
        QShortcut(
            QKeySequence("Ctrl+D"),
            self,
            lambda: self.btn_sel_delete.setChecked(not self.btn_sel_delete.isChecked()),
        )
        QShortcut(QKeySequence("Escape"), self, lambda: self._set_mode(EditMode.VIEW))
        QShortcut(QKeySequence("X"), self, self.btn_flag_delete.click)

    # ------------------------------------------------------------------
    # Sampling rate
    # ------------------------------------------------------------------

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp
        self.source_plot.set_fsamp(fsamp)
        self.fr_plot.set_fsamp(fsamp)

    # ------------------------------------------------------------------
    # Selection arm toggling
    # ------------------------------------------------------------------

    def _on_sel_add_toggled(self, checked: bool):
        if checked:
            # Ensure the other button is off (mutual exclusion without QButtonGroup
            # so we keep independent checkable state)
            self.btn_sel_delete.blockSignals(True)
            self.btn_sel_delete.setChecked(False)
            self.btn_sel_delete.blockSignals(False)
            self._sel_arm = SelectionArm.ADD
        else:
            self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(self._sel_arm)
        self._update_status()

    def _on_sel_delete_toggled(self, checked: bool):
        if checked:
            self.btn_sel_add.blockSignals(True)
            self.btn_sel_add.setChecked(False)
            self.btn_sel_add.blockSignals(False)
            self._sel_arm = SelectionArm.DELETE
        else:
            self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(self._sel_arm)
        self._update_status()

    def _disarm_selection(self):
        """Programmatically turn off both selection toggles."""
        for btn in (self.btn_sel_add, self.btn_sel_delete):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(SelectionArm.NONE)

    # ------------------------------------------------------------------
    # Region-selected slot — dispatches immediately
    # ------------------------------------------------------------------

    def _on_region_selected(self, x1: float, x2: float, y1: float, y2: float):
        if self._sel_arm == SelectionArm.ADD:
            self._apply_selection_add(x1, x2, y1, y2)
        elif self._sel_arm == SelectionArm.DELETE:
            self._apply_selection_delete(x1, x2, y1, y2)

    # ------------------------------------------------------------------
    # Selection operations
    # ------------------------------------------------------------------

    def _apply_selection_add(self, x1: float, x2: float, y1: float, y2: float):
        mu = self._current_mu()
        if mu is None:
            return
        s1 = max(0, int(x1 * self._fsamp))
        s2 = min(len(mu.source), int(x2 * self._fsamp))
        if s1 >= s2:
            return

        source_sq = np.nan_to_num(mu.source, nan=0.0, posinf=0.0, neginf=0.0) ** 2
        segment = source_sq[s1:s2]
        peaks, _ = sp_signal.find_peaks(
            segment, distance=max(1, int(0.005 * self._fsamp))
        )
        peaks_abs = peaks + s1
        existing = set(mu.timestamps.tolist())

        new_spikes = [
            int(p)
            for p in peaks_abs
            if 0 <= p < len(source_sq)
            and y1 <= source_sq[p] <= y2
            and int(p) not in existing
        ]
        if not new_spikes:
            self._update_status("No new peaks in selection")
            return

        old_ts = mu.timestamps.copy()
        new_ts = np.sort(
            np.concatenate([mu.timestamps, np.array(new_spikes, dtype=np.int64)])
        )
        self._push_undo(
            UndoAction(
                f"Sel-add {len(new_spikes)}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Added {len(new_spikes)} spikes from selection")

    def _apply_selection_delete(self, x1: float, x2: float, y1: float, y2: float):
        mu = self._current_mu()
        if mu is None:
            return
        s1 = int(x1 * self._fsamp)
        s2 = int(x2 * self._fsamp)
        source_sq = np.nan_to_num(mu.source, nan=0.0, posinf=0.0, neginf=0.0) ** 2

        in_box = np.array(
            [
                s1 <= ts < s2 and 0 <= ts < len(source_sq) and y1 <= source_sq[ts] <= y2
                for ts in mu.timestamps
            ],
            dtype=bool,
        )

        n_remove = int(np.sum(in_box))
        if n_remove == 0:
            self._update_status("No spikes in selection")
            return

        old_ts = mu.timestamps.copy()
        new_ts = mu.timestamps[~in_box]
        self._push_undo(
            UndoAction(
                f"Sel-del {n_remove}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted {n_remove} spikes from selection")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _update_file_label(self):
        if self._loaded_path:
            self._file_label.setText(f"📄 Current File: {self._loaded_path.name}")
        else:
            self._file_label.setText("")

    def load_from_path(self, path: Path):
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
        if "ports" not in data or "discharge_times" not in data:
            QMessageBox.warning(
                self,
                "Format Error",
                "File does not contain 'ports' and 'discharge_times'.",
            )
            return
        if data.get("skip_filter_recalc"):
            reply = QMessageBox.question(
                self,
                "Recalculate Filters?",
                "This file was previously edited.\n\nDo you want to recalculate the filters for each motor unit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                data["skip_filter_recalc"] = False

        try:
            self._load_decomposition_data(data)
            self._loaded_path = path
            self._update_status(f"Loaded: {path.name}")
            self._update_file_label()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to parse:\n{e}")

    def _load_decomposition_data(self, decomp_data: dict):
        self._disarm_selection()
        self._ports.clear()
        self._emg_data.clear()
        self._grid_info.clear()
        self._raw_port_channels.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._original_decomp_data = decomp_data

        fsamp = decomp_data.get("sampling_rate", decomp_data.get("fsamp", self._fsamp))
        self.set_fsamp(float(fsamp))

        raw_data = decomp_data.get("data")
        emg_full = None
        if raw_data is not None:
            emg_full = to_numpy(raw_data)
            if emg_full.ndim == 2 and emg_full.shape[0] > emg_full.shape[1]:
                emg_full = emg_full.T

        skip_recalc = bool(decomp_data.get("skip_filter_recalc", False))
        can_full, reason = supports_full_source_computation(decomp_data)
        full_port_results: Dict[int, list] = {}
        start_sample = 0
        end_sample = emg_full.shape[1] if emg_full is not None else 0

        if skip_recalc:
            print(
                "  [edition] skip_filter_recalc=True — using stored sources/timestamps as-is"
            )
            can_full = False
        elif can_full:
            try:
                self._update_status("Computing full-length sources (peel-off replay)…")
                full_port_results, start_sample, end_sample, err = (
                    compute_all_full_sources(decomp_data)
                )
                if err:
                    print(f"  [edition] Full source warning: {err}")
                    full_port_results = {}
            except Exception as e:
                print(f"  [edition] Full source failed: {e}")
                traceback.print_exc()
                full_port_results = {}
        else:
            sel_pts = decomp_data.get(
                "plateau_coords", decomp_data.get("selected_points")
            )
            if sel_pts is not None:
                try:
                    pts = to_numpy(np.asarray(sel_pts)).flatten()
                    start_sample, end_sample = int(pts[0]), int(pts[1])
                except (IndexError, TypeError, ValueError):
                    pass
            if end_sample <= start_sample and emg_full is not None:
                end_sample = emg_full.shape[1]

        self._start_sample = start_sample
        self._end_sample = end_sample
        self._full_source_mode = bool(full_port_results)

        if self._full_source_mode:
            print(
                "  [edition] ✓ Full-length source mode (peel-off + source_to_timestamps)"
            )
        else:
            print(
                f"  [edition] Fallback: plateau-only sources"
                f"{(' — ' + reason) if not can_full else ''}"
            )

        ports = decomp_data.get("ports", [])
        chans_per_electrode = decomp_data.get("chans_per_electrode", [])
        channel_indices_all = decomp_data.get("channel_indices")
        mask_list = decomp_data.get("emg_mask", [])
        electrode_list = decomp_data.get("electrodes", [])
        ch_offset = 0

        for port_idx, port_name in enumerate(ports):
            n_ch = (
                int(chans_per_electrode[port_idx])
                if port_idx < len(chans_per_electrode)
                else 64
            )

            if (
                channel_indices_all is not None
                and port_idx < len(channel_indices_all)
                and channel_indices_all[port_idx] is not None
            ):
                port_ch_idx = np.asarray(channel_indices_all[port_idx], dtype=int)
            else:
                port_ch_idx = np.arange(ch_offset, ch_offset + n_ch, dtype=int)

            if port_idx < len(mask_list) and mask_list[port_idx] is not None:
                local_active = np.where(
                    to_numpy(np.asarray(mask_list[port_idx])).flatten() == 0
                )[0]
            else:
                local_active = np.arange(n_ch)
            global_active = port_ch_idx[local_active[local_active < len(port_ch_idx)]]

            emg_port = None
            if emg_full is not None:
                valid_chs = global_active[global_active < emg_full.shape[0]]
                if len(valid_chs) > 0:
                    emg_port = emg_full[
                        valid_chs,
                        max(0, start_sample) : min(end_sample, emg_full.shape[1]),
                    ]
                    valid_port_chs = port_ch_idx[port_ch_idx < emg_full.shape[0]]
                    self._raw_port_channels[port_name] = emg_full[valid_port_chs, :]

            port_discharge = (
                decomp_data["discharge_times"][port_idx]
                if port_idx < len(decomp_data["discharge_times"])
                else []
            )
            port_sources = (
                decomp_data["pulse_trains"][port_idx]
                if port_idx < len(decomp_data["pulse_trains"])
                else []
            )
            port_filters_raw = decomp_data.get("mu_filters", [])
            port_filters = (
                port_filters_raw[port_idx] if port_idx < len(port_filters_raw) else None
            )

            ts_list = self._ensure_list_of_arrays(port_discharge)
            src_list = self._ensure_list_of_arrays(port_sources)
            filt_list = (
                self._ensure_list_of_arrays(port_filters)
                if port_filters is not None
                else [None] * len(ts_list)
            )

            full_results = full_port_results.get(port_idx, [])

            motor_units = []
            for mu_idx in range(len(ts_list)):
                filt = (
                    to_numpy(filt_list[mu_idx])
                    if mu_idx < len(filt_list) and filt_list[mu_idx] is not None
                    else None
                )

                if (
                    self._full_source_mode
                    and mu_idx < len(full_results)
                    and full_results[mu_idx][0] is not None
                ):
                    entry = full_results[mu_idx]
                    source = entry[0]
                    ts_abs = entry[1]
                    if len(entry) > 2 and entry[2] is not None:
                        filt = entry[2]
                else:
                    source = (
                        to_numpy(src_list[mu_idx]).flatten()
                        if mu_idx < len(src_list)
                        else np.zeros(1)
                    )
                    ts_plateau = to_numpy(ts_list[mu_idx]).flatten().astype(np.int64)
                    ts_abs = (
                        self._ts_to_absolute(ts_plateau)
                        if self._full_source_mode
                        else ts_plateau
                    )

                motor_units.append(
                    MotorUnit(
                        id=mu_idx,
                        timestamps=ts_abs,
                        source=source,
                        port_name=port_name,
                        mu_filter=filt,
                    )
                )

            etype = electrode_list[port_idx] if port_idx < len(electrode_list) else None
            grid_cfg = get_grid_config(etype)

            if motor_units:
                if self._full_source_mode:
                    props_ts = [
                        self._ts_to_plateau_local(mu.timestamps) for mu in motor_units
                    ]
                    props_src = [
                        (
                            mu.source[start_sample:end_sample]
                            if len(mu.source) > (end_sample - start_sample)
                            else mu.source
                        )
                        for mu in motor_units
                    ]
                else:
                    props_ts = [mu.timestamps for mu in motor_units]
                    props_src = [mu.source for mu in motor_units]

                props_list = compute_port_properties(
                    all_timestamps=props_ts,
                    all_sources=props_src,
                    emg_port=emg_port,
                    grid_positions=grid_cfg["positions"] if grid_cfg else None,
                    grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
                    fsamp=self._fsamp,
                )
                for mu, p in zip(motor_units, props_list):
                    mu.props = p

            self._ports[port_name] = motor_units
            self._grid_info[port_name] = grid_cfg
            if emg_port is not None:
                self._emg_data[port_name] = emg_port

            ch_offset += n_ch
            print(
                f"  Port '{port_name}': {len(motor_units)} MUs, "
                f"{sum(len(m.timestamps) for m in motor_units)} spikes"
                f"{' (full)' if self._full_source_mode else ''}"
            )

        self._refresh_port_combo()
        if ports:
            self.port_combo.setCurrentText(ports[0])
            self._on_port_changed(ports[0])
        self._reset_view_full()

        if self._full_source_mode:
            self.source_plot.set_plateau_region(start_sample, end_sample)

        self._refresh_aux_controls()

        ok, reason = supports_filter_recalculation(decomp_data)
        self._filter_recalc_available = ok
        self.btn_recalc_filter.setEnabled(ok)
        if not ok:
            self.btn_recalc_filter.setToolTip(f"Unavailable: {reason}")

        for btn in (
            self.btn_recalc_centroid,
            self.btn_remove_outliers,
            self.btn_flag_delete,
            self.btn_auto_edit_mu,
            self.btn_sel_add,
            self.btn_sel_delete,
            self.btn_auto_flag,
        ):
            btn.setEnabled(True)

        all_mus = [mu for mus in self._ports.values() for mu in mus]
        n_total = len(all_mus)
        n_reliable = sum(
            1 for mu in all_mus if mu.props is not None and mu.props.is_reliable
        )
        n_unreliable = n_total - n_reliable
        print(
            f"  Reliability: {n_reliable}/{n_total} reliable, "
            f"{n_unreliable}/{n_total} not reliable"
        )
        self._update_status(
            f"Loaded {n_total} MUs — "
            f"{n_reliable} reliable  |  {n_unreliable} not reliable"
        )

    @staticmethod
    def _ensure_list_of_arrays(data) -> list:
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            return []
        if isinstance(data, list):
            if len(data) == 0:
                return []
            first = data[0]
            if isinstance(first, (np.ndarray, list)) or hasattr(first, "detach"):
                return [to_numpy(x) for x in data]
            return [to_numpy(data)]
        arr = to_numpy(data)
        if arr.ndim == 0 or arr.size == 0:
            return []
        if arr.ndim == 1:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    def _load_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Decomposition", "", "Pickle (*.pkl);;All (*)"
        )
        if path:
            self.load_from_path(Path(path))

    def set_output_path(self, path: Path):
        """Set a fixed output path so Ctrl+S saves without a dialog."""
        self._output_path = Path(path)

    def set_quit_after_save(self, enabled: bool):
        """Close the application after every successful save while enabled."""
        self._quit_after_save = enabled

    def _save_file(self):
        if not self._ports:
            self._update_status("Nothing to save")
            return

        if self._output_path:
            save_path = self._output_path
        else:
            default = ""
            if self._loaded_path:
                default = str(
                    self._loaded_path.with_name(self._loaded_path.stem + "_edited.pkl")
                )
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save Decomposition", default, "Pickle (*.pkl)"
            )
            if not chosen:
                return
            save_path = Path(chosen)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(self._build_save_dict(), f)
            self._update_status(f"Saved: {save_path.name}")
            self._update_file_label()
            if self._quit_after_save:
                QApplication.quit()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _build_save_dict(self) -> dict:
        import dataclasses

        ports = list(self._ports.keys())
        discharge_times, pulse_trains, mu_filters, mu_properties = [], [], [], []
        kept_indices_per_port = {}

        for port_name in ports:
            mus = self._ports[port_name]
            kept_local = [
                (i, mu) for i, mu in enumerate(mus) if not mu.flagged_duplicate
            ]
            kept_indices_per_port[port_name] = [i for i, _ in kept_local]
            kept = [mu for _, mu in kept_local]

            if self._full_source_mode:
                save_ts = [self._ts_to_plateau_local(mu.timestamps) for mu in kept]
                save_src = [
                    (
                        mu.source[self._start_sample : self._end_sample]
                        if len(mu.source) > (self._end_sample - self._start_sample)
                        else mu.source
                    )
                    for mu in kept
                ]
            else:
                save_ts = [mu.timestamps for mu in kept]
                save_src = [mu.source for mu in kept]

            discharge_times.append(save_ts)
            pulse_trains.append(save_src)
            mu_filters.append([mu.mu_filter for mu in kept] or None)

            port_props = []
            for mu in kept:
                if mu.props is not None:
                    d = dataclasses.asdict(mu.props)
                    d.pop("muap_grid", None)
                    d.pop("duplicate_candidates", None)
                    port_props.append(d)
                else:
                    port_props.append({})
            mu_properties.append(port_props)

        save_data = {
            "ports": ports,
            "sampling_rate": self._fsamp,
            "discharge_times": discharge_times,
            "pulse_trains": pulse_trains,
            "mu_filters": mu_filters,
            "skip_filter_recalc": True,
            "mu_properties": mu_properties,
        }

        if self._original_decomp_data is not None:
            for key in [
                "data",
                "aux_channels",
                "plateau_coords",
                "chans_per_electrode",
                "channel_indices",
                "emg_mask",
                "electrodes",
                "dewhitened_filters",
                "version",
                "preprocessing_config",
                "w_mat",
                "selected_points",
            ]:
                val = self._original_decomp_data.get(key)
                if val is not None and key not in save_data:
                    save_data[key] = val

            # Remap peel_off_sequence to match kept MUs only
            orig_peel = self._original_decomp_data.get("peel_off_sequence")
            if orig_peel is not None:
                # Build a per-port mapping: old local index -> new local index
                # (or None if that MU was flagged/deleted)
                port_index_maps = {}
                for port_name in ports:
                    all_mus = self._ports[port_name]
                    kept_set = set(kept_indices_per_port[port_name])
                    old_to_new = {}
                    new_idx = 0
                    for old_idx in range(len(all_mus)):
                        if old_idx in kept_set:
                            old_to_new[old_idx] = new_idx
                            new_idx += 1
                        # else: deleted — not added to map
                    port_index_maps[port_name] = old_to_new

                def remap_port_peel_seq(port_seq, old_to_new):
                    remapped = []
                    for entry in port_seq:
                        uid = entry.get("accepted_unit_idx")
                        if uid is None:
                            # Rejected repeat — keep as-is, no index to remap
                            remapped.append(entry)
                        elif uid in old_to_new:
                            # Kept MU — update its index
                            remapped.append(
                                {**entry, "accepted_unit_idx": old_to_new[uid]}
                            )
                        # else: flagged MU — drop this entry entirely
                    return remapped

                # peel_off_sequence is list[list], one per port
                new_peel = []
                for port_idx, port_name in enumerate(ports):
                    port_seq = orig_peel[port_idx] if port_idx < len(orig_peel) else []
                    old_to_new = port_index_maps[port_name]
                    new_peel.append(remap_port_peel_seq(port_seq, old_to_new))
                save_data["peel_off_sequence"] = new_peel

            orig_filters = self._original_decomp_data.get("mu_filters")
            if orig_filters is not None:
                save_data["mu_filters_original"] = orig_filters

        if self._emg_data:
            save_data["emg_per_port"] = dict(self._emg_data)

        return save_data

    # ------------------------------------------------------------------
    # Edit mode (point-click)
    # ------------------------------------------------------------------

    def _set_mode(self, mode: EditMode):
        self._edit_mode = mode
        self.source_plot.set_edit_mode(mode)
        {
            EditMode.VIEW: self.action_view,
            EditMode.ADD: self.action_add,
            EditMode.DELETE: self.action_delete,
        }[mode].setChecked(True)
        # Entering any point-click mode disarms the rubberband selection
        self._disarm_selection()
        self._update_status()

    def _reset_view_full(self):
        """Reset both plots to show the entire signal length."""
        mu = self._current_mu()
        if mu is None or len(mu.source) == 0:
            return
        x_max = len(mu.source) / self._fsamp
        # Source plot: set X and Y explicitly.  Any autoRange() call (even on
        # the linked FR plot) feeds spike-marker bounds back through the X link
        # and re-clips the range to the plateau region, so we avoid it entirely.
        src_sq = np.nan_to_num(mu.source**2)
        y_max = float(np.max(src_sq)) if len(src_sq) > 0 else 1.0
        y_pad = y_max * 0.05
        self.source_plot.getViewBox().setRange(
            xRange=(0, x_max), yRange=(-y_pad, y_max + y_pad), padding=0
        )
        # FR plot X follows via the link; only auto-range its Y axis.
        self.fr_plot.getViewBox().enableAutoRange(axis=1, enable=True)

    def _reset_view(self):
        self._reset_view_full()
        self._update_status("View reset")

    # ------------------------------------------------------------------
    # Point-click spike editing
    # ------------------------------------------------------------------

    def _handle_add_click(self, sample: int):
        if self._edit_mode != EditMode.ADD:
            return
        mu = self._current_mu()
        if mu is None:
            return
        if sample < 0 or sample >= len(mu.source):
            self._update_status("Click outside source range")
            return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end = min(len(mu.source), int(view_x[1] * self._fsamp))
        peak = self._find_nearest_peak(mu.source**2, sample, view_start, view_end)
        if peak is None:
            self._update_status("No peak found near click")
            return
        if peak in mu.timestamps:
            self._update_status("Spike already exists")
            return
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.append(mu.timestamps, peak)).astype(np.int64)
        self._push_undo(
            UndoAction(
                "Add spike",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Added spike at {peak / self._fsamp:.3f}s")

    def _handle_delete_click(self, sample: int):
        if self._edit_mode != EditMode.DELETE:
            return
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) == 0:
            return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end = min(len(mu.source), int(view_x[1] * self._fsamp))
        visible_mask = (mu.timestamps >= view_start) & (mu.timestamps < view_end)
        visible_ts = mu.timestamps[visible_mask]
        if len(visible_ts) == 0:
            self._update_status("No spikes in view")
            return
        nearest = np.argmin(np.abs(visible_ts - sample))
        target = visible_ts[nearest]
        global_idx = np.where(mu.timestamps == target)[0][0]
        old_ts = mu.timestamps.copy()
        new_ts = np.delete(mu.timestamps, global_idx)
        self._push_undo(
            UndoAction(
                "Delete spike",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted spike at {target / self._fsamp:.3f}s")

    def _find_nearest_peak(
        self, source, click_sample: int, view_start: int, view_end: int
    ) -> Optional[int]:
        if view_start >= view_end:
            return None
        view_width = view_end - view_start
        half_window = max(int(0.005 * self._fsamp), int(0.05 * view_width))
        search_start = max(view_start, click_sample - half_window)
        search_end = min(view_end, click_sample + half_window)
        if search_start >= search_end:
            return None
        segment = source[search_start:search_end]
        peaks, _ = sp_signal.find_peaks(
            segment, distance=max(1, int(0.005 * self._fsamp))
        )
        if len(peaks) == 0:
            return int(search_start + np.argmax(segment))
        peaks_abs = peaks + search_start
        return int(peaks_abs[np.argmin(np.abs(peaks_abs - click_sample))])

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

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
        self._apply_undo_redo(action, is_undo=True)
        self._on_data_changed(
            f"Undo: {action.description}",
            source_changed=action.old_source is not None,
        )

    def _redo(self):
        if not self._redo_stack:
            self._update_status("Nothing to redo")
            return
        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        self._apply_undo_redo(action, is_undo=False)
        self._on_data_changed(
            f"Redo: {action.description}",
            source_changed=action.new_source is not None,
        )

    def _apply_undo_redo(self, action: UndoAction, is_undo: bool):
        if self._current_port != action.port_name:
            self.port_combo.setCurrentText(action.port_name)
        if self._current_mu_idx != action.mu_idx:
            self.mu_combo.setCurrentIndex(action.mu_idx)
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

    # ------------------------------------------------------------------
    # Filter recalculation
    # ------------------------------------------------------------------

    def _recalculate_filter(self):
        mu = self._current_mu()
        if mu is None:
            self._update_status("Select a motor unit first")
            return
        if not self._filter_recalc_available:
            QMessageBox.warning(
                self,
                "Unavailable",
                "Filter recalculation requires peel_off_sequence in the file.",
            )
            return
        if len(mu.timestamps) < 2:
            self._update_status("Need at least 2 spikes")
            return

        raw_port = self._raw_port_channels.get(self._current_port)
        if raw_port is None:
            QMessageBox.warning(self, "Missing Data", "Raw port EMG not available.")
            return

        port_idx = list(self._ports.keys()).index(self._current_port)
        global_idx = self._global_unit_idx(self._current_port, self._current_mu_idx)
        if global_idx is None:
            self._update_status("Could not determine global unit index")
            return

        current_port_filters = [m.mu_filter for m in self._ports[self._current_port]]
        ts_abs = mu.timestamps
        if not self._full_source_mode:
            ts_abs = self._ts_to_absolute(mu.timestamps)

        self._update_status(f"Recalculating filter for MU {mu.id}…")
        try:
            new_filter, new_source_full, new_ts_abs = recalculate_unit_filter(
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

            new_timestamps = (
                new_ts_abs
                if self._full_source_mode
                else self._ts_to_plateau_local(new_ts_abs)
            )

            old_source = mu.source.copy()
            old_filter = mu.mu_filter.copy() if mu.mu_filter is not None else None
            old_timestamps = mu.timestamps.copy()

            mu.source = new_source_full
            mu.mu_filter = new_filter
            mu.timestamps = new_timestamps

            self._push_undo(
                UndoAction(
                    description=f"Recalculate filter MU {mu.id}",
                    port_name=self._current_port,
                    mu_idx=self._current_mu_idx,
                    old_timestamps=old_timestamps,
                    new_timestamps=new_timestamps,
                    old_source=old_source,
                    old_filter=old_filter,
                    new_source=new_source_full,
                    new_filter=new_filter,
                )
            )
            self._on_data_changed(
                f"Filter recalculated for MU {mu.id}",
                source_changed=True,
            )

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Recalculation Error", str(e))
            self._update_status("Filter recalculation failed")

    def _recalculate_centroid(self):
        mu = self._current_mu()
        if mu is None:
            self._update_status("Select a motor unit first")
            return
        if mu.source is None or len(mu.source) == 0:
            self._update_status("No source available for centroid recalculation")
            return

        # In full-source mode mu.source spans the full recording — crop to the
        # plateau window and convert returned plateau-local indices to absolute.
        if self._full_source_mode:
            source_slice = mu.source[self._start_sample : self._end_sample]
        else:
            source_slice = mu.source

        if len(source_slice) == 0:
            self._update_status("Empty source slice — check plateau region")
            return

        self._update_status(f"Recalculating centroid for MU {mu.id}…")
        try:
            new_ts_local = recalculate_unit_centroid(source_slice)
            new_timestamps = (
                self._ts_to_absolute(new_ts_local)
                if self._full_source_mode
                else new_ts_local
            )

            old_timestamps = mu.timestamps.copy()
            mu.timestamps = new_timestamps

            self._push_undo(
                UndoAction(
                    description=f"Recalculate centroid MU {mu.id}",
                    port_name=self._current_port,
                    mu_idx=self._current_mu_idx,
                    old_timestamps=old_timestamps,
                    new_timestamps=new_timestamps,
                )
            )
            self._on_data_changed(f"Centroid recalculated for MU {mu.id}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Centroid Recalculation Error", str(e))
            self._update_status("Centroid recalculation failed")

    def _global_unit_idx(self, port_name: str, mu_idx: int) -> Optional[int]:
        offset = 0
        for pname, mus in self._ports.items():
            if pname == port_name:
                return offset + mu_idx
            offset += len(mus)
        return None

    # ------------------------------------------------------------------
    # Motor-unit accessors
    # ------------------------------------------------------------------

    def _current_mu(self) -> Optional[MotorUnit]:
        return self._get_mu(self._current_port, self._current_mu_idx)

    def _get_mu(self, port, idx) -> Optional[MotorUnit]:
        if port is None or idx < 0:
            return None
        mus = self._ports.get(port, [])
        return mus[idx] if 0 <= idx < len(mus) else None

    # ------------------------------------------------------------------
    # UI callbacks — port / MU selection
    # ------------------------------------------------------------------

    def _on_mu_selected(self, index: int):
        if self._current_port is None:
            return
        mus = self._ports.get(self._current_port, [])
        if index < 0 or index >= len(mus):
            return
        self._current_mu_idx = index
        mu = self._current_mu()
        if mu:
            self.btn_flag_delete.setText(
                "Unflag" if mu.flagged_duplicate else "🗑 Flag to Delete"
            )
        self._update_plots(reset_view=True)
        self._update_status()

    def _select_prev_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx > 0:
            self.mu_combo.setCurrentIndex(idx - 1)

    def _select_next_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx < self.mu_combo.count() - 1:
            self.mu_combo.setCurrentIndex(idx + 1)

    def _toggle_flag_delete(self):
        mu = self._current_mu()
        if mu is None:
            return
        mu.flagged_duplicate = not mu.flagged_duplicate
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_status(
            f"MU {mu.id} {'flagged' if mu.flagged_duplicate else 'unflagged'}"
        )

    def _remove_outliers(self):
        """Remove spikes causing outlier IFR in the current MU.

        For each consecutive pair of spikes whose instantaneous firing rate
        lies above the Tukey fence (Q75 + 1.5×IQR of the IFR distribution),
        the spike with the lower source amplitude is removed.  A floor of
        2× the median IFR prevents false positives in units with very regular
        firing (where IQR can be near zero).
        """
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) < 3:
            return

        ts = np.sort(mu.timestamps)
        isi = np.diff(ts) / self._fsamp  # seconds
        # Guard against zero-length ISIs (duplicate timestamps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ifr = np.where(isi > 0, 1.0 / isi, np.inf)

        finite_ifr = ifr[np.isfinite(ifr)]
        if len(finite_ifr) < 4:
            self._update_status("Not enough spikes for IFR outlier detection")
            return

        q25, q75 = np.percentile(finite_ifr, [25, 75])
        iqr = q75 - q25
        tukey_threshold = q75 + 1.5 * iqr
        # Floor: must be at least 2× the median to avoid flagging normal variation
        threshold = max(tukey_threshold, 2.0 * np.median(finite_ifr))

        # Find pairs whose IFR exceeds the threshold
        outlier_pair_indices = np.where(ifr > threshold)[0]

        if len(outlier_pair_indices) == 0:
            self._update_status("No IFR outliers found")
            return

        # For each outlier pair (i, i+1), mark the lower-amplitude spike for removal.
        # Collect unique removal indices so a spike in multiple pairs is only removed once.
        source = mu.source
        n_src = len(source)
        to_remove: set[int] = set()
        for i in outlier_pair_indices:
            idx_a, idx_b = i, i + 1
            if idx_a in to_remove or idx_b in to_remove:
                # One of the pair is already scheduled for removal; skip re-evaluation
                continue
            t_a, t_b = ts[idx_a], ts[idx_b]
            amp_a = source[t_a] if 0 <= t_a < n_src else 0.0
            amp_b = source[t_b] if 0 <= t_b < n_src else 0.0
            to_remove.add(idx_a if amp_a <= amp_b else idx_b)

        remove_timestamps = ts[sorted(to_remove)]
        old_ts = mu.timestamps.copy()
        new_ts = np.setdiff1d(mu.timestamps, remove_timestamps)
        self._push_undo(
            UndoAction(
                f"Remove IFR outliers ({len(to_remove)})",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(
            f"Removed {len(to_remove)} IFR outlier spike(s) "
            f"(threshold {threshold:.1f} Hz)"
        )

    def _run_auto_edit_current(self):
        """Apply rule-based auto-editing to the currently selected MU.

        Implements the four rules from Wen et al. (2024): removes spikes with
        low or high IPT/FR, and adds missed spikes based on IPT and FR criteria.
        The edit is pushed as a single undoable action.
        """
        mu = self._current_mu()
        if mu is None:
            return
        if len(mu.timestamps) < MIN_SPIKES:
            self._update_status(
                f"MU {mu.id}: fewer than {MIN_SPIKES} spikes — auto-edit skipped"
            )
            return

        old_ts = mu.timestamps.copy()
        result = auto_edit(old_ts, mu.source, self._fsamp)

        if result.skipped:
            self._update_status(
                f"MU {mu.id}: fewer than {MIN_SPIKES} spikes — auto-edit skipped"
            )
            return

        if np.array_equal(old_ts, result.new_timestamps):
            self._update_status(f"MU {mu.id}: no changes from auto-edit")
            return

        self._push_undo(
            UndoAction(
                f"Auto-edit MU {mu.id}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                result.new_timestamps,
            )
        )
        mu.timestamps = result.new_timestamps
        self._on_data_changed(
            f"Auto-edited MU {mu.id}: "
            f"{result.n_removed} removed, {result.n_added} added"
        )

    def _auto_flag_unreliable(self):
        """Flag all MUs with SIL < 0.9 across every port for deletion.

        Units with SIL >= 0.9 are considered borderline and left for manual
        review (they pass the primary quality criterion even if secondary
        metrics such as COV or PNR are still failing).

        Units whose properties have not yet been computed are skipped.
        """
        flagged_count = 0
        borderline_count = 0
        no_props_count = 0

        for port_name, mus in self._ports.items():
            for mu in mus:
                if mu.flagged_duplicate:
                    continue  # already flagged — leave it
                if mu.props is None:
                    no_props_count += 1
                    continue
                sil = mu.props.sil
                if np.isnan(sil) or sil < 0.9:
                    mu.flagged_duplicate = True
                    flagged_count += 1
                else:
                    borderline_count += 1

        # Refresh the combo for the current port so ⚠ labels appear
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)

        parts = [f"Auto-flagged {flagged_count} MU(s)"]
        if borderline_count:
            parts.append(f"{borderline_count} borderline left (SIL ≥ 0.9)")
        if no_props_count:
            parts.append(f"{no_props_count} skipped (no quality data)")
        self._update_status(" — ".join(parts))

    def _refresh_port_combo(self):
        cur = self.port_combo.currentText()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for name in self._ports:
            self.port_combo.addItem(name)
        if cur in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
            self.port_combo.setCurrentText(cur)
        self.port_combo.blockSignals(False)

    def _on_port_changed(self, port_name: str):
        if not port_name or port_name not in self._ports:
            return
        self._current_port = port_name
        self._current_mu_idx = -1
        self._clear_plots()
        self._refresh_mu_combo()
        if self.mu_combo.count() > 0:
            self.mu_combo.setCurrentIndex(0)
            self._on_mu_selected(0)

    def _refresh_mu_combo(self):
        self.mu_combo.blockSignals(True)
        self.mu_combo.clear()
        if self._current_port is not None:
            for mu in self._ports.get(self._current_port, []):
                label = f"MU {mu.id}  ({len(mu.timestamps)} spikes)"
                if mu.flagged_duplicate:
                    label += "  ⚠"
                if mu.props is not None:
                    label += "  ✓" if mu.props.is_reliable else "  ✗"
                self.mu_combo.addItem(label)
        self.mu_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # AUX / force overlay
    # ------------------------------------------------------------------

    def _refresh_aux_controls(self):
        aux_channels = []
        if self._original_decomp_data is not None:
            aux_channels = self._original_decomp_data.get("aux_channels") or []
        self.source_plot.set_aux_data(aux_channels, self._fsamp)

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _update_plots(self, reset_view: bool = False):
        mu = self._current_mu()
        if mu is None:
            self._clear_plots()
            return

        vb = self.source_plot.getViewBox()
        if not reset_view:
            x_range = vb.viewRange()[0]
            y_range = vb.viewRange()[1]
            had_custom_range = not vb.autoRangeEnabled()[0]

        self.source_plot.set_data(mu.source, mu.timestamps)
        if self._full_source_mode:
            self.source_plot.set_plateau_region(self._start_sample, self._end_sample)
        self.fr_plot.set_data(mu.timestamps)

        if reset_view:
            self._reset_view_full()
        elif had_custom_range:
            vb.setRange(xRange=x_range, yRange=y_range, padding=0)

        self._plot_muap()
        self._update_quality_panel(mu)

    def _clear_plots(self):
        self.source_plot.clear_data()
        self.fr_plot.clear_data()
        self._clear_muap_plot()
        self.quality_bar.clear_properties()

    def _on_data_changed(self, msg: str = "Modified", source_changed: bool = False):
        """Recompute properties after an edit. Preserves zoom unless source changed."""
        mu = self._current_mu()
        if mu is not None:
            grid_cfg = self._grid_info.get(self._current_port)
            emg_port = self._emg_data.get(self._current_port)

            if self._full_source_mode:
                props_ts = self._ts_to_plateau_local(mu.timestamps)
                props_source = (
                    mu.source[self._start_sample : self._end_sample]
                    if len(mu.source) > (self._end_sample - self._start_sample)
                    else mu.source
                )
            else:
                props_ts = mu.timestamps
                props_source = mu.source

            mu.props = recompute_unit_properties(
                mu_props=mu.props or MUProperties(),
                new_timestamps=props_ts,
                source=props_source,
                emg_port=emg_port,
                grid_positions=grid_cfg["positions"] if grid_cfg else None,
                grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
                fsamp=self._fsamp,
            )

            if source_changed:
                self._update_plots()
            else:
                # Lightweight refresh — zoom preserved
                self.source_plot.update_timestamps(mu.timestamps)
                self.fr_plot.set_data(mu.timestamps)
                self._plot_muap()
                self._update_quality_panel(mu)

        self._refresh_mu_combo()
        self.mu_combo.blockSignals(True)
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self.mu_combo.blockSignals(False)
        self._update_status(msg)
        self.data_modified.emit()

    def _update_quality_panel(self, mu: MotorUnit):
        if mu.props is not None:
            self.quality_bar.set_properties(mu.props)
        else:
            self.quality_bar.clear_properties()

    def _update_status(self, msg: Optional[str] = None):
        if msg:
            self.status_bar.showMessage(msg, 4000)
            return
        parts = []
        if self._current_port:
            parts.append(f"Port: {self._current_port}")
        n = len(self._ports.get(self._current_port, []))
        if n > 0:
            parts.append(f"{n} MUs")
        if self._current_mu_idx >= 0:
            parts.append(f"MU: {self._current_mu_idx}")

        if self._sel_arm == SelectionArm.ADD:
            parts.append("⬜ Drag to ADD  (armed)")
        elif self._sel_arm == SelectionArm.DELETE:
            parts.append("⬜ Drag to DELETE  (armed)")
        else:
            hints = {
                EditMode.VIEW: "View [V]",
                EditMode.ADD: "Add [A]  — click",
                EditMode.DELETE: "Delete [D]  — click",
            }
            parts.append(hints[self._edit_mode])

        if self._undo_stack:
            parts.append(f"Undo: {len(self._undo_stack)}")
        if self._full_source_mode:
            parts.append("Full signal")
        self.status_bar.showMessage("  |  ".join(parts))

    # ------------------------------------------------------------------
    # MUAP panel
    # ------------------------------------------------------------------

    def _open_muap_popout(self):
        if self._muap_popout is None or not self._muap_popout.isVisible():
            self._muap_popout = MuapPopoutDialog(parent=None)
            self._muap_popout.setAttribute(Qt.WA_DeleteOnClose)
            self._muap_popout.destroyed.connect(self._on_muap_popout_closed)
        self._muap_popout.show()
        self._muap_popout.raise_()
        self._muap_popout.activateWindow()
        self._plot_muap()

    def _on_muap_popout_closed(self):
        self._muap_popout = None

    def _plot_muap(self):
        mu = self._current_mu()
        if mu is None:
            self._clear_muap_plot("Select a Motor Unit")
            return
        if mu.props is None or mu.props.muap_grid is None:
            no_emg = self._emg_data.get(self._current_port) is None
            msg = "No EMG data in file" if no_emg else "MUAP unavailable"
            self._clear_muap_plot(msg)
            return

        muap_grid = mu.props.muap_grid
        grid_cfg = self._grid_info.get(self._current_port)
        if grid_cfg is not None:
            rows, cols = grid_cfg["grid_shape"]
            positions = grid_cfg["positions"]
            waveforms, ch_indices = [], []
            for ch in range(rows * cols):
                pos = positions.get(ch)
                if pos is None:
                    continue
                r, c = pos
                if r < rows and c < cols:
                    waveforms.append(muap_grid[r, c])
                    ch_indices.append(ch)
            self._render_muap_grid(waveforms, ch_indices, grid_cfg)
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_grid(
                    waveforms, ch_indices, grid_cfg, self._current_mu_idx
                )
        else:
            n_ch = muap_grid.shape[0]
            waveforms = [muap_grid[i, 0] for i in range(n_ch)]
            self._render_muap_stacked(waveforms, list(range(n_ch)))
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_stacked(
                    waveforms, list(range(n_ch)), self._current_mu_idx
                )

    def _render_muap_grid(self, waveforms, ch_indices, grid_cfg):
        self.muap_widget.clear()
        rows, cols = grid_cfg["grid_shape"]
        positions = grid_cfg["positions"]
        valid = [w for w in waveforms if len(w) > 0]
        amp = np.max(np.abs(np.concatenate(valid))) * 1.2 if valid else 1.0
        n_samples = len(waveforms[0]) if valid else 409

        # Transposed layout: data rows → display columns, data cols → display rows.
        # Row 0: MU label.  Row 1: data-row number strip.  Rows 2…cols+1: data plots.
        # Col 0: data-col number strip.  Cols 1…rows: data plots.
        label = (
            f"<span style='color:{COLORS['foreground']};font-size:10pt;'>"
            f"MU {self._current_mu_idx}</span>"
        )
        self.muap_widget.addLabel(label, row=0, col=0, colspan=rows + 1, justify="center")

        lbl_style = f"color:{COLORS.get('text_dim','#6c7086')}; font-size:7pt;"

        def _add_lbl(widget, text, row, col, **kw):
            lbl = widget.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        _add_lbl(self.muap_widget, f"<span style='{lbl_style}'><b>Ch</b></span>", 1, 0, justify="center")
        for r in range(rows):
            _add_lbl(self.muap_widget, f"<span style='{lbl_style}'>{r + 1}</span>", 1, r + 1, justify="center")
        for c in range(cols):
            _add_lbl(self.muap_widget, f"<span style='{lbl_style}'>{c + 1}</span>", c + 2, 0, justify="center")

        gl = self.muap_widget.ci.layout
        for c in range(cols):
            for r in range(rows):
                p = self.muap_widget.addPlot(row=c + 2, col=r + 1)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0)
                p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
                p.setMinimumWidth(0)
                p.setMinimumHeight(0)

        gl.setSpacing(0)
        gl.setColumnMinimumWidth(0, 0)
        gl.setColumnStretchFactor(0, 0)
        for r in range(rows):
            gl.setColumnMinimumWidth(r + 1, 0)
            gl.setColumnStretchFactor(r + 1, 1)
        for r in range(2):
            gl.setRowMinimumHeight(r, 0)
            gl.setRowStretchFactor(r, 0)
        for c in range(cols):
            gl.setRowMinimumHeight(c + 2, 0)
            gl.setRowStretchFactor(c + 2, 1)

        for idx, wav in enumerate(waveforms):
            if idx >= len(ch_indices):
                break
            pos = positions.get(int(idx))
            if pos is None:
                continue
            r, c = pos
            if r >= rows or c >= cols:
                continue
            item = self.muap_widget.getItem(c + 2, r + 1)
            if item is not None and len(wav) > 0:
                item.plot(wav, pen=pg.mkPen(color=COLORS["info"], width=1.5))

        # pyqtgraph's GraphicsView only updates its viewport transform in resizeEvent.
        # Calling resizeEvent(None) recomputes the transform for the current widget
        # size without requiring an actual geometry change.
        self.muap_widget.resizeEvent(None)

    def _render_muap_stacked(self, waveforms, ch_indices):
        self.muap_widget.clear()
        plot = self.muap_widget.addPlot(row=0, col=0)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return
        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["foreground"], width=1.5))
            txt = pg.TextItem(f"Ch {ch}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 7))
            plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        plot.setTitle(
            f"MU {self._current_mu_idx} — Stacked",
            color=COLORS["foreground"],
            size="10pt",
        )

    def _clear_muap_plot(self, message: str = "Select a Motor Unit"):
        self.muap_widget.clear()
        p = self.muap_widget.addPlot(row=0, col=0)
        t = pg.TextItem(message, color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        p.addItem(t)
        p.hideAxis("left")
        p.hideAxis("bottom")
        if self._muap_popout and self._muap_popout.isVisible():
            self._muap_popout.clear(message)
