"""
Edition Tab - EMG spike editing interface.

On load, filters are applied via peel-off replay on the full preprocessed EMG,
and timestamps are re-detected via source_to_timestamps.  The plateau region
is shown as a shaded band on the source plot.
"""

import logging
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
import pickle
import traceback

import numpy as np
from scipy import signal as sp_signal

from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt5.QtWidgets import (
    QWidget,
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
    QApplication,
)
from PyQt5.QtGui import QKeySequence, QFont, QPixmap, QIcon, QPainter, QColor
import pyqtgraph as pg

from scd_app.gui.style.styling import (
    COLORS,
    FONT_SIZES,
    FONT_FAMILY,
    get_section_header_style,
)
from scd_app.core.mu_model import EditMode, MotorUnit, UndoAction
from scd_app.core.mu_properties import (
    MUProperties,
    compute_port_properties,
    recompute_unit_properties,
    build_spike_train_matrix,
)
from scd_app.core.utils import to_numpy
from scd_app.core.constants import ROA_THRESHOLD
from scd_app.gui.widgets.mu_properties_panel import MUPropertiesPanel
from scd_app.gui.widgets.source_plot_widget import (
    SelectionArm,
    SourcePlotWidget,
    FiringRatePlotWidget,
)
from scd_app.gui.widgets.muap_popout import MuapPopoutDialog
from scd_app.core.filter_recalculation import (
    recalculate_unit_filter,
    supports_filter_recalculation,
    supports_full_source_computation,
    compute_all_full_sources,
)
from scd_app.core.auto_editor import auto_edit, MIN_SPIKES

logger = logging.getLogger(__name__)

try:
    from motor_unit_toolbox import spike_comp as _tb_spike_comp
    _SPIKE_COMP_AVAILABLE = True
except ImportError:
    _SPIKE_COMP_AVAILABLE = False


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
    # Key = 0-based OTBio channel index, value = (row, col) physical position
    # Col 0 = Side A (pads 1-20), Col 1 = Side B (pads 21-40)
    # Row 0 = pad 1/21 (proximal end), Row 19 = pad 20/40 (distal end)
    # Derived from DEMOVE->OTBio connector mapping table
    0: (18, 1),   # OTBio 1  -> DEMOVE 39 -> Side B pad 19
    1: (17, 1),   # OTBio 2  -> DEMOVE 38 -> Side B pad 18
    2: (16, 1),   # OTBio 3  -> DEMOVE 37 -> Side B pad 17
    3: (19, 1),   # OTBio 4  -> DEMOVE 40 -> Side B pad 20
    4: (12, 1),   # OTBio 5  -> DEMOVE 33 -> Side B pad 13
    5: (15, 1),   # OTBio 6  -> DEMOVE 36 -> Side B pad 16
    6: (14, 1),   # OTBio 7  -> DEMOVE 35 -> Side B pad 15
    7: (13, 1),   # OTBio 8  -> DEMOVE 34 -> Side B pad 14
    8: (10, 1),   # OTBio 9  -> DEMOVE 31 -> Side B pad 11
    9: (9, 1),    # OTBio 10 -> DEMOVE 30 -> Side B pad 10
    10: (6, 1),   # OTBio 11 -> DEMOVE 27 -> Side B pad 7
    11: (5, 1),   # OTBio 12 -> DEMOVE 26 -> Side B pad 6
    12: (2, 1),   # OTBio 13 -> DEMOVE 23 -> Side B pad 3
    13: (1, 1),   # OTBio 14 -> DEMOVE 22 -> Side B pad 2
    14: (18, 0),  # OTBio 15 -> DEMOVE 19 -> Side A pad 19
    15: (17, 0),  # OTBio 16 -> DEMOVE 18 -> Side A pad 18
    16: (14, 0),  # OTBio 17 -> DEMOVE 15 -> Side A pad 15
    17: (13, 0),  # OTBio 18 -> DEMOVE 14 -> Side A pad 14
    18: (10, 0),  # OTBio 19 -> DEMOVE 11 -> Side A pad 11
    19: (9, 0),   # OTBio 20 -> DEMOVE 10 -> Side A pad 10
    20: (6, 0),   # OTBio 21 -> DEMOVE 7  -> Side A pad 7
    21: (5, 0),   # OTBio 22 -> DEMOVE 6  -> Side A pad 6
    22: (2, 0),   # OTBio 23 -> DEMOVE 3  -> Side A pad 3
    23: (1, 0),   # OTBio 24 -> DEMOVE 2  -> Side A pad 2
    24: (0, 0),   # OTBio 25 -> DEMOVE 1  -> Side A pad 1
    25: (3, 0),   # OTBio 26 -> DEMOVE 4  -> Side A pad 4
    26: (4, 0),   # OTBio 27 -> DEMOVE 5  -> Side A pad 5
    27: (7, 0),   # OTBio 28 -> DEMOVE 8  -> Side A pad 8
    28: (8, 0),   # OTBio 29 -> DEMOVE 9  -> Side A pad 9
    29: (11, 0),  # OTBio 30 -> DEMOVE 12 -> Side A pad 12
    30: (12, 0),  # OTBio 31 -> DEMOVE 13 -> Side A pad 13
    31: (15, 0),  # OTBio 32 -> DEMOVE 16 -> Side A pad 16
    32: (16, 0),  # OTBio 33 -> DEMOVE 17 -> Side A pad 17
    33: (19, 0),  # OTBio 34 -> DEMOVE 20 -> Side A pad 20
    34: (0, 1),   # OTBio 35 -> DEMOVE 21 -> Side B pad 1
    35: (3, 1),   # OTBio 36 -> DEMOVE 24 -> Side B pad 4
    36: (4, 1),   # OTBio 37 -> DEMOVE 25 -> Side B pad 5
    37: (7, 1),   # OTBio 38 -> DEMOVE 28 -> Side B pad 8
    38: (8, 1),   # OTBio 39 -> DEMOVE 29 -> Side B pad 9
    39: (11, 1),  # OTBio 40 -> DEMOVE 32 -> Side B pad 12
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
    "GR04MM1305": {
        "grid_shape": (13, 5),
        "ied_mm": 4,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_13x5,
    },
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

def get_grid_config(electrode_type: Optional[str]) -> Optional[Dict]:
    if electrode_type is None:
        return None
    key = electrode_type.upper()
    for name, cfg in ELECTRODE_GRIDS.items():
        if name.upper() in key:
            return cfg
    return None


# ---------------------------------------------------------------------------
# EditionTab
# ---------------------------------------------------------------------------


class EditionTab(QWidget):
    data_modified = pyqtSignal()
    file_loaded = pyqtSignal()

    def __init__(self, fsamp: float = 2048.0, parent=None):
        super().__init__(parent)

        self._fsamp = fsamp
        self._ports: Dict[str, List[MotorUnit]] = {}
        self._emg_data: Dict[str, np.ndarray] = {}
        self._grid_info: Dict[str, Optional[Dict]] = {}
        self._raw_port_channels: Dict[str, np.ndarray] = {}
        self._rejected_ch_positions: Dict[str, set] = {}

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
        self._redetect_timestamps: bool = True

        self._undo_stack: List[UndoAction] = []
        self._redo_stack: List[UndoAction] = []
        self._original_decomp_data: Optional[dict] = None
        self._filter_recalc_available: bool = False
        self._muap_popout: Optional[MuapPopoutDialog] = None

        # Debounce: expensive recompute + MUAP render fire 120ms after the last edit
        self._props_timer = QTimer(self)
        self._props_timer.setSingleShot(True)
        self._props_timer.setInterval(120)
        self._props_timer.timeout.connect(self._flush_props_update)
        self._pending_source_changed: bool = False

        # MUAP grid reuse: keep cell PlotDataItems alive across MU switches
        self._muap_cell_plots: Dict[Tuple[int, int], object] = {}
        self._muap_waveform_items: Dict[Tuple[int, int], object] = {}
        self._muap_grid_key: Optional[Tuple] = None
        self._muap_title_label = None

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

        # ── Per-MU editing ────────────────────────────────────────────
        tb.addSeparator()

        self.btn_recalc_filter = QPushButton("⟳ Recalc Filter")
        self.btn_recalc_filter.setShortcut(QKeySequence("F"))
        self.btn_recalc_filter.setToolTip(
            "Replay peel-off and recompute filter + source + timestamps [F]"
        )
        self.btn_recalc_filter.clicked.connect(self._recalculate_filter)
        self.btn_recalc_filter.setEnabled(False)
        self.btn_recalc_filter.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_recalc_filter)

        self.btn_remove_outliers = QPushButton("⚡ Remove Outliers")
        self.btn_remove_outliers.setShortcut(QKeySequence("O"))
        self.btn_remove_outliers.setToolTip(
            "Remove spikes causing outlier instantaneous firing rate [O]\n"
            "Uses Tukey fence (Q75 + 1.5×IQR) on the IFR distribution."
        )
        self.btn_remove_outliers.clicked.connect(self._remove_outliers)
        self.btn_remove_outliers.setEnabled(False)
        self.btn_remove_outliers.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_remove_outliers)

        self.btn_flag_delete = QPushButton("🗑 Flag to Delete")
        self.btn_flag_delete.setToolTip("Toggle deletion flag for the current MU [X]")
        self.btn_flag_delete.clicked.connect(self._toggle_flag_delete)
        self.btn_flag_delete.setEnabled(False)
        self.btn_flag_delete.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_flag_delete)

        self.btn_auto_edit_mu = QPushButton("⚙ Auto-Edit MU")
        self.btn_auto_edit_mu.setShortcut(QKeySequence("E"))
        self.btn_auto_edit_mu.setToolTip(
            "Apply rule-based auto-editing to the current MU [E]\n"
            "Removes low/high-IPT spikes; adds missed spikes by FR/IPT criteria."
        )
        self.btn_auto_edit_mu.clicked.connect(self._run_auto_edit_current)
        self.btn_auto_edit_mu.setEnabled(False)
        self.btn_auto_edit_mu.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_auto_edit_mu)

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

        # ── Session-level actions ────────────────────────────────────────
        lay.addWidget(
            QLabel("SESSION", styleSheet=get_section_header_style("warning", margin_top=0))
        )

        self.btn_auto_flag = QPushButton("⚠ Auto-Flag Unreliable")
        self.btn_auto_flag.setIcon(self._make_warning_icon())
        self.btn_auto_flag.setStyleSheet(base_btn_style)
        self.btn_auto_flag.setToolTip(
            "Flag all MUs with SIL < 0.9 across all ports for deletion.\n"
            "Units with SIL ≥ 0.9 are left for manual review."
        )
        self.btn_auto_flag.clicked.connect(self._auto_flag_unreliable)
        self.btn_auto_flag.setEnabled(False)
        lay.addWidget(self.btn_auto_flag)

        self.btn_flag_within_dups = QPushButton("⧉ Within-Port Dups")
        self.btn_flag_within_dups.setStyleSheet(base_btn_style)
        self.btn_flag_within_dups.setToolTip(
            "Flag lower-quality duplicate MUs within each grid/probe for deletion.\n"
            "Uses rate-of-agreement (threshold 0.3) to identify duplicates."
        )
        self.btn_flag_within_dups.clicked.connect(self._flag_within_duplicates)
        self.btn_flag_within_dups.setEnabled(False)
        lay.addWidget(self.btn_flag_within_dups)

        self.btn_flag_cross_dups = QPushButton("⧉ Cross-Port Dups")
        self.btn_flag_cross_dups.setStyleSheet(base_btn_style)
        self.btn_flag_cross_dups.setToolTip(
            "Flag lower-quality duplicate MUs across different grids/probes for deletion.\n"
            "Uses rate-of-agreement (threshold 0.3) to identify duplicates."
        )
        self.btn_flag_cross_dups.clicked.connect(self._flag_cross_duplicates)
        self.btn_flag_cross_dups.setEnabled(False)
        lay.addWidget(self.btn_flag_cross_dups)

        self.btn_delete_flagged = QPushButton("🗑 Delete All Flagged MUs")
        self.btn_delete_flagged.setStyleSheet(base_btn_style)
        self.btn_delete_flagged.setToolTip(
            "Permanently remove all MUs flagged for deletion from the current session"
        )
        self.btn_delete_flagged.clicked.connect(self._delete_all_flagged)
        self.btn_delete_flagged.setEnabled(False)
        lay.addWidget(self.btn_delete_flagged)

        self.muap_widget = pg.GraphicsLayoutWidget()
        self.muap_widget.setBackground(COLORS["background"])
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

    def get_visualisation_data(self) -> dict:
        """Return a snapshot of all data needed by the Visualisation tab."""
        return {
            "ports": self._ports,
            "aux_channels": (self._original_decomp_data or {}).get("aux_channels") or [],
            "fsamp": self._fsamp,
            "start_sample": self._start_sample,
            "end_sample": self._end_sample,
            "file_stem": self._loaded_path.stem if self._loaded_path else "",
        }

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

        can_full, _ = supports_full_source_computation(data)
        if can_full and not data.get("skip_filter_recalc"):
            pts = data.get("plateau_coords", data.get("selected_points"))
            raw = data.get("data")
            full_len = (
                raw.shape[1]
                if raw is not None and hasattr(raw, "shape") and raw.ndim >= 2
                else None
            )
            is_partial_section = (
                pts is not None
                and full_len is not None
                and (int(pts[0]) > 0 or int(pts[1]) < full_len)
            )
            if is_partial_section:
                reply = QMessageBox.question(
                    self,
                    "Recalculate Timestamps?",
                    "Do you want to recalculate spike timestamps on the full signal?\n\n"
                    "Yes — re-detect timestamps from the source over the entire recording.\n"
                    "No  — keep the original timestamps from the decomposed section only.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                self._redetect_timestamps = reply == QMessageBox.Yes
        else:
            self._redetect_timestamps = True

        try:
            self._loaded_path = path  # must be set before _load_decomposition_data so _refresh_aux_controls sees the correct stem
            self._load_decomposition_data(data)
            self._update_status(f"Loaded: {path.name}")
            self._update_file_label()
            self.file_loaded.emit()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to parse:\n{e}")

    def _load_decomposition_data(self, decomp_data: dict):
        self._disarm_selection()
        self._ports.clear()
        self._emg_data.clear()
        self._grid_info.clear()
        self._raw_port_channels.clear()
        self._rejected_ch_positions.clear()
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
            logger.info("skip_filter_recalc=True — using stored sources/timestamps as-is")
            can_full = False
        elif can_full:
            try:
                self._update_status("Computing full-length sources (peel-off replay)…")
                full_port_results, start_sample, end_sample, err = (
                    compute_all_full_sources(decomp_data,
                                            redetect_timestamps=self._redetect_timestamps)
                )
                if err:
                    logger.warning("Full source warning: %s", err)
                    full_port_results = {}
            except Exception as e:
                logger.error("Full source computation failed: %s\n%s", e, traceback.format_exc())
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
            logger.info("Full-length source mode active (peel-off + source_to_timestamps)")
        else:
            logger.info("Plateau-only source mode%s",
                        f" — {reason}" if not can_full else "")

        ports = decomp_data.get("ports", [])
        ch_offset = 0
        for port_idx, port_name in enumerate(ports):
            ch_offset += self._load_single_port(
                port_idx, port_name, decomp_data,
                emg_full, start_sample, end_sample,
                full_port_results, ch_offset,
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
            self.btn_remove_outliers,
            self.btn_flag_delete,
            self.btn_delete_flagged,
            self.btn_auto_edit_mu,
            self.btn_sel_add,
            self.btn_sel_delete,
            self.btn_auto_flag,
            self.btn_flag_within_dups,
            self.btn_flag_cross_dups,
        ):
            btn.setEnabled(True)

        all_mus_have_props = all(
            mu.props is not None
            for mus in self._ports.values()
            for mu in mus
        )
        if not all_mus_have_props:
            _no_props_tip = "Compute MU properties first"
            self.btn_flag_within_dups.setEnabled(False)
            self.btn_flag_within_dups.setToolTip(_no_props_tip)
            self.btn_flag_cross_dups.setEnabled(False)
            self.btn_flag_cross_dups.setToolTip(_no_props_tip)

        all_mus = [mu for mus in self._ports.values() for mu in mus]
        n_total = len(all_mus)
        n_reliable = sum(
            1 for mu in all_mus if mu.props is not None and mu.props.is_reliable
        )
        n_unreliable = n_total - n_reliable
        logger.info("Reliability: %d/%d reliable, %d/%d not reliable",
                    n_reliable, n_total, n_unreliable, n_total)
        self._update_status(
            f"Loaded {n_total} MUs — "
            f"{n_reliable} reliable  |  {n_unreliable} not reliable"
        )

    def _load_single_port(
        self,
        port_idx: int,
        port_name: str,
        decomp_data: dict,
        emg_full,
        start_sample: int,
        end_sample: int,
        full_port_results: dict,
        ch_offset: int,
    ) -> int:
        chans_per_electrode = decomp_data.get("chans_per_electrode", [])
        channel_indices_all = decomp_data.get("channel_indices")
        mask_list = decomp_data.get("emg_mask", [])
        electrode_list = decomp_data.get("electrodes", [])

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
                if self._full_source_mode:
                    emg_port = emg_full[valid_chs, :]
                else:
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

        # Build corrected grid positions: maps new-0-based emg_port row -> (r, c).
        # The raw positions dicts use 1-based channel keys (matching hardware numbering)
        # while emg_port rows are 0-based and may have gaps due to rejected channels.
        # muap_mapping converts original port-local index -> grid key.
        corrected_positions = None
        rejected_pos_set: set = set()
        if grid_cfg is not None:
            muap_map = grid_cfg.get("muap_mapping", {})
            raw_pos = grid_cfg["positions"]
            corrected_positions = {}
            for new_idx, orig_idx in enumerate(local_active):
                key = muap_map.get(int(orig_idx), int(orig_idx))
                pos = raw_pos.get(key)
                if pos is not None:
                    corrected_positions[new_idx] = pos
            # Positions of rejected channels for visual masking
            for orig_idx in range(n_ch):
                if orig_idx in set(local_active):
                    continue
                key = muap_map.get(orig_idx, orig_idx)
                pos = raw_pos.get(key)
                if pos is not None:
                    rejected_pos_set.add(pos)

        if motor_units:
            props_ts = [mu.timestamps for mu in motor_units]
            props_src = [mu.source for mu in motor_units]

            props_list = compute_port_properties(
                all_timestamps=props_ts,
                all_sources=props_src,
                emg_port=emg_port,
                grid_positions=corrected_positions if corrected_positions is not None
                               else (grid_cfg["positions"] if grid_cfg else None),
                grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
                fsamp=self._fsamp,
            )
            for mu, p in zip(motor_units, props_list):
                mu.props = p

        self._ports[port_name] = motor_units

        # Restore flagged state persisted from a previous save
        for idx in decomp_data.get("flagged_mus", {}).get(port_name, []):
            if 0 <= idx < len(motor_units):
                motor_units[idx].flagged_duplicate = True

        self._grid_info[port_name] = grid_cfg
        self._rejected_ch_positions[port_name] = rejected_pos_set
        if emg_port is not None:
            self._emg_data[port_name] = emg_port

        logger.info("Port '%s': %d MUs, %d spikes%s",
                    port_name, len(motor_units),
                    sum(len(m.timestamps) for m in motor_units),
                    " (full)" if self._full_source_mode else "")
        return n_ch

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
        flagged_mus_per_port = {}

        for port_name in ports:
            mus = self._ports[port_name]
            flagged_mus_per_port[port_name] = [
                i for i, mu in enumerate(mus) if mu.flagged_duplicate
            ]

            if self._full_source_mode:
                save_ts = [self._ts_to_plateau_local(mu.timestamps) for mu in mus]
                save_src = [
                    (
                        mu.source[self._start_sample : self._end_sample]
                        if len(mu.source) > (self._end_sample - self._start_sample)
                        else mu.source
                    )
                    for mu in mus
                ]
            else:
                save_ts = [mu.timestamps for mu in mus]
                save_src = [mu.source for mu in mus]

            discharge_times.append(save_ts)
            pulse_trains.append(save_src)
            mu_filters.append([mu.mu_filter for mu in mus] or None)

            port_props = []
            for mu in mus:
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
            "flagged_mus": flagged_mus_per_port,
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

            # Pass peel_off_sequence through unchanged — all MUs are saved,
            # so no index remapping is needed. Remapping only happens when
            # units are physically deleted via "Delete All Flagged MUs".
            orig_peel = self._original_decomp_data.get("peel_off_sequence")
            if orig_peel is not None:
                save_data["peel_off_sequence"] = orig_peel

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

    def _delete_all_flagged(self):
        ports = list(self._ports.keys())
        total = sum(
            1 for p in ports for mu in self._ports[p] if mu.flagged_duplicate
        )
        if total == 0:
            self._update_status("No MUs are flagged for deletion")
            return

        reply = QMessageBox.question(
            self,
            "Delete Flagged MUs",
            f"Permanently delete {total} flagged MU(s) from the current session?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Remap peel_off_sequence before deletion so filter recalculation
        # continues to work correctly on the remaining units.
        if self._original_decomp_data is not None:
            orig_peel = self._original_decomp_data.get("peel_off_sequence")
            if orig_peel is not None:
                port_index_maps = {}
                for port_name in ports:
                    old_to_new = {}
                    new_idx = 0
                    for old_idx, mu in enumerate(self._ports[port_name]):
                        if not mu.flagged_duplicate:
                            old_to_new[old_idx] = new_idx
                            new_idx += 1
                    port_index_maps[port_name] = old_to_new

                new_peel = []
                for port_idx, port_name in enumerate(ports):
                    port_seq = orig_peel[port_idx] if port_idx < len(orig_peel) else []
                    old_to_new = port_index_maps[port_name]
                    remapped = []
                    for entry in port_seq:
                        uid = entry.get("accepted_unit_idx")
                        if uid is None:
                            remapped.append(entry)
                        elif uid in old_to_new:
                            remapped.append({**entry, "accepted_unit_idx": old_to_new[uid]})
                    new_peel.append(remapped)
                self._original_decomp_data["peel_off_sequence"] = new_peel

        for port_name in ports:
            kept = [mu for mu in self._ports[port_name] if not mu.flagged_duplicate]
            for i, mu in enumerate(kept):
                mu.id = i
            self._ports[port_name] = kept

        self._current_mu_idx = 0
        self._refresh_mu_combo()
        if self.mu_combo.count() > 0:
            self.mu_combo.setCurrentIndex(0)
        self._update_status(f"Deleted {total} flagged MU(s)")

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

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    @staticmethod
    def _mu_quality_key(mu: MotorUnit) -> tuple:
        """Return a sort key where a higher tuple means higher quality."""
        if mu.props is None:
            return (-float("inf"), -float("inf"), 0, -mu.id)
        sil = mu.props.sil if not np.isnan(mu.props.sil) else -float("inf")
        pnr = mu.props.pnr_db if not np.isnan(mu.props.pnr_db) else -float("inf")
        return (sil, pnr, mu.props.n_spikes, -mu.id)

    def _clear_duplicate_roles(self, kind: str):
        """Clear `kind` duplicate roles/partners; un-flag MUs not also deleted by the other kind."""
        other = "cross" if kind == "within" else "within"
        for mus in self._ports.values():
            for mu in mus:
                prev_delete = getattr(mu, f"{kind}_duplicate_role") == "delete"
                setattr(mu, f"{kind}_duplicate_role", None)
                setattr(mu, f"{kind}_duplicate_partners", [])
                if prev_delete and getattr(mu, f"{other}_duplicate_role") != "delete":
                    mu.flagged_duplicate = False

    def _flag_within_duplicates(self):
        """Detect and flag lower-quality within-port duplicate MUs for deletion."""
        if not _SPIKE_COMP_AVAILABLE:
            self._update_status("motor_unit_toolbox not available — cannot detect duplicates")
            return

        self._clear_duplicate_roles("within")

        for port_name, mus in self._ports.items():
            if len(mus) < 2:
                continue

            n_samples = max(len(mu.source) for mu in mus)
            spike_mat = build_spike_train_matrix(
                [mu.timestamps for mu in mus], n_samples
            )

            try:
                # rate_of_agreement_full returns full (n, n) RoA matrix
                roa, _ = _tb_spike_comp.rate_of_agreement_full(
                    spike_trains_ref=spike_mat,
                    spike_trains_test=spike_mat,
                    fs=int(round(self._fsamp)),
                )
            except Exception as exc:
                logger.warning("Within-port RoA failed for %s: %s", port_name, exc)
                continue

            n = len(mus)
            for i in range(n):
                for j in range(i + 1, n):
                    # Use max of both directions for a symmetric score
                    score = float(max(roa[i, j], roa[j, i]))
                    if score >= ROA_THRESHOLD:
                        mus[i].within_duplicate_partners.append(
                            (port_name, mus[j].id, score)
                        )
                        mus[j].within_duplicate_partners.append(
                            (port_name, mus[i].id, score)
                        )

            for mu in mus:
                if not mu.within_duplicate_partners:
                    continue
                partner_ids = {mid for (_, mid, _) in mu.within_duplicate_partners}
                partner_mus = [m for m in mus if m.id in partner_ids]
                best_partner = max(partner_mus, key=self._mu_quality_key)
                if self._mu_quality_key(mu) >= self._mu_quality_key(best_partner):
                    mu.within_duplicate_role = "keep"
                else:
                    mu.within_duplicate_role = "delete"
                    mu.flagged_duplicate = True

        n_flagged = sum(
            1 for mus in self._ports.values()
            for mu in mus if mu.within_duplicate_role == "delete"
        )
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(self._current_mu())
        self._update_status(
            f"Within-port duplicates: flagged {n_flagged} MU(s) for deletion"
        )

    def _flag_cross_duplicates(self):
        """Detect and flag lower-quality cross-port duplicate MUs for deletion."""
        if not _SPIKE_COMP_AVAILABLE:
            self._update_status("motor_unit_toolbox not available — cannot detect duplicates")
            return

        port_names = list(self._ports.keys())
        if len(port_names) < 2:
            self._update_status("Cross-port: only one port loaded — nothing to compare")
            return

        self._clear_duplicate_roles("cross")

        for idx_a in range(len(port_names)):
            for idx_b in range(idx_a + 1, len(port_names)):
                port_a, port_b = port_names[idx_a], port_names[idx_b]
                mus_a = self._ports[port_a]
                mus_b = self._ports[port_b]

                if not mus_a or not mus_b:
                    continue

                n_samples = max(
                    max(len(mu.source) for mu in mus_a),
                    max(len(mu.source) for mu in mus_b),
                )
                spike_mat_a = build_spike_train_matrix(
                    [mu.timestamps for mu in mus_a], n_samples
                )
                spike_mat_b = build_spike_train_matrix(
                    [mu.timestamps for mu in mus_b], n_samples
                )

                try:
                    # rate_of_agreement_full returns full (n_a, n_b) RoA matrix
                    roa, _ = _tb_spike_comp.rate_of_agreement_full(
                        spike_trains_ref=spike_mat_a,
                        spike_trains_test=spike_mat_b,
                        fs=int(round(self._fsamp)),
                    )
                except Exception as exc:
                    logger.warning("Cross-port RoA failed for %s vs %s: %s", port_a, port_b, exc)
                    continue

                na, nb = roa.shape[0], roa.shape[1]
                for i in range(min(len(mus_a), na)):
                    for j in range(min(len(mus_b), nb)):
                        score = float(roa[i, j])
                        if score >= ROA_THRESHOLD:
                            mus_a[i].cross_duplicate_partners.append(
                                (port_b, mus_b[j].id, score)
                            )
                            mus_b[j].cross_duplicate_partners.append(
                                (port_a, mus_a[i].id, score)
                            )

        # Assign cross-port roles: lower quality than any partner → delete
        for port_name, mus in self._ports.items():
            for mu in mus:
                if not mu.cross_duplicate_partners:
                    continue
                partner_mus = []
                for (pname, mid, _score) in mu.cross_duplicate_partners:
                    for pm in self._ports.get(pname, []):
                        if pm.id == mid:
                            partner_mus.append(pm)
                            break
                if not partner_mus:
                    continue
                best_partner = max(partner_mus, key=self._mu_quality_key)
                if self._mu_quality_key(mu) >= self._mu_quality_key(best_partner):
                    if mu.cross_duplicate_role != "delete":
                        mu.cross_duplicate_role = "keep"
                else:
                    mu.cross_duplicate_role = "delete"
                    mu.flagged_duplicate = True

        n_flagged = sum(
            1 for mus in self._ports.values()
            for mu in mus if mu.cross_duplicate_role == "delete"
        )
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(self._current_mu())
        self._update_status(
            f"Cross-port duplicates: flagged {n_flagged} MU(s) for deletion"
        )

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
                is_dup_delete = (
                    mu.within_duplicate_role == "delete"
                    or mu.cross_duplicate_role == "delete"
                )
                if is_dup_delete:
                    label += "  ⧉"
                self.mu_combo.addItem(label)
        self.mu_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # AUX / force overlay
    # ------------------------------------------------------------------

    def _refresh_aux_controls(self):
        aux_channels = []
        if self._original_decomp_data is not None:
            aux_channels = self._original_decomp_data.get("aux_channels") or []
        file_stem = self._loaded_path.stem if self._loaded_path else ""
        self.source_plot.set_aux_data(aux_channels, self._fsamp, file_stem)

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
        """Immediate cheap updates; expensive recompute+render deferred 120 ms."""
        mu = self._current_mu()
        if mu is not None:
            if source_changed:
                # Source signal changed (e.g. filter recalc) — redraw curve now
                self.source_plot.set_data(mu.source, mu.timestamps)
                if self._full_source_mode:
                    self.source_plot.set_plateau_region(self._start_sample, self._end_sample)
            self.fr_plot.set_data(mu.timestamps)
            self.source_plot.update_timestamps(mu.timestamps)

        self._refresh_mu_combo()
        self.mu_combo.blockSignals(True)
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self.mu_combo.blockSignals(False)
        self._update_status(msg)
        self.data_modified.emit()

        # Accumulate source_changed across rapid edits, then flush once
        self._pending_source_changed |= source_changed
        self._props_timer.start()

    def _flush_props_update(self):
        """Runs after editing pauses: recomputes MU properties and refreshes MUAP + quality."""
        mu = self._current_mu()
        if mu is None:
            self._pending_source_changed = False
            return
        grid_cfg = self._grid_info.get(self._current_port)
        emg_port = self._emg_data.get(self._current_port)
        mu.props = recompute_unit_properties(
            mu_props=mu.props or MUProperties(),
            new_timestamps=mu.timestamps,
            source=mu.source,
            emg_port=emg_port,
            grid_positions=grid_cfg["positions"] if grid_cfg else None,
            grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
            fsamp=self._fsamp,
        )
        self._pending_source_changed = False
        self._plot_muap()
        self._update_quality_panel(mu)

    def _update_quality_panel(self, mu: Optional[MotorUnit]):
        if mu is None:
            self.quality_bar.clear_properties()
            return
        if mu.props is not None:
            self.quality_bar.set_properties(
                mu.props,
                within_role=mu.within_duplicate_role,
                within_partners=mu.within_duplicate_partners,
                cross_role=mu.cross_duplicate_role,
                cross_partners=mu.cross_duplicate_partners,
            )
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
        rejected_pos = self._rejected_ch_positions.get(self._current_port, set())

        if grid_cfg is not None:
            self._render_muap_grid(muap_grid, grid_cfg, rejected_pos)
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_grid(
                    muap_grid, grid_cfg, rejected_pos, self._current_mu_idx
                )
        else:
            n_ch = muap_grid.shape[0]
            waveforms = [muap_grid[i, 0] for i in range(n_ch)]
            self._render_muap_stacked(waveforms, list(range(n_ch)))
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_stacked(
                    waveforms, list(range(n_ch)), self._current_mu_idx
                )

    def _render_muap_grid(self, muap_grid: np.ndarray, grid_cfg: dict, rejected_positions: set = None):
        """Render MUAPs in physical grid layout (portrait, rows × cols).

        muap_grid: (rows, cols, n_samples) from compute_port_properties.
        On the first call (or when grid shape changes) all PlotItems are built and
        stored; on subsequent calls only waveform data and amplitudes are updated,
        avoiding expensive scene teardown/rebuild.
        """
        if rejected_positions is None:
            rejected_positions = set()

        rows, cols = grid_cfg["grid_shape"]
        electrode_positions = set(grid_cfg["positions"].values())
        n_samples = muap_grid.shape[2] if muap_grid.ndim == 3 else 409

        valid_wavs = [
            muap_grid[r, c]
            for r in range(min(rows, muap_grid.shape[0]))
            for c in range(min(cols, muap_grid.shape[1]))
            if (r, c) in electrode_positions
            and (r, c) not in rejected_positions
            and len(muap_grid[r, c]) > 0
            and np.any(muap_grid[r, c] != 0)
        ]
        amp = np.max(np.abs(np.concatenate(valid_wavs))) * 1.2 if valid_wavs else 1.0
        grid_key = (rows, cols, n_samples, frozenset(rejected_positions))

        title_html = (
            f"<span style='color:{COLORS['foreground']};font-size:10pt;'>"
            f"MU {self._current_mu_idx}</span>"
        )

        if grid_key == self._muap_grid_key and self._muap_cell_plots:
            # Fast path: only update amplitudes and waveform data in existing plots
            for p in self._muap_cell_plots.values():
                p.setYRange(-amp, amp, padding=0)
            for (r, c), item in self._muap_waveform_items.items():
                wav = (muap_grid[r, c]
                       if r < muap_grid.shape[0] and c < muap_grid.shape[1]
                       else None)
                if wav is not None and len(wav) > 0 and np.any(wav != 0):
                    item.setData(wav)
                else:
                    item.setData([])
            if self._muap_title_label is not None:
                self._muap_title_label.setText(title_html)
            return

        # Slow path: full rebuild
        self.muap_widget.clear()
        self._muap_cell_plots = {}
        self._muap_waveform_items = {}

        lbl_style = f"color:{COLORS.get('text_dim','#6c7086')}; font-size:7pt;"

        def _add_lbl(widget, text, row, col, **kw):
            lbl = widget.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        # Row 0: title. Row 1: column headers. Rows 2+: data. Col 0: row labels.
        self._muap_title_label = _add_lbl(
            self.muap_widget, title_html, 0, 0, colspan=cols + 1, justify="center"
        )
        _add_lbl(self.muap_widget, f"<span style='{lbl_style}'></span>", 1, 0, justify="center")
        for c in range(cols):
            _add_lbl(self.muap_widget, f"<span style='{lbl_style}'>{c + 1}</span>", 1, c + 1, justify="center")
        for r in range(rows):
            _add_lbl(self.muap_widget, f"<span style='{lbl_style}'>{r + 1}</span>", r + 2, 0, justify="center")

        _rej_bg = (50, 30, 30)
        _empty_bg = (28, 28, 28)
        gl = self.muap_widget.ci.layout

        for r in range(rows):
            for c in range(cols):
                p = self.muap_widget.addPlot(row=r + 2, col=c + 1)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0)
                p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
                p.setMinimumWidth(0)
                p.setMinimumHeight(0)
                self._muap_cell_plots[(r, c)] = p

                rc = (r, c)
                if rc in rejected_positions:
                    p.getViewBox().setBackgroundColor(_rej_bg)
                    p.plot([0, n_samples], [0, 0], pen=pg.mkPen(color=(140, 60, 60), width=1))
                elif rc not in electrode_positions:
                    p.getViewBox().setBackgroundColor(_empty_bg)
                else:
                    # Pre-create waveform item; data filled below
                    item = p.plot([], pen=pg.mkPen(color=COLORS["info"], width=1.5))
                    self._muap_waveform_items[rc] = item

        gl.setSpacing(0)
        gl.setHorizontalSpacing(6)
        gl.setColumnMinimumWidth(0, 14)
        gl.setColumnStretchFactor(0, 0)
        for c in range(cols):
            gl.setColumnMinimumWidth(c + 1, 0)
            gl.setColumnStretchFactor(c + 1, 1)
        for r in range(2):
            gl.setRowMinimumHeight(r, 0)
            gl.setRowStretchFactor(r, 0)
        for r in range(rows):
            gl.setRowMinimumHeight(r + 2, 0)
            gl.setRowStretchFactor(r + 2, 1)

        for (r, c), item in self._muap_waveform_items.items():
            if r < muap_grid.shape[0] and c < muap_grid.shape[1]:
                wav = muap_grid[r, c]
                if len(wav) > 0 and np.any(wav != 0):
                    item.setData(wav)

        self._muap_grid_key = grid_key


    def _render_muap_stacked(self, waveforms, ch_indices):
        self.muap_widget.clear()
        self._muap_grid_key = None  # force grid rebuild on next _render_muap_grid call
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
        self._muap_grid_key = None
        p = self.muap_widget.addPlot(row=0, col=0)
        p.hideAxis("left")
        p.hideAxis("bottom")
        p.setMouseEnabled(x=False, y=False)
        p.setXRange(0, 1)
        p.setYRange(0, 1)
        t = pg.TextItem(message, color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        t.setPos(0.5, 0.5)
        p.addItem(t)
        if self._muap_popout and self._muap_popout.isVisible():
            self._muap_popout.clear(message)
