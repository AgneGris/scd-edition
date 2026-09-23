"""
Configuration Tab - EMG data loading and electrode configuration.
"""

import contextlib
import copy
import json
import logging
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scd_app.core.config import (
    ConfigManager,
    DecompositionConfig,
    ElectrodeConfig,
    FilterConfig,
    PortConfig,
)
from scd_app.gui.style.styling import (
    COLORS,
    FONT_FAMILY,
    FONT_SIZES,
    get_button_style,
    get_label_style,
)
from scd_app.gui.widgets.import_data_dialog import ImportDataDialog
from scd_app.io.data_inspector import inspect_recording
from scd_app.io.data_loader import (
    can_read_field,
    format_matches_extension,
    load_field,
    load_layout,
    load_metadata,
)

logger = logging.getLogger(__name__)


class ChannelAllocationBar(QFrame):
    """Visual bar showing channel allocation across all grids and aux channels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.max_channels = 256
        self.allocations = []

        self.setStyleSheet(
            f"""
            ChannelAllocationBar {{
                background-color: {COLORS["background_input"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 6px;
            }}
        """
        )

    def set_max_channels(self, n: int):
        self.max_channels = n
        self.update()

    def set_allocations(self, allocations: list[tuple[int, int, str, str]]):
        self.allocations = allocations
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.max_channels == 0:
            return

        from PySide6.QtCore import QRect
        from PySide6.QtGui import QColor, QFont, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 10
        bar_height = 25
        bar_y = (self.height() - bar_height) // 2
        bar_width = self.width() - 2 * margin

        bg_rect = QRect(margin, bar_y, bar_width, bar_height)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["background"]))
        painter.drawRoundedRect(bg_rect, 3, 3)

        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1))
        font = QFont(FONT_FAMILY, 8)
        painter.setFont(font)

        ticks = [0, 64, 128, 192, self.max_channels]
        ticks = [t for t in ticks if t <= self.max_channels]
        for ch in ticks:
            x = margin + int((ch / self.max_channels) * bar_width)
            painter.drawLine(x, bar_y, x, bar_y + bar_height)
            painter.drawText(x - 10, bar_y + bar_height + 15, f"{ch}")

        for start_ch, end_ch, name, color in self.allocations:
            if end_ch > self.max_channels:
                color = COLORS["error"]
                end_ch_display = self.max_channels
            else:
                end_ch_display = end_ch

            x_start = margin + int((start_ch / self.max_channels) * bar_width)
            x_end = margin + int((end_ch_display / self.max_channels) * bar_width)
            segment_width = max(x_end - x_start, 3)

            segment_rect = QRect(x_start, bar_y + 2, segment_width, bar_height - 4)
            painter.setPen(QPen(QColor(color), 2))
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(segment_rect, 2, 2)

            if segment_width > 40:
                painter.setPen(QPen(QColor("#ffffff")))
                label_font = QFont(FONT_FAMILY, 8, QFont.Weight.Bold)
                painter.setFont(label_font)
                painter.drawText(segment_rect, Qt.AlignmentFlag.AlignCenter, name)


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores touchpad/mouse-wheel scrolling."""

    def wheelEvent(self, event):
        event.ignore()


class GridCard(QFrame):
    """Card widget for configuring a single electrode grid."""

    remove_requested = Signal(object)
    changed = Signal()

    GRID_COLORS = ["#4a9eff", "#a78bfa", "#48BB78", "#F6AD55", "#ff6b9d", "#63B3ED"]

    ELECTRODE_CONFIGS = {
        "Surface": {
            "Grid (GR04MM1305)": {
                "rows": 13,
                "cols": 5,
                "spacing_mm": 4.0,
                "n_channels": 64,
            },
            "Grid (GR08MM1305)": {
                "rows": 13,
                "cols": 5,
                "spacing_mm": 8.0,
                "n_channels": 64,
            },
            "Grid (GR10MM0808)": {
                "rows": 8,
                "cols": 8,
                "spacing_mm": 10.0,
                "n_channels": 64,
            },
            "Grid (HD02MM0808)": {
                "rows": 8,
                "cols": 8,
                "spacing_mm": 2.0,
                "n_channels": 64,
            },
            "Grid (HD04MM1305)": {
                "rows": 13,
                "cols": 5,
                "spacing_mm": 4.0,
                "n_channels": 64,
            },
            "Grid (HD04MM1606)": {
                "rows": 16,
                "cols": 6,
                "spacing_mm": 4.0,
                "n_channels": 96,
            },
            "Grid (HD08MM1606)": {
                "rows": 16,
                "cols": 6,
                "spacing_mm": 8.0,
                "n_channels": 96,
            },
            "Grid (HD08MM1606, channels 17-96)": {
                "rows": 16,
                "cols": 5,
                "spacing_mm": 8.0,
                "n_channels": 80,
            },
            "Grid (HD08MM1305)": {
                "rows": 13,
                "cols": 5,
                "spacing_mm": 8.0,
                "n_channels": 64,
            },
            "Grid (HD10MM0804)": {
                "rows": 8,
                "cols": 4,
                "spacing_mm": 10.0,
                "n_channels": 32,
            },
            "Grid (HD05MM0804)": {
                "rows": 8,
                "cols": 4,
                "spacing_mm": 5.0,
                "n_channels": 32,
            },
            "Grid (SIM10X32)": {
                "rows": 10,
                "cols": 32,
                "spacing_mm": 4.0,
                "n_channels": 320,
            },
            # Ultra-high-density 4x4 array, 250 um pitch.
            "Grid (UltraHD 4x4)": {
                "rows": 4,
                "cols": 4,
                "spacing_mm": 0.25,
                "n_channels": 16,
            },
        },
        "Intramuscular": {
            "Thin-film (40ch)": {
                "rows": 20,
                "cols": 2,
                "spacing_mm": 2.5,
                "n_channels": 40,
            },
            "Wire needle": {"rows": 1, "cols": 16, "spacing_mm": 4.0, "n_channels": 16},
            "Myomatrix": {"rows": 1, "cols": 32, "spacing_mm": 4.0, "n_channels": 32},
        },
    }

    def __init__(self, index: int, color: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.color = color

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(55)

        self._setup_ui()
        self._apply_styling()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(8)

        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(4, 20)
        self.color_indicator.setStyleSheet(
            f"background-color: {self.color}; border-radius: 2px;"
        )
        main_layout.addWidget(self.color_indicator)

        # Type badge
        type_label = QLabel("EMG")
        type_label.setFixedWidth(36)
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {self.color}30;
                color: {self.color};
                border-radius: 3px;
                font-size: {FONT_SIZES["small"]};
                font-weight: bold;
                padding: 1px 4px;
            }}
        """
        )
        main_layout.addWidget(type_label)

        # Name
        self.name_edit = QLineEdit(f"Grid_{self.index}")
        self.name_edit.setPlaceholderText("e.g., Biceps")
        self.name_edit.textChanged.connect(self._notify_changed)
        main_layout.addWidget(self.name_edit, stretch=2)

        # Muscle
        self.muscle_edit = QLineEdit()
        self.muscle_edit.setPlaceholderText("Muscle")
        self.muscle_edit.textChanged.connect(self._notify_changed)
        main_layout.addWidget(self.muscle_edit, stretch=2)

        # Electrode type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Surface", "Intramuscular"])
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        main_layout.addWidget(self.type_combo, stretch=2)

        # Electrode config
        self.config_combo = QComboBox()
        self.config_combo.currentTextChanged.connect(self._notify_changed)
        main_layout.addWidget(self.config_combo, stretch=2)

        # Channel range — managed externally by ConfigTab._recalculate_all_channel_ranges
        self.start_spin = _NoScrollSpinBox()
        self.start_spin.setRange(0, 2048)
        self.start_spin.setValue(0)
        main_layout.addWidget(self.start_spin, stretch=1)

        self.end_spin = _NoScrollSpinBox()
        self.end_spin.setRange(0, 2048)
        self.end_spin.setValue(64)
        main_layout.addWidget(self.end_spin, stretch=1)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet(get_label_style(size="small"))
        main_layout.addWidget(self.status_label, stretch=1)

        # Remove
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(32, 32)
        self.remove_btn.setToolTip("Remove Grid")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_muted"]};
                border-radius: 10px;
                font-weight: bold; font-size: 18pt;
            }}
            QPushButton:hover {{
                background-color: {COLORS["error"]}40;
                color: {COLORS["error_bright"]};
            }}
        """
        )
        main_layout.addWidget(self.remove_btn)

        self._on_type_change()

    def _apply_styling(self):
        self.setStyleSheet(
            f"""
            GridCard {{
                background-color: {COLORS["background_light"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 6px;
            }}
            QLabel {{
                color: {COLORS["foreground"]};
                font-family: '{FONT_FAMILY}';
            }}
            QPushButton {{
                color: {COLORS["foreground"]};
            }}
        """
        )

    def update_index(self, index: int):
        self.index = index
        if self.name_edit.text().startswith("Grid_"):
            self.name_edit.setText(f"Grid_{index}")

    def _notify_changed(self, *_args):
        """Discard widget signal payloads before emitting the parameterless signal."""
        self.changed.emit()

    def _on_type_change(self):
        electrode_type = self.type_combo.currentText()
        current_config = self.config_combo.currentText()
        self.config_combo.clear()
        configs = list(self.ELECTRODE_CONFIGS[electrode_type].keys())
        self.config_combo.addItems(configs)
        if current_config in configs:
            self.config_combo.setCurrentText(current_config)
        self.changed.emit()

    def set_validation_status(self, is_valid: bool, message: str = ""):
        if is_valid:
            self.status_label.setText("")
        else:
            self.status_label.setText(f"⚠ {message}")
            self.status_label.setStyleSheet(
                get_label_style(size="small", color="warning")
            )

    def get_data(self) -> dict:
        return {
            "name": self.name_edit.text(),
            "muscle": self.muscle_edit.text(),
            "type": self.type_combo.currentText(),
            "config": self.config_combo.currentText(),
            "start_chan": self.start_spin.value(),
            "end_chan": self.end_spin.value(),
            "color": self.color,
        }

    def get_geometry(self) -> tuple[int, int, float, int]:
        electrode_type = self.type_combo.currentText()
        config_name = self.config_combo.currentText()
        if electrode_type in self.ELECTRODE_CONFIGS:
            configs = self.ELECTRODE_CONFIGS[electrode_type]
            if config_name in configs:
                cfg = configs[config_name]
                n_ch = cfg.get("n_channels", cfg["rows"] * cfg["cols"])
                return cfg["rows"], cfg["cols"], cfg["spacing_mm"], n_ch
        return 0, 0, 0.0, 0

    def get_channel_count(self) -> int:
        _, _, _, n_ch = self.get_geometry()
        return n_ch if n_ch > 0 else self.end_spin.value() - self.start_spin.value()

    def get_channel_range(self) -> tuple[int, int]:
        return self.start_spin.value(), self.end_spin.value()

    def set_start_channel(self, start: int):
        self.start_spin.setValue(start)

    def set_end_channel(self, end: int):
        self.end_spin.setValue(end)

    def set_values(
        self,
        name: str,
        muscle: str,
        electrode_type: str,
        config: str,
        start: int,
        end: int,
    ):
        self.name_edit.setText(name)
        self.muscle_edit.setText(muscle)
        type_idx = self.type_combo.findText(electrode_type)
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)
        config_idx = self.config_combo.findText(config)
        if config_idx >= 0:
            self.config_combo.setCurrentIndex(config_idx)
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)


class AuxChannelCard(QFrame):
    """Card widget for configuring an auxiliary channel group (force, target, etc.)."""

    remove_requested = Signal(object)
    changed = Signal()

    AUX_TYPES = [
        "Force",
        "Torque",
        "Trigger",
        "Target",
        "Path",
        "Angle",
        "Position",
        "Other",
    ]
    AUX_COLOR = "#F6AD55"

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(55)

        self._setup_ui()
        self._apply_styling()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(8)

        # Color indicator
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(4, 20)
        self.color_indicator.setStyleSheet(
            f"background-color: {self.AUX_COLOR}; border-radius: 2px;"
        )
        main_layout.addWidget(self.color_indicator)

        # Type badge
        type_label = QLabel("AUX")
        type_label.setFixedWidth(36)
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {self.AUX_COLOR}30;
                color: {self.AUX_COLOR};
                border-radius: 3px;
                font-size: {FONT_SIZES["small"]};
                font-weight: bold;
                padding: 1px 4px;
            }}
        """
        )
        main_layout.addWidget(type_label)

        # Name
        self.name_edit = QLineEdit(f"Aux_{self.index}")
        self.name_edit.setPlaceholderText("e.g., Force_raw")
        self.name_edit.textChanged.connect(self._notify_changed)
        main_layout.addWidget(self.name_edit, stretch=2)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.AUX_TYPES)
        self.type_combo.currentTextChanged.connect(self._notify_changed)
        main_layout.addWidget(self.type_combo, stretch=2)

        # Source: from main signal, an acquisition-system auxiliary stream, or a
        # named field in the data file.
        self.source_combo = QComboBox()
        self.source_combo.addItems(
            ["Signal channels", "Auxiliary stream", "Data file field"]
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_change)
        main_layout.addWidget(self.source_combo, stretch=2)

        # Channel range (for signal channels)
        self.start_spin = _NoScrollSpinBox()
        self.start_spin.setRange(0, 2048)
        self.start_spin.setValue(0)
        self.start_spin.valueChanged.connect(self._notify_changed)
        main_layout.addWidget(self.start_spin, stretch=1)

        self.end_spin = _NoScrollSpinBox()
        self.end_spin.setRange(0, 2048)
        self.end_spin.setValue(0)
        self.end_spin.valueChanged.connect(self._notify_changed)
        main_layout.addWidget(self.end_spin, stretch=1)

        # Field path (for "Data file field") — read straight out of the data file,
        # e.g. a MATLAB struct field holding an already-calibrated force trace.
        self.field_edit = QLineEdit()
        self.field_edit.setPlaceholderText("e.g. signal.path")
        self.field_edit.setToolTip(
            "Name of the field inside the data file holding this signal.\n"
            "Dot notation walks MATLAB structs (e.g. signal.path); "
            "HDF5 uses slashes (e.g. signal/force)."
        )
        self.field_edit.textChanged.connect(self._notify_changed)
        self.field_edit.setVisible(False)
        main_layout.addWidget(self.field_edit, stretch=2)

        # Unit
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("Unit")
        self.unit_edit.textChanged.connect(self._notify_changed)
        main_layout.addWidget(self.unit_edit, stretch=2)

        # MVC value
        mvc_label = QLabel("MVC:")
        mvc_label.setFixedWidth(32)
        mvc_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        mvc_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: {FONT_SIZES['small']};"
        )
        main_layout.addWidget(mvc_label)

        self.mvc_edit = QLineEdit()
        self.mvc_edit.setPlaceholderText("e.g. 70")
        self.mvc_edit.setFixedWidth(72)
        self.mvc_edit.setToolTip(
            "Maximum Voluntary Contraction value in mV (same units as the loaded signal).\n"
            "OTB displays force in Volts — multiply by 1000 to get mV (e.g. 0.049 V → 49).\n"
            "Used to normalise force as % MVC. Leave blank to use relative normalisation."
        )
        self.mvc_edit.textChanged.connect(self._notify_changed)
        main_layout.addWidget(self.mvc_edit)

        mvc_browse = QPushButton("...")
        mvc_browse.setFixedSize(24, 24)
        mvc_browse.setToolTip(
            "Load MVC: opens a data file and uses its maximum as the MVC value"
        )
        mvc_browse.clicked.connect(self._browse_mvc_file)
        mvc_browse.setStyleSheet(
            f"QPushButton {{ background: {COLORS['background_input']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px; font-size: {FONT_SIZES['small']}; }}"
            f"QPushButton:hover {{ background: {COLORS['background_hover']}; "
            f"color: {COLORS['foreground']}; }}"
        )
        main_layout.addWidget(mvc_browse)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet(get_label_style(size="small"))
        self.status_label.setMaximumWidth(80)
        main_layout.addWidget(self.status_label)

        # Remove
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setToolTip("Remove Channel")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_muted"]};
                border-radius: 10px;
                font-weight: bold; font-size: 14pt;
            }}
            QPushButton:hover {{
                background-color: {COLORS["error"]}40;
                color: {COLORS["error_bright"]};
            }}
        """
        )
        main_layout.addWidget(self.remove_btn)

    def _apply_styling(self):
        self.setStyleSheet(
            f"""
            AuxChannelCard {{
                background-color: {COLORS["background_light"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 6px;
            }}
            QLabel {{
                color: {COLORS["foreground"]};
                font-family: '{FONT_FAMILY}';
            }}
        """
        )

    def _notify_changed(self, *_args):
        """Discard widget signal payloads before emitting the parameterless signal."""
        self.changed.emit()

    def _on_source_change(self, idx: int):
        """Show the channel range for range-based sources, the field path otherwise."""
        is_range = idx in (0, 1)
        is_field = idx == 2
        self.start_spin.setVisible(is_range)
        self.end_spin.setVisible(is_range)
        self.field_edit.setVisible(is_field)
        self.changed.emit()

    def _browse_mvc_file(self):
        """Open a data file and use its maximum value as the MVC."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load MVC Signal",
            "",
            "Data Files (*.npy *.mat *.csv *.txt);;All Files (*.*)",
        )
        if not path:
            return
        try:
            p = Path(path)
            if p.suffix == ".npy":
                data = np.load(path)
            elif p.suffix == ".mat":
                import scipy.io

                mat = scipy.io.loadmat(path)
                data = next(v for k, v in mat.items() if not k.startswith("_"))
            else:
                data = np.loadtxt(path)
            mvc_val = float(np.asarray(data).flatten().max())
            self.mvc_edit.setText(f"{mvc_val:g}")
        except Exception as e:
            QMessageBox.warning(
                self, "Load Failed", f"Could not load MVC from file:\n{e}"
            )

    def update_index(self, index: int):
        self.index = index
        if self.name_edit.text().startswith("Aux_"):
            self.name_edit.setText(f"Aux_{index}")

    def set_validation_status(self, is_valid: bool, message: str = ""):
        if is_valid:
            self.status_label.setText("")
        else:
            self.status_label.setText(f"⚠ {message}")
            self.status_label.setStyleSheet(
                get_label_style(size="small", color="warning")
            )

    # Combo index ↔ the value persisted in the channel-config JSON
    SOURCES = ["signal", "aux_file", "data_field"]

    def get_source(self) -> str:
        """'signal' = main signal channels, 'aux_file' = .sip streams,
        'data_field' = a named field read straight out of the data file."""
        idx = self.source_combo.currentIndex()
        return self.SOURCES[idx] if 0 <= idx < len(self.SOURCES) else "signal"

    def get_field_path(self) -> str:
        return self.field_edit.text().strip()

    def get_channel_range(self) -> tuple[int, int]:
        return self.start_spin.value(), self.end_spin.value()

    def get_data(self) -> dict:
        d = {
            "index": self.index,
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText().lower(),
            "source": self.get_source(),
            "start_chan": self.start_spin.value(),
            "end_chan": self.end_spin.value(),
            "unit": self.unit_edit.text(),
        }
        field_path = self.get_field_path()
        if field_path:
            d["field_path"] = field_path
        mvc_text = self.mvc_edit.text().strip()
        if mvc_text:
            with contextlib.suppress(ValueError):
                d["mvc"] = float(mvc_text)
        return d

    def set_values(
        self,
        name: str,
        aux_type: str,
        source: str = "signal",
        start: int = 0,
        end: int = 0,
        unit: str = "",
        mvc: float | None = None,
        field_path: str = "",
    ):
        self.name_edit.setText(name)
        type_idx = self.type_combo.findText(aux_type, Qt.MatchFlag.MatchFixedString)
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)
        src_idx = self.SOURCES.index(source) if source in self.SOURCES else 0
        self.source_combo.setCurrentIndex(src_idx)
        # setCurrentIndex only fires _on_source_change when the index actually
        # changes, so apply the visibility rules explicitly.
        self._on_source_change(src_idx)
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.field_edit.setText(field_path)
        self.unit_edit.setText(unit)
        if mvc is not None:
            self.mvc_edit.setText(f"{mvc:g}")
        else:
            self.mvc_edit.clear()


class ConfigTab(QWidget):
    """Streamlined configuration tab for EMG data loading and channel setup."""

    config_applied = Signal(object, list)

    # 6 quaternion channels appended after each novecento-type grid (i.e. HD...) EMG channels
    HD_QUATERNION_CHANNELS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = ConfigManager()
        self.emg_path: Path | None = None
        self.emg_paths: list[Path] = []  # all selected files for batch
        self.max_channels: int = 256
        # False whenever max_channels is a placeholder rather than a count read
        # from the file — a wrong loader must not masquerade as a small file.
        self._channel_count_known: bool = False
        self.file_metadata: dict = {}
        self._metadata_key = None
        self._metadata_error = None
        self.grid_cards: list[GridCard] = []
        self.aux_cards: list[AuxChannelCard] = []

        self._setup_ui()
        self._show_initial_state()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Session Configuration")
        title.setStyleSheet(get_label_style(size="title", bold=True))
        main_layout.addWidget(title)

        main_layout.addWidget(self._create_file_section(), stretch=0)
        main_layout.addWidget(self._create_channels_section(), stretch=1)
        main_layout.addWidget(self._create_summary_section(), stretch=0)

    def _create_file_section(self) -> QGroupBox:
        group = QGroupBox("1. Load EMG Data")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Choosing data is one workflow. Format/array controls below describe
        # how the selected recording should be read; they are not alternative
        # file-selection actions.
        file_layout = QHBoxLayout()
        file_label = QLabel("Recording:")
        file_label.setStyleSheet(get_label_style(size="normal"))

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No recording selected")
        self.path_edit.setReadOnly(True)

        self.choose_recording_btn = QPushButton("Choose recording…")
        self.choose_recording_btn.setFixedWidth(180)
        self.choose_recording_btn.clicked.connect(self._browse_file)
        self.choose_recording_btn.setStyleSheet(
            get_button_style(bg_color="accent", padding=8)
        )

        self.choose_batch_btn = QPushButton("Choose batch…")
        self.choose_batch_btn.setFixedWidth(150)
        self.choose_batch_btn.setToolTip(
            "Select multiple recordings that share one data format and channel layout."
        )
        self.choose_batch_btn.clicked.connect(self._browse_files_batch)
        self.choose_batch_btn.setStyleSheet(
            get_button_style(bg_color="accent", padding=8)
        )

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.path_edit, stretch=1)
        file_layout.addWidget(self.choose_recording_btn)
        file_layout.addWidget(self.choose_batch_btn)
        layout.addLayout(file_layout)

        self.file_info_label = QLabel()
        self.file_info_label.setStyleSheet(
            get_label_style(size="small", color="text_dim")
        )
        layout.addWidget(self.file_info_label)

        loader_layout = QHBoxLayout()
        loader_label = QLabel("Data Format:")
        loader_label.setStyleSheet(get_label_style(size="normal"))
        self.loader_combo = QComboBox()
        self._populate_loader_presets()
        self.loader_combo.currentTextChanged.connect(self._on_loader_changed)
        self.inspect_arrays_btn = QPushButton("Inspect arrays…")
        self.inspect_arrays_btn.setEnabled(False)
        self.inspect_arrays_btn.setToolTip(
            "Advanced: choose the EMG array inside the selected MATLAB, HDF5, "
            "NumPy, CSV, or text recording."
        )
        self.inspect_arrays_btn.clicked.connect(self._inspect_selected_file)
        self.inspect_arrays_btn.setStyleSheet(
            get_button_style(bg_color="background_light", padding=8)
        )
        loader_layout.addWidget(loader_label)
        loader_layout.addWidget(self.loader_combo, stretch=1)
        loader_layout.addWidget(self.inspect_arrays_btn)
        layout.addLayout(loader_layout)

        # Path / orientation overrides for generic array-based formats.
        self.emg_path_row = QWidget()
        path_row_layout = QHBoxLayout(self.emg_path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(8)
        emg_path_label = QLabel("Array path:")
        emg_path_label.setStyleSheet(get_label_style(size="normal"))
        self.emg_path_edit = QLineEdit()
        self.emg_path_edit.setPlaceholderText("e.g. signal/data")
        self.emg_path_edit.textChanged.connect(self._on_emg_path_changed)
        orient_label = QLabel("Orientation:")
        orient_label.setStyleSheet(get_label_style(size="normal"))
        self.emg_orientation_combo = QComboBox()
        self.emg_orientation_combo.addItems(["auto", "samples_first", "channels_first"])
        self.emg_orientation_combo.currentTextChanged.connect(self._on_emg_path_changed)
        path_row_layout.addWidget(emg_path_label)
        path_row_layout.addWidget(self.emg_path_edit, stretch=2)
        path_row_layout.addWidget(orient_label)
        path_row_layout.addWidget(self.emg_orientation_combo)
        self.emg_path_row.setVisible(False)
        layout.addWidget(self.emg_path_row)

        fs_layout = QHBoxLayout()
        fs_label = QLabel("Sampling Rate:")
        fs_label.setStyleSheet(get_label_style(size="normal"))
        self.fsamp_edit = QLineEdit("2048")
        self.fsamp_edit.setFixedWidth(100)
        self.fsamp_edit.setValidator(QIntValidator(1, 100000))
        self.fsamp_edit.textChanged.connect(self._on_fsamp_changed)
        fs_hz = QLabel("Hz")
        fs_hz.setStyleSheet(get_label_style(size="normal", color="text_secondary"))
        fs_layout.addWidget(fs_label)
        fs_layout.addWidget(self.fsamp_edit)
        fs_layout.addWidget(fs_hz)

        # Sampling rate is the file's native rate; decimation is applied by the
        # loader on read, so the decomposition sees native / factor.
        fs_layout.addSpacing(16)
        decimate_label = QLabel("Decimate by:")
        decimate_label.setStyleSheet(get_label_style(size="normal"))
        self.decimate_spin = _NoScrollSpinBox()
        self.decimate_spin.setRange(1, 64)
        self.decimate_spin.setValue(1)
        # Match the sampling-rate box: the global QSpinBox style pads 12 px per
        # side and reserves the arrow column, so anything narrower clips the digits.
        self.decimate_spin.setFixedWidth(100)
        self.decimate_spin.setToolTip(
            "Integer factor by which the loader reduces the sampling rate "
            "(anti-aliased). 1 = keep the file's native rate."
        )
        self.decimate_spin.valueChanged.connect(self._on_decimate_changed)
        self.delivered_fs_label = QLabel("")
        self.delivered_fs_label.setStyleSheet(
            get_label_style(size="small", color="text_dim")
        )
        fs_layout.addWidget(decimate_label)
        fs_layout.addWidget(self.decimate_spin)
        fs_layout.addWidget(self.delivered_fs_label)
        fs_layout.addStretch()
        layout.addLayout(fs_layout)

        # Output directory
        out_layout = QHBoxLayout()
        out_label = QLabel("Output Folder:")
        out_label.setStyleSheet(get_label_style(size="normal"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Same as input file...")
        self.output_dir_edit.setReadOnly(True)
        out_browse = QPushButton("Browse...")
        out_browse.setFixedWidth(200)
        out_browse.clicked.connect(self._browse_output_dir)
        out_browse.setStyleSheet(get_button_style(bg_color="accent", padding=8))
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.output_dir_edit, stretch=1)
        out_layout.addWidget(out_browse)
        layout.addLayout(out_layout)

        return group

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(Path.cwd())
        )
        if path:
            self.output_dir_edit.setText(path)

    def _browse_files_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select EMG Data Files",
            str(Path.cwd()),
            "EMG Files (*.mat *.npy *.csv *.txt *.h5 *.hdf5 *.otb+ *.otb4 *.rhs);;"
            "All Files (*.*)",
        )
        if paths:
            self.emg_paths = [Path(p) for p in paths]
            self.emg_path = self.emg_paths[0]  # first file for channel estimation
            self.path_edit.setText(
                f"{len(paths)} files selected (first: {self.emg_path.name})"
            )
            self._auto_select_loader(self.emg_path)
            self._update_inspect_action()
            inspected = False
            if self._selected_file_needs_inspection():
                inspected = self._inspect_selected_file()
            if not inspected:
                self._refresh_file_metadata()
                self._update_file_info()
                self._update_summary()

            if not self.grid_cards:
                self._add_grid()

    def _create_channels_section(self) -> QGroupBox:
        group = QGroupBox("2. Configure Channels")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # Action bar
        action_layout = QHBoxLayout()

        add_grid_btn = QPushButton("+ Add Grid")
        add_grid_btn.clicked.connect(self._add_grid)
        add_grid_btn.setStyleSheet(get_button_style(bg_color="success", padding=8))

        add_aux_btn = QPushButton("+ Add Aux")
        add_aux_btn.clicked.connect(lambda: self._add_aux_channel())
        add_aux_btn.setStyleSheet(get_button_style(bg_color="accent", padding=8))

        self.channel_summary_label = QLabel()
        self.channel_summary_label.setStyleSheet(
            get_label_style(size="small", color="text_dim")
        )

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all_channels)
        clear_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_muted"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 8px 12px;
                font-size: {FONT_SIZES["normal"]};
            }}
            QPushButton:hover {{
                background-color: {COLORS["background_hover"]};
                color: {COLORS["foreground"]};
            }}
        """
        )

        self.skip_quaternions_cb = QCheckBox("Skip Quaternions (Novecento)")
        self.skip_quaternions_cb.setChecked(True)
        self.skip_quaternions_cb.setToolTip(
            "Data recorded with Novecento includes 6 additional quaternion channels after EMG data.\n"
            "When checked, these are automatically excluded and the next grid's start\n"
            "value is offset by 6."
        )
        self.skip_quaternions_cb.setStyleSheet(
            f"""
            QCheckBox {{
                color: {COLORS["foreground"]};
                font-size: {FONT_SIZES["small"]};
                background-color: {COLORS["background_input"]};
                border: 2px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QCheckBox:hover {{
                background-color: {COLORS["background_hover"]};
            }}
            """
        )
        self.skip_quaternions_cb.stateChanged.connect(
            self._recalculate_all_channel_ranges
        )

        action_layout.addWidget(add_grid_btn)
        action_layout.addWidget(add_aux_btn)
        action_layout.addWidget(self.skip_quaternions_cb)
        action_layout.addStretch()
        action_layout.addWidget(self.channel_summary_label)
        action_layout.addWidget(clear_btn)
        layout.addLayout(action_layout)

        # Scroll area
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {COLORS["background_input"]};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS["text_muted"]};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {COLORS['background']};")
        self.channels_layout = QVBoxLayout(scroll_content)
        self.channels_layout.setSpacing(6)
        self.channels_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)

        return group

    def _create_summary_section(self) -> QGroupBox:
        group = QGroupBox("3. Review & Apply")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.allocation_bar = ChannelAllocationBar()
        layout.addWidget(self.allocation_bar)

        # Bottom row: Save/Load on left, Apply on right
        bottom_layout = QHBoxLayout()

        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_config)
        save_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_muted"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 8px 16px;
                font-size: {FONT_SIZES["normal"]};
            }}
            QPushButton:hover {{
                background-color: {COLORS["background_hover"]};
                color: {COLORS["foreground"]};
            }}
        """
        )

        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self._load_config)
        load_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_muted"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 8px 16px;
                font-size: {FONT_SIZES["normal"]};
            }}
            QPushButton:hover {{
                background-color: {COLORS["background_hover"]};
                color: {COLORS["foreground"]};
            }}
        """
        )

        bottom_layout.addWidget(save_btn)
        bottom_layout.addWidget(load_btn)
        bottom_layout.addStretch()

        self.apply_btn = QPushButton("Apply Configuration →")
        self.apply_btn.setFixedHeight(40)
        self.apply_btn.setMinimumWidth(200)
        self.apply_btn.clicked.connect(self._apply_config)
        self.apply_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS["success"]}, stop:1 #38A169);
                color: white; border-radius: 6px; font-weight: bold;
                font-size: {FONT_SIZES["medium"]}; padding: 10px 24px;
            }}
            QPushButton:hover {{ background-color: #48BB78; }}
            QPushButton:pressed {{ background-color: #2F855A; }}
            QPushButton:disabled {{
                background-color: {COLORS["background_input"]};
                color: {COLORS["text_muted"]};
            }}
        """
        )

        bottom_layout.addWidget(self.apply_btn)
        layout.addLayout(bottom_layout)

        return group

    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                font-family: '{FONT_FAMILY}';
                font-size: {FONT_SIZES["large"]};
                font-weight: bold;
                color: {COLORS["info"]};
                border: 2px solid {COLORS["border"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }}
        """

    def _show_initial_state(self):
        self.apply_btn.setEnabled(False)
        self._update_summary()

    def _populate_loader_presets(self):
        presets_dir = Path(__file__).parent.parent.parent / "resources/loaders_configs"
        self.loader_combo.clear()
        self._loader_layouts = {}
        if presets_dir.exists():
            for yaml_file in sorted(presets_dir.glob("loader_*.yaml")):
                try:
                    layout = load_layout(yaml_file)
                    name = layout["name"]
                    self._loader_layouts[name] = layout
                    self.loader_combo.addItem(name)
                except Exception as e:
                    logger.warning("Could not load preset %s: %s", yaml_file.name, e)

    def _auto_select_loader(self, file_path: Path):
        """
        Select a loader for the file, preferring one that can actually read it.

        Matching the file extension against the preset name is not enough on its
        own: several presets can share an extension (a generic ".hdf5" and a
        study-specific one, say), so a name match alone would silently replace a
        working loader with one whose EMG dataset path is absent from the file.
        That failure surfaces much later as a bogus channel-count error.
        """
        ext = file_path.suffix.lower()

        def usable(layout) -> bool:
            """The layout is for this kind of file *and* resolves its EMG field."""
            return bool(
                format_matches_extension(layout.get("format", ""), ext)
                and can_read_field(file_path, layout, "emg")
            )

        current = self._get_current_layout()
        if current is not None and usable(current):
            return

        names = [
            self.loader_combo.itemText(i) for i in range(self.loader_combo.count())
        ]
        # Extension matches get first refusal; sorted() is stable, so the rest
        # keep their preset order.
        for name in sorted(names, key=lambda n: n.lower() != ext):
            layout = self._loader_layouts.get(name)
            if layout is not None and usable(layout):
                self._select_loader(name)
                return

        # No preset could be confirmed (or the format has no cheap probe):
        # fall back to the plain extension match.
        for name in names:
            if name.lower() == ext:
                self._select_loader(name)
                return

        # Some presets cover multiple extensions (.h5/.hdf5, .csv/.txt) and
        # therefore cannot be named after every extension they accept.
        for name in names:
            layout = self._loader_layouts.get(name)
            if layout and format_matches_extension(layout.get("format", ""), ext):
                self._select_loader(name)
                return

    def _select_loader(self, name: str):
        idx = self.loader_combo.findText(name)
        if idx >= 0:
            self.loader_combo.setCurrentIndex(idx)

    def _on_loader_changed(self):
        layout = self._get_current_layout()
        fmt = layout.get("format", "") if layout else ""
        show = fmt in ("h5", "mat", "npy", "csv")
        self.emg_path_row.setVisible(show)
        self.skip_quaternions_cb.setEnabled(fmt not in ("otb4", "rhs"))
        if fmt in ("otb4", "rhs"):
            # These loaders expose only physical EMG channels, so their
            # canonical channel arrays have no quaternion/buffer/ramp gaps.
            self.skip_quaternions_cb.setChecked(False)
        if show and layout:
            emg_spec = layout.get("fields", {}).get("emg", {})
            self.emg_path_edit.setText(emg_spec.get("path", ""))
            orient = emg_spec.get("orientation", "auto")
            idx = self.emg_orientation_combo.findText(orient)
            self.emg_orientation_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._set_decimate(self._layout_decimate(layout))
        self._update_inspect_action()
        if self.emg_path:
            self._refresh_file_metadata()
            self._update_file_info()

    @staticmethod
    def _layout_decimate(layout: dict | None) -> int:
        """Decimation factor a loader preset declares (1 when absent/invalid)."""
        try:
            return max(1, int((layout or {}).get("decimate") or 1))
        except (TypeError, ValueError):
            return 1

    def _set_decimate(self, factor: int):
        self.decimate_spin.blockSignals(True)
        self.decimate_spin.setValue(int(factor))
        self.decimate_spin.blockSignals(False)
        self._update_delivered_fs_label()

    def _on_decimate_changed(self):
        self._update_delivered_fs_label()
        if self.emg_path:
            self._refresh_file_metadata()
            self._update_file_info()
        self._update_summary()

    def _native_fs(self) -> int | None:
        try:
            return int(self.fsamp_edit.text())
        except ValueError:
            return None

    def _effective_fs(self) -> float | None:
        """Sampling rate the decomposition sees: native rate / decimation."""
        fs = self._native_fs()
        if fs is None:
            return None
        return fs / self.decimate_spin.value()

    def _update_delivered_fs_label(self):
        q = self.decimate_spin.value()
        fs = self._effective_fs()
        if q <= 1 or fs is None:
            self.delivered_fs_label.setText("")
        else:
            self.delivered_fs_label.setText(f"→ {fs:g} Hz delivered")

    def _on_emg_path_changed(self):
        if self.emg_path:
            self._refresh_file_metadata()
            self._update_file_info()

    def _get_current_layout(self) -> dict | None:
        return self._loader_layouts.get(self.loader_combo.currentText())

    def _get_layout_with_overrides(self) -> dict | None:
        """Return a deep-copy of the current layout with the user's path/orientation applied."""
        layout = self._get_current_layout()
        if layout is None:
            return None
        layout = copy.deepcopy(layout)
        path_override = self.emg_path_edit.text().strip()
        if path_override:
            layout["fields"]["emg"]["path"] = path_override
        orientation = self.emg_orientation_combo.currentText()
        layout["fields"]["emg"]["orientation"] = orientation
        layout["decimate"] = self.decimate_spin.value()
        return layout

    def _on_fsamp_changed(self):
        self._update_delivered_fs_label()
        if self.emg_path:
            self._update_file_info()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EMG Data",
            str(Path.cwd()),
            "EMG Files (*.mat *.npy *.csv *.txt *.h5 *.hdf5 *.otb+ *.otb4 *.rhs);;"
            "OTB Files (*.otb+ *.otb4);;Intan Files (*.rhs);;All Files (*.*)",
        )
        if path:
            self.emg_path = Path(path)
            self.emg_paths = [self.emg_path]
            self.path_edit.setText(path)
            self._auto_select_loader(self.emg_path)
            self._update_inspect_action()
            inspected = False
            if self._selected_file_needs_inspection():
                inspected = self._inspect_selected_file()
            if not inspected:
                self._refresh_file_metadata()
                self._update_file_info()
                self._update_summary()

            if not self.grid_cards:
                self._add_grid()

    @staticmethod
    def _supports_array_inspection(path: Path | None) -> bool:
        return path is not None and path.suffix.lower() in {
            ".mat",
            ".h5",
            ".hdf5",
            ".npy",
            ".csv",
            ".txt",
        }

    def _update_inspect_action(self):
        self.inspect_arrays_btn.setEnabled(
            self._supports_array_inspection(self.emg_path)
        )

    def _selected_file_needs_inspection(self) -> bool:
        if not self._supports_array_inspection(self.emg_path):
            return False
        layout = self._get_layout_with_overrides()
        return layout is None or not can_read_field(self.emg_path, layout, "emg")

    def _inspect_selected_file(self) -> bool:
        """Inspect the selected scientific file and apply its chosen EMG array."""
        if self.emg_path is None:
            return False
        path = self.emg_path
        try:
            inspection = inspect_recording(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Inspection failed", f"Could not inspect the recording:\n{exc}"
            )
            return False
        if inspection.suggested_array is None:
            QMessageBox.warning(
                self,
                "No EMG matrix found",
                "The file does not contain a numeric two-dimensional array.",
            )
            return False

        dialog = ImportDataDialog(
            inspection,
            parent=self,
        )
        if not dialog.exec():
            return False
        self._apply_import_selection(
            Path(path),
            field_path=dialog.selected_path,
            orientation=dialog.selected_orientation,
            sampling_rate=dialog.sampling_rate,
            preserve_batch=len(self.emg_paths) > 1,
        )
        return True

    def _apply_import_selection(
        self,
        file_path: Path,
        *,
        field_path: str,
        orientation: str,
        sampling_rate: int,
        preserve_batch: bool = False,
    ):
        """Apply a validated inspector selection to the normal configuration UI."""
        self.emg_path = Path(file_path)
        if not preserve_batch:
            self.emg_paths = [self.emg_path]
            self.path_edit.setText(str(self.emg_path))
        self._auto_select_loader(self.emg_path)
        self._update_inspect_action()

        layout = self._get_current_layout()
        if layout and layout.get("format") in {"h5", "mat", "npy", "csv"}:
            self.emg_path_edit.setText(field_path)
            index = self.emg_orientation_combo.findText(orientation)
            self.emg_orientation_combo.setCurrentIndex(index if index >= 0 else 0)
        self.fsamp_edit.setText(str(sampling_rate))
        self._metadata_key = None
        self._refresh_file_metadata()
        self._update_file_info()
        self._update_summary()
        if not self.grid_cards:
            self._add_grid()

    def _update_file_info(self):
        """
        Read the channel count from the selected file and report it.

        Sole owner of max_channels: when the file cannot be read, the count stays
        unknown rather than falling back to a placeholder, so the configuration
        is reported as unreadable instead of as too small.
        """
        self._channel_count_known = False
        layout = self._get_layout_with_overrides()
        if layout is None or self.emg_path is None:
            return
        try:
            if self._metadata_error:
                raise ValueError(self._metadata_error)
            if (
                self.file_metadata.get("emg_channel_count") is not None
                and self.file_metadata.get("n_samples") is not None
            ):
                n_samples = int(self.file_metadata["n_samples"])
                n_channels = int(self.file_metadata["emg_channel_count"])
                fs = float(
                    self.file_metadata.get("sampling_frequency")
                    or self._effective_fs()
                    or 2048
                )
                duration_sec = n_samples / fs
                n_grids = len(self.file_metadata.get("grids", []))
                n_aux = int(self.file_metadata.get("aux_channel_count", 0))
                self._set_channel_count(n_channels)
                self._set_file_info(
                    f"Loaded: {self.emg_path.name} | "
                    f"Shape: {n_samples} samples × {n_channels} channels | "
                    f"{n_grids} grid(s), {n_aux} aux | "
                    f"Duration: {duration_sec:.1f}s @ {fs:g} Hz"
                )
                return
            layout_full = copy.deepcopy(layout)
            layout_full["fields"]["emg"].pop("channels", None)
            emg = load_field(self.emg_path, layout_full, "emg")
            n_samples, n_channels = emg.shape
            fs = self._effective_fs() or 2048
            duration_sec = n_samples / fs
            self._set_channel_count(n_channels)
            self._set_file_info(
                f"Loaded: {self.emg_path.name} | "
                f"Shape: {n_samples} samples × {n_channels} channels | "
                f"Duration: {duration_sec:.1f}s @ {fs:g} Hz"
            )
        except Exception as e:
            self._set_file_info(
                f"⚠ Load failed with the '{self.loader_combo.currentText()}' "
                f"loader: {e}",
                error=True,
            )

    def _set_channel_count(self, n_channels: int):
        self.max_channels = n_channels
        self._channel_count_known = True
        self.allocation_bar.set_max_channels(n_channels)

    def _set_file_info(self, text: str, error: bool = False):
        self.file_info_label.setText(text)
        self.file_info_label.setStyleSheet(
            get_label_style(size="small", color="error" if error else "text_dim")
        )

    def _refresh_file_metadata(self):
        """Load cheap archive metadata and apply its sampling frequency."""
        layout = self._get_layout_with_overrides()
        if layout is None or self.emg_path is None:
            self.file_metadata = {}
            return
        emg_spec = layout.get("fields", {}).get("emg", {})
        key = (
            str(self.emg_path),
            layout.get("format"),
            layout.get("decimate"),
            emg_spec.get("path"),
            emg_spec.get("orientation"),
        )
        if key == self._metadata_key:
            return
        self._metadata_key = key
        self._metadata_error = None
        try:
            self.file_metadata = load_metadata(self.emg_path, layout)
            fs = self.file_metadata.get(
                "native_sampling_frequency",
                self.file_metadata.get("sampling_frequency"),
            )
            if fs is not None:
                self.fsamp_edit.setText(str(int(fs)))
        except Exception as exc:
            self.file_metadata = {}
            self._metadata_error = str(exc)

    def _add_grid(self):
        index = len(self.grid_cards) + 1
        color_idx = (index - 1) % len(GridCard.GRID_COLORS)
        color = GridCard.GRID_COLORS[color_idx]

        card = GridCard(index, color)
        card.remove_requested.connect(self._remove_grid)
        card.type_combo.currentTextChanged.connect(self._recalculate_all_channel_ranges)
        card.config_combo.currentTextChanged.connect(
            self._recalculate_all_channel_ranges
        )
        card.changed.connect(self._update_summary)
        card.start_spin.valueChanged.connect(self._update_summary)
        card.end_spin.valueChanged.connect(self._update_summary)

        self.channels_layout.insertWidget(len(self.grid_cards), card)
        self.grid_cards.append(card)

        # Only set the start of the NEW card based on where the previous one ends
        if len(self.grid_cards) > 1:
            prev_card = self.grid_cards[-2]
            next_start = prev_card.end_spin.value() + self._quaternion_gap(prev_card)
        else:
            next_start = 0

        _, _, _, n_ch = card.get_geometry()
        if n_ch <= 0:
            n_ch = 64
        card.start_spin.blockSignals(True)
        card.end_spin.blockSignals(True)
        card.start_spin.setValue(next_start)
        card.end_spin.setValue(next_start + n_ch)
        card.start_spin.blockSignals(False)
        card.end_spin.blockSignals(False)

        self._update_summary()

    @staticmethod
    def _is_hd_grid(card: "GridCard") -> bool:
        """True for OT Bioelettronica HD grids (HD<ied>MM<rows><cols>).

        Only those adapters append quaternion channels; other presets that
        merely contain "HD" in their name (e.g. the UltraHD 4x4 array) do not.
        """
        config_name = card.config_combo.currentText()
        return re.search(r"\bHD\d{2}MM", config_name.upper()) is not None

    def _quaternion_gap(self, card: "GridCard") -> int:
        """Return 6 if skip-quaternions is enabled and the card is an HD grid, else 0."""
        if self.skip_quaternions_cb.isChecked() and self._is_hd_grid(card):
            return self.HD_QUATERNION_CHANNELS
        return 0

    def _recalculate_all_channel_ranges(self):
        """Assign sequential channel ranges to all grid cards.

        When 'Skip Quaternions' is checked, a 6-channel gap is added after each
        HD grid to skip the quaternion channels in the recording.
        """
        next_start = 0
        for card in self.grid_cards:
            _, _, _, n_ch = card.get_geometry()
            if n_ch <= 0:
                n_ch = 64  # fallback if geometry unknown
            card.start_spin.blockSignals(True)
            card.end_spin.blockSignals(True)
            card.start_spin.setValue(next_start)
            card.end_spin.setValue(next_start + n_ch)
            card.start_spin.blockSignals(False)
            card.end_spin.blockSignals(False)
            next_start += n_ch + self._quaternion_gap(card)
        self._update_summary()

    def _remove_grid(self, card: GridCard):
        if card in self.grid_cards:
            # Capture the deleted card's range before removing it
            deleted_start, deleted_end = card.get_channel_range()
            deleted_count = deleted_end - deleted_start

            self.grid_cards.remove(card)
            self.channels_layout.removeWidget(card)
            card.deleteLater()
            self._renumber_grids()

            # Shift down only cards that started after the deleted one
            for remaining_card in self.grid_cards:
                start, end = remaining_card.get_channel_range()
                if start >= deleted_end:
                    remaining_card.start_spin.blockSignals(True)
                    remaining_card.end_spin.blockSignals(True)
                    remaining_card.start_spin.setValue(start - deleted_count)
                    remaining_card.end_spin.setValue(end - deleted_count)
                    remaining_card.start_spin.blockSignals(False)
                    remaining_card.end_spin.blockSignals(False)

            self._update_summary()

    def _add_grid_raw(self) -> GridCard:
        """Add a grid card without auto-assigning channel ranges. Used during config restore."""
        index = len(self.grid_cards) + 1
        color_idx = (index - 1) % len(GridCard.GRID_COLORS)
        color = GridCard.GRID_COLORS[color_idx]

        card = GridCard(index, color)
        card.remove_requested.connect(self._remove_grid)
        card.type_combo.currentTextChanged.connect(self._recalculate_all_channel_ranges)
        card.config_combo.currentTextChanged.connect(
            self._recalculate_all_channel_ranges
        )
        card.changed.connect(self._update_summary)
        card.start_spin.valueChanged.connect(self._update_summary)
        card.end_spin.valueChanged.connect(self._update_summary)

        self.channels_layout.insertWidget(len(self.grid_cards), card)
        self.grid_cards.append(card)
        return card

    def _renumber_grids(self):
        for i, card in enumerate(self.grid_cards):
            card.update_index(i + 1)
            color_idx = i % len(GridCard.GRID_COLORS)
            card.color = GridCard.GRID_COLORS[color_idx]
            card.color_indicator.setStyleSheet(
                f"background-color: {card.color}; border-radius: 2px;"
            )

    def _add_aux_channel(self, index: int | None = None):
        if index is None:
            index = len(self.aux_cards) + 1

        card = AuxChannelCard(index)
        card.remove_requested.connect(self._remove_aux_channel)
        card.changed.connect(self._update_summary)

        # Aux cards insert after all grids, before trailing stretch
        insert_pos = len(self.grid_cards) + len(self.aux_cards)
        self.channels_layout.insertWidget(insert_pos, card)
        self.aux_cards.append(card)
        self._update_summary()

    def _remove_aux_channel(self, card: AuxChannelCard):
        if card in self.aux_cards:
            self.aux_cards.remove(card)
            self.channels_layout.removeWidget(card)
            card.deleteLater()
            self._renumber_aux()
            self._update_summary()

    def _renumber_aux(self):
        for i, card in enumerate(self.aux_cards):
            card.update_index(i + 1)

    def _clear_all_channels(self):
        reply = QMessageBox.question(
            self,
            "Clear All Channels",
            "Remove all grid and auxiliary channel configurations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for card in self.grid_cards + self.aux_cards:
                card.deleteLater()
            self.grid_cards.clear()
            self.aux_cards.clear()
            self._update_summary()

    def _validate_configuration(self) -> tuple[bool, list[str]]:
        warnings = []

        if not self.grid_cards:
            warnings.append("No grids configured")
            return False, warnings

        for card in self.grid_cards:
            card.set_validation_status(True)
        for card in self.aux_cards:
            card.set_validation_status(True)

        # Without a channel count read from the file there is nothing to check
        # ranges against; say so rather than measuring them against a placeholder.
        if not self._channel_count_known:
            warnings.append(
                f"Channel count unknown: "
                f"{self.emg_path.name if self.emg_path else 'the selected file'} "
                f"could not be read with the "
                f"'{self.loader_combo.currentText()}' loader — check the data "
                f"format preset and the EMG dataset path"
            )

        # Collect all channel ranges (grids + signal-source aux)
        ranges = []

        for card in self.grid_cards:
            start, end = card.get_channel_range()
            name = card.get_data()["name"]

            expected = card.get_channel_count()
            if end - start != expected:
                msg = f"Expected {expected} channels"
                warnings.append(
                    f"{name}: configured range has {end - start} channels; "
                    f"the selected electrode expects {expected}"
                )
                card.set_validation_status(False, msg)

            if self._channel_count_known and end > self.max_channels:
                msg = "Exceeds available channels"
                warnings.append(
                    f"{name}: channels exceed file ({end} > {self.max_channels})"
                )
                card.set_validation_status(False, msg)

            if start >= end:
                msg = "Start >= End"
                warnings.append(f"{name}: Start channel must be < End channel")
                card.set_validation_status(False, msg)

            for other_start, other_end, other_name in ranges:
                if not (end <= other_start or start >= other_end):
                    msg = f"Overlaps with {other_name}"
                    warnings.append(f"{name} overlaps with {other_name}")
                    card.set_validation_status(False, msg)
                    break

            ranges.append((start, end, name))

        # Validate signal-source aux channels
        for card in self.aux_cards:
            source = card.get_source()
            if source == "data_field":
                # No channel range to check — but the field name is mandatory,
                # otherwise the worker has nothing to look up.
                if not card.get_field_path():
                    msg = "Field path required"
                    warnings.append(
                        f"{card.get_data()['name']}: "
                        "a field path is required for the 'Data file field' source"
                    )
                    card.set_validation_status(False, msg)
                continue
            if source == "aux_file":
                start, end = card.get_channel_range()
                name = card.get_data()["name"]
                available = self.file_metadata.get("aux_channel_count")
                if start >= end:
                    msg = "Start >= End"
                    warnings.append(f"{name}: Start channel must be < End channel")
                    card.set_validation_status(False, msg)
                elif available is not None and end > int(available):
                    msg = "Exceeds available aux channels"
                    warnings.append(
                        f"{name}: auxiliary range [{start},{end}) exceeds "
                        f"the {int(available)} channels declared by the file"
                    )
                    card.set_validation_status(False, msg)
                continue
            if source != "signal":
                continue
            start, end = card.get_channel_range()
            name = card.get_data()["name"]

            if self._channel_count_known and end > self.max_channels:
                msg = "Exceeds available channels"
                warnings.append(
                    f"{name}: channels exceed file ({end} > {self.max_channels})"
                )
                card.set_validation_status(False, msg)

            if start >= end:
                msg = "Start >= End"
                warnings.append(f"{name}: Start channel must be < End channel")
                card.set_validation_status(False, msg)

        metadata_fs = self.file_metadata.get(
            "native_sampling_frequency", self.file_metadata.get("sampling_frequency")
        )
        if metadata_fs is not None:
            configured_fs = self._native_fs()
            if configured_fs != int(metadata_fs):
                warnings.append(
                    f"Sampling rate {configured_fs!r} does not match file metadata "
                    f"({int(metadata_fs)} Hz)"
                )
        native_fs = self._native_fs()
        q = self.decimate_spin.value()
        if native_fs is not None and q > 1 and native_fs % q:
            warnings.append(
                f"Sampling rate {native_fs} Hz is not divisible by the decimation "
                f"factor {q}"
            )

        return len(warnings) == 0, warnings

    def _update_summary(self):
        # Allocation bar: grids + signal-source aux
        allocations = []
        for card in self.grid_cards:
            data = card.get_data()
            start, end = card.get_channel_range()
            allocations.append((start, end, data["name"], data["color"]))
        for card in self.aux_cards:
            if card.get_source() == "signal":
                data = card.get_data()
                start, end = card.get_channel_range()
                allocations.append((start, end, data["name"], AuxChannelCard.AUX_COLOR))
        self.allocation_bar.set_allocations(allocations)

        # Inline summary
        n_grids = len(self.grid_cards)
        n_aux = len(self.aux_cards)
        parts = []
        if n_grids:
            parts.append(f"{n_grids} grid{'s' if n_grids != 1 else ''}")
        if n_aux:
            parts.append(f"{n_aux} aux")
        self.channel_summary_label.setText(
            " · ".join(parts) if parts else "No channels"
        )

        # Apply button
        if not self.emg_path or not self.grid_cards:
            self.apply_btn.setEnabled(False)
            return
        is_valid, _ = self._validate_configuration()
        self.apply_btn.setEnabled(is_valid)

    def _config_to_dict(self) -> dict:
        """Serialize current UI state to a JSON-friendly dict."""
        d = {
            "version": 1,
            "loader": self.loader_combo.currentText(),
            "sampling_rate": int(self.fsamp_edit.text() or 2048),
            "decimate": self.decimate_spin.value(),
            "file_path": str(self.emg_path) if self.emg_path else None,
            "output_dir": self.output_dir_edit.text(),
            "skip_quaternions": self.skip_quaternions_cb.isChecked(),
            "grids": [
                {
                    **card.get_data(),
                    "muscle": card.muscle_edit.text(),
                }
                for card in self.grid_cards
            ],
            "aux_channels": [card.get_data() for card in self.aux_cards],
        }
        if self.emg_path_row.isVisible():
            d["emg_path"] = self.emg_path_edit.text().strip()
            d["emg_orientation"] = self.emg_orientation_combo.currentText()
        return d

    def _config_from_dict(self, cfg: dict):
        """Restore UI state from a previously saved dict."""
        # Loader preset
        loader_name = cfg.get("loader", "")
        idx = self.loader_combo.findText(loader_name)
        if idx >= 0:
            self.loader_combo.setCurrentIndex(idx)

        # Path / orientation overrides for generic array-based formats
        if "emg_path" in cfg:
            self.emg_path_edit.setText(cfg["emg_path"])
        if "emg_orientation" in cfg:
            idx = self.emg_orientation_combo.findText(cfg["emg_orientation"])
            if idx >= 0:
                self.emg_orientation_combo.setCurrentIndex(idx)

        # Sampling rate (native) and decimation. Configs saved before the
        # decimate field existed fall back to the loader preset's own value.
        self.fsamp_edit.setText(str(cfg.get("sampling_rate", 2048)))
        self._set_decimate(
            cfg.get("decimate", self._layout_decimate(self._get_current_layout()))
        )

        # Defaults to True for configs saved before this field existed
        self.skip_quaternions_cb.setChecked(cfg.get("skip_quaternions", True))

        # File path: only apply if no file has been selected yet.
        # If the user already loaded a file, keep it — the config's saved path
        # is just a reference from when the config was created.
        file_path = cfg.get("file_path")
        if file_path and Path(file_path).exists() and self.emg_path is None:
            self.emg_path = Path(file_path)
            self.emg_paths = [self.emg_path]
            self.path_edit.setText(file_path)
            self._update_inspect_action()
            self._refresh_file_metadata()
            self._update_file_info()

        # Clear existing
        for card in self.grid_cards + self.aux_cards:
            card.deleteLater()
        self.grid_cards.clear()
        self.aux_cards.clear()

        # Restore grids with their saved channel ranges. The recalculate signals are
        # disconnected during set_values to prevent them overwriting the saved start/end.
        for g in cfg.get("grids", []):
            card = self._add_grid_raw()
            with contextlib.suppress(TypeError):
                card.type_combo.currentTextChanged.disconnect(
                    self._recalculate_all_channel_ranges
                )
            with contextlib.suppress(TypeError):
                card.config_combo.currentTextChanged.disconnect(
                    self._recalculate_all_channel_ranges
                )
            card.blockSignals(True)
            card.set_values(
                name=g.get("name", ""),
                muscle=g.get("muscle", ""),
                electrode_type=g.get("type", "Surface"),
                config=g.get("config", ""),
                start=g.get("start_chan", 0),
                end=g.get("end_chan", 63),
            )
            card.blockSignals(False)
            # Reconnect recalculate signals
            card.type_combo.currentTextChanged.connect(
                self._recalculate_all_channel_ranges
            )
            card.config_combo.currentTextChanged.connect(
                self._recalculate_all_channel_ranges
            )

        # Restore aux
        for a in cfg.get("aux_channels", []):
            self._add_aux_channel()
            card = self.aux_cards[-1]
            card.set_values(
                name=a.get("name", ""),
                aux_type=a.get("type", "Other"),
                source=a.get("source", "signal"),
                start=a.get("start_chan", 0),
                end=a.get("end_chan", 0),
                unit=a.get("unit", ""),
                mvc=a.get("mvc"),
                field_path=a.get("field_path", ""),
            )

        self.output_dir_edit.setText(cfg.get("output_dir", self.output_dir_edit.text()))
        self._update_summary()

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Channel Configuration",
            str(Path.cwd() / "channel_config.json"),
            "JSON Files (*.json)",
        )
        if path:
            cfg = self._config_to_dict()
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)

    def _load_config(self):
        start_dir = str(self.emg_path.parent) if self.emg_path else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Channel Configuration", start_dir, "JSON Files (*.json)"
        )
        if path:
            with open(path) as f:
                cfg = json.load(f)
            self._config_from_dict(cfg)

    def _apply_config(self):
        is_valid, warnings = self._validate_configuration()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Configuration Invalid",
                "Please fix the following issues:\n\n" + "\n".join(warnings),
            )
            return

        try:
            fs = int(self.fsamp_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Sampling rate must be a number")
            return
        # The loader decimates on read, so downstream stages see this rate.
        fs //= self.decimate_spin.value()

        config = self.config_manager.create_default_session(
            name="Decomposition Session"
        )
        config.sampling_frequency = fs
        config.input_dir = str(self.emg_path.parent)

        # Grids → ports
        for card in self.grid_cards:
            data = card.get_data()
            electrode_type = data["type"]
            electrode_config = data["config"]
            channels = list(range(data["start_chan"], data["end_chan"]))

            if electrode_type in GridCard.ELECTRODE_CONFIGS:
                configs = GridCard.ELECTRODE_CONFIGS[electrode_type]
                if electrode_config in configs:
                    cfg = configs[electrode_config]
                    electrode = ElectrodeConfig(
                        name=electrode_config,
                        type=electrode_type.lower(),
                        channels=channels,
                        rows=cfg["rows"],
                        cols=cfg["cols"],
                        spacing_mm=cfg["spacing_mm"],
                    )
                    electrode.validate()
                    port = PortConfig(
                        name=data["name"],
                        electrode=electrode,
                        filter=FilterConfig(),
                        decomposition=DecompositionConfig(),
                        muscle=data.get("muscle", ""),
                    )
                    config.ports.append(port)

        # Aux channels
        config.aux_channels = [card.get_data() for card in self.aux_cards]

        config.data_layout = self._get_layout_with_overrides()
        config.output_dir = self.output_dir_edit.text() or str(self.emg_path.parent)
        config.emg_paths = [str(p) for p in self.emg_paths]
        self.config_applied.emit(config, self.emg_paths)
