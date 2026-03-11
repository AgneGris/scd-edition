"""
Configuration Tab - EMG data loading and electrode configuration.
"""

import copy
import json
from pathlib import Path
from typing import List, Optional, Tuple
from scd_app.io.data_loader import load_layout, load_field

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, 
    QSpinBox, QScrollArea, QFrame, 
    QMessageBox, QFileDialog, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator

from core.config import (
    ConfigManager, ElectrodeConfig, PortConfig,
    FilterConfig, DecompositionConfig
)

from gui.style.styling import (
    COLORS, FONT_SIZES, FONT_FAMILY,
    get_label_style, 
    get_button_style
)


class ChannelAllocationBar(QFrame):
    """Visual bar showing channel allocation across all grids and aux channels."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.max_channels = 256
        self.allocations = []
        
        self.setStyleSheet(f"""
            ChannelAllocationBar {{
                background-color: {COLORS['background_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
    
    def set_max_channels(self, n: int):
        self.max_channels = n
        self.update()
    
    def set_allocations(self, allocations: List[Tuple[int, int, str, str]]):
        self.allocations = allocations
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.max_channels == 0:
            return
        
        from PyQt5.QtGui import QPainter, QColor, QPen, QFont
        from PyQt5.QtCore import QRect
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margin = 10
        bar_height = 25
        bar_y = (self.height() - bar_height) // 2
        bar_width = self.width() - 2 * margin
        
        bg_rect = QRect(margin, bar_y, bar_width, bar_height)
        painter.setPen(QPen(QColor(COLORS['border']), 1))
        painter.setBrush(QColor(COLORS['background']))
        painter.drawRoundedRect(bg_rect, 3, 3)
        
        painter.setPen(QPen(QColor(COLORS['text_muted']), 1))
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
                color = COLORS['error']
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
                painter.setPen(QPen(QColor('#ffffff')))
                label_font = QFont(FONT_FAMILY, 8, QFont.Bold)
                painter.setFont(label_font)
                painter.drawText(segment_rect, Qt.AlignCenter, name)


class GridCard(QFrame):
    """Card widget for configuring a single electrode grid."""
    
    remove_requested = pyqtSignal(object)
    changed = pyqtSignal()
    
    GRID_COLORS = ['#4a9eff', '#a78bfa', '#48BB78', '#F6AD55', '#ff6b9d', '#63B3ED']
    
    ELECTRODE_CONFIGS = {
        "Surface": {
            "Grid (GR08MM1305)": {"rows": 13, "cols": 5, "spacing_mm": 8.0},
            "Grid (GR10MM0808)": {"rows": 8, "cols": 8, "spacing_mm": 10.0},
        },
        "Intramuscular": {
            "Thin-film (40ch)": {"rows": 20, "cols": 2, "spacing_mm": 2.5},
            "Wire needle": {"rows": 1, "cols": 16, "spacing_mm": 4.0},
            "Myomatrix": {"rows": 1, "cols": 32, "spacing_mm": 4.0},
        }
    }
    
    def __init__(self, index: int, color: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.color = color

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(55)

        self._setup_ui()
        self._apply_styling()
    
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(8)
        
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(4, 20)
        self.color_indicator.setStyleSheet(f"background-color: {self.color}; border-radius: 2px;")
        main_layout.addWidget(self.color_indicator)
        
        # Type badge
        type_label = QLabel("EMG")
        type_label.setFixedWidth(36)
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet(f"""
            QLabel {{
                background-color: {self.color}30;
                color: {self.color};
                border-radius: 3px;
                font-size: {FONT_SIZES['small']};
                font-weight: bold;
                padding: 1px 4px;
            }}
        """)
        main_layout.addWidget(type_label)
        
        # Name
        self.name_edit = QLineEdit(f"Grid_{self.index}")
        self.name_edit.setPlaceholderText("e.g., Biceps")
        self.name_edit.textChanged.connect(self.changed.emit)
        main_layout.addWidget(self.name_edit, stretch=2)
        
        # Muscle
        self.muscle_edit = QLineEdit()
        self.muscle_edit.setPlaceholderText("Muscle")
        self.muscle_edit.textChanged.connect(self.changed.emit)
        main_layout.addWidget(self.muscle_edit, stretch=2)
        
        # Electrode type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Surface", "Intramuscular"])
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        main_layout.addWidget(self.type_combo, stretch=2)
        
        # Electrode config
        self.config_combo = QComboBox()
        self.config_combo.currentTextChanged.connect(self._on_config_change)
        main_layout.addWidget(self.config_combo, stretch=2)
        
        # Channel range
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 2048)
        self.start_spin.setValue(0)
        self.start_spin.valueChanged.connect(self.changed.emit)
        main_layout.addWidget(self.start_spin, stretch=1)
        
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 2048)
        self.end_spin.setValue(64)
        self.end_spin.valueChanged.connect(self.changed.emit)
        main_layout.addWidget(self.end_spin, stretch=1)
        
        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet(get_label_style(size='small'))
        main_layout.addWidget(self.status_label, stretch=1)
        
        # Remove
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setToolTip("Remove Grid")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: transparent; 
                color: {COLORS['text_muted']}; 
                border-radius: 10px; 
                font-weight: bold; font-size: 14pt;
            }}
            QPushButton:hover {{ 
                background-color: {COLORS['error']}40; 
                color: {COLORS['error_bright']}; 
            }}
        """)
        main_layout.addWidget(self.remove_btn)
        
        self._on_type_change()
    
    def _apply_styling(self):
        self.setStyleSheet(f"""
            GridCard {{ 
                background-color: {COLORS['background_light']}; 
                border: 1px solid {COLORS['border']}; 
                border-radius: 6px; 
            }}
            QLabel {{ 
                color: {COLORS['foreground']}; 
                font-family: '{FONT_FAMILY}'; 
            }}
        """)
    
    def update_index(self, index: int):
        self.index = index
        if self.name_edit.text().startswith("Grid_"):
            self.name_edit.setText(f"Grid_{index}")
    
    def _on_type_change(self):
        electrode_type = self.type_combo.currentText()
        current_config = self.config_combo.currentText()
        self.config_combo.clear()
        configs = list(self.ELECTRODE_CONFIGS[electrode_type].keys())
        self.config_combo.addItems(configs)
        if current_config in configs:
            self.config_combo.setCurrentText(current_config)
        self.changed.emit()
    
    def _on_config_change(self):
        self.changed.emit()
    
    def set_validation_status(self, is_valid: bool, message: str = ""):
        if is_valid:
            self.status_label.setText("")
        else:
            self.status_label.setText(f"⚠ {message}")
            self.status_label.setStyleSheet(get_label_style(size='small', color='warning'))
    
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
    
    def get_geometry(self) -> Tuple[int, int, float]:
        electrode_type = self.type_combo.currentText()
        config_name = self.config_combo.currentText()
        if electrode_type in self.ELECTRODE_CONFIGS:
            configs = self.ELECTRODE_CONFIGS[electrode_type]
            if config_name in configs:
                cfg = configs[config_name]
                return cfg['rows'], cfg['cols'], cfg['spacing_mm']
        return 0, 0, 0.0
    
    def get_channel_count(self) -> int:
        return self.end_spin.value() - self.start_spin.value()
    
    def get_channel_range(self) -> Tuple[int, int]:
        return self.start_spin.value(), self.end_spin.value()
    
    def set_start_channel(self, start: int):
        self.start_spin.setValue(start)
    
    def set_end_channel(self, end: int):
        self.end_spin.setValue(end)
    
    def set_values(self, name: str, muscle: str, electrode_type: str, config: str,
                   start: int, end: int):
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
    
    remove_requested = pyqtSignal(object)
    changed = pyqtSignal()
    
    AUX_TYPES = ["Force", "Torque", "Trigger", "Target", "Path", "Angle", "Position", "Other"]
    AUX_COLOR = '#F6AD55'
    
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet(f"""
            QLabel {{
                background-color: {self.AUX_COLOR}30;
                color: {self.AUX_COLOR};
                border-radius: 3px;
                font-size: {FONT_SIZES['small']};
                font-weight: bold;
                padding: 1px 4px;
            }}
        """)
        main_layout.addWidget(type_label)
        
        # Name
        self.name_edit = QLineEdit(f"Aux_{self.index}")
        self.name_edit.setPlaceholderText("e.g., Force_raw")
        self.name_edit.textChanged.connect(self.changed.emit)
        main_layout.addWidget(self.name_edit, stretch=2)
        
        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.AUX_TYPES)
        self.type_combo.currentTextChanged.connect(self.changed.emit)
        main_layout.addWidget(self.type_combo, stretch=2)
        
        # Source: from main signal or from .sip aux files
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Signal channels", "Aux file (.sip)"])
        self.source_combo.currentIndexChanged.connect(self._on_source_change)
        main_layout.addWidget(self.source_combo, stretch=2)
        
        # Channel range (for signal channels)
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 2048)
        self.start_spin.setValue(0)
        self.start_spin.valueChanged.connect(self.changed.emit)
        main_layout.addWidget(self.start_spin, stretch=1)
        
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 2048)
        self.end_spin.setValue(0)
        self.end_spin.valueChanged.connect(self.changed.emit)
        main_layout.addWidget(self.end_spin, stretch=1)
        
        # Unit
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("Unit")
        self.unit_edit.setFixedWidth(60)
        self.unit_edit.textChanged.connect(self.changed.emit)
        main_layout.addWidget(self.unit_edit)
        
        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet(get_label_style(size='small'))
        main_layout.addWidget(self.status_label, stretch=1)
        
        # Remove
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setToolTip("Remove Channel")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: transparent; 
                color: {COLORS['text_muted']}; 
                border-radius: 10px; 
                font-weight: bold; font-size: 14pt;
            }}
            QPushButton:hover {{ 
                background-color: {COLORS['error']}40; 
                color: {COLORS['error_bright']}; 
            }}
        """)
        main_layout.addWidget(self.remove_btn)
    
    def _apply_styling(self):
        self.setStyleSheet(f"""
            AuxChannelCard {{ 
                background-color: {COLORS['background_light']}; 
                border: 1px solid {COLORS['border']}; 
                border-radius: 6px; 
            }}
            QLabel {{ 
                color: {COLORS['foreground']}; 
                font-family: '{FONT_FAMILY}'; 
            }}
        """)
    
    def _on_source_change(self, idx: int):
        """Toggle channel range visibility based on source."""
        is_signal = (idx == 0)
        self.start_spin.setVisible(is_signal)
        self.end_spin.setVisible(is_signal)
        self.changed.emit()
    
    def update_index(self, index: int):
        self.index = index
        if self.name_edit.text().startswith("Aux_"):
            self.name_edit.setText(f"Aux_{index}")
    
    def set_validation_status(self, is_valid: bool, message: str = ""):
        if is_valid:
            self.status_label.setText("")
        else:
            self.status_label.setText(f"⚠ {message}")
            self.status_label.setStyleSheet(get_label_style(size='small', color='warning'))
    
    def get_source(self) -> str:
        """'signal' if from main signal channels, 'aux_file' if from .sip files."""
        return "signal" if self.source_combo.currentIndex() == 0 else "aux_file"
    
    def get_channel_range(self) -> Tuple[int, int]:
        return self.start_spin.value(), self.end_spin.value()
    
    def get_data(self) -> dict:
        return {
            "index": self.index,
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText().lower(),
            "source": self.get_source(),
            "start_chan": self.start_spin.value(),
            "end_chan": self.end_spin.value(),
            "unit": self.unit_edit.text(),
        }
    
    def set_values(self, name: str, aux_type: str, source: str = "signal",
                   start: int = 0, end: int = 0, unit: str = ""):
        self.name_edit.setText(name)
        type_idx = self.type_combo.findText(aux_type, Qt.MatchFixedString)
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)
        self.source_combo.setCurrentIndex(0 if source == "signal" else 1)
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.unit_edit.setText(unit)


class ConfigTab(QWidget):
    """Streamlined configuration tab for EMG data loading and channel setup."""
    
    config_applied = pyqtSignal(object, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = ConfigManager()
        self.emg_path: Optional[Path] = None
        self.emg_paths: List[Path] = []  # all selected files for batch
        self.max_channels: int = 256
        self.grid_cards: List[GridCard] = []
        self.aux_cards: List[AuxChannelCard] = []
        
        self._setup_ui()
        self._show_initial_state()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        title = QLabel("Session Configuration")
        title.setStyleSheet(get_label_style(size='title', bold=True))
        main_layout.addWidget(title)
        
        main_layout.addWidget(self._create_file_section(), stretch=0)
        main_layout.addWidget(self._create_channels_section(), stretch=1)
        main_layout.addWidget(self._create_summary_section(), stretch=0)
    

    def _create_file_section(self) -> QGroupBox:
        group = QGroupBox("1. Load EMG Data")
        group.setStyleSheet(self._group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        loader_layout = QHBoxLayout()
        loader_label = QLabel("Data Format:")
        loader_label.setStyleSheet(get_label_style(size='normal'))
        self.loader_combo = QComboBox()
        self._populate_loader_presets()
        self.loader_combo.currentTextChanged.connect(self._on_loader_changed)
        loader_layout.addWidget(loader_label)
        loader_layout.addWidget(self.loader_combo, stretch=1)
        loader_layout.addStretch()
        layout.addLayout(loader_layout)

        fs_layout = QHBoxLayout()
        fs_label = QLabel("Sampling Rate:")
        fs_label.setStyleSheet(get_label_style(size='normal'))
        self.fsamp_edit = QLineEdit("2048")
        self.fsamp_edit.setFixedWidth(100)
        self.fsamp_edit.setValidator(QIntValidator(1, 100000))
        self.fsamp_edit.textChanged.connect(self._on_fsamp_changed)
        fs_hz = QLabel("Hz")
        fs_hz.setStyleSheet(get_label_style(size='normal', color='text_secondary'))
        fs_layout.addWidget(fs_label)
        fs_layout.addWidget(self.fsamp_edit)
        fs_layout.addWidget(fs_hz)
        fs_layout.addStretch()
        layout.addLayout(fs_layout)
    
        # Output directory
        out_layout = QHBoxLayout()
        out_label = QLabel("Output Folder:")
        out_label.setStyleSheet(get_label_style(size='normal'))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Same as input file...")
        self.output_dir_edit.setReadOnly(True)
        out_browse = QPushButton("Browse...")
        out_browse.setFixedWidth(200)
        out_browse.clicked.connect(self._browse_output_dir)
        out_browse.setStyleSheet(get_button_style(bg_color='accent', padding=8))
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.output_dir_edit, stretch=1)
        out_layout.addWidget(out_browse)
        layout.addLayout(out_layout)

        self.file_info_label = QLabel()
        self.file_info_label.setStyleSheet(get_label_style(size='small', color='text_dim'))
        layout.addWidget(self.file_info_label)
        
        file_layout = QHBoxLayout()

        file_label = QLabel("Input File:")
        file_label.setStyleSheet(get_label_style(size='normal'))

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select EMG data file(s)...")
        self.path_edit.setReadOnly(True)

        browse_btn = QPushButton("File...")
        browse_btn.setFixedWidth(200)
        browse_btn.clicked.connect(self._browse_file)
        browse_btn.setStyleSheet(get_button_style(bg_color='accent', padding=8))

        browse_multi_btn = QPushButton("Batch...")
        browse_multi_btn.setFixedWidth(200)
        browse_multi_btn.clicked.connect(self._browse_files_batch)
        browse_multi_btn.setStyleSheet(get_button_style(bg_color='accent', padding=8))

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.path_edit, stretch=1)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(browse_multi_btn)

        layout.addLayout(file_layout)

        return group

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(Path.cwd()))
        if path:
            self.output_dir_edit.setText(path)

    def _browse_files_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select EMG Data Files",
            str(Path.cwd()),
            "EMG Files (*.mat *.npy *.csv *.h5 *.otb+);;All Files (*.*)"
        )
        if paths:
            self.emg_paths = [Path(p) for p in paths]
            self.emg_path = self.emg_paths[0]  # first file for channel estimation
            self.path_edit.setText(f"{len(paths)} files selected (first: {self.emg_path.name})")

            self.max_channels = self._estimate_channels_from_file(self.emg_path)
            self._update_file_info()
            self.allocation_bar.set_max_channels(self.max_channels)
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
        add_grid_btn.setStyleSheet(get_button_style(bg_color='success', padding=8))
        
        add_aux_btn = QPushButton("+ Add Aux")
        add_aux_btn.clicked.connect(lambda: self._add_aux_channel())
        add_aux_btn.setStyleSheet(get_button_style(bg_color='accent', padding=8))
        
        self.channel_summary_label = QLabel()
        self.channel_summary_label.setStyleSheet(
            get_label_style(size='small', color='text_dim')
        )
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all_channels)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 12px;
                font-size: {FONT_SIZES['normal']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['background_hover']};
                color: {COLORS['foreground']};
            }}
        """)
        
        action_layout.addWidget(add_grid_btn)
        action_layout.addWidget(add_aux_btn)
        action_layout.addStretch()
        action_layout.addWidget(self.channel_summary_label)
        action_layout.addWidget(clear_btn)
        layout.addLayout(action_layout)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {COLORS['background_input']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['text_muted']};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

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
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 16px;
                font-size: {FONT_SIZES['normal']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['background_hover']};
                color: {COLORS['foreground']};
            }}
        """)
        
        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self._load_config)
        load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 16px;
                font-size: {FONT_SIZES['normal']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['background_hover']};
                color: {COLORS['foreground']};
            }}
        """)
        
        bottom_layout.addWidget(save_btn)
        bottom_layout.addWidget(load_btn)
        bottom_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply Configuration →")
        self.apply_btn.setFixedHeight(40)
        self.apply_btn.setMinimumWidth(200)
        self.apply_btn.clicked.connect(self._apply_config)
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {COLORS['success']}, stop:1 #38A169);
                color: white; border-radius: 6px; font-weight: bold;
                font-size: {FONT_SIZES['medium']}; padding: 10px 24px;
            }}
            QPushButton:hover {{ background-color: #48BB78; }}
            QPushButton:pressed {{ background-color: #2F855A; }}
            QPushButton:disabled {{
                background-color: {COLORS['background_input']};
                color: {COLORS['text_muted']};
            }}
        """)
        
        bottom_layout.addWidget(self.apply_btn)
        layout.addLayout(bottom_layout)
        
        return group
    

    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                font-family: '{FONT_FAMILY}';
                font-size: {FONT_SIZES['large']};
                font-weight: bold;
                color: {COLORS['info']};
                border: 2px solid {COLORS['border']};
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
        presets_dir = Path(__file__).parent.parent.parent.parent.parent / "resources/loaders_configs"
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
                    print(f"Warning: Could not load preset {yaml_file.name}: {e}")

    def _on_loader_changed(self):
        if self.emg_path:
            self._update_file_info()

    def _get_current_layout(self) -> Optional[dict]:
        return self._loader_layouts.get(self.loader_combo.currentText())
    
    def _on_fsamp_changed(self):
        if self.emg_path:
            self._update_file_info()
    

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EMG Data",
            str(Path.cwd()),
            "EMG Files (*.mat *.npy *.csv *.h5 *.otb+);;OTB+ Files (*.otb+);;All Files (*.*)"
        )
        if path:
            self.emg_path = Path(path)
            self.emg_paths = [self.emg_path]
            self.path_edit.setText(path)
            
            self.max_channels = self._estimate_channels_from_file(self.emg_path)
            self._update_file_info()
            
            self.allocation_bar.set_max_channels(self.max_channels)
            self._update_summary()
            
            if not self.grid_cards:
                self._add_grid()

    def _estimate_channels_from_file(self, file_path: Path) -> int:
        layout = self._get_current_layout()
        if layout is None:
            return 256
        try:
            layout_full = copy.deepcopy(layout)
            layout_full["fields"]["emg"].pop("channels", None)
            emg = load_field(file_path, layout_full, "emg")
            return emg.shape[1]
        except Exception as e:
            print(f"Warning: Could not determine channel count: {e}")
            return 256

    def _update_file_info(self):
        layout = self._get_current_layout()
        if layout is None or self.emg_path is None:
            return
        try:
            layout_full = copy.deepcopy(layout)
            layout_full["fields"]["emg"].pop("channels", None)
            emg = load_field(self.emg_path, layout_full, "emg")
            n_samples, n_channels = emg.shape
            fs = int(self.fsamp_edit.text() or 2048)
            duration_sec = n_samples / fs
            self.max_channels = n_channels
            self.allocation_bar.set_max_channels(n_channels)
            self.file_info_label.setText(
                f"Loaded: {self.emg_path.name} | "
                f"Shape: {n_samples} samples × {n_channels} channels | "
                f"Duration: {duration_sec:.1f}s @ {fs} Hz"
            )
        except Exception as e:
            self.file_info_label.setText(f"⚠ Load failed: {e}")
            self.file_info_label.setStyleSheet(get_label_style(size='small', color='error'))

    def _add_grid(self):
        index = len(self.grid_cards) + 1
        color_idx = (index - 1) % len(GridCard.GRID_COLORS)
        color = GridCard.GRID_COLORS[color_idx]
        
        card = GridCard(index, color)
        
        next_start = self._get_next_available_channel()
        card.set_start_channel(next_start)
        card.set_end_channel(next_start + 64)
        
        card.remove_requested.connect(self._remove_grid)
        card.changed.connect(self._update_summary)
        
        # Grids insert at top, before aux cards and trailing stretch
        self.channels_layout.insertWidget(len(self.grid_cards), card)
        self.grid_cards.append(card)
        self._update_summary()
    
    def _get_next_available_channel(self) -> int:
        """Next channel after all grids and signal-source aux cards."""
        max_end = -1
        for card in self.grid_cards:
            _, end = card.get_channel_range()
            max_end = max(max_end, end)
        for card in self.aux_cards:
            if card.get_source() == "signal":
                _, end = card.get_channel_range()
                max_end = max(max_end, end)
        return max_end if max_end >= 0 else 0
    
    def _remove_grid(self, card: GridCard):
        if card in self.grid_cards:
            self.grid_cards.remove(card)
            self.channels_layout.removeWidget(card)
            card.deleteLater()
            self._renumber_grids()
            self._update_summary()
    
    def _renumber_grids(self):
        for i, card in enumerate(self.grid_cards):
            card.update_index(i + 1)
            color_idx = i % len(GridCard.GRID_COLORS)
            card.color = GridCard.GRID_COLORS[color_idx]
            card.color_indicator.setStyleSheet(
                f"background-color: {card.color}; border-radius: 2px;"
            )
    
    def _add_aux_channel(self, index: int = None):
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
            self, "Clear All Channels",
            "Remove all grid and auxiliary channel configurations?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for card in self.grid_cards + self.aux_cards:
                card.deleteLater()
            self.grid_cards.clear()
            self.aux_cards.clear()
            self._update_summary()
    
    def _validate_configuration(self) -> Tuple[bool, List[str]]:
        warnings = []
        
        if not self.grid_cards:
            warnings.append("No grids configured")
            return False, warnings
        
        for card in self.grid_cards:
            card.set_validation_status(True)
        for card in self.aux_cards:
            card.set_validation_status(True)
        
        # Collect all channel ranges (grids + signal-source aux)
        ranges = []
        
        for card in self.grid_cards:
            start, end = card.get_channel_range()
            name = card.get_data()['name']
            
            if end > self.max_channels:
                msg = "Exceeds available channels"
                warnings.append(f"{name}: channels exceed file ({end} > {self.max_channels})")
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
            if card.get_source() != "signal":
                continue
            start, end = card.get_channel_range()
            name = card.get_data()['name']
            
            if end > self.max_channels:
                msg = "Exceeds available channels"
                warnings.append(f"{name}: channels exceed file ({end} > {self.max_channels})")
                card.set_validation_status(False, msg)

            if start >= end:
                msg = "Start >= End"
                warnings.append(f"{name}: Start channel must be < End channel")
                card.set_validation_status(False, msg)
        
        return len(warnings) == 0, warnings
    
    def _update_summary(self):
        # Allocation bar: grids + signal-source aux
        allocations = []
        for card in self.grid_cards:
            data = card.get_data()
            start, end = card.get_channel_range()
            allocations.append((start, end, data['name'], data['color']))
        for card in self.aux_cards:
            if card.get_source() == "signal":
                data = card.get_data()
                start, end = card.get_channel_range()
                allocations.append((start, end, data['name'], AuxChannelCard.AUX_COLOR))
        self.allocation_bar.set_allocations(allocations)
        
        # Inline summary
        n_grids = len(self.grid_cards)
        n_aux = len(self.aux_cards)
        parts = []
        if n_grids:
            parts.append(f"{n_grids} grid{'s' if n_grids != 1 else ''}")
        if n_aux:
            parts.append(f"{n_aux} aux")
        self.channel_summary_label.setText(" · ".join(parts) if parts else "No channels")
        
        # Apply button
        if not self.emg_path or not self.grid_cards:
            self.apply_btn.setEnabled(False)
            return
        is_valid, _ = self._validate_configuration()
        self.apply_btn.setEnabled(is_valid)
    
    def _config_to_dict(self) -> dict:
        """Serialize current UI state to a JSON-friendly dict."""
        return {
            "version": 1,
            "loader": self.loader_combo.currentText(),
            "sampling_rate": int(self.fsamp_edit.text() or 2048),
            "file_path": str(self.emg_path) if self.emg_path else None,
            "output_dir": self.output_dir_edit.text(),
            "grids": [
                {
                    **card.get_data(),
                    "muscle": card.muscle_edit.text(),
                }
                for card in self.grid_cards
            ],
            "aux_channels": [card.get_data() for card in self.aux_cards],
        }
    
    def _config_from_dict(self, cfg: dict):
        """Restore UI state from a previously saved dict."""
        # Loader preset
        loader_name = cfg.get("loader", "")
        idx = self.loader_combo.findText(loader_name)
        if idx >= 0:
            self.loader_combo.setCurrentIndex(idx)
        
        # Sampling rate
        self.fsamp_edit.setText(str(cfg.get("sampling_rate", 2048)))
        
        # File path (don't auto-load, just set the path)
        file_path = cfg.get("file_path")
        if file_path and Path(file_path).exists():
            self.emg_path = Path(file_path)
            self.path_edit.setText(file_path)
            self.max_channels = self._estimate_channels_from_file(self.emg_path)
            self._update_file_info()
            self.allocation_bar.set_max_channels(self.max_channels)
        
        # Clear existing
        for card in self.grid_cards + self.aux_cards:
            card.deleteLater()
        self.grid_cards.clear()
        self.aux_cards.clear()
        
        # Restore grids
        for g in cfg.get("grids", []):
            self._add_grid()
            card = self.grid_cards[-1]
            card.set_values(
                name=g.get("name", ""),
                muscle=g.get("muscle", ""),
                electrode_type=g.get("type", "Surface"),
                config=g.get("config", ""),
                start=g.get("start_chan", 0),
                end=g.get("end_chan", 63),
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
            )
        
        self.output_dir_edit.setText(self.output_dir_edit.text())  
        self._update_summary()
    
    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Channel Configuration",
            str(Path.cwd() / "channel_config.json"),
            "JSON Files (*.json)"
        )
        if path:
            cfg = self._config_to_dict()
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
    
    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Channel Configuration",
            str(Path.cwd()),
            "JSON Files (*.json)"
        )
        if path:
            with open(path, 'r') as f:
                cfg = json.load(f)
            self._config_from_dict(cfg)

    def _apply_config(self):
        is_valid, warnings = self._validate_configuration()
        if not is_valid:
            QMessageBox.warning(
                self, "Configuration Invalid",
                "Please fix the following issues:\n\n" + "\n".join(warnings)
            )
            return
        
        try:
            fs = int(self.fsamp_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Sampling rate must be a number")
            return
        
        config = self.config_manager.create_default_session(name="Decomposition Session")
        config.sampling_frequency = fs
        config.input_dir = str(self.emg_path.parent)
        
        # Grids → ports
        for card in self.grid_cards:
            data = card.get_data()
            electrode_type = data['type']
            electrode_config = data['config']
            channels = list(range(data['start_chan'], data['end_chan']))
            
            if electrode_type in GridCard.ELECTRODE_CONFIGS:
                configs = GridCard.ELECTRODE_CONFIGS[electrode_type]
                if electrode_config in configs:
                    cfg = configs[electrode_config]
                    electrode = ElectrodeConfig(
                        name=electrode_config,
                        type=electrode_type.lower(),
                        channels=channels,
                        rows=cfg['rows'],
                        cols=cfg['cols'],
                        spacing_mm=cfg['spacing_mm'],
                    )
                    electrode.validate()
                    port = PortConfig(
                        name=data['name'],
                        electrode=electrode,
                        filter=FilterConfig(),
                        decomposition=DecompositionConfig(),
                        muscle=data.get('muscle', ''),
                    )
                    config.ports.append(port)

        # Aux channels
        config.aux_channels = [card.get_data() for card in self.aux_cards]

        config.data_layout = self._get_current_layout()
        config.output_dir = self.output_dir_edit.text() or str(self.emg_path.parent)
        config.emg_paths = [str(p) for p in self.emg_paths]
        self.config_applied.emit(config, self.emg_paths)