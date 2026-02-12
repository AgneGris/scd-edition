"""
Configuration Tab - Streamlined EMG data loading and electrode configuration.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import scipy.io as sio
import torch

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, 
    QSpinBox, QScrollArea, QFrame, 
    QMessageBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator

from core.config import (
    ConfigManager, SessionConfig, ElectrodeConfig, PortConfig,
    FilterConfig, DecompositionConfig
)

from gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, 
    get_button_style
)


class ChannelAllocationBar(QFrame):
    """Visual bar showing channel allocation across all grids."""
    
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
            "Grid (GR04MM1305)": {"rows": 13, "cols": 5, "spacing_mm": 4.0},
            "Grid (GR08MM1305)": {"rows": 13, "cols": 5, "spacing_mm": 8.0},
            "Grid (GR10MM0808)": {"rows": 8, "cols": 8, "spacing_mm": 10.0},
        },
        "Intramuscular": {
            "Thin-film (40ch)": {"rows": 20, "cols": 2, "spacing_mm": 2.5},
            "Wire needle": {"rows": 1, "cols": 16, "spacing_mm": 4.0},
            "Myomatrix": {"rows": 1, "cols": 16, "spacing_mm": 4.0},
        }
    }
    
    def __init__(self, index: int, color: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.color = color
        self._setup_ui()
        self._apply_styling()
    
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(10)
        
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(4, 20)
        self.color_indicator.setStyleSheet(f"background-color: {self.color}; border-radius: 2px;")
        main_layout.addWidget(self.color_indicator)
        
        self.name_edit = QLineEdit(f"Grid_{self.index}")
        self.name_edit.setPlaceholderText("e.g., Biceps")
        self.name_edit.textChanged.connect(self.changed.emit)
        main_layout.addWidget(self.name_edit, stretch=2)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Surface", "Intramuscular"])
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        main_layout.addWidget(self.type_combo, stretch=2)
        
        self.config_combo = QComboBox()
        self.config_combo.currentTextChanged.connect(self._on_config_change)
        main_layout.addWidget(self.config_combo, stretch=3)
        
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
        
        self.status_label = QLabel()
        self.status_label.setStyleSheet(get_label_style(size='small'))
        main_layout.addWidget(self.status_label, stretch=2)
        
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setToolTip("Remove Grid")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        self.remove_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: transparent; 
                color: {COLORS['text_muted']}; 
                border-radius: 10px; 
                font-weight: bold;
                font-size: 14pt;
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
            "type": self.type_combo.currentText(),
            "config": self.config_combo.currentText(),
            "start_chan": self.start_spin.value(),
            "end_chan": self.end_spin.value(),
            "color": self.color
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
        """Get number of channels from user-set start and end."""
        return self.end_spin.value() - self.start_spin.value() + 1
    
    def get_channel_range(self) -> Tuple[int, int]:
        """Get (start, end) channel indices from user input."""
        return self.start_spin.value(), self.end_spin.value()
    
    def set_start_channel(self, start: int):
        self.start_spin.setValue(start)
    
    def set_end_channel(self, end: int):
        self.end_spin.setValue(end)
    
    def set_values(self, name: str, electrode_type: str, config: str, start: int, end: int):
        self.name_edit.setText(name)
        
        type_idx = self.type_combo.findText(electrode_type)
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)
        
        config_idx = self.config_combo.findText(config)
        if config_idx >= 0:
            self.config_combo.setCurrentIndex(config_idx)
        
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)


class ConfigTab(QWidget):
    """Streamlined configuration tab for EMG data loading and grid setup."""
    
    config_applied = pyqtSignal(object, Path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = ConfigManager()
        self.emg_path: Optional[Path] = None
        self.max_channels: int = 256
        self.grid_cards: List[GridCard] = []
        
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
        main_layout.addWidget(self._create_grids_section(), stretch=1)
        main_layout.addWidget(self._create_summary_section(), stretch=0)
    
    def _create_file_section(self) -> QGroupBox:
        group = QGroupBox("1. Load EMG Data")
        group.setStyleSheet(f"""
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
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        file_layout = QHBoxLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select an EMG data file (.mat, .npy, .csv)...")
        self.path_edit.setReadOnly(True)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self._browse_file)
        browse_btn.setStyleSheet(get_button_style(bg_color='accent', padding=8))
        
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        self.file_info_label = QLabel()
        self.file_info_label.setStyleSheet(get_label_style(size='small', color='text_dim'))
        layout.addWidget(self.file_info_label)
        
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
        
        return group
    
    def _create_grids_section(self) -> QGroupBox:
        group = QGroupBox("2. Configure Electrode Grids")
        group.setStyleSheet(f"""
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
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        action_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Grid")
        add_btn.clicked.connect(self._add_grid)
        add_btn.setStyleSheet(get_button_style(bg_color='success', padding=8))
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_grids)
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
        
        action_layout.addWidget(add_btn)
        action_layout.addStretch()
        action_layout.addWidget(clear_btn)
        
        layout.addLayout(action_layout)
        
        headers_layout = QHBoxLayout()
        headers_layout.setContentsMargins(12, 0, 12, 0)
        headers_layout.setSpacing(10)
        
        color_spacer = QLabel()
        color_spacer.setFixedWidth(4)
        headers_layout.addWidget(color_spacer)
        
        name_header = QLabel("Name")
        name_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(name_header, stretch=2)
        
        type_header = QLabel("Type")
        type_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(type_header, stretch=2)
        
        config_header = QLabel("Configuration")
        config_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(config_header, stretch=3)
        
        start_header = QLabel("Start Ch")
        start_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(start_header, stretch=1)
        
        end_header = QLabel("End Ch")
        end_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(end_header, stretch=1)
        
        status_header = QLabel("Status")
        status_header.setStyleSheet(get_label_style(size='small', bold=True, color='text_secondary'))
        headers_layout.addWidget(status_header, stretch=2)
        
        remove_spacer = QLabel()
        remove_spacer.setFixedWidth(20)
        headers_layout.addWidget(remove_spacer)
        
        layout.addLayout(headers_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(400)
        scroll.setStyleSheet(f"background-color: {COLORS['background']};")
        
        scroll_content = QWidget()
        self.grids_layout = QVBoxLayout(scroll_content)
        self.grids_layout.setSpacing(6)
        self.grids_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)
        
        return group
    
    def _create_summary_section(self) -> QGroupBox:
        group = QGroupBox("3. Review & Apply")
        group.setStyleSheet(f"""
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
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        self.allocation_bar = ChannelAllocationBar()
        layout.addWidget(self.allocation_bar)
        
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply Configuration →")
        self.apply_btn.setFixedHeight(40)
        self.apply_btn.setMinimumWidth(200)
        self.apply_btn.clicked.connect(self._apply_config)
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {COLORS['success']}, stop:1 #38A169);
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: {FONT_SIZES['medium']};
                padding: 10px 24px;
            }}
            QPushButton:hover {{
                background-color: #48BB78;
            }}
            QPushButton:pressed {{
                background-color: #2F855A;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_input']};
                color: {COLORS['text_muted']};
            }}
        """)
        
        apply_layout.addWidget(self.apply_btn)
        layout.addLayout(apply_layout)
        
        return group
    
    def _show_initial_state(self):
        self.apply_btn.setEnabled(False)
        self._update_summary()
    
    def _on_fsamp_changed(self):
        """Update file info when sampling rate changes."""
        if self.emg_path:
            self._update_file_info()
    
    def _update_file_info(self):
        """Update file info label with current sampling rate."""
        try:
            data = self._load_data(self.emg_path)
            n_samples, n_channels = data.shape
            fs = int(self.fsamp_edit.text() or 2048)
            duration_sec = n_samples / fs
            
            self.file_info_label.setText(
                f"Loaded: {self.emg_path.name} | "
                f"Shape: {n_samples} samples × {n_channels} channels | "
                f"Duration: {duration_sec:.1f}s @ {fs} Hz"
            )
        except Exception as e:
            self.file_info_label.setText(
                f"File: {self.emg_path.name} | "
                f"Available Channels: {self.max_channels} (estimated)"
            )
    
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EMG Data",
            str(Path.cwd()),
            "EMG Files (*.mat *.npy *.csv);;All Files (*.*)"
        )
        
        if path:
            self.emg_path = Path(path)
            self.path_edit.setText(path)
            
            self.max_channels = self._estimate_channels_from_file(self.emg_path)
            self._update_file_info()
            
            self.allocation_bar.set_max_channels(self.max_channels)
            self._update_summary()
            
            if not self.grid_cards:
                self._add_grid()
    
    def _estimate_channels_from_file(self, file_path: Path) -> int:
        try:
            if file_path.suffix.lower() == '.mat':
                mat = sio.loadmat(str(file_path))
                
                for key in ['emg', 'data', 'sig', 'signal']:
                    if key in mat:
                        data = mat[key]
                        if data.shape[1] > data.shape[0]:
                            n_channels = data.shape[0]
                        else:
                            n_channels = data.shape[1]
                        return n_channels
                
                for key, value in mat.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        data = value
                        if data.shape[1] > data.shape[0]:
                            n_channels = data.shape[0]
                        else:
                            n_channels = data.shape[1]
                        return n_channels
            
            elif file_path.suffix.lower() == '.npy':
                data = np.load(str(file_path))
                if data.ndim == 2:
                    if data.shape[1] > data.shape[0]:
                        n_channels = data.shape[0]
                    else:
                        n_channels = data.shape[1]
                    return n_channels
            
            elif file_path.suffix.lower() == '.csv':
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    n_channels = len(first_line.split(','))
                return n_channels
        
        except Exception as e:
            print(f"Warning: Could not determine channel count: {e}")
            return 256
        
        return 256
    
    def _load_data(self, path: Path, key: str = "emg") -> torch.Tensor:
        if path.suffix.lower() == ".mat":
            mat = sio.loadmat(str(path))
            
            if key in mat:
                data = mat[key]
            else:
                for common_key in ['emg', 'data', 'sig', 'signal']:
                    if common_key in mat:
                        data = mat[common_key]
                        break
                else:
                    for k, v in mat.items():
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            data = v
                            break
                    else:
                        raise ValueError(f"No suitable data array found")
        
        elif path.suffix.lower() == ".npy":
            data = np.load(str(path))
        
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        neural_data = torch.from_numpy(data).to(dtype=torch.float32)
        
        if neural_data.shape[1] > neural_data.shape[0]:
            neural_data = neural_data.T
        
        return neural_data
    
    def _add_grid(self):
        index = len(self.grid_cards) + 1
        color_idx = (index - 1) % len(GridCard.GRID_COLORS)
        color = GridCard.GRID_COLORS[color_idx]
        
        card = GridCard(index, color)
        
        next_start = self._get_next_available_channel()
        card.set_start_channel(next_start)
        card.set_end_channel(next_start + 63)  # 0-63 = 64 channels total
        
        card.remove_requested.connect(self._remove_grid)
        card.changed.connect(self._update_summary)
        
        self.grids_layout.insertWidget(self.grids_layout.count() - 1, card)
        self.grid_cards.append(card)
        
        self._update_summary()
    
    def _get_next_available_channel(self) -> int:
        if not self.grid_cards:
            return 0
        
        max_end = -1
        for card in self.grid_cards:
            _, end = card.get_channel_range()
            max_end = max(max_end, end)
        
        return max_end + 1
    
    def _remove_grid(self, card: GridCard):
        """Removes a specific grid card and refreshes the layout."""
        if card in self.grid_cards:
            self.grid_cards.remove(card)
            self.grids_layout.removeWidget(card)
            card.deleteLater()
            self._renumber_grids()
            self._update_summary()
    
    def _clear_grids(self):
        reply = QMessageBox.question(
            self,
            "Clear All Grids",
            "Remove all grid configurations?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for card in self.grid_cards:
                card.deleteLater()
            self.grid_cards.clear()
            self._update_summary()
    
    def _renumber_grids(self):
        for i, card in enumerate(self.grid_cards):
            new_index = i + 1
            card.update_index(new_index)
            color_idx = i % len(GridCard.GRID_COLORS)
            card.color = GridCard.GRID_COLORS[color_idx]
            card.color_indicator.setStyleSheet(
                f"background-color: {card.color}; border-radius: 2px;"
            )
    
    def _validate_configuration(self) -> Tuple[bool, List[str]]:
        warnings = []
        
        if not self.grid_cards:
            warnings.append("No grids configured")
            return False, warnings
        
        for card in self.grid_cards:
            card.set_validation_status(True)
        
        ranges = []
        for i, card in enumerate(self.grid_cards):
            start, end = card.get_channel_range()
            
            if end >= self.max_channels:
                msg = f"Exceeds available channels"
                warnings.append(f"{card.get_data()['name']} exceeds available channels ({end} >= {self.max_channels})")
                card.set_validation_status(False, msg)
            
            if start > end:
                msg = f"Start > End"
                warnings.append(f"{card.get_data()['name']}: Start channel must be <= End channel")
                card.set_validation_status(False, msg)
            
            for j, (other_start, other_end, other_name) in enumerate(ranges):
                if not (end < other_start or start > other_end):
                    msg = f"Overlaps with {other_name}"
                    warnings.append(f"{card.get_data()['name']} overlaps with {other_name}")
                    card.set_validation_status(False, msg)
                    break
            
            ranges.append((start, end, card.get_data()['name']))
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def _update_summary(self):
        allocations = []
        for card in self.grid_cards:
            data = card.get_data()
            start, end = card.get_channel_range()
            allocations.append((start, end, data['name'], data['color']))
        
        self.allocation_bar.set_allocations(allocations)
        
        if not self.emg_path or not self.grid_cards:
            self.apply_btn.setEnabled(False)
            return
        
        is_valid, warnings = self._validate_configuration()
        self.apply_btn.setEnabled(is_valid)
    
    def _apply_config(self):
        is_valid, warnings = self._validate_configuration()
        
        if not is_valid:
            QMessageBox.warning(
                self,
                "Configuration Invalid",
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
        
        for card in self.grid_cards:
            data = card.get_data()
            electrode_type = data['type']
            electrode_config = data['config']
            start_chan = data['start_chan']
            end_chan = data['end_chan']
            
            channels = list(range(start_chan, end_chan + 1))
            
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
                        decomposition=DecompositionConfig()
                    )
                    
                    config.ports.append(port)
        
        self.config_applied.emit(config, self.emg_path)
    