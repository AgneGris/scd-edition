"""
Edition tab for visual spike editing.
"""

from typing import Optional, List
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QAction, QActionGroup, QComboBox, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QFileDialog,
    QMessageBox, QShortcut, QStatusBar
)
from PyQt5.QtGui import QKeySequence, QIcon

from gui.widgets.plots import SourcePlot, RasterPlot, FiringRatePlot, QualityIndicator, EditMode
from core.data_handler import DataHandler, MotorUnit
from core.analysis import QualityAnalyzer, FilterRecalculator
from gui.style.styling import set_style_sheet

class EditionTab(QWidget):
    """
    Main edition interface for editing motor unit spike trains.
    
    Features:
    - Visual spike editing (add/delete single, ROI-based batch)
    - Source signal with spike markers
    - Raster plot overview
    - Instantaneous firing rate
    - Quality metrics display
    - Undo/redo support
    - Auto-save
    """
    
    data_modified = pyqtSignal()  # Emitted when data is edited
    
    def __init__(self, data_handler: DataHandler, parent=None):
        super().__init__(parent)
        set_style_sheet(self)
        self.data = data_handler
        self.analyzer = QualityAnalyzer(data_handler.fsamp)
        self.recalculator = FilterRecalculator(data_handler.fsamp)
        
        # Current state
        self._current_port: Optional[str] = None
        self._current_mu_idx: int = -1
        self._edit_mode = EditMode.VIEW
        
        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()
    
    def _setup_ui(self):
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Motor unit list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Port selector
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(self._on_port_changed)
        port_layout.addWidget(self.port_combo)
        left_layout.addLayout(port_layout)
        
        # Motor unit list
        self.mu_list = QListWidget()
        self.mu_list.currentRowChanged.connect(self._on_mu_selected)
        left_layout.addWidget(self.mu_list)
        
        # Unit controls
        btn_layout = QHBoxLayout()
        self.btn_disable = QPushButton("Disable")
        self.btn_disable.clicked.connect(self._toggle_unit_enabled)
        btn_layout.addWidget(self.btn_disable)
        self.btn_recalc = QPushButton("Recalculate")
        self.btn_recalc.clicked.connect(self._recalculate_filter)
        btn_layout.addWidget(self.btn_recalc)
        left_layout.addLayout(btn_layout)
        
        left_panel.setMaximumWidth(250)
        main_splitter.addWidget(left_panel)
        
        # Right panel: Plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Quality indicator bar
        self.quality_bar = QualityIndicator()
        right_layout.addWidget(self.quality_bar)
        
        # Plot splitter (vertical)
        plot_splitter = QSplitter(Qt.Vertical)
        
        # Source plot (main editing area)
        self.source_plot = SourcePlot(self.data.fsamp)
        self.source_plot.spike_clicked.connect(self._on_spike_click)
        self.source_plot.roi_selected.connect(self._on_roi_selected)
        plot_splitter.addWidget(self.source_plot)
        
        # Firing rate plot
        self.fr_plot = FiringRatePlot(self.data.fsamp)
        self.fr_plot.link_x_axis(self.source_plot)
        self.fr_plot.setMaximumHeight(150)
        plot_splitter.addWidget(self.fr_plot)
        
        # Raster plot
        self.raster_plot = RasterPlot(self.data.fsamp)
        self.raster_plot.unit_selected.connect(self._on_raster_unit_clicked)
        self.raster_plot.setMaximumHeight(200)
        # self.raster_plot.link_x_axis(self.source_plot)
        plot_splitter.addWidget(self.raster_plot)
        
        # Set initial sizes
        plot_splitter.setSizes([400, 100, 150])
        
        right_layout.addWidget(plot_splitter)
        main_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        main_splitter.setSizes([200, 800])
        
        layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self._update_status()
    
    def _create_toolbar(self) -> QToolBar:
        """Create the editing toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # File actions
        self.action_load = QAction("Load", self)
        self.action_load.triggered.connect(self._load_data)
        toolbar.addAction(self.action_load)
        
        self.action_save = QAction("Save", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self._save_data)
        toolbar.addAction(self.action_save)
        
        toolbar.addSeparator()
        
        # Edit mode actions (mutually exclusive)
        self.mode_group = QActionGroup(self)
        
        self.action_view = QAction("View", self)
        self.action_view.setCheckable(True)
        self.action_view.setChecked(True)
        self.action_view.triggered.connect(lambda: self._set_edit_mode(EditMode.VIEW))
        self.mode_group.addAction(self.action_view)
        toolbar.addAction(self.action_view)
        
        self.action_add = QAction("Add Spike", self)
        self.action_add.setCheckable(True)
        self.action_add.setShortcut(QKeySequence("A"))
        self.action_add.triggered.connect(lambda: self._set_edit_mode(EditMode.ADD))
        self.mode_group.addAction(self.action_add)
        toolbar.addAction(self.action_add)
        
        self.action_delete = QAction("Delete Spike", self)
        self.action_delete.setCheckable(True)
        self.action_delete.setShortcut(QKeySequence("D"))
        self.action_delete.triggered.connect(lambda: self._set_edit_mode(EditMode.DELETE))
        self.mode_group.addAction(self.action_delete)
        toolbar.addAction(self.action_delete)
        
        toolbar.addSeparator()
        
        # ROI actions
        self.action_add_roi = QAction("Add ROI", self)
        self.action_add_roi.setShortcut(QKeySequence("Shift+A"))
        self.action_add_roi.triggered.connect(lambda: self._set_edit_mode(EditMode.ADD_ROI))
        toolbar.addAction(self.action_add_roi)
        
        self.action_delete_roi = QAction("Delete ROI", self)
        self.action_delete_roi.setShortcut(QKeySequence("Shift+D"))
        self.action_delete_roi.triggered.connect(lambda: self._set_edit_mode(EditMode.DELETE_ROI))
        toolbar.addAction(self.action_delete_roi)
        
        toolbar.addSeparator()
        
        # Undo/Redo
        self.action_undo = QAction("Undo", self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.triggered.connect(self._undo)
        toolbar.addAction(self.action_undo)
        
        self.action_redo = QAction("Redo", self)
        self.action_redo.setShortcut(QKeySequence.Redo)
        self.action_redo.triggered.connect(self._redo)
        toolbar.addAction(self.action_redo)
        
        return toolbar
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Escape to return to view mode
        QShortcut(QKeySequence(Qt.Key_Escape), self, lambda: self._set_edit_mode(EditMode.VIEW))
        
        # Navigation
        QShortcut(QKeySequence("Up"), self, self._select_previous_unit)
        QShortcut(QKeySequence("Down"), self, self._select_next_unit)
    
    def _connect_signals(self):
        """Connect internal signals."""
        pass
    
    # === Data Loading ===
    
    def _load_data(self):
        """Load decomposition file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Decomposition",
            str(Path.cwd()),
            "Decomposition Files (*.pkl *.mat)"
        )
        
        if not path:
            return
        
        # TODO: Also load EMG if needed
        # For now, just load decomposition
        try:
            port_name = Path(path).stem
            self.data.load_decomposition(Path(path), port_name)
            self._refresh_port_list()
            self._update_status(f"Loaded: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
    
    def _save_data(self):
        """Save current data."""
        if self._current_port is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Decomposition",
            str(Path.cwd()),
            "Pickle Files (*.pkl)"
        )
        
        if not path:
            return
        
        try:
            self.data.save_decomposition(Path(path), self._current_port)
            self._update_status(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
    
    def load_port_data(self, port_name: str, motor_units: List[MotorUnit]):
        """Load data for a specific port (called externally)."""
        if port_name not in self.data.ports:
            from ...core.data_handler import PortData
            self.data.ports[port_name] = PortData(name=port_name)
        
        self.data.ports[port_name].motor_units = motor_units
        self._refresh_port_list()
        
        if len(self.data.ports) > 0:
            self.port_combo.setCurrentText(port_name)
    
    # === UI Updates ===
    
    def _refresh_port_list(self):
        """Refresh the port dropdown."""
        current = self.port_combo.currentText()
        self.port_combo.clear()
        
        for port_name in self.data.ports.keys():
            self.port_combo.addItem(port_name)
        
        if current in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
            self.port_combo.setCurrentText(current)
    
    def _refresh_mu_list(self):
        """Refresh the motor unit list."""
        self.mu_list.clear()
        
        if self._current_port is None:
            return
        
        port = self.data.ports.get(self._current_port)
        if port is None:
            return
        
        for mu in port.motor_units:
            text = f"MU {mu.id} ({len(mu.timestamps)} spikes)"
            if not mu.enabled:
                text += " [disabled]"
            if mu.flagged_duplicate:
                text += " ⚠"
            
            item = QListWidgetItem(text)
            if not mu.enabled:
                item.setForeground(Qt.gray)
            elif mu.flagged_duplicate:
                item.setForeground(Qt.yellow)
            
            self.mu_list.addItem(item)
    
    def _update_plots(self):
        """Update all plots with current motor unit data."""
        if self._current_port is None or self._current_mu_idx < 0:
            return
        
        try:
            mu = self.data.get_motor_unit(self._current_port, self._current_mu_idx)
        except KeyError:
            return
        
        # Update source plot
        self.source_plot.set_data(mu.source, mu.timestamps)
        
        # Update firing rate plot
        self.fr_plot.set_data(mu.timestamps)
        
        # Update raster plot
        port = self.data.ports[self._current_port]
        timestamps_list = [m.timestamps for m in port.motor_units]
        labels = [f"MU {m.id}" for m in port.motor_units]
        self.raster_plot.set_data(timestamps_list, labels)
        self.raster_plot.set_selected_unit(self._current_mu_idx)
        
        # Update quality metrics
        self._update_quality_metrics(mu)
    
    def _update_quality_metrics(self, mu: MotorUnit):
        """Update quality indicator display."""
        sil = self.analyzer.compute_sil(mu.source, mu.timestamps)
        cov = self.analyzer.compute_cov(mu.timestamps)
        fr = self.analyzer.compute_firing_rate(mu.timestamps)
        
        self.quality_bar.set_metrics(sil, cov, fr, len(mu.timestamps))
    
    def _update_status(self, message: str = None):
        """Update status bar."""
        if message:
            self.status_bar.showMessage(message, 3000)
        else:
            # Show current state
            parts = []
            if self._current_port:
                parts.append(f"Port: {self._current_port}")
            if self._current_mu_idx >= 0:
                parts.append(f"MU: {self._current_mu_idx}")
            parts.append(f"Mode: {self._edit_mode.value}")
            
            if self.data.can_undo():
                parts.append("(Undo available)")
            
            self.status_bar.showMessage(" | ".join(parts))
    
    # === Event Handlers ===
    
    def _on_port_changed(self, port_name: str):
        """Handle port selection change."""
        if not port_name:
            return
        
        self._current_port = port_name
        self._current_mu_idx = -1
        self._refresh_mu_list()
        
        # Select first unit
        if self.mu_list.count() > 0:
            self.mu_list.setCurrentRow(0)
    
    def _on_mu_selected(self, row: int):
        """Handle motor unit selection."""
        if row < 0:
            return
        
        self._current_mu_idx = row
        self._update_plots()
        self._update_status()
    
    def _on_raster_unit_clicked(self, unit_idx: int):
        """Handle click on raster plot."""
        if 0 <= unit_idx < self.mu_list.count():
            self.mu_list.setCurrentRow(unit_idx)
    
    def _on_spike_click(self, sample: int, action: str):
        """Handle spike add/delete click."""
        if self._current_port is None or self._current_mu_idx < 0:
            return
        
        if action == "add":
            # Find nearest peak
            mu = self.data.get_motor_unit(self._current_port, self._current_mu_idx)
            peak = self._find_nearest_peak(mu.source, sample)
            if peak is not None:
                self.data.add_spike(self._current_port, self._current_mu_idx, peak)
                self._on_data_modified()
        
        elif action == "delete":
            self.data.delete_spike(self._current_port, self._current_mu_idx, sample)
            self._on_data_modified()
    
    def _on_roi_selected(self, x1: float, x2: float, y1: float, y2: float):
        """Handle ROI selection for batch operations."""
        if self._current_port is None or self._current_mu_idx < 0:
            return
        
        if self._edit_mode == EditMode.ADD_ROI:
            self.data.add_spikes_roi(
                self._current_port, self._current_mu_idx,
                x1, x2, y1, y2
            )
        elif self._edit_mode == EditMode.DELETE_ROI:
            self.data.delete_spikes_roi(
                self._current_port, self._current_mu_idx,
                x1, x2, y1, y2
            )
        
        self._set_edit_mode(EditMode.VIEW)
        self._on_data_modified()
    
    def _on_data_modified(self):
        """Called after any data modification."""
        # Update plots
        self._update_plots()
        
        # Update list display
        self._refresh_mu_list()
        self.mu_list.setCurrentRow(self._current_mu_idx)
        
        # Update status
        self._update_status("Modified")
        
        # Emit signal
        self.data_modified.emit()
        
        # Auto-save if enabled
        # TODO: Implement auto-save
    
    # === Edit Operations ===
    
    def _set_edit_mode(self, mode: EditMode):
        """Set the current editing mode."""
        self._edit_mode = mode
        self.source_plot.set_edit_mode(mode)
        
        # Update toolbar
        if mode == EditMode.VIEW:
            self.action_view.setChecked(True)
        elif mode == EditMode.ADD:
            self.action_add.setChecked(True)
        elif mode == EditMode.DELETE:
            self.action_delete.setChecked(True)
        
        self._update_status()
    
    def _find_nearest_peak(self, source: np.ndarray, click_sample: int) -> Optional[int]:
        """Find nearest peak to click position."""
        from scipy import signal
        
        # Search window
        window = int(0.01 * self.data.fsamp)  # 10ms window
        start = max(0, click_sample - window)
        end = min(len(source), click_sample + window)
        
        segment = source[start:end]
        peaks, _ = signal.find_peaks(segment, distance=int(0.005 * self.data.fsamp))
        
        if len(peaks) == 0:
            return None
        
        peaks_abs = peaks + start
        distances = np.abs(peaks_abs - click_sample)
        return int(peaks_abs[np.argmin(distances)])
    
    def _undo(self):
        """Undo last action."""
        if self.data.undo():
            self._on_data_modified()
            self._update_status("Undone")
    
    def _redo(self):
        """Redo last undone action."""
        if self.data.redo():
            self._on_data_modified()
            self._update_status("Redone")
    
    def _toggle_unit_enabled(self):
        """Toggle current unit enabled state."""
        if self._current_port is None or self._current_mu_idx < 0:
            return
        
        mu = self.data.get_motor_unit(self._current_port, self._current_mu_idx)
        mu.enabled = not mu.enabled
        self._refresh_mu_list()
        self.mu_list.setCurrentRow(self._current_mu_idx)
    
    def _recalculate_filter(self):
        """Recalculate filter for current motor unit."""
        if self._current_port is None or self._current_mu_idx < 0:
            return
        
        port = self.data.ports[self._current_port]
        if port.emg_whitened is None:
            QMessageBox.warning(self, "Error", "No whitened EMG available for recalculation")
            return
        
        mu = self.data.get_motor_unit(self._current_port, self._current_mu_idx)
        
        try:
            new_source, new_timestamps = self.recalculator.recalculate(
                port.emg_whitened,
                mu.timestamps,
            )
            
            # Save for undo
            from ...core.data_handler import EditAction
            action = EditAction(
                action_type="recalculate",
                mu_id=mu.id,
                old_timestamps=mu.timestamps.copy(),
                new_timestamps=new_timestamps,
                old_source=mu.source.copy(),
                new_source=new_source,
            )
            self.data._save_undo(action)
            
            # Apply
            mu.timestamps = new_timestamps
            mu.source = new_source.squeeze()
            
            self._on_data_modified()
            self._update_status(f"Recalculated: {len(new_timestamps)} spikes")
            
        except Exception as e:
            QMessageBox.warning(self, "Recalculation Error", str(e))
    
    def _select_previous_unit(self):
        """Select previous motor unit."""
        current = self.mu_list.currentRow()
        if current > 0:
            self.mu_list.setCurrentRow(current - 1)
    
    def _select_next_unit(self):
        """Select next motor unit."""
        current = self.mu_list.currentRow()
        if current < self.mu_list.count() - 1:
            self.mu_list.setCurrentRow(current + 1)
