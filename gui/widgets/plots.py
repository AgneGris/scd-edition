"""
Custom plot widgets for SCD Suite edition GUI.
"""

from typing import Optional, Tuple, List, Callable
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QColor, QPen, QBrush


class EditMode(Enum):
    """Spike editing modes."""
    VIEW = "view"
    ADD = "add"
    DELETE = "delete"
    ADD_ROI = "add_roi"
    DELETE_ROI = "delete_roi"


class SourcePlot(pg.PlotWidget):
    """
    Interactive source signal plot with spike markers.
    
    Signals
    -------
    spike_clicked : (int, str)
        Emitted when user clicks to add/delete spike. Args: sample position, action ("add"/"delete")
    roi_selected : (float, float, float, float)
        Emitted when ROI selection complete. Args: x_start, x_end, y_min, y_max
    view_changed : (float, float)
        Emitted when view range changes. Args: x_start, x_end in seconds
    """
    
    spike_clicked = pyqtSignal(int, str)  # sample, action
    roi_selected = pyqtSignal(float, float, float, float)  # x1, x2, y1, y2
    view_changed = pyqtSignal(float, float)  # x_start, x_end in seconds
    
    def __init__(self, fsamp: int = 10240, parent=None):
        super().__init__(parent)
        
        self.fsamp = fsamp
        self.edit_mode = EditMode.VIEW
        
        # Data
        self._source: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        
        # Plot items
        self._source_curve: Optional[pg.PlotDataItem] = None
        self._spike_scatter: Optional[pg.ScatterPlotItem] = None
        self._preview_marker: Optional[pg.ScatterPlotItem] = None
        self._roi: Optional[pg.LinearRegionItem] = None
        
        # Styling
        self.source_color = QColor(100, 180, 255)
        self.spike_color = QColor(255, 100, 100)
        self.preview_add_color = QColor(100, 255, 100, 150)
        self.preview_delete_color = QColor(255, 100, 100, 150)
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize plot styling and items."""
        self.setBackground('k')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Source amplitude')
        
        # Enable mouse interaction
        self.setMouseEnabled(x=True, y=True)
        self.scene().sigMouseClicked.connect(self._on_mouse_click)
        self.sigRangeChanged.connect(self._on_range_changed)
        
        # Create placeholder items
        self._source_curve = self.plot([], [], pen=pg.mkPen(self.source_color, width=1))
        self._spike_scatter = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(self.spike_color, width=2),
            brush=pg.mkBrush(self.spike_color),
            symbol='o'
        )
        self.addItem(self._spike_scatter)
        
        # Preview marker (for add/delete preview)
        self._preview_marker = pg.ScatterPlotItem(
            size=15,
            pen=pg.mkPen('w', width=2),
            brush=pg.mkBrush(self.preview_add_color),
            symbol='o'
        )
        self._preview_marker.setZValue(100)
        self.addItem(self._preview_marker)
        self._preview_marker.setData([], [])
    
    def set_data(self, source: np.ndarray, timestamps: np.ndarray):
        """Set source signal and spike timestamps."""
        self._source = np.asarray(source).squeeze()
        self._timestamps = np.asarray(timestamps).astype(int)
        
        # Update source curve
        time_axis = np.arange(len(self._source)) / self.fsamp
        self._source_curve.setData(time_axis, self._source)
        
        # Update spike markers
        self._update_spike_markers()
        
        # Auto-range
        self.autoRange()
    
    def set_timestamps(self, timestamps: np.ndarray):
        """Update only the spike timestamps."""
        self._timestamps = np.asarray(timestamps).astype(int)
        self._update_spike_markers()
    
    def _update_spike_markers(self):
        """Update spike scatter plot."""
        if self._source is None or self._timestamps is None:
            return
        
        valid_ts = self._timestamps[(self._timestamps >= 0) & (self._timestamps < len(self._source))]
        if len(valid_ts) == 0:
            self._spike_scatter.setData([], [])
            return
        
        x_pos = valid_ts / self.fsamp
        y_pos = self._source[valid_ts]
        self._spike_scatter.setData(x_pos, y_pos)
    
    def set_edit_mode(self, mode: EditMode):
        """Set the current editing mode."""
        self.edit_mode = mode
        
        # Update cursor
        if mode == EditMode.VIEW:
            self.setCursor(Qt.ArrowCursor)
        elif mode == EditMode.ADD:
            self.setCursor(Qt.CrossCursor)
        elif mode == EditMode.DELETE:
            self.setCursor(Qt.PointingHandCursor)
        elif mode in (EditMode.ADD_ROI, EditMode.DELETE_ROI):
            self.setCursor(Qt.CrossCursor)
            self._start_roi_selection()
    
    def _start_roi_selection(self):
        """Start ROI selection mode."""
        if self._roi is not None:
            self.removeItem(self._roi)
        
        # Get current view range
        view_range = self.viewRange()
        x_range = view_range[0]
        center = (x_range[0] + x_range[1]) / 2
        width = (x_range[1] - x_range[0]) * 0.1
        
        self._roi = pg.LinearRegionItem(
            values=[center - width/2, center + width/2],
            brush=pg.mkBrush(100, 100, 255, 50),
            movable=True
        )
        self.addItem(self._roi)
        self._roi.sigRegionChangeFinished.connect(self._on_roi_finished)
    
    def _on_roi_finished(self):
        """Handle ROI selection complete."""
        if self._roi is None:
            return
        
        x_range = self._roi.getRegion()
        y_range = self.viewRange()[1]
        
        self.roi_selected.emit(x_range[0], x_range[1], y_range[0], y_range[1])
        
        # Remove ROI
        self.removeItem(self._roi)
        self._roi = None
        self.edit_mode = EditMode.VIEW
        self.setCursor(Qt.ArrowCursor)
    
    def _on_mouse_click(self, event):
        """Handle mouse click for spike editing."""
        if self.edit_mode == EditMode.VIEW:
            return
        
        if self._source is None:
            return
        
        # Get click position in data coordinates
        pos = event.scenePos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        click_time = mouse_point.x()
        click_sample = int(click_time * self.fsamp)
        
        if click_sample < 0 or click_sample >= len(self._source):
            return
        
        if self.edit_mode == EditMode.ADD:
            self.spike_clicked.emit(click_sample, "add")
        elif self.edit_mode == EditMode.DELETE:
            self.spike_clicked.emit(click_sample, "delete")
    
    def _on_range_changed(self):
        """Handle view range change."""
        x_range = self.viewRange()[0]
        self.view_changed.emit(x_range[0], x_range[1])
    
    def show_preview(self, sample: int, action: str):
        """Show preview marker for add/delete action."""
        if self._source is None or sample < 0 or sample >= len(self._source):
            self._preview_marker.setData([], [])
            return
        
        x = sample / self.fsamp
        y = self._source[sample]
        
        if action == "add":
            self._preview_marker.setBrush(pg.mkBrush(self.preview_add_color))
        else:
            self._preview_marker.setBrush(pg.mkBrush(self.preview_delete_color))
        
        self._preview_marker.setData([x], [y])
    
    def clear_preview(self):
        """Clear preview marker."""
        self._preview_marker.setData([], [])
    
    def get_visible_range(self) -> Tuple[int, int]:
        """Get visible sample range."""
        x_range = self.viewRange()[0]
        return int(x_range[0] * self.fsamp), int(x_range[1] * self.fsamp)
    
    def get_y_range(self) -> Tuple[float, float]:
        """Get visible Y range."""
        return tuple(self.viewRange()[1])


class RasterPlot(pg.PlotWidget):
    """
    Raster plot showing spike times for multiple motor units.
    
    Signals
    -------
    unit_selected : int
        Emitted when a motor unit row is clicked
    """
    
    unit_selected = pyqtSignal(int)
    
    def __init__(self, fsamp: int = 10240, parent=None):
        super().__init__(parent)
        
        self.fsamp = fsamp
        self._timestamps_list: List[np.ndarray] = []
        self._scatter_items: List[pg.ScatterPlotItem] = []
        self._selected_unit: int = -1
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize plot styling."""
        self.setBackground('k')
        self.showGrid(x=True, y=False, alpha=0.3)
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Motor Unit')
        
        # Connect click
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def set_data(self, timestamps_list: List[np.ndarray], labels: List[str] = None):
        """Set spike data for all motor units."""
        # Clear existing
        for item in self._scatter_items:
            self.removeItem(item)
        self._scatter_items.clear()
        
        self._timestamps_list = timestamps_list
        n_units = len(timestamps_list)
        
        # Generate colors
        colors = [pg.intColor(i, n_units, maxValue=200) for i in range(n_units)]
        
        for i, timestamps in enumerate(timestamps_list):
            if len(timestamps) == 0:
                continue
            
            x = timestamps / self.fsamp
            y = np.full(len(timestamps), i)
            
            scatter = pg.ScatterPlotItem(
                x=x, y=y,
                size=3,
                pen=None,
                brush=pg.mkBrush(colors[i]),
                symbol='s'
            )
            self.addItem(scatter)
            self._scatter_items.append(scatter)
        
        # Set y-axis ticks
        if labels is None:
            labels = [f"MU {i}" for i in range(n_units)]
        
        y_axis = self.getAxis('left')
        ticks = [(i, labels[i]) for i in range(n_units)]
        y_axis.setTicks([ticks])
        
        self.setYRange(-0.5, n_units - 0.5)
    
    def set_selected_unit(self, unit_idx: int):
        """Highlight selected unit."""
        self._selected_unit = unit_idx
        
        # Update visual appearance
        for i, scatter in enumerate(self._scatter_items):
            if i == unit_idx:
                scatter.setSize(5)
                scatter.setSymbol('o')
            else:
                scatter.setSize(3)
                scatter.setSymbol('s')
    
    def _on_click(self, event):
        """Handle click to select unit."""
        pos = event.scenePos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        unit_idx = int(round(mouse_point.y()))
        
        if 0 <= unit_idx < len(self._timestamps_list):
            self.unit_selected.emit(unit_idx)


class FiringRatePlot(pg.PlotWidget):
    """
    Plot showing instantaneous firing rate over time.
    """
    
    def __init__(self, fsamp: int = 10240, parent=None):
        super().__init__(parent)
        
        self.fsamp = fsamp
        self._curve: Optional[pg.PlotDataItem] = None
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize plot styling."""
        self.setBackground('k')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Firing Rate', units='pps')
        
        self._curve = self.plot([], [], pen=pg.mkPen(QColor(100, 255, 150), width=1))
    
    def set_data(self, timestamps: np.ndarray):
        """Set spike timestamps and compute firing rate."""
        if len(timestamps) < 2:
            self._curve.setData([], [])
            return
        
        timestamps = np.sort(timestamps)
        
        # Compute instantaneous firing rate
        isi = np.diff(timestamps) / self.fsamp  # ISI in seconds
        rates = 1.0 / isi  # Instantaneous rate in Hz
        
        # Time points (midpoint between spikes)
        time_points = (timestamps[:-1] + timestamps[1:]) / (2 * self.fsamp)
        
        # Filter extreme values
        valid_mask = rates < 200  # Max 200 Hz
        
        self._curve.setData(time_points[valid_mask], rates[valid_mask])
    
    def link_x_axis(self, other_plot: pg.PlotWidget):
        """Link X axis with another plot."""
        self.setXLink(other_plot)


class QualityIndicator(QWidget):
    """Widget showing quality metrics for a motor unit."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(15)
        
        # Labels
        self.sil_label = QLabel("SIL: --")
        self.cov_label = QLabel("CoV: --")
        self.fr_label = QLabel("FR: -- Hz")
        self.spikes_label = QLabel("Spikes: --")
        
        for label in [self.sil_label, self.cov_label, self.fr_label, self.spikes_label]:
            label.setStyleSheet("color: #aaa; font-size: 11px;")
            layout.addWidget(label)
        
        layout.addStretch()
    
    def set_metrics(self, sil: float, cov: float, mean_fr: float, n_spikes: int):
        """Update displayed metrics."""
        # SIL with color coding
        sil_color = "#4a4" if sil >= 0.85 else "#a44" if sil < 0.7 else "#aa4"
        self.sil_label.setText(f"SIL: {sil:.2f}")
        self.sil_label.setStyleSheet(f"color: {sil_color}; font-size: 11px; font-weight: bold;")
        
        # CoV with color coding
        cov_color = "#4a4" if cov < 0.3 else "#a44" if cov > 0.6 else "#aa4"
        cov_text = f"{cov:.2f}" if cov < 10 else ">10"
        self.cov_label.setText(f"CoV: {cov_text}")
        self.cov_label.setStyleSheet(f"color: {cov_color}; font-size: 11px;")
        
        # Firing rate
        fr_color = "#4a4" if 2 < mean_fr < 50 else "#aa4"
        self.fr_label.setText(f"FR: {mean_fr:.1f} Hz")
        self.fr_label.setStyleSheet(f"color: {fr_color}; font-size: 11px;")
        
        # Spike count
        spikes_color = "#4a4" if n_spikes >= 20 else "#a44" if n_spikes < 5 else "#aa4"
        self.spikes_label.setText(f"Spikes: {n_spikes}")
        self.spikes_label.setStyleSheet(f"color: {spikes_color}; font-size: 11px;")
    
    def clear(self):
        """Clear all metrics."""
        for label in [self.sil_label, self.cov_label, self.fr_label, self.spikes_label]:
            label.setText(label.text().split(":")[0] + ": --")
            label.setStyleSheet("color: #aaa; font-size: 11px;")
