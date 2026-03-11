from typing import Optional, Tuple
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import pyqtgraph as pg

from gui.style.styling import COLORS
from scd_app.core.mu_model import EditMode

class SourcePlotWidget(pg.PlotWidget):
    """Source signal plot with spike markers and ROI support."""

    spike_add_requested = pyqtSignal(int)
    spike_delete_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS['background'])

        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get('text_dim', '#6c7086'))
        self.setLabel("left", "Amplitude", color=COLORS.get('text_dim', '#6c7086'))
        self.getAxis("bottom").setPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("left").setPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("bottom").setTextPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("left").setTextPen(COLORS.get('text_dim', '#6c7086'))

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
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
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

