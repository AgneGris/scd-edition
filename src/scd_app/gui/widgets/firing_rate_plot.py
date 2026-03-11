from typing import Optional
import numpy as np

from PyQt5.QtGui import QColor
import pyqtgraph as pg

from gui.style.styling import COLORS

class FiringRatePlotWidget(pg.PlotWidget):
    """Instantaneous firing rate display."""

    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS['background'])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get('text_dim', '#6c7086'))
        self.setLabel("left", "IFR (Hz)", color=COLORS.get('text_dim', '#6c7086'))
        self.getAxis("bottom").setPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("left").setPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("bottom").setTextPen(COLORS.get('text_dim', '#6c7086'))
        self.getAxis("left").setTextPen(COLORS.get('text_dim', '#6c7086'))
        self._curve = self.plot([], pen=pg.mkPen(COLORS.get('warning', '#f9e2af'), width=1.5))
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

