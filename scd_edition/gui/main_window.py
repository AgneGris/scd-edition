"""
Main application window.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog,
    QMessageBox, QButtonGroup, QShortcut, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QColor

from scd_edition.core.data_handler import DataHandler
from scd_edition.core.spike_editor import SpikeEditor
from scd_edition.core.analysis import SpikeTrain, QualityMetrics, DuplicateDetector
from scd_edition.gui.styles import PlotColors
from scd_edition.export.emglab import export_to_emglab


class MainWindow(QMainWindow):
    """Main application window for EMG decomposition editing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCD Edition - EMG Decomposition Editor")
        self.setGeometry(100, 100, 1800, 1500)
        
        # Core modules
        self.data = DataHandler()
        self.editor = SpikeEditor()
        self.detector = DuplicateDetector()
        
        # State
        self.flagged_units: list = []
        self.output_folder: Optional[Path] = None
        self.y_range_fixed: Optional[tuple] = None
        self.x_range_fixed: Optional[tuple] = None
        self.edit_mode: str = "view"
        self.roi_spikes = None
        
        self._setup_ui()
        self._setup_shortcuts()
    
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        
        self._create_top_row(main_layout)
        self._create_plots_section(main_layout)
        self._create_bottom_row(main_layout)
    
    def _create_top_row(self, main_layout):
        """Create top control row."""
        top_row = QHBoxLayout()
        
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self._load_data)
        top_row.addWidget(load_btn)
        
        output_btn = QPushButton("Set Output Folder")
        output_btn.clicked.connect(self._set_output_folder)
        top_row.addWidget(output_btn)
        
        self.output_label = QLabel("Output: Not Set")
        self.output_label.setMinimumWidth(200)
        top_row.addWidget(self.output_label)
        
        top_row.addWidget(QLabel("Motor Unit:"))
        self.unit_selector = QComboBox()
        self.unit_selector.currentIndexChanged.connect(self._update_plots)
        self.unit_selector.setMinimumWidth(150)
        top_row.addWidget(self.unit_selector)
        
        self.sil_label = QLabel("SIL: N/A")
        self.sil_label.setMinimumWidth(120)
        top_row.addWidget(self.sil_label)
        
        top_row.addStretch()
        main_layout.addLayout(top_row)
    
    def _create_plots_section(self, main_layout):
        """Create plots section with controls."""
        plots_widget = QWidget()
        plots_layout = QGridLayout(plots_widget)
        plots_layout.setSpacing(15)
        
        source_controls = self._create_source_controls()
        plots_layout.addLayout(source_controls, 0, 0)
        
        muap_controls = self._create_muap_controls()
        plots_layout.addLayout(muap_controls, 0, 1)
        
        self.mode_hint = QLabel("💡 Ctrl+Click to add | Alt+Click to delete")
        self.mode_hint.setStyleSheet("color: #7a7f87; font-size: 11pt;")
        plots_layout.addWidget(self.mode_hint, 1, 0)
        
        plots_layout.addWidget(QLabel("Source Signal with Spikes"), 2, 0)
        self.muap_label = QLabel("Motor Unit Action Potentials")
        plots_layout.addWidget(self.muap_label, 2, 1)
        
        self.source_plot = pg.PlotWidget()
        self.source_plot.showGrid(x=True, y=True, alpha=0.3)
        self.source_plot.setLabel('bottom', 'Time', units='s')
        self.source_plot.setLabel('left', 'Amplitude')
        self.source_plot.scene().sigMouseClicked.connect(self._on_plot_click)
        plots_layout.addWidget(self.source_plot, 3, 0)
        
        self.muap_plot = pg.PlotWidget()
        self.muap_plot.showGrid(x=True, y=True, alpha=0.3)
        self.muap_plot.setLabel('bottom', 'Time', units='ms')
        self.muap_plot.setLabel('left', 'Amplitude')
        plots_layout.addWidget(self.muap_plot, 3, 1)
        
        plots_layout.addWidget(QLabel("Instantaneous Discharge Rate"), 4, 0, 1, 2)
        
        self.discharge_plot = pg.PlotWidget()
        self.discharge_plot.showGrid(x=True, y=True, alpha=0.3)
        self.discharge_plot.setLabel('bottom', 'Time', units='s')
        self.discharge_plot.setLabel('left', 'Rate', units='pps')
        plots_layout.addWidget(self.discharge_plot, 5, 0, 1, 2)
        
        self.source_plot.sigRangeChanged.connect(self._sync_discharge_x_range)
        
        plots_layout.setColumnStretch(0, 3)
        plots_layout.setColumnStretch(1, 2)
        plots_layout.setRowStretch(3, 2)
        plots_layout.setRowStretch(5, 1)
        
        main_layout.addWidget(plots_widget, stretch=1)
    
    def _create_source_controls(self) -> QHBoxLayout:
        """Create source plot control buttons."""
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Edit Mode:"))
        
        self.mode_button_group = QButtonGroup()
        
        self.view_btn = QPushButton("View (V)")
        self.view_btn.setCheckable(True)
        self.view_btn.setChecked(True)
        self.view_btn.clicked.connect(lambda: self._set_edit_mode("view"))
        self.mode_button_group.addButton(self.view_btn)
        controls.addWidget(self.view_btn)
        
        self.add_btn = QPushButton("Add (A)")
        self.add_btn.setCheckable(True)
        self.add_btn.clicked.connect(lambda: self._set_edit_mode("add"))
        self.mode_button_group.addButton(self.add_btn)
        controls.addWidget(self.add_btn)
        
        self.delete_btn = QPushButton("Delete (D)")
        self.delete_btn.setCheckable(True)
        self.delete_btn.clicked.connect(lambda: self._set_edit_mode("delete"))
        self.mode_button_group.addButton(self.delete_btn)
        controls.addWidget(self.delete_btn)
        
        controls.addWidget(QLabel("|"))
        
        self.roi_checkbox = QCheckBox("ROI Mode (R)")
        self.roi_checkbox.toggled.connect(self._toggle_roi)
        controls.addWidget(self.roi_checkbox)
        
        roi_add_btn = QPushButton("ROI Add")
        roi_add_btn.clicked.connect(self._add_spikes_roi)
        controls.addWidget(roi_add_btn)
        
        roi_del_btn = QPushButton("ROI Delete")
        roi_del_btn.clicked.connect(self._delete_spikes_roi)
        controls.addWidget(roi_del_btn)
        
        controls.addWidget(QLabel("|"))
        
        undo_btn = QPushButton("Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self._undo)
        controls.addWidget(undo_btn)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        return controls
    
    def _create_muap_controls(self) -> QHBoxLayout:
        """Create MUAP plot control buttons."""
        controls = QHBoxLayout()
        
        flag_btn = QPushButton("Flag Unit")
        flag_btn.clicked.connect(self._flag_unit)
        controls.addWidget(flag_btn)
        
        duplicates_btn = QPushButton("Remove Duplicates")
        duplicates_btn.clicked.connect(self._remove_duplicates)
        controls.addWidget(duplicates_btn)
        
        recalc_btn = QPushButton("Recalculate Filters")
        recalc_btn.clicked.connect(self._recalculate_filters)
        controls.addWidget(recalc_btn)
        
        delete_flagged_btn = QPushButton("Delete Flagged")
        delete_flagged_btn.clicked.connect(self._delete_flagged_units)
        controls.addWidget(delete_flagged_btn)
        
        controls.addStretch()
        return controls
    
    def _create_bottom_row(self, main_layout):
        """Create bottom save buttons row."""
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        
        convert_btn = QPushButton("Convert to EMGlab (.eaf)")
        convert_btn.clicked.connect(self._convert_to_emglab)
        bottom_row.addWidget(convert_btn)
        
        save_btn = QPushButton("Save Edited Data")
        save_btn.clicked.connect(self._save_data)
        bottom_row.addWidget(save_btn)
        
        main_layout.addLayout(bottom_row)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        shortcuts = [
            (Qt.Key_Return, self._handle_enter),
            (Qt.Key_Enter, self._handle_enter),
            (Qt.Key_Escape, self._handle_escape),
            (Qt.Key_V, lambda: self._set_edit_mode("view")),
            (Qt.Key_A, lambda: self._set_edit_mode("add")),
            (Qt.Key_D, lambda: self._set_edit_mode("delete")),
            (Qt.Key_R, lambda: self.roi_checkbox.setChecked(not self.roi_checkbox.isChecked())),
        ]
        
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
        
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.setContext(Qt.ApplicationShortcut)
        undo_shortcut.activated.connect(self._undo)
    
    # === Data Loading (SIMPLE) ===
    
    def _load_data(self):
        """Load decomposition and EMG data."""
        # Select decomposition file
        decomp_file, _ = QFileDialog.getOpenFileName(
            self, "Select Decomposition File (.pkl)", "", "Pickle Files (*.pkl)"
        )
        if not decomp_file:
            return
        
        # Select EMG file
        emg_file, _ = QFileDialog.getOpenFileName(
            self, "Select EMG File (.mat or .npy)", "", "Data Files (*.mat *.npy)"
        )
        if not emg_file:
            return
        
        try:
            # Load data using simple interface
            self.data.load(Path(decomp_file), Path(emg_file))
            
            # Reset state
            self.flagged_units = []
            self.y_range_fixed = None
            self.x_range_fixed = None
            self.editor = SpikeEditor()
            self.detector = DuplicateDetector(fsamp=self.data.fsamp)
            
            # Update UI
            self.unit_selector.clear()
            for i in range(self.data.n_units):
                self.unit_selector.addItem(f"Unit {i + 1}")
            
            self._update_plots()
            
            QMessageBox.information(self, "Success", f"Loaded {self.data.n_units} motor units")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _set_output_folder(self):
        """Set output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = Path(folder)
            self.output_label.setText(f"Output: {self.output_folder.name}")
    
    # === Plot Updates ===
    
    def _update_plots(self, preserve_view: bool = False):
        """Update all plots for selected unit."""
        if self.data.sources is None or self.data.n_units == 0:
            return
        
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx < 0 or unit_idx >= self.data.n_units:
            return
        
        x_range, y_range = None, None
        if preserve_view:
            x_range, y_range = self.source_plot.viewRange()
        
        source, timestamps = self.data.get_unit_data(unit_idx)
        time_axis = np.arange(len(source)) / self.data.fsamp
        
        if self.y_range_fixed is None:
            ymin, ymax = float(np.min(source)), float(np.max(source))
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
            self.y_range_fixed = (ymin - margin, ymax + margin)
        if self.x_range_fixed is None:
            self.x_range_fixed = (0.0, float(time_axis[-1]))
        
        # Plot source
        self.source_plot.clear()
        self.source_plot.plot(
            time_axis, source,
            pen=pg.mkPen(color=PlotColors.SOURCE_LINE, width=1)
        )
        
        # Plot spikes
        if len(timestamps) > 0:
            valid_idx = timestamps[timestamps < len(source)]
            spike_times = valid_idx / self.data.fsamp
            spike_values = source[valid_idx]
            self.source_plot.plot(
                spike_times, spike_values,
                pen=None, symbol='o',
                symbolBrush=PlotColors.SPIKE_MARKER, symbolSize=8
            )
        
        if preserve_view and x_range and y_range:
            self.source_plot.setXRange(*x_range, padding=0)
            self.source_plot.setYRange(*y_range, padding=0)
        else:
            self.source_plot.setXRange(*self.x_range_fixed, padding=0)
            self.source_plot.setYRange(*self.y_range_fixed, padding=0)
        
        title = f"Unit {unit_idx + 1} - {len(timestamps)} spikes"
        if unit_idx in self.flagged_units:
            title = f"<span style='color: red;'>{title} - FLAGGED</span>"
        self.source_plot.setTitle(title)
        
        sil = QualityMetrics.calculate_sil(source, timestamps, self.data.fsamp)
        self.sil_label.setText(f"SIL: {sil:.3f}")
        
        self._plot_muap(unit_idx)
        self._plot_discharge_rate(unit_idx)
        
        if not self.editor.preview_active:
            self.muap_label.setText("Motor Unit Action Potentials")
    
    def _plot_muap(self, unit_idx: int, preview_snippet=None, preview_mode="add"):
        """Plot MUAP for all channels."""
        self.muap_plot.clear()
        
        timestamps = self.data.timestamps[unit_idx]
        emg = self.data.emg_data_filtered
        
        if len(timestamps) < 5 or emg is None:
            return
        
        hw = int(0.02 * self.data.fsamp)  # half window = 20ms
        window = 2 * hw  # full window
        time_ms = (np.arange(window) - hw) / self.data.fsamp * 1000
        
        emg_windows = []
        for ts in timestamps:
            ts = int(ts)
            if ts - hw >= 0 and ts + hw <= emg.shape[1]:
                emg_windows.append(emg[:, ts - hw:ts + hw])
        
        if len(emg_windows) < 5:
            return
        
        sta = np.mean(emg_windows, axis=0)
        n_channels = sta.shape[0]
        
        max_channel = np.argmax(np.ptp(sta, axis=1))
        max_amplitude = np.max(np.abs(sta))
        offset_step = max_amplitude * 0.5
        
        preview_emg = None
        if preview_snippet is not None and self.editor.preview_location is not None:
            ts = int(self.editor.preview_location)
            if ts - hw >= 0 and ts + hw <= emg.shape[1]:
                preview_emg = emg[:, ts - hw:ts + hw]
        
        preview_color = PlotColors.PREVIEW_ADD if preview_mode == "add" else PlotColors.PREVIEW_DELETE
        
        for ch in range(n_channels):
            y_offset = (n_channels - ch - 1) * offset_step
            muap_y = sta[ch, :] + y_offset
            
            color = PlotColors.MUAP_MAX_CHANNEL if ch == max_channel else PlotColors.MUAP_OTHER_CHANNEL
            width = 3 if ch == max_channel else 2
            
            self.muap_plot.plot(time_ms, muap_y, pen=pg.mkPen(color=color, width=width))
            
            if preview_emg is not None:
                preview_y = preview_emg[ch, :] + y_offset
                style = Qt.SolidLine if ch == max_channel else Qt.DashLine
                self.muap_plot.plot(
                    time_ms, preview_y,
                    pen=pg.mkPen(color=preview_color, width=width, style=style)
                )
        
        self.muap_plot.setYRange(-offset_step, n_channels * offset_step, padding=0.05)
    
    def _plot_discharge_rate(self, unit_idx: int):
        """Plot instantaneous discharge rate."""
        self.discharge_plot.clear()
        
        timestamps = self.data.timestamps[unit_idx]
        if len(timestamps) < 2:
            return
        
        time_points, rates = SpikeTrain.calculate_discharge_rate(timestamps, self.data.fsamp)
        
        if len(rates) == 0:
            return
        
        self.discharge_plot.plot(
            time_points, rates,
            pen=pg.mkPen(color=PlotColors.SPIKE_MARKER, width=2),
            symbol='o', symbolBrush=None,
            symbolPen=pg.mkPen(color=PlotColors.SPIKE_MARKER, width=2),
            symbolSize=8
        )
        
        median_rate = np.median(rates)
        self.discharge_plot.addLine(
            y=median_rate,
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        
        mean_rate = np.mean(rates)
        cv = np.std(rates) / mean_rate if mean_rate > 0 else 0
        self.discharge_plot.setTitle(
            f"Discharge Rate - Mean: {mean_rate:.1f} pps, Median: {median_rate:.1f} pps, CV: {cv:.2f}"
        )
    
    def _sync_discharge_x_range(self):
        """Sync discharge plot X-range with source plot."""
        x_range, _ = self.source_plot.viewRange()
        self.discharge_plot.setXRange(*x_range, padding=0)
    
    def _reset_view(self):
        """Reset plot views to original."""
        if self.y_range_fixed and self.x_range_fixed:
            self.source_plot.setYRange(*self.y_range_fixed, padding=0)
            self.source_plot.setXRange(*self.x_range_fixed, padding=0)
            self.discharge_plot.setXRange(*self.x_range_fixed, padding=0)
            self.discharge_plot.enableAutoRange(axis='y')
            self.muap_plot.enableAutoRange()
    
    # === Edit Mode ===
    
    def _set_edit_mode(self, mode: str):
        """Set edit mode and update UI."""
        if self.editor.preview_active:
            self._cancel_preview()
        
        self.edit_mode = mode
        self.view_btn.setChecked(mode == "view")
        self.add_btn.setChecked(mode == "add")
        self.delete_btn.setChecked(mode == "delete")
        
        hints = {
            "view": "💡 Ctrl+Click to add | Alt+Click to delete",
            "add": "✏️ ADD MODE: Click to preview → Enter to confirm",
            "delete": "🗑️ DELETE MODE: Click to preview → Enter to confirm"
        }
        cursors = {
            "view": Qt.ArrowCursor,
            "add": Qt.CrossCursor,
            "delete": Qt.PointingHandCursor
        }
        
        self.mode_hint.setText(hints.get(mode, ""))
        self.source_plot.setCursor(cursors.get(mode, Qt.ArrowCursor))
    
    def _on_plot_click(self, event):
        """Handle mouse clicks on source plot."""
        if self.data.sources is None:
            return
        
        mouse_point = self.source_plot.plotItem.vb.mapSceneToView(event.scenePos())
        click_sample = int(mouse_point.x() * self.data.fsamp)
        
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx < 0:
            return
        
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ControlModifier or self.edit_mode == "add":
            if not self.editor.preview_active:
                self._show_add_preview(click_sample)
        elif modifiers == Qt.AltModifier or self.edit_mode == "delete":
            if not self.editor.preview_active:
                self._show_delete_preview(click_sample)
    
    def _show_add_preview(self, click_sample: int):
        """Show preview for adding a spike."""
        unit_idx = self.unit_selector.currentIndex()
        source, timestamps = self.data.get_unit_data(unit_idx)
        
        x_range, y_range = self.source_plot.viewRange()
        view_range = (int(x_range[0] * self.data.fsamp), int(x_range[1] * self.data.fsamp))
        
        nearest_peak = self.editor.find_nearest_peak(
            source, click_sample, view_range, tuple(y_range),
            self.data.fsamp, timestamps
        )
        
        if nearest_peak is None:
            return
        
        self.editor.start_preview(nearest_peak, "add")
        self._plot_muap(unit_idx, preview_snippet=True, preview_mode="add")
        self.muap_label.setText(
            f"<span style='color: #00ff88;'>⚡ PREVIEW ADD at {nearest_peak/self.data.fsamp:.3f}s</span>"
        )
    
    def _show_delete_preview(self, click_sample: int):
        """Show preview for deleting a spike."""
        unit_idx = self.unit_selector.currentIndex()
        timestamps = self.data.timestamps[unit_idx]
        
        x_range, _ = self.source_plot.viewRange()
        view_range = (int(x_range[0] * self.data.fsamp), int(x_range[1] * self.data.fsamp))
        
        nearest_spike = self.editor.find_nearest_spike(timestamps, click_sample, view_range)
        
        if nearest_spike is None:
            return
        
        self.editor.start_preview(nearest_spike, "delete")
        self._plot_muap(unit_idx, preview_snippet=True, preview_mode="delete")
        self.muap_label.setText(
            f"<span style='color: #ff6666;'>🗑️ PREVIEW DELETE at {nearest_spike/self.data.fsamp:.3f}s</span>"
        )
    
    def _handle_enter(self):
        """Handle Enter key - confirm preview."""
        if not self.editor.preview_active:
            return
        
        unit_idx = self.unit_selector.currentIndex()
        
        self.editor.save_state(self.data.timestamps, self.data.sources)
        
        if self.editor.preview_action == "add":
            self.data.timestamps[unit_idx] = self.editor.add_spike(
                self.data.timestamps[unit_idx],
                self.editor.preview_location
            )
        elif self.editor.preview_action == "delete":
            self.data.timestamps[unit_idx] = self.editor.delete_spike(
                self.data.timestamps[unit_idx],
                self.editor.preview_location
            )
        
        self.editor.clear_preview()
        self._update_plots(preserve_view=True)
    
    def _handle_escape(self):
        """Handle Escape key - cancel preview."""
        self._cancel_preview()
    
    def _cancel_preview(self):
        """Cancel current preview."""
        if not self.editor.preview_active:
            return
        
        self.editor.clear_preview()
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx >= 0:
            self._plot_muap(unit_idx)
            self.muap_label.setText("Motor Unit Action Potentials")
    
    def _undo(self):
        """Undo last edit."""
        state = self.editor.undo()
        if state is None:
            QMessageBox.information(self, "No History", "Nothing to undo")
            return
        
        self.data.timestamps = [ts.astype(int) for ts in state.timestamps]
        if state.sources is not None:
            self.data.sources = state.sources
        
        self._update_plots(preserve_view=True)
    
    # === ROI Operations ===
    
    def _toggle_roi(self, checked: bool):
        """Toggle ROI selection mode."""
        if checked:
            x_range, y_range = self.source_plot.viewRange()
            roi_width = (x_range[1] - x_range[0]) * 0.3
            roi_height = (y_range[1] - y_range[0]) * 0.5
            roi_x = x_range[0] + (x_range[1] - x_range[0] - roi_width) * 0.5
            roi_y = y_range[0] + (y_range[1] - y_range[0] - roi_height) * 0.5
            
            self.roi_spikes = pg.RectROI(
                [roi_x, roi_y], [roi_width, roi_height],
                pen=pg.mkPen('r', width=2)
            )
            self.roi_spikes.addScaleHandle([1, 1], [0, 0])
            self.roi_spikes.addScaleHandle([0, 0], [1, 1])
            self.source_plot.addItem(self.roi_spikes)
        else:
            if self.roi_spikes:
                self.source_plot.removeItem(self.roi_spikes)
                self.roi_spikes = None
    
    def _add_spikes_roi(self):
        """Add spikes within ROI."""
        if self.roi_spikes is None:
            QMessageBox.warning(self, "Warning", "Enable ROI first")
            return
        
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx < 0:
            return
        
        self.editor.save_state(self.data.timestamps, self.data.sources)
        
        pos = self.roi_spikes.pos()
        size = self.roi_spikes.size()
        roi_bounds = (pos.x(), pos.x() + size.x(), pos.y(), pos.y() + size.y())
        
        source, timestamps = self.data.get_unit_data(unit_idx)
        self.data.timestamps[unit_idx] = self.editor.add_spikes_in_roi(
            source, timestamps, roi_bounds, self.data.fsamp
        )
        
        self._update_plots(preserve_view=True)
        self.roi_checkbox.setChecked(False)
    
    def _delete_spikes_roi(self):
        """Delete spikes within ROI."""
        if self.roi_spikes is None:
            QMessageBox.warning(self, "Warning", "Enable ROI first")
            return
        
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx < 0:
            return
        
        self.editor.save_state(self.data.timestamps, self.data.sources)
        
        pos = self.roi_spikes.pos()
        size = self.roi_spikes.size()
        roi_bounds = (pos.x(), pos.x() + size.x(), pos.y(), pos.y() + size.y())
        
        source, timestamps = self.data.get_unit_data(unit_idx)
        self.data.timestamps[unit_idx] = self.editor.delete_spikes_in_roi(
            source, timestamps, roi_bounds, self.data.fsamp
        )
        
        self._update_plots(preserve_view=True)
        self.roi_checkbox.setChecked(False)
    
    # === Unit Management ===
    
    def _flag_unit(self):
        """Toggle flag on current unit."""
        unit_idx = self.unit_selector.currentIndex()
        if unit_idx < 0:
            return
        
        if unit_idx in self.flagged_units:
            self.flagged_units.remove(unit_idx)
        else:
            self.flagged_units.append(unit_idx)
        
        self._update_plots(preserve_view=True)
        self._update_unit_selector_colors()
    
    def _update_unit_selector_colors(self):
        """Update dropdown colors for flagged units."""
        for idx in range(self.unit_selector.count()):
            color = QColor(*PlotColors.FLAGGED_UNIT) if idx in self.flagged_units else QColor(*PlotColors.NORMAL_UNIT)
            self.unit_selector.setItemData(idx, color, Qt.ForegroundRole)
    
    def _remove_duplicates(self):
        """Detect and flag duplicates."""
        if self.data.n_units < 2:
            return
        
        groups, units_to_flag = self.detector.find_duplicates(self.data.timestamps)
        
        for idx in units_to_flag:
            if idx not in self.flagged_units:
                self.flagged_units.append(idx)
        
        if units_to_flag:
            QMessageBox.information(
                self, "Duplicates Found",
                f"Found {len(groups)} duplicate groups.\n"
                f"Flagged {len(units_to_flag)} units for deletion."
            )
            self._update_plots(preserve_view=True)
            self._update_unit_selector_colors()
        else:
            QMessageBox.information(self, "No Issues", "No duplicates found")
    
    def _recalculate_filters(self):
        """Recalculate filter for current unit."""
        if self.data.whitened_emg is None:
            QMessageBox.warning(self, "Error", "EMG data not prepared for recalculation")
            return
        
        unit_idx = self.unit_selector.currentIndex()
        timestamps = self.data.timestamps[unit_idx]
        
        if len(timestamps) < 5:
            QMessageBox.warning(self, "Error", "Not enough spikes for recalculation")
            return
        
        try:
            self.editor.save_state(self.data.timestamps, self.data.sources)
            
            new_source, new_timestamps = self.editor.recalculate_filter(
                self.data.whitened_emg,
                timestamps,
                self.data.fsamp
            )
            
            self.data.sources[unit_idx] = new_source
            self.data.timestamps[unit_idx] = new_timestamps
            
            self._update_plots(preserve_view=True)
            
            QMessageBox.information(
                self, "Success",
                f"Recalculated Unit {unit_idx + 1}\n"
                f"New spike count: {len(new_timestamps)}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Recalculation failed:\n{str(e)}")
    
    def _delete_flagged_units(self):
        """Delete all flagged units."""
        if not self.flagged_units:
            QMessageBox.information(self, "No Units", "No units flagged")
            return
        
        for idx in sorted(self.flagged_units, reverse=True):
            del self.data.timestamps[idx]
            if self.data.sources is not None:
                self.data.sources = np.delete(self.data.sources, idx, axis=0)
        
        self.unit_selector.clear()
        for i in range(self.data.n_units):
            self.unit_selector.addItem(f"Unit {i + 1}")
        
        self.flagged_units = []
        self._update_plots()
        
        QMessageBox.information(self, "Success", "Flagged units deleted")
    
    # === Save/Export ===
    
    def _save_data(self):
        """Save edited data."""
        default_name = ""
        if self.output_folder:
            default_name = str(self.output_folder / f"{self.data.filename}_edited.pkl")
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Edited Data", default_name, "Pickle Files (*.pkl)"
        )
        
        if filename:
            self.data.save(Path(filename))
            QMessageBox.information(self, "Success", "Data saved")
    
    def _convert_to_emglab(self):
        """Convert to EMGlab format."""
        if self.data.n_units == 0:
            QMessageBox.warning(self, "No Data", "No data loaded")
            return
        
        default_name = ""
        if self.output_folder:
            default_name = str(self.output_folder / f"{self.data.filename}_edited.eaf")
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save EMGlab File", default_name, "EMGlab Files (*.eaf)"
        )
        
        if filename:
            try:
                export_to_emglab(
                    Path(filename),
                    self.data.timestamps,
                    self.data.emg_data_filtered,
                    self.data.fsamp
                )
                QMessageBox.information(self, "Success", f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")