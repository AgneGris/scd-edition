"""
Decomposition Tab - Manages EMG signal decomposition.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, 
    QCheckBox, QStackedWidget, QSizePolicy, QFrame,
     QMessageBox, QSplitter, QSpinBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEventLoop

# Visualization
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import TextBox, Button
import matplotlib.pyplot as plt

from gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, get_button_style
)

from core.config import SessionConfig
import torch
from scd_app.core.decomp_worker import DecompositionWorker

class DecompositionTab(QWidget):
    """Decomposition tab for EMG signal decomposition."""
    
    # Signal emits the decomp file path so Edition tab can load it
    decomposition_complete = pyqtSignal(Path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.config: Optional[SessionConfig] = None
        self.emg_path: Optional[Path] = None
        self.grid_configs: Dict = {}
        self.worker = None
        
        # EMG data
        self.emg_data = None
        self.sampling_rate = 2048
        self.rejected_channels = []
        self.selected_points = []
        self.plateau_coords = None
        
        # UI References
        self.grid_selector = None
        self.param_stack = None
        self.param_widgets = {}
        
        # Matplotlib cleanup
        self.cid = None
        self.start_time_box = None
        self.end_time_box = None
        self.clear_button = None
        self.confirm_button = None
        
        self.init_ui()

    def setup_session(self, config: SessionConfig, emg_paths: list):
        """Called by MainWindow when Configuration is applied."""
        self.config = config
        self.emg_paths = [Path(p) if not isinstance(p, Path) else p for p in emg_paths]
        self.emg_path = self.emg_paths[0]
        self.sampling_rate = config.sampling_frequency
        
        n_files = len(self.emg_paths)
        self.file_path_label.setText(
            f"\U0001f4c4 {self.emg_path.name}" if n_files == 1 
            else f"\U0001f4c4 {n_files} files (first: {self.emg_path.name})"
        )
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px; font-weight: bold;"
        )
        
        self._load_emg_data()
        self._load_grid_configs()
        print(f"Decomposition Tab Ready: {len(self.grid_configs)} grids, {n_files} file(s).")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        
        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 20)
        layout.setSpacing(4)
        
        # === Global parameters (all grids) ===
        layout.addWidget(QLabel("GLOBAL PARAMETERS", styleSheet=get_section_header_style('info')))
        
        global_grid = QGridLayout()
        global_grid.setSpacing(2)
        self.global_widgets = {}
        row = 0
        
        def add_global(label, widget, key):
            nonlocal row
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['info_light']};")
            global_grid.addWidget(lbl, row, 0)
            global_grid.addWidget(widget, row, 1)
            self.global_widgets[key] = widget
            row += 1
        
        add_global("SIL Threshold:", QLineEdit("0.85"), "sil_threshold")
        add_global("Iterations:", QLineEdit("20"), "iterations")
        
        clamp = QComboBox()
        clamp.addItems(["True", "False"])
        add_global("Clamping:", clamp, "clamp")
        
        fitness = QComboBox()
        fitness.addItems(["CoV", "SIL"])
        add_global("Fitness:", fitness, "fitness")
        
        peel = QComboBox()
        peel.addItems(["True", "False"])
        add_global("Peel Off:", peel, "peel_off")
        
        swarm = QComboBox()
        swarm.addItems(["True", "False"])
        add_global("Swarm:", swarm, "swarm")
        
        exp_spin = QSpinBox()
        exp_spin.setRange(2, 7)
        exp_spin.setValue(3)
        add_global("Fixed Exponent:", exp_spin, "fixed_exponent")
        
        layout.addLayout(global_grid)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep)
    
        # === Batch options ===
        batch_label = QLabel("BATCH OPTIONS", styleSheet=get_section_header_style('info'))
        layout.addWidget(batch_label)
        
        batch_grid = QGridLayout()
        batch_grid.setSpacing(2)
        
        lbl1 = QLabel("Channel Rejection:")
        lbl1.setStyleSheet(f"color: {COLORS['info_light']};")
        self.rejection_mode = QComboBox()
        self.rejection_mode.addItems(["Per file", "First file only"])
        batch_grid.addWidget(lbl1, 0, 0)
        batch_grid.addWidget(self.rejection_mode, 0, 1)
        
        lbl2 = QLabel("Time Window:")
        lbl2.setStyleSheet(f"color: {COLORS['info_light']};")
        self.time_mode = QComboBox()
        self.time_mode.addItems(["Manual selection", "Full file"])
        batch_grid.addWidget(lbl2, 1, 0)
        batch_grid.addWidget(self.time_mode, 1, 1)
        
        layout.addLayout(batch_grid)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep2)

        # === Per-grid parameters ===
        layout.addWidget(QLabel("PER-GRID PARAMETERS", styleSheet=get_section_header_style('info')))
        
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Grid:"))
        self.grid_selector = QComboBox()
        self.grid_selector.currentIndexChanged.connect(self._on_grid_changed)
        sel_layout.addWidget(self.grid_selector)
        layout.addLayout(sel_layout)
        
        self.param_stack = QStackedWidget()
        layout.addWidget(self.param_stack)
        
        layout.addStretch()

        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Decomposition")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_decomposition)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {COLORS['success']}, stop:1 #2F855A);
                color: white; border-radius: 6px; font-weight: bold; 
                font-size: {FONT_SIZES['medium']};
            }}
            QPushButton:hover {{ background-color: #48BB78; }}
            QPushButton:pressed {{ background-color: #276749; }}
            QPushButton:disabled {{ 
                background-color: {COLORS['background_input']}; 
                color: {COLORS['text_muted']}; 
            }}
        """)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_decomposition)
        self.stop_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 6px; font-weight: bold;"
        )
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        return container
    
    def _create_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(
            f"background-color: {COLORS['background']}; "
            f"border-left: 2px solid {COLORS['border']};"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setAlignment(Qt.AlignLeft)
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px;"
        )
        
        self.grid_indicator_label = QLabel("")
        self.grid_indicator_label.setStyleSheet(
            f"color: {COLORS['info']}; font-size: 10pt; font-weight: bold; "
            f"padding: 5px; margin: 5px;"
        )
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.file_path_label)
        header_layout.addWidget(self.grid_indicator_label)
        header_layout.addStretch()
        layout.addLayout(header_layout, stretch=0)

        self.figure = Figure(facecolor=COLORS['background'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['background'])
        self.ax.text(
            0.5, 0.5, "Waiting for configuration...", 
            color=COLORS['text_muted'], 
            ha='center', va='center', transform=self.ax.transAxes
        )
        self.ax.axis('off')
        
        layout.addWidget(self.canvas, stretch=1)
        return container

    def _load_emg_data(self):
        """Load EMG data using the layout from config."""
        import copy
        from scd_app.io.data_loader import load_field

        try:
            layout = getattr(self.config, 'data_layout', None)
            if layout is None:
                raise ValueError("No data layout configured -- select a Data Format in Configuration tab")

            # Strip layout-level channel slice so all channels are loaded.
            # Per-grid channel selection is done downstream via port.electrode.channels
            # (absolute indices into the full array).
            layout_full = copy.deepcopy(layout)
            layout_full["fields"]["emg"].pop("channels", None)

            emg = load_field(self.emg_path, layout_full, "emg")
            # load_field returns (samples, channels) as torch.Tensor
            self.emg_data = emg
            print(f"Loaded EMG data: {self.emg_data.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load EMG data:\n{str(e)}")
            print(f"Error loading EMG data: {e}")

    def _load_grid_configs(self):
        """Populate grid configurations based on SessionConfig."""
        self.grid_configs.clear()
        self.grid_selector.clear()
        
        while self.param_stack.count():
            w = self.param_stack.widget(0)
            self.param_stack.removeWidget(w)
            w.deleteLater()
            
        if not self.config:
            return

        for port in self.config.ports:
            if not port.enabled:
                continue
            
            n_channels = len(port.electrode.channels)
            extension_factor = int(np.ceil(1000 / n_channels))
            
            # Set filter defaults based on electrode type
            is_surface = port.electrode.type == "surface"
            lowpass = 500 if is_surface else 4400
            highpass = 10
            
            defaults = {
                "extension_factor": extension_factor,
                "lowpass_hz": lowpass,
                "highpass_hz": highpass,
                "notch_filter": "None",
                "notch_harmonics": False,
            }
            
            electrode_type_label = "Surface" if is_surface else "Intramuscular"
            
            self.grid_configs[port.name] = {
                "params": defaults,
                "channels": port.electrode.channels,
                "num_channels": n_channels,
                "electrode_type": port.electrode.name,
                "electrode_class": port.electrode.type,
            }
            
            self.grid_selector.addItem(
                f"{port.name} ({port.electrode.name})", 
                port.name
            )
            
            page = self._create_param_page(port.name, defaults, electrode_type_label)
            self.param_stack.addWidget(page)

    def _create_param_page(self, port_name: str, defaults: dict, electrode_type: str = "") -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(10)
        
        widgets = {}
        row = 0
        
        def add_row(label, widget):
            nonlocal row
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['info_light']};")
            layout.addWidget(lbl, row, 0)
            layout.addWidget(widget, row, 1)
            row += 1
            return widget
        
        # Type indicator (read-only)
        type_label = QLabel(electrode_type if electrode_type else "Unknown")
        type_label.setStyleSheet(f"color: {COLORS['foreground']}; font-weight: bold;")
        add_row("Electrode Type:", type_label)
        
        widgets['extension_factor'] = add_row("Extension Factor:", QLineEdit(str(defaults['extension_factor'])))
        widgets['highpass_hz'] = add_row("High-pass (Hz):", QLineEdit(str(defaults['highpass_hz'])))
        widgets['lowpass_hz'] = add_row("Low-pass (Hz):", QLineEdit(str(defaults['lowpass_hz'])))
        
        notch = QComboBox()
        notch.addItems(["None", "50", "60"])
        notch.setCurrentText(defaults['notch_filter'])
        widgets['notch_filter'] = add_row("Notch Filter:", notch)
        
        harmonics_cb = QCheckBox("Include Harmonics")
        harmonics_cb.setChecked(defaults['notch_harmonics'])
        widgets['notch_harmonics'] = add_row("", harmonics_cb)

        self.param_widgets[port_name] = widgets
        return page
    
    def _on_grid_changed(self, index: int):
        if index >= 0:
            self.param_stack.setCurrentIndex(index)

    def _sync_params_from_ui(self):
        """Update grid_configs with values from global + per-grid UI."""
        # Read global params once
        global_params = {
            "sil_threshold": float(self.global_widgets['sil_threshold'].text()),
            "iterations": int(self.global_widgets['iterations'].text()),
            "clamp": self.global_widgets['clamp'].currentText() == "True",
            "fitness": self.global_widgets['fitness'].currentText(),
            "peel_off": self.global_widgets['peel_off'].currentText() == "True",
            "swarm": self.global_widgets['swarm'].currentText() == "True",
            "fixed_exponent": self.global_widgets['fixed_exponent'].value(),
        }
        
        # Merge into each grid
        for port_name, widgets in self.param_widgets.items():
            try:
                params = self.grid_configs[port_name]["params"]
                
                # Global
                params.update(global_params)
                
                # Per-grid
                params["extension_factor"] = int(widgets['extension_factor'].text())
                params["highpass_hz"] = float(widgets['highpass_hz'].text())
                params["lowpass_hz"] = float(widgets['lowpass_hz'].text())
                params["notch_filter"] = widgets['notch_filter'].currentText()
                params["notch_harmonics"] = widgets['notch_harmonics'].isChecked()
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not sync parameter for {port_name}: {e}")

    def _manual_channel_rejection(self):
        """Show EMG channels per grid and let user select which to remove."""
        if self.emg_data is None:
            return
        
        self._cleanup_matplotlib_widgets()
        self.figure.set_facecolor(COLORS['background'])
        
        # Build grid list
        grid_list = list(self.grid_configs.items())
        if not grid_list:
            return
        
        # Init rejection masks
        self.rejected_channels = []
        for port_idx, (port_name, config) in enumerate(grid_list):
            n_channels = len(config['channels'])
            if port_idx < len(self.rejected_channels):
                pass  # keep existing
            else:
                self.rejected_channels.append(np.zeros(n_channels, dtype=int))
        
        import time
        nav = {'current': 0}

        def draw_grid(grid_idx):
            """Draw a single grid's channels."""
            self.figure.clf()
            
            port_name, config = grid_list[grid_idx]
            channels = config['channels']
            n_channels = len(channels)
            mask = self.rejected_channels[grid_idx]
            
            ax = self.figure.add_axes([0.03, 0.12, 0.94, 0.82])
            ax.set_facecolor(COLORS['background'])
            self.figure.set_facecolor(COLORS['background'])
            
            # Title
            self.figure.text(
                0.5, 0.96,
                f"{port_name}  ({grid_idx + 1}/{len(grid_list)})",
                ha="center", va="center",
                fontsize=13, weight='bold', color=COLORS['foreground'],
            )
            
            # Instructions
            self.figure.text(
                0.5, 0.08,
                "Click = Toggle channel | Scroll = Zoom | Right-drag = Pan | R = Reset",
                ha="center", va="center",
                fontsize=10, weight='bold', color=COLORS['info'],
            )

            grid_data = self.emg_data[:, channels].numpy()
            std_dev = np.std(grid_data)
            separation = std_dev * 15 if std_dev > 0 else 1.0
            
            step = max(1, grid_data.shape[0] // 4000)
            disp_data = grid_data[::step, :]
            max_len = disp_data.shape[0]
            
            line_info = {}
            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                color = COLORS['error'] if is_rejected else COLORS['info']
                alpha = 0.5 if is_rejected else 0.8
                y_pos = ch * separation
                
                line, = ax.plot(
                    disp_data[:, ch] + y_pos,
                    color=color, alpha=alpha, linewidth=1.0,
                )
                line.set_pickradius(8)
                line_info[line] = ch
            
            # Channel labels on the left
            for ch in range(n_channels):
                ax.text(
                    -max_len * 0.01, ch * separation,
                    f"{ch}", color=COLORS['text_muted'],
                    fontsize=7, ha='right', va='center',
                )
            
            total_height = n_channels * separation
            ax.set_xlim(0, max_len)
            ax.set_ylim(-separation, total_height)
            ax.axis('off')
            ax.margins(0)

            # --- Navigation buttons ---
            prev_ax = self.figure.add_axes([0.05, 0.01, 0.12, 0.05])
            next_ax = self.figure.add_axes([0.83, 0.01, 0.12, 0.05])
            confirm_ax = self.figure.add_axes([0.4, 0.01, 0.2, 0.05])
            
            is_last = (grid_idx == len(grid_list) - 1)
            is_first = (grid_idx == 0)

            prev_btn = Button(prev_ax, "<- Previous",
                              color=COLORS['background_light'], hovercolor=COLORS['background_hover'])
            prev_btn.label.set_color(COLORS['foreground'] if not is_first else COLORS['text_muted'])
            prev_btn.label.set_fontsize(10)
            
            next_btn = Button(next_ax, "Next ->",
                              color=COLORS['background_light'], hovercolor=COLORS['background_hover'])
            next_btn.label.set_color(COLORS['foreground'] if not is_last else COLORS['text_muted'])
            next_btn.label.set_fontsize(10)

            confirm_btn = Button(confirm_ax, "CONFIRM ALL",
                                 color=COLORS['success'], hovercolor='#2EA043')
            confirm_btn.label.set_color("white")
            confirm_btn.label.set_weight('bold')
            confirm_btn.label.set_fontsize(10)

            # Rejected count for this grid
            n_rej = np.sum(mask)
            rej_text = self.figure.text(
                0.5, 0.04,
                f"{n_rej} channel{'s' if n_rej != 1 else ''} rejected" if n_rej > 0 else "",
                ha="center", va="center",
                fontsize=9, color=COLORS['error'] if n_rej > 0 else COLORS['text_muted'],
            )

            # --- Interaction state ---
            state = {
                'press_event': None,
                'last_scroll_time': 0,
                'panning': False,
                'pan_start': None,
                'pan_xlim': None,
                'pan_ylim': None,
            }
            CLICK_TOLERANCE_PX = 5
            SCROLL_GUARD_MS = 300

            def on_press(event):
                if event.inaxes != ax:
                    return
                if event.button == 1:
                    state['press_event'] = event
                elif event.button == 3:
                    state['panning'] = True
                    state['pan_start'] = (event.x, event.y)
                    state['pan_xlim'] = ax.get_xlim()
                    state['pan_ylim'] = ax.get_ylim()

            def on_motion(event):
                if not state['panning'] or event.inaxes != ax:
                    return
                dx = event.x - state['pan_start'][0]
                dy = event.y - state['pan_start'][1]
                xlim = state['pan_xlim']
                ylim = state['pan_ylim']
                bbox = ax.get_window_extent()
                data_dx = -(dx / bbox.width) * (xlim[1] - xlim[0])
                data_dy = -(dy / bbox.height) * (ylim[1] - ylim[0])
                ax.set_xlim(xlim[0] + data_dx, xlim[1] + data_dx)
                ax.set_ylim(ylim[0] + data_dy, ylim[1] + data_dy)
                self.canvas.draw_idle()

            def on_release(event):
                if event.button == 3:
                    state['panning'] = False
                    return
                if event.button != 1 or state['press_event'] is None:
                    return
                press = state['press_event']
                state['press_event'] = None
                
                elapsed_ms = (time.time() - state['last_scroll_time']) * 1000
                if elapsed_ms < SCROLL_GUARD_MS:
                    return
                if abs(event.x - press.x) > CLICK_TOLERANCE_PX or \
                   abs(event.y - press.y) > CLICK_TOLERANCE_PX:
                    return
                if event.inaxes != ax:
                    return
                
                closest = None
                min_dist = float('inf')
                for line, c_idx in line_info.items():
                    contains, info = line.contains(event)
                    if contains:
                        inds = info.get('ind', [])
                        if inds.any():
                            ydata = line.get_ydata()
                            d = min(abs(ydata[i] - event.ydata) for i in inds)
                            if d < min_dist:
                                min_dist = d
                                closest = (line, c_idx)
                
                if closest is None:
                    return
                
                line, c_idx = closest
                new_state = 1 - mask[c_idx]
                mask[c_idx] = new_state
                
                line.set_color(COLORS['error'] if new_state == 1 else COLORS['info'])
                line.set_alpha(0.5 if new_state == 1 else 0.8)
                
                n_rej = np.sum(mask)
                rej_text.set_text(
                    f"{n_rej} channel{'s' if n_rej != 1 else ''} rejected" if n_rej > 0 else ""
                )
                rej_text.set_color(COLORS['error'] if n_rej > 0 else COLORS['text_muted'])
                self.canvas.draw_idle()

            def on_scroll(event):
                if event.inaxes != ax:
                    return
                state['last_scroll_time'] = time.time()
                scale = 0.85 if event.button == 'up' else 1.18
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xdata, ydata = event.xdata, event.ydata
                ax.set_xlim(xdata - (xdata - xlim[0]) * scale,
                            xdata + (xlim[1] - xdata) * scale)
                ax.set_ylim(ydata - (ydata - ylim[0]) * scale,
                            ydata + (ylim[1] - ydata) * scale)
                self.canvas.draw_idle()

            def on_key(event):
                if event.key in ('r', 'R'):
                    ax.set_xlim(0, max_len)
                    ax.set_ylim(-separation, total_height)
                    self.canvas.draw_idle()

            def go_prev(event):
                if not is_first:
                    disconnect()
                    nav['current'] -= 1
                    draw_grid(nav['current'])

            def go_next(event):
                if not is_last:
                    disconnect()
                    nav['current'] += 1
                    draw_grid(nav['current'])

            def on_confirm(event):
                disconnect()
                self.figure.clf()
                self.figure.set_facecolor(COLORS['background'])
                
                total_rej = sum(np.sum(m) for m in self.rejected_channels)
                self.figure.text(
                    0.5, 0.5,
                    f"Rejected {total_rej} channels across {len(grid_list)} grids",
                    ha="center", va="center",
                    fontsize=14, weight='bold', color=COLORS['success'],
                )
                self.canvas.draw()
                
                print("\nChannel Rejection Summary:")
                for pidx, (pname, _) in enumerate(grid_list):
                    n = np.sum(self.rejected_channels[pidx])
                    print(f"  {pname}: {n} channels rejected")
                
                QTimer.singleShot(100, event_loop.quit)

            # Connect
            cids = [
                self.canvas.mpl_connect('button_press_event', on_press),
                self.canvas.mpl_connect('button_release_event', on_release),
                self.canvas.mpl_connect('motion_notify_event', on_motion),
                self.canvas.mpl_connect('scroll_event', on_scroll),
                self.canvas.mpl_connect('key_press_event', on_key),
            ]
            prev_btn.on_clicked(go_prev)
            next_btn.on_clicked(go_next)
            confirm_btn.on_clicked(on_confirm)
            
            # Store for cleanup
            nav['cids'] = cids
            nav['buttons'] = [prev_btn, next_btn, confirm_btn]
            
            self.canvas.draw()

        def disconnect():
            # Disconnect canvas events
            for cid in nav.get('cids', []):
                self.canvas.mpl_disconnect(cid)
                
            # Disconnect and remove all navigation buttons
            for btn in nav.get('buttons', []):
                btn.disconnect_events()
                try:
                    btn.ax.remove()
                except Exception:
                    pass  # Ignore if already removed
            nav['buttons'] = []

        event_loop = QEventLoop()
        draw_grid(0)
        event_loop.exec_()

    def _select_time_window(self):
        """Let user select time window for decomposition."""
        if self.emg_data is None:
            return
        
        self._cleanup_matplotlib_widgets()
        self.figure.clf()
        
        total_duration = self.emg_data.shape[0] / self.sampling_rate
        time_axis = np.arange(self.emg_data.shape[0]) / self.sampling_rate
        
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        
        instruction_text = self.figure.text(
            0.5, 0.9,
            "Click plot to restart selection OR type in boxes (Press ENTER to apply)",
            ha="center", va="center",
            fontsize=12, weight='bold', color=COLORS['info'],
        )
        
        colors = ['#4a9eff', '#a78bfa', '#48BB78', '#F6AD55', '#ff6b9d']
        for idx, (port_name, config) in enumerate(self.grid_configs.items()):
            channels = config['channels']
            channel_data = self.emg_data[:, channels].numpy()
            
            rejected = self.rejected_channels[idx]
            active_channels = np.where(rejected == 0)[0]
            channel_data = channel_data[:, active_channels]
            
            rms = np.sqrt(np.mean(channel_data ** 2, axis=1))
            window = int(self.sampling_rate * 0.1)
            kernel = np.ones(window) / window
            rms_smooth = np.convolve(rms, kernel, mode='same')
            
            color = colors[idx % len(colors)]
            ax.plot(time_axis, rms_smooth, color=color, label=port_name, 
                   linewidth=2.5, alpha=0.85)
        
        ax.set_xlabel("Time (s)", color=COLORS['foreground'], fontsize=12, weight='bold')
        ax.set_ylabel("RMS Amplitude", color=COLORS['foreground'], fontsize=12, weight='bold')
        ax.tick_params(colors=COLORS['foreground'])
        ax.set_xlim(0, total_duration)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['background_light'])
        ax.spines['bottom'].set_color(COLORS['background_light'])
        
        start_box_ax = self.figure.add_axes([0.12, 0.955, 0.18, 0.025])
        end_box_ax = self.figure.add_axes([0.38, 0.955, 0.18, 0.025])
        clear_btn_ax = self.figure.add_axes([0.63, 0.95, 0.12, 0.035])
        confirm_btn_ax = self.figure.add_axes([0.77, 0.95, 0.14, 0.035])
        
        start_box = TextBox(start_box_ax, "Start:", initial="0.00", 
                           color=COLORS['background_light'], hovercolor=COLORS['background_light'])
        end_box = TextBox(end_box_ax, "End:", initial=f"{total_duration:.2f}", 
                         color=COLORS['background_light'], hovercolor=COLORS['background_light'])
        clear_btn = Button(clear_btn_ax, "RESET", color=COLORS['background_light'], hovercolor=COLORS['text_muted'])
        confirm_btn = Button(confirm_btn_ax, "CONFIRM", color=COLORS['success'], hovercolor="#2EA043")
        
        for w in [start_box, end_box]:
            w.label.set_color(COLORS['foreground'])
            w.text_disp.set_color(COLORS['foreground'])
        confirm_btn.label.set_color("white")

        self.sel_start = 0.0
        self.sel_end = total_duration
        self.selection_state = "complete"

        self.line_start = ax.axvline(x=self.sel_start, color=COLORS['success'], linestyle="--", linewidth=2)
        self.line_end = ax.axvline(x=self.sel_end, color=COLORS['error'], linestyle="--", linewidth=2)
        
        event_loop = QEventLoop()

        def update_visuals():
            if self.sel_start is not None:
                self.line_start.set_xdata([self.sel_start, self.sel_start])
                self.line_start.set_visible(True)
                try:
                    if abs(float(start_box.text) - self.sel_start) > 0.01:
                        start_box.set_val(f"{self.sel_start:.2f}")
                except:
                    start_box.set_val(f"{self.sel_start:.2f}")
            else:
                self.line_start.set_visible(False)
                start_box.set_val("")

            if self.sel_end is not None:
                self.line_end.set_xdata([self.sel_end, self.sel_end])
                self.line_end.set_visible(True)
                try:
                    if abs(float(end_box.text) - self.sel_end) > 0.01:
                        end_box.set_val(f"{self.sel_end:.2f}")
                except:
                    end_box.set_val(f"{self.sel_end:.2f}")
            else:
                self.line_end.set_visible(False)
                end_box.set_val("")
            
            if self.selection_state == "start_set":
                instruction_text.set_text(f"Start: {self.sel_start:.2f}s set. Click for End point.")
                instruction_text.set_color(COLORS['warning'])
            elif self.selection_state == "complete":
                dur = abs(self.sel_end - self.sel_start)
                instruction_text.set_text(f"Selected: {min(self.sel_start, self.sel_end):.2f}s - {max(self.sel_start, self.sel_end):.2f}s (Dur: {dur:.2f}s)")
                instruction_text.set_color(COLORS['success'])
            
            self.canvas.draw()

        def on_click(event):
            if event.inaxes != ax: return
            val = event.xdata
            
            if self.selection_state == "complete":
                self.sel_start = val
                self.sel_end = None
                self.selection_state = "start_set"
            elif self.selection_state == "start_set":
                self.sel_end = val
                if self.sel_end < self.sel_start:
                    self.sel_start, self.sel_end = self.sel_end, self.sel_start
                self.selection_state = "complete"
            
            update_visuals()

        def on_start_submit(text):
            try:
                val = float(text)
                self.sel_start = val
                if self.selection_state == "complete" and self.sel_end is not None:
                    if self.sel_start > self.sel_end:
                         self.sel_start, self.sel_end = self.sel_end, self.sel_start
                if self.sel_start is not None and self.sel_end is None:
                     self.selection_state = "start_set"
                update_visuals()
            except ValueError:
                pass

        def on_end_submit(text):
            try:
                val = float(text)
                self.sel_end = val
                if self.sel_start is None:
                    self.sel_start = 0.0
                self.selection_state = "complete"
                if self.sel_start > self.sel_end:
                     self.sel_start, self.sel_end = self.sel_end, self.sel_start
                update_visuals()
            except ValueError:
                pass

        def on_reset(event):
            self.sel_start = 0.0
            self.sel_end = total_duration
            self.selection_state = "complete"
            update_visuals()

        def on_confirm(event):
            if self.sel_start is not None and self.sel_end is not None:
                confirm_btn.color = "#1F6E2C"
                confirm_btn.label.set_text("Processing...")
                self.canvas.draw()
                self.canvas.flush_events()
                
                final_s = min(self.sel_start, self.sel_end)
                final_e = max(self.sel_start, self.sel_end)
                
                self.plateau_coords = np.array([
                    int(final_s * self.sampling_rate),
                    int(final_e * self.sampling_rate)
                ])
                QTimer.singleShot(100, event_loop.quit)
            else:
                instruction_text.set_text("Please select start and end points first")
                instruction_text.set_color(COLORS['error'])
                self.canvas.draw()

        def on_scroll(event):
            if event.inaxes != ax:
                return
            scale = 0.8 if event.button == 'up' else 1.25
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Zoom centered on cursor
            xdata, ydata = event.xdata, event.ydata
            ax.set_xlim([xdata - (xdata - xlim[0]) * scale,
                         xdata + (xlim[1] - xdata) * scale])
            ax.set_ylim([ydata - (ydata - ylim[0]) * scale,
                         ydata + (ylim[1] - ydata) * scale])
            self.canvas.draw_idle()

        self.cid = self.canvas.mpl_connect("button_press_event", on_click)
        self.scroll_cid = self.canvas.mpl_connect("scroll_event", on_scroll)
        start_box.on_submit(on_start_submit)
        end_box.on_submit(on_end_submit)
        clear_btn.on_clicked(on_reset)
        confirm_btn.on_clicked(on_confirm)

        self.start_time_box = start_box
        self.end_time_box = end_box
        self.clear_button = clear_btn
        self.confirm_button = confirm_btn
        
        update_visuals()
        self.canvas.draw()
        event_loop.exec_()
        
        print(f"Selected time window: {self.plateau_coords / self.sampling_rate} seconds")

    def _cleanup_matplotlib_widgets(self):
        """Clean up matplotlib widgets before creating new ones."""
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None
        
        if self.start_time_box is not None:
            self.start_time_box.disconnect_events()
            self.start_time_box = None
        
        if self.end_time_box is not None:
            self.end_time_box.disconnect_events()
            self.end_time_box = None
        
        if self.clear_button is not None:
            self.clear_button.disconnect_events()
            self.clear_button = None
        
        if self.confirm_button is not None:
            self.confirm_button.disconnect_events()
            self.confirm_button = None

        if hasattr(self, 'scroll_cid') and self.scroll_cid is not None:
            self.canvas.mpl_disconnect(self.scroll_cid)
            self.scroll_cid = None

    def _start_decomposition(self):
        if not self.config or not self.emg_paths or self.emg_data is None:
            QMessageBox.warning(self, "Error", "No session loaded.")
            return
        
        self._sync_params_from_ui()
        
        is_batch = len(self.emg_paths) > 1
        share_rejection = self.rejection_mode.currentText() == "First file only"
        use_full_file = self.time_mode.currentText() == "Full file"
        
        # --- Step 1: Channel rejection (always for first file) ---
        self._manual_channel_rejection()
        shared_mask = [m.copy() for m in self.rejected_channels]
        
        # --- Step 2: Time window (if manual) ---
        if not use_full_file:
            self._select_time_window()
            if self.plateau_coords is None:
                QMessageBox.warning(self, "Error", "No time window selected.")
                return
        
        # --- Step 3: Decompose all files ---
        output_dir = Path(self.config.output_dir) if self.config.output_dir else self.emg_path.parent
        
        self._cleanup_matplotlib_widgets()
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        
        self._file_queue = list(self.emg_paths)
        self._file_idx = 0
        self._output_dir = output_dir
        self._shared_mask = shared_mask
        self._share_rejection = share_rejection
        self._use_full_file = use_full_file
        
        self._decompose_next_file()
    
    def _decompose_next_file(self):
        """Process next file in the queue."""
        from scd_app.io.data_loader import load_field
        
        if self._file_idx >= len(self._file_queue):
            # All files done — now emit signal to switch to Edition tab
            self.grid_indicator_label.setText("All files complete")
            self._reset_ui_state()
            if hasattr(self, '_last_decomp_path') and self._last_decomp_path:
                self.decomposition_complete.emit(self._last_decomp_path)
            return
        
        file_path = self._file_queue[self._file_idx]
        n_total = len(self._file_queue)
        self.grid_indicator_label.setText(
            f"File {self._file_idx + 1}/{n_total}: {file_path.name}"
        )
        
        # Load this file's data
        try:
            layout = getattr(self.config, 'data_layout', None)
            emg = load_field(file_path, layout, "emg")
            self.emg_data = emg
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load {file_path.name}:\n{e}")
            self._file_idx += 1
            self._decompose_next_file()
            return
        
        # Channel rejection: per-file or shared
        if self._file_idx == 0 or self._share_rejection:
            self.rejected_channels = [m.copy() for m in self._shared_mask]
        else:
            self._manual_channel_rejection()
        
        # Time window: full file or manual
        if self._use_full_file:
            self.plateau_coords = np.array([0, self.emg_data.shape[0]])
        elif self._file_idx > 0:
            # For subsequent files in manual mode, ask again
            self._select_time_window()
            if self.plateau_coords is None:
                self._file_idx += 1
                self._decompose_next_file()
                return
        
        # Show decomposing state
        self.figure.clf()
        self.figure.set_facecolor(COLORS['background'])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        ax.text(0.5, 0.5, f"Decomposing {file_path.name}...",
                color=COLORS['warning'], fontsize=16, weight='bold',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()
        
        save_path = self._output_dir / f"{file_path.stem}_decomp_output.pkl"
        
        self.worker = DecompositionWorker(
            self.emg_data, self.grid_configs, self.rejected_channels,
            self.plateau_coords, self.sampling_rate, save_path
        )
        self.worker.progress.connect(self._update_grid_indicator)
        self.worker.finished.connect(self._on_file_decomposition_finished)
        self.worker.error.connect(self._on_decomposition_error)
        self.worker.source_found.connect(self._on_source_found)
        self.worker.start()
    
    def _on_file_decomposition_finished(self, results):
        """One file done, move to next."""
        decomp_path = Path(results.get('path'))
        self._last_decomp_path = decomp_path
        
        self._file_idx += 1
        self._decompose_next_file()

    def _update_grid_indicator(self, message: str):
        """Update grid indicator when progress says 'Processing ...'"""
        if message.startswith("Processing "):
            # Extract "Processing GridName (2/6)..."
            self.grid_indicator_label.setText(f"{message}")

    def _stop_decomposition(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self._reset_ui_state()

    def _on_decomposition_error(self, err_msg):
        QMessageBox.critical(self, "Error", f"Decomposition Failed:\n{err_msg}")
        self._reset_ui_state()

    def _on_source_found(self, source, timestamps, iteration, silhouette):
        self._plot_source_realtime(source, timestamps, iteration, silhouette)
        QApplication.processEvents()

    def _plot_source_realtime(self, source, timestamps, iteration, silhouette):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        
        source_np = source.detach().cpu().numpy() if torch.is_tensor(source) else source
        timestamps_np = timestamps.detach().cpu().numpy() if torch.is_tensor(timestamps) else timestamps
        
        if len(timestamps_np) > 0:
            idx = timestamps_np.astype(int)
            idx = idx[idx < len(source_np)] 
            y_values = source_np[idx]
        else:
            idx = []
            y_values = []
        
        ax.plot(source_np, color='#2b6cb0', linewidth=1.2, alpha=0.9) 
        
        if len(idx) > 0:
            ax.plot(idx, y_values, 'o', color='#ed8936', markersize=4, alpha=0.9)

        ax.set_title(
            f"Iteration {iteration} | Silhouette: {silhouette:.3f} | {len(timestamps_np)} spikes",
            color=COLORS['foreground'], fontsize=12, weight='bold'
        )
        ax.set_xlabel("Sample", color=COLORS['foreground'])
        ax.set_ylabel("Amplitude", color=COLORS['foreground'])
        ax.tick_params(colors=COLORS['foreground'])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['foreground'])
        ax.spines['bottom'].set_color(COLORS['foreground'])
        
        self.canvas.draw()

    def _reset_ui_state(self):
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)