"""
Decomposition Tab - Manages EMG signal decomposition.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import argparse
import pickle
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, 
    QCheckBox, QStackedWidget, QSizePolicy, QFrame,
    QScrollArea, QMessageBox, QSplitter, QSpinBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEventLoop

# Visualization
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import TextBox, Button
import matplotlib.pyplot as plt

# Styling
from gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, get_button_style
)

# Core & Backend
from core.config import SessionConfig
import torch
from scd.config.structures import Config
from scd.models.scd import SwarmContrastiveDecomposition

# =============================================================================
# WORKER THREAD
# =============================================================================

class DecompositionWorker(QThread):
    """Worker thread to run the SCD decomposition algorithm."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    electrode_completed = pyqtSignal(int, int)
    source_found = pyqtSignal(object, object, int, float)

    def __init__(self, emg_data: torch.Tensor, grid_configs: dict, 
                 rejected_channels: List[np.ndarray], plateau_coords: np.ndarray,
                 sampling_rate: int, save_path: Path):
        super().__init__()
        self.emg_data = emg_data
        self.grid_configs = grid_configs
        self.rejected_channels = rejected_channels
        self.plateau_coords = plateau_coords
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        self._is_running = True

    def run(self):
        try:
            self.progress.emit("Starting decomposition...")
            
            results = {
                "pulse_trains": [],
                "discharge_times": [],
                "mu_filters": [],
                "ports": []
            }
            
            total_mus = 0
            
            # Process each grid
            for grid_idx, (port_name, config) in enumerate(self.grid_configs.items()):
                if not self._is_running:
                    break
                
                self.progress.emit(f"Processing {port_name} ({grid_idx + 1}/{len(self.grid_configs)})...")
                
                # Extract data for this grid
                channels = config['channels']
                grid_data = self.emg_data[:, channels]  # (time, channels)
                
                # Remove rejected channels
                rejected = self.rejected_channels[grid_idx]
                active_channels = np.where(rejected == 0)[0]
                grid_data = grid_data[:, active_channels]
                
                # Slice to selected time window
                start_sample = int(self.plateau_coords[0])
                end_sample = int(self.plateau_coords[1])
                grid_data = grid_data[start_sample:end_sample, :]
                
                # Create SCD config from GUI parameters
                scd_config = self._create_scd_config(config['params'])
                
                # Run decomposition
                dictionary, timestamps = self._decompose_grid(grid_data, 
                                                              scd_config,
                                                              grid_idx=grid_idx)
                
                # Store results
                if dictionary and "filters" in dictionary:
                    results["pulse_trains"].append(dictionary["source"])
                    results["discharge_times"].append(timestamps)
                    results["mu_filters"].append(dictionary["filters"])
                    results["ports"].append(port_name)
                    
                    n_mus = len(timestamps) if isinstance(timestamps, list) else 1
                    total_mus += n_mus
                    
                    self.progress.emit(f"{port_name}: {n_mus} MUs found")
                else:
                    results["pulse_trains"].append(np.array([]))
                    results["discharge_times"].append([])
                    results["mu_filters"].append(np.array([]))
                    results["ports"].append(port_name)
                    
                    self.progress.emit(f"{port_name}: 0 MUs found")
                
                self.electrode_completed.emit(grid_idx + 1, len(self.grid_configs))
            
            # Save results
            self.progress.emit("Saving results...")
            self._save_results(results)
            
            results_summary = {
                "status": "success",
                "path": str(self.save_path),
                "n_units": total_mus
            }
            
            self.finished.emit(results_summary)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _create_notch_params(self, params: dict) -> Optional[Tuple[int, float, bool]]:
        """Create notch_params tuple from params dict."""
        notch_freq = self._parse_notch(params["notch_filter"])
        if notch_freq is None:
            return None
        return (notch_freq, 2.0, params["notch_harmonics"])  # (frequency, bandwidth, harmonics)

    def _create_scd_config(self, params: dict) -> Config:
        """Create SCD Config object from GUI parameters."""
        return Config(
            device="cuda" if torch.cuda.is_available() else "cpu",
            sampling_frequency=self.sampling_rate,
            start_time=0,
            end_time=-1,
            
            # Decomposition parameters
            acceptance_silhouette=params["sil_threshold"],
            max_iterations=params["iterations"],
            extension_factor=params["extension_factor"],
            
            # Filter parameters (corrected names)
            low_pass_cutoff=params["lowpass_hz"],
            high_pass_cutoff=params["highpass_hz"],
            notch_params=self._create_notch_params(params),  # Must be tuple (freq, bandwidth, harmonics)
            
            # Algorithm parameters
            clamp_percentile=params["clamp"],
            use_coeff_var_fitness=(params["fitness"] == "CoV"),
            
            # Additional parameters that exist in Config
            peel_off=params["peel_off"],
            peel_off_window_size_ms=20,
            peel_off_repeats=params.get("peel_off_repeats", True),

            swarm=params["swarm"],
            fixed_exponent=params["fixed_exponent"],

            bad_channels=None,
            remove_bad_fr=False)

    def _parse_notch(self, notch_str: str) -> Optional[float]:
        """Parse notch filter string to frequency."""
        if notch_str == "50Hz":
            return 50.0
        elif notch_str == "60Hz":
            return 60.0
        return None

    def _decompose_grid(self, grid_data: torch.Tensor, config: Config, grid_idx: int) -> Tuple:
        """Run SCD decomposition on a single grid."""
        # Ensure data is on correct device
        grid_data = grid_data.to(device=config.device, dtype=torch.float32)
        
        # Create callback function
        def on_source_found(source, timestamps, iteration, silhouette):
            """Callback when SCD finds a source - emit signal to GUI"""
            self.source_found.emit(source, timestamps, iteration, silhouette)
        
        # Run SCD with callback
        model = SwarmContrastiveDecomposition()
        timestamps, dictionary = model.run(
            grid_data, 
            config,
            source_callback=on_source_found 
        )
        
        return dictionary, timestamps

    def _save_results(self, results: dict):
        """Save decomposition results to file."""
        save_dict = {
            "pulse_trains": results["pulse_trains"],
            "discharge_times": results["discharge_times"],
            "mu_filters": results["mu_filters"],
            "ports": results["ports"],
            "sampling_rate": self.sampling_rate,
            "plateau_coords": self.plateau_coords.tolist(),
        }
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def stop(self):
        self._is_running = False


# =============================================================================
# MAIN TAB CLASS
# =============================================================================

class DecompositionTab(QWidget):
    """Decomposition tab for EMG signal decomposition."""
    
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

    def setup_session(self, config: SessionConfig, emg_path: Path):
        """Called by MainWindow when Configuration is applied."""
        self.config = config
        self.emg_path = emg_path
        self.sampling_rate = config.sampling_frequency
        
        # Update file path display
        self.file_path_label.setText(f"📄 {self.emg_path.name}")
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px; font-weight: bold;"
        )
        
        # Load EMG data
        self._load_emg_data()
        
        # Initialize Grids
        self._load_grid_configs()
        print(f"Decomposition Tab Ready: {len(self.grid_configs)} grids loaded.")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        
        # Panels
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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # --- GRID PARAMETERS ---
        layout.addWidget(QLabel("GRID PARAMETERS", styleSheet=get_section_header_style('info')))
        
        # Grid Selector
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Grid:"))
        self.grid_selector = QComboBox()
        self.grid_selector.currentIndexChanged.connect(self._on_grid_changed)
        sel_layout.addWidget(self.grid_selector)
        layout.addLayout(sel_layout)
        
        # Parameter Stack
        self.param_stack = QStackedWidget()
        layout.addWidget(self.param_stack)
        
        layout.addStretch()
        
        # --- DECOMPOSITION BUTTON ---
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10pt;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

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
        
        # File path label at top
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setAlignment(Qt.AlignLeft)
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px;"
        )
        layout.addWidget(self.file_path_label, stretch=0)
        
        # Canvas for visualization
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

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_emg_data(self):
        """Load EMG data from file."""
        import scipy.io as sio
        
        try:
            if self.emg_path.suffix.lower() == ".mat":
                mat = sio.loadmat(str(self.emg_path))
                
                # Find data array
                for key in ['emg', 'data', 'sig', 'signal']:
                    if key in mat:
                        data = mat[key]
                        break
                else:
                    for k, v in mat.items():
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            data = v
                            break
                    else:
                        raise ValueError(f"No suitable data array found in {self.emg_path}")
            
            elif self.emg_path.suffix.lower() == ".npy":
                data = np.load(str(self.emg_path))
            
            else:
                raise ValueError(f"Unsupported format: {self.emg_path.suffix}")
            
            # Convert to tensor and ensure (samples, channels) format
            self.emg_data = torch.from_numpy(data).to(dtype=torch.float32)
            
            if self.emg_data.shape[1] > self.emg_data.shape[0]:
                self.emg_data = self.emg_data.T
            
            print(f"✓ Loaded EMG data: {self.emg_data.shape}")
            
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
                
            # Determine extension factor based on number of channels
            n_channels = len(port.electrode.channels)
            extension_factor = int(np.ceil(1000 / n_channels))
            
            defaults = {
                "sil_threshold": 0.85,
                "iterations": 20,
                "extension_factor": extension_factor,
                "lowpass_hz": 4400,
                "highpass_hz": 10,
                "notch_filter": "None",
                "notch_harmonics": False,
                "clamp": True,
                "fitness": "CoV",
                "peel_off": True,
                "swarm": True,
                "fixed_exponent": 3
            }
            
            self.grid_configs[port.name] = {
                "params": defaults,
                "channels": port.electrode.channels,
                "num_channels": len(port.electrode.channels)
            }
            
            self.grid_selector.addItem(
                f"{port.name} ({port.electrode.name})", 
                port.name
            )
            
            page = self._create_param_page(port.name, defaults)
            self.param_stack.addWidget(page)

    def _create_param_page(self, port_name: str, defaults: dict) -> QWidget:
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

        widgets['sil_threshold'] = add_row("SIL Threshold:", QLineEdit(str(defaults['sil_threshold'])))
        widgets['iterations'] = add_row("Iterations:", QLineEdit(str(defaults['iterations'])))
        widgets['extension_factor'] = add_row("Extension Factor:", QLineEdit(str(defaults['extension_factor'])))
        widgets['highpass_hz'] = add_row("High-pass (Hz):", QLineEdit(str(defaults['highpass_hz'])))
        widgets['lowpass_hz'] = add_row("Low-pass (Hz):", QLineEdit(str(defaults['lowpass_hz'])))
        
        notch = QComboBox()
        notch.addItems(["None", "50Hz", "60Hz"])
        notch.setCurrentText(defaults['notch_filter'])
        widgets['notch_filter'] = add_row("Notch Filter:", notch)
        
        harmonics_cb = QCheckBox("Include Harmonics")
        harmonics_cb.setChecked(defaults['notch_harmonics'])
        widgets['notch_harmonics'] = add_row("", harmonics_cb)
        
        clamp = QComboBox()
        clamp.addItems(["True", "False"])
        clamp.setCurrentText(str(defaults['clamp']))
        widgets['clamp'] = add_row("Clamping:", clamp)
        
        fitness = QComboBox()
        fitness.addItems(["CoV", "SIL"])
        fitness.setCurrentText(defaults['fitness'])
        widgets['fitness'] = add_row("Fitness:", fitness)
        
        peel = QComboBox()
        peel.addItems(["True", "False"])
        peel.setCurrentText(str(defaults['peel_off']))
        widgets['peel_off'] = add_row("Peel Off:", peel)
        
        swarm = QComboBox()
        swarm.addItems(["True", "False"])
        swarm.setCurrentText(str(defaults['swarm']))
        widgets['swarm'] = add_row("Swarm:", swarm)
        
        exp_spin = QSpinBox()
        exp_spin.setRange(2, 7)
        exp_spin.setValue(defaults['fixed_exponent'])
        widgets['fixed_exponent'] = add_row("Fixed Exponent:", exp_spin)

        self.param_widgets[port_name] = widgets
        return page

    def _on_grid_changed(self, index: int):
        if index >= 0:
            self.param_stack.setCurrentIndex(index)

    def _sync_params_from_ui(self):
        """Update grid_configs with values from UI."""
        for port_name, widgets in self.param_widgets.items():
            try:
                self.grid_configs[port_name]["params"]["sil_threshold"] = float(widgets['sil_threshold'].text())
                self.grid_configs[port_name]["params"]["iterations"] = int(widgets['iterations'].text())
                self.grid_configs[port_name]["params"]["extension_factor"] = int(widgets['extension_factor'].text())
                self.grid_configs[port_name]["params"]["highpass_hz"] = float(widgets['highpass_hz'].text())
                self.grid_configs[port_name]["params"]["lowpass_hz"] = float(widgets['lowpass_hz'].text())
                self.grid_configs[port_name]["params"]["notch_filter"] = widgets['notch_filter'].currentText()
                self.grid_configs[port_name]["params"]["notch_harmonics"] = widgets['notch_harmonics'].isChecked()
                self.grid_configs[port_name]["params"]["clamp"] = widgets['clamp'].currentText() == "True"
                self.grid_configs[port_name]["params"]["fitness"] = widgets['fitness'].currentText()
                self.grid_configs[port_name]["params"]["peel_off"] = widgets['peel_off'].currentText() == "True"
                self.grid_configs[port_name]["params"]["swarm"] = widgets['swarm'].currentText() == "True"
                self.grid_configs[port_name]["params"]["fixed_exponent"] = widgets['fixed_exponent'].value()
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not sync parameter for {port_name}: {e}")

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _manual_channel_rejection(self):
        """Show EMG channels and let user select which to remove."""
        if self.emg_data is None:
            return
        
        # 1. Cleanup
        self._cleanup_matplotlib_widgets()
        self.figure.clf()
        
        # 2. Setup Figure (Full Panel)
        ax = self.figure.add_axes([0.01, 0.12, 0.98, 0.86])
        ax.set_facecolor(COLORS['background'])
        self.figure.set_facecolor(COLORS['background'])
        ax.axis('off')

        # 3. Instructions
        instruction_text = self.figure.text(
            0.5, 0.08,
            "Click traces to toggle: RED = Rejected | BLUE = Kept",
            ha="center", va="center",
            fontsize=11, weight='bold', color=COLORS['info'],
        )

        # 4. Plot Channels
        self.rejected_channels = [] 
        plot_lines = [] 
        
        current_vertical_offset = 0
        
        # Calculate separation
        # Increased to *15 to prevent overlap which causes clicking issues
        std_dev = torch.std(self.emg_data).item()
        separation = std_dev * 15 if std_dev > 0 else 1.0
        
        max_len = 0

        for port_idx, (port_name, config) in enumerate(self.grid_configs.items()):
            channels = config['channels']
            n_channels = len(channels)
            
            # Initialize mask
            if len(self.rejected_channels) <= port_idx:
                mask = np.zeros(n_channels, dtype=int)
                self.rejected_channels.append(mask)
            else:
                mask = self.rejected_channels[port_idx]

            # Extract data
            grid_data = self.emg_data[:, channels].numpy()
            step = max(1, grid_data.shape[0] // 4000)
            disp_data = grid_data[::step, :]
            max_len = max(max_len, disp_data.shape[0])
            
            # Plot
            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                color = COLORS['error'] if is_rejected else COLORS['info']
                alpha = 0.5 if is_rejected else 0.8
                
                # Logic: y_pos increases with channel index
                # This stacks them visually bottom-to-top
                y_pos = current_vertical_offset + (ch * separation)
                
                line, = ax.plot(
                    disp_data[:, ch] + y_pos,
                    color=color,
                    alpha=alpha,
                    linewidth=1.0,
                    # Reduced tolerance to prevent clicking the neighbor
                    picker=5.0 
                )
                
                # Strict metadata attachment
                line.port_idx = port_idx
                line.channel_idx = ch
                plot_lines.append(line)
            
            # Label
            grid_center_y = current_vertical_offset + (n_channels * separation / 2)
            ax.text(
                0, grid_center_y, 
                f" {port_name} ", 
                color=COLORS['text_muted'], 
                fontsize=9, weight='bold', ha='right', va='center'
            )

            current_vertical_offset += (n_channels * separation) + (separation * 2)

        ax.set_xlim(0, max_len)
        ax.set_ylim(-separation, current_vertical_offset)
        ax.margins(0)

        # 5. Confirm Button
        confirm_ax = self.figure.add_axes([0.4, 0.01, 0.2, 0.05])
        confirm_btn = Button(confirm_ax, "CONFIRM", 
                            color=COLORS['success'], hovercolor='#2EA043')
        confirm_btn.label.set_color("white")
        confirm_btn.label.set_weight('bold')
        confirm_btn.label.set_fontsize(10)

        # 6. Event Logic
        event_loop = QEventLoop()

        def on_pick(event):
            """Handle click"""
            line = event.artist
            
            # Double check we have the metadata
            if not hasattr(line, 'port_idx') or not hasattr(line, 'channel_idx'): 
                return
            
            p_idx = line.port_idx
            c_idx = line.channel_idx
            
            # Toggle Logic
            current_state = self.rejected_channels[p_idx][c_idx]
            new_state = 1 - current_state
            self.rejected_channels[p_idx][c_idx] = new_state
            
            # Update visual immediately
            if new_state == 1:
                line.set_color(COLORS['error'])
                line.set_alpha(0.5)
            else:
                line.set_color(COLORS['info'])
                line.set_alpha(0.8)
            
            # Force redraw to show change immediately
            self.canvas.draw()

        def on_confirm(event):
            confirm_btn.color = "#1F6E2C"
            confirm_btn.label.set_text("Done")
            self.canvas.draw()
            self.canvas.flush_events()
            
            total_rej = sum([np.sum(m) for m in self.rejected_channels])
            instruction_text.set_text(f"✓ Rejected {total_rej} channels")
            instruction_text.set_color(COLORS['success'])
            
            QTimer.singleShot(100, event_loop.quit)

        self.cid = self.canvas.mpl_connect("pick_event", on_pick)
        confirm_btn.on_clicked(on_confirm)
        self.confirm_button = confirm_btn

        self.canvas.draw()
        event_loop.exec_()
        
        print("\nChannel Rejection Summary:")
        for port_idx, (port_name, _) in enumerate(self.grid_configs.items()):
            n_rej = np.sum(self.rejected_channels[port_idx])
            print(f"  {port_name}: {n_rej} channels rejected")

    def _select_time_window(self):
        """Let user select time window for decomposition."""
        if self.emg_data is None:
            return
        
        # Clean up previous matplotlib widgets
        self._cleanup_matplotlib_widgets()
        
        self.figure.clf()
        
        # Calculate constants
        total_duration = self.emg_data.shape[0] / self.sampling_rate
        time_axis = np.arange(self.emg_data.shape[0]) / self.sampling_rate
        
        # Create plot
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        
        # Instructions
        instruction_text = self.figure.text(
            0.5, 0.9,
            "Click plot to restart selection OR type in boxes (Press ENTER to apply)",
            ha="center", va="center",
            fontsize=12, weight='bold', color=COLORS['info'],
        )
        
        # Plot RMS signals
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
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['background_light'])
        ax.spines['bottom'].set_color(COLORS['background_light'])
        
        # --- WIDGET SETUP ---
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
        
        # Styling
        for w in [start_box, end_box]:
            w.label.set_color(COLORS['foreground'])
            w.text_disp.set_color(COLORS['foreground'])
        confirm_btn.label.set_color("white")

        # --- STATE MANAGEMENT ---
        # We maintain explicitly two values. None means "not set".
        self.sel_start = 0.0
        self.sel_end = total_duration
        self.selection_state = "complete" # options: "start_set", "complete"

        # Create the two vertical lines immediately (initialized to start/end)
        self.line_start = ax.axvline(x=self.sel_start, color=COLORS['success'], linestyle="--", linewidth=2)
        self.line_end = ax.axvline(x=self.sel_end, color=COLORS['error'], linestyle="--", linewidth=2)
        
        event_loop = QEventLoop()

        def update_visuals():
            """Updates lines and text boxes based on internal state"""
            # Update Start Line
            if self.sel_start is not None:
                self.line_start.set_xdata([self.sel_start, self.sel_start])
                self.line_start.set_visible(True)
                # Only update text if it differs significantly to avoid typing interference
                try:
                    if abs(float(start_box.text) - self.sel_start) > 0.01:
                        start_box.set_val(f"{self.sel_start:.2f}")
                except:
                    start_box.set_val(f"{self.sel_start:.2f}")
            else:
                self.line_start.set_visible(False)
                start_box.set_val("")

            # Update End Line
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
            
            # Update Instructions
            if self.selection_state == "start_set":
                instruction_text.set_text(f"Start: {self.sel_start:.2f}s set. Click for End point.")
                instruction_text.set_color(COLORS['warning'])
            elif self.selection_state == "complete":
                dur = abs(self.sel_end - self.sel_start)
                instruction_text.set_text(f"Selected: {min(self.sel_start, self.sel_end):.2f}s - {max(self.sel_start, self.sel_end):.2f}s (Dur: {dur:.2f}s)")
                instruction_text.set_color(COLORS['success'])
            
            self.canvas.draw()

        def on_click(event):
            """Handle plot clicks"""
            if event.inaxes != ax: return
            
            val = event.xdata
            
            if self.selection_state == "complete":
                # RESET: Start a new selection from scratch
                self.sel_start = val
                self.sel_end = None
                self.selection_state = "start_set"
            
            elif self.selection_state == "start_set":
                # FINISH: Set the end point
                self.sel_end = val
                # Auto-sort
                if self.sel_end < self.sel_start:
                    self.sel_start, self.sel_end = self.sel_end, self.sel_start
                self.selection_state = "complete"
            
            update_visuals()

        def on_start_submit(text):
            """User pressed Enter in Start Box"""
            try:
                val = float(text)
                self.sel_start = val
                
                # If we have a complete pair and start > end, swap them
                if self.selection_state == "complete" and self.sel_end is not None:
                    if self.sel_start > self.sel_end:
                         self.sel_start, self.sel_end = self.sel_end, self.sel_start
                
                # If we were in the middle of clicking, let's treat manual entry as valid start
                if self.sel_start is not None and self.sel_end is None:
                     self.selection_state = "start_set"
                
                update_visuals()
            except ValueError:
                pass # Ignore non-numbers

        def on_end_submit(text):
            """User pressed Enter in End Box"""
            try:
                val = float(text)
                self.sel_end = val
                
                # If incomplete, treat this as completing it (using 0 as start if missing)
                if self.sel_start is None:
                    self.sel_start = 0.0

                self.selection_state = "complete"
                
                # Swap if needed
                if self.sel_start > self.sel_end:
                     self.sel_start, self.sel_end = self.sel_end, self.sel_start

                update_visuals()
            except ValueError:
                pass

        def on_reset(event):
            """Reset to full duration"""
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
                instruction_text.set_text("⚠ Please select start and end points first")
                instruction_text.set_color(COLORS['error'])
                self.canvas.draw()

        # Connect Events
        self.cid = self.canvas.mpl_connect("button_press_event", on_click)
        start_box.on_submit(on_start_submit)
        end_box.on_submit(on_end_submit)
        clear_btn.on_clicked(on_reset)
        confirm_btn.on_clicked(on_confirm)

        # Cleanup refs
        self.start_time_box = start_box
        self.end_time_box = end_box
        self.clear_button = clear_btn
        self.confirm_button = confirm_btn
        
        # Initial Draw
        update_visuals()
        
        self.canvas.draw()
        event_loop.exec_()
        
        print(f"✓ Selected time window: {self.plateau_coords / self.sampling_rate} seconds")

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

    # =========================================================================
    # DECOMPOSITION
    # =========================================================================

    def _start_decomposition(self):
        if not self.config or not self.emg_path or self.emg_data is None:
            QMessageBox.warning(self, "Error", "No session loaded.")
            return
        
        # Step 1: Sync parameters from UI
        self._sync_params_from_ui()
        
        # Step 2: Manual channel rejection
        self.status_label.setText("Select channels to reject...")
        self._manual_channel_rejection()
        
        # Step 3: Time window selection
        self.status_label.setText("Select time window...")
        self._select_time_window()
        
        if self.plateau_coords is None:
            QMessageBox.warning(self, "Error", "No time window selected.")
            return
        
        # Clean up matplotlib
        self._cleanup_matplotlib_widgets()
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        ax.text(
            0.5, 0.5, "Decomposing...", 
            color=COLORS['warning'], fontsize=16, weight='bold',
            ha='center', va='center', transform=ax.transAxes
        )
        ax.axis('off')
        self.canvas.draw()
        
        # Step 4: Start decomposition worker
        self.status_label.setText("Starting decomposition...")
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        
        save_path = Path(self.config.output_dir) if self.config.output_dir else self.emg_path.parent
        save_path = save_path / f"{self.emg_path.stem}_decomposed.pkl"
        
        self.worker = DecompositionWorker(
            self.emg_data,
            self.grid_configs,
            self.rejected_channels,
            self.plateau_coords,
            self.sampling_rate,
            save_path
        )
        self.worker.progress.connect(self.status_label.setText)
        self.worker.electrode_completed.connect(self._on_electrode_complete)
        self.worker.finished.connect(self._on_decomposition_finished)
        self.worker.error.connect(self._on_decomposition_error)
        self.worker.source_found.connect(self._on_source_found)
        self.worker.start()

    def _on_electrode_complete(self, current: int, total: int):
        """Update progress when an electrode completes."""
        self.status_label.setText(f"Completed {current}/{total} grids...")

    def _stop_decomposition(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping...")
            self.worker.wait()
        self._reset_ui_state()

    def _on_decomposition_finished(self, results):
        self.status_label.setText("Decomposition Complete")
        QMessageBox.information(
            self, "Done", 
            f"Found {results.get('n_units', 0)} Motor Units.\n"
            f"Saved to: {results.get('path')}"
        )
        self._reset_ui_state()

    def _on_decomposition_error(self, err_msg):
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Decomposition Failed:\n{err_msg}")
        self._reset_ui_state()

    def _on_source_found(self, source, timestamps, iteration, silhouette):
        """Update GUI when a source is found during decomposition"""
        # Update status
        self.status_label.setText(
            f"Iteration {iteration}: Found MU (SIL: {silhouette:.3f})"
        )

        self._plot_source_realtime(source, timestamps, iteration, silhouette)
        
        # Force GUI update
        QApplication.processEvents()

    def _plot_source_realtime(self, source, timestamps, iteration, silhouette):
        """Plot source in real-time as it's found"""
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        
        # Convert to numpy for plotting
        source_np = source.detach().cpu().numpy() if torch.is_tensor(source) else source
        timestamps_np = timestamps.detach().cpu().numpy() if torch.is_tensor(timestamps) else timestamps
        
        # Ensure timestamps are valid integers for indexing
        if len(timestamps_np) > 0:
            # Cast to int and ensure they are within bounds of the source array
            idx = timestamps_np.astype(int)
            idx = idx[idx < len(source_np)] 
            y_values = source_np[idx]
        else:
            idx = []
            y_values = []
        
        # 1. Plot source waveform in BLUE
        # using a standard blue hex or 'tab:blue'
        ax.plot(source_np, color='#2b6cb0', linewidth=1.2, alpha=0.9) 
        
        # 2. Plot markers in ORANGE at timestamp locations
        if len(idx) > 0:
            ax.plot(idx, y_values, 'o', color='#ed8936', markersize=4, alpha=0.9)

        ax.set_title(
            f"Iteration {iteration} | Silhouette: {silhouette:.3f} | {len(timestamps_np)} spikes",
            color=COLORS['foreground'], fontsize=12, weight='bold'
        )
        ax.set_xlabel("Sample", color=COLORS['foreground'])
        ax.set_ylabel("Amplitude", color=COLORS['foreground'])
        ax.tick_params(colors=COLORS['foreground'])
        
        # Remove top/right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['foreground'])
        ax.spines['bottom'].set_color(COLORS['foreground'])
        
        self.canvas.draw()

    def _reset_ui_state(self):
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)