"""
Decomposition Tab - Manages EMG signal decomposition.
"""

from pathlib import Path
from typing import Optional, Dict, Set
import re
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QStackedWidget,
    QSizePolicy,
    QFrame,
    QMessageBox,
    QDialog,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QApplication,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEventLoop

# Visualization
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from scd_app.gui.style.styling import (
    COLORS,
    FONT_SIZES,
    SPACING,
    FONT_FAMILY,
    get_section_header_style,
    get_label_style,
    get_button_style,
)

from scd_app.core.config import SessionConfig
import torch
from scd_app.core.decomp_worker import DecompositionWorker

# ── Filename → active-aux helpers ────────────────────────────────────────────
_FINGER_ABBREV: Dict[str, str] = {
    "T": "Thumb", "I": "Index", "M": "Middle", "R": "Ring", "L": "Little"
}
_MOTION_ABBREV: Dict[str, str] = {"ext": "Ext", "flex": "Flex"}
_TASK_PATTERN = re.compile(r"mvc-\d+(ext|flex)_fing-([TIMRL]+)", re.IGNORECASE)


def _parse_task_targets(file_stem: str) -> Optional[Set[str]]:
    """Return expected active unit strings (e.g. {'Index Ext'}) from the filename."""
    m = _TASK_PATTERN.search(file_stem)
    if not m:
        return None
    motion = _MOTION_ABBREV.get(m.group(1).lower())
    if not motion:
        return None
    targets = {
        f"{_FINGER_ABBREV[c]} {motion}"
        for c in m.group(2).upper()
        if c in _FINGER_ABBREV
    }
    return targets or None


def _filter_aux_for_task(aux_cfgs: list, file_stem: str) -> list:
    """Return the subset of aux_cfgs whose unit label matches the task in the filename.
    Falls back to showing all when the filename doesn't parse or nothing matches."""
    targets = _parse_task_targets(file_stem)
    if not targets:
        return aux_cfgs
    matched = [a for a in aux_cfgs if a.get("unit", "") in targets]
    return matched if matched else aux_cfgs


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
        self.plateau_coords = None

        # UI References
        self.grid_selector = None
        self.param_stack = None
        self.param_widgets = {}

        # Matplotlib cleanup
        self.cid = None
        self._setup_cancelled = False
        self._discard_on_stop = False  # True = stop now (discard), False = save on stop

        # Time window selection state
        self.sel_start: float = 0.0
        self.sel_end = None
        self._selection_state: str = "idle"  # 'idle', 'start_set', 'complete'
        self._rms_ax = None
        self._line_start = None
        self._line_end = None
        self._canvas_click_cid = None

        self.init_ui()

    def setup_session(self, config: SessionConfig, emg_paths: list):
        """Called by MainWindow when Configuration is applied."""
        self.config = config
        self.emg_paths = [Path(p) if not isinstance(p, Path) else p for p in emg_paths]
        self.emg_path = self.emg_paths[0]
        self.sampling_rate = config.sampling_frequency

        n_files = len(self.emg_paths)
        self.file_path_label.setText(
            f"\U0001f4c4 {self.emg_path.name}"
            if n_files == 1
            else f"\U0001f4c4 {n_files} files (first: {self.emg_path.name})"
        )
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px; font-weight: bold;"
        )

        self._load_emg_data()
        self._load_grid_configs()

        # Show a ready prompt — channel rejection + RMS happen when user clicks Start
        self.figure.clf()
        self.figure.set_facecolor(COLORS["background"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["background"])
        ax.text(
            0.5,
            0.5,
            "Session loaded.\nClick  'Start Decomposition'  to begin.",
            color=COLORS["text_muted"],
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self.canvas.draw()

        print(
            f"Decomposition Tab Ready: {len(self.grid_configs)} grids, {n_files} file(s)."
        )

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
        layout.addWidget(
            QLabel("GLOBAL PARAMETERS", styleSheet=get_section_header_style("info"))
        )

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

        muap_win_spin = QDoubleSpinBox()
        muap_win_spin.setRange(1, 200)
        muap_win_spin.setSingleStep(1)
        muap_win_spin.setValue(20)
        muap_win_spin.setSuffix(" ms")
        add_global("MUAP Window:", muap_win_spin, "muap_window_ms")

        layout.addLayout(global_grid)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep)

        # === Batch options ===
        batch_label = QLabel(
            "BATCH OPTIONS", styleSheet=get_section_header_style("info")
        )
        layout.addWidget(batch_label)

        batch_grid = QGridLayout()
        batch_grid.setSpacing(2)

        lbl1 = QLabel("Channel Rejection:")
        lbl1.setStyleSheet(f"color: {COLORS['info_light']};")
        self.rejection_mode = QComboBox()
        self.rejection_mode.addItems(["Per file", "First file only"])
        batch_grid.addWidget(lbl1, 0, 0)
        batch_grid.addWidget(self.rejection_mode, 0, 1)

        downsample_cb = QCheckBox("Downsample EMG for Display")
        downsample_cb.setChecked(False)
        downsample_cb.setToolTip(
            "Reduces the number of points plotted during channel rejection for faster rendering.\n"
            "Useful for long recordings at high sampling rates.\n"
            "Does not affect the data used for decomposition."
        )
        add_global("", downsample_cb, "downsample_display")

        lbl2 = QLabel("Time Window:")
        lbl2.setStyleSheet(f"color: {COLORS['info_light']};")
        self.time_mode = QComboBox()
        self.time_mode.addItems(["Manual selection", "Full file"])
        self.time_mode.currentTextChanged.connect(self._on_time_mode_changed)
        batch_grid.addWidget(lbl2, 1, 0)
        batch_grid.addWidget(self.time_mode, 1, 1)

        layout.addLayout(batch_grid)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep2)

        # === Per-grid parameters ===
        layout.addWidget(
            QLabel("PER-GRID PARAMETERS", styleSheet=get_section_header_style("info"))
        )

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
        self.start_btn.setStyleSheet(
            f"""
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
        """
        )

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_decomposition)
        self.stop_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 6px; font-weight: bold;"
        )

        self.cancel_setup_btn = QPushButton("Cancel")
        self.cancel_setup_btn.setMinimumHeight(45)
        self.cancel_setup_btn.setVisible(False)
        self.cancel_setup_btn.clicked.connect(self._cancel_setup)
        self.cancel_setup_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 6px; font-weight: bold;"
        )

        btn_layout.addWidget(self.stop_btn, stretch=2)
        btn_layout.addWidget(self.cancel_setup_btn, stretch=2)
        btn_layout.addWidget(self.start_btn, stretch=3)
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

        # --- Time window bar (shown below header when Manual selection) ---
        _edit_style = (
            f"QLineEdit {{ background-color: {COLORS.get('background_input','#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 3px; padding: 2px 6px;"
            f" font-size: {FONT_SIZES.get('small','9pt')}; }}"
        )
        _ts_btn_style = (
            f"QPushButton {{ background-color: {COLORS.get('background_input','#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 4px; padding: 3px 10px;"
            f" font-size: {FONT_SIZES.get('small','9pt')}; }}"
            f"QPushButton:hover {{ border-color: {COLORS.get('info','#4a9eff')}; }}"
        )
        self.time_sel_widget = QWidget()
        ts_bar = QHBoxLayout(self.time_sel_widget)
        ts_bar.setContentsMargins(4, 0, 4, 4)
        ts_bar.setSpacing(6)

        lbl_s = QLabel("Start (s):")
        lbl_s.setStyleSheet(
            f"color: {COLORS['info_light']}; font-size: {FONT_SIZES.get('small','9pt')};"
        )
        self.start_time_edit = QLineEdit("0.00")
        self.start_time_edit.setStyleSheet(_edit_style)
        self.start_time_edit.setFixedWidth(72)
        self.start_time_edit.editingFinished.connect(self._on_start_time_entered)

        lbl_e = QLabel("End (s):")
        lbl_e.setStyleSheet(
            f"color: {COLORS['info_light']}; font-size: {FONT_SIZES.get('small','9pt')};"
        )
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("…")
        self.end_time_edit.setStyleSheet(_edit_style)
        self.end_time_edit.setFixedWidth(72)
        self.end_time_edit.editingFinished.connect(self._on_end_time_entered)

        self.clear_time_btn = QPushButton("✕  Clear")
        self.clear_time_btn.setStyleSheet(_ts_btn_style)
        self.clear_time_btn.clicked.connect(self._on_clear_time_selection)

        instr_lbl = QLabel("  Click plot: 1st = start,  2nd = end")
        instr_lbl.setStyleSheet(
            f"color: {COLORS.get('text_muted','#6c7086')}; font-size: 8pt; font-style: italic;"
        )

        ts_bar.addWidget(lbl_s)
        ts_bar.addWidget(self.start_time_edit)
        ts_bar.addWidget(lbl_e)
        ts_bar.addWidget(self.end_time_edit)
        ts_bar.addWidget(self.clear_time_btn)
        ts_bar.addWidget(instr_lbl)
        ts_bar.addStretch()

        layout.addWidget(self.time_sel_widget, stretch=0)
        self.time_sel_widget.setVisible(False)

        self.figure = Figure(facecolor=COLORS["background"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS["background"])
        self.ax.text(
            0.5,
            0.5,
            "Waiting for configuration...",
            color=COLORS["text_muted"],
            ha="center",
            va="center",
            transform=self.ax.transAxes,
        )
        self.ax.axis("off")

        layout.addWidget(self.canvas, stretch=1)
        return container

    def _load_emg_data(self):
        """Load EMG data using the layout from config."""
        import copy
        from scd_app.io.data_loader import load_field

        try:
            layout = getattr(self.config, "data_layout", None)
            if layout is None:
                raise ValueError(
                    "No data layout configured -- select a Data Format in Configuration tab"
                )

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
            QMessageBox.critical(
                self, "Load Error", f"Failed to load EMG data:\n{str(e)}"
            )
            print(f"Error loading EMG data: {e}")

    def _load_grid_configs(self):
        """Populate grid configurations based on SessionConfig."""
        self.grid_configs.clear()
        self.param_widgets.clear()
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
                f"{port.name} ({port.electrode.name})", port.name
            )

            page = self._create_param_page(port.name, defaults, electrode_type_label)
            self.param_stack.addWidget(page)

    def _create_param_page(
        self, port_name: str, defaults: dict, electrode_type: str = ""
    ) -> QWidget:
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

        widgets["extension_factor"] = add_row(
            "Extension Factor:", QLineEdit(str(defaults["extension_factor"]))
        )
        widgets["highpass_hz"] = add_row(
            "High-pass (Hz):", QLineEdit(str(defaults["highpass_hz"]))
        )
        widgets["lowpass_hz"] = add_row(
            "Low-pass (Hz):", QLineEdit(str(defaults["lowpass_hz"]))
        )

        notch = QComboBox()
        notch.addItems(["None", "50", "60"])
        notch.setCurrentText(defaults["notch_filter"])
        widgets["notch_filter"] = add_row("Notch Filter:", notch)

        harmonics_cb = QCheckBox("Include Harmonics")
        harmonics_cb.setChecked(defaults["notch_harmonics"])
        widgets["notch_harmonics"] = add_row("", harmonics_cb)

        self.param_widgets[port_name] = widgets
        return page

    def _on_grid_changed(self, index: int):
        if index >= 0:
            self.param_stack.setCurrentIndex(index)

    def _sync_params_from_ui(self):
        """Update grid_configs with values from global + per-grid UI."""
        # Read global params once
        global_params = {
            "sil_threshold": float(self.global_widgets["sil_threshold"].text()),
            "iterations": int(self.global_widgets["iterations"].text()),
            "clamp": self.global_widgets["clamp"].currentText() == "True",
            "fitness": self.global_widgets["fitness"].currentText(),
            "peel_off": self.global_widgets["peel_off"].currentText() == "True",
            "swarm": self.global_widgets["swarm"].currentText() == "True",
            "muap_window_ms": self.global_widgets["muap_window_ms"].value(),
        }

        # Merge into each grid
        for port_name, widgets in self.param_widgets.items():
            try:
                params = self.grid_configs[port_name]["params"]

                # Global
                params.update(global_params)

                # Per-grid
                params["extension_factor"] = int(widgets["extension_factor"].text())
                params["highpass_hz"] = float(widgets["highpass_hz"].text())
                params["lowpass_hz"] = float(widgets["lowpass_hz"].text())
                params["notch_filter"] = widgets["notch_filter"].currentText()
                params["notch_harmonics"] = widgets["notch_harmonics"].isChecked()
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not sync parameter for {port_name}: {e}")

    # ------------------------------------------------------------------ #
    #  Aux / force channel helpers                                        #
    # ------------------------------------------------------------------ #


    @staticmethod
    def _downsample_for_display(data: np.ndarray, canvas_width_px: int) -> tuple:
        """Downsample EMG data to a pixel-budget limit for display performance."""
        max_points = canvas_width_px * 3
        step = max(1, data.shape[0] // max_points)
        if step <= 1:
            return data, 1
        return data[::step, :], step

    def _manual_channel_rejection(self):
        """Show EMG channels per grid and let user select which to remove."""
        if self.emg_data is None:
            return

        self._cleanup_matplotlib_widgets()
        self.figure.set_facecolor(COLORS["background"])

        # Build grid list
        grid_list = list(self.grid_configs.items())
        if not grid_list:
            return

        # Init rejection masks
        self.rejected_channels = []
        for port_idx, (port_name, config) in enumerate(grid_list):
            n_channels = len(config["channels"])
            if port_idx < len(self.rejected_channels):
                pass  # keep existing
            else:
                self.rejected_channels.append(np.zeros(n_channels, dtype=int))

        # ── Aux / force overlay setup ─────────────────────────────────────
        _aux_cfgs = getattr(self.config, "aux_channels", []) if self.config else []
        _stem = self.emg_path.stem if self.emg_path else ""
        _aux_to_show = _filter_aux_for_task(_aux_cfgs, _stem)

        import time

        nav = {"current": 0, "show_force": bool(_aux_to_show)}

        def draw_grid(grid_idx, restore_view=None, fixed_separation=None):
            """Draw a single grid's channels.

            Args:
                restore_view:      optional ((x0, x1), (y0, y1)) — preserves zoom/pan.
                fixed_separation:  optional float — reuse an existing separation value
                                   so the channel geometry doesn't shift when a channel
                                   is toggled and the active-channel std changes.
            """
            self.figure.clf()

            port_name, config = grid_list[grid_idx]
            channels = config["channels"]
            n_channels = len(channels)
            mask = self.rejected_channels[grid_idx]

            ax = self.figure.add_axes([0.03, 0.14, 0.94, 0.80])
            ax.set_facecolor(COLORS["background"])
            self.figure.set_facecolor(COLORS["background"])

            # Title
            self.figure.text(
                0.5,
                0.96,
                f"{port_name}  ({grid_idx + 1}/{len(grid_list)})",
                ha="center",
                va="center",
                fontsize=13,
                weight="bold",
                color=COLORS["foreground"],
            )

            # Instructions
            self.figure.text(
                0.5,
                0.08,
                "Click = Toggle channel  |  Scroll = Zoom X  |  Shift+Scroll = Pan  |  Ctrl+Scroll = Zoom XY  |  Drag = Pan ",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color=COLORS["info"],
            )

            raw_data = self.emg_data[:, channels].numpy()

            # Downsample for display option (if enabled, does not affect actual decomposition data)
            use_downsample = self.global_widgets["downsample_display"].isChecked()
            if use_downsample:
                canvas_width_px = self.canvas.get_width_height()[0] or 1000
                disp_data, step = self._downsample_for_display(
                    raw_data, canvas_width_px
                )
            else:
                disp_data = raw_data
                step = 1
            max_len = disp_data.shape[0]

            # Normalise separation using only active (non-rejected) channels.
            if fixed_separation is not None:
                separation = fixed_separation
            else:
                active_idx = np.where(mask == 0)[0]
                ref_data = (
                    disp_data[:, active_idx] if len(active_idx) > 0 else disp_data
                )
                active_std = np.std(ref_data)
                separation = active_std * 15 if active_std > 0 else 1.0

            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                y_pos = ch * separation
                if is_rejected:
                    # Faint dashed flat line — visually removed but still clickable
                    ax.plot(
                        [0, max_len],
                        [y_pos, y_pos],
                        color=COLORS["error"],
                        alpha=0.25,
                        linewidth=0.8,
                        linestyle="--",
                    )
                else:
                    ax.plot(
                        disp_data[:, ch] + y_pos,
                        color=COLORS["info"],
                        alpha=0.8,
                        linewidth=1.0,
                    )

            # Channel labels on the left
            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                ax.text(
                    -max_len * 0.01,
                    ch * separation,
                    f"{ch}",
                    color=COLORS["error"] if is_rejected else COLORS["text_muted"],
                    fontsize=7,
                    ha="right",
                    va="center",
                )

            total_height = n_channels * separation
            ax.set_xlim(0, max_len)
            ax.set_ylim(-separation, total_height)
            ax.margins(0)

            # Restore zoom/pan if the user was already viewing a sub-region
            if restore_view is not None:
                ax.set_xlim(restore_view[0])
                ax.set_ylim(restore_view[1])

            # Hide everything except the bottom time axis
            ax.set_yticks([])
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(COLORS.get("text_muted", "#888888"))
            ax.tick_params(
                axis="x",
                colors=COLORS.get("text_muted", "#888888"),
                labelsize=8,
                length=4,
            )

            # Format x-axis ticks as time (seconds), accounting for downsampling
            def _time_fmt(x, pos):
                seconds = x * step / self.sampling_rate
                if seconds >= 60:
                    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
                return f"{seconds:.1f}s"

            ax.xaxis.set_major_formatter(FuncFormatter(_time_fmt))

            # ── Force / aux overlay ───────────────────────────────────────────
            if _aux_to_show and nav.get("show_force", False):
                total_height = n_channels * separation
                force_color = "#FFD700"  # gold — visible on dark background
                first_label = True
                for a in _aux_to_show:
                    s = int(a.get("start_chan", 0))
                    e = int(a.get("end_chan", s + 1))
                    n_total_ch = self.emg_data.shape[1]
                    if s >= e or e > n_total_ch:
                        continue
                    raw = self.emg_data[:, s:e].numpy().mean(axis=1)  # (samples,)
                    sig = raw - raw.mean()  # remove DC offset
                    peak = max(abs(sig.max()), abs(sig.min()), 1e-9)
                    # Scale to span ~70 % of total channel height, centred
                    sig_scaled = sig / peak * (0.35 * total_height) + total_height / 2
                    label = (
                        a.get("unit", a.get("name", "force")) if first_label else "_"
                    )
                    ax.plot(
                        sig_scaled,
                        color=force_color,
                        alpha=0.55,
                        linewidth=1.2,
                        zorder=3,
                        label=label,
                    )
                    first_label = False
                ax.legend(
                    loc="upper right",
                    fontsize=7,
                    facecolor=COLORS["background_light"],
                    labelcolor=force_color,
                    framealpha=0.7,
                )

            # --- Navigation buttons ---
            prev_ax = self.figure.add_axes([0.05, 0.01, 0.12, 0.05])
            next_ax = self.figure.add_axes([0.83, 0.01, 0.12, 0.05])
            confirm_ax = self.figure.add_axes([0.34, 0.01, 0.18, 0.05])
            cancel_rej_ax = self.figure.add_axes([0.53, 0.01, 0.13, 0.05])
            reset_ax = self.figure.add_axes([0.87, 0.95, 0.10, 0.04])

            is_last = grid_idx == len(grid_list) - 1
            is_first = grid_idx == 0

            prev_btn = Button(
                prev_ax,
                "<- Previous",
                color=COLORS["background_light"],
                hovercolor=COLORS["background_hover"],
            )
            prev_btn.label.set_color(
                COLORS["foreground"] if not is_first else COLORS["text_muted"]
            )
            prev_btn.label.set_fontsize(10)

            next_btn = Button(
                next_ax,
                "Next  →",
                color=COLORS["info"] if not is_last else COLORS["background_light"],
                hovercolor=COLORS["info_light"],
            )
            next_btn.label.set_color("white" if not is_last else COLORS["text_muted"])
            next_btn.label.set_fontsize(11)
            next_btn.label.set_weight("bold")

            confirm_btn = Button(
                confirm_ax, "CONFIRM ALL", color=COLORS["success"], hovercolor="#2EA043"
            )
            confirm_btn.label.set_color("white")
            confirm_btn.label.set_weight("bold")
            confirm_btn.label.set_fontsize(10)

            cancel_rej_btn = Button(
                cancel_rej_ax,
                "Cancel",
                color=COLORS["error"],
                hovercolor="#c0392b",
            )
            cancel_rej_btn.label.set_color("white")
            cancel_rej_btn.label.set_weight("bold")
            cancel_rej_btn.label.set_fontsize(10)

            reset_btn = Button(
                reset_ax,
                "⟳ Reset View",
                color="#4b5563",
                hovercolor="#6b7280",
            )
            reset_btn.label.set_color("white")
            reset_btn.label.set_fontsize(9)

            # ── "Show force" checkbox (only when aux channels available) ────
            if _aux_to_show:
                from matplotlib.widgets import CheckButtons

                cb_ax = self.figure.add_axes([0.20, 0.005, 0.13, 0.06])
                cb_ax.set_facecolor(COLORS["background"])
                check = CheckButtons(
                    cb_ax, ["Show force"], [nav.get("show_force", True)]
                )
                check.labels[0].set_color(COLORS["foreground"])
                check.labels[0].set_fontsize(9)
                for r in getattr(check, "rectangles", []):
                    r.set_edgecolor(COLORS.get("text_muted", "#888"))
                    r.set_facecolor(COLORS["background_light"])

                def on_force_toggle(_label, _check=check):
                    nav["show_force"] = _check.get_status()[0]
                    disconnect()
                    draw_grid(nav["current"])

                check.on_clicked(on_force_toggle)
                nav.setdefault("buttons", []).append(check)

            # Rejected count for this grid
            n_rej = np.sum(mask)
            rej_text = self.figure.text(
                0.285,
                0.03,
                (f"{n_rej} ch. rejected" if n_rej > 0 else ""),
                ha="center",
                va="center",
                fontsize=9,
                color=COLORS["error"] if n_rej > 0 else COLORS["text_muted"],
            )

            # ── Interaction state ───────────────────────────────────────────
            state = {
                "press_event": None,
                "last_scroll_time": 0,
                # Left-button drag pan
                "dragging": False,
                "drag_start": None,
                "drag_xlim": None,
                "drag_ylim": None,
                # Scroll debounce
                "pending_xlim": None,
                "pending_ylim": None,
                "scroll_timer": QTimer(),
            }
            state["scroll_timer"].setSingleShot(True)
            state["scroll_timer"].setInterval(16)  # ~60 fps cap

            # How many pixels the mouse must move before a press becomes a drag
            DRAG_THRESHOLD_PX = 4
            SCROLL_GUARD_MS = 150

            def _apply_pending_scroll():
                """Flush buffered scroll limits in one redraw.
                Also prevents zooming out further than the full reset view."""
                if state["pending_xlim"] is not None:
                    lo, hi = _clamp_xlim(*state["pending_xlim"])
                    # Don't allow wider than the full data range
                    if hi - lo > _x_max - _x_min:
                        lo, hi = _x_min, _x_max
                    ax.set_xlim(lo, hi)
                    state["pending_xlim"] = None
                if state["pending_ylim"] is not None:
                    lo, hi = _clamp_ylim(*state["pending_ylim"])
                    if hi - lo > _y_max - _y_min:
                        lo, hi = _y_min, _y_max
                    ax.set_ylim(lo, hi)
                    state["pending_ylim"] = None
                self.canvas.draw_idle()

            state["scroll_timer"].timeout.connect(_apply_pending_scroll)

            def on_press(event):
                if event.inaxes != ax or event.button != 1:
                    return
                state["press_event"] = event
                state["dragging"] = False
                state["drag_start"] = (event.x, event.y)
                state["drag_xlim"] = ax.get_xlim()
                state["drag_ylim"] = ax.get_ylim()

            def on_motion(event):
                if state["drag_start"] is None or event.inaxes != ax:
                    return
                dx = event.x - state["drag_start"][0]
                dy = event.y - state["drag_start"][1]
                if not state["dragging"]:
                    if abs(dx) > DRAG_THRESHOLD_PX or abs(dy) > DRAG_THRESHOLD_PX:
                        state["dragging"] = True
                    else:
                        return
                xlim = state["drag_xlim"]
                ylim = state["drag_ylim"]
                bbox = ax.get_window_extent()
                data_dx = -(dx / bbox.width) * (xlim[1] - xlim[0])
                data_dy = -(dy / bbox.height) * (ylim[1] - ylim[0])
                ax.set_xlim(_clamp_xlim(xlim[0] + data_dx, xlim[1] + data_dx))
                ax.set_ylim(_clamp_ylim(ylim[0] + data_dy, ylim[1] + data_dy))
                self.canvas.draw_idle()

            def on_release(event):
                if event.button != 1:
                    return

                was_dragging = state["dragging"]
                state["dragging"] = False
                state["drag_start"] = None

                # If panning, don't treat this as a channel click
                if was_dragging:
                    state["press_event"] = None
                    return

                if state["press_event"] is None:
                    return
                press = state["press_event"]
                state["press_event"] = None

                # Guard: ignore click if it arrived shortly after a scroll
                elapsed_ms = (time.time() - state["last_scroll_time"]) * 1000
                if elapsed_ms < SCROLL_GUARD_MS:
                    return
                if event.inaxes != ax or event.ydata is None:
                    return

                # Find the closest channel by y-position proximity
                closest_ch = None
                min_dist = float("inf")
                for ch in range(n_channels):
                    d = abs(event.ydata - ch * separation)
                    if d < separation * 0.6 and d < min_dist:
                        min_dist = d
                        closest_ch = ch

                if closest_ch is None:
                    return

                mask[closest_ch] = 1 - mask[closest_ch]
                # Preserve zoom/pan and channel geometry across the redraw
                saved_view = (ax.get_xlim(), ax.get_ylim())
                saved_sep = separation
                disconnect()
                draw_grid(
                    nav["current"], restore_view=saved_view, fixed_separation=saved_sep
                )

            def on_scroll(event):
                if event.inaxes != ax:
                    return
                state["last_scroll_time"] = time.time()

                modifiers = QApplication.keyboardModifiers()
                ctrl_held = bool(modifiers & Qt.ControlModifier)
                shift_held = bool(modifiers & Qt.ShiftModifier)

                zoom_scale = 0.85 if event.button == "up" else 1.18

                if shift_held:
                    # Pan horizontally — shift the window left/right by 10 % of its width
                    xlim = (
                        state["pending_xlim"]
                        if state["pending_xlim"] is not None
                        else ax.get_xlim()
                    )
                    span = xlim[1] - xlim[0]
                    pan_amount = span * 0.10 * (1 if event.button == "up" else -1)
                    state["pending_xlim"] = (xlim[0] + pan_amount, xlim[1] + pan_amount)

                elif ctrl_held:
                    # Zoom both axes symmetrically around the cursor
                    xlim = (
                        state["pending_xlim"]
                        if state["pending_xlim"] is not None
                        else ax.get_xlim()
                    )
                    ylim = (
                        state["pending_ylim"]
                        if state["pending_ylim"] is not None
                        else ax.get_ylim()
                    )
                    xdata = (
                        event.xdata
                        if event.xdata is not None
                        else (xlim[0] + xlim[1]) / 2
                    )
                    ydata = (
                        event.ydata
                        if event.ydata is not None
                        else (ylim[0] + ylim[1]) / 2
                    )
                    state["pending_xlim"] = (
                        xdata - (xdata - xlim[0]) * zoom_scale,
                        xdata + (xlim[1] - xdata) * zoom_scale,
                    )
                    state["pending_ylim"] = (
                        ydata - (ydata - ylim[0]) * zoom_scale,
                        ydata + (ylim[1] - ydata) * zoom_scale,
                    )

                else:
                    # Plain scroll — zoom X axis only, anchored at cursor
                    xlim = (
                        state["pending_xlim"]
                        if state["pending_xlim"] is not None
                        else ax.get_xlim()
                    )
                    xdata = (
                        event.xdata
                        if event.xdata is not None
                        else (xlim[0] + xlim[1]) / 2
                    )
                    state["pending_xlim"] = (
                        xdata - (xdata - xlim[0]) * zoom_scale,
                        xdata + (xlim[1] - xdata) * zoom_scale,
                    )

                if not state["scroll_timer"].isActive():
                    state["scroll_timer"].start()

            # ── View boundary limits (5 % margin) ────────────────────────
            _x_margin = max_len * 0.05
            _y_margin = total_height * 0.05
            _x_min = -_x_margin
            _x_max = max_len + _x_margin
            _y_min = -separation - _y_margin
            _y_max = total_height + _y_margin

            def _clamp_xlim(lo, hi):
                """Shift window to stay within data bounds, preserving width."""
                w = hi - lo
                if lo < _x_min:
                    lo, hi = _x_min, _x_min + w
                if hi > _x_max:
                    lo, hi = _x_max - w, _x_max
                return lo, hi

            def _clamp_ylim(lo, hi):
                """Shift window to stay within data bounds, preserving height."""
                h = hi - lo
                if lo < _y_min:
                    lo, hi = _y_min, _y_min + h
                if hi > _y_max:
                    lo, hi = _y_max - h, _y_max
                return lo, hi

            def _reset_view():
                ax.set_xlim(0, max_len)
                ax.set_ylim(-separation, total_height)
                self.canvas.draw_idle()

            def go_prev(event):
                if not is_first:
                    disconnect()
                    nav["current"] -= 1
                    draw_grid(nav["current"])

            def go_next(event):
                if not is_last:
                    disconnect()
                    nav["current"] += 1
                    draw_grid(nav["current"])

            def on_confirm(event):
                disconnect()
                self.figure.clf()
                self.figure.set_facecolor(COLORS["background"])

                total_rej = sum(np.sum(m) for m in self.rejected_channels)
                self.figure.text(
                    0.5,
                    0.5,
                    f"Rejected {total_rej} channels across {len(grid_list)} grids",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                    color=COLORS["success"],
                )
                self.canvas.draw()

                print("\nChannel Rejection Summary:")
                for pidx, (pname, _) in enumerate(grid_list):
                    n = np.sum(self.rejected_channels[pidx])
                    print(f"  {pname}: {n} channels rejected")

                QTimer.singleShot(100, event_loop.quit)

            def on_cancel_rejection(event):
                self._setup_cancelled = True
                disconnect()
                self.figure.clf()
                self.figure.set_facecolor(COLORS["background"])
                self.canvas.draw()
                event_loop.quit()

            # Connect
            cids = [
                self.canvas.mpl_connect("button_press_event", on_press),
                self.canvas.mpl_connect("button_release_event", on_release),
                self.canvas.mpl_connect("motion_notify_event", on_motion),
                self.canvas.mpl_connect("scroll_event", on_scroll),
            ]
            prev_btn.on_clicked(go_prev)
            next_btn.on_clicked(go_next)
            confirm_btn.on_clicked(on_confirm)
            cancel_rej_btn.on_clicked(on_cancel_rejection)
            reset_btn.on_clicked(lambda _: _reset_view())

            # Store for cleanup
            nav["cids"] = cids
            nav.setdefault("buttons", []).extend(
                [prev_btn, next_btn, confirm_btn, cancel_rej_btn, reset_btn]
            )
            nav["scroll_timer"] = state["scroll_timer"]

            self.canvas.draw()

        def disconnect():
            # Stop and clean up the scroll debounce timer
            timer = nav.get("scroll_timer")
            if timer is not None:
                timer.stop()

            # Disconnect canvas events
            for cid in nav.get("cids", []):
                self.canvas.mpl_disconnect(cid)

            # Disconnect and remove all navigation buttons
            for btn in nav.get("buttons", []):
                btn.disconnect_events()
                try:
                    btn.ax.remove()
                except Exception:
                    pass  # Ignore if already removed
            nav["buttons"] = []

        event_loop = QEventLoop()
        draw_grid(0)
        event_loop.exec_()

    def _cleanup_matplotlib_widgets(self):
        """Disconnect all active matplotlib event handlers before redrawing the canvas."""
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

        if hasattr(self, "scroll_cid") and self.scroll_cid is not None:
            self.canvas.mpl_disconnect(self.scroll_cid)
            self.scroll_cid = None

        if self._canvas_click_cid is not None:
            self.canvas.mpl_disconnect(self._canvas_click_cid)
            self._canvas_click_cid = None

    # ------------------------------------------------------------------ #
    #  RMS plot & time-window selection                                   #
    # ------------------------------------------------------------------ #

    def _show_rms_plot(self):
        """Display RMS amplitude on the canvas and restore selection lines."""
        if self.emg_data is None or not self.grid_configs:
            return

        self.start_btn.setEnabled(False)

        self._cleanup_matplotlib_widgets()
        self._rms_ax = None
        self.figure.clf()
        self.figure.set_facecolor(COLORS["background"])

        total_duration = self.emg_data.shape[0] / self.sampling_rate
        time_axis = np.arange(self.emg_data.shape[0]) / self.sampling_rate

        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["background"])
        self._rms_ax = ax

        colors = ["#4a9eff", "#a78bfa", "#48BB78", "#F6AD55", "#ff6b9d"]
        for idx, (port_name, config) in enumerate(self.grid_configs.items()):
            channels = config["channels"]
            channel_data = self.emg_data[:, channels].numpy()

            if idx < len(self.rejected_channels):
                rejected = self.rejected_channels[idx]
                active_channels = np.where(rejected == 0)[0]
                channel_data = channel_data[:, active_channels]

            rms = np.sqrt(np.mean(channel_data**2, axis=1))
            window = int(self.sampling_rate * 0.1)
            kernel = np.ones(window) / window
            rms_smooth = np.convolve(rms, kernel, mode="same")
            color = colors[idx % len(colors)]
            ax.plot(
                time_axis,
                rms_smooth,
                color=color,
                label=port_name,
                linewidth=2.5,
                alpha=0.85,
            )

        ax.set_xlabel(
            "Time (s)", color=COLORS["foreground"], fontsize=12, weight="bold"
        )
        ax.set_ylabel(
            "RMS Amplitude", color=COLORS["foreground"], fontsize=12, weight="bold"
        )
        ax.tick_params(colors=COLORS["foreground"])
        ax.set_xlim(0, total_duration)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS.get("background_light", "#2a2a3c"))
        ax.spines["bottom"].set_color(COLORS.get("background_light", "#2a2a3c"))
        if len(self.grid_configs) > 1:
            ax.legend(
                facecolor=COLORS.get("background_light", "#2a2a3c"),
                labelcolor=COLORS["foreground"],
                fontsize=9,
            )

        # Selection lines
        self._line_start = ax.axvline(
            x=0,
            color=COLORS.get("success", "#48BB78"),
            linestyle="--",
            linewidth=2,
            visible=False,
        )
        self._line_end = ax.axvline(
            x=total_duration,
            color=COLORS.get("error", "#f38ba8"),
            linestyle="--",
            linewidth=2,
            visible=False,
        )

        # Pre-fill text boxes with the full range as a convenience hint.
        if self._selection_state == "idle":
            self.sel_start = 0.0
            self.sel_end = total_duration
            self.start_time_edit.setText(f"{self.sel_start:.2f}")
            self.end_time_edit.setText(f"{self.sel_end:.2f}")
            # Draw lines but keep state idle — don't call _update_time_visuals
            # yet as it would trigger _update_confirm_btn_visibility too early
            if self._line_start is not None:
                self._line_start.set_xdata([self.sel_start, self.sel_start])
                self._line_start.set_visible(True)
            if self._line_end is not None:
                self._line_end.set_xdata([self.sel_end, self.sel_end])
                self._line_end.set_visible(True)
        else:
            self._update_time_visuals()

        self.canvas.draw()

        if self.time_mode.currentText() == "Manual selection":
            self._connect_canvas_click()

    def _on_time_mode_changed(self, text: str):
        is_manual = text == "Manual selection"
        if is_manual and self._rms_ax is not None:
            self._connect_canvas_click()
        else:
            self._disconnect_canvas_click()

    def _connect_canvas_click(self):
        self._disconnect_canvas_click()
        self._canvas_click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

    def _disconnect_canvas_click(self):
        if self._canvas_click_cid is not None:
            self.canvas.mpl_disconnect(self._canvas_click_cid)
            self._canvas_click_cid = None

    def _on_canvas_click(self, event):
        if self._rms_ax is None or event.inaxes != self._rms_ax:
            return
        if event.button != 1 or event.xdata is None:
            return
        val = event.xdata
        if self._selection_state in ("idle", "complete"):
            self.sel_start = val
            self.sel_end = None
            self._selection_state = "start_set"
        else:  # start_set
            self.sel_end = val
            if self.sel_end < self.sel_start:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._selection_state = "complete"
        self._update_time_visuals()

    def _on_start_time_entered(self):
        try:
            val = float(self.start_time_edit.text())
            self.sel_start = val
            if self._selection_state == "idle":
                self._selection_state = "start_set"
            if self.sel_end is not None and self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._update_time_visuals()
        except ValueError:
            pass

    def _on_end_time_entered(self):
        try:
            val = float(self.end_time_edit.text())
            self.sel_end = val
            if self.sel_start is None:
                self.sel_start = 0.0
            if self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._selection_state = "complete"
            self._update_time_visuals()
        except ValueError:
            pass

    def _on_clear_time_selection(self):
        if self.emg_data is not None:
            total_duration = self.emg_data.shape[0] / self.sampling_rate
            self.sel_start = 0.0
            self.sel_end = total_duration
        else:
            self.sel_start = 0.0
            self.sel_end = None
        # Always reset to idle so user must re-confirm
        self._selection_state = "idle"
        self._update_time_visuals()

    def _update_time_visuals(self):
        """Sync text boxes and plot lines with current sel_start / sel_end."""
        if self.sel_start is not None:
            self.start_time_edit.setText(f"{self.sel_start:.2f}")
        if self.sel_end is not None:
            self.end_time_edit.setText(f"{self.sel_end:.2f}")
        else:
            self.end_time_edit.clear()

        if self._line_start is not None:
            if self.sel_start is not None:
                self._line_start.set_xdata([self.sel_start, self.sel_start])
                self._line_start.set_visible(True)
            else:
                self._line_start.set_visible(False)

        if self._line_end is not None:
            if self.sel_end is not None:
                self._line_end.set_xdata([self.sel_end, self.sel_end])
                self._line_end.set_visible(True)
            else:
                self._line_end.set_visible(False)

        if self._rms_ax is not None:
            self.canvas.draw_idle()
        self._update_confirm_btn_visibility()

    def _start_decomposition(self):
        if not self.config or not self.emg_paths or self.emg_data is None:
            QMessageBox.warning(self, "Error", "No session loaded.")
            return

        # ── Phase 2: user confirmed time window → run current file ───────
        if getattr(self, "_awaiting_time_confirmation", False):
            if not self._use_full_file:
                if (
                    self._selection_state == "complete"
                    and self.sel_start is not None
                    and self.sel_end is not None
                ):
                    self.plateau_coords = np.array(
                        [
                            int(self.sel_start * self.sampling_rate),
                            int(self.sel_end * self.sampling_rate),
                        ]
                    )
                else:
                    # No selection made — use the full file as default
                    self.plateau_coords = np.array([0, self.emg_data.shape[0]])
            self._awaiting_time_confirmation = False
            self._run_current_file()
            return

        # ── Phase 1: initialise queue and prepare the first file ─────────
        self._sync_params_from_ui()
        self._set_params_enabled(False)
        self.stop_btn.setVisible(False)
        self._use_full_file = self.time_mode.currentText() == "Full file"
        self._share_rejection = self.rejection_mode.currentText() == "First file only"
        self._output_dir = (
            Path(self.config.output_dir)
            if self.config.output_dir
            else self.emg_path.parent
        )
        self._file_queue = list(self.emg_paths)
        self._file_idx = 0
        self._prepare_current_file()

    def _prepare_current_file(self):
        """Load file, do channel rejection + RMS, then wait for time window (per file)."""
        from scd_app.io.data_loader import load_field

        if self._file_idx >= len(self._file_queue):
            self.grid_indicator_label.setText("All files complete")
            self._reset_ui_state()
            if hasattr(self, "_last_decomp_path") and self._last_decomp_path:
                self.decomposition_complete.emit(self._last_decomp_path)
            return

        file_path = self._file_queue[self._file_idx]
        n_total = len(self._file_queue)
        self.grid_indicator_label.setText(
            f"File {self._file_idx + 1}/{n_total}: {file_path.name}"
        )

        # Load this file's EMG data — remove any channel filter so that
        # force/aux "signal" channels (which may sit beyond the EMG grid range)
        # are included in the array and available for the worker to save.
        try:
            import copy
            layout = getattr(self.config, "data_layout", None)
            layout_full = copy.deepcopy(layout) if layout else layout
            if (
                layout_full
                and "fields" in layout_full
                and "emg" in layout_full["fields"]
            ):
                layout_full["fields"]["emg"].pop("channels", None)
            emg = load_field(file_path, layout_full, "emg")
            self.emg_data = emg
            self.emg_path = file_path
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Failed to load {file_path.name}:\n{e}"
            )
            self._file_idx += 1
            self._prepare_current_file()
            return

        # Disable start_btn during channel rejection
        self.start_btn.setEnabled(False)

        # Channel rejection: manual per file, or reuse shared mask
        if self._file_idx == 0 or not self._share_rejection:
            self._manual_channel_rejection()
        else:
            self.rejected_channels = [m.copy() for m in self._shared_mask]

        # If the user pressed Cancel during channel rejection, bail out cleanly
        if self._setup_cancelled:
            self._setup_cancelled = False
            self._cancel_setup()
            return

        # Save shared mask after first file's rejection
        if self._file_idx == 0:
            self._shared_mask = [m.copy() for m in self.rejected_channels]

        # Show cleaned RMS plot (reset selection so lines are cleared)
        self._selection_state = "idle"
        self._show_rms_plot()

        if self._use_full_file:
            self.plateau_coords = np.array([0, self.emg_data.shape[0]])
            self._run_current_file()
        else:
            self.stop_btn.setVisible(False)
            self._awaiting_time_confirmation = True
            self.start_btn.setText("▶  Confirm && Run")
            self.start_btn.setEnabled(True)
            self.time_sel_widget.setVisible(True)
            self.cancel_setup_btn.setVisible(True)

    def _run_current_file(self):
        """Start the decomposition worker for the current file."""
        file_path = self._file_queue[self._file_idx]

        self._cleanup_matplotlib_widgets()
        self.start_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.cancel_setup_btn.setVisible(False)
        self.time_sel_widget.setVisible(False)

        self.figure.clf()
        self.figure.set_facecolor(COLORS["background"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["background"])
        ax.text(
            0.5,
            0.5,
            f"Decomposing {file_path.name}...",
            color=COLORS["warning"],
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self.canvas.draw()

        save_path = self._output_dir / f"{file_path.stem}_decomp_output.pkl"

        aux_configs = getattr(self.config, "aux_channels", [])
        self.worker = DecompositionWorker(
            self.emg_data,
            self.grid_configs,
            self.rejected_channels,
            self.plateau_coords,
            self.sampling_rate,
            save_path,
            aux_configs=aux_configs,
            emg_file_path=file_path,
        )
        self.worker.progress.connect(self._update_grid_indicator)
        self.worker.finished.connect(self._on_file_decomposition_finished)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.error.connect(self._on_decomposition_error)
        self.worker.source_found.connect(self._on_source_found)
        self.worker.start()

    def _on_file_decomposition_finished(self, results):
        """One file done — move to next (which repeats channel rejection + time selection)."""
        decomp_path = Path(results.get("path"))
        self._last_decomp_path = decomp_path

        self._file_idx += 1
        self._prepare_current_file()

    def _update_grid_indicator(self, message: str):
        """Update grid indicator when progress says 'Processing ...'"""
        if message.startswith("Processing "):
            # Extract "Processing GridName (2/6)..."
            self.grid_indicator_label.setText(f"{message}")

    def _stop_decomposition(self):
        """Show stop options immediately. Worker keeps running until user chooses.
        Uses a QDialog instead of QMessageBox so button order is fully controlled."""
        if not self.worker or not self.worker.isRunning():
            self._reset_ui_state()
            return

        partial = self.worker._partial_results
        n_complete = len(partial[0]["ports"]) if partial else 0
        total = len(self.grid_configs)

        dlg = QDialog(self)
        dlg.setWindowTitle("Stop Decomposition?")
        dlg.setMinimumWidth(520)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        title = QLabel("<b>Stop Decomposition?</b>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 12pt;")
        outer.addWidget(title)

        disc_note = (
            "  \u2014  closing the app will discard the current grid."
            if n_complete > 0
            else "  \u2014  no grids completed yet."
        )
        subtitle = QLabel(f"Grids completed: <b>{n_complete} / {total}</b>{disc_note}")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {COLORS.get('text_muted', '#888')}; font-size: 10pt;"
        )
        outer.addWidget(subtitle)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        wait_btn = QPushButton("Stop after current grid")
        wait_btn.setStyleSheet(
            f"background-color: {COLORS.get('info', '#4a9eff')}; color: white; "
            f"border-radius: 6px; font-weight: bold; padding: 10px 16px; font-size: 10pt;"
        )

        stop_btn = QPushButton("Stop now and close application")
        stop_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 6px; font-weight: bold; padding: 10px 16px; font-size: 10pt;"
        )

        cancel_btn = QPushButton("Continue")
        cancel_btn.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"color: {COLORS['foreground']}; border-radius: 6px; "
            f"padding: 10px 16px; font-size: 10pt; "
            f"border: 1px solid {COLORS.get('border', '#444')};"
        )

        btn_row.addWidget(wait_btn)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(cancel_btn)
        outer.addLayout(btn_row)

        # Wire up — store choice then close
        choice = [None]

        def _pick_wait():
            choice[0] = "wait"
            dlg.accept()

        def _pick_stop():
            choice[0] = "stop"
            dlg.accept()

        def _pick_cancel():
            choice[0] = "cancel"
            dlg.accept()

        wait_btn.clicked.connect(_pick_wait)
        stop_btn.clicked.connect(_pick_stop)
        cancel_btn.clicked.connect(_pick_cancel)

        dlg.exec_()

        if choice[0] == "wait":
            self._show_waiting_dialog()
        elif choice[0] == "stop":
            import sys

            sys.exit(0)
        # else: cancel — worker continues untouched

    def _show_waiting_dialog(self):
        """Keep a dialogue open while the current grid finishes.
        Updates live with number of iterations completed.
        Closes automatically when the worker emits stopped()."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Waiting for current grid…")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)
        dlg.setMinimumWidth(420)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        title_lbl = QLabel("<b>Finishing current grid…</b>")
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet("font-size: 13pt;")
        layout.addWidget(title_lbl)

        info_lbl = QLabel(
            "The current grid will be allowed to finish normally.<br>"
            "All previously completed grids will be saved."
        )
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet(f"color: {COLORS.get('text_muted', '#888')};")
        layout.addWidget(info_lbl)

        iter_lbl = QLabel("Current iteration: <b>0</b>")
        iter_lbl.setAlignment(Qt.AlignCenter)
        iter_lbl.setStyleSheet(
            f"color: {COLORS.get('info', '#4a9eff')}; font-size: 11pt;"
        )
        layout.addWidget(iter_lbl)

        cancel_now_btn = QPushButton(
            "Stop now and close application (results will be deleted)"
        )
        cancel_now_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 4px; padding: 8px; font-weight: bold;"
        )
        layout.addWidget(cancel_now_btn)

        def _on_source(_source, _timestamps, iteration, _silhouette):
            iterations = self.global_widgets["iterations"].text()
            iter_lbl.setText(
                f"Current iteration: <b>{iteration + 1} / {iterations}</b>"
            )

        def _on_stopped(_info):
            """Worker finished the current grid and stopped naturally."""
            try:
                self.worker.source_found.disconnect(_on_source)
            except Exception:
                pass
            dlg.accept()

        def _on_cancel_now():
            """Close the application — only reliable way to stop mid-grid."""
            import sys

            sys.exit(0)

        self.worker.source_found.connect(_on_source)
        self.worker.stopped.connect(_on_stopped)
        cancel_now_btn.clicked.connect(_on_cancel_now)

        # Tell the worker to stop after this grid completes
        self.worker.stop()

        dlg.exec_()

    def _on_worker_stopped(self, _info: dict):
        """Called when the worker emits stopped() after finishing the current grid.
        Saves completed grids unless _discard_on_stop is set."""
        discard = self._discard_on_stop
        self._discard_on_stop = False  # always reset

        if not discard:
            partial = self.worker._partial_results
            n_complete = len(partial[0]["ports"]) if partial else 0

            if n_complete > 0:
                try:
                    results, _ = partial
                    self.worker._save_results(results)
                    decomp_path = Path(self.worker.save_path)
                    self._last_decomp_path = decomp_path
                    self.decomposition_complete.emit(decomp_path)
                except Exception as e:
                    QMessageBox.critical(
                        self, "Save Error", f"Could not save results:\n{e}"
                    )

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
        ax.set_facecolor(COLORS["background"])

        source_np = source.detach().cpu().numpy() if torch.is_tensor(source) else source
        timestamps_np = (
            timestamps.detach().cpu().numpy()
            if torch.is_tensor(timestamps)
            else timestamps
        )

        if len(timestamps_np) > 0:
            idx = timestamps_np.astype(int)
            idx = idx[idx < len(source_np)]
            y_values = source_np[idx]
        else:
            idx = []
            y_values = []

        ax.plot(source_np, color="#2b6cb0", linewidth=1.2, alpha=0.9)

        if len(idx) > 0:
            ax.plot(idx, y_values, "o", color="#ed8936", markersize=4, alpha=0.9)

        ax.set_title(
            f"Iteration {iteration} | Silhouette: {silhouette:.3f} | {len(timestamps_np)} spikes",
            color=COLORS["foreground"],
            fontsize=12,
            weight="bold",
        )
        ax.set_xlabel("Sample", color=COLORS["foreground"])
        ax.set_ylabel("Amplitude", color=COLORS["foreground"])
        ax.tick_params(colors=COLORS["foreground"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["foreground"])
        ax.spines["bottom"].set_color(COLORS["foreground"])

        self.canvas.draw()

    def _set_params_enabled(self, enabled: bool) -> None:
        """Enable or disable all parameter input widgets on the left panel.
        Also hides/shows the start button — it reappears only when a valid
        time window is confirmed (see _update_confirm_btn_visibility)."""
        for w in self.global_widgets.values():
            w.setEnabled(enabled)
        for widgets in self.param_widgets.values():
            for w in widgets.values():
                w.setEnabled(enabled)
        self.rejection_mode.setEnabled(enabled)
        self.time_mode.setEnabled(enabled)
        self.grid_selector.setEnabled(enabled)

        # Downsample checkbox stays interactive at all times
        self.global_widgets["downsample_display"].setEnabled(True)

    def _update_confirm_btn_visibility(self):
        """Keep Confirm & Run enabled while awaiting time confirmation."""
        if not getattr(self, "_awaiting_time_confirmation", False):
            return
        self.start_btn.setEnabled(True)
        self.start_btn.setVisible(True)

    def _cancel_setup(self):
        """Cancel channel rejection / time window and return to the ready state."""
        self._awaiting_time_confirmation = False
        self._cleanup_matplotlib_widgets()
        self.figure.clf()
        self.figure.set_facecolor(COLORS["background"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["background"])
        ax.text(
            0.5,
            0.5,
            "Session loaded.\nClick  'Start Decomposition'  to begin.",
            color=COLORS["text_muted"],
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self.canvas.draw()
        self._reset_ui_state()

    def _reset_ui_state(self):
        self._awaiting_time_confirmation = False
        self.start_btn.setText("Start Decomposition")
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setVisible(False)
        self.time_sel_widget.setVisible(False)
        self._set_params_enabled(True)
        self.cancel_setup_btn.setVisible(False)
        self._cleanup_matplotlib_widgets()
        self.figure.clf()
        self.figure.set_facecolor(COLORS["background"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["background"])
        ax.text(
            0.5,
            0.5,
            "Session loaded.\nClick  'Start Decomposition'  to begin.",
            color=COLORS["text_muted"],
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self.canvas.draw()
