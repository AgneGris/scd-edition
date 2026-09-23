"""
Edition Tab - EMG spike editing interface.

On load, filters are applied via peel-off replay on the full preprocessed EMG,
and timestamps are re-detected via source_to_timestamps.  The plateau region
is shown as a shaded band on the source plot.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from scipy import signal as sp_signal

from scd_app.core.auto_editor import MIN_SPIKES, auto_edit
from scd_app.core.constants import ROA_THRESHOLD
from scd_app.core.duplicate_detection import (
    DUPLICATE_DETECTION_AVAILABLE,
    clear_duplicate_roles,
    scan_cross_port_duplicates,
    scan_within_port_duplicates,
)
from scd_app.core.filter_recalculation import (
    compute_all_full_sources,
    recalculate_unit_filter,
    supports_filter_recalculation,
    supports_full_source_computation,
)
from scd_app.core.mu_model import EditMode, MotorUnit, UndoAction
from scd_app.core.mu_properties import (
    MUProperties,
    compute_port_properties,
    recompute_unit_properties,
)
from scd_app.core.utils import to_numpy
from scd_app.gui.style.styling import (
    COLORS,
    FONT_FAMILY,
    FONT_SIZES,
    get_section_header_style,
)
from scd_app.gui.widgets.mu_properties_panel import MUPropertiesPanel
from scd_app.gui.widgets.muap_popout import MuapPopoutDialog
from scd_app.gui.widgets.source_plot_widget import (
    FiringRatePlotWidget,
    SelectionArm,
    SourcePlotWidget,
)
from scd_app.io.atomic_pickle import atomic_pickle_dump
from scd_app.io.audit_report import write_audit_report
from scd_app.io.decomposition_loader import load_decomposition_file
from scd_app.io.edition_session import (
    EditionSaveState,
    build_edition_save_data,
    load_edition_port,
    normalise_aux_channels,
    normalise_notes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EditionTab
# ---------------------------------------------------------------------------


class EditionTab(QWidget):
    data_modified = Signal()
    file_loaded = Signal()

    def __init__(self, fsamp: float = 2048.0, parent=None):
        super().__init__(parent)

        self._fsamp = fsamp
        self._ports: dict[str, list[MotorUnit]] = {}
        self._emg_data: dict[str, np.ndarray] = {}
        self._grid_info: dict[str, dict | None] = {}
        self._raw_port_channels: dict[str, np.ndarray] = {}
        self._rejected_ch_positions: dict[str, set] = {}
        self._notes: list[str] = []

        self._current_port: str | None = None
        self._current_mu_idx: int = -1
        self._edit_mode = EditMode.VIEW
        self._sel_arm = SelectionArm.NONE
        self._loaded_path: Path | None = None
        self._output_path: Path | None = None
        self._quit_after_save: bool = False
        self._dirty: bool = False
        self._config_aux_channels: list = []  # from the last applied config, used to fill missing MVC on load

        self._start_sample: int = 0
        self._end_sample: int = 0
        self._full_source_mode: bool = False
        self._redetect_timestamps: bool = True

        self._undo_stack: dict[tuple, list[UndoAction]] = {}
        self._redo_stack: dict[tuple, list[UndoAction]] = {}
        self._edit_history: list = []  # accumulated across sessions; saved in pickle
        self._original_decomp_data: dict | None = None
        self._filter_recalc_available: bool = False
        self._muap_popout: MuapPopoutDialog | None = None
        self._last_action_msg: str | None = None

        # Debounce: expensive recompute + MUAP render fire 120ms after the last edit
        self._props_timer = QTimer(self)
        self._props_timer.setSingleShot(True)
        self._props_timer.setInterval(120)
        self._props_timer.timeout.connect(self._flush_props_update)
        self._pending_source_changed: bool = False
        self._pending_props_key: tuple[str, int] | None = None

        # MUAP grid reuse: keep cell PlotDataItems alive across MU switches
        self._muap_cell_plots: dict[tuple[int, int], object] = {}
        self._muap_waveform_items: dict[tuple[int, int], object] = {}
        self._muap_grid_key: tuple | None = None
        self._muap_title_label = None

        self._build_ui()
        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _ts_to_plateau_local(self, timestamps: np.ndarray) -> np.ndarray:
        return timestamps - self._start_sample

    def _ts_to_absolute(self, plateau_local: np.ndarray) -> np.ndarray:
        return plateau_local + self._start_sample

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        self.quality_bar = MUPropertiesPanel()
        self.quality_bar.setMinimumHeight(110)
        self.quality_bar.reliability_toggled.connect(self._toggle_reliability)
        self.quality_bar.reliability_reset.connect(self._reset_reliability)
        root.addWidget(self.quality_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 1020])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"background-color: {COLORS.get('background_light', '#2a2a3c')}; "
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')}; "
            f"padding-right: 6px;"
        )
        self.status_bar.addPermanentWidget(self._file_label)
        root.addWidget(self.status_bar)
        self._update_status()

    # ── Style helpers ────────────────────────────────────────────────────

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox {{ background-color: {COLORS.get('background_input', '#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 4px; padding: 4px 8px; }}"
        )

    @staticmethod
    def _sel_btn_style(accent: str) -> str:
        """Checkable push-button style with a coloured 'on' state."""
        return (
            f"QPushButton {{"
            f"  color: {COLORS['foreground']};"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"  padding: 4px 10px;"
            f"  font-size: {FONT_SIZES.get('small', '9pt')};"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {COLORS.get('background_input', '#33334d')};"
            f"  border-color: {COLORS['border']};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background-color: {accent}30;"
            f"  border: 2px solid {accent};"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:disabled {{ color: {COLORS.get('text_dim', '#6c7086')}; }}"
        )

    @staticmethod
    def _make_warning_icon(size: int = 14) -> QIcon:
        """Render ⚠ in warning yellow to a QIcon, independent of button text colour."""
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        f = QFont()
        f.setPixelSize(size)
        p.setFont(f)
        p.setPen(QColor(COLORS["warning"]))
        p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, "⚠")
        p.end()
        return QIcon(px)

    @staticmethod
    def _warn_toolbar_btn_style() -> str:
        """Subtle toolbar button style — transparent background, default text colour."""
        fg = COLORS["foreground"]
        bg_hover = COLORS.get("background_input", "#1a1f24")
        fs = FONT_SIZES.get("small", "9pt")
        return (
            f"QPushButton {{"
            f"  color: {fg};"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"  padding: 4px 10px;"
            f"  font-size: {fs};"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {bg_hover};"
            f"  border-color: {COLORS['border']};"
            f"}}"
        )

    # ── Toolbar ──────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar()
        tb.setMovable(False)
        tb.setStyleSheet(
            f"""
            QToolBar {{
                background-color: {COLORS.get("background_light", "#2a2a3c")};
                border-bottom: 1px solid {COLORS["border"]};
                spacing: 4px;
                padding: 2px;
            }}
            QToolBar QLabel {{
                color: {COLORS["foreground"]};
                font-size: {FONT_SIZES.get("small", "9pt")};
            }}
            QToolButton {{
                color: {COLORS["foreground"]};
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {FONT_SIZES.get("small", "9pt")};
            }}
            QToolButton:hover {{
                background-color: {COLORS.get("background_input", "#33334d")};
                border-color: {COLORS["border"]};
            }}
            QToolButton:checked {{
                background-color: {COLORS.get("info", "#89b4fa")}30;
                border-color: {COLORS.get("info", "#89b4fa")};
            }}
        """
        )

        # ── File ──────────────────────────────────────────────────────
        self.action_load = QAction("📂 Load", self)
        self.action_load.triggered.connect(self._load_file_dialog)
        tb.addAction(self.action_load)

        self.action_save = QAction("💾 Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_file)
        tb.addAction(self.action_save)

        self.action_reset = QAction("⟲ Reset View", self)
        self.action_reset.setShortcut(QKeySequence("Home"))
        self.action_reset.triggered.connect(self._reset_view)
        tb.addAction(self.action_reset)

        # ── Rubberband-drag selection toggles ─────────────────────────
        tb.addSeparator()
        tb.addWidget(QLabel("  Select & "))

        success_color = COLORS.get("success", "#a6e3a1")
        error_color = COLORS.get("error", "#f38ba8")

        self.btn_sel_add = QPushButton("✅ Add in Selection")
        self.btn_sel_add.setCheckable(True)
        self.btn_sel_add.setChecked(False)
        self.btn_sel_add.setStyleSheet(self._sel_btn_style(success_color))
        self.btn_sel_add.setToolTip(
            "Toggle ON: drag a rectangle on the plot to add all peaks inside it.\n"
            "Stays armed — drag as many times as needed.\n"
            "Click again to turn off."
        )
        self.btn_sel_add.toggled.connect(self._on_sel_add_toggled)
        self.btn_sel_add.setEnabled(False)
        tb.addWidget(self.btn_sel_add)

        self.btn_sel_delete = QPushButton("🗑 Del in Selection")
        self.btn_sel_delete.setCheckable(True)
        self.btn_sel_delete.setChecked(False)
        self.btn_sel_delete.setStyleSheet(self._sel_btn_style(error_color))
        self.btn_sel_delete.setToolTip(
            "Toggle ON: drag a rectangle on the plot to delete all spikes inside it.\n"
            "Stays armed — drag as many times as needed.\n"
            "Click again to turn off."
        )
        self.btn_sel_delete.toggled.connect(self._on_sel_delete_toggled)
        self.btn_sel_delete.setEnabled(False)
        tb.addWidget(self.btn_sel_delete)

        # ── Undo / Redo ───────────────────────────────────────────────
        tb.addSeparator()
        self.action_undo = QAction("↩ Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.triggered.connect(self._undo)
        tb.addAction(self.action_undo)

        self.action_redo = QAction("↪ Redo", self)
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.triggered.connect(self._redo)
        tb.addAction(self.action_redo)

        # ── Per-MU editing ────────────────────────────────────────────
        tb.addSeparator()

        self.btn_recalc_filter = QPushButton("⟳ Recalc Filter")
        self.btn_recalc_filter.setShortcut(QKeySequence("F"))
        self.btn_recalc_filter.setToolTip(
            "Replay peel-off and recompute filter + source + timestamps [F]"
        )
        self.btn_recalc_filter.clicked.connect(self._recalculate_filter)
        self.btn_recalc_filter.setEnabled(False)
        self.btn_recalc_filter.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_recalc_filter)

        self.btn_flag_delete = QPushButton("⚑ Flag Unit")
        self.btn_flag_delete.setToolTip("Toggle deletion flag for the current MU [X]")
        self.btn_flag_delete.clicked.connect(self._toggle_flag_delete)
        self.btn_flag_delete.setEnabled(False)
        self.btn_flag_delete.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_flag_delete)

        self.btn_remove_outliers = QPushButton("⚡ Remove Outliers")
        self.btn_remove_outliers.setShortcut(QKeySequence("O"))
        self.btn_remove_outliers.setToolTip(
            "Remove spikes causing outlier instantaneous firing rate [O]\n"
            "Uses Tukey fence (Q75 + 1.5×IQR) on the IFR distribution."
        )
        self.btn_remove_outliers.clicked.connect(self._remove_outliers)
        self.btn_remove_outliers.setEnabled(False)
        self.btn_remove_outliers.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_remove_outliers)

        self.btn_auto_edit_mu = QPushButton("⚙ Auto-Edit MU")
        self.btn_auto_edit_mu.setShortcut(QKeySequence("E"))
        self.btn_auto_edit_mu.setToolTip(
            "Apply rule-based auto-editing to the current MU [E]\n"
            "Removes low/high-IPT spikes; adds missed spikes by FR/IPT criteria."
        )
        self.btn_auto_edit_mu.clicked.connect(self._run_auto_edit_current)
        self.btn_auto_edit_mu.setEnabled(False)
        self.btn_auto_edit_mu.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_auto_edit_mu)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.btn_notes = QPushButton("📝 Notes")
        self.btn_notes.setToolTip(
            "Open file notes; new entries are tagged with the current port and unit"
        )
        self.btn_notes.clicked.connect(self._open_notes_dialog)
        self.btn_notes.setEnabled(False)
        self.btn_notes.setStyleSheet(self._warn_toolbar_btn_style())
        tb.addWidget(self.btn_notes)

        return tb

    # ── Left panel ───────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {COLORS['background']};"
            f" border-right: 2px solid {COLORS['border']};"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        port_header = QLabel("PORT SELECTION")
        port_header.setStyleSheet(get_section_header_style("info"))
        lay.addWidget(port_header)
        port_row = QHBoxLayout()
        port_lbl = QLabel("Port:")
        port_lbl.setStyleSheet(
            f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        port_row.addWidget(port_lbl)
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(self._combo_style())
        self.port_combo.currentTextChanged.connect(self._on_port_changed)
        port_row.addWidget(self.port_combo, stretch=1)
        lay.addLayout(port_row)

        mu_header = QLabel("MOTOR UNIT")
        mu_header.setStyleSheet(get_section_header_style("info"))
        lay.addWidget(mu_header)
        mu_row = QHBoxLayout()
        mu_lbl = QLabel("Unit:")
        mu_lbl.setStyleSheet(
            f"color: {COLORS['foreground']}; font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        mu_row.addWidget(mu_lbl)
        self.mu_combo = QComboBox()
        self.mu_combo.setStyleSheet(self._combo_style())
        self.mu_combo.currentIndexChanged.connect(self._on_mu_selected)
        mu_row.addWidget(self.mu_combo, stretch=1)
        lay.addLayout(mu_row)

        # ── Button styles ────────────────────────────────────────────────
        _fs = FONT_SIZES.get("small", "9pt")
        _bg = COLORS.get("background_input", "#1a1f24")
        _fg = COLORS["foreground"]
        _border = COLORS["border"]
        _bg_hover = COLORS.get("background_hover", "#1e242b")

        base_btn_style = f"""
            QPushButton {{
                background-color: {_bg};
                color: {_fg};
                border: 1px solid {_border};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: {_fs};
            }}
            QPushButton:hover {{
                background-color: {_bg_hover};
                border-color: {COLORS.get("info", "#4a9eff")};
            }}
            QPushButton:disabled {{
                color: {COLORS.get("text_dim", "#6c7086")};
                border-color: {_border};
            }}"""

        review_row = QHBoxLayout()
        self.btn_reviewed = QPushButton("☐ Mark Reviewed")
        self.btn_reviewed.setCheckable(True)
        self.btn_reviewed.setEnabled(False)
        self.btn_reviewed.setStyleSheet(
            self._sel_btn_style(COLORS.get("success", "#a6e3a1"))
        )
        self.btn_reviewed.setToolTip(
            "Mark or unmark the current motor unit as manually reviewed [M]"
        )
        self.btn_reviewed.clicked.connect(self._toggle_reviewed)
        review_row.addWidget(self.btn_reviewed)

        self.btn_next_unreviewed = QPushButton("Next Unreviewed →")
        self.btn_next_unreviewed.setEnabled(False)
        self.btn_next_unreviewed.setStyleSheet(base_btn_style)
        self.btn_next_unreviewed.setToolTip(
            "Jump to the next unreviewed motor unit, including other ports [N]"
        )
        self.btn_next_unreviewed.clicked.connect(self._select_next_unreviewed)
        review_row.addWidget(self.btn_next_unreviewed)
        lay.addLayout(review_row)

        self.review_progress_label = QLabel("Reviewed 0/0")
        self.review_progress_label.setStyleSheet(
            f"color: {COLORS.get('text_dim', '#6c7086')}; "
            f"font-size: {FONT_SIZES.get('small', '9pt')};"
        )
        self.review_progress_label.setToolTip(
            "Manual review progress across all loaded motor units"
        )
        lay.addWidget(self.review_progress_label)

        # ── Session-level actions ────────────────────────────────────────
        session_header = QLabel("SESSION")
        session_header.setStyleSheet(get_section_header_style("warning", margin_top=0))
        lay.addWidget(session_header)

        self.btn_delete_flagged = QPushButton("🗑 Delete All Flagged Units")
        self.btn_delete_flagged.setStyleSheet(base_btn_style)
        self.btn_delete_flagged.setToolTip(
            "Permanently remove all flagged units from every port.\n"
            "The confirmation lists the number that will be removed per port."
        )
        self.btn_delete_flagged.clicked.connect(self._delete_all_flagged)
        self.btn_delete_flagged.setEnabled(False)
        lay.addWidget(self.btn_delete_flagged)

        self.btn_flag_within_dups = QPushButton("⧉ Check Duplicates in Current Port")
        self.btn_flag_within_dups.setStyleSheet(base_btn_style)
        self.btn_flag_within_dups.setToolTip(
            "Check every unit in the selected grid/probe, including flagged units,\n"
            "and suggest lower-priority duplicates for deletion.\n"
            "Uses rate-of-agreement (threshold 0.3) to identify duplicates."
        )
        self.btn_flag_within_dups.clicked.connect(self._flag_within_duplicates)
        self.btn_flag_within_dups.setEnabled(False)
        lay.addWidget(self.btn_flag_within_dups)

        self.btn_flag_cross_dups = QPushButton("⧉ Check Duplicates Across Ports")
        self.btn_flag_cross_dups.setStyleSheet(base_btn_style)
        self.btn_flag_cross_dups.setToolTip(
            "Optionally check every unit across different grids/probes, including\n"
            "flagged units, and suggest lower-priority duplicates for deletion.\n"
            "Uses rate-of-agreement (threshold 0.3) to identify duplicates."
        )
        self.btn_flag_cross_dups.clicked.connect(self._flag_cross_duplicates)
        self.btn_flag_cross_dups.setEnabled(False)
        lay.addWidget(self.btn_flag_cross_dups)

        self.muap_widget = pg.GraphicsLayoutWidget()
        self.muap_widget.setBackground(COLORS["background"])
        self.muap_widget.setStyleSheet(
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;"
        )
        lay.addWidget(self.muap_widget, stretch=1)

        self.btn_muap_popout = QPushButton("↗", self.muap_widget)
        self.btn_muap_popout.setToolTip("Pop out MUAP view")
        self.btn_muap_popout.setFixedSize(20, 20)
        self.btn_muap_popout.setFont(QFont("Arial, Helvetica", 10))
        self.btn_muap_popout.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.get('background_hover', '#1e242b')};"
            f" color: {COLORS.get('text_dim', '#6c7086')}; padding: 0px;"
            f" border: 1px solid {COLORS['border']}; border-radius: 4px; }}"
            f"QPushButton:hover {{ color: {COLORS['foreground']};"
            f" border-color: {COLORS.get('info', '#4a9eff')}; }}"
        )
        self.btn_muap_popout.clicked.connect(self._open_muap_popout)
        self.btn_muap_popout.raise_()
        self.muap_widget.installEventFilter(self)

        return panel

    # ── Right panel ──────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['background']};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.setHandleWidth(2)

        self.source_plot = SourcePlotWidget()
        self.source_plot.set_fsamp(self._fsamp)
        self.source_plot.spike_add_requested.connect(self._handle_add_click)
        self.source_plot.spike_delete_requested.connect(self._handle_delete_click)
        self.source_plot.region_selected.connect(self._on_region_selected)
        plot_splitter.addWidget(self.source_plot)

        self.fr_plot = FiringRatePlotWidget()
        self.fr_plot.set_fsamp(self._fsamp)
        self.fr_plot.link_x(self.source_plot)
        self.fr_plot.setMaximumHeight(150)
        plot_splitter.addWidget(self.fr_plot)
        plot_splitter.setSizes([500, 150])

        lay.addWidget(plot_splitter)
        return panel

    def eventFilter(self, obj, event):
        if obj is self.muap_widget and event.type() == QEvent.Type.Resize:
            self._reposition_muap_popout_btn()
        return super().eventFilter(obj, event)

    def _reposition_muap_popout_btn(self):
        btn = self.btn_muap_popout
        btn.move(self.muap_widget.width() - btn.width() - 6, 6)
        btn.raise_()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Up"), self, self._select_prev_mu)
        QShortcut(QKeySequence("Down"), self, self._select_next_mu)
        QShortcut(QKeySequence("Ctrl+Up"), self, self._select_prev_port)
        QShortcut(QKeySequence("Ctrl+Down"), self, self._select_next_port)
        QShortcut(QKeySequence("Left"), self, lambda: self._pan_source(-0.02))
        QShortcut(QKeySequence("Right"), self, lambda: self._pan_source(0.02))
        QShortcut(QKeySequence("A"), self, lambda: self.btn_sel_add.setChecked(True))
        QShortcut(QKeySequence("D"), self, lambda: self.btn_sel_delete.setChecked(True))
        QShortcut(QKeySequence("Escape"), self, lambda: self._set_mode(EditMode.VIEW))
        QShortcut(QKeySequence("X"), self, self.btn_flag_delete.click)
        QShortcut(QKeySequence("M"), self, self.btn_reviewed.click)
        QShortcut(QKeySequence("N"), self, self.btn_next_unreviewed.click)
        QShortcut(QKeySequence("T"), self, self._toggle_reliability)
        QShortcut(QKeySequence("Shift+T"), self, self._reset_reliability)

    def _pan_source(self, fraction: float):
        """Pan the source plot by `fraction` of the current visible width."""
        vb = self.source_plot.getViewBox()
        (x_min, x_max), _ = vb.viewRange()
        vb.translateBy(x=(x_max - x_min) * fraction, y=0)

    # ------------------------------------------------------------------
    # Sampling rate
    # ------------------------------------------------------------------

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp
        self.source_plot.set_fsamp(fsamp)
        self.fr_plot.set_fsamp(fsamp)

    def set_aux_configs(self, aux_configs: list):
        """Store the current config's aux channel definitions (including MVC) so they
        can be used to fill in missing MVC values when loading an old decomposition."""
        self._config_aux_channels = list(aux_configs)

    # ------------------------------------------------------------------
    # Selection arm toggling
    # ------------------------------------------------------------------

    def _on_sel_add_toggled(self, checked: bool):
        if checked:
            # Ensure the other button is off (mutual exclusion without QButtonGroup
            # so we keep independent checkable state)
            self.btn_sel_delete.blockSignals(True)
            self.btn_sel_delete.setChecked(False)
            self.btn_sel_delete.blockSignals(False)
            self._sel_arm = SelectionArm.ADD
        else:
            self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(self._sel_arm)
        self._update_status()

    def _on_sel_delete_toggled(self, checked: bool):
        if checked:
            self.btn_sel_add.blockSignals(True)
            self.btn_sel_add.setChecked(False)
            self.btn_sel_add.blockSignals(False)
            self._sel_arm = SelectionArm.DELETE
        else:
            self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(self._sel_arm)
        self._update_status()

    def _disarm_selection(self):
        """Programmatically turn off both selection toggles."""
        for btn in (self.btn_sel_add, self.btn_sel_delete):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._sel_arm = SelectionArm.NONE
        self.source_plot.set_selection_arm(SelectionArm.NONE)

    # ------------------------------------------------------------------
    # Region-selected slot — dispatches immediately
    # ------------------------------------------------------------------

    def _on_region_selected(self, x1: float, x2: float, y1: float, y2: float):
        if self._sel_arm == SelectionArm.ADD:
            self._apply_selection_add(x1, x2, y1, y2)
        elif self._sel_arm == SelectionArm.DELETE:
            self._apply_selection_delete(x1, x2, y1, y2)

    # ------------------------------------------------------------------
    # Selection operations
    # ------------------------------------------------------------------

    def _apply_selection_add(self, x1: float, x2: float, y1: float, y2: float):
        mu = self._current_mu()
        if mu is None:
            return
        s1 = max(0, int(x1 * self._fsamp))
        s2 = min(len(mu.source), int(x2 * self._fsamp))
        if s1 >= s2:
            return

        source_sq = np.nan_to_num(mu.source, nan=0.0, posinf=0.0, neginf=0.0) ** 2
        segment = source_sq[s1:s2]
        peaks, _ = sp_signal.find_peaks(
            segment, distance=max(1, int(0.005 * self._fsamp))
        )
        peaks_abs = peaks + s1
        existing = set(mu.timestamps.tolist())

        new_spikes = [
            int(p)
            for p in peaks_abs
            if 0 <= p < len(source_sq)
            and y1 <= source_sq[p] <= y2
            and int(p) not in existing
        ]
        if not new_spikes:
            self._update_status("No new peaks in selection")
            return

        old_ts = mu.timestamps.copy()
        new_ts = np.sort(
            np.concatenate([mu.timestamps, np.array(new_spikes, dtype=np.int64)])
        )
        self._push_undo(
            UndoAction(
                f"Sel-add {len(new_spikes)}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Added {len(new_spikes)} spikes from selection")

    def _apply_selection_delete(self, x1: float, x2: float, y1: float, y2: float):
        mu = self._current_mu()
        if mu is None:
            return
        s1 = int(x1 * self._fsamp)
        s2 = int(x2 * self._fsamp)
        source_sq = np.nan_to_num(mu.source, nan=0.0, posinf=0.0, neginf=0.0) ** 2

        in_box = np.array(
            [
                s1 <= ts < s2 and 0 <= ts < len(source_sq) and y1 <= source_sq[ts] <= y2
                for ts in mu.timestamps
            ],
            dtype=bool,
        )

        n_remove = int(np.sum(in_box))
        if n_remove == 0:
            self._update_status("No spikes in selection")
            return

        old_ts = mu.timestamps.copy()
        new_ts = mu.timestamps[~in_box]
        self._push_undo(
            UndoAction(
                f"Sel-del {n_remove}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted {n_remove} spikes from selection")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _update_file_label(self):
        if self._loaded_path:
            dirty_marker = " *" if self._dirty else ""
            self._file_label.setText(
                f"📄 Current File: {self._loaded_path.name}{dirty_marker}"
            )
        else:
            self._file_label.setText("")

    @property
    def is_dirty(self) -> bool:
        """Whether the current edition contains changes not written to disk."""
        return self._dirty

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self._update_file_label()

    def confirm_save_changes(self, action: str = "continuing") -> bool:
        """Offer to save dirty data, returning whether *action* may proceed."""
        if not self._dirty:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Save changes before {action}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_file()
        return reply == QMessageBox.StandardButton.Discard

    def get_visualisation_data(self) -> dict:
        """Return a snapshot of all data needed by the Visualisation tab."""
        return {
            "ports": self._ports,
            "aux_channels": (self._original_decomp_data or {}).get("aux_channels")
            or [],
            "fsamp": self._fsamp,
            "start_sample": self._start_sample,
            "end_sample": self._end_sample,
            "timestamps_are_absolute": self._full_source_mode,
            "file_stem": self._loaded_path.stem if self._loaded_path else "",
        }

    def _capture_session_state(self) -> dict:
        attributes = (
            "_fsamp",
            "_ports",
            "_emg_data",
            "_grid_info",
            "_raw_port_channels",
            "_rejected_ch_positions",
            "_notes",
            "_current_port",
            "_current_mu_idx",
            "_edit_mode",
            "_sel_arm",
            "_loaded_path",
            "_dirty",
            "_start_sample",
            "_end_sample",
            "_full_source_mode",
            "_redetect_timestamps",
            "_undo_stack",
            "_redo_stack",
            "_edit_history",
            "_original_decomp_data",
            "_filter_recalc_available",
            "_last_action_msg",
            "_pending_source_changed",
            "_pending_props_key",
        )
        controls = (
            "btn_recalc_filter",
            "btn_remove_outliers",
            "btn_flag_delete",
            "btn_delete_flagged",
            "btn_auto_edit_mu",
            "btn_sel_add",
            "btn_sel_delete",
            "btn_flag_within_dups",
            "btn_flag_cross_dups",
            "btn_notes",
            "btn_reviewed",
            "btn_next_unreviewed",
        )
        return {
            "attributes": {name: getattr(self, name) for name in attributes},
            "controls": {
                name: (getattr(self, name).isEnabled(), getattr(self, name).toolTip())
                for name in controls
            },
            "props_timer_remaining": (
                self._props_timer.remainingTime()
                if self._props_timer.isActive()
                else -1
            ),
        }

    def _restore_session_state(self, state: dict) -> None:
        self._props_timer.stop()
        for name, value in state["attributes"].items():
            setattr(self, name, value)

        self._refresh_port_combo()
        self.port_combo.blockSignals(True)
        self.port_combo.setCurrentText(self._current_port or "")
        self.port_combo.blockSignals(False)
        self._refresh_mu_combo()
        self.mu_combo.blockSignals(True)
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self.mu_combo.blockSignals(False)
        if self._current_mu() is None:
            self._clear_plots()
        else:
            self._update_plots(reset_view=True)
        self._refresh_aux_controls()

        self.source_plot.set_edit_mode(self._edit_mode)
        self.source_plot.set_selection_arm(self._sel_arm)
        self.btn_sel_add.blockSignals(True)
        self.btn_sel_delete.blockSignals(True)
        self.btn_sel_add.setChecked(self._sel_arm == SelectionArm.ADD)
        self.btn_sel_delete.setChecked(self._sel_arm == SelectionArm.DELETE)
        self.btn_sel_add.blockSignals(False)
        self.btn_sel_delete.blockSignals(False)

        for name, (enabled, tooltip) in state["controls"].items():
            control = getattr(self, name)
            control.setEnabled(enabled)
            control.setToolTip(tooltip)
        self._update_review_controls()
        self._update_file_label()
        self._update_status()

        remaining = state["props_timer_remaining"]
        if remaining >= 0 and self._pending_props_key is not None:
            self._props_timer.start(max(1, remaining))

    def load_from_path(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            QMessageBox.critical(self, "Load Error", f"File not found:\n{path}")
            return False
        if not self.confirm_save_changes("opening another file"):
            return False
        try:
            data = load_decomposition_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{e}")
            return False
        if data.get("skip_filter_recalc"):
            reply = QMessageBox.question(
                self,
                "Recalculate Filters?",
                "This file was previously edited.\n\nDo you want to recalculate the filters for each motor unit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                data["skip_filter_recalc"] = False

        can_full, _ = supports_full_source_computation(data)
        if can_full and not data.get("skip_filter_recalc"):
            pts = data.get("plateau_coords", data.get("selected_points"))
            raw = data.get("data")
            full_len = (
                raw.shape[1]
                if raw is not None and hasattr(raw, "shape") and raw.ndim >= 2
                else None
            )
            is_partial_section = (
                pts is not None
                and full_len is not None
                and (int(pts[0]) > 0 or int(pts[1]) < full_len)
            )
            if is_partial_section:
                reply = QMessageBox.question(
                    self,
                    "Recalculate Timestamps?",
                    "Do you want to recalculate spike timestamps on the full signal?\n\n"
                    "Yes — re-detect timestamps from the source over the entire recording.\n"
                    "No  — keep the original timestamps from the decomposed section only.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                redetect_timestamps = reply == QMessageBox.StandardButton.Yes
            else:
                redetect_timestamps = True
        else:
            redetect_timestamps = True

        previous_session = self._capture_session_state()
        try:
            # Set this before loading so _refresh_aux_controls sees the correct stem.
            self._loaded_path = path
            self._redetect_timestamps = redetect_timestamps
            self._load_decomposition_data(data)
            imported_format = data.get("import_provenance", {}).get("format")
            if imported_format:
                self._update_status(
                    f"Loaded: {path.name} (converted from {imported_format})"
                )
            else:
                self._update_status(f"Loaded: {path.name}")
            self._set_dirty(False)
            self._update_file_label()
            self.file_loaded.emit()
            return True
        except Exception as e:
            logger.exception("Failed to parse decomposition file %s", path)
            self._restore_session_state(previous_session)
            QMessageBox.critical(self, "Load Error", f"Failed to parse:\n{e}")
            return False

    def _load_decomposition_data(self, decomp_data: dict):
        self._props_timer.stop()
        self._pending_props_key = None
        self._pending_source_changed = False
        self._disarm_selection()
        # Allocate new containers so a failed load can restore the previous
        # session by reference without retaining partially parsed data.
        self._ports = {}
        self._emg_data = {}
        self._grid_info = {}
        self._notes = []
        self._raw_port_channels = {}
        self._rejected_ch_positions = {}
        self._undo_stack = {}
        self._redo_stack = {}
        # Carry forward any edit history already stored in the file so the log
        # accumulates across multiple editing sessions.
        prior_history = decomp_data.get("edit_history", [])
        if isinstance(prior_history, list):
            self._edit_history = prior_history
        else:
            self._edit_history = []

        self._notes = normalise_notes(decomp_data.get("notes"))
        acquisition_format = decomp_data.get("acquisition_metadata", {}).get("format")
        normalise_aux_channels(
            decomp_data.get("aux_channels", []),
            acquisition_format=acquisition_format,
            configured_channels=self._config_aux_channels,
        )

        self._original_decomp_data = decomp_data

        fsamp = decomp_data.get("sampling_rate", decomp_data.get("fsamp", self._fsamp))
        self.set_fsamp(float(fsamp))

        raw_data = decomp_data.get("data")
        emg_full = None
        if raw_data is not None:
            emg_full = to_numpy(raw_data)
            if emg_full.ndim == 2 and emg_full.shape[0] > emg_full.shape[1]:
                emg_full = emg_full.T

        skip_recalc = bool(decomp_data.get("skip_filter_recalc", False))
        can_full, reason = supports_full_source_computation(decomp_data)
        full_port_results: dict[int, list] = {}
        start_sample = 0
        end_sample = emg_full.shape[1] if emg_full is not None else 0

        if skip_recalc:
            logger.info(
                "skip_filter_recalc=True — using stored sources/timestamps as-is"
            )
            can_full = False
        elif can_full:
            try:
                self._update_status("Computing full-length sources (peel-off replay)…")
                (
                    full_port_results,
                    start_sample,
                    end_sample,
                    err,
                ) = compute_all_full_sources(
                    decomp_data, redetect_timestamps=self._redetect_timestamps
                )
                if err:
                    logger.warning("Full source warning: %s", err)
                    full_port_results = {}
            except Exception as e:
                logger.exception("Full source computation failed: %s", e)
                full_port_results = {}
        else:
            sel_pts = decomp_data.get(
                "plateau_coords", decomp_data.get("selected_points")
            )
            if sel_pts is not None:
                try:
                    pts = to_numpy(np.asarray(sel_pts)).flatten()
                    start_sample, end_sample = int(pts[0]), int(pts[1])
                except (IndexError, TypeError, ValueError):
                    pass
            if end_sample <= start_sample and emg_full is not None:
                end_sample = emg_full.shape[1]

        self._start_sample = start_sample
        self._end_sample = end_sample
        self._full_source_mode = bool(full_port_results)

        if self._full_source_mode:
            logger.info(
                "Full-length source mode active (peel-off + source_to_timestamps)"
            )
        else:
            logger.info(
                "Plateau-only source mode%s", f" — {reason}" if not can_full else ""
            )

        ports = decomp_data.get("ports", [])
        ch_offset = 0
        for port_idx, port_name in enumerate(ports):
            ch_offset += self._load_single_port(
                port_idx,
                port_name,
                decomp_data,
                emg_full,
                start_sample,
                end_sample,
                full_port_results,
                ch_offset,
            )

        self._refresh_port_combo()
        if ports:
            self.port_combo.setCurrentText(ports[0])
            self._on_port_changed(ports[0])
        self._reset_view_full()

        if self._full_source_mode:
            self.source_plot.set_plateau_region(start_sample, end_sample)

        self._refresh_aux_controls()

        ok, reason = supports_filter_recalculation(decomp_data)
        self._filter_recalc_available = ok
        self.btn_recalc_filter.setEnabled(ok)
        if not ok:
            self.btn_recalc_filter.setToolTip(f"Unavailable: {reason}")

        for btn in (
            self.btn_remove_outliers,
            self.btn_flag_delete,
            self.btn_delete_flagged,
            self.btn_auto_edit_mu,
            self.btn_sel_add,
            self.btn_sel_delete,
            self.btn_flag_within_dups,
            self.btn_flag_cross_dups,
        ):
            btn.setEnabled(True)

        all_mus_have_props = all(
            mu.props is not None for mus in self._ports.values() for mu in mus
        )
        if not all_mus_have_props:
            _no_props_tip = "Compute MU properties first"
            self.btn_flag_within_dups.setEnabled(False)
            self.btn_flag_within_dups.setToolTip(_no_props_tip)
            self.btn_flag_cross_dups.setEnabled(False)
            self.btn_flag_cross_dups.setToolTip(_no_props_tip)

        all_mus = [mu for mus in self._ports.values() for mu in mus]
        n_total = len(all_mus)
        n_reliable = sum(
            1 for mu in all_mus if mu.props is not None and mu.props.is_reliable
        )
        n_unreliable = n_total - n_reliable
        logger.info(
            "Reliability: %d/%d reliable, %d/%d not reliable",
            n_reliable,
            n_total,
            n_unreliable,
            n_total,
        )
        self._update_status(
            f"Loaded {n_total} MUs — "
            f"{n_reliable} reliable  |  {n_unreliable} not reliable"
        )

    def _load_single_port(
        self,
        port_idx: int,
        port_name: str,
        decomp_data: dict,
        emg_full,
        start_sample: int,
        end_sample: int,
        full_port_results: dict,
        ch_offset: int,
    ) -> int:
        loaded = load_edition_port(
            port_index=port_idx,
            port_name=port_name,
            decomposition=decomp_data,
            emg_full=emg_full,
            start_sample=start_sample,
            end_sample=end_sample,
            full_port_results=full_port_results,
            channel_offset=ch_offset,
            full_source_mode=self._full_source_mode,
            sampling_rate=self._fsamp,
            existing_notes=self._notes,
            property_computer=compute_port_properties,
        )

        self._ports[port_name] = loaded.motor_units
        self._grid_info[port_name] = loaded.grid_config
        self._rejected_ch_positions[port_name] = loaded.rejected_channel_positions
        self._notes.extend(loaded.migrated_notes)
        if loaded.raw_channels is not None:
            self._raw_port_channels[port_name] = loaded.raw_channels
        if loaded.emg is not None:
            self._emg_data[port_name] = loaded.emg
        return loaded.channel_count

    def _load_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Decomposition", "", "Pickle (*.pkl);;All (*)"
        )
        if path:
            self.load_from_path(Path(path))

    def set_output_path(self, path: Path):
        """Set a fixed output path so Ctrl+S saves without a dialog."""
        self._output_path = Path(path)

    def set_quit_after_save(self, enabled: bool):
        """Close the application after every successful save while enabled."""
        self._quit_after_save = enabled

    def _save_file(self) -> bool:
        if not self._ports:
            self._update_status("Nothing to save")
            return False

        # Persist derived quality metrics for the latest spike edit even when
        # the user saves inside the short UI debounce window.
        if self._pending_props_key is not None:
            self._props_timer.stop()
            self._flush_props_update()

        if self._output_path:
            save_path = self._output_path
        else:
            default = ""
            if self._loaded_path:
                default = str(
                    self._loaded_path.with_name(self._loaded_path.stem + "_edited.pkl")
                )
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save Decomposition", default, "Pickle (*.pkl)"
            )
            if not chosen:
                return False
            save_path = Path(chosen)

        try:
            save_data = self._build_save_dict()
            atomic_pickle_dump(save_data, save_path)
            try:
                audit_path = write_audit_report(
                    save_path,
                    save_data,
                    operation="edition",
                    derived_from=self._loaded_path,
                )
                status = f"Saved: {save_path.name} + {audit_path.name}"
            except Exception:
                logger.exception(
                    "Edition was saved, but its audit report could not be written"
                )
                status = f"Saved: {save_path.name} (audit report not written)"
            self._set_dirty(False)
            self._update_status(status)
            self._update_file_label()
            if self._quit_after_save:
                QApplication.quit()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False

    def _build_save_dict(self) -> dict:
        state = EditionSaveState(
            ports=self._ports,
            sampling_rate=self._fsamp,
            full_source_mode=self._full_source_mode,
            start_sample=self._start_sample,
            end_sample=self._end_sample,
            edit_history=self._edit_history,
            notes=self._notes,
            original_decomposition=self._original_decomp_data,
            emg_data=self._emg_data,
        )
        return build_edition_save_data(state)

    # ------------------------------------------------------------------
    # Edit mode (point-click)
    # ------------------------------------------------------------------

    def _set_mode(self, mode: EditMode):
        self._edit_mode = mode
        self.source_plot.set_edit_mode(mode)
        # Entering view mode disarms the rubberband selection
        self._disarm_selection()
        self._update_status()

    def _reset_view_full(self):
        """Reset both plots to show the entire signal length."""
        mu = self._current_mu()
        if mu is None or len(mu.source) == 0:
            return
        x_max = len(mu.source) / self._fsamp
        # Source plot: set X and Y explicitly.  Any autoRange() call (even on
        # the linked FR plot) feeds spike-marker bounds back through the X link
        # and re-clips the range to the plateau region, so we avoid it entirely.
        src_sq = np.nan_to_num(mu.source**2)
        window_sq = (
            src_sq[self._start_sample : self._end_sample]
            if self._full_source_mode
            else src_sq
        )
        y_max = float(np.max(window_sq)) if window_sq.size else 1.0
        y_pad = y_max * 0.05
        self.source_plot.getViewBox().setRange(
            xRange=(0, x_max), yRange=(-y_pad, y_max + y_pad), padding=0
        )
        # FR plot X follows via the link; reset Y to the full data range.
        self.fr_plot.reset_y_range()

    def _reset_view(self):
        self._reset_view_full()
        self._update_status("View reset")

    # ------------------------------------------------------------------
    # Point-click spike editing
    # ------------------------------------------------------------------

    def _handle_add_click(self, sample: int):
        if self._edit_mode != EditMode.ADD:
            return
        mu = self._current_mu()
        if mu is None:
            return
        if sample < 0 or sample >= len(mu.source):
            self._update_status("Click outside source range")
            return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end = min(len(mu.source), int(view_x[1] * self._fsamp))
        peak = self._find_nearest_peak(mu.source**2, sample, view_start, view_end)
        if peak is None:
            self._update_status("No peak found near click")
            return
        if peak in mu.timestamps:
            self._update_status("Spike already exists")
            return
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.append(mu.timestamps, peak)).astype(np.int64)
        self._push_undo(
            UndoAction(
                "Add spike",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Added spike at {peak / self._fsamp:.3f}s")

    def _handle_delete_click(self, sample: int):
        if self._edit_mode != EditMode.DELETE:
            return
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) == 0:
            return
        view_x = self.source_plot.getViewBox().viewRange()[0]
        view_start = max(0, int(view_x[0] * self._fsamp))
        view_end = min(len(mu.source), int(view_x[1] * self._fsamp))
        visible_mask = (mu.timestamps >= view_start) & (mu.timestamps < view_end)
        visible_ts = mu.timestamps[visible_mask]
        if len(visible_ts) == 0:
            self._update_status("No spikes in view")
            return
        nearest = np.argmin(np.abs(visible_ts - sample))
        target = visible_ts[nearest]
        global_idx = np.where(mu.timestamps == target)[0][0]
        old_ts = mu.timestamps.copy()
        new_ts = np.delete(mu.timestamps, global_idx)
        self._push_undo(
            UndoAction(
                "Delete spike",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(f"Deleted spike at {target / self._fsamp:.3f}s")

    def _find_nearest_peak(
        self, source, click_sample: int, view_start: int, view_end: int
    ) -> int | None:
        if view_start >= view_end:
            return None
        view_width = view_end - view_start
        half_window = max(int(0.005 * self._fsamp), int(0.05 * view_width))
        search_start = max(view_start, click_sample - half_window)
        search_end = min(view_end, click_sample + half_window)
        if search_start >= search_end:
            return None
        segment = source[search_start:search_end]
        peaks, _ = sp_signal.find_peaks(
            segment, distance=max(1, int(0.005 * self._fsamp))
        )
        if len(peaks) == 0:
            return int(search_start + np.argmax(segment))
        peaks_abs = peaks + search_start
        return int(peaks_abs[np.argmin(np.abs(peaks_abs - click_sample))])

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Edit history logging
    # ------------------------------------------------------------------

    def _log_event(
        self, event_type: str, action: str, port_name: str, mu_idx: int, **extra
    ):
        """Append one entry to _edit_history for any kind of event.

        event_type: "edit" | "undo" | "redo" | "flag" | "unflag" |
                    "delete_flagged" | "flag_within_duplicates" |
                    "flag_cross_duplicates" | "reliability_override"
        extra: any additional serialisable fields to include in the record.
        """
        self._edit_history.append(
            {
                "datetime": datetime.now().isoformat(),
                "event_type": event_type,
                "action": action,
                "port_name": port_name,
                "mu_idx": mu_idx,
                "fsamp": self._fsamp,
                **extra,
            }
        )

    def _log_edit(self, event_type: str, action: UndoAction):
        """Append one entry to _edit_history for an undo-tracked spike edit.

        event_type: "edit" | "undo" | "redo"
        Timestamps are stored as plain Python lists so the history is
        serialisable without numpy and easy to consume for ML training.
        """

        def _to_list(arr):
            if arr is None:
                return None
            return np.asarray(arr).flatten().tolist()

        if event_type in ("edit", "redo"):
            ts_before = _to_list(action.old_timestamps)
            ts_after = _to_list(action.new_timestamps)
        else:  # undo
            ts_before = _to_list(action.new_timestamps)
            ts_after = _to_list(action.old_timestamps)

        added, removed = [], []
        if ts_before is not None and ts_after is not None:
            before_set = set(ts_before)
            after_set = set(ts_after)
            added = sorted(after_set - before_set)
            removed = sorted(before_set - after_set)

        self._log_event(
            event_type,
            action.description,
            action.port_name,
            action.mu_idx,
            samples_added=added,
            samples_removed=removed,
        )

    # ------------------------------------------------------------------

    def _push_undo(self, action: UndoAction):
        key = (action.port_name, action.mu_idx)
        stack = self._undo_stack.setdefault(key, [])
        stack.append(action)
        self._redo_stack.pop(key, None)  # new edit clears redo for this unit only
        if len(stack) > 100:
            stack.pop(0)
        self._log_edit("edit", action)

    def _undo(self):
        key = (self._current_port or "", self._current_mu_idx)
        stack = self._undo_stack.get(key, [])
        if not stack:
            self._update_status("Nothing to undo")
            return
        action = stack.pop()
        self._redo_stack.setdefault(key, []).append(action)
        self._log_edit("undo", action)
        self._apply_undo_redo(action, is_undo=True)
        self._on_data_changed(
            f"Undo: {action.description}",
            source_changed=action.old_source is not None,
        )

    def _redo(self):
        key = (self._current_port or "", self._current_mu_idx)
        stack = self._redo_stack.get(key, [])
        if not stack:
            self._update_status("Nothing to redo")
            return
        action = stack.pop()
        self._undo_stack.setdefault(key, []).append(action)
        self._log_edit("redo", action)
        self._apply_undo_redo(action, is_undo=False)
        self._on_data_changed(
            f"Redo: {action.description}",
            source_changed=action.new_source is not None,
        )

    def _apply_undo_redo(self, action: UndoAction, is_undo: bool):
        if self._current_port != action.port_name:
            self.port_combo.setCurrentText(action.port_name)
        if self._current_mu_idx != action.mu_idx:
            self.mu_combo.setCurrentIndex(action.mu_idx)
        mu = self._get_mu(action.port_name, action.mu_idx)
        if mu is None:
            return
        if is_undo:
            if action.old_timestamps is not None:
                mu.timestamps = action.old_timestamps
            if action.old_source is not None:
                mu.source = action.old_source
            if action.old_filter is not None:
                mu.mu_filter = action.old_filter
        else:
            if action.new_timestamps is not None:
                mu.timestamps = action.new_timestamps
            if action.new_source is not None:
                mu.source = action.new_source
            if action.new_filter is not None:
                mu.mu_filter = action.new_filter

    # ------------------------------------------------------------------
    # Filter recalculation
    # ------------------------------------------------------------------

    def _recalculate_filter(self):
        mu = self._current_mu()
        if mu is None:
            self._update_status("Select a motor unit first")
            return
        if not self._filter_recalc_available:
            QMessageBox.warning(
                self,
                "Unavailable",
                "Filter recalculation requires peel_off_sequence in the file.",
            )
            return
        if len(mu.timestamps) < 2:
            self._update_status("Need at least 2 spikes")
            return

        raw_port = self._raw_port_channels.get(self._current_port)
        if raw_port is None:
            QMessageBox.warning(self, "Missing Data", "Raw port EMG not available.")
            return

        port_idx = list(self._ports.keys()).index(self._current_port)
        global_idx = self._global_unit_idx(self._current_port, self._current_mu_idx)
        if global_idx is None:
            self._update_status("Could not determine global unit index")
            return

        current_port_filters = [m.mu_filter for m in self._ports[self._current_port]]
        ts_abs = mu.timestamps
        if not self._full_source_mode:
            ts_abs = self._ts_to_absolute(mu.timestamps)

        self._update_status(f"Recalculating filter for MU {mu.id}…")
        try:
            new_filter, new_source_full, new_ts_abs = recalculate_unit_filter(
                raw_port_channels=raw_port,
                decomp_data=self._original_decomp_data,
                port_idx=port_idx,
                local_mu_idx=self._current_mu_idx,
                edited_timestamps_abs=ts_abs,
                global_unit_idx=global_idx,
                start_sample=self._start_sample,
                end_sample=self._end_sample,
                current_port_filters=current_port_filters,
            )

            new_timestamps = (
                new_ts_abs
                if self._full_source_mode
                else self._ts_to_plateau_local(new_ts_abs)
            )

            old_source = mu.source.copy()
            old_filter = mu.mu_filter.copy() if mu.mu_filter is not None else None
            old_timestamps = mu.timestamps.copy()

            mu.source = new_source_full
            mu.mu_filter = new_filter
            mu.timestamps = new_timestamps

            self._push_undo(
                UndoAction(
                    description=f"Recalculate filter MU {mu.id}",
                    port_name=self._current_port,
                    mu_idx=self._current_mu_idx,
                    old_timestamps=old_timestamps,
                    new_timestamps=new_timestamps,
                    old_source=old_source,
                    old_filter=old_filter,
                    new_source=new_source_full,
                    new_filter=new_filter,
                )
            )
            self._on_data_changed(
                f"Filter recalculated for MU {mu.id}",
                source_changed=True,
            )

        except Exception as e:
            logger.exception("Filter recalculation failed")
            QMessageBox.critical(self, "Recalculation Error", str(e))
            self._update_status("Filter recalculation failed")

    def _global_unit_idx(self, port_name: str, mu_idx: int) -> int | None:
        offset = 0
        for pname, mus in self._ports.items():
            if pname == port_name:
                return offset + mu_idx
            offset += len(mus)
        return None

    # ------------------------------------------------------------------
    # Motor-unit accessors
    # ------------------------------------------------------------------

    def _current_mu(self) -> MotorUnit | None:
        return self._get_mu(self._current_port, self._current_mu_idx)

    def _get_mu(self, port, idx) -> MotorUnit | None:
        if port is None or idx < 0:
            return None
        mus = self._ports.get(port, [])
        return mus[idx] if 0 <= idx < len(mus) else None

    # ------------------------------------------------------------------
    # UI callbacks — port / MU selection
    # ------------------------------------------------------------------

    def _on_mu_selected(self, index: int):
        if self._current_port is None:
            return
        mus = self._ports.get(self._current_port, [])
        if index < 0 or index >= len(mus):
            return
        self._current_mu_idx = index
        mu = self._current_mu()
        if mu:
            self.btn_flag_delete.setText(
                "Unflag Unit" if mu.flagged_for_deletion else "⚑ Flag Unit"
            )
        self._update_review_controls()
        self._update_plots(reset_view=True)
        self._update_status()

    def _select_prev_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx > 0:
            self.mu_combo.setCurrentIndex(idx - 1)

    def _select_next_mu(self):
        idx = self.mu_combo.currentIndex()
        if idx < self.mu_combo.count() - 1:
            self.mu_combo.setCurrentIndex(idx + 1)

    def _select_prev_port(self):
        idx = self.port_combo.currentIndex()
        if idx > 0:
            self.port_combo.setCurrentIndex(idx - 1)

    def _select_next_port(self):
        idx = self.port_combo.currentIndex()
        if idx < self.port_combo.count() - 1:
            self.port_combo.setCurrentIndex(idx + 1)

    def _review_counts(self) -> tuple[int, int]:
        total = sum(len(motor_units) for motor_units in self._ports.values())
        reviewed = sum(
            1
            for motor_units in self._ports.values()
            for motor_unit in motor_units
            if motor_unit.reviewed
        )
        return reviewed, total

    def _update_review_controls(self) -> None:
        motor_unit = self._current_mu()
        reviewed, total = self._review_counts()
        all_reviewed = total > 0 and reviewed == total

        self.btn_reviewed.blockSignals(True)
        self.btn_reviewed.setChecked(
            motor_unit.reviewed if motor_unit is not None else False
        )
        self.btn_reviewed.setText(
            "☑ Reviewed"
            if motor_unit is not None and motor_unit.reviewed
            else "☐ Mark Reviewed"
        )
        self.btn_reviewed.setEnabled(motor_unit is not None)
        self.btn_reviewed.blockSignals(False)

        self.btn_next_unreviewed.setEnabled(total > reviewed)
        self.review_progress_label.setText(f"Reviewed {reviewed}/{total}")
        progress_color = (
            COLORS.get("success", "#a6e3a1")
            if all_reviewed
            else COLORS.get("text_dim", "#6c7086")
        )
        self.review_progress_label.setStyleSheet(
            f"color: {progress_color}; font-size: {FONT_SIZES.get('small', '9pt')};"
        )

    def _toggle_reviewed(self) -> None:
        motor_unit = self._current_mu()
        if motor_unit is None:
            return

        motor_unit.reviewed = not motor_unit.reviewed
        action = "reviewed" if motor_unit.reviewed else "unreviewed"
        self._log_event(
            "review_status",
            f"marked MU {motor_unit.id} as {action}",
            self._current_port or "",
            self._current_mu_idx,
        )
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_review_controls()
        self._update_status(f"MU {motor_unit.id} marked {action.upper()}")
        self._mark_modified()

    def _select_next_unreviewed(self) -> None:
        positions = [
            (port_name, unit_index)
            for port_name, motor_units in self._ports.items()
            for unit_index in range(len(motor_units))
        ]
        if not positions:
            self._update_status("No motor units are loaded")
            return

        unreviewed = {
            (port_name, unit_index)
            for port_name, unit_index in positions
            if not self._ports[port_name][unit_index].reviewed
        }
        if not unreviewed:
            self._update_review_controls()
            self._update_status("All motor units have been reviewed")
            return

        current = (self._current_port, self._current_mu_idx)
        start_index = positions.index(current) if current in positions else -1
        target = next(
            positions[(start_index + step) % len(positions)]
            for step in range(1, len(positions) + 1)
            if positions[(start_index + step) % len(positions)] in unreviewed
        )
        target_port, target_index = target
        if target == current:
            self._update_status("Current MU is the only unreviewed unit")
            return

        if target_port != self._current_port:
            self.port_combo.setCurrentText(target_port)
        self.mu_combo.setCurrentIndex(target_index)
        target_unit = self._ports[target_port][target_index]
        self._update_status(f"Next unreviewed: {target_port}, MU {target_unit.id}")

    def _toggle_flag_delete(self):
        mu = self._current_mu()
        if mu is None:
            return
        was_flagged = mu.flagged_for_deletion
        if was_flagged:
            mu.flagged_duplicate = False
            if mu.within_duplicate_role == "delete":
                mu.within_duplicate_role = "keep"
            if mu.cross_duplicate_role == "delete":
                mu.cross_duplicate_role = "keep"
        else:
            mu.flagged_duplicate = True
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(mu)
        event = "unflag" if was_flagged else "flag"
        self._log_event(
            event,
            f"{event} MU for deletion",
            self._current_port or "",
            self._current_mu_idx,
        )
        self._update_status(
            f"MU {mu.id} {'flagged' if mu.flagged_for_deletion else 'unflagged'}"
        )
        self._mark_modified()

    # ------------------------------------------------------------------
    # Reliability override
    # ------------------------------------------------------------------

    def _toggle_reliability(self):
        """Flip the current MU's reliability verdict.

        Flipping to a value that matches the automatic verdict clears the
        override, so the unit goes back to tracking its quality metrics.
        """
        mu = self._current_mu()
        if mu is None or mu.props is None:
            self._update_status("No quality data for this MU")
            return

        wanted = not mu.props.is_reliable
        if wanted == mu.props.auto_reliable:
            mu.props.reliability_override = None
            how = "automatic"
        else:
            mu.props.reliability_override = wanted
            how = "manual"

        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(mu)
        self._log_event(
            "reliability_override",
            f"set MU {mu.id} to {'reliable' if wanted else 'unreliable'} ({how})",
            self._current_port or "",
            self._current_mu_idx,
        )
        self._update_status(
            f"MU {mu.id} marked {'RELIABLE' if wanted else 'UNRELIABLE'} ({how})"
        )
        self._mark_modified()

    def _reset_reliability(self):
        """Drop the current MU's manual override and follow the metrics again."""
        mu = self._current_mu()
        if mu is None or mu.props is None:
            return
        if not mu.props.reliability_is_overridden:
            self._update_status("MU reliability already follows the quality metrics")
            return

        mu.props.reliability_override = None
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(mu)
        self._log_event(
            "reliability_override",
            f"cleared manual reliability for MU {mu.id}",
            self._current_port or "",
            self._current_mu_idx,
        )
        self._update_status(
            f"MU {mu.id} reliability back to automatic: "
            f"{'RELIABLE' if mu.props.is_reliable else 'UNRELIABLE'}"
        )
        self._mark_modified()

    def _delete_all_flagged(self):
        ports = list(self._ports.keys())
        flagged_by_port = {
            port_name: [
                motor_unit.id
                for motor_unit in self._ports[port_name]
                if motor_unit.flagged_for_deletion
            ]
            for port_name in ports
        }
        total = sum(len(unit_ids) for unit_ids in flagged_by_port.values())
        if total == 0:
            self._update_status("No MUs are flagged for deletion")
            return

        counts = "\n".join(
            f"• {port_name}: {len(unit_ids)}"
            for port_name, unit_ids in flagged_by_port.items()
            if unit_ids
        )

        reply = QMessageBox.question(
            self,
            "Delete Flagged MUs",
            f"Permanently delete {total} flagged MU(s) from the current session?\n\n"
            f"{counts}\n\nThis cannot be undone and clears the undo/redo history.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remap peel_off_sequence before deletion so filter recalculation
        # continues to work correctly on the remaining units.
        if self._original_decomp_data is not None:
            orig_peel = self._original_decomp_data.get("peel_off_sequence")
            if orig_peel is not None:
                port_index_maps = {}
                for port_name in ports:
                    old_to_new = {}
                    new_idx = 0
                    for old_idx, mu in enumerate(self._ports[port_name]):
                        if not mu.flagged_for_deletion:
                            old_to_new[old_idx] = new_idx
                            new_idx += 1
                    port_index_maps[port_name] = old_to_new

                new_peel = []
                for port_idx, port_name in enumerate(ports):
                    port_seq = orig_peel[port_idx] if port_idx < len(orig_peel) else []
                    old_to_new = port_index_maps[port_name]
                    remapped = []
                    for entry in port_seq:
                        uid = entry.get("accepted_unit_idx")
                        if uid is None:
                            remapped.append(entry)
                        elif uid in old_to_new:
                            remapped.append(
                                {**entry, "accepted_unit_idx": old_to_new[uid]}
                            )
                    new_peel.append(remapped)
                self._original_decomp_data["peel_off_sequence"] = new_peel

        deleted_by_port = flagged_by_port
        for port_name in ports:
            kept = [mu for mu in self._ports[port_name] if not mu.flagged_for_deletion]
            for i, mu in enumerate(kept):
                mu.id = i
            self._ports[port_name] = kept

        # Duplicate partner IDs and undo keys refer to the pre-deletion unit
        # indices. Clear them after this irreversible operation so retained
        # units cannot display or replay stale state.
        clear_duplicate_roles(self._ports, "within")
        clear_duplicate_roles(self._ports, "cross")
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._props_timer.stop()
        self._pending_props_key = None
        self._pending_source_changed = False

        self._log_event(
            "delete_flagged",
            f"deleted {total} flagged MU(s)",
            "",
            -1,
            deleted_by_port=deleted_by_port,
        )
        # Jump back to the first unit of the first port that still has units.
        # `_on_port_changed` is called directly rather than relying on the
        # combo's signal: the signal does not fire when the port is unchanged,
        # which would leave the plots showing a unit that no longer exists.
        self._current_mu_idx = -1
        remaining_ports = [p for p in ports if self._ports.get(p)]
        if remaining_ports:
            first_port = remaining_ports[0]
            self.port_combo.blockSignals(True)
            self.port_combo.setCurrentText(first_port)
            self.port_combo.blockSignals(False)
            self._on_port_changed(first_port)
        else:
            self._current_port = None
            self._refresh_mu_combo()
            self._clear_plots()
        self._update_status(f"Deleted {total} flagged MU(s)")
        self._mark_modified()

    def _remove_outliers(self):
        """Remove spikes causing outlier IFR in the current MU.

        For each consecutive pair of spikes whose instantaneous firing rate
        lies above the Tukey fence (Q75 + 1.5×IQR of the IFR distribution),
        the spike with the lower source amplitude is removed.  A floor of
        2× the median IFR prevents false positives in units with very regular
        firing (where IQR can be near zero).
        """
        mu = self._current_mu()
        if mu is None or len(mu.timestamps) < 3:
            return

        ts = np.sort(mu.timestamps)
        isi = np.diff(ts) / self._fsamp  # seconds
        # Guard against zero-length ISIs (duplicate timestamps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ifr = np.where(isi > 0, 1.0 / isi, np.inf)

        finite_ifr = ifr[np.isfinite(ifr)]
        if len(finite_ifr) < 4:
            self._update_status("Not enough spikes for IFR outlier detection")
            return

        q25, q75 = np.percentile(finite_ifr, [25, 75])
        iqr = q75 - q25
        tukey_threshold = q75 + 1.5 * iqr
        # Floor: must be at least 2× the median to avoid flagging normal variation
        threshold = max(tukey_threshold, 2.0 * np.median(finite_ifr))

        # Find pairs whose IFR exceeds the threshold
        outlier_pair_indices = np.where(ifr > threshold)[0]

        if len(outlier_pair_indices) == 0:
            self._update_status("No IFR outliers found")
            return

        # For each outlier pair (i, i+1), mark the lower-amplitude spike for removal.
        # Collect unique removal indices so a spike in multiple pairs is only removed once.
        source = mu.source
        n_src = len(source)
        to_remove: set[int] = set()
        for i in outlier_pair_indices:
            idx_a, idx_b = i, i + 1
            if idx_a in to_remove or idx_b in to_remove:
                # One of the pair is already scheduled for removal; skip re-evaluation
                continue
            t_a, t_b = ts[idx_a], ts[idx_b]
            amp_a = source[t_a] if 0 <= t_a < n_src else 0.0
            amp_b = source[t_b] if 0 <= t_b < n_src else 0.0
            to_remove.add(idx_a if amp_a <= amp_b else idx_b)

        remove_timestamps = ts[sorted(to_remove)]
        old_ts = mu.timestamps.copy()
        new_ts = np.setdiff1d(mu.timestamps, remove_timestamps)
        self._push_undo(
            UndoAction(
                f"Remove IFR outliers ({len(to_remove)})",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                new_ts,
            )
        )
        mu.timestamps = new_ts
        self._on_data_changed(
            f"Removed {len(to_remove)} IFR outlier spike(s) "
            f"(threshold {threshold:.1f} Hz)"
        )

    def _run_auto_edit_current(self):
        """Apply rule-based auto-editing to the currently selected MU.

        Implements the four rules from Wen et al. (2024): removes spikes with
        low or high IPT/FR, and adds missed spikes based on IPT and FR criteria.
        The edit is pushed as a single undoable action.
        """
        mu = self._current_mu()
        if mu is None:
            return
        if len(mu.timestamps) < MIN_SPIKES:
            self._update_status(
                f"MU {mu.id}: fewer than {MIN_SPIKES} spikes — auto-edit skipped"
            )
            return

        old_ts = mu.timestamps.copy()
        result = auto_edit(old_ts, mu.source, self._fsamp)

        if result.skipped:
            self._update_status(
                f"MU {mu.id}: fewer than {MIN_SPIKES} spikes — auto-edit skipped"
            )
            return

        if np.array_equal(old_ts, result.new_timestamps):
            self._update_status(f"MU {mu.id}: no changes from auto-edit")
            return

        self._push_undo(
            UndoAction(
                f"Auto-edit MU {mu.id}",
                self._current_port,
                self._current_mu_idx,
                old_ts,
                result.new_timestamps,
            )
        )
        mu.timestamps = result.new_timestamps
        self._on_data_changed(
            f"Auto-edited MU {mu.id}: "
            f"{result.n_removed} removed, {result.n_added} added"
        )

    def _auto_flag_unreliable(self):
        """Flag all MUs with SIL < 0.9 across every port for deletion.

        Units with SIL >= 0.9 are considered borderline and left for manual
        review (they pass the primary quality criterion even if secondary
        metrics such as CoV or discharge rate are still failing).

        Units whose properties have not yet been computed are skipped.
        """
        flagged_count = 0
        borderline_count = 0
        no_props_count = 0

        for mus in self._ports.values():
            for mu in mus:
                if mu.flagged_for_deletion:
                    continue  # already flagged — leave it
                if mu.props is None:
                    no_props_count += 1
                    continue
                sil = mu.props.sil
                if np.isnan(sil) or sil < 0.9:
                    mu.flagged_duplicate = True
                    flagged_count += 1
                else:
                    borderline_count += 1

        # Refresh the combo for the current port so ⚠ labels appear
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)

        parts = [f"Auto-flagged {flagged_count} MU(s)"]
        if borderline_count:
            parts.append(f"{borderline_count} borderline left (SIL ≥ 0.9)")
        if no_props_count:
            parts.append(f"{no_props_count} skipped (no quality data)")
        self._update_status(" — ".join(parts))
        if flagged_count:
            self._mark_modified()

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _flag_within_duplicates(self):
        """Suggest lower-priority duplicate MUs in the current port."""
        if not DUPLICATE_DETECTION_AVAILABLE:
            self._update_status(
                "motor_unit_toolbox not available — cannot detect duplicates"
            )
            return

        if self._current_port is None:
            self._update_status("Select a port first")
            return
        motor_units = self._ports.get(self._current_port, [])
        if len(motor_units) < 2:
            self._update_status(
                f"{self._current_port}: fewer than 2 MUs — nothing to compare"
            )
            return

        current_port = self._current_port
        result = scan_within_port_duplicates({current_port: motor_units}, self._fsamp)
        n_flagged = result.n_flagged
        self._log_event(
            "flag_within_duplicates",
            f"suggested {n_flagged} duplicate MU(s) in {current_port}",
            current_port,
            -1,
            flagged_by_port=result.flagged_by_port,
        )
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(self._current_mu())
        self._update_status(
            f"{current_port}: suggested {n_flagged} duplicate MU(s) for deletion"
        )
        self._show_duplicate_report(
            title=f"Duplicates in {current_port}",
            scope=f"within {current_port}",
            pairs=result.pairs,
            flagged_by_port=result.flagged_by_port,
            n_compared=result.n_compared,
            skipped_ports=result.skipped_ports,
            failed_ports=result.failed_ports,
            skipped_reason="fewer than 2 MUs",
        )
        self._mark_modified()

    def _flag_cross_duplicates(self):
        """Detect and flag lower-quality cross-port duplicate MUs for deletion."""
        if not DUPLICATE_DETECTION_AVAILABLE:
            self._update_status(
                "motor_unit_toolbox not available — cannot detect duplicates"
            )
            return

        if len(self._ports) < 2:
            self._update_status("Cross-port: only one port loaded — nothing to compare")
            return

        result = scan_cross_port_duplicates(self._ports, self._fsamp)
        n_flagged = result.n_flagged
        self._log_event(
            "flag_cross_duplicates",
            f"flagged {n_flagged} cross-port duplicate MU(s)",
            "",
            -1,
            flagged_by_port=result.flagged_by_port,
        )
        self._refresh_mu_combo()
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self._update_quality_panel(self._current_mu())
        self._update_status(
            f"Cross-port duplicates: flagged {n_flagged} MU(s) for deletion"
        )
        self._show_duplicate_report(
            title="Cross-Port Duplicates",
            scope="across grids/probes",
            pairs=result.pairs,
            flagged_by_port=result.flagged_by_port,
            n_compared=result.n_compared,
            skipped_ports=result.skipped_ports,
            failed_ports=result.failed_ports,
            skipped_reason="",
        )
        self._mark_modified()

    def _show_duplicate_report(
        self,
        *,
        title: str,
        scope: str,
        pairs: list,
        flagged_by_port: dict,
        n_compared: int,
        skipped_ports: list,
        failed_ports: list,
        skipped_reason: str,
    ):
        """Summarise a duplicate scan in a dialog so the result isn't missed.

        `pairs` holds one (port_a, id_a, port_b, id_b, roa) tuple per detected
        duplicate pair; `flagged_by_port` maps a port name to the ids flagged
        for deletion in it.
        """
        n_pairs = len(pairs)
        n_flagged = sum(len(v) for v in flagged_by_port.values())

        if n_compared == 0:
            reason = (
                f"every port was skipped ({skipped_reason})"
                if skipped_reason
                else "no units are loaded"
            )
            summary = f"Nothing to compare {scope} — {reason}."
        elif n_pairs == 0:
            summary = (
                f"No duplicate pairs found {scope}.<br><br>"
                f"Compared {n_compared} MU(s) at a rate-of-agreement threshold "
                f"of {ROA_THRESHOLD:.0%}."
            )
        else:
            summary = (
                f"Found <b>{n_pairs}</b> duplicate pair(s) {scope} among "
                f"{n_compared} MU(s), and suggested <b>{n_flagged}</b> "
                f"lower-priority MU(s) for deletion.<br><br>"
                f"Rate-of-agreement threshold: {ROA_THRESHOLD:.0%}."
            )

        # Details: every pair with its RoA score, strongest first, grouped by
        # the port (or port pair) it was found in.
        grouped: dict = {}
        for port_a, id_a, port_b, id_b, score in pairs:
            label = port_a if port_a == port_b else f"{port_a} ↔ {port_b}"
            grouped.setdefault(label, []).append((port_a, id_a, port_b, id_b, score))

        detail_lines = []
        for label, group in grouped.items():
            detail_lines.append(label)
            for port_a, id_a, port_b, id_b, score in sorted(group, key=lambda x: -x[4]):
                to_delete = [
                    f"MU {mid}"
                    for pname, mid in ((port_a, id_a), (port_b, id_b))
                    if mid in flagged_by_port.get(pname, [])
                ]
                marks = "   → delete " + ", ".join(to_delete) if to_delete else ""
                detail_lines.append(
                    f"    MU {id_a} ↔ MU {id_b}   RoA {score:.1%}{marks}"
                )
            detail_lines.append("")

        for port_name, ids in flagged_by_port.items():
            if ids:
                detail_lines.append(
                    f"Flagged in {port_name}: "
                    + ", ".join(f"MU {i}" for i in sorted(ids))
                )
        if skipped_ports:
            detail_lines.append("")
            detail_lines.append(
                f"Skipped ({skipped_reason}): " + ", ".join(skipped_ports)
            )
        if failed_ports:
            detail_lines.append("")
            detail_lines.append(
                "Comparison failed (see log): " + ", ".join(failed_ports)
            )

        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(
            QMessageBox.Icon.Warning if failed_ports else QMessageBox.Icon.Information
        )
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(summary)
        if n_flagged:
            box.setInformativeText(
                "Flagged units are marked ⚠ in the MU list — review them, "
                "then use “Delete All Flagged Units” to remove them."
            )
        detail = "\n".join(detail_lines).strip()
        if detail:
            box.setDetailedText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _refresh_port_combo(self):
        cur = self.port_combo.currentText()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for name in self._ports:
            self.port_combo.addItem(name)
        if cur in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
            self.port_combo.setCurrentText(cur)
        self.port_combo.blockSignals(False)

    def _on_port_changed(self, port_name: str):
        if not port_name or port_name not in self._ports:
            return
        self._current_port = port_name
        self._current_mu_idx = -1
        self._clear_plots()
        self._refresh_mu_combo()
        if self.mu_combo.count() > 0:
            self.mu_combo.setCurrentIndex(0)
            self._on_mu_selected(0)

    def _refresh_mu_combo(self):
        self.mu_combo.blockSignals(True)
        self.mu_combo.clear()
        if self._current_port is not None:
            for mu in self._ports.get(self._current_port, []):
                label = f"MU {mu.id}  ({len(mu.timestamps)} spikes)"
                if mu.flagged_for_deletion:
                    label += "  ⚠"
                if mu.props is not None:
                    label += "  ✓" if mu.props.is_reliable else "  ✗"
                    if mu.props.reliability_is_overridden:
                        label += "*"  # verdict set by hand, not by the metrics
                is_dup_delete = (
                    mu.within_duplicate_role == "delete"
                    or mu.cross_duplicate_role == "delete"
                )
                if is_dup_delete:
                    label += "  ⧉"
                if mu.reviewed:
                    label += "  ☑ reviewed"
                self.mu_combo.addItem(label)
        self.mu_combo.blockSignals(False)
        self._update_review_controls()

    # ------------------------------------------------------------------
    # AUX / force overlay
    # ------------------------------------------------------------------

    def _refresh_aux_controls(self):
        aux_channels = []
        if self._original_decomp_data is not None:
            aux_channels = self._original_decomp_data.get("aux_channels") or []
        file_stem = self._loaded_path.stem if self._loaded_path else ""
        self.source_plot.set_aux_data(aux_channels, self._fsamp, file_stem)

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _update_plots(self, reset_view: bool = False):
        mu = self._current_mu()
        if mu is None:
            self._clear_plots()
            return

        vb = self.source_plot.getViewBox()
        if not reset_view:
            x_range = vb.viewRange()[0]
            y_range = vb.viewRange()[1]
            had_custom_range = not vb.autoRangeEnabled()[0]

        self.source_plot.set_data(mu.source, mu.timestamps)
        # In full-source mode mu.source covers the whole recording (offset = 0);
        # otherwise it covers only the plateau window (offset = _start_sample).
        self.source_plot.set_force_offset(
            0 if self._full_source_mode else self._start_sample
        )
        if self._full_source_mode:
            self.source_plot.set_plateau_region(self._start_sample, self._end_sample)
        self.fr_plot.set_data(mu.timestamps)

        if reset_view:
            self._reset_view_full()
        elif had_custom_range:
            vb.setRange(xRange=x_range, yRange=y_range, padding=0)

        self._plot_muap()
        self._update_quality_panel(mu)

    def _clear_plots(self):
        self.source_plot.clear_data()
        self.fr_plot.clear_data()
        self._clear_muap_plot()
        self.quality_bar.clear_properties()

    def _on_data_changed(self, msg: str = "Modified", source_changed: bool = False):
        """Immediate cheap updates; expensive recompute+render deferred 120 ms."""
        mu = self._current_mu()
        review_reset = False
        if mu is not None:
            review_reset = mu.reviewed
            mu.reviewed = False
            key = (self._current_port or "", self._current_mu_idx)
            if self._pending_props_key is not None and self._pending_props_key != key:
                self._props_timer.stop()
                self._flush_props_update()
            self._pending_props_key = key
            if source_changed:
                # Source signal changed (e.g. filter recalc) — redraw curve now
                self.source_plot.set_data(mu.source, mu.timestamps)
                self.source_plot.set_force_offset(
                    0 if self._full_source_mode else self._start_sample
                )
                if self._full_source_mode:
                    self.source_plot.set_plateau_region(
                        self._start_sample, self._end_sample
                    )
            self.fr_plot.set_data(mu.timestamps)
            self.source_plot.update_timestamps(mu.timestamps)

        self._refresh_mu_combo()
        self.mu_combo.blockSignals(True)
        self.mu_combo.setCurrentIndex(self._current_mu_idx)
        self.mu_combo.blockSignals(False)
        if review_reset:
            msg = f"{msg} — review reset"
        self._update_status(msg)
        self._mark_modified()

        # Accumulate source_changed across rapid edits, then flush once
        self._pending_source_changed |= source_changed
        self._props_timer.start()

    def _mark_modified(self) -> None:
        """Record a persisted state change without forcing a plot recomputation."""
        self._set_dirty(True)
        self.data_modified.emit()

    def _flush_props_update(self):
        """Runs after editing pauses: recomputes MU properties and refreshes MUAP + quality."""
        key = self._pending_props_key
        self._pending_props_key = None
        if key is None:
            self._pending_source_changed = False
            return

        port_name, mu_idx = key
        mus = self._ports.get(port_name, [])
        if not (0 <= mu_idx < len(mus)):
            self._pending_source_changed = False
            return

        mu = mus[mu_idx]
        grid_cfg = self._grid_info.get(port_name)
        emg_port = self._emg_data.get(port_name)
        mu.props = recompute_unit_properties(
            mu_props=mu.props or MUProperties(),
            new_timestamps=mu.timestamps,
            source=mu.source,
            emg_port=emg_port,
            grid_positions=grid_cfg["positions"] if grid_cfg else None,
            grid_shape=grid_cfg["grid_shape"] if grid_cfg else None,
            fsamp=self._fsamp,
        )
        self._pending_source_changed = False
        if (self._current_port, self._current_mu_idx) == key:
            self._plot_muap()
            self._update_quality_panel(mu)

    def _update_quality_panel(self, mu: MotorUnit | None):
        if mu is None:
            self.quality_bar.clear_properties()
            self.btn_notes.setEnabled(False)
            return
        if mu.props is not None:
            self.quality_bar.set_properties(
                mu.props,
                within_role=mu.within_duplicate_role,
                within_partners=mu.within_duplicate_partners,
                cross_role=mu.cross_duplicate_role,
                cross_partners=mu.cross_duplicate_partners,
            )
        else:
            self.quality_bar.clear_properties()
        self.btn_notes.setEnabled(True)

    def _open_notes_dialog(self):
        mu = self._current_mu()
        if mu is None:
            return

        dlg = QDialog(self)
        file_name = self._loaded_path.name if self._loaded_path else "Current File"
        window_title = f"File Notes — {file_name}"
        dlg.setWindowTitle(window_title)
        dlg.resize(860, 320)
        dlg.setStyleSheet(
            f"background-color: {COLORS['background']}; color: {COLORS['foreground']};"
        )

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # History with previous notes
        history = QPlainTextEdit()
        history.setPlainText("\n".join(self._notes))
        history.setToolTip(
            f"Edit notes stored with this file and "
            f"press {QKeySequence(QKeySequence.StandardKey.Save).toString()} to save changes. "
            f"Press {QKeySequence(Qt.Key.Key_Escape).toString()} to quit."
        )
        history.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        history.verticalScrollBar().setValue(history.verticalScrollBar().maximum())
        history.setTabChangesFocus(True)
        history.setStyleSheet(
            f"QPlainTextEdit {{"
            f" background-color: {COLORS.get('background', '#1a1f24')};"
            f" color: {COLORS['foreground']};"
            f" font-size: {FONT_SIZES.get('small', '9pt')};"
            f" border: 0px;"
            f" outline: none;"
            f"}}"
        )
        lay.addWidget(history, stretch=1)

        # Editor to add new notes
        editor = QLineEdit()
        editor.setPlaceholderText(f"Add note for {mu.port_name}, MU {mu.id}…")
        editor.setToolTip(
            "Add a file note tagged with the current port and unit, then press Enter"
        )
        editor.setStyleSheet(
            f"QLineEdit {{"
            f" background-color: {COLORS.get('background_input', '#1a1f24')};"
            f" color: {COLORS['foreground']};"
            f" border: 1px solid {COLORS['border']};"
            f" border-radius: 4px;"
            f" font-size: {FONT_SIZES.get('small', '9pt')};"
            f"}}"
        )
        lay.addWidget(editor)

        # Connections:
        def edited_notes() -> list[str]:
            return normalise_notes(history.toPlainText().splitlines())

        def mark_changes():
            """Mark history was changed in window title."""
            if self._notes != edited_notes():
                dlg.setWindowTitle(window_title + " *")
            else:
                dlg.setWindowTitle(window_title)

        def save_notes():
            """Save history in self._notes and inform scd-edit that notes were modified."""
            self._notes = edited_notes()
            dlg.setWindowTitle(window_title)
            self._log_event(
                "notes",
                f"updated notes for {self._current_mu().port_name} (MU {self._current_mu().id})",
                self._current_port or "",
                self._current_mu_idx,
            )
            self._mark_modified()

        def append_note():
            """Read new note, refresh window, and save history"""
            note = editor.text().strip()
            if note:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                full_note = f"{timestamp} ({self._current_mu().port_name}, MU {self._current_mu().id}): {note}"
                history.appendPlainText(full_note)
                editor.clear()  # clear text in editor
                history.verticalScrollBar().setValue(
                    history.verticalScrollBar().maximum()
                )
                save_notes()

        history.textChanged.connect(mark_changes)
        editor.returnPressed.connect(append_note)
        editor.setFocus()

        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, dlg)
        save_shortcut.activated.connect(save_notes)

        dlg.exec()

    def _update_status(self, msg: str | None = None):
        if msg:
            self._last_action_msg = msg
            self.status_bar.showMessage(msg)
            return
        parts = []
        if self._current_port:
            parts.append(f"Port: {self._current_port}")
        n = len(self._ports.get(self._current_port, []))
        if n > 0:
            parts.append(f"{n} MUs")
        if self._current_mu_idx >= 0:
            parts.append(f"MU: {self._current_mu_idx}")

        if self._sel_arm == SelectionArm.ADD:
            parts.append("⬜ Drag to ADD  (armed)")
        elif self._sel_arm == SelectionArm.DELETE:
            parts.append("⬜ Drag to DELETE  (armed)")

        key = (self._current_port or "", self._current_mu_idx)
        n_undo = len(self._undo_stack.get(key, []))
        if n_undo:
            parts.append(f"Undo: {n_undo}")
        if self._full_source_mode:
            parts.append("Full signal")
        if getattr(self, "_last_action_msg", None):
            parts.append(f"Last: {self._last_action_msg}")
        self.status_bar.showMessage("  |  ".join(parts))

    # ------------------------------------------------------------------
    # MUAP panel
    # ------------------------------------------------------------------

    def _open_muap_popout(self):
        if self._muap_popout is None or not self._muap_popout.isVisible():
            self._muap_popout = MuapPopoutDialog(parent=None)
            self._muap_popout.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._muap_popout.destroyed.connect(self._on_muap_popout_closed)
        self._muap_popout.show()
        self._muap_popout.raise_()
        self._muap_popout.activateWindow()
        self._plot_muap()

    def _on_muap_popout_closed(self):
        self._muap_popout = None

    def _plot_muap(self):
        mu = self._current_mu()
        if mu is None:
            self._clear_muap_plot("Select a Motor Unit")
            return
        if mu.props is None or mu.props.muap_grid is None:
            no_emg = self._emg_data.get(self._current_port) is None
            msg = "No EMG data in file" if no_emg else "MUAP unavailable"
            self._clear_muap_plot(msg)
            return

        muap_grid = mu.props.muap_grid
        grid_cfg = self._grid_info.get(self._current_port)
        rejected_pos = self._rejected_ch_positions.get(self._current_port, set())

        if grid_cfg is not None:
            self._render_muap_grid(muap_grid, grid_cfg, rejected_pos)
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_grid(
                    muap_grid, grid_cfg, rejected_pos, self._current_mu_idx
                )
        else:
            n_ch = muap_grid.shape[0]
            waveforms = [muap_grid[i, 0] for i in range(n_ch)]
            self._render_muap_stacked(waveforms, list(range(n_ch)))
            if self._muap_popout and self._muap_popout.isVisible():
                self._muap_popout.render_stacked(
                    waveforms, list(range(n_ch)), self._current_mu_idx
                )

    def _render_muap_grid(
        self,
        muap_grid: np.ndarray,
        grid_cfg: dict,
        rejected_positions: set | None = None,
    ):
        """Render MUAPs in physical grid layout (portrait, rows × cols).

        muap_grid: (rows, cols, n_samples) from compute_port_properties.
        On the first call (or when grid shape changes) all PlotItems are built and
        stored; on subsequent calls only waveform data and amplitudes are updated,
        avoiding expensive scene teardown/rebuild.
        """
        if rejected_positions is None:
            rejected_positions = set()

        rows, cols = grid_cfg["grid_shape"]
        electrode_positions = set(grid_cfg["positions"].values())
        n_samples = muap_grid.shape[2] if muap_grid.ndim == 3 else 409

        valid_wavs = [
            muap_grid[r, c]
            for r in range(min(rows, muap_grid.shape[0]))
            for c in range(min(cols, muap_grid.shape[1]))
            if (r, c) in electrode_positions
            and (r, c) not in rejected_positions
            and len(muap_grid[r, c]) > 0
            and np.any(muap_grid[r, c] != 0)
        ]
        amp = np.max(np.abs(np.concatenate(valid_wavs))) * 1.2 if valid_wavs else 1.0
        grid_key = (rows, cols, n_samples, frozenset(rejected_positions))

        title_html = (
            f"<span style='color:{COLORS['foreground']};font-size:10pt;'>"
            f"MU {self._current_mu_idx}</span>"
        )

        if grid_key == self._muap_grid_key and self._muap_cell_plots:
            # Fast path: only update amplitudes and waveform data in existing plots
            for p in self._muap_cell_plots.values():
                p.setYRange(-amp, amp, padding=0)
            for (r, c), item in self._muap_waveform_items.items():
                wav = (
                    muap_grid[r, c]
                    if r < muap_grid.shape[0] and c < muap_grid.shape[1]
                    else None
                )
                if wav is not None and len(wav) > 0 and np.any(wav != 0):
                    item.setData(wav)
                else:
                    item.setData([])
            if self._muap_title_label is not None:
                self._muap_title_label.setText(title_html)
            return

        # Slow path: full rebuild
        self.muap_widget.clear()
        self._muap_cell_plots = {}
        self._muap_waveform_items = {}

        lbl_style = f"color:{COLORS.get('text_dim', '#6c7086')}; font-size:7pt;"

        def _add_lbl(widget, text, row, col, **kw):
            lbl = widget.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        # Row 0: title. Row 1: column headers. Rows 2+: data. Col 0: row labels.
        self._muap_title_label = _add_lbl(
            self.muap_widget, title_html, 0, 0, colspan=cols + 1, justify="center"
        )
        _add_lbl(
            self.muap_widget,
            f"<span style='{lbl_style}'></span>",
            1,
            0,
            justify="center",
        )
        for c in range(cols):
            _add_lbl(
                self.muap_widget,
                f"<span style='{lbl_style}'>{c + 1}</span>",
                1,
                c + 1,
                justify="center",
            )
        for r in range(rows):
            _add_lbl(
                self.muap_widget,
                f"<span style='{lbl_style}'>{r + 1}</span>",
                r + 2,
                0,
                justify="center",
            )

        _rej_bg = (50, 30, 30)
        _empty_bg = (28, 28, 28)
        gl = self.muap_widget.ci.layout

        for r in range(rows):
            for c in range(cols):
                p = self.muap_widget.addPlot(row=r + 2, col=c + 1)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0)
                p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
                p.setMinimumWidth(0)
                p.setMinimumHeight(0)
                self._muap_cell_plots[(r, c)] = p

                rc = (r, c)
                if rc in rejected_positions:
                    p.getViewBox().setBackgroundColor(_rej_bg)
                    p.plot(
                        [0, n_samples],
                        [0, 0],
                        pen=pg.mkPen(color=(140, 60, 60), width=1),
                    )
                elif rc not in electrode_positions:
                    p.getViewBox().setBackgroundColor(_empty_bg)
                else:
                    # Pre-create waveform item; data filled below
                    item = p.plot([], pen=pg.mkPen(color=COLORS["info"], width=1.5))
                    self._muap_waveform_items[rc] = item

        gl.setSpacing(0)
        gl.setHorizontalSpacing(6)
        gl.setColumnMinimumWidth(0, 14)
        gl.setColumnStretchFactor(0, 0)
        for c in range(cols):
            gl.setColumnMinimumWidth(c + 1, 0)
            gl.setColumnStretchFactor(c + 1, 1)
        for r in range(2):
            gl.setRowMinimumHeight(r, 0)
            gl.setRowStretchFactor(r, 0)
        for r in range(rows):
            gl.setRowMinimumHeight(r + 2, 0)
            gl.setRowStretchFactor(r + 2, 1)

        for (r, c), item in self._muap_waveform_items.items():
            if r < muap_grid.shape[0] and c < muap_grid.shape[1]:
                wav = muap_grid[r, c]
                if len(wav) > 0 and np.any(wav != 0):
                    item.setData(wav)

        self._muap_grid_key = grid_key

    def _render_muap_stacked(self, waveforms, ch_indices):
        self.muap_widget.clear()
        self._muap_grid_key = None  # force grid rebuild on next _render_muap_grid call
        plot = self.muap_widget.addPlot(row=0, col=0)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return
        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["foreground"], width=1.5))
            txt = pg.TextItem(f"Ch {ch}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 7))
            plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        plot.setTitle(
            f"MU {self._current_mu_idx} — Stacked",
            color=COLORS["foreground"],
            size="10pt",
        )

    def _clear_muap_plot(self, message: str = "Select a Motor Unit"):
        self.muap_widget.clear()
        self._muap_grid_key = None
        p = self.muap_widget.addPlot(row=0, col=0)
        p.hideAxis("left")
        p.hideAxis("bottom")
        p.setMouseEnabled(x=False, y=False)
        p.setXRange(0, 1)
        p.setYRange(0, 1)
        t = pg.TextItem(message, color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        t.setPos(0.5, 0.5)
        p.addItem(t)
        if self._muap_popout and self._muap_popout.isVisible():
            self._muap_popout.clear(message)
