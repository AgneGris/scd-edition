"""
Main application window for SCD-edition.
"""

import logging
import os
import sys
from pathlib import Path

import torch
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scd_app.core.config import ConfigManager, SessionConfig
from scd_app.core.logging_config import (
    configure_logging,
    install_exception_hook,
    log_startup_diagnostics,
    runtime_diagnostics,
)
from scd_app.gui.style.styling import set_style_sheet
from scd_app.gui.tabs.config_tab import ConfigTab
from scd_app.gui.tabs.decomposition_tab import DecompositionTab
from scd_app.gui.tabs.edition_tab import EditionTab
from scd_app.gui.tabs.visualisation_tab import VisualisationTab

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window with tabbed interface.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCD - EMG Decomposition & Edition")

        # Dynamically size to screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(1400, screen.width()), min(1000, screen.height()))
        self.resize(
            min(1400, int(screen.width() * 0.95)),
            min(1000, int(screen.height() * 0.95)),
        )

        # Core objects
        self.config_manager = ConfigManager()
        self.config: SessionConfig | None = None
        self._close_pending = False

        self._setup_ui()
        self._setup_menu()
        self._setup_connections()

        self._reset_session()

    def _setup_ui(self):
        """Build the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # 1. Configuration Tab
        self.config_tab = ConfigTab()
        self.tabs.addTab(self.config_tab, "1. Configuration")

        # 2. Decomposition Tab
        self.decomp_tab = DecompositionTab()
        self.tabs.addTab(self.decomp_tab, "2. Decomposition")

        # 3. Edition Tab
        self.edition_tab = EditionTab(fsamp=2048.0)
        self.tabs.addTab(self.edition_tab, "3. Edition")

        # 4. Visualisation Tab (disabled until a file is loaded)
        self.vis_tab = VisualisationTab(edition_tab=self.edition_tab)
        self.tabs.addTab(self.vis_tab, "4. Visualisation")

        self._set_tabs_enabled(False)

        layout.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Please configure session")

    def _setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu("&View")
        for i, name in enumerate(
            ["Configuration", "Decomposition", "Edition", "Visualisation"]
        ):
            action = QAction(f"&{i + 1}. {name}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            action.triggered.connect(
                lambda checked, idx=i: self.tabs.setCurrentIndex(idx)
            )
            view_menu.addAction(action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        open_logs_action = QAction("Open &Log Folder", self)
        open_logs_action.triggered.connect(self._open_log_folder)
        help_menu.addAction(open_logs_action)

        copy_diagnostics_action = QAction("&Copy Diagnostics", self)
        copy_diagnostics_action.triggered.connect(self._copy_diagnostics)
        help_menu.addAction(copy_diagnostics_action)

        help_menu.addSeparator()
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_connections(self):
        """Setup signal connections between tabs."""
        # Configuration → Decomposition
        self.config_tab.config_applied.connect(self._on_config_applied)

        # Decomposition → Edition
        self.decomp_tab.decomposition_complete.connect(self._on_decomposition_complete)

        # Edition → Visualisation
        self.edition_tab.file_loaded.connect(self._on_file_loaded_into_edition)

        # Tab switch → Visualisation refresh
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        if index == 3:
            self.vis_tab.on_tab_activated()

    def _reset_session(self):
        """Reset session state."""
        self.config = None
        self._set_tabs_enabled(False)

    def _set_tabs_enabled(self, enabled: bool):
        # Only control Decomposition tab (index 1)
        self.tabs.setTabEnabled(1, enabled)
        # Always allow Edition (index 2)
        self.tabs.setTabEnabled(2, True)
        # Visualisation tab (index 3) starts disabled; enabled on first file load
        self.tabs.setTabEnabled(3, False)

    def _on_config_applied(self, config: SessionConfig, emg_paths: list):
        """Handle the 'Apply' event from the Configuration tab."""
        self.config = config

        # Update Edition tab sampling rate and aux channel configs (for MVC lookup on load)
        self.edition_tab.set_fsamp(config.sampling_frequency)
        self.edition_tab.set_aux_configs(config.aux_channels)

        # Configure Decomposition Tab
        if hasattr(
            self.decomp_tab, "setup_session"
        ) and not self.decomp_tab.setup_session(config, emg_paths):
            self.config = None
            self._set_tabs_enabled(False)
            self.status_bar.showMessage("Configuration could not be loaded")
            return

        # Enable tabs and switch to Decomposition
        self._set_tabs_enabled(True)
        self.tabs.setCurrentIndex(1)

        self.status_bar.showMessage(
            f"✓ Configuration Applied: {len(config.ports)} grid(s) configured"
        )

    def _on_file_loaded_into_edition(self):
        """Enable the Visualisation tab and mark it stale after a file load."""
        self.tabs.setTabEnabled(3, True)
        self.vis_tab.on_data_modified()

    def _on_decomposition_complete(self, decomp_path: Path):
        """Handle decomposition completion and auto-load into Edition tab."""
        try:
            if self.edition_tab.load_from_path(decomp_path, trusted=True):
                self.tabs.setCurrentWidget(self.edition_tab)
                self.status_bar.showMessage(
                    "✓ Decomposition complete — loaded into Edition tab"
                )
            else:
                self.status_bar.showMessage(
                    f"Decomposition saved to {decomp_path.name}, but was not loaded"
                )
        except Exception as e:
            logger.exception("Could not load completed decomposition %s", decomp_path)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Decomposition finished but failed to load into Edition:\n{e}",
            )

    def _open_log_folder(self):
        log_directory = configure_logging().parent
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_directory))):
            QMessageBox.warning(
                self,
                "Log Folder",
                f"Could not open the log folder automatically:\n{log_directory}",
            )

    def _copy_diagnostics(self):
        diagnostics = runtime_diagnostics(configure_logging())
        QApplication.clipboard().setText(diagnostics)
        self.status_bar.showMessage("Diagnostics copied to the clipboard", 5000)
        logger.info("Runtime diagnostics copied to the clipboard")

    def _show_about(self):
        QMessageBox.about(
            self,
            "About SCD Edition",
            "SCD Edition\nEMG Decomposition & Edition\n\n"
            "Motor unit decomposition, review and spike editing\n\n"
            "Licensed under the BSD 3-Clause License.",
        )

    def closeEvent(self, event):
        if self._close_pending:
            event.ignore()
            return

        if self.decomp_tab.has_running_worker():
            reply = QMessageBox.question(
                self,
                "Decomposition Running",
                "Stop after the current grid, save completed results, and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            worker = self.decomp_tab.worker
            self._close_pending = True
            worker.finished.connect(self._finish_pending_close)
            self.decomp_tab.request_shutdown()
            self.status_bar.showMessage(
                "Finishing the current grid safely before closing…"
            )
            event.ignore()
            return

        if not self.edition_tab.confirm_save_changes("closing"):
            event.ignore()
            return

        event.accept()

    def _finish_pending_close(self):
        self._close_pending = False
        if self.decomp_tab.shutdown_save_failed:
            self.status_bar.showMessage(
                "Close canceled because completed decomposition results could not be saved"
            )
            return
        QTimer.singleShot(0, self.close)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="scd-edition",
        description="SCD EMG Decomposition & Edition GUI",
    )
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--open",
        dest="open_path",
        metavar="FILE",
        help="PKL decomposition file to load directly into the Edition tab on startup",
    )
    startup.add_argument(
        "--example",
        action="store_true",
        help="open the bundled example recording with its configuration filled in",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        metavar="FILE",
        help="Default output path used when saving (skips the save dialog)",
    )
    parser.add_argument(
        "--quit-after-save",
        dest="quit_after_save",
        action="store_true",
        help="Close the application automatically after the file is saved",
    )
    # parse_known_args so Qt's own flags (e.g. -platform) are left in sys.argv
    args, qt_argv = parser.parse_known_args()

    log_path = configure_logging()
    install_exception_hook()
    log_startup_diagnostics(log_path)

    app = QApplication([sys.argv[0], *qt_argv])
    app.setApplicationName("SCD-Edition")

    cuda_available = torch.cuda.is_available()
    cuda_required = (
        os.environ.get("SCD_REQUIRE_CUDA", "").strip().lower()
        in {"1", "true", "yes", "on"}
        or (Path(sys.prefix) / ".scd-require-cuda").is_file()
    )
    if cuda_required and not cuda_available:
        message = (
            "CUDA is required for this SCD Edition environment, but PyTorch "
            f"cannot use it (PyTorch {torch.__version__}, CUDA build "
            f"{torch.version.cuda or 'none'}).\n\n"
            "For this source checkout, run:\n"
            "uv sync --python 3.13 --managed-python --extra cuda\n\n"
            "For later uv commands, include --extra cuda or --no-sync; "
            "a bare uv run may reinstall CPU-only PyTorch."
        )
        logger.error("SCD Edition startup error: %s", message)
        QMessageBox.critical(None, "CUDA Required", message)
        return 1

    if cuda_available:
        logger.info(
            "SCD Edition device: CUDA (%s) because PyTorch detected a "
            "working CUDA device.",
            torch.cuda.get_device_name(0),
        )
    else:
        logger.info(
            "SCD Edition device: CPU because PyTorch did not detect a working "
            "CUDA device."
        )

    set_style_sheet(app)

    window = MainWindow()

    if args.output_path:
        window.edition_tab.set_output_path(Path(args.output_path))

    if args.quit_after_save:
        window.edition_tab.set_quit_after_save(True)

    if args.open_path:
        open_path = Path(args.open_path)
        # Defer until the event loop is running so the window is fully shown
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: _open_on_startup(window, open_path))
    elif args.example:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: _configure_example_on_startup(window))

    window.show()
    return app.exec()


def _open_on_startup(window: "MainWindow", path: Path):
    try:
        if window.edition_tab.load_from_path(path):
            window.tabs.setCurrentWidget(window.edition_tab)
    except Exception as e:
        logger.exception("Could not open file '%s'", path)
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(window, "Load Error", f"Could not open file:\n{e}")
        sys.exit(1)


def _configure_example_on_startup(window: "MainWindow"):
    """Fill the Configuration tab with the packaged example recording."""
    from scd_app.examples import bundled_example_config, bundled_example_path

    try:
        sample_path = bundled_example_path()
        config = bundled_example_config()
        config["file_path"] = str(sample_path)
        config["output_dir"] = str(Path.cwd() / "scd-edition-output")
        window.config_tab._config_from_dict(config)
        window.tabs.setCurrentWidget(window.config_tab)
        window.status_bar.showMessage(
            "Bundled example ready — review the settings and click Apply Configuration"
        )
    except Exception as exc:
        logger.exception("Could not configure bundled example")
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(
            window, "Example Error", f"Could not configure bundled example:\n{exc}"
        )


if __name__ == "__main__":
    sys.exit(main())
