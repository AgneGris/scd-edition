"""
Main application window for SCD-edition.
"""

import sys
import pickle
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QMenuBar,
    QMenu,
    QAction,
    QStatusBar,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtGui import QKeySequence

from scd_app.gui.tabs.config_tab import ConfigTab
from scd_app.gui.tabs.decomposition_tab import DecompositionTab
from scd_app.gui.tabs.edition_tab import EditionTab
from scd_app.gui.tabs.visualisation_tab import VisualisationTab

from scd_app.core.config import ConfigManager, SessionConfig
from scd_app.gui.style.styling import set_style_sheet


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
        self.config: Optional[SessionConfig] = None

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
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu("&View")
        for i, name in enumerate(["Configuration", "Decomposition", "Edition", "Visualisation"]):
            action = QAction(f"&{i+1}. {name}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
            action.triggered.connect(
                lambda checked, idx=i: self.tabs.setCurrentIndex(idx)
            )
            view_menu.addAction(action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
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

        # Update Edition tab sampling rate
        self.edition_tab.set_fsamp(config.sampling_frequency)

        # Configure Decomposition Tab
        if hasattr(self.decomp_tab, "setup_session"):
            self.decomp_tab.setup_session(config, emg_paths)

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
            self.edition_tab.load_from_path(decomp_path)
            self.tabs.setCurrentWidget(self.edition_tab)
            self.status_bar.showMessage(
                "✓ Decomposition complete — loaded into Edition tab"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Decomposition finished but failed to load into Edition:\n{e}",
            )

    def _show_about(self):
        QMessageBox.about(
            self,
            "About SCD Suite",
            "SCD Suite\nEMG Decomposition & Edition\n\n"
            "Real-time motor unit decomposition and spike editing",
        )

    def closeEvent(self, event):
        # Check if edition tab has unsaved edits via undo stack
        has_edits = len(self.edition_tab._undo_stack) > 0
        if has_edits:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.edition_tab._save_file()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="scd-edition",
        description="SCD EMG Decomposition & Edition GUI",
    )
    parser.add_argument(
        "--open", dest="open_path", metavar="FILE",
        help="PKL decomposition file to load directly into the Edition tab on startup",
    )
    parser.add_argument(
        "--output", dest="output_path", metavar="FILE",
        help="Default output path used when saving (skips the save dialog)",
    )
    parser.add_argument(
        "--quit-after-save", dest="quit_after_save", action="store_true",
        help="Close the application automatically after the file is saved",
    )
    # parse_known_args so Qt's own flags (e.g. -platform) are left in sys.argv
    args, qt_argv = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + qt_argv)
    app.setApplicationName("SCD-Edition")

    set_style_sheet(app)

    window = MainWindow()

    if args.output_path:
        window.edition_tab.set_output_path(Path(args.output_path))

    if args.quit_after_save:
        window.edition_tab.set_quit_after_save(True)

    if args.open_path:
        open_path = Path(args.open_path)
        # Defer until the event loop is running so the window is fully shown
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: _open_on_startup(window, open_path))

    window.show()
    sys.exit(app.exec_())


def _open_on_startup(window: "MainWindow", path: Path):
    try:
        window.edition_tab.load_from_path(path)
        window.tabs.setCurrentWidget(window.edition_tab)
    except Exception as e:
        print(f"ERROR: Could not open file '{path}': {e}", file=sys.stderr)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(window, "Load Error", f"Could not open file:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
