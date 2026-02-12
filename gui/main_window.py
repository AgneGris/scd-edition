"""
Main application window for SCD Suite.
"""

import sys
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QMenuBar, QMenu, QAction, QStatusBar,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QKeySequence

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Tabs
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.config_tab import ConfigTab
from gui.tabs.decomposition_tab import DecompositionTab
from gui.tabs.edition_tab import EditionTab

# Import Core
from core.config import ConfigManager, SessionConfig
from core.data_handler import DataHandler

# Import Styling
from gui.style.styling import set_style_sheet


class MainWindow(QMainWindow):
    """
    Main application window with tabbed interface.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SCD - EMG Decomposition & Edition")
        self.setMinimumSize(1400, 1000)
        
        # Core objects
        self.config_manager = ConfigManager()
        self.config: Optional[SessionConfig] = None
        self.data_handler: Optional[DataHandler] = None
        
        self._setup_ui()
        self._setup_menu()
        
        # Start with a clean state
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
        self.config_tab.config_applied.connect(self._on_config_applied)
        self.tabs.addTab(self.config_tab, "1. Configuration")
        
        # 2. Decomposition Tab
        self.decomp_tab = DecompositionTab()
        self.tabs.addTab(self.decomp_tab, "2. Decomposition")
        
        # 3. Edition Tab
        # Initialize placeholder data handler
        self.data_handler = DataHandler()
        self.edition_tab = EditionTab(self.data_handler)
        self.tabs.addTab(self.edition_tab, "3. Edition")
        
        # 4. Analysis Tab
        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, "4. Analysis")
        
        # Initially disable tabs 2-4 until config is applied
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
        for i, name in enumerate(["Configuration", "Decomposition", "Edition", "Analysis"]):
            action = QAction(f"&{i+1}. {name}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
            action.triggered.connect(lambda checked, idx=i: self.tabs.setCurrentIndex(idx))
            view_menu.addAction(action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _reset_session(self):
        """Reset session state."""
        self.config = None
        self._set_tabs_enabled(False)

    def _set_tabs_enabled(self, enabled: bool):
        """Enable or disable operational tabs."""
        for i in range(1, 4):
            self.tabs.setTabEnabled(i, enabled)

    def _on_config_applied(self, config: SessionConfig, emg_path: Path):
        """
        Handle the 'Apply' event from the Configuration tab.
        """
        self.config = config
        
        # Update DataHandler
        self.data_handler = DataHandler(
            fsamp=config.sampling_frequency,
            max_undo=config.undo_levels
        )
        
        # Update Tabs
        self.edition_tab.data = self.data_handler
        self.edition_tab.analyzer.fsamp = self.data_handler.fsamp
        
        # Configure Decomposition Tab
        if hasattr(self.decomp_tab, 'setup_session'):
            self.decomp_tab.setup_session(config, emg_path)
        
        # Enable tabs and switch to Decomposition
        self._set_tabs_enabled(True)
        self.tabs.setCurrentIndex(1)
        
        self.status_bar.showMessage(f"Configuration Applied: {len(config.ports)} probes configured.")

    def _show_about(self):
        QMessageBox.about(self, "About SCD Suite", "SCD Suite\nEMG Decomposition & Edition")

    def closeEvent(self, event):
        if self.data_handler and self.data_handler.modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes", "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.edition_tab._save_data()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SCD Suite")
    
    # === APPLY GLOBAL STYLESHEET ===
    # This applies the dark theme to every widget in the app
    set_style_sheet(app)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()