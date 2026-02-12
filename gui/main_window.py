"""
Main application window for SCD Suite.
"""

import sys
import os
import pickle
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
from gui.tabs.config_tab import ConfigTab
from gui.tabs.decomposition_tab import DecompositionTab
from gui.tabs.edition_tab import EditionTab

# Import Core
from core.config import ConfigManager, SessionConfig

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
        
        self._setup_ui()
        self._setup_menu()
        self._setup_connections()
        
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
        self.tabs.addTab(self.config_tab, "1. Configuration")
        
        # 2. Decomposition Tab
        self.decomp_tab = DecompositionTab()
        self.tabs.addTab(self.decomp_tab, "2. Decomposition")
        
        # 3. Edition Tab — self-contained, takes fsamp
        self.edition_tab = EditionTab(fsamp=2048.0)
        self.tabs.addTab(self.edition_tab, "3. Edition")
        
        # Initially disable tabs 2-3 until config is applied
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
        for i, name in enumerate(["Configuration", "Decomposition", "Edition"]):
            action = QAction(f"&{i+1}. {name}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
            action.triggered.connect(lambda checked, idx=i: self.tabs.setCurrentIndex(idx))
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

    def _reset_session(self):
        """Reset session state."""
        self.config = None
        self._set_tabs_enabled(False)

    def _set_tabs_enabled(self, enabled: bool):
        """Enable or disable operational tabs."""
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, enabled)

    def _on_config_applied(self, config: SessionConfig, emg_path: Path):
        """Handle the 'Apply' event from the Configuration tab."""
        self.config = config
        
        # Update Edition tab sampling rate
        self.edition_tab.set_fsamp(config.sampling_frequency)
        
        # Configure Decomposition Tab
        if hasattr(self.decomp_tab, 'setup_session'):
            self.decomp_tab.setup_session(config, emg_path)
        
        # Enable tabs and switch to Decomposition
        self._set_tabs_enabled(True)
        self.tabs.setCurrentIndex(1)
        
        self.status_bar.showMessage(
            f"✓ Configuration Applied: {len(config.ports)} grid(s) configured"
        )
    
    def _on_decomposition_complete(self, decomp_path: Path):
        """Handle decomposition completion and auto-load into Edition tab."""
        try:
            self.edition_tab.load_from_path(decomp_path)
            self.tabs.setCurrentIndex(2)
            self.status_bar.showMessage(
                "✓ Decomposition complete — loaded into Edition tab"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Decomposition finished but failed to load into Edition:\n{e}"
            )

    def _show_about(self):
        QMessageBox.about(
            self, 
            "About SCD Suite", 
            "SCD Suite\nEMG Decomposition & Edition\n\n"
            "Real-time motor unit decomposition and spike editing"
        )

    def closeEvent(self, event):
        # Check if edition tab has unsaved edits via undo stack
        has_edits = len(self.edition_tab._undo_stack) > 0
        if has_edits:
            reply = QMessageBox.question(
                self, "Unsaved Changes", "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
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
    app = QApplication(sys.argv)
    app.setApplicationName("SCD Suite")
    
    # === APPLY GLOBAL STYLESHEET ===
    set_style_sheet(app)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()