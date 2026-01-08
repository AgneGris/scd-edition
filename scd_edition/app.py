"""
SCD Edition - Application entry point.
"""

import sys

from PyQt5.QtWidgets import QApplication

from scd_edition.gui.main_window import MainWindow
from scd_edition.gui.styles import get_dark_stylesheet, load_fonts

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyqtgraph")

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Setup font
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    
    # Load custom fonts (optional)
    load_fonts("Figtree")
    
    # Create and show window
    window = MainWindow()
    window.setStyleSheet(get_dark_stylesheet("Figtree"))
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()