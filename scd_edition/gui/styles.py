"""
GUI styling and theming.
"""

from PyQt5.QtGui import QFontDatabase


def load_fonts(font_type: str = "Figtree") -> None:
    """Load custom fonts if available."""
    font_files = {
        "Figtree": [
            "fonts/Figtree-Regular.ttf",
            "fonts/Figtree-Medium.ttf",
            "fonts/Figtree-SemiBold.ttf",
            "fonts/Figtree-Bold.ttf"
        ],
        "Inter": [
            "fonts/Inter-Regular.ttf",
            "fonts/Inter-Medium.ttf",
            "fonts/Inter-SemiBold.ttf",
            "fonts/Inter-ExtraBold.ttf"
        ]
    }
    
    for font_file in font_files.get(font_type, []):
        try:
            QFontDatabase.addApplicationFont(font_file)
        except Exception:
            pass


def get_dark_stylesheet(font_type: str = "Figtree") -> str:
    """Return dark theme stylesheet."""
    return f"""
        QMainWindow, QWidget {{
            background-color: #1a1d23;
            color: #e8eaed;
            font-family: '{font_type}', sans-serif;
            font-size: 13pt;
        }}
        
        QPushButton {{
            background-color: #003E74;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 500;
            font-size: 12pt;
        }}
        QPushButton:hover {{
            background-color: #005a9e;
        }}
        QPushButton:pressed {{
            background-color: #002e52;
        }}
        QPushButton:disabled {{
            background-color: #3d4248;
            color: #7a7f87;
        }}
        QPushButton:checked {{
            background-color: #007acc;
            border: 2px solid #4da6ff;
        }}
        
        QComboBox {{
            background-color: #21252b;
            color: #e8eaed;
            border: 1px solid #3d4248;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12pt;
        }}
        QComboBox::drop-down {{
            border-left: 1px solid #3d4248;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }}
        QComboBox QAbstractItemView {{
            background-color: #21252b;
            selection-background-color: #003E74;
            border: 1px solid #3d4248;
        }}
        
        QCheckBox {{
            spacing: 10px;
        }}
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border: 2px solid #3d4248;
            border-radius: 4px;
            background-color: #21252b;
        }}
        QCheckBox::indicator:checked {{
            background-color: #003E74;
            border: 2px solid #003E74;
        }}
        
        QLabel {{
            color: #e8eaed;
        }}
        
        QLineEdit {{
            background-color: #21252b;
            color: #e8eaed;
            border: 1px solid #3d4248;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12pt;
        }}
        
        QMessageBox {{
            background-color: #1a1d23;
        }}
        QMessageBox QLabel {{
            color: #e8eaed;
        }}
        
        QInputDialog {{
            background-color: #1a1d23;
        }}
    """


class PlotColors:
    """Color scheme for plots."""
    
    # Source plot
    SOURCE_LINE = (120, 180, 255)
    SPIKE_MARKER = (255, 150, 50)
    
    # MUAP plot
    MUAP_MAX_CHANNEL = (0, 255, 255)
    MUAP_OTHER_CHANNEL = (150, 180, 220)
    
    # Preview colors
    PREVIEW_ADD = (0, 255, 136, 180)
    PREVIEW_DELETE = (255, 102, 102, 180)
    
    # Force colors
    FORCE_COLORS = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
    ]
    
    # UI colors
    FLAGGED_UNIT = (255, 100, 100)
    NORMAL_UNIT = (232, 234, 237)