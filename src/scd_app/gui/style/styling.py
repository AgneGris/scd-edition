from PyQt5.QtGui import QFontDatabase, QFont
import pyqtgraph as pg

FONT_FAMILY = 'Lexend'

COLORS = {
    # Backgrounds
    'background': '#0e1318',
    'background_light': '#161b20',
    'background_input': '#1a1f24',
    'background_hover': '#1e242b',
    
    # Text
    'foreground': '#e8eaed',
    'text_dim': '#9da3ab',
    'text_muted': '#5a6169',
    'text_secondary': '#A0AEC0',
    
    # Borders
    'border': '#2d3139',
    
    # Accent (Imperial Blue)
    'accent': '#003E74',
    'accent_hover': '#004a8f',
    'accent_light': '#005aa6',
    
    # Semantic colors
    'success': '#48BB78',
    'warning': '#F6AD55',
    'error': '#FC8181',
    'error_bright': '#F85149',
    'info': '#4a9eff',
    'info_light': '#63B3ED',
    
    # Additional accent colors
    'pink': '#ff6b9d',
    'purple': '#a78bfa',
}

FONT_SIZES = {
    'tiny': '9pt',       # File paths, metadata
    'small': '10pt',     # Status text, secondary info
    'normal': '11pt',    # Default body text, buttons
    'medium': '12pt',    # Emphasized text, form labels
    'large': '13pt',     # Section headers
    'xlarge': '14pt',    # Major headings, dialog section titles
    'title': '16pt',     # Dialog titles, main headers
    'huge': '18pt',      # Big dialog titles
    'display': '22pt',   # Large display text (timers, counters)
    'countdown': '24pt', # Countdown displays
}


SPACING = {
    'xs': 5,
    'sm': 10,
    'md': 15,
    'lg': 20,
    'xl': 30,
}

def get_label_style(
    color: str = None,
    size: str = 'normal',
    bold: bool = False,
    italic: bool = False,
    margin_top: int = 0,
    margin_bottom: int = 0,
    padding: int = 0
) -> str:
    """
    Generate consistent label stylesheet string.
    
    Args:
        color: Color key from COLORS dict or hex string
        size: Size key from FONT_SIZES dict ('tiny', 'small', 'normal', etc.)
        bold: Whether to use bold font weight
        italic: Whether to use italic font style
        margin_top: Top margin in pixels
        margin_bottom: Bottom margin in pixels
        padding: Padding in pixels
    
    Returns:
        CSS stylesheet string
    
    Example:
        label.setStyleSheet(get_label_style(color='warning', size='large', bold=True))
    """
    # Resolve color
    if color is None:
        color_value = COLORS['foreground']
    elif color in COLORS:
        color_value = COLORS[color]
    else:
        color_value = color  # Assume it's a hex string
    
    # Resolve font size
    font_size = FONT_SIZES.get(size, FONT_SIZES['normal'])
    
    # Build style
    style_parts = [
        f"color: {color_value}",
        f"font-size: {font_size}",
    ]
    
    if bold:
        style_parts.append("font-weight: bold")
    
    if italic:
        style_parts.append("font-style: italic")
    
    if margin_top > 0:
        style_parts.append(f"margin-top: {margin_top}px")
    
    if margin_bottom > 0:
        style_parts.append(f"margin-bottom: {margin_bottom}px")
    
    if padding > 0:
        style_parts.append(f"padding: {padding}px")
    
    return "; ".join(style_parts) + ";"


def get_section_header_style(color: str = 'info', margin_top: int = None) -> str:
    """
    Get consistent section header style.
    
    Args:
        color: Color key from COLORS dict
        margin_top: Top margin in pixels (default: SPACING['lg'] = 20)
    
    Returns:
        CSS stylesheet string for section headers
    """
    if margin_top is None:
        margin_top = SPACING['lg']
    return get_label_style(color=color, size='large', bold=True, margin_top=margin_top)


def get_file_label_style(color: str = 'text_dim') -> str:
    """
    Get consistent file path label style.
    
    Args:
        color: Color key from COLORS dict
    
    Returns:
        CSS stylesheet string for file path labels
    """
    return get_label_style(color=color, size='small', padding=2)


def get_dialog_title_style(color: str = 'foreground') -> str:
    """
    Get consistent dialog title style.
    
    Args:
        color: Color key from COLORS dict
    
    Returns:
        CSS stylesheet string for dialog titles
    """
    return get_label_style(color=color, size='huge', bold=True)


def get_dialog_subtitle_style(color: str = 'info') -> str:
    """
    Get consistent dialog subtitle/info style.
    
    Args:
        color: Color key from COLORS dict
    
    Returns:
        CSS stylesheet string for dialog subtitles
    """
    return get_label_style(color=color, size='medium')


def get_timer_style(color: str = 'foreground') -> str:
    """
    Get consistent timer/countdown display style.
    
    Args:
        color: Color key from COLORS dict
    
    Returns:
        CSS stylesheet string for timer displays
    """
    return get_label_style(color=color, size='xlarge', bold=True)


def get_countdown_style(color: str = 'foreground') -> str:
    """
    Get consistent large countdown display style.
    
    Args:
        color: Color key from COLORS dict
    
    Returns:
        CSS stylesheet string for countdown displays
    """
    return get_label_style(color=color, size='countdown', bold=True)


def get_html_span(text: str, color: str = 'foreground', size: str = 'normal', bold: bool = False) -> str:
    """
    Generate HTML span with consistent styling for use in rich text labels.
    
    Args:
        text: The text content
        color: Color key from COLORS dict or hex string
        size: Size key from FONT_SIZES dict
        bold: Whether to use bold font weight
    
    Returns:
        HTML span string
    
    Example:
        label.setText(get_html_span("Hello", color='success', size='large', bold=True))
    """
    # Resolve color
    if color in COLORS:
        color_value = COLORS[color]
    else:
        color_value = color
    
    # Resolve font size
    font_size = FONT_SIZES.get(size, FONT_SIZES['normal'])
    
    # Build style
    weight = "bold" if bold else "normal"
    
    return f'<span style="font-family: {FONT_FAMILY}; font-size: {font_size}; font-weight: {weight}; color: {color_value};">{text}</span>'


def get_button_style(
    bg_color: str = None,
    text_color: str = 'foreground',
    size: str = 'normal',
    padding: int = 10
) -> str:
    """
    Generate custom button stylesheet.
    
    Args:
        bg_color: Background color key from COLORS dict
        text_color: Text color key from COLORS dict
        size: Font size key from FONT_SIZES dict
        padding: Padding in pixels
    
    Returns:
        CSS stylesheet string for buttons
    """
    bg = COLORS.get(bg_color, COLORS['text_muted']) if bg_color else COLORS['text_muted']
    fg = COLORS.get(text_color, COLORS['foreground'])
    font_size = FONT_SIZES.get(size, FONT_SIZES['normal'])
    
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {fg};
            font-size: {font_size};
            padding: {padding}px;
        }}
    """

def load_font(font_type="Lexend"):
    """Load custom fonts into Qt application."""
    
    if font_type == "Lexend":
        font_files = [
            "src/scd_app/gui/style/fonts/Lexend-Regular.ttf",
            "src/scd_app/gui/style/fonts/Lexend-Light.ttf",
            "src/scd_app/gui/style/fonts/Lexend-Medium.ttf",
            "src/scd_app/gui/style/fonts/Lexend-SemiBold.ttf",
            "src/scd_app/gui/style/fonts/Lexend-Bold.ttf",
            "src/scd_app/gui/style/fonts/Lexend-ExtraBold.ttf",
        ]
    elif font_type == "Figtree":
        font_files = [
            "src/scd_app/gui/style/fonts/Figtree-Regular.ttf",
            "src/scd_app/gui/style/fonts/Figtree-Medium.ttf",
            "src/scd_app/gui/style/fonts/Figtree-SemiBold.ttf",
            "src/scd_app/gui/style/fonts/Figtree-Bold.ttf",
        ]
    elif font_type == "Inter":
        font_files = [
            "src/scd_app/gui/style/fonts/Inter-Regular.ttf",
            "src/scd_app/gui/style/fonts/Inter-Medium.ttf",
            "src/scd_app/gui/style/fonts/Inter-SemiBold.ttf",
            "src/scd_app/gui/style/fonts/Inter-ExtraBold.ttf",
        ]
    else:
        raise ValueError("Unsupported font type. Choose 'Lexend', 'Figtree', or 'Inter'.")
    
    for font_file in font_files:
        font_id = QFontDatabase.addApplicationFont(font_file)
        if font_id < 0:
            print(f"Warning: Could not load font {font_file}")
    
    return font_type


def set_style_sheet(widget, font_type="Lexend"):
    """
    Modern dark theme with Imperial Blue and Lexend font.
    Clean, professional styling optimized for data visualization applications.
    """
    font_family = load_font(font_type=font_type)

    return widget.setStyleSheet(
        f"""
            /* ========== BASE WIDGET ========== */
            QWidget {{
                background-color: {COLORS['background']}; 
                color: {COLORS['foreground']};
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                font-size: {FONT_SIZES['medium']};
                font-weight: 400;
            }}

            /* ========== LABELS ========== */
            QLabel {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                color: {COLORS['foreground']};
                background-color: transparent;
                padding: 2px;
            }}
            
            QLabel[heading="true"] {{
                font-size: {FONT_SIZES['xlarge']};
                font-weight: 600;
                color: #ffffff;
            }}
            
            QLabel#PinkLabel {{ color: {COLORS['pink']}; font-weight: 600; }}
            QLabel#BlueLabel {{ color: {COLORS['info']}; font-weight: 600; }}
            QLabel#PurpleLabel {{ color: {COLORS['purple']}; font-weight: 600; }}
            QLabel#GreenLabel {{ color: {COLORS['success']}; font-weight: 600; }}
            QLabel#OrangeLabel {{ color: {COLORS['warning']}; font-weight: 600; }}

            /* ========== TAB WIDGET ========== */
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['background_light']};  
                margin-top: -1px;
            }}

            QTabBar {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
            }}

            QTabBar::tab {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background: transparent;  
                color: {COLORS['text_dim']};
                border: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 12px 24px; 
                min-width: 80px;
                margin-right: 4px;
                font-weight: 500;
                font-size: {FONT_SIZES['normal']};
            }}

            QTabBar::tab:hover {{
                background: {COLORS['background_hover']};  
                color: {COLORS['foreground']};
            }}

            QTabBar::tab:selected {{
                background: {COLORS['background_light']}; 
                color: #ffffff;  
                font-weight: 600;
                border-bottom: 3px solid {COLORS['accent']};  
            }}

            /* ========== BUTTONS ========== */
            QPushButton {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['accent_hover']}, 
                    stop:1 {COLORS['accent']}
                );  
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;  
                font-size: {FONT_SIZES['normal']};
            }}
            
            QPushButton:hover {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['accent_light']}, 
                    stop:1 {COLORS['accent_hover']}
                );  
            }}
            
            QPushButton:pressed {{
                background-color: #002952;
                padding: 11px 20px 9px 20px;  
            }}
            
            QPushButton:disabled {{
                background-color: #2a2f36;  
                color: {COLORS['text_muted']};  
            }}

            /* ========== INPUT FIELDS ========== */
            QLineEdit {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']}; 
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES['normal']};
                border: 2px solid {COLORS['border']}; 
                border-radius: 6px;
                padding: 8px 12px;
                selection-background-color: {COLORS['accent']};
            }}

            QLineEdit:focus {{
                border: 2px solid {COLORS['accent']};  
                background-color: {COLORS['background_light']};
            }}
            
            QLineEdit:read-only {{
                background-color: {COLORS['background_input']};
                color: {COLORS['text_dim']};
            }}

            /* ========== COMBO BOX ========== */
            QComboBox {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']}; 
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES['normal']};
                border: 2px solid {COLORS['border']}; 
                border-radius: 6px;
                padding: 8px 12px;
                selection-background-color: {COLORS['accent']};
            }}

            QComboBox:focus {{
                border: 2px solid {COLORS['accent']};  
                background-color: {COLORS['background_light']};
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 2px solid {COLORS['border']};  
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: {COLORS['background_hover']}; 
            }}

            QComboBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['text_dim']};
                margin-right: 8px;
            }}
            
            QComboBox::down-arrow:hover {{
                border-top: 6px solid {COLORS['foreground']};
            }}

            QComboBox QAbstractItemView {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                border: 2px solid {COLORS['accent']};  
                background-color: {COLORS['background_light']};  
                selection-background-color: {COLORS['accent']};
                selection-color: #ffffff;
                border-radius: 6px;
                padding: 4px;
            }}
            
            QComboBox:disabled {{
                background-color: {COLORS['background_input']};
                color: {COLORS['text_muted']};
                border-color: {COLORS['background_hover']};
            }}

            /* ========== SPIN BOX ========== */
            QSpinBox, QDoubleSpinBox {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']}; 
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES['normal']};
                border: 2px solid {COLORS['border']}; 
                border-radius: 6px;
                padding: 8px 12px;
                selection-background-color: {COLORS['accent']};
            }}

            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {COLORS['accent']};  
                background-color: {COLORS['background_light']};
            }}

            /* ========== CHECKBOX ========== */
            QCheckBox {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                spacing: 10px;
                font-size: {FONT_SIZES['normal']};
            }}

            QCheckBox::indicator {{
                width: 20px;  
                height: 20px;
                border: 2px solid {COLORS['border']};
                border-radius: 4px; 
                background-color: {COLORS['background']};
            }}

            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent']};  
                border: 2px solid {COLORS['accent']};
            }}

            QCheckBox::indicator:hover {{
                border: 2px solid {COLORS['accent_hover']};
            }}
            
            QCheckBox:disabled {{
                color: {COLORS['text_muted']};
            }}
            
            QCheckBox::indicator:disabled {{
                background-color: {COLORS['background_input']};
                border-color: {COLORS['border']};
            }}

            /* ========== RADIO BUTTON ========== */
            QRadioButton {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                spacing: 10px;
                font-size: {FONT_SIZES['normal']};
            }}

            QRadioButton::indicator {{
                width: 18px;  
                height: 18px;
                border: 2px solid {COLORS['border']};
                border-radius: 10px; 
                background-color: {COLORS['background']};
            }}

            QRadioButton::indicator:checked {{
                background-color: {COLORS['accent']};  
                border: 2px solid {COLORS['accent']};
            }}

            QRadioButton::indicator:hover {{
                border: 2px solid {COLORS['accent_hover']};
            }}

            /* ========== SCROLL BAR ========== */
            QScrollBar:vertical {{
                border: none;
                background: {COLORS['background']};
                width: 10px; 
                margin: 2px;
                border-radius: 5px;
            }}

            QScrollBar::handle:vertical {{
                background: {COLORS['border']};
                min-height: 30px;
                border-radius: 5px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: {COLORS['accent']}; 
            }}

            QScrollBar:horizontal {{
                border: none;
                background: {COLORS['background']};
                height: 10px; 
                margin: 2px;
                border-radius: 5px;
            }}

            QScrollBar::handle:horizontal {{
                background: {COLORS['border']};
                min-width: 30px;
                border-radius: 5px;
            }}

            QScrollBar::handle:horizontal:hover {{
                background: {COLORS['accent']}; 
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                height: 0px;
                width: 0px;
            }}
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
            }}

            /* ========== GROUP BOX ========== */
            QGroupBox {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                border: 2px solid {COLORS['border']};  
                border-radius: 8px;
                margin-top: 24px;
                padding-top: 20px;  
                font-weight: 600;
                font-size: {FONT_SIZES['medium']};  
                background-color: {COLORS['background_light']};
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 16px;
                padding: 4px 12px;
                color: {COLORS['accent']};  
                background-color: {COLORS['background']};
                border-radius: 4px;
            }}

            /* ========== PROGRESS BAR ========== */
            QProgressBar {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                border: none;
                border-radius: 6px;
                background-color: {COLORS['background_hover']};
                text-align: center;
                color: white;
                font-size: {FONT_SIZES['small']};
                font-weight: 500;
            }}

            QProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']},
                    stop:1 {COLORS['accent_light']}
                );  
                border-radius: 6px;
            }}

            /* ========== SLIDER ========== */
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {COLORS['border']};
                border-radius: 3px;
            }}

            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}

            QSlider::handle:horizontal:hover {{
                background: {COLORS['accent_light']};
            }}

            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 3px;
            }}

            /* ========== TOOLTIP ========== */
            QToolTip {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background_hover']};  
                color: #ffffff; 
                border: 1px solid {COLORS['accent']};  
                border-radius: 6px;
                padding: 8px 12px;  
                font-size: {FONT_SIZES['small']};
            }}

            /* ========== SPLITTER ========== */
            QSplitter::handle {{
                background-color: {COLORS['border']};  
                width: 2px;  
                height: 2px;
            }}

            QSplitter::handle:hover {{
                background-color: {COLORS['accent']};
            }}

            /* ========== TABLE VIEW ========== */
            QTableView, QTableWidget {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
                alternate-background-color: {COLORS['background_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                gridline-color: {COLORS['border']};
                selection-background-color: {COLORS['accent']};
                font-size: {FONT_SIZES['small']};
            }}

            QTableView::item, QTableWidget::item {{
                padding: 8px;
            }}

            QHeaderView::section {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background_hover']};
                color: {COLORS['foreground']};
                padding: 10px;
                border: none;
                border-bottom: 2px solid {COLORS['accent']};
                font-weight: 600;
                font-size: {FONT_SIZES['small']};
            }}

            /* ========== LIST VIEW ========== */
            QListView, QListWidget {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                selection-background-color: {COLORS['accent']};
                font-size: {FONT_SIZES['normal']};
            }}

            QListView::item, QListWidget::item {{
                padding: 8px;
                border-radius: 4px;
            }}

            QListView::item:hover, QListWidget::item:hover {{
                background-color: {COLORS['background_hover']};
            }}

            QListView::item:selected, QListWidget::item:selected {{
                background-color: {COLORS['accent']};
            }}

            /* ========== MENU ========== */
            QMenuBar {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
                color: {COLORS['foreground']};
                border-bottom: 1px solid {COLORS['border']};
                font-size: {FONT_SIZES['normal']};
            }}

            QMenuBar::item {{
                padding: 8px 16px;
                background: transparent;
            }}

            QMenuBar::item:selected {{
                background-color: {COLORS['background_hover']};
            }}

            QMenu {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 4px;
                font-size: {FONT_SIZES['normal']};
            }}

            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
            }}

            QMenu::item:selected {{
                background-color: {COLORS['accent']};
            }}

            QMenu::separator {{
                height: 1px;
                background: {COLORS['border']};
                margin: 4px 8px;
            }}

            /* ========== DIALOG ========== */
            QDialog {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
            }}

            QMessageBox {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
            }}

            QMessageBox QLabel {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                color: {COLORS['foreground']};
                font-size: {FONT_SIZES['normal']};
            }}

            /* ========== TEXT EDIT ========== */
            QTextEdit, QPlainTextEdit {{
                font-family: '{font_family}', 'Consolas', monospace;
                background-color: {COLORS['background']};
                color: {COLORS['foreground']};
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
                font-size: {FONT_SIZES['small']};
                selection-background-color: {COLORS['accent']};
            }}

            QTextEdit:focus, QPlainTextEdit:focus {{
                border: 2px solid {COLORS['accent']};
            }}

            /* ========== STATUS BAR ========== */
            QStatusBar {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background_light']};
                color: {COLORS['text_dim']};
                border-top: 1px solid {COLORS['border']};
                font-size: {FONT_SIZES['small']};
            }}

            /* ========== TOOL BAR ========== */
            QToolBar {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background_light']};
                border: none;
                spacing: 4px;
                padding: 4px;
            }}

            QToolButton {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }}

            QToolButton:hover {{
                background-color: {COLORS['background_hover']};
            }}

            QToolButton:pressed {{
                background-color: {COLORS['accent']};
            }}

            /* ========== TREE WIDGET ========== */
            QTreeWidget, QTreeView {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                selection-background-color: {COLORS['accent']};
                font-size: {FONT_SIZES['normal']};
            }}

            QTreeWidget::item, QTreeView::item {{
                padding: 4px;
            }}

            QTreeWidget::item:hover, QTreeView::item:hover {{
                background-color: {COLORS['background_hover']};
            }}

            QTreeWidget::item:selected, QTreeView::item:selected {{
                background-color: {COLORS['accent']};
            }}

            /* ========== FRAME ========== */
            QFrame {{
                font-family: '{font_family}', 'Segoe UI', sans-serif;
            }}

            QFrame[frameShape="4"] {{ /* HLine */
                background-color: {COLORS['border']};
                max-height: 1px;
            }}

            QFrame[frameShape="5"] {{ /* VLine */
                background-color: {COLORS['border']};
                max-width: 1px;
            }}
        """
    )


# =============================================================================
# PYQTGRAPH PLOT STYLING
# =============================================================================

def style_plot_widget(
    plot_widget: pg.PlotWidget,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    font_family: str = None
) -> pg.PlotWidget:
    """
    Apply consistent styling to any pyqtgraph PlotWidget.
    
    Args:
        plot_widget: The pyqtgraph PlotWidget to style
        title: Optional title for the plot
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        font_family: Font family to use (default: FONT_FAMILY)
    
    Returns:
        The styled plot widget
    """
    if font_family is None:
        font_family = FONT_FAMILY
    
    # Fonts
    title_font = QFont(font_family, 14, QFont.DemiBold)
    label_font = QFont(font_family, 12, QFont.Medium)
    tick_font = QFont(font_family, 10)
    
    # Background
    plot_widget.setBackground(COLORS['background'])
    
    # Configure axes
    for axis_name in ['left', 'bottom']:
        axis = plot_widget.getAxis(axis_name)
        axis.setPen(pg.mkPen(color=COLORS['border'], width=1))
        axis.setTextPen(pg.mkPen(color=COLORS['foreground']))
        axis.setTickFont(tick_font)
        axis.label.setFont(label_font)
    
    # Labels
    label_style = {
        'color': COLORS['foreground'],
        'font-size': FONT_SIZES['xlarge'],
        'font-family': font_family
    }
    
    if y_label:
        plot_widget.setLabel('left', y_label, **label_style)
    if x_label:
        plot_widget.setLabel('bottom', x_label, **label_style)
    
    # Title
    if title:
        plot_widget.setTitle(title, color=COLORS['foreground'], size=FONT_SIZES['title'])
        plot_widget.getPlotItem().titleLabel.item.setFont(title_font)
    
    # Grid
    plot_widget.showGrid(x=True, y=True, alpha=0.2)
    
    return plot_widget


def get_plot_colors(n: int = 10) -> list:
    """
    Get a list of distinct colors for plotting multiple traces.
    
    Args:
        n: Number of colors needed
    
    Returns:
        List of hex color strings
    """
    import colorsys
    
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def get_pen(color_name: str = 'accent', width: int = 2) -> pg.mkPen:
    """
    Get a styled pen for plotting.
    
    Args:
        color_name: Key from COLORS dict (e.g., 'accent', 'success', 'error')
        width: Line width in pixels
    
    Returns:
        pyqtgraph pen object
    """
    color = COLORS.get(color_name, COLORS['accent'])
    return pg.mkPen(color=color, width=width)


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., '#FF5733' or 'FF5733')
    
    Returns:
        Tuple of (r, g, b) values (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))