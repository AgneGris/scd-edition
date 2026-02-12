#!/usr/bin/env python3
"""

"""

import sys
import argparse
from pathlib import Path


def run_gui():
    """Launch the GUI application."""
    from gui.main_window import main
    main()



if __name__ == "__main__":
    run_gui()
