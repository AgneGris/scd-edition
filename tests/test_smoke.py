import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_constructs_headlessly():
    import scd_app
    from scd_app.gui.main_window import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert scd_app is not None
    assert window.windowTitle() == "SCD - EMG Decomposition & Edition"
    assert window.tabs.count() == 4

    window.close()
    app.processEvents()
