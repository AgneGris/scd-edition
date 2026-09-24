import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_safe_plot_removes_transforms_and_keeps_safe_actions():
    from scd_app.gui.widgets.plot_tools import SafePlotWidget

    app = _application()
    widget = SafePlotWidget()

    assert widget.plotItem.menuEnabled() is False
    assert widget.getViewBox().menuEnabled() is True
    action_names = [action.text() for action in widget._actions_button.menu().actions()]
    assert action_names == ["View all", "", "Export plot…"]
    assert all("spectrum" not in name.lower() for name in action_names)

    widget.close()
    app.processEvents()


def test_plot_actions_button_stylesheet_parses_without_qt_warnings():
    from PySide6.QtCore import qInstallMessageHandler

    from scd_app.gui.widgets.plot_tools import SafePlotWidget

    app = _application()
    messages = []

    def record_message(_message_type, _context, message):
        messages.append(message)

    previous_handler = qInstallMessageHandler(record_message)
    widget = None
    try:
        widget = SafePlotWidget()
        widget.show()
        app.processEvents()
    finally:
        if widget is not None:
            widget.close()
        qInstallMessageHandler(previous_handler)

    assert not any("Could not parse stylesheet" in message for message in messages)
