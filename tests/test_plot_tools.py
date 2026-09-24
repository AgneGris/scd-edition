import os
from unittest.mock import MagicMock, patch

import numpy as np

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


def test_x_zoom_view_box_uses_source_plot_wheel_controls():
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from scd_app.gui.widgets.plot_tools import XZoomViewBox

    app = _application()
    view_box = XZoomViewBox()
    event = MagicMock()

    event.modifiers.return_value = Qt.KeyboardModifier.NoModifier
    with patch.object(pg.ViewBox, "wheelEvent") as base_wheel:
        view_box.wheelEvent(event)
    base_wheel.assert_called_once_with(event, axis=0)

    event.reset_mock()
    event.modifiers.return_value = Qt.KeyboardModifier.ControlModifier
    with patch.object(pg.ViewBox, "wheelEvent") as base_wheel:
        view_box.wheelEvent(event)
    base_wheel.assert_called_once_with(event, axis=None)

    event.reset_mock()
    event.modifiers.return_value = Qt.KeyboardModifier.ShiftModifier
    event.delta.return_value = 120
    with patch.object(view_box, "translateBy") as translate:
        view_box.wheelEvent(event)
    translate.assert_called_once_with(x=-0.6, y=0)
    event.accept.assert_called_once_with()

    view_box.close()
    app.processEvents()


def test_stacked_muap_views_use_x_axis_zoom():
    import pyqtgraph as pg

    from scd_app.gui.tabs.edition_tab import EditionTab
    from scd_app.gui.widgets.muap_popout import MuapPopoutDialog
    from scd_app.gui.widgets.plot_tools import XZoomViewBox

    app = _application()
    waveform = np.array([0.0, 1.0, 0.0])

    tab = EditionTab()
    tab._render_muap_stacked([waveform], [0])
    embedded_plots = [
        item for item in tab.muap_widget.ci.items if isinstance(item, pg.PlotItem)
    ]
    assert len(embedded_plots) == 1
    assert isinstance(embedded_plots[0].getViewBox(), XZoomViewBox)

    popout = MuapPopoutDialog()
    popout.render_stacked([waveform], [0], 0)
    popout_plots = [
        item for item in popout._plot.ci.items if isinstance(item, pg.PlotItem)
    ]
    assert len(popout_plots) == 1
    assert isinstance(popout_plots[0].getViewBox(), XZoomViewBox)

    tab.close()
    popout.close()
    app.processEvents()
