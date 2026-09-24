"""Safe, reusable actions for PyQtGraph plots."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu, QToolButton

from scd_app.gui.style.styling import COLORS


class XZoomViewBox(pg.ViewBox):
    """ViewBox whose default wheel interaction changes only the time axis."""

    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            delta = ev.delta()
            self.translateBy(x=-delta / 200.0, y=0)
            ev.accept()
        elif mods & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(ev, axis=None)
        else:
            super().wheelEvent(ev, axis=0)


def make_plot_item_safe(plot_item: pg.PlotItem) -> None:
    """Remove PyQtGraph's state-changing transform submenu from a plot.

    ViewBox navigation and the scene's Export action remain available. This
    specifically removes options such as FFT mode, derivative mode, log axes,
    and Y vs Y', all of which mutate the live plot and can disrupt linked views.
    """

    plot_item.setMenuEnabled(False, enableViewBoxMenu=True)


class SafePlotWidget(pg.PlotWidget):
    """PlotWidget without generic transforms and with safe plot actions."""

    def __init__(self, *args, **kwargs):
        # GraphicsView invokes resizeEvent during its own constructor.
        self._actions_button: QToolButton | None = None
        super().__init__(*args, **kwargs)
        make_plot_item_safe(self.plotItem)

        self._actions_button = QToolButton(self)
        self._actions_button.setText("...")
        self._actions_button.setToolTip("Plot actions")
        self._actions_button.setAccessibleName("Plot actions")
        self._actions_button.setFixedSize(28, 22)
        self._actions_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._actions_button.setStyleSheet(
            f"""
            QToolButton {{
                background-color: {COLORS["background_light"]};
                color: {COLORS["text_dim"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 0px;
            }}
            QToolButton:hover {{
                color: {COLORS["foreground"]};
                border-color: {COLORS["info"]};
            }}
            """
        )

        menu = QMenu(self._actions_button)
        view_all = menu.addAction("View all")
        view_all.triggered.connect(self.autoRange)
        menu.addSeparator()
        export = menu.addAction("Export plot…")
        export.triggered.connect(self._show_export_dialog)
        self._actions_button.setMenu(menu)
        self._actions_button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        button = self._actions_button
        if button is None:
            return
        button.move(
            self.width() - button.width() - 6,
            6,
        )
        button.raise_()

    def _show_export_dialog(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        scene.contextMenuItem = self.plotItem
        scene.showExportDialog()
