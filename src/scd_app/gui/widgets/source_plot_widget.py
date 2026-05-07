"""
source_plot_widget.py — EMG source signal plot and related widgets.

Contains:
    SelectionArm         — rubberband selection state constants
    XZoomViewBox         — Shift+scroll pans, plain scroll zooms X only
    _AuxLegend           — click-to-toggle floating AUX force legend
    SourcePlotWidget     — main source/IPT plot with point-click and rubberband editing
    FiringRatePlotWidget — instantaneous firing rate plot
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtWidgets import QRubberBand

from scd_app.core.mu_model import EditMode
from scd_app.gui.style.styling import COLORS

_MIN_RUBBERBAND_PX = 5  # drags smaller than this are ignored in selection-arm mode

# ── AUX colour palette ────────────────────────────────────────────────────────
_AUX_COLORS_HEX = ["#FFD700", "#C0EFFF", "#FFB347"]
_AUX_COLORS_RGB = [(255, 215, 0), (192, 239, 255), (255, 179, 71)]

# ── Filename → active-aux helpers ────────────────────────────────────────────
_FINGER_ABBREV: Dict[str, str] = {
    "T": "Thumb", "I": "Index", "M": "Middle", "R": "Ring", "L": "Little"
}
_MOTION_ABBREV: Dict[str, str] = {"ext": "Ext", "flex": "Flex"}
_TASK_PATTERN = re.compile(r"mvc-\d+(ext|flex)_fing-([TIMRL]+)", re.IGNORECASE)


def _parse_task_targets(file_stem: str) -> Optional[Set[str]]:
    """Return expected active unit strings from the filename, or None."""
    m = _TASK_PATTERN.search(file_stem)
    if not m:
        return None
    motion = _MOTION_ABBREV.get(m.group(1).lower())
    if not motion:
        return None
    targets = {
        f"{_FINGER_ABBREV[c]} {motion}"
        for c in m.group(2).upper()
        if c in _FINGER_ABBREV
    }
    return targets or None


def _default_aux_states(channels: list, file_stem: str) -> List[bool]:
    """Return initial on/off states for aux channels based on the filename."""
    targets = _parse_task_targets(file_stem)
    if not targets:
        return [True] * len(channels)
    states = []
    for ch in channels:
        meta = ch.get("meta", {})
        unit = meta.get("unit") or ch.get("unit") or ""
        states.append(unit in targets)
    return states if any(states) else [True] * len(channels)


class SelectionArm:
    """Which (if any) rubberband-selection operation is currently armed."""
    NONE = "none"
    ADD = "add"
    DELETE = "delete"


class XZoomViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & Qt.ShiftModifier:
            delta = ev.delta()
            self.translateBy(x=-delta / 200.0, y=0)
            ev.accept()
        elif mods & Qt.ControlModifier:
            super().wheelEvent(ev, axis=None)
        else:
            super().wheelEvent(ev, axis=0)


class _AuxLegend(pg.LegendItem):
    """Floating click-to-toggle legend for AUX force traces.

    Labels are added directly to the grid layout (no ItemSample swatch).
    Toggling uses pen-alpha (70 → 5) rather than setVisible to avoid the
    pyqtgraph 'hidden' eye-slash glyph.
    """

    def __init__(self):
        super().__init__(offset=(-10, 10))
        self._curves: list = []
        self._names: list = []
        self._colors: list = []
        self._on_states: list = []

    def clear(self):
        for _sample, label in self.items:
            self.layout.removeItem(label)
            label.close()
        self.items = []
        self.updateSize()

    def populate(self, channels: list, curves: list, initial_states: Optional[List[bool]] = None):
        self.clear()
        self._curves = list(curves)
        self._names = []
        self._colors = []
        self._on_states = list(initial_states) if initial_states is not None else [True] * len(channels)
        for i, (ch, _curve) in enumerate(zip(channels, curves)):
            meta = ch.get("meta", {})
            name = meta.get("name") or ch.get("name") or f"AUX {i + 1}"
            unit = meta.get("unit") or ch.get("unit") or ""
            display = f"{name} ({unit})" if unit else name
            on = self._on_states[i] if i < len(self._on_states) else True
            color = _AUX_COLORS_HEX[i % len(_AUX_COLORS_HEX)]
            self._names.append(display)
            self._colors.append(color)
            dot = "●" if on else "○"
            label_color = color if on else "#555555"
            label = pg.LabelItem(f"{dot} {display}", color=label_color, justify="left")
            self.layout.addItem(label, i, 0)
            self.items.append((None, label))
        self.updateSize()
        self.setVisible(bool(channels))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            y = event.pos().y()
            for i, (_sample, label) in enumerate(self.items):
                rect = label.mapRectToParent(label.boundingRect())
                if rect.top() <= y <= rect.bottom():
                    self._toggle(i)
                    event.accept()
                    return
        event.accept()

    def _toggle(self, idx: int):
        self._on_states[idx] = not self._on_states[idx]
        on = self._on_states[idx]
        r, g, b = _AUX_COLORS_RGB[idx % len(_AUX_COLORS_RGB)]
        alpha = 70 if on else 5
        self._curves[idx].setPen(pg.mkPen(color=(r, g, b, alpha), width=2.5))
        _sample, label = self.items[idx]
        dot = "●" if on else "○"
        color = self._colors[idx] if on else "#555555"
        label.setText(f"{dot} {self._names[idx]}", color=color)
        self.update()


class SourcePlotWidget(pg.PlotWidget):
    """
    Source-signal plot with two independent interaction modes:

    1. Point-click editing  (EditMode.ADD / DELETE via set_edit_mode)
       Click  →  spike_add_requested / spike_delete_requested signal.

    2. Rubberband-drag selection  (armed via set_selection_arm)
       Drag  →  region_selected(x1, x2, y1, y2) fires on mouse-release.
       Stays armed until explicitly disarmed.  Each drag fires immediately.
    """

    spike_add_requested = pyqtSignal(int)
    spike_delete_requested = pyqtSignal(int)
    region_selected = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(
            parent, background=COLORS["background"], viewBox=XZoomViewBox()
        )
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left", "Amplitude", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))

        self._fsamp = 1.0
        self._edit_mode = EditMode.VIEW
        self._sel_arm = SelectionArm.NONE

        self._source: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None

        self._signal_curve = self.plot([], pen=pg.mkPen("#2b6cb0", width=1))
        self._signal_curve.setDownsampling(auto=True, method="peak")
        self._signal_curve.setClipToView(True)
        self._spike_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush("#ed8936"),
            symbol="o", hoverable=True,
        )
        self.addItem(self._spike_scatter)
        self._plateau_region: Optional[pg.LinearRegionItem] = None

        self._aux_curves: list = []
        self._aux_raw: list = []
        self._legend = _AuxLegend()
        self._legend.setParentItem(self.plotItem.vb)

        self._rb_widget: Optional[QRubberBand] = None
        self._rb_origin: Optional[QPoint] = None

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def set_edit_mode(self, mode: EditMode):
        self._edit_mode = mode
        self._apply_cursor()

    def set_selection_arm(self, arm: str):
        self._sel_arm = arm
        self._apply_cursor()
        if arm == SelectionArm.NONE:
            self._cancel_rubberband()

    def _apply_cursor(self):
        if self._sel_arm != SelectionArm.NONE:
            self.setCursor(Qt.CrossCursor)
        else:
            cursors = {
                EditMode.VIEW: Qt.ArrowCursor,
                EditMode.ADD: Qt.CrossCursor,
                EditMode.DELETE: Qt.PointingHandCursor,
            }
            self.setCursor(cursors.get(self._edit_mode, Qt.ArrowCursor))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_data(self, source: np.ndarray, timestamps: np.ndarray):
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
        self._source = source**2
        self._timestamps = timestamps
        t = np.arange(len(self._source)) / self._fsamp
        self._signal_curve.setData(t, self._source)
        self._update_spike_markers()
        self._redraw_aux()

    def set_plateau_region(self, start_sample: int, end_sample: int):
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
        t0 = start_sample / self._fsamp
        t1 = end_sample / self._fsamp
        self._plateau_region = pg.LinearRegionItem(
            values=(t0, t1),
            movable=False,
            brush=pg.mkBrush(137, 180, 250, 20),
            pen=pg.mkPen(color=(137, 180, 250, 60), width=1, style=Qt.DashLine),
        )
        self._plateau_region.setZValue(-10)
        self.addItem(self._plateau_region)

    def set_aux_data(self, channels: list, fsamp: float, file_stem: str = ""):
        for curve in self._aux_curves:
            self.removeItem(curve)
        self._aux_curves.clear()
        self._aux_raw.clear()
        initial_states = _default_aux_states(channels, file_stem)
        for i, ch in enumerate(channels):
            raw = np.asarray(ch["data"]).squeeze()
            raw = np.nan_to_num(raw, nan=0.0)
            self._aux_raw.append(raw)
            r, g, b = _AUX_COLORS_RGB[i % len(_AUX_COLORS_RGB)]
            on = initial_states[i] if i < len(initial_states) else True
            alpha = 70 if on else 5
            curve = pg.PlotCurveItem(pen=pg.mkPen(color=(r, g, b, alpha), width=2.5))
            curve.setZValue(-20)
            self.addItem(curve)
            self._aux_curves.append(curve)
        self._legend.populate(channels, self._aux_curves, initial_states)
        self._redraw_aux()

    def _redraw_aux(self):
        if self._source is not None and len(self._source) > 0:
            src_min = float(self._source.min())
            src_max = float(self._source.max())
        else:
            src_min, src_max = 0.0, 1.0
        src_range = max(src_max * 1.05 - src_min, 1e-9)
        n_src = len(self._source) if self._source is not None else 0
        for raw, curve in zip(self._aux_raw, self._aux_curves):
            sig = raw - float(raw.min())
            sig_range = max(float(sig.max()), 1e-9)
            sig_scaled = (sig / sig_range) * src_range + src_min
            n_plot = min(len(sig_scaled), n_src) if n_src > 0 else len(sig_scaled)
            step = max(1, n_plot // 2000)
            t = np.arange(0, n_plot, step) / self._fsamp
            curve.setData(t, sig_scaled[:n_plot:step])

    def update_timestamps(self, timestamps: np.ndarray):
        self._timestamps = timestamps
        self._update_spike_markers()

    def clear_data(self):
        self._signal_curve.setData([], [])
        self._spike_scatter.setData([], [])
        for curve in self._aux_curves:
            curve.setData([], [])
        self._source = None
        self._timestamps = None
        if self._plateau_region is not None:
            self.removeItem(self._plateau_region)
            self._plateau_region = None
        self._cancel_rubberband()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            super().mousePressEvent(ev)
            return
        if self._sel_arm != SelectionArm.NONE:
            self._rb_origin = ev.pos()
            if self._rb_widget is None:
                self._rb_widget = QRubberBand(QRubberBand.Rectangle, self)
            self._rb_widget.setGeometry(QRect(self._rb_origin, QSize()))
            self._rb_widget.show()
            ev.accept()
        elif self._edit_mode in (EditMode.ADD, EditMode.DELETE):
            self._rb_origin = ev.pos()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if (self._rb_widget is not None
                and self._rb_widget.isVisible()
                and self._rb_origin is not None):
            self._rb_widget.setGeometry(QRect(self._rb_origin, ev.pos()).normalized())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._rb_origin is None:
            super().mouseReleaseEvent(ev)
            return

        origin = self._rb_origin
        self._rb_origin = None

        if self._sel_arm != SelectionArm.NONE and self._rb_widget is not None:
            rect_px = QRect(origin, ev.pos()).normalized()
            self._rb_widget.hide()
            if rect_px.width() > _MIN_RUBBERBAND_PX and rect_px.height() > _MIN_RUBBERBAND_PX:
                vb = self.getViewBox()
                tl = vb.mapSceneToView(self.mapToScene(rect_px.topLeft()))
                br = vb.mapSceneToView(self.mapToScene(rect_px.bottomRight()))
                x1, x2 = sorted([tl.x(), br.x()])
                y1, y2 = sorted([tl.y(), br.y()])
                self.region_selected.emit(x1, x2, y1, y2)
            ev.accept()
            return

        if self._edit_mode in (EditMode.ADD, EditMode.DELETE):
            rect_px = QRect(origin, ev.pos()).normalized()
            if rect_px.width() <= _MIN_RUBBERBAND_PX and rect_px.height() <= _MIN_RUBBERBAND_PX:
                vb = self.getViewBox()
                pos = vb.mapSceneToView(self.mapToScene(ev.pos()))
                sample = int(pos.x() * self._fsamp)
                if self._edit_mode == EditMode.ADD:
                    self.spike_add_requested.emit(sample)
                elif self._edit_mode == EditMode.DELETE:
                    self.spike_delete_requested.emit(sample)
            ev.accept()
            return

        super().mouseReleaseEvent(ev)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_rubberband(self):
        self._rb_origin = None
        if self._rb_widget is not None:
            self._rb_widget.hide()

    def _update_spike_markers(self):
        if (self._source is None
                or self._timestamps is None
                or len(self._timestamps) == 0):
            self._spike_scatter.setData([], [])
            return
        valid = self._timestamps[self._timestamps < len(self._source)]
        if len(valid) == 0:
            self._spike_scatter.setData([], [])
            return
        self._spike_scatter.setData(valid / self._fsamp, self._source[valid])


class FiringRatePlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background=COLORS["background"])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Time (s)", color=COLORS.get("text_dim", "#6c7086"))
        self.setLabel("left", "IFR (Hz)", color=COLORS.get("text_dim", "#6c7086"))
        for axis in ("bottom", "left"):
            self.getAxis(axis).setPen(COLORS.get("text_dim", "#6c7086"))
            self.getAxis(axis).setTextPen(COLORS.get("text_dim", "#6c7086"))
        self._curve = self.plot([], pen=pg.mkPen(COLORS["warning"], width=1.5))
        self._curve.setDownsampling(auto=True, method="peak")
        self._curve.setClipToView(True)
        self._fsamp = 1.0

    def set_fsamp(self, fsamp: float):
        self._fsamp = fsamp

    def link_x(self, other: pg.PlotWidget):
        self.setXLink(other)

    def set_data(self, timestamps: np.ndarray):
        ts = np.sort(timestamps)
        if len(ts) < 2:
            self._curve.setData([], [])
            return
        isi = np.diff(ts) / self._fsamp
        ifr = np.where(isi > 0.01, 1.0 / isi, 0.0)
        t_mid = (ts[:-1] + ts[1:]) / 2 / self._fsamp
        self._curve.setData(t_mid, ifr)

    def clear_data(self):
        self._curve.setData([], [])
