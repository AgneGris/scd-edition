"""
muap_popout.py — Floating MUAP shapes window.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from scd_app.gui.style.styling import COLORS, FONT_FAMILY


class MuapPopoutDialog(QDialog):
    """Floating window that mirrors the MUAP panel and live-updates with MU selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MUAP Shapes")
        self.resize(850, 620)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._plot = pg.GraphicsLayoutWidget()
        self._plot.setBackground(COLORS["background"])
        lay.addWidget(self._plot)

    def render_grid(
        self,
        muap_grid: np.ndarray,
        grid_cfg: dict,
        rejected_positions: set,
        mu_idx: int,
    ):
        self._plot.clear()
        rows, cols = grid_cfg["grid_shape"]
        positions = grid_cfg["positions"]
        electrode_positions = set(positions.values())

        valid_wavs = [
            muap_grid[r, c]
            for r in range(min(rows, muap_grid.shape[0]))
            for c in range(min(cols, muap_grid.shape[1]))
            if (r, c) in electrode_positions
            and (r, c) not in rejected_positions
            and len(muap_grid[r, c]) > 0
            and np.any(muap_grid[r, c] != 0)
        ]
        amp = np.max(np.abs(np.concatenate(valid_wavs))) * 1.2 if valid_wavs else 1.0
        n_samples = muap_grid.shape[2] if muap_grid.ndim == 3 else 409

        label = (
            f"<span style='color:{COLORS['foreground']};font-size:11pt;'>"
            f"MU {mu_idx}</span>"
        )
        self._plot.addLabel(label, row=0, col=0, colspan=cols + 1, justify="center")

        lbl_style = f"color:{COLORS.get('text_dim','#6c7086')}; font-size:8pt;"

        def _add_lbl(text, row, col, **kw):
            lbl = self._plot.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        _add_lbl(f"<span style='{lbl_style}'></span>", 1, 0, justify="center")
        for c in range(cols):
            _add_lbl(f"<span style='{lbl_style}'><b>{c + 1}</b></span>", 1, c + 1, justify="center")
        for r in range(rows):
            _add_lbl(f"<span style='{lbl_style}'><b>{r + 1}</b></span>", r + 2, 0, justify="center")

        _rej_bg = (50, 30, 30)
        _empty_bg = (28, 28, 28)
        gl = self._plot.ci.layout
        cell_plots: Dict[Tuple[int, int], object] = {}

        for r in range(rows):
            for c in range(cols):
                p = self._plot.addPlot(row=r + 2, col=c + 1)
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.setMouseEnabled(x=False, y=False)
                p.enableAutoRange(enable=False)
                p.setYRange(-amp, amp, padding=0)
                p.setXRange(0, n_samples, padding=0)
                p.setLimits(xMin=0, xMax=n_samples, yMin=-amp, yMax=amp)
                p.setMinimumWidth(0)
                p.setMinimumHeight(0)
                cell_plots[(r, c)] = p
                rc = (r, c)
                if rc in rejected_positions:
                    p.getViewBox().setBackgroundColor(_rej_bg)
                    p.plot([0, n_samples], [0, 0], pen=pg.mkPen(color=(140, 60, 60), width=1))
                elif rc not in electrode_positions:
                    p.getViewBox().setBackgroundColor(_empty_bg)

        gl.setSpacing(0)
        gl.setHorizontalSpacing(6)
        gl.setColumnMinimumWidth(0, 14)
        gl.setColumnStretchFactor(0, 0)
        for c in range(cols):
            gl.setColumnMinimumWidth(c + 1, 0)
            gl.setColumnStretchFactor(c + 1, 1)
        for r in range(2):
            gl.setRowMinimumHeight(r, 0)
            gl.setRowStretchFactor(r, 0)
        for r in range(rows):
            gl.setRowMinimumHeight(r + 2, 0)
            gl.setRowStretchFactor(r + 2, 1)

        for r in range(min(rows, muap_grid.shape[0])):
            for c in range(min(cols, muap_grid.shape[1])):
                rc = (r, c)
                if rc not in electrode_positions or rc in rejected_positions:
                    continue
                wav = muap_grid[r, c]
                if len(wav) == 0 or not np.any(wav != 0):
                    continue
                p = cell_plots.get(rc)
                if p is not None:
                    p.plot(wav, pen=pg.mkPen(color=COLORS["info"], width=1.5))

        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx}")

    def render_stacked(self, waveforms, ch_indices, mu_idx):
        self._plot.clear()
        plot = self._plot.addPlot(row=0, col=0)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return
        all_data = np.concatenate([w for _, w in valid])
        spacing = np.max(np.abs(all_data)) * 0.6 if len(all_data) > 0 else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["foreground"], width=1.5))
            txt = pg.TextItem(f"Ch {ch}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 8))
            plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        plot.setTitle(f"MU {mu_idx} — Stacked", color=COLORS["foreground"], size="11pt")
        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx} (Stacked)")

    def clear(self, message="Select a Motor Unit"):
        self._plot.clear()
        p = self._plot.addPlot(row=0, col=0)
        p.hideAxis("left")
        p.hideAxis("bottom")
        p.setMouseEnabled(x=False, y=False)
        p.setXRange(0, 1)
        p.setYRange(0, 1)
        t = pg.TextItem(message, color=(120, 120, 120), anchor=(0.5, 0.5))
        t.setFont(QFont(FONT_FAMILY, 14))
        t.setPos(0.5, 0.5)
        p.addItem(t)
        self.setWindowTitle("MUAP Shapes")
