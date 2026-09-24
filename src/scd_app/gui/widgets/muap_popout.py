"""
muap_popout.py — Floating MUAP shapes window.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QVBoxLayout

from scd_app.core.spike_muap import SpikeMUAPInspection
from scd_app.gui.style.styling import COLORS, FONT_FAMILY
from scd_app.gui.widgets.plot_tools import make_plot_item_safe


class MuapPopoutDialog(QDialog):
    """Floating window that mirrors the MUAP panel and live-updates with MU selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MUAP Shapes")
        self.resize(850, 620)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
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
        *,
        inspection: SpikeMUAPInspection | None = None,
        fsamp: float = 1.0,
        show_selected: bool = True,
        remove_other_units: bool = False,
    ):
        self._plot.clear()
        rows, cols = grid_cfg["grid_shape"]
        positions = grid_cfg["positions"]
        electrode_positions = set(positions.values())
        reference_grid = (
            inspection.reference_grid if inspection is not None else muap_grid
        )
        selected_grid = (
            inspection.selected_view(remove_other_units)[0]
            if inspection is not None and show_selected
            else None
        )

        valid_wavs = [
            grid[r, c]
            for grid in (
                [reference_grid, selected_grid]
                if selected_grid is not None
                else [reference_grid]
            )
            for r in range(min(rows, grid.shape[0]))
            for c in range(min(cols, grid.shape[1]))
            if (r, c) in electrode_positions
            and (r, c) not in rejected_positions
            and len(grid[r, c]) > 0
            and np.any(np.isfinite(grid[r, c]) & (grid[r, c] != 0))
        ]
        all_values = np.concatenate(valid_wavs) if valid_wavs else np.array([])
        finite_values = all_values[np.isfinite(all_values)]
        amp = float(np.max(np.abs(finite_values))) * 1.2 if finite_values.size else 1.0
        n_samples = reference_grid.shape[2] if reference_grid.ndim == 3 else 409

        if inspection is None:
            label = (
                f"<span style='color:{COLORS['foreground']};font-size:11pt;'>"
                f"MU {mu_idx}</span>"
            )
        else:
            spike_time = inspection.selected_sample / fsamp
            _, similarity, amplitude_ratio, lag_ms = inspection.selected_view(
                remove_other_units
            )
            selected_label = (
                "selected spike" if show_selected else "selected spike hidden"
            )
            selected_mode = "earlier units removed" if remove_other_units else "raw EMG"
            label = (
                f"<span style='color:{COLORS['foreground']};font-size:11pt;'>"
                f"MU {mu_idx} · spike {spike_time:.3f} s · "
                f"r {similarity:.3f} · "
                f"amplitude {amplitude_ratio:.2f}× · "
                f"lag {lag_ms:+.2f} ms</span><br>"
                f"<span style='color:{COLORS['info']};font-size:9pt;'>"
                f"reference (other {inspection.n_reference_spikes})</span> · "
                f"<span style='color:#ed8936;font-size:9pt;'>"
                f"{selected_label} ({selected_mode})</span>"
            )
        self._plot.addLabel(label, row=0, col=0, colspan=cols + 1, justify="center")

        lbl_style = f"color:{COLORS.get('text_dim', '#6c7086')}; font-size:8pt;"

        def _add_lbl(text, row, col, **kw):
            lbl = self._plot.addLabel(text, row=row, col=col, **kw)
            lbl.setMinimumWidth(0)
            lbl.setMinimumHeight(0)
            return lbl

        _add_lbl(f"<span style='{lbl_style}'></span>", 1, 0, justify="center")
        for c in range(cols):
            _add_lbl(
                f"<span style='{lbl_style}'><b>{c + 1}</b></span>",
                1,
                c + 1,
                justify="center",
            )
        for r in range(rows):
            _add_lbl(
                f"<span style='{lbl_style}'><b>{r + 1}</b></span>",
                r + 2,
                0,
                justify="center",
            )

        _rej_bg = (50, 30, 30)
        _empty_bg = (28, 28, 28)
        gl = self._plot.ci.layout
        cell_plots: dict[tuple[int, int], object] = {}

        for r in range(rows):
            for c in range(cols):
                p = self._plot.addPlot(row=r + 2, col=c + 1)
                make_plot_item_safe(p)
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
                    p.plot(
                        [0, n_samples],
                        [0, 0],
                        pen=pg.mkPen(color=(140, 60, 60), width=1),
                    )
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

        for r in range(min(rows, reference_grid.shape[0])):
            for c in range(min(cols, reference_grid.shape[1])):
                rc = (r, c)
                if rc not in electrode_positions or rc in rejected_positions:
                    continue
                wav = reference_grid[r, c]
                if len(wav) == 0 or not np.any(np.isfinite(wav)):
                    continue
                p = cell_plots.get(rc)
                if p is not None:
                    p.plot(wav, pen=pg.mkPen(color=COLORS["info"], width=3.0))
                    if selected_grid is not None:
                        selected = selected_grid[r, c]
                        if len(selected) > 0 and np.any(np.isfinite(selected)):
                            p.plot(
                                selected,
                                pen=pg.mkPen(
                                    color=(237, 137, 54, 210),
                                    width=1.5,
                                    style=Qt.PenStyle.SolidLine,
                                ),
                            )

        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx}")

    def render_stacked(
        self,
        waveforms,
        ch_indices,
        mu_idx,
        *,
        selected_waveforms=None,
        inspection: SpikeMUAPInspection | None = None,
        fsamp: float = 1.0,
        remove_other_units: bool = False,
    ):
        self._plot.clear()
        plot = self._plot.addPlot(row=0, col=0)
        make_plot_item_safe(plot)
        valid = [(i, w) for i, w in enumerate(waveforms) if len(w) > 0]
        if not valid:
            return
        spacing_waveforms = [w for _, w in valid]
        if selected_waveforms is not None:
            spacing_waveforms.extend(selected_waveforms)
        all_data = np.concatenate(spacing_waveforms)
        finite_data = all_data[np.isfinite(all_data)]
        spacing = float(np.max(np.abs(finite_data))) * 0.6 if finite_data.size else 1.0
        n = len(valid)
        for rank, (pidx, wav) in enumerate(valid):
            offset = (n - rank - 1) * spacing
            ch = int(ch_indices[pidx]) if pidx < len(ch_indices) else pidx
            plot.plot(wav + offset, pen=pg.mkPen(COLORS["info"], width=3.0))
            if selected_waveforms is not None and pidx < len(selected_waveforms):
                selected = selected_waveforms[pidx]
                if len(selected) > 0 and np.any(np.isfinite(selected)):
                    plot.plot(
                        selected + offset,
                        pen=pg.mkPen(
                            color=(237, 137, 54, 210),
                            width=1.5,
                            style=Qt.PenStyle.SolidLine,
                        ),
                    )
            txt = pg.TextItem(f"Ch {ch}", color=(150, 150, 150), anchor=(1, 0.5))
            txt.setPos(-1, offset)
            txt.setFont(QFont(FONT_FAMILY, 8))
            plot.addItem(txt)
        plot.getAxis("left").setVisible(False)
        if inspection is None:
            title = f"MU {mu_idx} — Stacked"
        else:
            spike_time = inspection.selected_sample / fsamp
            _, similarity, amplitude_ratio, lag_ms = inspection.selected_view(
                remove_other_units
            )
            selected_mode = "earlier units removed" if remove_other_units else "raw EMG"
            title = (
                f"MU {mu_idx} · spike {spike_time:.3f} s · "
                f"r {similarity:.3f} · "
                f"amplitude {amplitude_ratio:.2f}× · "
                f"lag {lag_ms:+.2f} ms · {selected_mode}"
            )
        plot.setTitle(title, color=COLORS["foreground"], size="11pt")
        self.setWindowTitle(f"MUAP Shapes — MU {mu_idx} (Stacked)")

    def clear(self, message="Select a Motor Unit"):
        self._plot.clear()
        p = self._plot.addPlot(row=0, col=0)
        make_plot_item_safe(p)
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
