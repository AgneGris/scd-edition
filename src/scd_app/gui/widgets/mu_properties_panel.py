"""
mu_properties_panel.py — Rich motor unit properties display widget.

Replaces the thin QualityBar with a structured panel that shows:
  • Firing properties  (DR, CoV, n_spikes, min_ISI)
  • Quality metrics    (SIL, PNR)
  • MUAP features      (PTP, peak frequency, waveform length …)
  • Reliability flag
  • Duplicate candidates
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QGridLayout, QLabel, QSizePolicy, QWidget, QHBoxLayout,
    QVBoxLayout, QGroupBox,
)
from PyQt5.QtGui import QFont

# Re-use the app's colour / size tokens
from gui.style.styling import COLORS, FONT_SIZES, FONT_FAMILY

from core.mu_properties import MUProperties


# ── colour helpers ─────────────────────────────────────────────────────────────

_C_OK   = COLORS.get("success", "#a6e3a1")
_C_WARN = COLORS.get("warning", "#f9e2af")
_C_ERR  = COLORS.get("error",   "#f38ba8")
_C_DIM  = COLORS.get("text_dim","#6c7086")
_C_FG   = COLORS.get("foreground", "#cdd6f4")
_C_BG   = COLORS.get("background", "#1e1e2e")
_C_BGS  = COLORS.get("background_light", "#2a2a3c")
_C_BORD = COLORS.get("border", "#45475a")
_C_INFO = COLORS.get("info",   "#89b4fa")


def _color_for_flag(ok: bool, na: bool = False) -> str:
    if na:
        return _C_DIM
    return _C_OK if ok else _C_ERR


def _fmt(value: float, decimals: int = 2, unit: str = "") -> str:
    if math.isnan(value) or math.isinf(value):
        return "—"
    text = f"{value:.{decimals}f}"
    return f"{text} {unit}".strip() if unit else text


# ── small reusable widgets ─────────────────────────────────────────────────────

class _MetricRow(QWidget):
    """One label + value row inside the properties grid."""

    _LABEL_STYLE = (
        f"color: {_C_DIM}; font-size: {FONT_SIZES.get('small','9pt')};"
        f"font-family: {FONT_FAMILY};"
    )
    _VALUE_STYLE = (
        f"color: {_C_FG}; font-size: {FONT_SIZES.get('small','9pt')};"
        f"font-family: {FONT_FAMILY}; font-weight: bold;"
    )

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._lbl = QLabel(label + ":")
        self._lbl.setStyleSheet(self._LABEL_STYLE)
        self._lbl.setFixedWidth(130)

        self._val = QLabel("—")
        self._val.setStyleSheet(self._VALUE_STYLE)
        self._val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        lay.addWidget(self._lbl)
        lay.addWidget(self._val, stretch=1)

    def set_value(self, text: str, color: Optional[str] = None):
        self._val.setText(text)
        c = color or _C_FG
        style = self._VALUE_STYLE + f" color: {c};"
        self._val.setStyleSheet(style)

    def clear(self):
        self._val.setText("—")
        self._val.setStyleSheet(self._VALUE_STYLE)


class _Section(QGroupBox):
    """Titled group box for a cluster of metrics."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"""
            QGroupBox {{
                color: {_C_INFO};
                font-size: {FONT_SIZES.get('small','9pt')};
                font-family: {FONT_FAMILY};
                font-weight: bold;
                border: 1px solid {_C_BORD};
                border-radius: 4px;
                margin-top: 20px;
                padding-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)
        self._inner = QVBoxLayout(self)
        self._inner.setContentsMargins(8, 4, 8, 8)
        self._inner.setSpacing(2)

    def add_row(self, row: _MetricRow):
        self._inner.addWidget(row)


# ══════════════════════════════════════════════════════════════════════════════
# Main panel
# ══════════════════════════════════════════════════════════════════════════════

class MUPropertiesPanel(QFrame):
    """Scrollable panel that shows all MU properties.

    Drop-in replacement for QualityBar.  Call ``set_properties(props)``
    whenever the current motor unit changes or is edited.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {_C_BGS}; border-top: 1px solid {_C_BORD};"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        # ── reliability badge (top row) ──────────────────────────────────
        badge_row = QHBoxLayout()
        self._reliability_badge = QLabel("● RELIABLE")
        self._reliability_badge.setStyleSheet(
            f"color: {_C_OK}; font-weight: bold; "
            f"font-size: {FONT_SIZES.get('small','9pt')}; font-family: {FONT_FAMILY};"
        )
        badge_row.addWidget(self._reliability_badge)
        badge_row.addStretch()
        self._duplicate_label = QLabel()
        self._duplicate_label.setStyleSheet(
            f"color: {_C_WARN}; font-size: {FONT_SIZES.get('small','9pt')};"
        )
        badge_row.addWidget(self._duplicate_label)
        root.addLayout(badge_row)

        # ── metric rows ──────────────────────────────────────────────────
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(8)

        # — Firing —
        sec_firing = _Section("Firing")
        self._r_nspikes  = _MetricRow("N spikes")
        self._r_dr       = _MetricRow("Discharge rate")
        self._r_cov      = _MetricRow("CoV ISI")
        self._r_min_isi  = _MetricRow("Min ISI")
        for r in (self._r_nspikes, self._r_dr, self._r_cov, self._r_min_isi):
            sec_firing.add_row(r)
        sections_layout.addWidget(sec_firing)

        # — Quality —
        sec_quality = _Section("Quality")
        self._r_sil = _MetricRow("SIL")
        self._r_pnr = _MetricRow("PNR")
        for r in (self._r_sil, self._r_pnr):
            sec_quality.add_row(r)
        sections_layout.addWidget(sec_quality)

        # — MUAP —
        sec_muap = _Section("MUAP features")
        self._r_ptp      = _MetricRow("Max PTP")
        self._r_wl       = _MetricRow("Max WL")
        self._r_peak_f   = _MetricRow("Peak freq")
        self._r_med_f    = _MetricRow("Median freq")
        for r in (self._r_ptp, self._r_wl, self._r_peak_f, self._r_med_f):
            sec_muap.add_row(r)
        sections_layout.addWidget(sec_muap)

        root.addLayout(sections_layout)

        self.clear_properties()

    # ── public API ─────────────────────────────────────────────────────────

    def set_properties(self, props: MUProperties):
        flags = props.quality_flags

        # — Firing —
        self._r_nspikes.set_value(
            str(props.n_spikes),
            _color_for_flag(flags["n_spikes"])
        )
        self._r_dr.set_value(
            _fmt(props.discharge_rate_hz, 1, "Hz"),
            _color_for_flag(flags["dr"], na=math.isnan(props.discharge_rate_hz))
        )
        self._r_cov.set_value(
            _fmt(props.cov_pct, 1, "%"),
            _color_for_flag(flags["cov"], na=math.isnan(props.cov_pct))
        )
        self._r_min_isi.set_value(_fmt(props.min_isi_ms, 1, "ms"))

        # — Quality —
        self._r_sil.set_value(
            _fmt(props.sil, 3),
            _color_for_flag(flags["sil"], na=math.isnan(props.sil))
        )
        self._r_pnr.set_value(
            _fmt(props.pnr_db, 1, "dB"),
            _color_for_flag(flags["pnr"], na=math.isnan(props.pnr_db))
        )

        # — MUAP —
        self._r_ptp.set_value(_fmt(props.muap_max_ptp_uv, 2, "µV"))
        self._r_wl.set_value(_fmt(props.muap_max_wl, 1))
        self._r_peak_f.set_value(_fmt(props.muap_peak_freq_hz, 0, "Hz"))
        self._r_med_f.set_value(_fmt(props.muap_median_freq_hz, 0, "Hz"))

        # — Reliability badge —
        if props.is_reliable:
            self._reliability_badge.setText("● RELIABLE")
            self._reliability_badge.setStyleSheet(
                f"color: {_C_OK}; font-weight: bold; "
                f"font-size: {FONT_SIZES.get('small','9pt')};"
            )
        else:
            self._reliability_badge.setText("● UNRELIABLE")
            self._reliability_badge.setStyleSheet(
                f"color: {_C_ERR}; font-weight: bold; "
                f"font-size: {FONT_SIZES.get('small','9pt')};"
            )

        # — Duplicate candidates —
        if props.duplicate_candidates:
            ids = ", ".join(
                f"MU{k} ({v:.0%})"
                for k, v in sorted(props.duplicate_candidates.items(),
                                   key=lambda x: -x[1])
            )
            self._duplicate_label.setText(f"⚠ Possible duplicates: {ids}")
        else:
            self._duplicate_label.setText("")

    def clear_properties(self):
        for row in (
            self._r_nspikes, self._r_dr, self._r_cov, self._r_min_isi,
            self._r_sil, self._r_pnr,
            self._r_ptp, self._r_wl, self._r_peak_f, self._r_med_f,
        ):
            row.clear()
        self._reliability_badge.setText("● —")
        self._reliability_badge.setStyleSheet(
            f"color: {_C_DIM}; font-weight: bold; "
            f"font-size: {FONT_SIZES.get('small','9pt')};"
        )
        self._duplicate_label.setText("")


# ── backwards-compatible thin bar  (for easy migration) ───────────────────────

class QualityBar(MUPropertiesPanel):
    """Kept for import compatibility; routes to MUPropertiesPanel."""

    def set_metrics(self, sil: float, cov: float, fr: float, n_spikes: int):
        """Legacy API — builds a minimal MUProperties and forwards it."""
        from core.mu_properties import MUProperties
        p = MUProperties(
            n_spikes=n_spikes,
            sil=sil,
            cov_pct=cov * 100,       # old code passed 0–1 range
            discharge_rate_hz=fr,
        )
        self.set_properties(p)