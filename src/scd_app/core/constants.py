"""Shared numeric constants used across core modules."""

# Duplicate detection
ROA_THRESHOLD: float = 0.3  # rate-of-agreement threshold for flagging duplicates

# Spike detection / MUAP computation
MIN_PEAK_SEP: int = 30  # minimum sample separation between detected spikes
MUAP_WIN_MS: int = 25  # total MUAP window duration in milliseconds

# ── Reliability thresholds ────────────────────────────────────────────────────
# These are the single source of truth for both the per-metric colouring in the
# properties panel and the RELIABLE / UNRELIABLE badge: a unit is automatically
# considered reliable when — and only when — every criterion below passes.
SIL_THRESHOLD: float = 0.9
COV_THRESHOLD_PCT: float = 40.0
DR_MIN_HZ: float = 3.0
DR_MAX_HZ: float = 40.0
MIN_N_SPIKES: int = 10

# Human-readable descriptions, used for the badge tooltip.
RELIABILITY_CRITERIA: dict = {
    "sil": f"SIL ≥ {SIL_THRESHOLD:g}",
    "cov": f"CoV ISI ≤ {COV_THRESHOLD_PCT:g} %",
    "dr": f"Discharge rate {DR_MIN_HZ:g}–{DR_MAX_HZ:g} Hz",
    "n_spikes": f"N spikes ≥ {MIN_N_SPIKES}",
}
