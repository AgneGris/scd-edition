"""Shared numeric constants used across core modules."""

# Duplicate detection
ROA_THRESHOLD: float = 0.3  # rate-of-agreement threshold for flagging duplicates

# Spike detection / MUAP computation
MIN_PEAK_SEP: int = 30      # minimum sample separation between detected spikes
MUAP_WIN_MS: int = 25       # MUAP window half-width in milliseconds

# Reliability thresholds (used by motor_unit_toolbox.props.find_reliable_units)
SIL_THRESHOLD: float = 0.9
PNR_THRESHOLD_DB: float = 30.0
COV_THRESHOLD_PCT: float = 40.0
