"""
mu_properties.py  —  Motor Unit property computation via motor_unit_toolbox.

Responsibility:
    Convert between the app's internal data formats and the formats expected
    by motor_unit_toolbox, then compute and cache all MU properties.

Data-format contracts
─────────────────────
  motor_unit_toolbox expects:
    spike_train  : np.ndarray  (n_samples, n_units)  bool / int
    ipts         : np.ndarray  (n_samples, n_units)  float  (pulse trains)
    emg_ch_array : np.ndarray  (rows, cols, n_samples)  — 3-D grid

  App stores:
    timestamps  : 1-D int64 array of sample indices
    source      : 1-D float array (IPT / pulse train)
    emg_port    : np.ndarray  (n_active_channels, n_samples)  flat

Public API
──────────
    MUProperties          — dataclass that holds every metric for one unit
    compute_port_props()  — compute everything for all units in one port
    recompute_unit_props()— recompute one unit after an edit
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scd_app.core.constants import MIN_PEAK_SEP, MUAP_WIN_MS, ROA_THRESHOLD
from scd_app.core.utils import to_numpy  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)

# ── toolbox imports ────────────────────────────────────────────────────────────
try:
    from motor_unit_toolbox import props as tb_props
    from motor_unit_toolbox import spike_comp as tb_spike
    _TOOLBOX_AVAILABLE = True
except ImportError:
    _TOOLBOX_AVAILABLE = False
    import warnings
    warnings.warn(
        "motor_unit_toolbox not installed – quality metrics will be unavailable. "
        "Install with: pip install git+https://github.com/imendezguerra/motor_unit_toolbox.git",
        ImportWarning,
        stacklevel=2,
    )


def _center_muaps(muaps: np.ndarray) -> np.ndarray:
    """Center MUAPs so the peak-amplitude sample sits at the window midpoint.

    Fixes a bug in tb_props.center_muaps where per-channel peak *sample*
    indices are passed to np.unravel_index with the *spatial* grid shape,
    causing an IndexError whenever a peak sample index >= rows*cols.
    """
    if muaps.ndim < 4 or muaps.size == 0:
        return muaps
    result = muaps.copy()
    units, rows, cols, samples = muaps.shape
    center = samples // 2
    for u in range(units):
        muap = muaps[u]  # (rows, cols, samples)
        flat_idx = int(np.nanargmax(np.nanmax(np.abs(muap), axis=-1)))
        ch_row, ch_col = np.unravel_index(flat_idx, (rows, cols))
        peak_sample = int(np.nanargmax(np.abs(muap[ch_row, ch_col])))
        result[u] = np.roll(muap, center - peak_sample, axis=-1)
    return result


@dataclass
class MUProperties:
    """All quality and morphological properties for a single motor unit.

    Naming follows motor_unit_toolbox conventions where possible.
    Values are NaN when computation failed or data was insufficient.
    """
    # ── firing properties ──────────────────────────────────────────────────
    n_spikes: int = 0
    discharge_rate_hz: float = float("nan")    # mean DR (active period)
    cov_pct: float = float("nan")              # CoV of ISI × 100  (%)
    min_isi_ms: float = float("nan")           # minimum ISI in ms

    # ── quality metrics ───────────────────────────────────────────────────
    sil: float = float("nan")                  # silhouette measure  [0-1]
    pnr_db: float = float("nan")               # pulse-to-noise ratio  (dB)
    spike_centroid: float = float("nan")       # mean source² at spike peaks
    noise_centroid: float = float("nan")       # mean source² at non-spike peaks

    # ── MUAP features (scalar summaries over the selected channels) ────────
    muap_max_ptp_uv: float = float("nan")      # max PTP across channels
    muap_max_energy: float = float("nan")      # max energy across channels
    muap_max_wl: float = float("nan")          # max waveform length
    muap_peak_freq_hz: float = float("nan")    # peak spectral freq (IQR chs)
    muap_median_freq_hz: float = float("nan")  # median spectral freq
    muap_mean_freq_hz: float = float("nan")    # mean spectral freq

    # ── reliability flag  (from toolbox thresholds) ────────────────────────
    is_reliable: bool = False

    # ── full MUAP array in grid layout  (rows × cols × win_samples) ───────
    #    stored here so the GUI can render it without re-computation
    muap_grid: Optional[np.ndarray] = field(default=None, repr=False)

    # ── RoA duplicate candidates  {other_mu_idx: roa_score} ────────────────
    duplicate_candidates: Dict[int, float] = field(default_factory=dict)

    @property
    def quality_flags(self) -> Dict[str, bool]:
        """Return a dict of pass/fail flags for each quality criterion."""
        return {
            "sil": self.sil >= 0.9 if not np.isnan(self.sil) else False,
            "pnr": self.pnr_db >= 30 if not np.isnan(self.pnr_db) else False,
            "cov": self.cov_pct <= 40 if not np.isnan(self.cov_pct) else False,
            "dr": (3 <= self.discharge_rate_hz <= 40)
                  if not np.isnan(self.discharge_rate_hz) else False,
            "n_spikes": self.n_spikes >= 10,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Format-conversion helpers
# ══════════════════════════════════════════════════════════════════════════════

def timestamps_to_spike_train(
    timestamps: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Convert a 1-D array of spike sample indices → binary spike train.

    Returns:
        np.ndarray: shape (n_samples,), dtype bool
    """
    st = np.zeros(n_samples, dtype=bool)
    valid = timestamps[(timestamps >= 0) & (timestamps < n_samples)]
    st[valid.astype(np.int64)] = True
    return st


def build_spike_train_matrix(
    all_timestamps: List[np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Stack several timestamp arrays into a (n_samples, n_units) bool matrix."""
    n_units = len(all_timestamps)
    mat = np.zeros((n_samples, n_units), dtype=bool)
    for i, ts in enumerate(all_timestamps):
        valid = ts[(ts >= 0) & (ts < n_samples)]
        mat[valid.astype(np.int64), i] = True
    return mat


def sources_to_ipts_matrix(
    sources: List[np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Stack per-unit source signals into a (n_samples, n_units) float matrix.

    Source signals may be shorter than n_samples (e.g. plateau only); in that
    case the remaining samples are zero-padded.
    """
    n_units = len(sources)
    mat = np.zeros((n_samples, n_units), dtype=np.float64)
    for i, src in enumerate(sources):
        src = np.asarray(src, dtype=np.float64).flatten()
        length = min(len(src), n_samples)
        mat[:length, i] = src[:length]
    return mat


def flat_channels_to_grid(
    emg_flat: np.ndarray,
    grid_positions: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Re-arrange a (n_channels, n_samples) flat EMG into (rows, cols, n_samples).

    Channels that have no entry in grid_positions are left as zeros.

    Args:
        emg_flat: (n_channels, n_samples)
        grid_positions: dict mapping local channel index (0-based) → (row, col)
        grid_shape: (n_rows, n_cols)

    Returns:
        np.ndarray: (rows, cols, n_samples)
    """
    rows, cols = grid_shape
    n_ch, n_samples = emg_flat.shape
    grid = np.zeros((rows, cols, n_samples), dtype=emg_flat.dtype)

    for local_ch in range(n_ch):
        pos = grid_positions.get(local_ch)
        if pos is None:
            continue
        r, c = pos
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = emg_flat[local_ch]

    return grid


def timestamps_to_time_axis(
    n_samples: int,
    fsamp: float,
) -> np.ndarray:
    """Return a time-axis array in seconds."""
    return np.arange(n_samples) / fsamp


# ══════════════════════════════════════════════════════════════════════════════
# Core computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_centroids(
    source: np.ndarray,
    timestamps: np.ndarray,
    min_peak_sep: int = MIN_PEAK_SEP,
) -> tuple:
    """Return (spike_centroid, noise_centroid) from source² peak amplitudes.

    Spike centroid  = mean of source² at spike positions.
    Noise centroid  = mean of source² at all detected peaks that are NOT spikes.
    Uses scipy.signal.find_peaks; returns (nan, nan) if unavailable.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return float("nan"), float("nan")

    src_sq = np.asarray(source, dtype=np.float64) ** 2
    if len(src_sq) == 0 or len(timestamps) == 0:
        return float("nan"), float("nan")

    ts = timestamps[(timestamps >= 0) & (timestamps < len(src_sq))]
    spike_centroid = float(np.mean(src_sq[ts])) if len(ts) > 0 else float("nan")

    all_peaks, _ = find_peaks(src_sq, distance=min_peak_sep)
    spike_set = set(ts.tolist())
    noise_peaks = all_peaks[np.array([p not in spike_set for p in all_peaks])]
    noise_centroid = float(np.mean(src_sq[noise_peaks])) if len(noise_peaks) > 0 else float("nan")

    return spike_centroid, noise_centroid


def _nanval(arr: np.ndarray, idx: int) -> float:
    """Safely extract a scalar from a toolbox-returned array."""
    try:
        v = float(arr[idx])
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def compute_unit_properties(
    timestamps: np.ndarray,
    source: np.ndarray,
    spike_train_col: np.ndarray,      # (n_samples,) bool — column for this unit
    ipts_col: np.ndarray,             # (n_samples,) float — column for this unit
    time_axis: np.ndarray,            # (n_samples,) float  seconds
    fsamp: float,
    muap_grid: Optional[np.ndarray],  # (rows, cols, win_samples) or None
    fsamp_int: int,
    win_ms: int = MUAP_WIN_MS,
) -> MUProperties:
    """Compute all properties for a single motor unit.

    All toolbox calls operate on single-column 2-D arrays so we can reuse
    batch functions without modification.

    Args:
        spike_train_col: Binary spike train for this unit  (n_samples,)
        ipts_col:        Source/pulse train for this unit  (n_samples,)
        time_axis:       Time in seconds  (n_samples,)
        muap_grid:       Pre-computed MUAP, or None  (rows, cols, win_samples)
    """
    props = MUProperties()
    props.n_spikes = int(np.sum(spike_train_col))

    if props.n_spikes == 0:
        props.muap_grid = None
        return props

    if not _TOOLBOX_AVAILABLE or props.n_spikes < 2:
        props.muap_grid = muap_grid
        return props

    # Reshape single-unit arrays to (n, 1) as toolbox expects
    st1 = spike_train_col.reshape(-1, 1)
    ipts1 = ipts_col.reshape(-1, 1)

    # ── Firing properties ──────────────────────────────────────────────────
    try:
        dr = tb_props.get_discharge_rate(st1, time_axis)
        props.discharge_rate_hz = _nanval(dr, 0)
    except Exception as e:
        logger.debug("discharge_rate failed: %s", e)

    try:
        cov = tb_props.get_coefficient_of_variation(st1, time_axis)
        props.cov_pct = _nanval(cov, 0)
    except Exception as e:
        logger.debug("coefficient_of_variation failed: %s", e)

    try:
        if props.n_spikes >= 2:
            isi_samples = np.diff(np.sort(timestamps))
            if len(isi_samples) > 0:
                props.min_isi_ms = float(np.min(isi_samples)) / fsamp * 1000.0
    except Exception as e:
        logger.debug("min_isi failed: %s", e)

    # ── Quality metrics ───────────────────────────────────────────────────
    try:
        sil = tb_props.get_silhouette_measure(st1, ipts1)
        props.sil = _nanval(sil, 0)
    except Exception as e:
        logger.debug("silhouette_measure failed: %s", e)

    try:
        pnr = tb_props.get_pulse_to_noise_ratio(st1, ipts1)
        props.pnr_db = _nanval(pnr, 0)
    except Exception as e:
        logger.debug("pulse_to_noise_ratio failed: %s", e)

    props.spike_centroid, props.noise_centroid = _compute_centroids(source, timestamps)

    # ── MUAP features ─────────────────────────────────────────────────────
    if muap_grid is not None and muap_grid.ndim == 3:
        try:
            muap4 = muap_grid[np.newaxis]   # (1, rows, cols, win_samples)

            ptp = tb_props.get_muap_ptp(muap4, sel_chs_by=None)
            props.muap_max_ptp_uv = float(np.nanmax(ptp))

            energy = tb_props.get_muap_energy(muap4, sel_chs_by=None)
            props.muap_max_energy = float(np.nanmax(energy))

            wl = tb_props.get_muap_waveform_length(muap4, sel_chs_by=None)
            props.muap_max_wl = float(np.nanmax(wl))

            peak_f = tb_props.get_muap_peak_frequency(muap4, sel_chs_by="iqr", fs=fsamp_int)
            props.muap_peak_freq_hz = float(np.nanmean(peak_f[np.isfinite(peak_f)]))

            med_f = tb_props.get_muap_median_frequency(muap4, sel_chs_by="iqr", fs=fsamp_int)
            props.muap_median_freq_hz = float(np.nanmean(med_f[np.isfinite(med_f)]))

            mean_f = tb_props.get_muap_mean_frequency(muap4, sel_chs_by="iqr", fs=fsamp_int)
            props.muap_mean_freq_hz = float(np.nanmean(mean_f[np.isfinite(mean_f)]))

        except Exception as e:
            logger.debug("MUAP feature computation failed: %s", e)

    props.muap_grid = muap_grid

    # ── Reliability ───────────────────────────────────────────────────────
    try:
        dr_arr = np.array([props.discharge_rate_hz])
        cov_arr = np.array([props.cov_pct])
        sil_arr = np.array([props.sil])
        pnr_arr = np.array([props.pnr_db])
        reliable = tb_props.find_reliable_units(dr_arr, cov_arr, sil_arr, pnr_arr)
        props.is_reliable = bool(reliable[0])
    except Exception as e:
        logger.debug("find_reliable_units failed: %s", e)
        props.is_reliable = False

    return props


def compute_port_properties(
    all_timestamps: List[np.ndarray],
    all_sources: List[np.ndarray],
    emg_port: Optional[np.ndarray],          # (n_channels, n_samples)
    grid_positions: Optional[Dict[int, Tuple[int, int]]],
    grid_shape: Optional[Tuple[int, int]],
    fsamp: float,
    win_ms: int = MUAP_WIN_MS,
    roa_threshold: float = ROA_THRESHOLD,
) -> List[MUProperties]:
    """Compute all properties for every motor unit in a port.

    Args:
        all_timestamps:  List of timestamp arrays, one per MU
        all_sources:     List of source/IPT arrays, one per MU
        emg_port:        (n_channels, n_samples) flat EMG, or None
        grid_positions:  channel-index → (row, col) mapping, or None
        grid_shape:      (rows, cols), or None for stacked layout
        fsamp:           Sampling frequency in Hz
        win_ms:          MUAP window half-width in ms
        roa_threshold:   Rate-of-agreement threshold for flagging duplicates

    Returns:
        List[MUProperties], one per MU in the same order as inputs.
    """
    n_units = len(all_timestamps)
    if n_units == 0:
        return []

    n_samples = 0
    for src in all_sources:
        n_samples = max(n_samples, len(src))
    if emg_port is not None:
        n_samples = max(n_samples, emg_port.shape[1])
    if n_samples == 0:
        return [MUProperties() for _ in range(n_units)]

    fsamp_int = int(round(fsamp))

    spike_mat = build_spike_train_matrix(all_timestamps, n_samples)
    ipts_mat  = sources_to_ipts_matrix(all_sources, n_samples)
    time_axis = timestamps_to_time_axis(n_samples, fsamp)

    muap_grids: List[Optional[np.ndarray]] = [None] * n_units

    if emg_port is None:
        logger.info("MUAP computation skipped — no EMG data for this port")
    elif not _TOOLBOX_AVAILABLE:
        logger.warning("motor_unit_toolbox not available — MUAP computation skipped")
    else:
        try:
            if grid_positions is not None and grid_shape is not None:
                emg_grid = flat_channels_to_grid(emg_port, grid_positions, grid_shape)
            else:
                n_ch = emg_port.shape[0]
                emg_grid = emg_port.reshape(n_ch, 1, -1)
                grid_shape = (n_ch, 1)
                logger.info("No grid config — using stacked fallback (%d ch × 1 col)", n_ch)

            muaps_all = tb_props.get_muaps(
                spike_mat, emg_grid, fs=fsamp_int, win_ms=win_ms
            )
            muaps_all = _center_muaps(muaps_all)

            for i in range(n_units):
                muap_grids[i] = muaps_all[i]

        except Exception as exc:
            logger.error("MUAP computation failed: %s\n%s", exc, traceback.format_exc())

    results: List[MUProperties] = []
    for i in range(n_units):
        p = compute_unit_properties(
            timestamps=all_timestamps[i],
            source=all_sources[i],
            spike_train_col=spike_mat[:, i],
            ipts_col=ipts_mat[:, i],
            time_axis=time_axis,
            fsamp=fsamp,
            muap_grid=muap_grids[i],
            fsamp_int=fsamp_int,
            win_ms=win_ms,
        )
        results.append(p)

    if _TOOLBOX_AVAILABLE and n_units > 1:
        try:
            roa, _ = tb_spike.rate_of_agreement_full(
                spike_trains_ref=spike_mat,
                spike_trains_test=spike_mat,
                fs=fsamp_int,
            )
            for i in range(n_units):
                for j in range(n_units):
                    if i != j and roa[i, j] >= roa_threshold:
                        results[i].duplicate_candidates[j] = float(roa[i, j])
        except Exception as e:
            logger.debug("Within-port RoA computation failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Single-unit recomputation  (after an edit)
# ══════════════════════════════════════════════════════════════════════════════

def recompute_unit_properties(
    mu_props: MUProperties,
    new_timestamps: np.ndarray,
    source: np.ndarray,
    emg_port: Optional[np.ndarray],
    grid_positions: Optional[Dict[int, Tuple[int, int]]],
    grid_shape: Optional[Tuple[int, int]],
    fsamp: float,
    win_ms: int = MUAP_WIN_MS,
) -> MUProperties:
    """Lightweight re-computation after a spike edit.

    Re-uses the stored muap_grid when it exists (MUAP doesn't change with
    small spike edits); recomputes it from scratch when emg_port is provided.
    Falls back to the existing muap_grid if EMG is not available.
    """
    n_samples = len(source)
    if n_samples == 0:
        return MUProperties(n_spikes=len(new_timestamps))

    fsamp_int = int(round(fsamp))
    time_axis = timestamps_to_time_axis(n_samples, fsamp)
    spike_train_col = timestamps_to_spike_train(new_timestamps, n_samples)
    ipts_col = np.asarray(source, dtype=np.float64).flatten()[:n_samples]

    muap_grid = None if len(new_timestamps) == 0 else mu_props.muap_grid
    if len(new_timestamps) > 0 and emg_port is not None and _TOOLBOX_AVAILABLE:
        try:
            if grid_positions is not None and grid_shape is not None:
                emg_grid = flat_channels_to_grid(emg_port, grid_positions, grid_shape)
            else:
                n_ch = emg_port.shape[0]
                emg_grid = emg_port.reshape(n_ch, 1, -1)

            st1 = spike_train_col.reshape(-1, 1)
            muaps_new = tb_props.get_muaps(st1, emg_grid, fs=fsamp_int, win_ms=win_ms)
            muaps_new = _center_muaps(muaps_new)
            muap_grid = muaps_new[0]
        except Exception as e:
            logger.debug("MUAP recompute failed: %s", e)

    return compute_unit_properties(
        timestamps=new_timestamps,
        source=source,
        spike_train_col=spike_train_col,
        ipts_col=ipts_col,
        time_axis=time_axis,
        fsamp=fsamp,
        muap_grid=muap_grid,
        fsamp_int=fsamp_int,
        win_ms=win_ms,
    )
