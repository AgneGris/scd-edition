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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

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
    win_ms: int = 25,
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

    if not _TOOLBOX_AVAILABLE or props.n_spikes < 2:
        return props

    # Reshape single-unit arrays to (n, 1) as toolbox expects
    st1 = spike_train_col.reshape(-1, 1)
    ipts1 = ipts_col.reshape(-1, 1)

    # ── Firing properties ──────────────────────────────────────────────────
    try:
        dr = tb_props.get_discharge_rate(st1, time_axis)
        props.discharge_rate_hz = _nanval(dr, 0)
    except Exception:
        pass

    try:
        cov = tb_props.get_coefficient_of_variation(st1, time_axis)
        props.cov_pct = _nanval(cov, 0)
    except Exception:
        pass

    try:
        if props.n_spikes >= 2:
            isi_samples = np.diff(np.sort(timestamps))
            if len(isi_samples) > 0:
                props.min_isi_ms = float(np.min(isi_samples)) / fsamp * 1000.0
    except Exception:
        pass

    # ── Quality metrics ───────────────────────────────────────────────────
    try:
        sil = tb_props.get_silhouette_measure(st1, ipts1)
        props.sil = _nanval(sil, 0)
    except Exception:
        pass

    try:
        pnr = tb_props.get_pulse_to_noise_ratio(st1, ipts1)
        props.pnr_db = _nanval(pnr, 0)
    except Exception:
        pass

    # ── MUAP features ─────────────────────────────────────────────────────
    if muap_grid is not None and muap_grid.ndim == 3:
        try:
            # Wrap single-unit MUAP in a (1, rows, cols, win) array
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

        except Exception:
            pass

    props.muap_grid = muap_grid

    # ── Reliability ───────────────────────────────────────────────────────
    try:
        dr_arr = np.array([props.discharge_rate_hz])
        cov_arr = np.array([props.cov_pct])
        sil_arr = np.array([props.sil])
        pnr_arr = np.array([props.pnr_db])
        reliable = tb_props.find_reliable_units(dr_arr, cov_arr, sil_arr, pnr_arr)
        props.is_reliable = bool(reliable[0])
    except Exception:
        props.is_reliable = False

    return props


def compute_port_properties(
    all_timestamps: List[np.ndarray],
    all_sources: List[np.ndarray],
    emg_port: Optional[np.ndarray],          # (n_channels, n_samples)
    grid_positions: Optional[Dict[int, Tuple[int, int]]],
    grid_shape: Optional[Tuple[int, int]],
    fsamp: float,
    win_ms: int = 25,
    roa_threshold: float = 0.3,
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

    # Determine n_samples from the longest source or EMG
    n_samples = 0
    for src in all_sources:
        n_samples = max(n_samples, len(src))
    if emg_port is not None:
        n_samples = max(n_samples, emg_port.shape[1])
    if n_samples == 0:
        return [MUProperties() for _ in range(n_units)]

    fsamp_int = int(round(fsamp))

    # ── Build shared matrices ─────────────────────────────────────────────
    spike_mat = build_spike_train_matrix(all_timestamps, n_samples)   # (n, m)
    ipts_mat  = sources_to_ipts_matrix(all_sources, n_samples)         # (n, m)
    time_axis = timestamps_to_time_axis(n_samples, fsamp)               # (n,)

    # ── Build EMG grid and compute MUAPs ─────────────────────────────────
    muap_grids: List[Optional[np.ndarray]] = [None] * n_units

    if emg_port is None:
        print("  [mu_props] emg_port is None — MUAP computation skipped "
              "(PKL has no 'data' key or all channels were masked)")
    elif not _TOOLBOX_AVAILABLE:
        print("  [mu_props] motor_unit_toolbox not available — MUAP computation skipped")
    else:
        try:
            if grid_positions is not None and grid_shape is not None:
                emg_grid = flat_channels_to_grid(emg_port, grid_positions, grid_shape)
            else:
                # Fallback: treat channels as a (n_ch, 1) pseudo-grid
                n_ch = emg_port.shape[0]
                emg_grid = emg_port.reshape(n_ch, 1, -1)
                grid_shape = (n_ch, 1)
                print(f"  [mu_props] No grid config — using stacked fallback "
                      f"({n_ch} ch × 1 col)")

            # Compute MUAPs via toolbox: returns (n_units, rows, cols, win)
            muaps_all = tb_props.get_muaps(
                spike_mat, emg_grid, fs=fsamp_int, win_ms=win_ms
            )
            # Center the MUAPs — wrap in try/except: center_muaps has a known
            # bug where it uses the full flat index (spatial × temporal) but
            # unravels with spatial size only → ValueError for some grids.
            try:
                muaps_all = tb_props.center_muaps(muaps_all)
            except (ValueError, IndexError):
                pass   # use uncentred MUAPs — still valid for display

            for i in range(n_units):
                muap_grids[i] = muaps_all[i]   # (rows, cols, win)

        except Exception as exc:
            import traceback
            print(f"  [mu_props] MUAP computation failed: {exc}")
            traceback.print_exc()

    # ── Per-unit properties ────────────────────────────────────────────────
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

    # ── Duplicate detection (within-port RoA) ─────────────────────────────
    if _TOOLBOX_AVAILABLE and n_units > 1:
        try:
            roa, pair_idx, _ = tb_spike.rate_of_agreement(
                spike_trains_ref=None,
                spike_trains_test=spike_mat,
                fs=fsamp_int,
            )
            # roa is an (n_units, n_units) matrix when ref=None
            if roa.ndim == 2:
                for i in range(n_units):
                    for j in range(n_units):
                        if i != j and roa[i, j] >= roa_threshold:
                            results[i].duplicate_candidates[j] = float(roa[i, j])
        except Exception:
            pass

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
    win_ms: int = 25,
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

    # Try to recompute MUAP if we have fresh EMG
    muap_grid = mu_props.muap_grid  # reuse existing by default
    if emg_port is not None and _TOOLBOX_AVAILABLE:
        try:
            if grid_positions is not None and grid_shape is not None:
                emg_grid = flat_channels_to_grid(emg_port, grid_positions, grid_shape)
            else:
                n_ch = emg_port.shape[0]
                emg_grid = emg_port.reshape(n_ch, 1, -1)

            st1 = spike_train_col.reshape(-1, 1)
            muaps_new = tb_props.get_muaps(st1, emg_grid, fs=fsamp_int, win_ms=win_ms)
            muaps_new = tb_props.center_muaps(muaps_new)
            muap_grid = muaps_new[0]
        except Exception:
            pass

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