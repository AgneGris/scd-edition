"""
filter_recalculation.py
───────────────────────
Computes full-length sources on load and recalculates filters after edits.

On load (compute_all_full_sources):
    For each port:
    1. Preprocess FULL raw EMG (using saved w_mat).
    2. Walk the peel-off sequence in order:
       a. Accepted unit → apply saved filter → source_to_timestamps
          → store (full_source, abs_timestamps) → peel with new timestamps.
       b. Rejected repeat → peel with original timestamps (offset to absolute).

On recalculate (recalculate_unit_filter):
    1. Preprocess FULL raw EMG.
    2. Replay peel-off for entries before target unit (same as load).
    3. STA(peeled_emg, edited_timestamps) → new filter.
    4. Apply new filter to peeled EMG → new source.
    5. source_to_timestamps → new timestamps.

Coordinate spaces:
    plateau-local — 0-based within plateau window (stored in .pkl files).
    absolute      — 0-based within full signal.  absolute = local + start_sample.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch

from scd_app.core.constants import MIN_PEAK_SEP
from scd_app.core.utils import to_numpy as _to_numpy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SCD module loader
# ═══════════════════════════════════════════════════════════════════════════════

def _get_scd_modules():
    try:
        from scd.processing.preprocess import (
            whiten, autocorrelation_whiten, extend, time_differentiate,
            notch_filter, low_pass_filter, high_pass_filter,
        )
        from scd.models.timestamping import (
            spike_triggered_average, peel_off_source, source_to_timestamps,
        )
        return {
            "whiten": whiten,
            "autocorrelation_whiten": autocorrelation_whiten,
            "extend": extend,
            "time_differentiate": time_differentiate,
            "notch_filter": notch_filter,
            "low_pass_filter": low_pass_filter,
            "high_pass_filter": high_pass_filter,
            "spike_triggered_average": spike_triggered_average,
            "peel_off_source": peel_off_source,
            "source_to_timestamps": source_to_timestamps,
        }
    except ImportError as e:
        raise ImportError(
            "SCD package not found. Filter recalculation requires scd."
        ) from e


def _replace_bad_channels(
    raw_port: np.ndarray,          # (n_ch, samples)
    decomp_data: dict,
    port_idx: int,
    seed: int = 42,
) -> np.ndarray:
    """Replace rejected channels with baseline noise, matching decomp_worker.

    The decomposition replaced bad channels with torch.randn * noise_std
    BEFORE SCD saw the data, so the whitening matrix was computed on
    noise-replaced data.  We must do the same here.
    """
    mask_list = decomp_data.get("emg_mask", [])
    if port_idx >= len(mask_list) or mask_list[port_idx] is None:
        return raw_port

    mask = np.asarray(mask_list[port_idx]).flatten()
    bad_ch  = np.where(mask == 1)[0]
    if len(bad_ch) == 0:
        return raw_port

    good_ch = np.where(mask == 0)[0]
    noise_std = float(raw_port[good_ch, :].std()) if len(good_ch) > 0 else 1e-6

    # Fixed seed so the noise is identical across load / recalculate calls
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.randn(len(bad_ch), raw_port.shape[1], generator=gen).numpy() * noise_std
    raw_port[bad_ch, :] = noise

    logger.debug("Port %d: replaced %d bad channel(s) with noise (std=%.4f)",
                 port_idx, len(bad_ch), noise_std)
    return raw_port


def _get_plateau_bounds(decomp_data: dict, full_samples: int) -> Tuple[int, int]:
    """Extract plateau start/end from decomp_data."""
    sel_pts = decomp_data.get("plateau_coords",
                              decomp_data.get("selected_points"))
    start, end = 0, full_samples
    if sel_pts is not None:
        try:
            pts = _to_numpy(np.asarray(sel_pts)).flatten()
            start, end = int(pts[0]), int(pts[1])
        except (IndexError, TypeError, ValueError):
            pass
    if end <= start:
        end = full_samples
    return start, end


def _normalise_filters(port_filters_raw, n_units: int) -> List[Optional[np.ndarray]]:
    """Normalise saved filters into a list of numpy arrays."""
    if port_filters_raw is None:
        return [None] * n_units
    if isinstance(port_filters_raw, list):
        result = [_to_numpy(f) if f is not None else None for f in port_filters_raw]
    elif isinstance(port_filters_raw, np.ndarray) and port_filters_raw.ndim >= 2:
        result = [port_filters_raw[i] for i in range(port_filters_raw.shape[0])]
    else:
        result = [_to_numpy(port_filters_raw)]
    while len(result) < n_units:
        result.append(None)
    return result


def _partition_peel_sequence(
    peel_sequence: list, units_per_port: List[int],
) -> List[list]:
    """Split global peel_off_sequence into per-port sub-sequences."""
    boundaries = []
    offset = 0
    for n in units_per_port:
        boundaries.append((offset, offset + n))
        offset += n

    port_seqs: List[list] = [[] for _ in units_per_port]
    current_port = 0
    for entry in peel_sequence:
        uid = entry.get("accepted_unit_idx")
        if uid is not None:
            for p, (lo, hi) in enumerate(boundaries):
                if lo <= uid < hi:
                    current_port = p
                    break
        port_seqs[current_port].append(entry)
    return port_seqs


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_emg(
    raw_emg: torch.Tensor,        # (samples, active_channels)
    config: Dict[str, Any],
    device: torch.device,
    w_mat: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Preprocess raw EMG exactly as SCD does. Returns (samples, ext_ch)."""
    fn  = _get_scd_modules()
    emg = raw_emg.to(device).float()
    fs  = float(config["sampling_frequency"])

    if config.get("notch_params") is not None:
        emg = fn["notch_filter"](emg, fs, config["notch_params"],
                                 config.get("low_pass_cutoff"))
    if config.get("low_pass_cutoff") is not None:
        emg = fn["low_pass_filter"](emg, fs, config["low_pass_cutoff"])
    if config.get("high_pass_cutoff") is not None:
        emg = fn["high_pass_filter"](emg, fs, config["high_pass_cutoff"])
    if config.get("time_differentiate", False):
        emg = fn["time_differentiate"](emg)

    R = int(config.get("extension_factor", 1))
    if R > 1:
        emg = fn["extend"](emg, R)

    if w_mat is not None:
        w_t = torch.from_numpy(w_mat.astype(np.float32)).to(device)
        emg = torch.matmul(emg, w_t.T)
    elif config.get("whitening") == "autocorrelation":
        emg = fn["autocorrelation_whiten"](emg)
    else:
        emg = fn["whiten"](emg)

    return emg


def _apply_filter_torch(
    emg: torch.Tensor,
    filt: np.ndarray,
    device: torch.device,
    norm_slice: Optional[slice] = None,
) -> torch.Tensor:
    """Apply filter and z-score normalize source."""
    filt_t = torch.from_numpy(filt.astype(np.float32)).to(device)
    source = torch.matmul(emg, filt_t).squeeze(-1)
    sl = source[norm_slice] if norm_slice is not None else source
    mu  = sl.mean()
    std = sl.std().clamp(min=1e-8)
    return (source - mu) / std


def _extract_timestamps(
    source_t: torch.Tensor,
    fn: dict,
    min_peak_sep: int = MIN_PEAK_SEP,
    square_source: bool = True,
) -> np.ndarray:
    """Run source_to_timestamps and return absolute sample indices."""
    source_clean = torch.nan_to_num(source_t, nan=0.0, posinf=0.0, neginf=0.0)
    if square_source:
        source_clean = source_clean ** 2
    locs, _heights, _sil = fn["source_to_timestamps"](
        source_clean, min_peak_separation=min_peak_sep,
    )
    return locs.cpu().numpy().astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# Peel-off helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _peel_with_original(
    emg: torch.Tensor,
    entry: dict,
    start_sample: int,
    window_size: int,
    device: torch.device,
    fn: dict,
):
    """Peel one entry using its original plateau-local timestamps."""
    ts_raw = _to_numpy(np.asarray(entry["timestamps"])).flatten()
    ts_abs = (ts_raw + start_sample).astype(np.int64)
    ts_abs = ts_abs[(ts_abs >= 0) & (ts_abs < emg.shape[0])]
    if len(ts_abs) > 0:
        ts_t = torch.from_numpy(ts_abs).to(device)
        fn["peel_off_source"](emg, ts_t, window_size)


def _process_saved_peel_entry(
    emg_running: torch.Tensor,
    entry: dict,
    filt: Optional[np.ndarray],
    port_filters: List[Optional[np.ndarray]],
    local_idx: int,
    start_sample: int,
    end_sample: int,
    window_size: int,
    min_peak_sep: int,
    device: torch.device,
    fn: dict,
    recalculate_filters: bool,
    redetect_timestamps: bool,
    square_source: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Handle one accepted peel entry when use_saved_peel_timestamps=True.

    Returns (source_np, ts_display, new_filt_np).
    """
    plateau_norm = slice(start_sample, end_sample)
    ts_saved = _to_numpy(np.asarray(entry["timestamps"])).flatten().astype(np.int64)
    ts_peel  = ts_saved + start_sample
    ts_peel  = ts_peel[(ts_peel >= 0) & (ts_peel < emg_running.shape[0])]

    new_filt_np = None
    source_t = None

    if recalculate_filters and len(ts_peel) >= 2:
        # Recompute filter via STA so the displayed source is consistent with
        # the spike train.  Updates port_filters in-place for later edits.
        ts_t  = torch.from_numpy(ts_peel).to(device)
        sta   = fn["spike_triggered_average"](emg_running, ts_t, 1)
        norm  = sta.abs().sum().clamp(min=1e-8)
        new_filt = sta.t() / norm
        new_filt_np = new_filt.detach().cpu().numpy()
        if 0 <= local_idx < len(port_filters):
            port_filters[local_idx] = new_filt_np
        source_t  = torch.matmul(emg_running, new_filt).squeeze(-1)
        plateau_sl = source_t[plateau_norm]
        source_t  = (source_t - plateau_sl.mean()) / plateau_sl.std().clamp(min=1e-8)
        source_t  = torch.nan_to_num(source_t, nan=0.0, posinf=0.0, neginf=0.0)
    elif filt is not None:
        source_t = _apply_filter_torch(emg_running, filt, device, norm_slice=plateau_norm)
        source_t = torch.nan_to_num(source_t, nan=0.0, posinf=0.0, neginf=0.0)

    source_np = source_t.cpu().numpy().astype(np.float64) if source_t is not None else None

    if redetect_timestamps and source_t is not None:
        ts_display = _extract_timestamps(source_t, fn, min_peak_sep, square_source=square_source)
    else:
        ts_display = ts_peel

    # Peel using saved timestamps so subsequent units see identical residual EMG
    if len(ts_peel) > 0:
        ts_t = torch.from_numpy(ts_peel).to(device)
        emg_running = fn["peel_off_source"](emg_running, ts_t, window_size)

    return source_np, ts_display, new_filt_np


def _process_recalc_entry(
    emg_running: torch.Tensor,
    filt: np.ndarray,
    window_size: int,
    min_peak_sep: int,
    device: torch.device,
    fn: dict,
    square_source: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle one accepted peel entry in recalculate mode.

    Returns (source_np, ts_abs).
    """
    source_t = _apply_filter_torch(emg_running, filt, device)
    ts_abs   = _extract_timestamps(source_t, fn, min_peak_sep, square_source=square_source)
    if len(ts_abs) > 0:
        ts_t = torch.from_numpy(ts_abs).to(device)
        fn["peel_off_source"](emg_running, ts_t, window_size)
    return source_t.cpu().numpy().astype(np.float64), ts_abs


def _replay_peel_off_for_port(
    emg_full: torch.Tensor,                    # (full_samples, ext_ch) — will be cloned
    port_peel_seq: list,                       # this port's peel_off_sequence entries
    port_filters: List[Optional[np.ndarray]],  # current filters for this port
    global_offset: int,                        # sum of units in earlier ports
    start_sample: int,
    end_sample: int,
    window_size: int,
    min_peak_sep: int,
    device: torch.device,
    stop_before_local_idx: Optional[int] = None,  # None = process all
    square_source: bool = True,
    use_saved_peel_timestamps: bool = False,
    recalculate_filters: bool = False,
    redetect_timestamps: bool = True,
) -> Tuple[torch.Tensor, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Replay peel-off for one port, optionally stopping before a given unit.

    Args:
        use_saved_peel_timestamps: When True (initial load), peel with the
            timestamps that were stored in peel_off_sequence.
        recalculate_filters: When True (and use_saved_peel_timestamps is True),
            recompute each unit's filter via STA on the peeled EMG.

    Returns:
        emg_running:  peeled EMG tensor (modified clone)
        results_dict: {local_mu_idx: (source_np, timestamps_abs_np, new_filt_np|None)}
    """
    fn = _get_scd_modules()
    emg_running = emg_full.clone()
    results_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for entry in port_peel_seq:
        uid = entry.get("accepted_unit_idx")

        if uid is None:
            # Rejected repeat — peel with original timestamps
            _peel_with_original(emg_running, entry, start_sample, window_size, device, fn)
            continue

        local_idx = uid - global_offset

        if stop_before_local_idx is not None and local_idx == stop_before_local_idx:
            break

        filt = port_filters[local_idx] if 0 <= local_idx < len(port_filters) else None

        if use_saved_peel_timestamps:
            source_np, ts_display, new_filt_np = _process_saved_peel_entry(
                emg_running, entry, filt, port_filters, local_idx,
                start_sample, end_sample, window_size, min_peak_sep,
                device, fn, recalculate_filters, redetect_timestamps, square_source,
            )
            results_dict[local_idx] = (source_np, ts_display, new_filt_np)
        elif filt is not None:
            source_np, ts_abs = _process_recalc_entry(
                emg_running, filt, window_size, min_peak_sep, device, fn, square_source,
            )
            results_dict[local_idx] = (source_np, ts_abs, None)
        else:
            results_dict[local_idx] = (None, None, None)
            _peel_with_original(emg_running, entry, start_sample, window_size, device, fn)

    return emg_running, results_dict


def compute_all_full_sources(
    decomp_data: dict,
    device: Optional[torch.device] = None,
    redetect_timestamps: bool = True,
) -> Tuple[Dict[int, List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]],
           int, int, str]:
    """Compute full-length sources and timestamps for all MUs.

    redetect_timestamps: when True (default) re-detect spike times from the
    full source via source_to_timestamps.  When False the stored plateau-local
    timestamps are used as-is (offset to absolute coords) — faster and avoids
    picking up artefact peaks outside the decomposition window.
    """
    if device is None:
        device = torch.device("cpu")

    for key in ("preprocessing_config", "peel_off_sequence", "data", "mu_filters", "ports"):
        if key not in decomp_data:
            return {}, 0, 0, f"Missing key '{key}'."

    config_raw   = decomp_data["preprocessing_config"]
    peel_seq_raw = decomp_data["peel_off_sequence"]

    # Detect format:
    #   New: config_raw is list[dict], peel_seq_raw is list[list]  (one per port)
    #   Old: config_raw is dict,       peel_seq_raw is flat list   (port-0 only)
    _is_per_port_config = isinstance(config_raw, list)
    _is_per_port_peel   = (isinstance(peel_seq_raw, list) and
                           len(peel_seq_raw) > 0 and
                           isinstance(peel_seq_raw[0], list))

    raw_full = _to_numpy(decomp_data["data"])
    if raw_full.ndim == 2 and raw_full.shape[0] > raw_full.shape[1]:
        raw_full = raw_full.T                                   # (ch, samples)

    start_sample, end_sample = _get_plateau_bounds(decomp_data, raw_full.shape[1])

    ports               = decomp_data["ports"]
    chans_per_electrode = decomp_data.get("chans_per_electrode", [])
    channel_indices_all = decomp_data.get("channel_indices")
    mu_filters_all      = decomp_data.get("mu_filters", [])
    w_mat_list          = decomp_data.get("w_mat")
    discharge_times     = decomp_data.get("discharge_times", [])

    units_per_port = []
    for pidx in range(len(ports)):
        if pidx < len(discharge_times):
            dt = discharge_times[pidx]
            units_per_port.append(len(dt) if isinstance(dt, list) else 1)
        else:
            units_per_port.append(0)

    _port_peel_seqs_old: list = []
    if not _is_per_port_peel:
        _port_peel_seqs_old = _partition_peel_sequence(peel_seq_raw, units_per_port)

    port_results: Dict[int, list] = {}
    ch_offset = 0

    for port_idx, port_name in enumerate(ports):
        n_ch    = int(chans_per_electrode[port_idx]) if port_idx < len(chans_per_electrode) else 64
        n_units = units_per_port[port_idx]

        if n_units == 0:
            port_results[port_idx] = []
            ch_offset += n_ch
            continue

        config = config_raw[port_idx] if _is_per_port_config and port_idx < len(config_raw) else (
            config_raw[0] if _is_per_port_config else config_raw
        )
        window_size   = int(config["peel_off_window_size"])
        min_peak_sep  = int(config.get("min_peak_separation", MIN_PEAK_SEP))
        # square_sources_spike_det not saved by SCD; default True for back-compat
        square_source = bool(config.get("square_sources_spike_det", True))

        if _is_per_port_peel:
            port_peel_seq = peel_seq_raw[port_idx] if port_idx < len(peel_seq_raw) else []
            global_offset = 0
        else:
            port_peel_seq = _port_peel_seqs_old[port_idx]
            global_offset = sum(units_per_port[:port_idx])

        if (channel_indices_all is not None
                and port_idx < len(channel_indices_all)
                and channel_indices_all[port_idx] is not None):
            ch_idx = np.asarray(channel_indices_all[port_idx], dtype=int)
            raw_port = raw_full[ch_idx, :].copy()
        else:
            raw_port = raw_full[ch_offset:ch_offset+n_ch, :].copy()
        _replace_bad_channels(raw_port, decomp_data, port_idx)

        w_mat = None
        if w_mat_list is not None:
            entry = w_mat_list[port_idx] if isinstance(w_mat_list, list) else w_mat_list
            if isinstance(entry, np.ndarray) and entry.size > 0:
                w_mat = entry

        pf_raw = mu_filters_all[port_idx] if port_idx < len(mu_filters_all) else None
        port_filters = _normalise_filters(pf_raw, n_units)

        raw_tensor = torch.from_numpy(raw_port.astype(np.float32).T).to(device)
        try:
            emg_proc = preprocess_emg(raw_tensor, config, device, w_mat=w_mat)
        except Exception as e:
            logger.error("Preprocessing failed for '%s': %s", port_name, e)
            port_results[port_idx] = [(None, None, None)] * n_units
            ch_offset += n_ch
            continue

        _, results_dict = _replay_peel_off_for_port(
            emg_proc, port_peel_seq, port_filters,
            global_offset, start_sample, end_sample, window_size, min_peak_sep, device,
            stop_before_local_idx=None, square_source=square_source,
            use_saved_peel_timestamps=True, recalculate_filters=True,
            redetect_timestamps=redetect_timestamps,
        )

        port_results[port_idx] = [
            results_dict.get(i, (None, None, None)) for i in range(n_units)
        ]
        n_ok = sum(1 for r in port_results[port_idx] if r[0] is not None)
        logger.info("Port '%s': %d/%d full sources computed", port_name, n_ok, n_units)

        ch_offset += n_ch

    return port_results, start_sample, end_sample, ""


def recalculate_unit_filter(
    raw_port_channels: np.ndarray,             # (n_ch, full_samples)
    decomp_data: dict,
    port_idx: int,
    local_mu_idx: int,
    edited_timestamps_abs: np.ndarray,         # absolute timestamps from editor
    global_unit_idx: int,
    start_sample: int,
    end_sample: int,
    current_port_filters: List[Optional[np.ndarray]],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recalculate filter + source + timestamps for one MU after edits."""
    if device is None:
        device = torch.device("cpu")

    for key in ("preprocessing_config", "peel_off_sequence"):
        if key not in decomp_data:
            raise KeyError(f"'{key}' not found in decomp_data.")

    config_raw   = decomp_data["preprocessing_config"]
    peel_seq_raw = decomp_data["peel_off_sequence"]

    _is_per_port_config = isinstance(config_raw, list)
    _is_per_port_peel   = (isinstance(peel_seq_raw, list) and
                           len(peel_seq_raw) > 0 and
                           isinstance(peel_seq_raw[0], list))

    config = config_raw[port_idx] if _is_per_port_config else config_raw
    window_size   = int(config["peel_off_window_size"])
    min_peak_sep  = int(config.get("min_peak_separation", MIN_PEAK_SEP))
    square_source = bool(config.get("square_sources_spike_det", True))

    w_mat_list = decomp_data.get("w_mat")
    w_mat = None
    if w_mat_list is not None:
        entry = w_mat_list[port_idx] if isinstance(w_mat_list, list) else w_mat_list
        if isinstance(entry, np.ndarray) and entry.size > 0:
            w_mat = entry

    fn = _get_scd_modules()

    if _is_per_port_peel:
        port_peel_seq = peel_seq_raw[port_idx] if port_idx < len(peel_seq_raw) else []
        global_offset = 0
    else:
        discharge_times = decomp_data.get("discharge_times", [])
        units_per_port = []
        for pidx in range(len(decomp_data["ports"])):
            if pidx < len(discharge_times):
                dt = discharge_times[pidx]
                units_per_port.append(len(dt) if isinstance(dt, list) else 1)
            else:
                units_per_port.append(0)
        port_peel_seqs = _partition_peel_sequence(peel_seq_raw, units_per_port)
        port_peel_seq  = port_peel_seqs[port_idx]
        global_offset  = sum(units_per_port[:port_idx])

    # ── Step 1: Preprocess FULL EMG ───────────────────────────────────────
    raw_port_channels = raw_port_channels.copy()
    _replace_bad_channels(raw_port_channels, decomp_data, port_idx)
    raw_tensor = torch.from_numpy(raw_port_channels.astype(np.float32).T).to(device)
    emg_proc = preprocess_emg(raw_tensor, config, device, w_mat=w_mat)

    # ── Step 2: Replay peel-off up to this unit ───────────────────────────
    emg_peeled, _ = _replay_peel_off_for_port(
        emg_proc, port_peel_seq, current_port_filters,
        global_offset, start_sample, end_sample, window_size, min_peak_sep, device,
        stop_before_local_idx=local_mu_idx, square_source=square_source,
    )

    # ── Step 3: STA filter from edited timestamps (plateau only) ──────────
    ts_in_plateau = edited_timestamps_abs[
        (edited_timestamps_abs >= start_sample) &
        (edited_timestamps_abs < end_sample) &
        (edited_timestamps_abs < emg_peeled.shape[0])
    ]
    if len(ts_in_plateau) < 2:
        raise ValueError("Need at least 2 spikes within the plateau for filter recalculation.")

    ts_tensor = torch.from_numpy(np.asarray(ts_in_plateau, dtype=np.int64)).to(device)
    sta  = fn["spike_triggered_average"](emg_peeled, ts_tensor, 1)
    filt = sta.t()
    filt = filt / filt.abs().sum().clamp(min=1e-8)

    # ── Step 4: Apply new filter → full source ────────────────────────────
    source_t = torch.matmul(emg_peeled, filt).squeeze(-1)
    plateau_slice = source_t[start_sample:end_sample]
    source_t = (source_t - plateau_slice.mean()) / plateau_slice.std().clamp(min=1e-8)
    source_t = torch.nan_to_num(source_t, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Step 5: Re-detect spike times from new source ─────────────────────
    new_timestamps_abs = _extract_timestamps(
        source_t, fn, min_peak_sep=min_peak_sep, square_source=square_source
    )

    return (
        filt.detach().cpu().numpy(),
        source_t.detach().cpu().numpy().astype(np.float64),
        new_timestamps_abs,
    )


def supports_filter_recalculation(decomp_data: dict) -> Tuple[bool, str]:
    missing = [k for k in ("preprocessing_config", "peel_off_sequence", "data")
               if k not in decomp_data]
    if missing:
        return False, f"Missing keys: {missing}."
    peel = decomp_data["peel_off_sequence"]
    if isinstance(peel, list) and len(peel) > 0 and isinstance(peel[0], list):
        if not any(len(p) > 0 for p in peel):
            return False, "peel_off_sequence is empty."
    elif len(peel) == 0:
        return False, "peel_off_sequence is empty."
    return True, ""


def supports_full_source_computation(decomp_data: dict) -> Tuple[bool, str]:
    missing = [k for k in ("preprocessing_config", "peel_off_sequence", "data",
                            "mu_filters") if k not in decomp_data]
    if missing:
        return False, f"Missing keys: {missing}."
    if decomp_data.get("mu_filters") is None:
        return False, "mu_filters is None."
    return True, ""
