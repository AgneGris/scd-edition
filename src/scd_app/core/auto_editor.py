"""
Rule-based automatic editor for motor unit spike trains.

Algorithm reference:
    Wen, Y., Kim, S. J., & Pons, J. L. (2024). A Rule-based Framework for
    Automatic Editing of Motor Unit Spike Trains. TechRxiv.
    https://doi.org/10.36227/techrxiv.170792859.96539206/v1

The paper uses four rules, applied in sequence, and iterated several times for completely automatic editing.
The rules are as follows:
    1. Remove spikes with low innervation pulse train (IPT) heights
    2. Remove spikes with abnormally high firing rate
    3. Add candidate spikes with high IPT heights
    4. Add missing spikes indicated by abnormally low firing rate

Our implementation applies rules 1 and 2 in a first pass, then rules 3 and 4,
then rules 1 and 2 again to clean up any low IPT/high frequency noise additions from the previous pass.

Parameters are taken from the Bayesian-optimised set reported in the paper:
    [rl=0.331, rg=0.339, al=0.539, ag=0.508]

Usage:
    from scd_app.core.auto_editor import auto_edit

    result = auto_edit(mu.timestamps, mu.source, fs=fsamp)
    if not result.skipped:
        mu.timestamps = result.new_timestamps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optimised parameters (Wen et al. 2024, Table / Section II-B5)
# ---------------------------------------------------------------------------
RL: float = 0.331  # local IPT threshold ratio for removal
RG: float = 0.339  # global IPT threshold ratio for removal
AL: float = 0.539  # local IPT threshold ratio for adding
AG: float = 0.508  # global IPT threshold ratio for adding
MAX_FR_HZ: float = 50.0  # global maximum allowable firing rate (Hz)
MIN_SPIKES: int = 4  # minimum spikes required to attempt any rule


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class AutoEditResult:
    new_timestamps: np.ndarray  # sorted int64 spike indices
    n_removed: int  # net spikes removed (rules 1 & 2)
    n_added: int  # net spikes added (rules 3 & 4)
    skipped: bool  # True when MU had < MIN_SPIKES spikes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ipt_heights(ts: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return source amplitude at each spike location.

    Out-of-bounds indices (shouldn't happen in practice) return 0.0 so that
    they are treated as very low IPT and will be removed by Rule 1.
    """
    heights = np.zeros(len(ts), dtype=np.float64)
    valid = (ts >= 0) & (ts < len(source))
    heights[valid] = source[ts[valid]]
    return heights


def _local_mean_heights(heights: np.ndarray, k: int, window: int = 2) -> float:
    """Mean IPT of up to 2*window surrounding spikes (excludes k itself)."""
    n = len(heights)
    neighbours = [
        heights[j]
        for offset in range(-window, window + 1)
        if offset != 0 and 0 <= (j := k + offset) < n
    ]
    return float(np.mean(neighbours)) if neighbours else float(heights[k])


def _local_mean_fr(fr_hz: np.ndarray, k: int, window: int = 5) -> Tuple[float, float]:
    """Mean and std of up to 2*window firing rates surrounding index k.

    `fr_hz[k]` is the firing rate between spike k and spike k+1, so
    fr_hz has length n_spikes - 1.
    """
    n = len(fr_hz)
    lo = max(0, k - window)
    hi = min(n, k + window)
    neighbourhood = fr_hz[lo:hi]
    if len(neighbourhood) < 2:
        return (float(fr_hz[k]) if n > 0 else 0.0), 0.0
    return float(np.mean(neighbourhood)), float(np.std(neighbourhood))


# ---------------------------------------------------------------------------
# Rule 1 — Remove spikes with low IPT heights
# ---------------------------------------------------------------------------


def _apply_rule1(ts: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Remove false-positive spikes whose IPT is low relative to neighbours.

    Subrule 1-1: Remove spike k if its IPT < min(RL * local_mean, RG * global_mean).
    Subrule 1-2: If two consecutive spikes are both below RG * global_mean,
                 remove both (catches mixed spike trains with distinct IPT levels).
    """
    heights = _ipt_heights(ts, source)
    Hm = float(np.mean(heights))
    n = len(ts)
    to_remove: set[int] = set()

    for k in range(n):
        if k in to_remove:
            continue
        hk = heights[k]
        hm_local = _local_mean_heights(heights, k)
        hr = min(RL * hm_local, RG * Hm)

        if hk < hr:
            # Subrule 1-2: also remove the next spike if it is also low
            if k + 1 < n and heights[k + 1] < RG * Hm:
                to_remove.add(k)
                to_remove.add(k + 1)
            else:
                to_remove.add(k)

    if not to_remove:
        return ts
    return np.delete(ts, sorted(to_remove))


# ---------------------------------------------------------------------------
# Rule 2 — Remove spikes with abnormally high firing rate
# ---------------------------------------------------------------------------


def _apply_rule2(ts: np.ndarray, source: np.ndarray, fs: float) -> np.ndarray:
    """Remove false-positive spikes that cause an abnormally high firing rate.

    For each consecutive pair (k, k+1) where the firing rate is too high,
    the spike with the lower IPT is removed.

    Subrule 2-1: FR > MAX_FR_HZ or FR > 2 * local_mean_FR.
    Subrule 2-2: FR > local_mean + 2*local_std, with an additional IPT
                 difference check to avoid removing valid spikes.
    """
    n = len(ts)
    if n < 2:
        return ts

    heights = _ipt_heights(ts, source)
    Hm = float(np.mean(heights))
    Hs = float(np.std(heights))

    isi = np.diff(ts).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(isi > 0, fs / isi, np.inf)

    to_remove: set[int] = set()

    for k in range(n - 1):
        if k in to_remove or (k + 1) in to_remove:
            continue

        fk = fr[k]
        fm, fs_std = _local_mean_fr(fr, k)

        # Index of the lower-IPT spike in this pair
        lower_idx = k if heights[k] <= heights[k + 1] else k + 1

        # Subrule 2-1
        if fk > MAX_FR_HZ or fk > 2.0 * fm:
            to_remove.add(lower_idx)
            continue

        # Subrule 2-2
        if fk > fm + 2.0 * fs_std:
            h_lo = min(heights[k], heights[k + 1])
            h_hi = max(heights[k], heights[k + 1])
            threshold_low = max(Hm - 2.0 * Hs, 0.0)
            if (h_hi - h_lo) > 0.5 * h_lo and h_lo < threshold_low:
                to_remove.add(lower_idx)

    if not to_remove:
        return ts
    return np.delete(ts, sorted(to_remove))


# ---------------------------------------------------------------------------
# Rule 3 — Add candidate spikes with high IPT heights
# ---------------------------------------------------------------------------


def _apply_rule3(ts: np.ndarray, source: np.ndarray, fs: float) -> np.ndarray:
    """Add missed spikes where the IPT has a prominent peak.

    The active region [ts[0], ts[-1]] is scanned in non-overlapping windows
    of size fs/F_hat samples.  Within each window the highest source peak is
    evaluated as a candidate spike.

    Subrule 3-1: Accept if IPT > ha_plus and candidate FR < MAX_FR_HZ.
    Subrule 3-2: Accept if 2 Hz <= candidate FR <= F_hat and IPT > ha_minus.

    The ha_plus / ha_minus thresholds are computed from the *original* spike
    context (before any additions in this pass), consistent with the paper.
    """
    n = len(ts)
    if n < 2:
        return ts

    # Original heights used for threshold context throughout this pass
    orig_heights = _ipt_heights(ts, source)
    Hm = float(np.mean(orig_heights))

    Fm = float(np.mean(fs / np.diff(ts).astype(np.float64)))  # mean FR (Hz)
    F_hat = min(MAX_FR_HZ, 2.0 * Fm)
    win_size = max(1, int(fs / F_hat))

    existing = set(ts.tolist())
    new_spikes: list[int] = []

    active_start = int(ts[0])
    active_end = int(ts[-1])
    current = active_start

    while current < active_end:
        win_end = min(current + win_size, active_end)
        window = source[current:win_end]
        if len(window) == 0:
            current = win_end
            continue

        peak_local = int(np.argmax(window))
        peak_abs = current + peak_local

        if peak_abs in existing:
            current = win_end
            continue

        peak_ipt = float(source[peak_abs])

        # Threshold context: neighbours in the original ts around insertion point
        k_ins = int(np.searchsorted(ts, peak_abs))
        neighbour_idxs = [i for i in range(k_ins - 2, k_ins + 2) if 0 <= i < n]
        hm = float(np.mean(orig_heights[neighbour_idxs])) if neighbour_idxs else Hm
        ha_plus = max(AL * hm, AG * Hm)
        ha_minus = min(AL * hm, AG * Hm)

        # Firing rates this candidate would create with its immediate neighbours
        fr_before = (
            (fs / (peak_abs - ts[k_ins - 1]))
            if k_ins > 0 and ts[k_ins - 1] < peak_abs
            else None
        )
        fr_after = (
            (fs / (ts[k_ins] - peak_abs))
            if k_ins < n and ts[k_ins] > peak_abs
            else None
        )

        candidates = [x for x in (fr_before, fr_after) if x is not None]
        fr_cand = max(candidates) if candidates else MAX_FR_HZ + 1.0

        accepted = False
        # Subrule 3-1
        if peak_ipt > ha_plus and fr_cand < MAX_FR_HZ:
            accepted = True
        # Subrule 3-2
        elif 2.0 <= fr_cand <= F_hat and peak_ipt > ha_minus:
            accepted = True

        if accepted:
            new_spikes.append(peak_abs)
            existing.add(peak_abs)

        current = win_end

    if not new_spikes:
        return ts
    return np.sort(np.concatenate([ts, np.array(new_spikes, dtype=ts.dtype)]))


# ---------------------------------------------------------------------------
# Rule 4 — Add missing spikes for abnormally low firing rate gaps
# ---------------------------------------------------------------------------


def _apply_rule4(ts: np.ndarray, source: np.ndarray, fs: float) -> np.ndarray:
    """Add missing spikes in gaps where the firing rate drops below 70% of local mean.

    Subrule 4-1: Accept if peak IPT > 0.5*ha_plus and both resulting FRs
                 are within the local mean + 2*std bounds.
    Subrule 4-2: Accept if the two resulting FRs are within 30% of each other
                 and IPT > 0.5*ha_minus.
    """
    n = len(ts)
    if n < 2:
        return ts

    # Original heights and FRs used for context throughout this pass
    orig_heights = _ipt_heights(ts, source)
    Hm = float(np.mean(orig_heights))

    isi = np.diff(ts).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(isi > 0, fs / isi, np.inf)

    existing = set(ts.tolist())
    new_spikes: list[int] = []

    for k in range(n - 1):
        fk = fr[k]
        fm, fs_std = _local_mean_fr(fr, k)

        if not np.isfinite(fk) or fm <= 0.0 or fk >= 0.7 * fm:
            continue

        # Find peak in the gap between spike k and spike k+1
        seg_start = int(ts[k]) + 1
        seg_end = int(ts[k + 1])
        if seg_end <= seg_start:
            continue

        segment = source[seg_start:seg_end]
        if len(segment) == 0:
            continue

        peak_local = int(np.argmax(segment))
        peak_abs = seg_start + peak_local

        if peak_abs in existing:
            continue

        peak_ipt = float(source[peak_abs])

        # Threshold context: neighbours around the gap
        neighbour_idxs = [i for i in range(k - 1, k + 3) if 0 <= i < n]
        hm = float(np.mean(orig_heights[neighbour_idxs])) if neighbour_idxs else Hm
        ha_plus = max(AL * hm, AG * Hm)
        ha_minus = min(AL * hm, AG * Hm)

        fr_L = fs / (peak_abs - int(ts[k]))
        fr_R = fs / (int(ts[k + 1]) - peak_abs)

        accepted = False
        # Subrule 4-1
        if (
            peak_ipt > 0.5 * ha_plus
            and fr_L < fm + 2.0 * fs_std
            and fr_R < fm + 2.0 * fs_std
        ):
            accepted = True
        # Subrule 4-2 — ratio test: both FRs within 30% of each other
        elif (
            peak_ipt > 0.5 * ha_minus and min(fr_L, fr_R) / max(fr_L, fr_R, 1e-9) >= 0.7
        ):
            accepted = True

        if accepted:
            new_spikes.append(peak_abs)
            existing.add(peak_abs)

    if not new_spikes:
        return ts
    return np.sort(np.concatenate([ts, np.array(new_spikes, dtype=ts.dtype)]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def auto_edit(
    timestamps: np.ndarray,
    source: np.ndarray,
    fs: float,
) -> AutoEditResult:
    """Apply the four-rule auto-editor to a single motor unit spike train.

    Parameters
    ----------
    timestamps : np.ndarray
        Spike sample indices (int64).  Must be valid indices into `source`.
    source : np.ndarray
        IPT / source signal (float).  May be full-recording length or
        plateau-local — the algorithm is agnostic to coordinate space.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    AutoEditResult
        Contains the refined timestamps, net removal/addition counts, and a
        `skipped` flag that is True when the MU had fewer than MIN_SPIKES spikes.
    """
    ts = np.sort(np.asarray(timestamps, dtype=np.int64))
    source = np.asarray(source, dtype=np.float64)

    if len(ts) < MIN_SPIKES:
        return AutoEditResult(
            new_timestamps=ts,
            n_removed=0,
            n_added=0,
            skipped=True,
        )

    n_before = len(ts)

    ts = _apply_rule1(ts, source)
    ts = _apply_rule2(ts, source, fs)

    n_after_removal = len(ts)
    n_removed = n_before - n_after_removal

    # Rules 3 & 4 need at least 2 spikes to compute FRs
    if len(ts) >= 2:
        ts = _apply_rule3(ts, source, fs)
        ts = _apply_rule4(ts, source, fs)

    # Second removal pass: re-evaluate rules 1 & 2 against the updated spike train.
    # At low SNR, rules 3 & 4 can add noise peaks that look acceptable against the
    # original context but are clearly wrong once evaluated against their actual
    # neighbours (very low IPT or implausibly high IFR).  n_added below reflects
    # net additions that survive this cleanup.
    if len(ts) >= MIN_SPIKES:
        ts = _apply_rule1(ts, source)
        ts = _apply_rule2(ts, source, fs)

    n_added = len(ts) - n_after_removal

    return AutoEditResult(
        new_timestamps=ts,
        n_removed=n_removed,
        n_added=n_added,
        skipped=False,
    )
