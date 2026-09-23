"""Single-discharge MUAP inspection helpers.

The Edition GUI uses this module to compare the raw EMG waveform around one
selected discharge with a leave-one-out template made from the unit's other
discharges.  Keeping the numerical work outside Qt makes the result easy to
test and prevents a diagnostic interaction from mutating the decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scd_app.core.constants import MUAP_WIN_MS
from scd_app.core.mu_properties import flat_channels_to_grid


class SpikeMUAPUnavailable(ValueError):
    """Raised when a meaningful single-discharge comparison cannot be made."""


@dataclass(frozen=True)
class SpikeMUAPInspection:
    """Waveforms and summary metrics for one inspected discharge."""

    selected_sample: int
    reference_grid: np.ndarray
    selected_grid: np.ndarray
    similarity: float
    amplitude_ratio: float
    lag_ms: float
    n_reference_spikes: int
    n_informative_channels: int


def _shift_without_wrap(values: np.ndarray, samples: int) -> np.ndarray:
    """Shift the sample axis, padding with NaN instead of wrapping data."""
    shifted = np.full(values.shape, np.nan, dtype=np.float64)
    n_samples = values.shape[-1]
    if abs(samples) >= n_samples:
        return shifted
    if samples > 0:
        shifted[..., samples:] = values[..., : n_samples - samples]
    elif samples < 0:
        shifted[..., : n_samples + samples] = values[..., -samples:]
    else:
        shifted[...] = values
    return shifted


def _demean_channels(values: np.ndarray) -> np.ndarray:
    """Remove each channel's temporal mean while preserving invalid samples."""
    finite_counts = np.sum(np.isfinite(values), axis=-1, keepdims=True)
    means = np.divide(
        np.nansum(values, axis=-1, keepdims=True),
        finite_counts,
        out=np.zeros((values.shape[0], 1), dtype=np.float64),
        where=finite_counts > 0,
    )
    return values - means


def _normalised_similarity(
    reference: np.ndarray,
    selected: np.ndarray,
    informative: np.ndarray,
) -> tuple[float, float]:
    """Return signed cosine similarity and RMS amplitude ratio."""
    ref = reference[informative]
    event = selected[informative]
    finite = np.isfinite(ref) & np.isfinite(event)
    if not np.any(finite):
        return float("nan"), float("nan")
    ref_values = ref[finite]
    event_values = event[finite]
    ref_norm = float(np.linalg.norm(ref_values))
    event_norm = float(np.linalg.norm(event_values))
    if ref_norm <= np.finfo(float).eps or event_norm <= np.finfo(float).eps:
        return float("nan"), float("nan")
    similarity = float(np.dot(ref_values, event_values) / (ref_norm * event_norm))
    ratio = event_norm / ref_norm
    return float(np.clip(similarity, -1.0, 1.0)), float(ratio)


def _as_grid(
    waveforms: np.ndarray,
    grid_positions: dict[int, tuple[int, int]] | None,
    grid_shape: tuple[int, int] | None,
) -> np.ndarray:
    if grid_positions is not None and grid_shape is not None:
        return flat_channels_to_grid(waveforms, grid_positions, grid_shape)
    return waveforms.reshape(waveforms.shape[0], 1, waveforms.shape[1])


def inspect_spike_muap(
    emg_port: np.ndarray,
    timestamps: np.ndarray,
    selected_sample: int,
    fsamp: float,
    *,
    grid_positions: dict[int, tuple[int, int]] | None = None,
    grid_shape: tuple[int, int] | None = None,
    win_ms: int = MUAP_WIN_MS,
    max_lag_ms: float = 2.0,
    informative_fraction: float = 0.1,
) -> SpikeMUAPInspection:
    """Compare one discharge with a leave-one-out multichannel MUAP template.

    Correlation is calculated after temporal demeaning on channels whose
    reference energy is at least ``informative_fraction`` of the strongest
    channel.  The selected waveform may move by at most ``max_lag_ms`` to
    accommodate small timestamp jitter.  ``lag_ms`` reports the selected
    waveform's latency relative to the reference (positive means later).
    """
    try:
        emg = np.asarray(emg_port, dtype=np.float64)
        raw_timestamps = np.asarray(timestamps).reshape(-1)
        sample = int(selected_sample)
        sampling_rate = float(fsamp)
    except (TypeError, ValueError) as exc:
        raise SpikeMUAPUnavailable("Invalid EMG or timestamp data") from exc

    if emg.ndim != 2 or emg.shape[0] == 0 or emg.shape[1] == 0:
        raise SpikeMUAPUnavailable("Raw multichannel EMG is unavailable")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise SpikeMUAPUnavailable("The sampling rate is invalid")

    try:
        finite_timestamps = raw_timestamps[np.isfinite(raw_timestamps)].astype(
            np.int64, copy=False
        )
    except TypeError as exc:
        raise SpikeMUAPUnavailable("Spike timestamps are invalid") from exc
    timestamps_unique = np.unique(finite_timestamps)
    if sample not in timestamps_unique:
        raise SpikeMUAPUnavailable("The selected marker is no longer a spike")

    half_window = max(1, int(round(float(win_ms) / 2.0 / 1000.0 * sampling_rate)))
    if sample < half_window or sample + half_window > emg.shape[1]:
        raise SpikeMUAPUnavailable(
            "This spike is too close to the recording edge for MUAP inspection"
        )

    valid = timestamps_unique[
        (timestamps_unique >= half_window)
        & (timestamps_unique + half_window <= emg.shape[1])
    ]
    reference_timestamps = valid[valid != sample]
    if reference_timestamps.size == 0:
        raise SpikeMUAPUnavailable(
            "At least one other complete spike is needed for comparison"
        )

    reference_events = np.stack(
        [emg[:, t - half_window : t + half_window] for t in reference_timestamps],
        axis=0,
    )
    finite_counts = np.sum(np.isfinite(reference_events), axis=0)
    reference = np.divide(
        np.nansum(reference_events, axis=0),
        finite_counts,
        out=np.full(reference_events.shape[1:], np.nan),
        where=finite_counts > 0,
    )
    selected = emg[:, sample - half_window : sample + half_window].copy()
    reference = _demean_channels(reference)
    selected = _demean_channels(selected)

    channel_energy = np.nansum(reference**2, axis=1)
    finite_energy = channel_energy[np.isfinite(channel_energy)]
    peak_energy = float(np.max(finite_energy)) if finite_energy.size else 0.0
    if peak_energy <= np.finfo(float).eps:
        raise SpikeMUAPUnavailable("The reference MUAP has no measurable signal")
    fraction = float(np.clip(informative_fraction, 0.0, 1.0))
    informative = np.isfinite(channel_energy) & (
        channel_energy >= peak_energy * fraction
    )
    if not np.any(informative):
        raise SpikeMUAPUnavailable("No informative EMG channels are available")

    max_lag = max(0, int(round(float(max_lag_ms) / 1000.0 * sampling_rate)))
    max_lag = min(max_lag, max(0, reference.shape[1] - 2))
    best_shift = 0
    best_similarity = -np.inf
    best_ratio = float("nan")
    best_selected = selected
    for shift in range(-max_lag, max_lag + 1):
        shifted = _shift_without_wrap(selected, shift)
        similarity, ratio = _normalised_similarity(reference, shifted, informative)
        if np.isfinite(similarity) and similarity > best_similarity:
            best_similarity = similarity
            best_ratio = ratio
            best_shift = shift
            best_selected = shifted

    if not np.isfinite(best_similarity):
        raise SpikeMUAPUnavailable("Waveform similarity could not be calculated")

    dominant_channel = int(np.nanargmax(channel_energy))
    dominant_waveform = reference[dominant_channel]
    if not np.any(np.isfinite(dominant_waveform)):
        raise SpikeMUAPUnavailable("The reference MUAP has no finite waveform")
    peak_sample = int(np.nanargmax(np.abs(dominant_waveform)))
    center_shift = reference.shape[1] // 2 - peak_sample
    reference_display = _shift_without_wrap(reference, center_shift)
    selected_display = _shift_without_wrap(best_selected, center_shift)

    return SpikeMUAPInspection(
        selected_sample=sample,
        reference_grid=_as_grid(
            reference_display, grid_positions=grid_positions, grid_shape=grid_shape
        ),
        selected_grid=_as_grid(
            selected_display, grid_positions=grid_positions, grid_shape=grid_shape
        ),
        similarity=float(best_similarity),
        amplitude_ratio=float(best_ratio),
        lag_ms=float(-best_shift / sampling_rate * 1000.0),
        n_reference_spikes=int(reference_timestamps.size),
        n_informative_channels=int(np.count_nonzero(informative)),
    )
