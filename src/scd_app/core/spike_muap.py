"""Single-discharge MUAP inspection helpers.

The Edition GUI uses this module to compare one discharge with a raw,
leave-one-out template made from the unit's other discharges.  The selected
waveform is available both as recorded and after timestamp-aligned MUAP
estimates for the other units have been removed.  Keeping the numerical work
outside Qt makes the result easy to test and prevents a diagnostic interaction
from mutating the decomposition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scd_app.core.constants import MUAP_WIN_MS
from scd_app.core.mu_properties import flat_channels_to_grid


class SpikeMUAPUnavailable(ValueError):
    """Raised when a meaningful single-discharge comparison cannot be made."""


@dataclass(frozen=True)
class SpikeMUAPInspection:
    """Reference plus raw and interference-reduced views of one discharge.

    ``selected_grid`` and its metrics describe the interference-reduced view.
    The ``raw_*`` fields describe the same discharge before subtraction.  The
    blue ``reference_grid`` is shared by both views.
    """

    selected_sample: int
    reference_grid: np.ndarray
    selected_grid: np.ndarray
    similarity: float
    amplitude_ratio: float
    lag_ms: float
    n_reference_spikes: int
    n_informative_channels: int
    n_subtracted_units: int = 0
    n_subtracted_events: int = 0
    raw_selected_grid: np.ndarray | None = None
    raw_similarity: float = float("nan")
    raw_amplitude_ratio: float = float("nan")
    raw_lag_ms: float = float("nan")

    def selected_view(
        self, remove_other_units: bool
    ) -> tuple[np.ndarray, float, float, float]:
        """Return the selected waveform and metrics for the requested view."""
        if not remove_other_units and self.raw_selected_grid is not None:
            return (
                self.raw_selected_grid,
                self.raw_similarity,
                self.raw_amplitude_ratio,
                self.raw_lag_ms,
            )
        return self.selected_grid, self.similarity, self.amplitude_ratio, self.lag_ms


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


def _align_to_reference(
    reference: np.ndarray,
    selected: np.ndarray,
    informative: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, float, float, int]:
    """Align one selected waveform to a fixed reference and return its metrics."""
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
    return best_selected, float(best_similarity), float(best_ratio), best_shift


def _as_grid(
    waveforms: np.ndarray,
    grid_positions: dict[int, tuple[int, int]] | None,
    grid_shape: tuple[int, int] | None,
) -> np.ndarray:
    if grid_positions is not None and grid_shape is not None:
        return flat_channels_to_grid(waveforms, grid_positions, grid_shape)
    return waveforms.reshape(waveforms.shape[0], 1, waveforms.shape[1])


def _finite_integer_timestamps(values: np.ndarray | Sequence[int]) -> np.ndarray:
    """Return sorted, unique, finite timestamps as int64."""
    raw = np.asarray(values).reshape(-1)
    try:
        finite = raw[np.isfinite(raw)].astype(np.int64, copy=False)
    except TypeError as exc:
        raise SpikeMUAPUnavailable("Spike timestamps are invalid") from exc
    return np.unique(finite)


def _mean_event_template(
    emg: np.ndarray,
    timestamps: np.ndarray,
    half_window: int,
) -> np.ndarray | None:
    """Estimate a zero-baseline, timestamp-aligned MUAP contribution."""
    complete = timestamps[
        (timestamps >= half_window) & (timestamps + half_window <= emg.shape[1])
    ]
    if complete.size == 0:
        return None

    events = np.stack(
        [emg[:, t - half_window : t + half_window] for t in complete], axis=0
    )
    finite_counts = np.sum(np.isfinite(events), axis=0)
    template = np.divide(
        np.nansum(events, axis=0),
        finite_counts,
        out=np.full(events.shape[1:], np.nan),
        where=finite_counts > 0,
    )
    if not np.any(np.isfinite(template)):
        return None
    # EMG is normally high-pass filtered, but removing the finite-window mean
    # prevents repeated template subtraction from shifting the local baseline.
    return _demean_channels(template)


def _other_unit_contributions(
    emg: np.ndarray,
    target_timestamps: np.ndarray,
    other_unit_timestamps: Sequence[np.ndarray] | None,
    half_window: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build subtraction templates for the other units in the current port.

    Where possible, a unit's template is estimated from discharges that do not
    overlap a target-unit discharge.  This avoids baking the target MUAP into
    the contribution that is subsequently subtracted.  If a unit has no such
    isolated discharge, all of its complete discharges are used as a fallback.
    """
    contributions: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_other in other_unit_timestamps or ():
        other = _finite_integer_timestamps(raw_other)
        if other.size == 0:
            continue

        complete = other[(other >= half_window) & (other + half_window <= emg.shape[1])]
        if complete.size == 0:
            continue

        # Two windows of half-width ``half_window`` overlap when their centres
        # are less than one full window apart.
        distance = np.abs(complete[:, None] - target_timestamps[None, :])
        isolated = complete[np.all(distance >= 2 * half_window, axis=1)]
        template_timestamps = isolated if isolated.size else complete
        template = _mean_event_template(emg, template_timestamps, half_window)
        if template is not None:
            contributions.append((other, template))
    return contributions


def _interference_reduced_events(
    emg: np.ndarray,
    event_timestamps: np.ndarray,
    contributions: Sequence[tuple[np.ndarray, np.ndarray]],
    half_window: int,
) -> tuple[np.ndarray, int]:
    """Extract event windows after removing overlapping other-unit MUAPs."""
    events = np.stack(
        [emg[:, t - half_window : t + half_window] for t in event_timestamps],
        axis=0,
    ).astype(np.float64, copy=True)
    removed_events = 0

    for event_index, target in enumerate(event_timestamps):
        target_start = int(target) - half_window
        target_end = int(target) + half_window
        for other_timestamps, template in contributions:
            overlapping = other_timestamps[
                (other_timestamps - half_window < target_end)
                & (other_timestamps + half_window > target_start)
            ]
            for other in overlapping:
                contribution_start = int(other) - half_window
                contribution_end = int(other) + half_window
                overlap_start = max(target_start, contribution_start, 0)
                overlap_end = min(target_end, contribution_end, emg.shape[1])
                if overlap_start >= overlap_end:
                    continue

                target_slice = slice(
                    overlap_start - target_start, overlap_end - target_start
                )
                template_slice = slice(
                    overlap_start - contribution_start,
                    overlap_end - contribution_start,
                )
                contribution = template[:, template_slice]
                events[event_index, :, target_slice] -= np.nan_to_num(
                    contribution, nan=0.0, posinf=0.0, neginf=0.0
                )
                removed_events += 1

    return events, removed_events


def inspect_spike_muap(
    emg_port: np.ndarray,
    timestamps: np.ndarray,
    selected_sample: int,
    fsamp: float,
    *,
    grid_positions: dict[int, tuple[int, int]] | None = None,
    grid_shape: tuple[int, int] | None = None,
    other_unit_timestamps: Sequence[np.ndarray] | None = None,
    win_ms: int = MUAP_WIN_MS,
    max_lag_ms: float = 2.0,
    informative_fraction: float = 0.1,
) -> SpikeMUAPInspection:
    """Compare raw and interference-reduced views with one fixed MUAP template.

    The reference is the raw leave-one-out average of the inspected unit's
    other discharges. Timestamp-aligned templates from ``other_unit_timestamps``
    are subtracted only from the selected event, so switching views never moves
    the reference. The inspected unit is never removed.

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

    timestamps_unique = _finite_integer_timestamps(raw_timestamps)
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
    raw_selected = emg[:, sample - half_window : sample + half_window].copy()
    contributions = _other_unit_contributions(
        emg,
        timestamps_unique,
        other_unit_timestamps,
        half_window,
    )
    reduced_events, removed_events = _interference_reduced_events(
        emg, np.array([sample], dtype=np.int64), contributions, half_window
    )
    selected = reduced_events[0]
    reference = _demean_channels(reference)
    raw_selected = _demean_channels(raw_selected)
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
    best_selected, best_similarity, best_ratio, best_shift = _align_to_reference(
        reference, selected, informative, max_lag
    )
    raw_best_selected, raw_similarity, raw_ratio, raw_best_shift = _align_to_reference(
        reference, raw_selected, informative, max_lag
    )

    dominant_channel = int(np.nanargmax(channel_energy))
    dominant_waveform = reference[dominant_channel]
    if not np.any(np.isfinite(dominant_waveform)):
        raise SpikeMUAPUnavailable("The reference MUAP has no finite waveform")
    peak_sample = int(np.nanargmax(np.abs(dominant_waveform)))
    center_shift = reference.shape[1] // 2 - peak_sample
    reference_display = _shift_without_wrap(reference, center_shift)
    selected_display = _shift_without_wrap(best_selected, center_shift)
    raw_selected_display = _shift_without_wrap(raw_best_selected, center_shift)

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
        n_subtracted_units=len(contributions),
        n_subtracted_events=removed_events,
        raw_selected_grid=_as_grid(
            raw_selected_display,
            grid_positions=grid_positions,
            grid_shape=grid_shape,
        ),
        raw_similarity=raw_similarity,
        raw_amplitude_ratio=raw_ratio,
        raw_lag_ms=float(-raw_best_shift / sampling_rate * 1000.0),
    )
