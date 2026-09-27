"""Automatic suggestions for splitting a merged motor-unit discharge train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitSuggestion:
    """A complete two-way timestamp partition proposed from source peak heights."""

    group_a: np.ndarray
    group_b: np.ndarray
    threshold: float
    separation_score: float
    lower_mean_height: float
    upper_mean_height: float


def suggest_split_by_peak_height(
    timestamps: np.ndarray,
    source: np.ndarray,
    *,
    min_group_size: int = 2,
) -> SplitSuggestion:
    """Partition all spikes at the best two-mode source-height threshold.

    The displayed source is squared, so each spike is represented by
    ``source[timestamp] ** 2``.  Heights are log-scaled to prevent a handful of
    very large peaks from dominating the suggestion.  Every possible sorted
    threshold that leaves at least ``min_group_size`` spikes on each side is
    evaluated, and the threshold with the smallest total within-group squared
    error is chosen.  This is the exact deterministic solution to one-
    dimensional two-means clustering.

    Group A is always the higher-height mode and group B the lower-height mode.
    ``separation_score`` is the fraction of total log-height variance explained
    by the two groups (0 to 1).  The result is a suggestion for user review,
    never an automatic committed edit.
    """
    raw_timestamps = np.asarray(timestamps).reshape(-1)
    source_array = np.asarray(source, dtype=float).reshape(-1)
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    if raw_timestamps.size < 2 * min_group_size:
        raise ValueError(
            f"Need at least {2 * min_group_size} spikes for an automatic split"
        )
    if not np.all(np.isfinite(raw_timestamps)) or not np.allclose(
        raw_timestamps, np.rint(raw_timestamps)
    ):
        raise ValueError("Spike timestamps must be finite integer sample indices")

    timestamp_array = np.rint(raw_timestamps).astype(np.int64)
    if np.any(timestamp_array < 0) or np.any(timestamp_array >= source_array.size):
        raise ValueError("Spike timestamps fall outside the source signal")

    with np.errstate(over="ignore", invalid="ignore"):
        heights = np.square(source_array[timestamp_array])
    if not np.all(np.isfinite(heights)):
        raise ValueError("Source heights at the spike timestamps are not finite")

    positive = heights[heights > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    features = np.log1p(heights / max(scale, np.finfo(float).tiny))
    order = np.argsort(features, kind="stable")
    sorted_features = features[order]
    sorted_heights = heights[order]

    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_features)))
    prefix_square_sum = np.concatenate(
        ([0.0], np.cumsum(np.square(sorted_features)))
    )

    def interval_sse(start: int, stop: int) -> float:
        count = stop - start
        total = prefix_sum[stop] - prefix_sum[start]
        square_total = prefix_square_sum[stop] - prefix_square_sum[start]
        return max(0.0, float(square_total - total * total / count))

    candidates = []
    n_spikes = len(sorted_features)
    for cut in range(min_group_size, n_spikes - min_group_size + 1):
        # A threshold cannot separate equal-height observations consistently.
        if sorted_features[cut - 1] >= sorted_features[cut]:
            continue
        within_sse = interval_sse(0, cut) + interval_sse(cut, n_spikes)
        boundary_gap = float(sorted_features[cut] - sorted_features[cut - 1])
        balance = abs(cut - n_spikes / 2)
        candidates.append((within_sse, -boundary_gap, balance, cut))

    if not candidates:
        raise ValueError("Spike peak heights do not contain two separable levels")

    best_sse, _negative_gap, _balance, best_cut = min(candidates)
    total_mean = float(np.mean(features))
    total_sse = float(np.sum(np.square(features - total_mean)))
    if total_sse <= np.finfo(float).eps:
        raise ValueError("Spike peak heights do not contain two separable levels")

    lower_boundary = float(sorted_heights[best_cut - 1])
    upper_boundary = float(sorted_heights[best_cut])
    threshold = (lower_boundary + upper_boundary) / 2.0
    in_group_a = heights > threshold
    group_a = timestamp_array[in_group_a]
    group_b = timestamp_array[~in_group_a]
    if len(group_a) < min_group_size or len(group_b) < min_group_size:
        raise ValueError("Automatic split did not produce two usable groups")

    separation_score = float(np.clip(1.0 - best_sse / total_sse, 0.0, 1.0))
    return SplitSuggestion(
        group_a=group_a,
        group_b=group_b,
        threshold=threshold,
        separation_score=separation_score,
        lower_mean_height=float(np.mean(heights[~in_group_a])),
        upper_mean_height=float(np.mean(heights[in_group_a])),
    )
