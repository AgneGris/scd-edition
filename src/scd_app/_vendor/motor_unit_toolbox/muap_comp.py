"""MUAP channel-selection helpers from Motor Unit Toolbox 1.0.

This is the subset imported by ``props.py``. It is preserved from upstream
commit d84dc6c943daa5d686b2911b48b13bf718628130 under the MIT License.
"""

from typing import Optional

import numpy as np


def get_highest_iqr_ch(muap: np.ndarray) -> np.ndarray:
    """Compute the highest interquartile range (IQR) channels."""
    peak_amps = np.amax(np.abs(muap), axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    q1 = np.percentile(peak_amps, 25)
    q3 = np.percentile(peak_amps, 75)
    iqr = q3 - q1
    whis = 1.5
    outliers = peak_amps > q3 + whis * iqr
    return np.reshape(outliers, (muap.shape[0], muap.shape[1]))


def get_percentile_ch(muap: np.ndarray, thr: Optional[int] = 90):
    """Compute channels whose amplitude exceeds a percentile threshold."""
    peak_amps = np.amax(np.abs(muap), axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    outliers = peak_amps > np.percentile(peak_amps, thr)
    return np.reshape(outliers, (muap.shape[0], muap.shape[1]))


def get_highest_iqr_ptp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute channels with outlying peak-to-peak amplitude."""
    peak_amps = np.ptp(muap, axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    q1 = np.percentile(peak_amps, 25)
    q3 = np.percentile(peak_amps, 75)
    iqr = q3 - q1
    whis = 1.5
    outliers = peak_amps > q3 + whis * iqr
    return np.reshape(outliers, (muap.shape[0], muap.shape[1]))


def get_highest_amp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute channels with the highest absolute amplitude."""
    peak_amps = np.amax(np.abs(muap), axis=-1)
    amp_thr = 3 * np.std(peak_amps)
    return peak_amps > amp_thr


def get_highest_ptp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute channels with the highest peak-to-peak amplitude."""
    ptp_amps = np.ptp(muap, axis=-1)
    amp_thr = 2 * np.std(ptp_amps)
    return ptp_amps > amp_thr
