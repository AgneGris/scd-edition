"""Functions to compute motor unit properties"""

import itertools
from copy import copy
from typing import Union, Optional, Tuple
import numpy as np
from scipy import signal
from .muap_comp import (
    get_percentile_ch,
    get_highest_amp_ch,
    get_highest_ptp_ch,
    get_highest_iqr_ch,
    get_highest_iqr_ptp_ch,
)


def _check_mu_format(data: np.ndarray) -> np.ndarray:
    """Check data is 2D and return it.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: 2D data array.
    """

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    return data


def get_discharge_rate(
        spike_train: np.ndarray,
        timestamps: Union[list, np.ndarray]
        ) -> np.ndarray:
    """Compute the discharge rate of motor units.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        timestamps (Union[list, np.ndarray]): Array of timestamps corresponding
            to the spike train.

    Returns:
        np.ndarray: Array of discharge rates for each motor unit.
    """

    # Get number of motor units and initialise dr
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    dr = np.zeros(units)

    for unit in range(units):
        # Compute the total number of firings
        n_spikes = np.sum(spike_train[:, unit].astype(int))

        if n_spikes == 0:
            continue

        # Get firing times
        times_spikes = timestamps[spike_train[:, unit]]

        # Get total active period
        total_period = times_spikes[-1] - times_spikes[0]

        if total_period == 0:
            continue

        # Calculate interspike interval (ISI)
        isi = np.diff(times_spikes)

        # Find silent periods (i.e. where the ISI is larger than the minimum
        # motor unit discharge rate). This is 4 Hz according to "Negro F (2016)
        # Multi-channel intramuscular and surface EMG decomposition by
        # convolutive blind source separation."
        silent_period = np.sum(isi[isi > 0.25])

        # Calculate the actual active period of the motor unit
        active_period = total_period - silent_period

        # Return mean discharge rate
        dr[unit] = n_spikes / active_period

    return dr


def get_number_of_spikes(spike_train: np.ndarray) -> np.ndarray:
    """Function to compute the number of spikes.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.

    Returns:
        np.ndarray: number of spikes for each motor unit
    """

    # Compute the number of spikes
    n_spikes = np.sum(spike_train.astype(int), axis=0)

    return n_spikes


def get_inst_discharge_rate(
        spike_train: np.ndarray,
        fs: Optional[int] = 2048
        ) -> np.ndarray:
    """Compute the instantaneous discharge rate of motor units.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.

    Returns:
        np.ndarray: Array of instantaneous discharge rates for each motor unit.
    """

    # Get number of motor units and initialise ints_DR
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    inst_dr = np.zeros(spike_train.shape)

    # Define hanning window
    dur = 1  # (s) for the moving average
    hann_win = np.hanning(np.round(dur * fs))

    for unit in range(units):
        # Convolve the hanning window and the binary spikes
        inst_dr[:, unit] = np.convolve(
            spike_train[:, unit], hann_win, mode='same'
            ) * 2

    return inst_dr


def get_coefficient_of_variation(
        spike_train: np.ndarray,
        timestamps: Union[list, np.ndarray]
        ) -> np.ndarray:
    """
    Calculate the coefficient of variation (CoV) for each motor unit in a spike
    train.

    Parameters:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        timestamps (Union[list, np.ndarray]): Array of timestamps
            corresponding to each time point in the spike train.

    Returns:
        np.ndarray: Array of CoV values for each motor unit, scaled by 100.

    Notes:
        - The CoV is calculated as the standard deviation of the interspike
          intervals divided by the mean interspike interval.
        - Interspike intervals greater than 0.25 s (or discharge rate less than
          4 Hz) and intervals less than 0.02 s (or discharge rate greater than
          50 Hz) are discarded, based on "Negro F (2016). Multi-channel
          intramuscular and surface EMG decomposition by convolutive blind
          source separation."
    """
    # Function implementation
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    cov = np.zeros(units)
    cov[:] = np.nan

    for unit in range(units):
        if not np.any(spike_train[:, unit]):
            continue

        times_spikes = timestamps[spike_train[:, unit]]
        isi = np.diff(times_spikes)
        isi = isi[isi < 0.25]
        cov[unit] = np.std(isi) / np.mean(isi)

    return cov * 100


def get_pulse_to_noise_ratio(
    spike_train: np.ndarray,
    ipts: np.ndarray,
    ext_fact: int = 8
    ) -> np.ndarray:
    """Compute the pulse-to-noise ratio (PNR) for each motor unit.

    Args:
    spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
        where n is the number of time points and m is the number of motor
        units.
    ipts (np.ndarray): Innervated pulse trains (IPTs) with shape (n, m),
        where n is the number of time points and m is the number of motor
        units.
    ext_fact (int, optional): Extension factor to discard initial spikes.
        Defaults to 8.

    Returns:
    np.ndarray: Array of PNR values for each motor unit.

    Notes:
    - The PNR is calculated as 20 * log10(spikes_mean / baseline_mean),
      where spikes_mean is the mean of the IPTs corresponding to the spikes
      and baseline_mean is the mean of the IPTs corresponding to the
      baseline.
    - The baseline is defined as the IPTs with amplitude lower than the
      lowest spike.
    - IPTs greater than the extension factor are considered for both spikes
      and baseline.
    """

    # Get number of motor units and initialise PNR
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    pnr = np.zeros(units)
    pnr[:] = np.nan

    # Square IPTs
    ipts2 = ipts ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline = ipts2[baseline_peaks_idx, unit]

        if len(spikes) == 0:
            continue

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes)
        baseline_mean = np.mean(baseline)

        # Compute PNR
        pnr[unit] = 20 * np.log10(spikes_mean / baseline_mean)

    return pnr


def get_silhouette_measure(
    spike_train: np.ndarray,
    ipts: np.ndarray,
    ext_fact: int = 8
) -> np.ndarray:
    """Compute the silhouette measure for each motor unit.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ips (np.ndarray): Innervated pulse trains (IPTs) with shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ext_fact (int, optional): Extension factor to discard initial spikes.
            Defaults to 8.

    Returns:
        np.ndarray: Array of silhouette measures for each motor unit.

    Notes:
        - The silhouette measure is a measure of how well-separated the spikes
          and baseline are in terms of their IPTs.
        - The silhouette measure is calculated as
          (dist_sum_baseline - dist_sum_spikes) / max_dist, where
          dist_sum_baseline is the sum of squared distances between each
          baseline IPT and the mean baseline IPT, dist_sum_spikes is the sum
          of squared distances between each spike IPT and the mean spike IPT,
          and max_dist is the maximum of dist_sum_baseline and dist_sum_spikes.
    """

    # Get number of motor units and initialise sil
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    sil = np.zeros(units)
    sil[:] = np.nan

    # Square IPTs
    ipts2 = ipts ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes_amp = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes_amp)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline_amp = ipts2[baseline_peaks_idx, unit]

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes_amp)
        baseline_mean = np.mean(baseline_amp)

        # Compute distances
        dist_sum_spikes = np.sum(np.power((spikes_amp - spikes_mean), 2))
        dist_sum_baseline = np.sum(np.power((spikes_amp - baseline_mean), 2))

        # Compute sil
        max_dist = np.amax([dist_sum_spikes, dist_sum_baseline])
        if max_dist == 0:
            sil[unit] = 0
        else:
            sil[unit] = (dist_sum_baseline - dist_sum_spikes) / max_dist

    return sil


def get_spike_baseline_amp(
    spike_train: np.ndarray,
    ipts: np.ndarray,
    ext_fact: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the median amplitude of spikes and baseline for each motor unit.

    Parameters:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ipts (np.ndarray): Innervated pulse trains (IPTs) with shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ext_fact (int, optional): Extension factor to discard initial spikes.
            Defaults to 8.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing two arrays:
            - spikes_amp: Array of median amplitudes of spikes for each motor unit.
            - base_amp: Array of median amplitudes of baseline for each motor unit.
    """
    # Get number of motor units and initialise output variables
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    spikes_amp = np.zeros(units)
    spikes_amp[:] = np.nan
    base_amp = np.zeros(units)
    base_amp[:] = np.nan

    for unit in range(units):
        # Get the spikes indexes discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if np.any(spikes_idx):
            spikes_amp[unit] = np.median(ipts[spikes_idx, unit])

        # Get the baseline indexes discarding the extension factor
        baseline_peaks_idx = np.nonzero(
            np.logical_not(spike_train[:, unit].astype(bool))
            )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]

        if np.any(baseline_peaks_idx):
            base_amp[unit] = np.median(ipts[baseline_peaks_idx, unit])

    return spikes_amp, base_amp


def find_reliable_units(
    dr: np.ndarray,
    cov: np.ndarray,
    sil: np.ndarray,
    pnr: np.ndarray,
    dr_low_thr: Optional[float] = 3,
    dr_upp_thr: Optional[float] = 40,
    cov_thr: Optional[float] = 40,
    sil_thr: Optional[float] = 0.9,
    pnr_thr: Optional[float] = 30
) -> np.ndarray:
    """
    Find reliable motor units based on specified thresholds.

    Args:
        dr (np.ndarray): Array of discharge rates for each motor unit.
        cov (np.ndarray): Array of coefficient of variation for each motor unit.
        sil (np.ndarray): Array of silhouette measures for each motor unit.
        pnr (np.ndarray): Array of pulse-to-noise ratios for each motor unit.
        dr_low_thr (float, optional): Lower threshold for discharge rate. Defaults to 3.
        dr_upp_thr (float, optional): Upper threshold for discharge rate. Defaults to 40.
        cov_thr (float, optional): Threshold for coefficient of variation. Defaults to 40.
        sil_thr (float, optional): Threshold for silhouette measure. Defaults to 0.9.
        pnr_thr (float, optional): Threshold for pulse-to-noise ratio. Defaults to 30.

    Returns:
        np.ndarray: Boolean array indicating which motor units are reliable.
    """
    aux = np.vstack((
        dr >= dr_low_thr,
        dr <= dr_upp_thr,
        cov <= cov_thr,
        sil >= sil_thr,
        pnr >= pnr_thr
    ))
    reliable_units = np.all(aux, axis=0)
    return reliable_units


def get_muaps(
        spike_trains: np.ndarray,
        emg_ch_array: np.ndarray,
        fs: Optional[int] = 2048,
        win_ms: Optional[int] = 25
        ) -> np.ndarray:
    """
    Compute the motor unit action potentials (MUAPs) from spike trains and
    surface electromyography (sEMG) signals.

    Args:
        spike_trains (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        emg_ch_array (np.ndarray): sEMG signal matrix of shape (r, c, n), 
            where r is the number of rows, c is the number of columns, and n 
            is the number of time points.
        fs (int, optional): Sampling frequency of the sEMG signal. Defaults to 
            2048 Hz.
        win_ms (int, optional): Window size in milliseconds. Defaults to 25.

    Returns:
        np.ndarray: Array of MUAPs with shape (m, r, c, win_samples), where m is
            the number of motor units, r is the number of rows, c is the number
            of columns, and win_samples is the number of samples in the window.

    Notes:
        - The MUAPs are computed by aligning the sEMG signals to the spike
          occurrences in the spike trains and averaging the aligned signals.
        - The window size is centered around each spike occurrence.
    """

    # Initialise dimensions
    spike_trains = _check_mu_format(spike_trains.astype(bool))
    rows, cols, samples = emg_ch_array.shape
    half_win = round(win_ms/2/1000*fs)

    # Check spike train dimensions
    if len(spike_trains) > 0:
        if len(spike_trains.shape) == 1:
            spike_trains = np.expand_dims(spike_trains, axis=-1)

        # Initialise units and muaps
        units = spike_trains.shape[1]
        muaps = np.empty((units, rows, cols, half_win * 2))

        for unit in range(units):
            # Get the firings that fit a window around them
            unit_firings = np.nonzero(spike_trains[:, unit])[0]
            unit_firings = unit_firings[
                (unit_firings - half_win >= 0) &
                (unit_firings + half_win <= samples-1)
                ]
            n_unit_firings = len(unit_firings)

            # Initialise muap samples
            muaps_aux = np.empty((n_unit_firings, rows, cols, half_win*2))

            # Get all the muap samples for each unit firing
            for i, firing in enumerate(unit_firings):

                mask = np.arange(half_win * 2) - half_win + firing
                muaps_aux[i] = emg_ch_array[:, :, mask]

            # Compute mean
            muaps[unit] = np.mean(muaps_aux, axis=0)
    else:
        muaps = np.empty((0, rows, cols, half_win*2))

    return muaps


def center_muaps(muaps: np.ndarray) -> np.ndarray:
    """
    Center the motor unit action potentials (MUAPs) around the peak amplitude.

    Parameters:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.

    Returns:
        np.ndarray: Array of centered MUAPs with shape (m, r, c, win_samples).

    Notes:
        - The MUAPs are centered by shifting the samples so that the peak
          amplitude is at the center of the window.
    """

    # Check muaps dimensions
    if len(muaps) > 0:
        # There are units
        if len(muaps.shape) < 4:
            muaps = np.expand_dims(muaps, axis=0)

        # Initialise variables
        units, rows, cols, samples = muaps.shape
        center_sample = samples//2
        centered_muaps = copy(muaps)

        for unit in range(units):
            # Get current muap and peak amplitude channel
            muap = muaps[unit]
            ch_row, ch_col = np.unravel_index(
                np.nanargmax(np.abs(muap), axis=-1), (rows, cols)
                )

            #  Get the sample at which the amplitude is max
            max_sample = np.nanargmax(np.abs(muap[ch_row, ch_col]))

            # Center muap
            centered_muaps[unit] = np.roll(
                centered_muaps[unit], center_sample-max_sample, axis=-1
                )
    else:
        #  No units
        centered_muaps = copy(muaps)

    return centered_muaps


def get_muap_waveform_length(
    muaps: np.ndarray,
    sel_chs_by: Optional[str] = "iqr"
) -> np.ndarray:
    """
    Compute the waveform length of motor unit action potentials (MUAPs).

    Parameters:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".

    Returns:
        np.ndarray: Array of waveform lengths with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The waveform length is computed as the sum of absolute differences
          between consecutive samples of each MUAP.
        - If sel_chs_by is specified, the waveform length is computed only for
          the selected channels based on the specified method.
    """
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units = muaps.shape[0]

    if sel_chs_by is None:
        # Compute muap length for each muap channel
        wl = np.sum(np.abs(np.diff(muaps, axis=-1)), axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap length
        wl = np.empty(muaps.shape[0:3])
        wl[:] = np.nan

        # Compute muap length
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            wl[unit][sel_chs_mask] = np.sum(np.abs(
                np.diff(muap_sel_chs, axis=-1)
                ), axis=-1)

    return wl


def get_muap_energy(
    muaps: np.ndarray,
    sel_chs_by: Optional[str] = "iqr"
) -> np.ndarray:
    """
    Compute the energy of motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".

    Returns:
        np.ndarray: Array of energy values with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The energy is computed as the sum of squared values of each MUAP.
        - If sel_chs_by is specified, the energy is computed only for the
          selected channels based on the specified method.
    """
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units = muaps.shape[0]

    if sel_chs_by is None:
        # Compute muap energy for each muap channel
        energy = np.sum(np.power(muaps, 2), axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap energy
        energy = np.empty(muaps.shape[0:3])
        energy[:] = np.nan

        # Compute muap energy
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            energy[unit][sel_chs_mask] = np.sum(
                np.power(muap_sel_chs, 2), axis=-1
            )

    return energy


def get_muap_ptp(muaps: np.ndarray, sel_chs_by: Optional[str] = "iqr") -> np.ndarray:
    """
    Compute the peak-to-peak amplitude of motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".

    Returns:
        np.ndarray: Array of peak-to-peak amplitudes with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The peak-to-peak amplitude is computed as the difference between the
          maximum and minimum values of each MUAP.
        - If sel_chs_by is specified, the peak-to-peak amplitude is computed only
          for the selected channels based on the specified method.
    """
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units = muaps.shape[0]

    if sel_chs_by is None:
        # Compute muap peak to peak amplitude for each muap channel
        ptp = np.ptp(muaps, axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap peak to peak amplitude
        ptp = np.empty(muaps.shape[0:3])
        ptp[:] = np.nan

        # Compute muap peak to peak amplitude
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            ptp[unit][sel_chs_mask] = np.ptp(muap_sel_chs, axis=-1)

    return ptp


def get_muap_ptp_time(
    muaps: np.ndarray,
    sel_chs_by: Optional[str] = "iqr",
) -> np.ndarray:
    """
    Compute the peak-to-peak time (in samples) of motor unit action potentials (MUAPs).

    Parameters:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".

    Returns:
        np.ndarray: Array of peak-to-peak times (in samples) with shape (m, r, c),
            where m is the number of motor units, r is the number of rows, and
            c is the number of columns.

    Notes:
        - The peak-to-peak time is computed as the absolute difference between
          the sample index of the maximum and minimum values of each MUAP.
        - If sel_chs_by is specified, the peak-to-peak time is computed only for
          the selected channels based on the specified method.
    """
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units = muaps.shape[0]

    if sel_chs_by is None:
        # Compute muap peak to peak time (in samples) for each muap channel
        ptp_time = np.abs(
            np.argmax(muaps, axis=-1) - np.argmin(muaps, axis=-1)
        )
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap peak to peak time (in samples)
        ptp_time = np.empty(muaps.shape[0:3])
        ptp_time[:] = np.nan

        # Compute muap peak to peak time (in samples)
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            ptp_time[unit][sel_chs_mask] = np.abs(
                np.argmax(muap_sel_chs, axis=-1) -
                np.argmin(muap_sel_chs, axis=-1)
            )

    return ptp_time


def get_muap_peak_frequency(
    muaps: np.ndarray,
    sel_chs_by: Optional[str] = "iqr",
    fs: Optional[int] = 2048
) -> np.ndarray:
    """
    Compute the peak frequency of motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".
        fs (int, optional): Sampling frequency in Hz. Defaults to 2048.

    Returns:
        np.ndarray: Array of peak frequencies with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The peak frequency is computed as the frequency corresponding to the
          maximum value in the power spectrum of each MUAP.
        - If sel_chs_by is specified, the peak frequency is computed only for
          the selected channels based on the specified method.
    """
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, rows, cols, samples = muaps.shape
    ps = np.power(
        np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
        )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]
    max_ps_idx = np.argmax(ps, axis=-1)

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)
    else:
        sel_chs_mask = np.ones((muaps.shape[1:3])).astype(bool)

    # Initialise peak frequency
    peak_freq = np.empty(muaps.shape[0:3])
    peak_freq[:] = np.nan
    for unit, row, col in itertools.product(
            range(units), range(rows), range(cols)
            ):
        if sel_chs_mask[row, col] is False:
            continue
        peak_freq[unit, row, col] = freq[max_ps_idx[unit, row, col]]

    return peak_freq


def get_muap_median_frequency(
        muaps: np.ndarray,
        sel_chs_by: Optional[str] = "iqr",
        fs: Optional[int] = 2048
    ) -> np.ndarray:
    """
    Compute the median frequency of motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".
        fs (int, optional): Sampling frequency in Hz. Defaults to 2048.

    Returns:
        np.ndarray: Array of median frequencies with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The median frequency is computed as the frequency corresponding to the
          cumulative power spectrum reaching half of its total power.
        - If sel_chs_by is specified, the median frequency is computed only for
          the selected channels based on the specified method.
    """

    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, rows, cols, samples = muaps.shape
    ps = np.power(
        np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
        )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]

    # Compute peak frequency
    cum_ps = np.cumsum(ps, axis=-1)
    med_ps = np.sum(ps, axis=-1)/2

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)
    else:
        sel_chs_mask = np.ones((muaps.shape[1:3])).astype(bool)

    # Initialise peak frequency
    med_freq = np.empty(muaps.shape[0:3])
    med_freq[:] = np.nan
    for unit, row, col in itertools.product(
            range(units), range(rows), range(cols)
            ):
        if sel_chs_mask[row, col] is False:
            continue
        med_freq[unit, row, col] = freq[np.argmin(np.abs(
            cum_ps[unit, row, col] -
            med_ps[unit, row, col]
            ))]
    return med_freq


def get_muap_mean_frequency(
        muaps: np.ndarray,
        sel_chs_by: Optional[str] = "iqr",
        fs: Optional[int] = 2048
    ) -> np.ndarray:
    """
    Compute the mean frequency of motor unit action potentials (MUAPs).

    Parameters:
        muaps (np.ndarray): Array of MUAPs with shape (m, r, c, win_samples),
            where m is the number of motor units, r is the number of rows, c
            is the number of columns, and win_samples is the number of samples
            in the window.
        sel_chs_by (str, optional): Method for selecting channels. Defaults to "iqr".
        fs (int, optional): Sampling frequency in Hz. Defaults to 2048.

    Returns:
        np.ndarray: Array of mean frequencies with shape (m, r, c), where m is
            the number of motor units, r is the number of rows, and c is the
            number of columns.

    Notes:
        - The mean frequency is computed as the weighted average of the frequency
          components in the power spectrum of each MUAP.
        - If sel_chs_by is specified, the mean frequency is computed only for
          the selected channels based on the specified method.
    """

    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, _, _, samples = muaps.shape
    ps = np.power(
            np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
            )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]

    # Compute peak frequency
    mean_freq = np.sum(ps * freq, axis=-1)/np.sum(ps, axis=-1)

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": get_highest_iqr_ch,
            "iqr_ptp": get_highest_iqr_ptp_ch,
            "max_amp": get_highest_amp_ch,
            "ptp": get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Discard non selected channels
        for unit in range(units):
            mean_freq[unit][np.logical_not(sel_chs_mask)] = np.nan

    return mean_freq
