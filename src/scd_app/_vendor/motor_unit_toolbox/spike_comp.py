"""Functions to compare spike trains"""
from typing import Tuple, Optional, List, Union
import itertools
import numpy as np
from scipy import signal


def rate_of_agreement_paired(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the rate of agreement between two sets of paired spike trains.

    Args:
        spike_trains_ref (np.ndarray): Reference spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains,
              with shape (n).
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n).

    Note:
        - The function assumes that the spike trains between the sets are matched 
          and in the same order.
    """
    # Check spike trains shape
    if len(spike_trains_ref.shape) == 1:
        spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if len(spike_trains_test.shape) == 1:
        spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)

    if spike_trains_ref.shape != spike_trains_test.shape:
        raise ValueError(f'Dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms / 1000 * fs)
    tol_train = round(tol_train_ms / 1000 * fs)

    # Initialise test variables
    n_units = spike_trains_test.shape[1]

    #  If there are no spikes return empty RoA
    if (not np.any(spike_trains_ref)) | (not np.any(spike_trains_test)):
        pair_idx = np.arange(n_units)
        pair_lag = np.zeros((n_units))
        roa = np.zeros((n_units))
        return roa, pair_idx, pair_lag

    # Compute the RoA between the sets
    #  --------------------------------
    # Initialise correlation variables
    spikes_corr = np.zeros((n_units))
    roa = np.empty((n_units))
    pair_lag = np.zeros((n_units))
    pair_idx = [(unit, unit) for unit in range(n_units)]

    for unit in range(n_units):
        #  Align spike trains based on their correlation and spike tol
        # -----------------------------------------------------------
        #  Get trains
        train_ref = spike_trains_ref[:, unit]
        train_test = spike_trains_test[:, unit]
        # Apply spike tolerance
        train_ref = np.convolve(train_ref, np.ones(tol_spike), mode="same")
        train_test = np.convolve(train_test, np.ones(tol_spike), mode="same")
        # Compute correlation and lags
        curr_corr = signal.correlate(train_ref, train_test, mode="full")
        curr_lags = signal.correlation_lags(
            len(train_ref), len(train_test), mode="full"
        )
        # Apply train shift tolerance
        train_tol_idxs = np.nonzero(np.abs(curr_lags) == tol_train)[0]
        train_tol_mask = np.arange(train_tol_idxs[0], train_tol_idxs[-1] + 1).astype(
            int
        )
        curr_corr = curr_corr[train_tol_mask]
        curr_lags = curr_lags[train_tol_mask]
        # Identify optimal lag for alignment
        trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
        if not np.isscalar(trains_lag):
            # If there is more than one possible lag, choose the minimum
            trains_lag = np.amin(trains_lag)
        # Fill matrices
        spikes_corr[unit] = np.amax(curr_corr)
        pair_lag[unit] = trains_lag

        #  Compute rate of agreement between the aligned spike trains
        # ----------------------------------------------------------
        # Align spike trains
        firings_ref = np.nonzero(spike_trains_ref[:, unit])[0]
        firings_test = np.nonzero(spike_trains_test[:, unit])[0] + pair_lag[unit]
        # Initialise variables
        firings_common = 0
        firings_ref_only = 0
        firings_test_only = 0
        # Pair firings
        for firing in firings_ref:
            curr_firing_diff = np.abs(firings_test - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A common firing
                firings_common += 1
                firings_test = np.delete(firings_test, np.argmin(curr_firing_diff))
            else:
                # Only in reference firings
                firings_ref_only += 1
        firings_test_only = len(firings_test)
        # Compute rate of agreement
        roa[unit] = firings_common / (
            firings_common + firings_ref_only + firings_test_only
        )

    return roa, pair_idx, pair_lag


def rate_of_agreement(
    spike_trains_ref: Union[np.ndarray, None],
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the rate of agreement between two sets of spike trains.

    Args:
        spike_trains_ref (Union[np.ndarray, None]): Reference spike trains 
            with shape (m, n1) where m is the number of samples and n1 is the
            number of motor units in the reference set. If None are provided, 
            the function will compute the RoA within the test set.
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n2),
            where m is the number of samples and n2 is the number of motor units
            in the test set.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains.
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units.

    Note:
        - The function does not assume that the spike trains between the sets are 
          matched nor in the same order.
        - The dimensions of the output arrays will depend on the number of matched
          pairs between the sets.
    """

    # Check spike trains shape
    if spike_trains_ref is not None:
        if len( spike_trains_ref.shape ) == 1:
            spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if spike_trains_test is not None:
        if len( spike_trains_test.shape ) == 1:
            spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)
    
    if spike_trains_ref.shape[0] != spike_trains_test.shape[0]:
        raise ValueError(f'Time dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms/1000 * fs)
    tol_train = round(tol_train_ms/1000 * fs)

    # Initialise test variables
    n_units_test = spike_trains_test.shape[1]

    #  If no spike trains to test are provided, return empty RoA
    if not np.any(spike_trains_test):
        pair_idx = np.arange(n_units_test)
        pair_lag = np.zeros((n_units_test))
        roa = np.zeros((n_units_test))
        return roa, pair_idx, pair_lag

    if spike_trains_ref is None:
        # Only one set provided, compute the RoA within set
        # -------------------------------------------------
        # Initialise correlation variables
        spikes_corr = np.zeros((n_units_test, n_units_test))
        spikes_lag = np.zeros((n_units_test, n_units_test))

        #  Align spike trains based on their correlation and spike tol
        pairs = itertools.combinations(range(n_units_test), 2)
        for pair in pairs:
            #  Get trains
            train_0 = spike_trains_test[:, pair[0]]
            train_1 = spike_trains_test[:, pair[1]]
            # Apply spike tolerance
            train_0 = np.convolve(train_0, np.ones(tol_spike), mode="same")
            train_1 = np.convolve(train_1, np.ones(tol_spike), mode="same")
            # Compute correlation and lags
            curr_corr = signal.correlate(train_0, train_1, mode="full")
            curr_lags = signal.correlation_lags(len(train_0), len(train_1), mode="full")
            # Identify optimal lag for alignment
            trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
            if not np.isscalar(trains_lag):
                # If there is more than one possible lag, choose the minimum
                trains_lag = np.amin(trains_lag)
            # Ensure alignment is within tolerance
            if np.abs(trains_lag) > tol_train:
                trains_lag = 0
            # Fill matrices
            spikes_corr[pair] = np.amax(curr_corr)
            spikes_lag[pair] = int(trains_lag)

    else:
        # Compute the RoA between the sets
        #  --------------------------------
        # Initialise reference variables
        n_units_ref = spike_trains_ref.shape[-1]

        # Initialise correlation variables
        spikes_corr = np.zeros((n_units_ref, n_units_test))
        spikes_lag = np.zeros((n_units_ref, n_units_test))

        #  Align spike trains based on their correlation and spike tol
        for unit_ref in range(n_units_ref):
            for unit_test in range(n_units_test):
                #  Get trains
                train_0 = spike_trains_ref[:, unit_ref]
                train_1 = spike_trains_test[:, unit_test]
                # Apply spike tolerance
                train_0 = np.convolve(train_0, np.ones(tol_spike), mode="same")
                train_1 = np.convolve(train_1, np.ones(tol_spike), mode="same")
                # Compute correlation and lags
                curr_corr = signal.correlate(train_0, train_1, mode="full")
                curr_lags = signal.correlation_lags(
                    len(train_0), len(train_1), mode="full"
                )
                # Identify optimal lag for alignment
                trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
                if not np.isscalar(trains_lag):
                    # If there is more than one possible lag, choose the minimum
                    trains_lag = np.amin(trains_lag)
                # Fill matrices
                spikes_corr[unit_ref, unit_test] = np.amax(curr_corr)
                spikes_lag[unit_ref, unit_test] = int(trains_lag)

    # Find most likely pairs by progressively taking the max corr
    pair_idx = []
    pair_lag = []
    while np.any(sum(spikes_corr)):
        idx_max_corr = np.unravel_index(np.argmax(spikes_corr), spikes_corr.shape)
        pair_idx.append(idx_max_corr)
        pair_lag.append(int(spikes_lag[idx_max_corr]))
        spikes_corr[idx_max_corr[0], :] = 0
        spikes_corr[:, idx_max_corr[1]] = 0
        if spike_trains_ref is None:
            spikes_corr[idx_max_corr[1], :] = 0
            spikes_corr[:, idx_max_corr[0]] = 0

    # Compute rate of agreement
    roa = np.empty((len(pair_idx)))
    for i, pair in enumerate(pair_idx):
        # Get corresponding firings and apply optimal lag
        if spike_trains_ref is None:
            firings_0 = np.nonzero(spike_trains_test[:, pair[0]])[0]
        else:
            firings_0 = np.nonzero(spike_trains_ref[:, pair[0]])[0]
        firings_1 = np.nonzero(spike_trains_test[:, pair[1]])[0] + pair_lag[i]

        # Initialise variables
        # len_firings_1 = len(firings_1)
        firings_common = 0
        firings_0_only = 0
        firings_1_only = 0
        # Pair firings
        for firing in firings_0:
            curr_firing_diff = np.abs(firings_1 - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A common firing
                firings_common += 1
                firings_1 = np.delete(firings_1, np.argmin(curr_firing_diff))
            else:
                # Only in firings 0
                firings_0_only += 1
        firings_1_only = len(firings_1)
        # Compute rate of agreement
        roa[i] = firings_common / (firings_common + firings_0_only + firings_1_only)

    #  Align the indexes to the reference
    first_pair = [pair[0] for pair in pair_idx]
    pairs_sort_idx = np.argsort(first_pair)

    roa_sorted = roa[pairs_sort_idx]
    pair_idx_sorted = [pair_idx[i] for i in pairs_sort_idx]
    pair_lag_sorted = [int(pair_lag[i]) for i in pairs_sort_idx]

    return roa_sorted, pair_idx_sorted, pair_lag_sorted

def rate_of_agreement_full(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rate of agreement between two sets of spike trains.

    Args:
        spike_trains_ref (np.ndarray): Reference spike trains  with shape 
            (m, n1) where m is the number of samples and n1 is the number of 
            motor units in the reference set. 
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n2),
            where m is the number of samples and n2 is the number of motor units
            in the test set.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains,
              with shape (n1, n2).
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n1, n2).

    Note:
        - The function does not assume that the spike trains between the sets are 
          matched nor in the same order.
    """

    # Dimensionality checks
    # ---------------------
    # Check spike trains shape
    if len( spike_trains_ref.shape ) == 1:
        spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if len( spike_trains_test.shape ) == 1:
        spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)
    
    if spike_trains_ref.shape[0] != spike_trains_test.shape[0]:
        raise ValueError(f'Time dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Initialisation
    # --------------
    # Put tolerances into samples
    tol_spike = round(tol_spike_ms/1000 * fs)
    tol_train = round(tol_train_ms/1000 * fs)

    # Initialise variables
    n_units_test = spike_trains_test.shape[1]
    n_units_ref = spike_trains_ref.shape[-1]

    # Initialise correlation variables
    spikes_corr = np.zeros((n_units_ref, n_units_test))
    spikes_lag = np.zeros((n_units_ref, n_units_test))
    roa = np.empty((n_units_ref, n_units_test))

    # Compute cross correlation between units to align the spike trains
    # -----------------------------------------------------------------
    for unit_ref, unit_test in itertools.product(range(n_units_ref), range(n_units_test)):
        #  Get trains
        train_0 = spike_trains_ref[:, unit_ref]
        train_1 = spike_trains_test[:, unit_test]
        # Apply spike tolerance
        train_0 = np.convolve(train_0, np.ones(tol_spike), mode="same")
        train_1 = np.convolve(train_1, np.ones(tol_spike), mode="same")
        # Compute correlation and lags
        curr_corr = signal.correlate(train_0, train_1, mode="full")
        curr_lags = signal.correlation_lags(
            len(train_0), len(train_1), mode="full"
        )
        # Identify optimal lag for alignment
        trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
        if not np.isscalar(trains_lag):
            # If there is more than one possible lag, choose the minimum
            trains_lag = np.amin(trains_lag)
            # Check if the lag is within tolerance
            if np.abs(trains_lag) > trains_lag:
                trains_lag = np.sign(trains_lag) * tol_train

        # Fill matrices
        spikes_corr[unit_ref, unit_test] = np.amax(curr_corr)
        spikes_lag[unit_ref, unit_test] = int(trains_lag)

    # Compute RoA between aligned spike trains for all pairs
    # ------------------------------------------------------
    for unit_ref, unit_test in itertools.product(range(n_units_ref), range(n_units_test)):
        # Get corresponding firings and apply optimal lag
        firings_0 = np.nonzero(spike_trains_ref[:, unit_ref])[0]
        firings_1 = np.nonzero(spike_trains_test[:, unit_test])[0] + spikes_lag[unit_ref, unit_test]

        # Initialise variables
        # len_firings_1 = len(firings_1)
        firings_common = 0
        firings_0_only = 0
        firings_1_only = 0
        # Pair firings
        for firing in firings_0:
            curr_firing_diff = np.abs(firings_1 - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A common firing
                firings_common += 1
                firings_1 = np.delete(firings_1, np.argmin(curr_firing_diff))
            else:
                # Only in firings 0
                firings_0_only += 1
        firings_1_only = len(firings_1)
        # Compute rate of agreement
        roa[unit_ref, unit_test] = firings_common / (firings_common + firings_0_only + firings_1_only)

    return roa, spikes_lag

def rate_of_agreement_all(
    spike_trains: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the rate of agreement within a set of spike trains.

    Args:
        spike_trains (np.ndarray): Test spike trains with shape (m, n), where m
          is the number of samples and n is the number of motor units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains,
              with shape (n).
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n).

    Note:
        - The function does not assume that the spike trains between the sets are 
          matched nor in the same order.
    """

    if len(spike_trains) < 1:
        # No units
        spikes_roa = np.empty(0)
        spikes_lags = np.empty(0)
    else:
        # Put tolerances into samples
        tol_spike = round(tol_spike_ms / 1000 * fs)
        if tol_train_ms is not None:
            tol_train = round(tol_train_ms / 1000 * fs)
        else:
            tol_train = None

        # Ensure right format and get number of units
        if len(spike_trains.shape) == 1:
            spike_trains = np.expand_dims(spike_trains, axis=-1)
        samples, n_units = spike_trains.shape
        if samples < n_units:
            raise Warning("The number of samples is less than the number of units. Consider transposing the spike trains.")

        # Compute RoA across all unique pairs
        if n_units == 1:
            spikes_roa = np.array([1])
            spikes_lags = np.array([0])
        else:

            # Initialise correlation variables
            spikes_roa = np.ones((n_units, n_units))
            spikes_lags = np.zeros((n_units, n_units))

            pairs = itertools.combinations(range(n_units), 2)

            for pair in pairs:

                #  Get alignment between spike trains
                # ----------------------------------
                #  Get trains
                train_0 = spike_trains[:, pair[0]]
                train_1 = spike_trains[:, pair[1]]

                # Apply spike tolerance (On both sides *2)
                train_0 = np.convolve(train_0, np.ones(tol_spike * 2), mode="same")
                train_1 = np.convolve(train_1, np.ones(tol_spike * 2), mode="same")

                # Compute correlation and lags
                curr_corr = signal.correlate(train_0, train_1, mode="full")
                curr_lags = signal.correlation_lags(
                    len(train_0), len(train_1), mode="full"
                )

                # Identify optimal lag for alignment
                trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
                if not np.isscalar(trains_lag):
                    # If there is more than one possible lag, choose the minimum
                    trains_lag = np.amin(trains_lag)
                # Ensure alignment is within tolerance
                if tol_train is not None:
                    if np.abs(trains_lag) > tol_train:
                        trains_lag = 0

                # Store lag
                spikes_lags[pair] = trains_lag
                spikes_lags[pair[1], pair[0]] = -trains_lag

                # Compute rate of agreement between the aligned trains
                # ----------------------------------------------------

                # Transform spike trains into firings
                firings_0 = np.nonzero(train_0)[0]
                firings_1 = np.nonzero(train_1)[0] + trains_lag

                # Initialise variables
                firings_common = 0
                firings_0_only = 0
                firings_1_only = 0

                # Pair firings
                for firing in firings_0:
                    curr_firing_diff = np.abs(firings_1 - firing)
                    if np.any(curr_firing_diff <= tol_spike):
                        # A common firing
                        firings_common += 1
                        firings_1 = np.delete(firings_1, np.argmin(curr_firing_diff))
                    else:
                        # Only in firings 0
                        firings_0_only += 1
                firings_1_only = len(firings_1)
                # Compute rate of agreement
                roa = firings_common / (
                    firings_common + firings_0_only + firings_1_only
                )

                # Store RoA
                spikes_roa[pair] = roa
                spikes_roa[pair[1], pair[0]] = roa

    return spikes_roa, spikes_lags.astype(int)


def precision_sensitivity_f1_paired(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:

    """Compute the precision, sensitivity, and f1-score between two sets of
       paired spike trains.

    Args:
        spike_trains_ref (np.ndarray): Reference spike trains with shape (m, n) 
            where m is the number of samples and n is the number of motor units. 
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
            A tuple containing:
            - precision (np.ndarray): Precision values for each motor unit, with 
              shape (n).
            - sensitivity (np.ndarray): Sensitivity values for each motor unit, with
              shape (n).
            - f1score (np.ndarray): F1-score values for each motor unit, with shape
              (n).
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n).

    Note:
        - The function assumes that the spike trains between the sets are matched 
          and in the same order.
    """

    # Check spike trains shape
    if len(spike_trains_ref.shape) == 1:
        spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if len(spike_trains_test.shape) == 1:
        spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)

    if spike_trains_ref.shape != spike_trains_test.shape:
        raise ValueError(f'Dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms / 1000 * fs)
    tol_train = round(tol_train_ms / 1000 * fs)

    # Initialise test variables
    n_units = spike_trains_test.shape[1]

    #  If there are no spikes return zeros
    if (not np.any(spike_trains_ref)) | (not np.any(spike_trains_test)):
        pair_idx = np.arange(n_units)
        pair_lag = np.zeros((n_units))
        precision = np.zeros((n_units))
        sensitivity = np.zeros((n_units))
        f1score = np.zeros((n_units))
        return precision, sensitivity, f1score, pair_idx, pair_lag

    # Compute the PRECISION, SENSITIVITY, and F1-SCORE between the sets
    #  -----------------------------------------------------------------
    #  precision = TP / (TP + FP)
    # sensitivity (recall) = TP / (TP + FN)
    # f1-score = 2 * precision * sensitivity / (precision + sensitivity)

    tp = np.zeros((n_units))  # Actual spikes
    fp = np.zeros((n_units))  #  Incorrect spikes
    fn = np.zeros((n_units))  # Missed spikes

    spikes_corr = np.zeros((n_units))
    pair_lag = np.zeros((n_units))
    pair_idx = [(unit, unit) for unit in range(n_units)]

    for unit in range(n_units):

        #  Align spike trains based on their correlation and spike tol
        # -----------------------------------------------------------
        #  Get trains
        train_ref = spike_trains_ref[:, unit]
        train_test = spike_trains_test[:, unit]
        # Apply spike tolerance
        train_ref = np.convolve(train_ref, np.ones(tol_spike), mode="same")
        train_test = np.convolve(train_test, np.ones(tol_spike), mode="same")
        # Compute correlation and lags
        curr_corr = signal.correlate(train_ref, train_test, mode="full")
        curr_lags = signal.correlation_lags(
            len(train_ref), len(train_test), mode="full"
        )
        # Apply train shift tolerance
        train_tol_idxs = np.nonzero(np.abs(curr_lags) == tol_train)[0]
        train_tol_mask = np.arange(train_tol_idxs[0], train_tol_idxs[-1] + 1).astype(
            int
        )
        curr_corr = curr_corr[train_tol_mask]
        curr_lags = curr_lags[train_tol_mask]
        # Identify optimal lag for alignment
        trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
        if not np.isscalar(trains_lag):
            # If there is more than one possible lag, choose the minimum
            trains_lag = np.amin(trains_lag)
        # Fill matrices
        spikes_corr[unit] = np.amax(curr_corr)
        pair_lag[unit] = trains_lag

        #  Compute metrics between the aligned spike trains
        # ------------------------------------------------
        # Align spike trains
        firings_ref = np.nonzero(spike_trains_ref[:, unit])[0]
        firings_test = np.nonzero(spike_trains_test[:, unit])[0] + pair_lag[unit]
        # Pair firings
        for firing in firings_test:
            curr_firing_diff = np.abs(firings_ref - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A correctly detected firing (TP)
                tp[unit] += 1
                firings_ref = np.delete(firings_ref, np.argmin(curr_firing_diff))
            else:
                # Incorrectly detected firings (FP)
                fp[unit] += 1

        # A missed firing (FN)
        fn[unit] = len(firings_ref)

        assert (tp[unit] + fp[unit]) == np.sum(spike_trains_test[:, unit])
        assert (tp[unit] + fn[unit]) == np.sum(spike_trains_ref[:, unit])

    #  Compute metrics
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1score = 2 * precision * sensitivity / (precision + sensitivity)

    return precision, sensitivity, f1score, pair_idx, pair_lag


def get_tp_fp_fn_paired(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the true positives (tp), false positives (fp), and false negatives (fn),
       between two sets of paired spike trains.

    Args:
        spike_trains_ref (np.ndarray): Reference spike trains with shape (m, n) 
            where m is the number of samples and n is the number of motor units. 
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]: 
        A tuple containing:
            - tp (np.ndarray): True positives, with shape (m, n).
            - fp (np.ndarray): False positives, with shape (m, n).
            - fn (np.ndarray): False negatives, with shape (m, n).
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n).

    Note:
        - The function assumes that the spike trains between the sets are matched 
          and in the same order.
    """


    # Check spike trains shape
    if len(spike_trains_ref.shape) == 1:
        spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if len(spike_trains_test.shape) == 1:
        spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)

    if spike_trains_ref.shape != spike_trains_test.shape:
        raise ValueError(f'Dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms / 1000 * fs)
    tol_train = round(tol_train_ms / 1000 * fs)

    # Initialise test variables
    n_units = spike_trains_test.shape[1]

    #  If there are no spikes return zeros
    if (not np.any(spike_trains_ref)) | (not np.any(spike_trains_test)):
        pair_idx = np.arange(n_units)
        pair_lag = np.zeros((n_units))
        tp = np.zeros((n_units))  # Correctly detected spikes
        fp = np.zeros((n_units))  # Incorrectly detected spikes
        fn = np.zeros((n_units))  # Missed spikes
        return tp, fp, fn, pair_idx, pair_lag

    # Compute the TRUE POSITIVES, FALSE POSITIVES, and FALSE NEGATIVES between the sets
    #  ---------------------------------------------------------------------------------

    tp = np.zeros_like(spike_trains_test)  # Actual spikes
    fp = np.zeros_like(spike_trains_test)  #  Incorrect spikes
    fn = np.zeros_like(spike_trains_test)  # Missed spikes

    spikes_corr = np.zeros((n_units))
    pair_lag = np.zeros((n_units))
    pair_idx = [(unit, unit) for unit in range(n_units)]

    for unit in range(n_units):

        #  Align spike trains based on their correlation and spike tol
        # -----------------------------------------------------------
        #  Get trains
        train_ref = spike_trains_ref[:, unit]
        train_test = spike_trains_test[:, unit]
        # Apply spike tolerance
        train_ref = np.convolve(train_ref, np.ones(tol_spike), mode="same")
        train_test = np.convolve(train_test, np.ones(tol_spike), mode="same")
        # Compute correlation and lags
        curr_corr = signal.correlate(train_ref, train_test, mode="full")
        curr_lags = signal.correlation_lags(
            len(train_ref), len(train_test), mode="full"
        )
        # Apply train shift tolerance
        train_tol_idxs = np.nonzero(np.abs(curr_lags) == tol_train)[0]
        train_tol_mask = np.arange(train_tol_idxs[0], train_tol_idxs[-1] + 1).astype(
            int
        )
        curr_corr = curr_corr[train_tol_mask]
        curr_lags = curr_lags[train_tol_mask]
        # Identify optimal lag for alignment
        trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
        if not np.isscalar(trains_lag):
            # If there is more than one possible lag, choose the minimum
            trains_lag = np.amin(trains_lag)
        # Fill matrices
        spikes_corr[unit] = np.amax(curr_corr)
        pair_lag[unit] = trains_lag.astype(int)

        #  Compute metrics between the aligned spike trains
        # ------------------------------------------------
        # Align spike trains
        firings_ref = np.nonzero(spike_trains_ref[:, unit])[0] - pair_lag[unit].astype(
            int
        )
        firings_test = np.nonzero(spike_trains_test[:, unit])[0]
        # Pair firings
        for firing in firings_test:
            curr_firing_diff = np.abs(firings_ref - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A correctly detected firing (TP)
                tp[firing, unit] = 1
                firings_ref = np.delete(firings_ref, np.argmin(curr_firing_diff))
            else:
                # Incorrectly detected firings (FP)
                fp[firing, unit] = 1

        # A missed firing (FN)
        if np.any(firings_ref):
            fn[
                firings_ref[
                    np.logical_and(firings_ref >= 0, firings_ref < fn.shape[0])
                ],
                unit,
            ] = np.ones(len(firings_ref)).astype(int)

        assert np.sum(tp[:, unit] + fp[:, unit]) == np.sum(spike_trains_test[:, unit])
        assert np.sum(tp[:, unit] + fn[:, unit]) == np.sum(spike_trains_ref[:, unit])

    # Output
    return tp, fp, fn, pair_idx, pair_lag
