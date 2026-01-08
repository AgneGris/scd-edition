"""
Analysis module - spike train analysis, quality metrics, and duplicate detection.
"""

from typing import List, Tuple, Set
import itertools

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


class SpikeTrain:
    """Utility class for spike train analysis."""
    
    @staticmethod
    def calculate_isi(timestamps: np.ndarray, fsamp: int) -> np.ndarray:
        """Calculate inter-spike intervals in seconds."""
        if len(timestamps) < 2:
            return np.array([])
        return np.diff(np.sort(timestamps)) / fsamp
    
    @staticmethod
    def calculate_discharge_rate(timestamps: np.ndarray, fsamp: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate instantaneous discharge rate.
        
        Returns
        -------
        time_points : np.ndarray
            Time points (midpoint between spikes) in seconds
        rates : np.ndarray
            Instantaneous rates in pulses per second
        """
        if len(timestamps) < 2:
            return np.array([]), np.array([])
        
        timestamps_sorted = np.sort(timestamps)
        isi = np.diff(timestamps_sorted) / fsamp
        rates = 1.0 / isi
        time_points = (timestamps_sorted[:-1] + timestamps_sorted[1:]) / (2 * fsamp)
        
        return time_points, rates
    
    @staticmethod
    def calculate_cv(timestamps: np.ndarray, fsamp: int = 1) -> float:
        """Calculate coefficient of variation of ISI."""
        if len(timestamps) < 2:
            return float('inf')
        
        isi = np.diff(np.sort(timestamps))
        if len(isi) == 0 or np.mean(isi) == 0:
            return float('inf')
        
        return np.std(isi) / np.mean(isi)


class QualityMetrics:
    """Quality metrics for motor unit decomposition."""
    
    @staticmethod
    def calculate_sil(source: np.ndarray, timestamps: np.ndarray, fsamp: int) -> float:
        """
        Calculate Silhouette metric for source quality.
        
        Parameters
        ----------
        source : np.ndarray
            Source signal
        timestamps : np.ndarray
            Spike timestamps
        fsamp : int
            Sampling frequency
        
        Returns
        -------
        float
            Silhouette value between -1 and 1
        """
        if len(timestamps) < 2:
            return 0.0
        
        # Find all peaks
        min_distance = int(0.005 * fsamp)
        peaks, _ = signal.find_peaks(source, distance=min_distance)
        
        if len(peaks) <= 1:
            return 0.0
        
        # Normalize source
        source_norm = source.copy()
        top_peaks = source[peaks][np.argsort(source[peaks])[-10:]]
        if len(top_peaks) > 0 and np.mean(top_peaks) > 0:
            source_norm = source_norm / np.mean(top_peaks)
        
        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=2, n_init=1, random_state=1337)
            labels = kmeans.fit_predict(source_norm[peaks].reshape(-1, 1))
            
            spikes_idx = np.argmax(kmeans.cluster_centers_)
            noise_idx = np.argmin(kmeans.cluster_centers_)
            
            spikes = peaks[labels == spikes_idx]
            spikes_centroid = kmeans.cluster_centers_[spikes_idx]
            noise_centroid = kmeans.cluster_centers_[noise_idx]
            
            # Calculate silhouette
            intra = np.sum((source_norm[spikes] - spikes_centroid) ** 2)
            inter = np.sum((source_norm[spikes] - noise_centroid) ** 2)
            
            sil = (inter - intra) / max(intra, inter) if max(intra, inter) > 0 else 0.0
            return float(sil)
        except Exception:
            return 0.0


class DuplicateDetector:
    """Detects duplicate and subset motor units."""
    
    def __init__(self, fsamp: int = 10240):
        self.fsamp = fsamp
    
    def find_duplicates(
        self,
        timestamps_list: List[np.ndarray],
        roa_threshold: float = 0.3,
        subset_threshold: float = 0.8
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Find duplicate and subset motor units.
        
        Returns
        -------
        groups : List[List[int]]
            Groups of related units
        units_to_flag : List[int]
            Indices of units recommended for removal
        """
        n_units = len(timestamps_list)
        if n_units < 2:
            return [], []
        
        # Build spike train matrix
        max_time = max(np.max(ts) if len(ts) > 0 else 0 for ts in timestamps_list)
        if max_time == 0:
            return [], []
            
        spike_trains = np.zeros((int(max_time) + 1, n_units))
        
        for idx, ts in enumerate(timestamps_list):
            if len(ts) > 0:
                valid_ts = ts[ts <= max_time].astype(int)
                spike_trains[valid_ts, idx] = 1
        
        # Calculate Rate of Agreement matrix
        roa_matrix = self._calculate_roa_matrix(spike_trains)
        
        # Calculate subset ratios
        subset_matrix1, subset_matrix2 = self._calculate_subset_matrices(timestamps_list)
        
        # Find pairs
        duplicate_pairs = []
        subset_pairs = []
        
        for i in range(n_units):
            for j in range(i + 1, n_units):
                if roa_matrix[i, j] > roa_threshold:
                    duplicate_pairs.append((i, j, roa_matrix[i, j], 'roa'))
                elif subset_matrix1[i, j] > subset_threshold or subset_matrix2[i, j] > subset_threshold:
                    subset_pairs.append((i, j, max(subset_matrix1[i, j], subset_matrix2[i, j]), 'subset'))
        
        all_pairs = duplicate_pairs + subset_pairs
        
        # Group connected units
        groups = self._group_connected_units(
            n_units, all_pairs, roa_matrix, subset_matrix1, subset_matrix2,
            roa_threshold, subset_threshold
        )
        
        # Determine which units to flag
        cv_values = [SpikeTrain.calculate_cv(ts) for ts in timestamps_list]
        spike_counts = [len(ts) for ts in timestamps_list]
        
        units_to_flag = []
        for group in groups:
            if len(group) < 2:
                continue
            
            # Check if subset relationship exists
            has_subset = any(
                subset_matrix1[i, j] > subset_threshold or subset_matrix2[i, j] > subset_threshold
                for i in group for j in group if i != j
            )
            
            if has_subset:
                # Keep unit with most spikes
                keep_idx = max(group, key=lambda idx: spike_counts[idx])
            else:
                # Keep unit with lowest CV
                keep_idx = min(group, key=lambda idx: cv_values[idx])
            
            units_to_flag.extend([idx for idx in group if idx != keep_idx])
        
        return groups, units_to_flag
    
    def _calculate_roa_matrix(
        self,
        spike_trains: np.ndarray,
        tol_spike_ms: float = 1.0,
        tol_train_ms: float = 40.0
    ) -> np.ndarray:
        """Calculate Rate of Agreement matrix."""
        n_units = spike_trains.shape[1]
        roa_matrix = np.ones((n_units, n_units))
        
        tol_spike = round(tol_spike_ms / 1000 * self.fsamp)
        tol_train = round(tol_train_ms / 1000 * self.fsamp)
        
        for i, j in itertools.combinations(range(n_units), 2):
            train_0 = spike_trains[:, i]
            train_1 = spike_trains[:, j]
            
            # Convolve for tolerance
            kernel = np.ones(tol_spike * 2)
            train_0_conv = np.convolve(train_0, kernel, mode="same")
            train_1_conv = np.convolve(train_1, kernel, mode="same")
            
            # Find optimal lag
            curr_corr = signal.correlate(train_0_conv, train_1_conv, mode="full")
            curr_lags = signal.correlation_lags(len(train_0_conv), len(train_1_conv), mode="full")
            
            trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
            if not np.isscalar(trains_lag):
                trains_lag = np.amin(trains_lag)
            
            if np.abs(trains_lag) > tol_train:
                trains_lag = 0
            
            # Count common firings
            firings_0 = np.nonzero(train_0)[0]
            firings_1 = np.nonzero(train_1)[0] + trains_lag
            
            firings_common = 0
            firings_1_remaining = list(firings_1)
            
            for firing in firings_0:
                if len(firings_1_remaining) == 0:
                    break
                diffs = np.abs(np.array(firings_1_remaining) - firing)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= tol_spike:
                    firings_common += 1
                    firings_1_remaining.pop(min_idx)
            
            total = firings_common + (len(firings_0) - firings_common) + len(firings_1_remaining)
            roa = firings_common / total if total > 0 else 0.0
            
            roa_matrix[i, j] = roa
            roa_matrix[j, i] = roa
        
        return roa_matrix
    
    def _calculate_subset_matrices(
        self,
        timestamps_list: List[np.ndarray],
        tolerance_ms: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate subset ratio matrices."""
        n_units = len(timestamps_list)
        subset_matrix1 = np.zeros((n_units, n_units))
        subset_matrix2 = np.zeros((n_units, n_units))
        
        tolerance_samples = int(tolerance_ms * self.fsamp / 1000)
        
        for i in range(n_units):
            for j in range(i + 1, n_units):
                times1, times2 = timestamps_list[i], timestamps_list[j]
                
                if len(times1) == 0 or len(times2) == 0:
                    continue
                
                # Count matches
                matches_1_in_2 = sum(
                    1 for t1 in times1
                    if np.any(np.abs(times2 - t1) <= tolerance_samples)
                )
                matches_2_in_1 = sum(
                    1 for t2 in times2
                    if np.any(np.abs(times1 - t2) <= tolerance_samples)
                )
                
                subset_matrix1[i, j] = matches_1_in_2 / len(times1)
                subset_matrix2[i, j] = matches_2_in_1 / len(times2)
                subset_matrix1[j, i] = subset_matrix2[i, j]
                subset_matrix2[j, i] = subset_matrix1[i, j]
        
        return subset_matrix1, subset_matrix2
    
    def _group_connected_units(
        self,
        n_units: int,
        all_pairs: List[Tuple],
        roa_matrix: np.ndarray,
        subset_matrix1: np.ndarray,
        subset_matrix2: np.ndarray,
        roa_threshold: float,
        subset_threshold: float
    ) -> List[List[int]]:
        """Group connected units using union-find."""
        groups = []
        processed: Set[int] = set()
        
        for i, j, score, dtype in all_pairs:
            if i in processed and j in processed:
                continue
            
            group = {i, j}
            added = True
            
            while added:
                added = False
                for k in range(n_units):
                    if k not in group:
                        for member in group:
                            is_connected = (
                                roa_matrix[min(k, member), max(k, member)] > roa_threshold or
                                subset_matrix1[k, member] > subset_threshold or
                                subset_matrix1[member, k] > subset_threshold
                            )
                            if is_connected:
                                group.add(k)
                                added = True
                                break
            
            groups.append(sorted(list(group)))
            processed.update(group)
        
        return groups