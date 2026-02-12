"""
Analysis module for SCD Suite.
Quality metrics, duplicate detection (within and across ports/files), and validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import itertools

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


@dataclass
class QualityReport:
    """Quality metrics for a motor unit."""
    mu_id: int
    port_name: str
    n_spikes: int
    sil: float
    cov: float
    mean_fr_hz: float
    min_isi_ms: float
    
    # Flags
    low_quality: bool = False
    low_spike_count: bool = False
    abnormal_fr: bool = False
    high_cov: bool = False
    
    @property
    def is_valid(self) -> bool:
        return not (self.low_quality or self.low_spike_count or self.abnormal_fr)


@dataclass
class DuplicatePair:
    """Represents a pair of potentially duplicate motor units."""
    mu1_id: int
    mu1_port: str
    mu2_id: int
    mu2_port: str
    roa: float              # Rate of Agreement
    subset_ratio: float     # How much of smaller is in larger
    relationship: str       # "duplicate", "subset", "related"
    
    @property
    def cross_port(self) -> bool:
        return self.mu1_port != self.mu2_port


class QualityAnalyzer:
    """Computes quality metrics for motor units."""
    
    def __init__(self, fsamp: int = 10240):
        self.fsamp = fsamp
    
    def compute_sil(self, source: np.ndarray, timestamps: np.ndarray) -> float:
        """
        Compute Silhouette (SIL) metric for source quality.
        
        Higher SIL indicates better separation between spikes and noise.
        Range: -1 to 1, typically >0.85 is good.
        """
        if len(timestamps) < 2:
            return 0.0
        
        source = np.asarray(source).squeeze()
        
        # Find all peaks
        min_distance = int(0.005 * self.fsamp)
        peaks, _ = signal.find_peaks(source, distance=min_distance)
        
        if len(peaks) <= 1:
            return 0.0
        
        # Normalize source by top peaks
        top_indices = np.argsort(source[peaks])[-10:]
        top_values = source[peaks][top_indices]
        if len(top_values) > 0 and np.mean(top_values) > 0:
            source_norm = source / np.mean(top_values)
        else:
            source_norm = source.copy()
        
        try:
            # KMeans to separate spikes from noise
            kmeans = KMeans(n_clusters=2, n_init=1, random_state=42)
            labels = kmeans.fit_predict(source_norm[peaks].reshape(-1, 1))
            
            spike_cluster = np.argmax(kmeans.cluster_centers_)
            noise_cluster = np.argmin(kmeans.cluster_centers_)
            
            spike_peaks = peaks[labels == spike_cluster]
            spike_centroid = kmeans.cluster_centers_[spike_cluster, 0]
            noise_centroid = kmeans.cluster_centers_[noise_cluster, 0]
            
            if len(spike_peaks) == 0:
                return 0.0
            
            # Compute silhouette
            intra = np.sum((source_norm[spike_peaks] - spike_centroid) ** 2)
            inter = np.sum((source_norm[spike_peaks] - noise_centroid) ** 2)
            
            denominator = max(intra, inter)
            if denominator == 0:
                return 0.0
            
            return float((inter - intra) / denominator)
        
        except Exception:
            return 0.0
    
    def compute_cov(self, timestamps: np.ndarray) -> float:
        """
        Compute Coefficient of Variation (CoV) of inter-spike intervals.
        
        Lower CoV indicates more regular firing. Typical range: 0.1-0.5.
        """
        if len(timestamps) < 2:
            return float('inf')
        
        isi = np.diff(np.sort(timestamps))
        if len(isi) == 0 or np.mean(isi) == 0:
            return float('inf')
        
        return float(np.std(isi) / np.mean(isi))
    
    def compute_firing_rate(self, timestamps: np.ndarray, duration_samples: int = None) -> float:
        """Compute mean firing rate in Hz."""
        if len(timestamps) < 2:
            return 0.0
        
        timestamps = np.sort(timestamps)
        
        if duration_samples is None:
            duration_samples = timestamps[-1] - timestamps[0]
        
        if duration_samples <= 0:
            return 0.0
        
        duration_sec = duration_samples / self.fsamp
        return len(timestamps) / duration_sec
    
    def compute_min_isi(self, timestamps: np.ndarray) -> float:
        """Compute minimum ISI in milliseconds."""
        if len(timestamps) < 2:
            return float('inf')
        
        isi_samples = np.diff(np.sort(timestamps))
        return float(np.min(isi_samples) / self.fsamp * 1000)
    
    def analyze_motor_unit(
        self,
        mu_id: int,
        port_name: str,
        timestamps: np.ndarray,
        source: np.ndarray,
        duration_samples: int = None,
        sil_threshold: float = 0.85,
        min_spikes: int = 10,
        min_fr_hz: float = 2.0,
        max_fr_hz: float = 100.0,
        max_cov: float = 0.8,
    ) -> QualityReport:
        """Generate complete quality report for a motor unit."""
        
        n_spikes = len(timestamps)
        sil = self.compute_sil(source, timestamps)
        cov = self.compute_cov(timestamps)
        mean_fr = self.compute_firing_rate(timestamps, duration_samples)
        min_isi = self.compute_min_isi(timestamps)
        
        report = QualityReport(
            mu_id=mu_id,
            port_name=port_name,
            n_spikes=n_spikes,
            sil=sil,
            cov=cov,
            mean_fr_hz=mean_fr,
            min_isi_ms=min_isi,
            low_quality=sil < sil_threshold,
            low_spike_count=n_spikes < min_spikes,
            abnormal_fr=(mean_fr < min_fr_hz or mean_fr > max_fr_hz) if mean_fr > 0 else True,
            high_cov=cov > max_cov if cov < float('inf') else True,
        )
        
        return report


class DuplicateDetector:
    """
    Detects duplicate and subset motor units within and across ports/files.
    """
    
    def __init__(self, fsamp: int = 10240):
        self.fsamp = fsamp
    
    def compute_roa(
        self,
        timestamps1: np.ndarray,
        timestamps2: np.ndarray,
        tolerance_ms: float = 1.0,
        max_lag_ms: float = 40.0,
    ) -> Tuple[float, int]:
        """
        Compute Rate of Agreement between two spike trains.
        
        Returns
        -------
        roa : float
            Rate of agreement (0-1)
        optimal_lag : int
            Optimal lag in samples
        """
        if len(timestamps1) == 0 or len(timestamps2) == 0:
            return 0.0, 0
        
        tolerance = int(tolerance_ms * self.fsamp / 1000)
        max_lag = int(max_lag_ms * self.fsamp / 1000)
        
        # Build spike trains
        max_time = max(np.max(timestamps1), np.max(timestamps2))
        train1 = np.zeros(int(max_time) + 1)
        train2 = np.zeros(int(max_time) + 1)
        
        train1[timestamps1.astype(int)] = 1
        train2[timestamps2.astype(int)] = 1
        
        # Convolve for tolerance
        kernel = np.ones(tolerance * 2 + 1)
        train1_conv = np.convolve(train1, kernel, mode='same')
        train2_conv = np.convolve(train2, kernel, mode='same')
        
        # Find optimal lag via cross-correlation
        corr = signal.correlate(train1_conv, train2_conv, mode='full')
        lags = signal.correlation_lags(len(train1_conv), len(train2_conv), mode='full')
        
        # Restrict to max_lag
        valid_mask = np.abs(lags) <= max_lag
        if not np.any(valid_mask):
            optimal_lag = 0
        else:
            valid_corr = corr[valid_mask]
            valid_lags = lags[valid_mask]
            optimal_lag = valid_lags[np.argmax(valid_corr)]
        
        # Count common firings with optimal lag
        ts2_shifted = timestamps2 + optimal_lag
        
        common = 0
        ts2_remaining = list(ts2_shifted)
        
        for t1 in timestamps1:
            if len(ts2_remaining) == 0:
                break
            diffs = np.abs(np.array(ts2_remaining) - t1)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tolerance:
                common += 1
                ts2_remaining.pop(min_idx)
        
        # ROA = common / total unique firings
        total = common + (len(timestamps1) - common) + len(ts2_remaining)
        roa = common / total if total > 0 else 0.0
        
        return float(roa), int(optimal_lag)
    
    def compute_subset_ratio(
        self,
        timestamps1: np.ndarray,
        timestamps2: np.ndarray,
        tolerance_ms: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute what fraction of each train appears in the other.
        
        Returns
        -------
        ratio_1_in_2 : float
            Fraction of timestamps1 that appear in timestamps2
        ratio_2_in_1 : float
            Fraction of timestamps2 that appear in timestamps1
        """
        if len(timestamps1) == 0 or len(timestamps2) == 0:
            return 0.0, 0.0
        
        tolerance = int(tolerance_ms * self.fsamp / 1000)
        
        # How many of ts1 are in ts2
        count_1_in_2 = sum(
            1 for t1 in timestamps1
            if np.any(np.abs(timestamps2 - t1) <= tolerance)
        )
        
        # How many of ts2 are in ts1
        count_2_in_1 = sum(
            1 for t2 in timestamps2
            if np.any(np.abs(timestamps1 - t2) <= tolerance)
        )
        
        return count_1_in_2 / len(timestamps1), count_2_in_1 / len(timestamps2)
    
    def find_duplicates_within_port(
        self,
        timestamps_list: List[np.ndarray],
        port_name: str,
        roa_threshold: float = 0.3,
        subset_threshold: float = 0.8,
    ) -> List[DuplicatePair]:
        """Find duplicates within a single port."""
        
        n_units = len(timestamps_list)
        pairs = []
        
        for i, j in itertools.combinations(range(n_units), 2):
            ts1, ts2 = timestamps_list[i], timestamps_list[j]
            
            roa, _ = self.compute_roa(ts1, ts2)
            ratio_1_in_2, ratio_2_in_1 = self.compute_subset_ratio(ts1, ts2)
            
            if roa > roa_threshold:
                pairs.append(DuplicatePair(
                    mu1_id=i, mu1_port=port_name,
                    mu2_id=j, mu2_port=port_name,
                    roa=roa,
                    subset_ratio=max(ratio_1_in_2, ratio_2_in_1),
                    relationship="duplicate",
                ))
            elif ratio_1_in_2 > subset_threshold or ratio_2_in_1 > subset_threshold:
                pairs.append(DuplicatePair(
                    mu1_id=i, mu1_port=port_name,
                    mu2_id=j, mu2_port=port_name,
                    roa=roa,
                    subset_ratio=max(ratio_1_in_2, ratio_2_in_1),
                    relationship="subset",
                ))
        
        return pairs
    
    def find_duplicates_across_ports(
        self,
        port_data: Dict[str, List[np.ndarray]],
        roa_threshold: float = 0.3,
        subset_threshold: float = 0.8,
    ) -> List[DuplicatePair]:
        """
        Find duplicates across different ports.
        
        Parameters
        ----------
        port_data : Dict[str, List[np.ndarray]]
            Dictionary mapping port names to lists of timestamp arrays
        
        Returns
        -------
        pairs : List[DuplicatePair]
            List of duplicate pairs found across ports
        """
        pairs = []
        port_names = list(port_data.keys())
        
        for p1, p2 in itertools.combinations(port_names, 2):
            ts_list1 = port_data[p1]
            ts_list2 = port_data[p2]
            
            for i, ts1 in enumerate(ts_list1):
                for j, ts2 in enumerate(ts_list2):
                    roa, _ = self.compute_roa(ts1, ts2)
                    ratio_1_in_2, ratio_2_in_1 = self.compute_subset_ratio(ts1, ts2)
                    
                    if roa > roa_threshold:
                        pairs.append(DuplicatePair(
                            mu1_id=i, mu1_port=p1,
                            mu2_id=j, mu2_port=p2,
                            roa=roa,
                            subset_ratio=max(ratio_1_in_2, ratio_2_in_1),
                            relationship="duplicate",
                        ))
                    elif ratio_1_in_2 > subset_threshold or ratio_2_in_1 > subset_threshold:
                        pairs.append(DuplicatePair(
                            mu1_id=i, mu1_port=p1,
                            mu2_id=j, mu2_port=p2,
                            roa=roa,
                            subset_ratio=max(ratio_1_in_2, ratio_2_in_1),
                            relationship="subset",
                        ))
        
        return pairs
    
    def find_all_duplicates(
        self,
        port_data: Dict[str, List[np.ndarray]],
        roa_threshold: float = 0.3,
        subset_threshold: float = 0.8,
    ) -> Tuple[List[DuplicatePair], List[DuplicatePair]]:
        """
        Find all duplicates both within and across ports.
        
        Returns
        -------
        within_pairs : List[DuplicatePair]
            Duplicates found within individual ports
        across_pairs : List[DuplicatePair]
            Duplicates found across different ports
        """
        within_pairs = []
        for port_name, ts_list in port_data.items():
            within_pairs.extend(
                self.find_duplicates_within_port(ts_list, port_name, roa_threshold, subset_threshold)
            )
        
        across_pairs = self.find_duplicates_across_ports(port_data, roa_threshold, subset_threshold)
        
        return within_pairs, across_pairs
    
    def recommend_removals(
        self,
        pairs: List[DuplicatePair],
        timestamps_dict: Dict[Tuple[str, int], np.ndarray],
        quality_reports: Dict[Tuple[str, int], QualityReport] = None,
    ) -> List[Tuple[str, int]]:
        """
        Recommend which motor units to remove based on duplicates.
        
        Strategy:
        - For subsets: keep the unit with more spikes
        - For duplicates: keep the unit with better quality (lower CoV)
        
        Returns list of (port_name, mu_id) tuples to remove.
        """
        to_remove: Set[Tuple[str, int]] = set()
        
        for pair in pairs:
            key1 = (pair.mu1_port, pair.mu1_id)
            key2 = (pair.mu2_port, pair.mu2_id)
            
            # Skip if already marked for removal
            if key1 in to_remove or key2 in to_remove:
                continue
            
            ts1 = timestamps_dict.get(key1, np.array([]))
            ts2 = timestamps_dict.get(key2, np.array([]))
            
            if pair.relationship == "subset":
                # Keep unit with more spikes
                if len(ts1) < len(ts2):
                    to_remove.add(key1)
                else:
                    to_remove.add(key2)
            else:
                # Keep unit with better quality
                if quality_reports:
                    q1 = quality_reports.get(key1)
                    q2 = quality_reports.get(key2)
                    if q1 and q2:
                        if q1.cov < q2.cov:
                            to_remove.add(key2)
                        else:
                            to_remove.add(key1)
                        continue
                
                # Fallback: keep unit with more spikes
                if len(ts1) >= len(ts2):
                    to_remove.add(key2)
                else:
                    to_remove.add(key1)
        
        return list(to_remove)


class FilterRecalculator:
    """Recalculates motor unit filters from edited spike times."""
    
    def __init__(self, fsamp: int = 10240):
        self.fsamp = fsamp
    
    def recalculate(
        self,
        whitened_emg: np.ndarray,
        timestamps: np.ndarray,
        min_spikes: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recalculate filter and detect new spikes.
        
        Parameters
        ----------
        whitened_emg : np.ndarray
            Whitened EMG data [extended_channels x samples]
        timestamps : np.ndarray
            Current spike timestamps
        min_spikes : int
            Minimum spikes required for recalculation
        
        Returns
        -------
        new_source : np.ndarray
            Recalculated source signal
        new_timestamps : np.ndarray
            Refined spike timestamps
        """
        if len(timestamps) < min_spikes:
            raise ValueError(f"Need at least {min_spikes} spikes for recalculation")
        
        # Calculate filter by summing whitened EMG at spike locations
        mu_filter = np.zeros(whitened_emg.shape[0])
        for ts in timestamps:
            ts = int(ts)
            if ts < whitened_emg.shape[1]:
                mu_filter += whitened_emg[:, ts]
        
        # Apply filter
        new_source = np.dot(mu_filter, whitened_emg)
        
        # Normalize
        if np.std(new_source) > 0:
            new_source = new_source / np.std(new_source)
        
        # Find peaks (20ms minimum distance)
        min_distance = int(0.02 * self.fsamp)
        peaks, _ = signal.find_peaks(new_source, distance=min_distance)
        
        if len(peaks) > 10:
            # Normalize by top 10 peaks
            peak_values = new_source[peaks]
            top10 = np.sort(peak_values)[-10:]
            norm_factor = np.mean(top10)
            source_norm = new_source / norm_factor
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=2, n_init=1, random_state=42)
            labels = kmeans.fit_predict(source_norm[peaks].reshape(-1, 1))
            
            spike_cluster = np.argmax(kmeans.cluster_centers_)
            new_timestamps = peaks[labels == spike_cluster]
        else:
            new_timestamps = peaks
        
        return new_source, new_timestamps.astype(np.int64)
