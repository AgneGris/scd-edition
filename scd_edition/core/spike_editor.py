"""
Spike editing module - handles adding, deleting, and managing spikes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


@dataclass
class EditState:
    """Stores the state for undo functionality."""
    timestamps: List[np.ndarray]
    sources: Optional[np.ndarray] = None


class SpikeEditor:
    """Handles spike editing operations with undo support."""
    
    def __init__(self, max_undo: int = 10):
        self.max_undo = max_undo
        self.undo_stack: deque = deque(maxlen=max_undo)
        
        # Preview state
        self.preview_active: bool = False
        self.preview_location: Optional[int] = None
        self.preview_action: Optional[str] = None  # "add" or "delete"
    
    def save_state(self, timestamps: List[np.ndarray], sources: Optional[np.ndarray] = None):
        """Save current state to undo stack."""
        state = EditState(
            timestamps=[ts.copy().astype(int) for ts in timestamps],
            sources=sources.copy() if sources is not None else None
        )
        self.undo_stack.append(state)
    
    def undo(self) -> Optional[EditState]:
        """Pop and return the last saved state."""
        if self.undo_stack:
            return self.undo_stack.pop()
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def find_nearest_peak(
        self,
        source: np.ndarray,
        click_sample: int,
        view_range: Tuple[int, int],
        y_range: Tuple[float, float],
        fsamp: int,
        existing_timestamps: np.ndarray
    ) -> Optional[int]:
        """
        Find the nearest peak to a click position within the visible range.
        
        Returns None if no valid peak found or spike already exists.
        """
        view_start, view_end = view_range
        view_start = max(0, view_start)
        view_end = min(len(source), view_end)
        
        if click_sample < view_start or click_sample > view_end:
            return None
        
        # Extract visible segment
        visible_segment = source[view_start:view_end]
        if len(visible_segment) == 0:
            return None
        
        # Find peaks in visible range (both X and Y)
        min_distance = int(0.005 * fsamp)  # 5ms minimum
        peaks, _ = signal.find_peaks(
            visible_segment, 
            distance=min_distance,
            height=(y_range[0], y_range[1])
        )
        
        if len(peaks) == 0:
            return None
        
        # Convert to absolute positions
        peak_samples = peaks + view_start
        
        # Find closest peak to click
        distances = np.abs(peak_samples - click_sample)
        nearest_peak = peak_samples[np.argmin(distances)]
        
        # Check if spike already exists
        tolerance = int(0.005 * fsamp)
        if len(existing_timestamps) > 0 and np.any(np.abs(existing_timestamps - nearest_peak) < tolerance):
            return None
        
        return int(nearest_peak)
    
    def find_nearest_spike(
        self,
        timestamps: np.ndarray,
        click_sample: int,
        view_range: Tuple[int, int]
    ) -> Optional[int]:
        """
        Find the nearest existing spike to a click position.
        
        Returns the spike sample position or None if no visible spikes.
        """
        view_start, view_end = view_range
        
        # Filter to visible spikes
        visible_mask = (timestamps >= view_start) & (timestamps <= view_end)
        visible_spikes = timestamps[visible_mask]
        
        if len(visible_spikes) == 0:
            return None
        
        # Find closest
        distances = np.abs(visible_spikes - click_sample)
        return int(visible_spikes[np.argmin(distances)])
    
    def add_spike(self, timestamps: np.ndarray, new_spike: int) -> np.ndarray:
        """Add a spike and return sorted timestamps."""
        return np.sort(np.concatenate([timestamps, [new_spike]])).astype(int)
    
    def delete_spike(self, timestamps: np.ndarray, spike_to_delete: int) -> np.ndarray:
        """Delete a spike and return updated timestamps."""
        idx = np.where(timestamps == spike_to_delete)[0]
        if len(idx) > 0:
            return np.delete(timestamps, idx[0]).astype(int)
        return timestamps
    
    def add_spikes_in_roi(
        self,
        source: np.ndarray,
        timestamps: np.ndarray,
        roi_bounds: Tuple[float, float, float, float],
        fsamp: int
    ) -> np.ndarray:
        """
        Add all peaks within ROI bounds.
        
        Parameters
        ----------
        roi_bounds : Tuple[x_left, x_right, y_bottom, y_top]
        """
        x_left, x_right, y_bottom, y_top = roi_bounds
        sample_start = int(x_left * fsamp)
        sample_end = int(x_right * fsamp)
        
        # Create segment mask
        segment = np.zeros_like(source)
        sample_end = min(sample_end, len(source))
        sample_start = max(0, sample_start)
        segment[sample_start:sample_end] = source[sample_start:sample_end]
        
        # Find peaks in ROI
        min_distance = int(0.005 * fsamp)
        peaks, _ = signal.find_peaks(segment, height=y_bottom, distance=min_distance)
        peaks_in_roi = peaks[source[peaks] <= y_top]
        
        if len(peaks_in_roi) > 0:
            return np.unique(np.concatenate([timestamps, peaks_in_roi])).astype(int)
        
        return timestamps
    
    def delete_spikes_in_roi(
        self,
        source: np.ndarray,
        timestamps: np.ndarray,
        roi_bounds: Tuple[float, float, float, float],
        fsamp: int
    ) -> np.ndarray:
        """Delete all spikes within ROI bounds."""
        x_left, x_right, y_bottom, y_top = roi_bounds
        sample_start = int(x_left * fsamp)
        sample_end = int(x_right * fsamp)
        
        spikes_to_keep = []
        for ts in timestamps:
            ts_int = int(ts)
            in_x_range = sample_start <= ts_int < sample_end
            
            if in_x_range and ts_int < len(source):
                amplitude = source[ts_int]
                in_y_range = y_bottom <= amplitude <= y_top
                if in_y_range:
                    continue  # Skip (delete) this spike
            
            spikes_to_keep.append(ts)
        
        if len(spikes_to_keep) == 0:
            return np.array([], dtype=int)
        return np.array(spikes_to_keep).astype(int)
    
    def recalculate_filter(
        self,
        whitened_emg: np.ndarray,
        timestamps: np.ndarray,
        fsamp: int,
        min_spikes: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recalculate filter and detect new spikes based on current timestamps.
        
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
            if ts < whitened_emg.shape[1]:
                mu_filter += whitened_emg[:, int(ts)]
        
        # Apply filter to get new source
        new_source = np.dot(mu_filter, whitened_emg)
        
        # Normalize
        if np.std(new_source) > 0:
            new_source = new_source / np.std(new_source)
        
        # Find peaks with 20ms distance constraint
        peaks, _ = signal.find_peaks(
            np.squeeze(new_source),
            distance=int(np.round(fsamp * 0.02) + 1)
        )
        
        if len(peaks) > 10:
            # Normalize by mean of top 10 peaks
            peak_values = new_source[peaks]
            top10_peaks = np.sort(peak_values)[-10:]
            normalization_factor = np.mean(top10_peaks)
            source_normalized = new_source / normalization_factor
            
            # KMeans clustering to separate spikes from noise
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
            kmeans.fit(source_normalized[peaks].reshape(-1, 1))
            
            # Select cluster with higher centroid
            spikes_idx = np.argmax(kmeans.cluster_centers_)
            new_timestamps = peaks[kmeans.labels_ == spikes_idx]
        else:
            new_timestamps = peaks
        
        return np.expand_dims(new_source, 1), new_timestamps.astype(int)
    
    # Preview management
    def start_preview(self, location: int, action: str):
        """Start a preview for add/delete action."""
        self.preview_active = True
        self.preview_location = location
        self.preview_action = action
    
    def clear_preview(self):
        """Clear the current preview."""
        self.preview_active = False
        self.preview_location = None
        self.preview_action = None