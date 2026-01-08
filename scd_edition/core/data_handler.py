"""
Data handler - stores data for the GUI.
Uses load_data.py for file loading and scd.processing.preprocess for signal processing.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from scd_edition.load_data import load_emg, load_decomposition, save_decomposition

# Import preprocessing functions from scd package
from scd.processing.preprocess import high_pass_filter, low_pass_filter, extend, whiten


class DataHandler:
    """Stores and manages data for the GUI."""
    
    def __init__(self, fsamp: int = 10240):
        self.fsamp = fsamp
        self.emg_data_filtered: Optional[np.ndarray] = None
        self.timestamps: List[np.ndarray] = []
        self.sources: Optional[np.ndarray] = None
        self.whitened_emg: Optional[np.ndarray] = None
        self.filename: str = ""
        self._raw_data: dict = {}
    
    def load(self, decomp_path: Path, emg_path: Path):
        """Load decomposition and EMG files."""
        # Load decomposition using load_data.py
        data = load_decomposition(str(decomp_path))
        self.timestamps = data["timestamps"]
        self.sources = data["source"]
        self._raw_data = data["raw"]
        self.filename = decomp_path.stem
        
        # Load EMG using load_data.py - returns [channels x samples]
        raw_emg = load_emg(str(emg_path))
        
        # Filter using scd functions
        self.emg_data_filtered = self._bandpass_filter(raw_emg)
        
        # Prepare whitened EMG for recalculation
        self._prepare_whitened()
    
    def save(self, filepath: Path):
        """Save edited data."""
        save_decomposition(str(filepath), self.timestamps, self.sources, self._raw_data)
    
    @property
    def n_units(self) -> int:
        return len(self.timestamps)
    
    def get_unit_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get source and timestamps for a unit."""
        return self.sources[idx].squeeze(), self.timestamps[idx]
    
    def _bandpass_filter(self, emg: np.ndarray, lowcut: int = 10, highcut: int = 4400) -> np.ndarray:
        """
        Apply bandpass filter using scd's high_pass and low_pass filters.
        
        Input: emg [channels x samples]
        Output: filtered [channels x samples]
        """
        # scd functions expect [samples x channels] torch tensor
        emg_torch = torch.from_numpy(emg.T).float()  # [samples x channels]
        
        # Apply high-pass then low-pass
        filtered = high_pass_filter(emg_torch, self.fsamp, lowcut)
        filtered = low_pass_filter(filtered, self.fsamp, highcut)
        
        # Convert back to numpy [channels x samples]
        return filtered.numpy().T
    
    def _prepare_whitened(self):
        """Prepare whitened EMG for filter recalculation using scd functions."""
        if self.emg_data_filtered is None:
            return
        try:
            n_ch = self.emg_data_filtered.shape[0]
            ext_factor = int(np.ceil(1000 / n_ch))
            
            # scd functions expect [samples x channels] torch tensor
            emg_torch = torch.from_numpy(self.emg_data_filtered.T).float()
            
            # Extend and whiten using scd functions
            extended = extend(emg_torch, ext_factor)
            whitened = whiten(extended, method="zca")
            
            # Convert back to numpy [channels x samples] for GUI use
            self.whitened_emg = whitened.numpy().T
            
        except Exception as e:
            print(f"Whitening failed: {e}")
            self.whitened_emg = None