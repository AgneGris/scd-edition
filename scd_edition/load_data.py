"""
Data loading functions - MODIFY THIS FILE FOR YOUR DATA FORMAT

Two simple functions:
- load_emg: returns EMG array [channels x samples]
- load_decomposition: returns dict with 'timestamps' and 'source'
"""

import pickle as pkl
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def load_emg(filepath: str) -> np.ndarray:
    """
    Load EMG data. Modify this for your file format.
    
    Returns
    -------
    emg : np.ndarray
        Shape [channels x samples]
    """
    filepath = Path(filepath)
    
    # === MODIFY BELOW FOR YOUR DATA ===
    
    if filepath.suffix == ".mat":
        mat = sio.loadmat(str(filepath))
        # Change 'emg' to match your variable name
        emg = mat["emg"]
    
    elif filepath.suffix == ".npy":
        emg = np.load(filepath)
    
    else:
        raise ValueError(f"Unknown format: {filepath.suffix}")
    
    # === END MODIFY ===
    
    # Ensure [channels x samples]
    emg = np.array(emg).squeeze()
    if emg.shape[0] > emg.shape[1]:
        emg = emg.T
    
    return emg.astype(np.float64)


def load_decomposition(filepath: str) -> dict:
    """
    Load decomposition results.
    
    Returns
    -------
    dict with:
        - 'timestamps': List[np.ndarray] - spike indices for each unit
        - 'source': np.ndarray - source signals [n_units x samples]
    """
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    
    # Convert torch tensors to numpy
    timestamps = [
        ts.cpu().numpy().astype(int) if torch.is_tensor(ts) else np.array(ts).astype(int)
        for ts in data["timestamps"]
    ]
    
    source = data.get("source", None)
    
    return {
        "timestamps": timestamps,
        "source": source,
        "raw": data  # Keep original for saving
    }


def save_decomposition(filepath: str, timestamps: list, source: np.ndarray, raw_data: dict):
    """Save edited decomposition."""
    data = raw_data.copy()
    data["timestamps"] = [torch.from_numpy(ts.astype(int)) for ts in timestamps]
    data["source"] = source
    
    with open(filepath, "wb") as f:
        pkl.dump(data, f)