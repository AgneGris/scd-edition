"""
Universal EMG data loader.

Reads any EMG file given a YAML layout descriptor.
One function, one path — no format-specific branches scattered across the codebase.

Usage:
    from core.data_loader import load_layout, load_field

    layout = load_layout("presets/loader_dc.yaml")
    emg  = load_field(Path("recording.h5"), layout, "emg")    # → torch.Tensor (samples, channels)
    force = load_field(Path("recording.h5"), layout, "force")  # → torch.Tensor (samples, channels)
    ts   = load_field(Path("recording.h5"), layout, "timestamps")  # → torch.Tensor (samples,)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
import numpy as np
import torch


def load_layout(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML layout descriptor."""
    with open(yaml_path, 'r') as f:
        layout = yaml.safe_load(f)
    
    if "name" not in layout or "format" not in layout or "fields" not in layout:
        raise ValueError(
            f"Invalid layout file: must contain 'name', 'format', and 'fields' keys. "
            f"Got: {list(layout.keys())}"
        )
    return layout


def load_field(
    file_path: Path,
    layout: Dict[str, Any],
    field: str,
) -> torch.Tensor:
    """
    Load a single field (emg, force, timestamps) from a data file.

    Parameters
    ----------
    file_path : Path
        Path to the data file (.mat, .h5, .hdf5, .npy)
    layout : dict
        Parsed YAML layout descriptor (from load_layout)
    field : str
        Which field to load: "emg", "force", "timestamps", etc.

    Returns
    -------
    torch.Tensor
        For 2D fields: (samples, channels) — always this orientation.
        For 1D fields: (samples,)
    """
    file_path = Path(file_path)
    fmt = layout["format"]

    field_spec = layout["fields"].get(field)
    if field_spec is None:
        raise KeyError(f"Field '{field}' not defined in layout '{layout['name']}'")

    # Read raw array from file
    raw = _read_array(file_path, fmt, field_spec)

    # Slice channels if specified
    raw = _slice_channels(raw, field_spec.get("channels"))

    # Fix orientation → always (samples, channels) for 2D
    if raw.ndim == 2:
        raw = _fix_orientation(raw, field_spec.get("orientation", "auto"))

    return torch.from_numpy(raw).to(dtype=torch.float32)


def load_all_fields(
    file_path: Path,
    layout: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Load all fields defined in the layout. Returns dict of field_name → tensor."""
    result = {}
    for field_name in layout["fields"]:
        try:
            result[field_name] = load_field(file_path, layout, field_name)
        except (KeyError, ValueError) as e:
            print(f"  Warning: Could not load field '{field_name}': {e}")
    return result


# =============================================================================
# Internal: file reading
# =============================================================================

def _read_array(file_path: Path, fmt: str, field_spec: Dict) -> np.ndarray:
    """Read a raw numpy array from file using the field spec."""
    primary_path = field_spec["path"]
    fallbacks = field_spec.get("fallback_keys", [])

    if fmt == "h5":
        return _read_h5(file_path, primary_path, fallbacks)
    elif fmt == "mat":
        return _read_mat(file_path, primary_path, fallbacks)
    elif fmt == "npy":
        return np.load(str(file_path))
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")


def _read_h5(file_path: Path, dataset_path: str, fallbacks: List[str]) -> np.ndarray:
    """Read from HDF5 file."""
    import h5py

    with h5py.File(file_path, 'r') as f:
        # Try primary path
        if dataset_path in f:
            return np.array(f[dataset_path])

        # Try fallbacks (as top-level or nested paths)
        for key in fallbacks:
            if key in f:
                return np.array(f[key])

        available = []
        f.visit(lambda name: available.append(name))
        raise KeyError(
            f"Dataset '{dataset_path}' not found in {file_path.name}. "
            f"Available: {available[:20]}"
        )


def _read_mat(file_path: Path, var_name: str, fallbacks: List[str]) -> np.ndarray:
    """Read from .mat file (v5/v7 via scipy, v7.3 via h5py)."""
    import scipy.io as sio

    try:
        mat = sio.loadmat(str(file_path))
    except NotImplementedError:
        # v7.3 .mat files are HDF5
        return _read_h5(file_path, var_name, fallbacks)

    # Try primary key
    if var_name in mat:
        return np.asarray(mat[var_name])

    # Try fallbacks
    for key in fallbacks:
        if key in mat:
            return np.asarray(mat[key])

    # List available keys (skip MATLAB metadata)
    available = [k for k in mat.keys() if not k.startswith('__')]
    raise KeyError(
        f"Variable '{var_name}' not found in {file_path.name}. "
        f"Available: {available}"
    )


# =============================================================================
# Internal: channel slicing and orientation
# =============================================================================

def _slice_channels(data: np.ndarray, channels_spec) -> np.ndarray:
    """
    Slice channels from data.

    channels_spec can be:
        null/None  → return all
        [start, end]  → slice rows start:end (assumes channels_first before orientation fix)
        [0, 1, 5, 10] → pick specific indices (len > 2)
    """
    if channels_spec is None:
        return data

    if data.ndim != 2:
        return data

    ch = list(channels_spec)

    if len(ch) == 2 and ch[1] > ch[0]:
        # Could be a range [start, end] or two specific indices
        # Heuristic: if end > number of rows when interpreted as index, treat as range
        # Simple rule: [0, 64] is a range; [3, 7] with only 8 channels could be either
        # We use range interpretation for [start, end] pairs
        return data[ch[0]:ch[1], :]
    else:
        # Explicit list of indices
        return data[ch, :]


def _fix_orientation(data: np.ndarray, orientation: str) -> np.ndarray:
    """
    Ensure 2D data is (samples, channels).

    orientation:
        "channels_first"  → data is (channels, samples), transpose it
        "samples_first"   → data is already (samples, channels)
        "auto"            → larger dim is samples
    """
    if orientation == "channels_first":
        return data.T
    elif orientation == "samples_first":
        return data
    elif orientation == "auto":
        if data.shape[1] > data.shape[0]:
            return data.T
        return data
    else:
        raise ValueError(f"Unknown orientation: '{orientation}'. Use 'channels_first', 'samples_first', or 'auto'.")