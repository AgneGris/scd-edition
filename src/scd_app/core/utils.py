"""Shared utility functions used across core modules."""

from __future__ import annotations
import numpy as np


def to_numpy(obj) -> np.ndarray:
    """Convert array-like objects (torch tensors, lists, None) to numpy arrays."""
    if obj is None:
        return np.array([])
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)
