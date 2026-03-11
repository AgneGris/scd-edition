"""
Data handler for SCD-Edition.
Manages EMG data, decomposition results, and motor unit editing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pickle
import json

import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import h5py

from scd_app.core.mu_model import MotorUnit, EditAction


@dataclass
class PortData:
    """Data for a single port."""
    name: str
    emg_raw: Optional[np.ndarray] = None       # [channels x samples]
    emg_filtered: Optional[np.ndarray] = None  # [channels x samples]
    emg_whitened: Optional[np.ndarray] = None  # [extended_channels x samples]
    motor_units: List[MotorUnit] = field(default_factory=list)
    
    @property
    def n_units(self) -> int:
        return len(self.motor_units)
    
    @property
    def n_channels(self) -> int:
        if self.emg_filtered is not None:
            return self.emg_filtered.shape[0]
        return 0


class DataHandler:
    """
    Central data management for the edition GUI.
    Handles loading, saving, and organizing data across multiple ports.
    """
    
    def __init__(self, fsamp: int = 10240, max_undo: int = 20):
        self.fsamp = fsamp
        self.max_undo = max_undo
        
        # Data storage
        self.ports: Dict[str, PortData] = {}
        self.session_name: str = ""
        
        # Edit history
        self.undo_stack: deque = deque(maxlen=max_undo)
        self.redo_stack: deque = deque(maxlen=max_undo)
        
        # State tracking
        self.modified: bool = False
        self._last_save_path: Optional[Path] = None
    
    # === File Loading ===
    
    def load_emg(self, path: Path, port_name: str, channels: List[int]) -> np.ndarray:
        """
        Load EMG data for a specific port.
        
        Parameters
        ----------
        path : Path
            Path to EMG file (.mat, .h5, .bin)
        port_name : str
            Name of the port
        channels : List[int]
            Channel indices to extract
        
        Returns
        -------
        emg : np.ndarray
            EMG data [channels x samples]
        """
        path = Path(path)
        
        if path.suffix == '.mat':
            emg = self._load_mat(path)
        elif path.suffix in ['.h5', '.hdf5']:
            emg = self._load_h5(path)
        elif path.suffix == '.bin':
            emg = self._load_bin(path)
        else:
            raise ValueError(f"Unsupported EMG format: {path.suffix}")
        
        # Ensure [channels x samples] format
        if emg.shape[0] > emg.shape[1]:
            emg = emg.T
        
        # Extract specified channels
        max_ch = max(channels)
        if max_ch >= emg.shape[0]:
            raise ValueError(f"Channel {max_ch} exceeds available channels ({emg.shape[0]})")
        
        emg_port = emg[channels, :]
        
        # Store in port
        if port_name not in self.ports:
            self.ports[port_name] = PortData(name=port_name)
        self.ports[port_name].emg_raw = emg_port
        
        return emg_port
    
    def load_decomposition(self, path: Path, port_name: str) -> List[MotorUnit]:
        """
        Load decomposition results for a port.
        
        Supports .pkl (SCD output) and .mat formats.
        """
        path = Path(path)
        
        if path.suffix == '.pkl':
            data = self._load_decomp_pkl(path)
        elif path.suffix == '.mat':
            data = self._load_decomp_mat(path)
        else:
            raise ValueError(f"Unsupported decomposition format: {path.suffix}")
        
        # Create motor units
        timestamps_list = data["timestamps"]
        sources = data.get("sources")
        filters = data.get("filters")
        
        if port_name not in self.ports:
            self.ports[port_name] = PortData(name=port_name)
        
        motor_units = []
        for i, ts in enumerate(timestamps_list):
            source = sources[i] if sources is not None else np.zeros(1)
            mu_filter = filters[i] if filters is not None and i < len(filters) else None
            
            mu = MotorUnit(
                id=i,
                timestamps=ts,
                source=source,
                port_name=port_name,
                filter=mu_filter,
            )
            motor_units.append(mu)
        
        self.ports[port_name].motor_units = motor_units
        return motor_units
    
    def _load_mat(self, path: Path) -> np.ndarray:
        """Load EMG from .mat file."""
        try:
            # Try scipy first (v7.2 and earlier)
            data = loadmat(str(path))
        except NotImplementedError:
            # HDF5-based .mat (v7.3)
            with h5py.File(path, 'r') as f:
                # Find EMG variable
                for key in ['emg', 'EMG', 'data', 'signal', 'signals']:
                    if key in f:
                        return np.array(f[key])
                # Take first numeric array
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        return np.array(f[key])
            raise ValueError("Could not find EMG data in .mat file")
        
        # scipy.io.loadmat result
        for key in ['emg', 'EMG', 'data', 'signal', 'signals']:
            if key in data:
                return np.asarray(data[key])
        
        # Find first numeric array
        for key, val in data.items():
            if not key.startswith('_') and isinstance(val, np.ndarray):
                if val.ndim == 2 and min(val.shape) > 1:
                    return val
        
        raise ValueError("Could not find EMG data in .mat file")
    
    def _load_h5(self, path: Path) -> np.ndarray:
        """Load EMG from HDF5 file."""
        with h5py.File(path, 'r') as f:
            for key in ['emg', 'EMG', 'data', 'signal']:
                if key in f:
                    return np.array(f[key])
            # Take first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return np.array(f[key])
        raise ValueError("Could not find EMG data in HDF5 file")
    
    def _load_bin(self, path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
        """Load EMG from binary file."""
        return np.fromfile(path, dtype=dtype)
    
    def _load_decomp_pkl(self, path: Path) -> Dict[str, Any]:
        """Load SCD decomposition output."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # SCD saves as dictionary with 'MUPulses' and 'sources'
        if 'MUPulses' in data:
            timestamps = [np.asarray(ts, dtype=np.int64) for ts in data['MUPulses']]
        elif 'timestamps' in data:
            timestamps = [np.asarray(ts, dtype=np.int64) for ts in data['timestamps']]
        else:
            raise ValueError("Could not find timestamps in decomposition file")
        
        sources = data.get('sources', data.get('source'))
        if sources is not None:
            sources = np.asarray(sources)
            if sources.ndim == 1:
                sources = sources.reshape(1, -1)
        
        filters = data.get('filters', data.get('mu_filters'))
        
        return {"timestamps": timestamps, "sources": sources, "filters": filters}
    
    def _load_decomp_mat(self, path: Path) -> Dict[str, Any]:
        """Load decomposition from .mat file."""
        try:
            data = loadmat(str(path))
        except NotImplementedError:
            with h5py.File(path, 'r') as f:
                timestamps = []
                for key in ['MUPulses', 'timestamps', 'firings']:
                    if key in f:
                        arr = np.array(f[key])
                        if arr.ndim == 1:
                            timestamps = [arr]
                        else:
                            timestamps = [arr[i] for i in range(arr.shape[0])]
                        break
                sources = None
                if 'sources' in f:
                    sources = np.array(f['sources'])
                filters = None
                if 'filters' in f or 'mu_filters' in f:
                    key = 'filters' if 'filters' in f else 'mu_filters'
                    filters = np.array(f[key])
                return {"timestamps": timestamps, "sources": sources, "filters": filters}
        
        # scipy.io result
        timestamps = []
        for key in ['MUPulses', 'timestamps', 'firings']:
            if key in data:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.dtype == object:
                        timestamps = [np.asarray(a).flatten() for a in arr.flatten()]
                    elif arr.ndim == 1:
                        timestamps = [arr]
                    else:
                        timestamps = [arr[i] for i in range(arr.shape[0])]
                break
        
        sources = data.get('sources', data.get('source'))
        filters = data.get('filters', data.get('mu_filters'))
        
        return {"timestamps": timestamps, "sources": sources, "filters": filters}
    
    # === Saving ===
    
    def save_decomposition(self, path: Path, port_name: str):
        """Save decomposition results for a port."""
        path = Path(path)
        port = self.ports.get(port_name)
        if port is None:
            raise KeyError(f"Port '{port_name}' not found")
        
        timestamps = [mu.timestamps for mu in port.motor_units if mu.enabled]
        sources = np.array([mu.source for mu in port.motor_units if mu.enabled])
        filters = np.array([mu.filter for mu in port.motor_units if mu.enabled and mu.filter is not None])
        
        data = {
            "MUPulses": timestamps,
            "sources": sources,
            "filters": filters if len(filters) > 0 else None,
            "fsamp": self.fsamp,
            "port_name": port_name,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif path.suffix == '.mat':
            savemat(str(path), data)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        self.modified = False
        self._last_save_path = path
    
    def save_all(self, output_dir: Path):
        """Save all ports to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for port_name in self.ports:
            path = output_dir / f"{self.session_name}_{port_name}.pkl"
            self.save_decomposition(path, port_name)
    
    # === Motor Unit Access ===
    
    def get_motor_unit(self, port_name: str, mu_id: int) -> MotorUnit:
        """Get a motor unit by port and ID."""
        port = self.ports.get(port_name)
        if port is None:
            raise KeyError(f"Port '{port_name}' not found")
        
        for mu in port.motor_units:
            if mu.id == mu_id:
                return mu
        raise KeyError(f"Motor unit {mu_id} not found in port '{port_name}'")
    
    def get_all_motor_units(self) -> List[Tuple[str, MotorUnit]]:
        """Get all motor units across all ports."""
        result = []
        for port_name, port in self.ports.items():
            for mu in port.motor_units:
                result.append((port_name, mu))
        return result
    
    # === Editing with Undo/Redo ===
    
    def _save_undo(self, action: EditAction):
        """Save an action to the undo stack."""
        self.undo_stack.append(action)
        self.redo_stack.clear()
        self.modified = True
    
    def add_spike(self, port_name: str, mu_id: int, sample: int):
        """Add a spike to a motor unit."""
        mu = self.get_motor_unit(port_name, mu_id)
        
        # Check if spike already exists
        if len(mu.timestamps) > 0 and np.any(np.abs(mu.timestamps - sample) < 0.005 * self.fsamp):
            return  # Spike too close to existing one
        
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.concatenate([mu.timestamps, [sample]])).astype(np.int64)
        
        action = EditAction(
            action_type="add",
            mu_id=mu_id,
            old_timestamps=old_ts,
            new_timestamps=new_ts,
        )
        self._save_undo(action)
        mu.timestamps = new_ts
    
    def delete_spike(self, port_name: str, mu_id: int, sample: int, tolerance_ms: float = 5.0):
        """Delete a spike from a motor unit."""
        mu = self.get_motor_unit(port_name, mu_id)
        
        tolerance_samples = int(tolerance_ms * self.fsamp / 1000)
        distances = np.abs(mu.timestamps - sample)
        
        if len(distances) == 0 or np.min(distances) > tolerance_samples:
            return  # No spike close enough
        
        idx = np.argmin(distances)
        old_ts = mu.timestamps.copy()
        new_ts = np.delete(mu.timestamps, idx)
        
        action = EditAction(
            action_type="delete",
            mu_id=mu_id,
            old_timestamps=old_ts,
            new_timestamps=new_ts,
        )
        self._save_undo(action)
        mu.timestamps = new_ts
    
    def add_spikes_roi(
        self,
        port_name: str,
        mu_id: int,
        x_start: float,
        x_end: float,
        y_min: float,
        y_max: float,
    ):
        """Add all peaks within ROI."""
        mu = self.get_motor_unit(port_name, mu_id)
        
        sample_start = int(x_start * self.fsamp)
        sample_end = int(x_end * self.fsamp)
        sample_end = min(sample_end, len(mu.source))
        sample_start = max(0, sample_start)
        
        # Find peaks in ROI
        segment = mu.source[sample_start:sample_end]
        min_distance = int(0.005 * self.fsamp)
        peaks, _ = signal.find_peaks(segment, distance=min_distance, height=(y_min, y_max))
        
        if len(peaks) == 0:
            return
        
        peaks_abs = peaks + sample_start
        
        # Filter peaks already close to existing spikes
        tolerance = int(0.005 * self.fsamp)
        new_peaks = []
        for p in peaks_abs:
            if len(mu.timestamps) == 0 or np.all(np.abs(mu.timestamps - p) > tolerance):
                new_peaks.append(p)
        
        if len(new_peaks) == 0:
            return
        
        old_ts = mu.timestamps.copy()
        new_ts = np.sort(np.concatenate([mu.timestamps, new_peaks])).astype(np.int64)
        
        action = EditAction(
            action_type="add_roi",
            mu_id=mu_id,
            old_timestamps=old_ts,
            new_timestamps=new_ts,
        )
        self._save_undo(action)
        mu.timestamps = new_ts
    
    def delete_spikes_roi(
        self,
        port_name: str,
        mu_id: int,
        x_start: float,
        x_end: float,
        y_min: float,
        y_max: float,
    ):
        """Delete all spikes within ROI."""
        mu = self.get_motor_unit(port_name, mu_id)
        
        sample_start = int(x_start * self.fsamp)
        sample_end = int(x_end * self.fsamp)
        
        keep_mask = np.ones(len(mu.timestamps), dtype=bool)
        
        for i, ts in enumerate(mu.timestamps):
            if sample_start <= ts < sample_end:
                if 0 <= ts < len(mu.source):
                    amplitude = mu.source[ts]
                    if y_min <= amplitude <= y_max:
                        keep_mask[i] = False
        
        if np.all(keep_mask):
            return  # Nothing to delete
        
        old_ts = mu.timestamps.copy()
        new_ts = mu.timestamps[keep_mask]
        
        action = EditAction(
            action_type="delete_roi",
            mu_id=mu_id,
            old_timestamps=old_ts,
            new_timestamps=new_ts,
        )
        self._save_undo(action)
        mu.timestamps = new_ts
    
    def undo(self) -> bool:
        """Undo the last action. Returns True if successful."""
        if not self.undo_stack:
            return False
        
        action = self.undo_stack.pop()
        
        # Find the motor unit
        for port in self.ports.values():
            for mu in port.motor_units:
                if mu.id == action.mu_id:
                    mu.timestamps = action.old_timestamps.copy()
                    if action.old_source is not None:
                        mu.source = action.old_source.copy()
                    if action.old_filter is not None:
                        mu.filter = action.old_filter.copy()
                    break
        
        self.redo_stack.append(action)
        return True
    
    def redo(self) -> bool:
        """Redo the last undone action. Returns True if successful."""
        if not self.redo_stack:
            return False
        
        action = self.redo_stack.pop()
        
        # Find the motor unit
        for port in self.ports.values():
            for mu in port.motor_units:
                if mu.id == action.mu_id:
                    mu.timestamps = action.new_timestamps.copy()
                    if action.new_source is not None:
                        mu.source = action.new_source.copy()
                    if action.new_filter is not None:
                        mu.filter = action.new_filter.copy()
                    break
        
        self.undo_stack.append(action)
        return True
    
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0