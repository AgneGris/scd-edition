"""
EMG data loader.
Reads any EMG file given a YAML layout descriptor.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import yaml
import numpy as np
import torch


def load_layout(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML layout descriptor."""
    with open(yaml_path, "r") as f:
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
        Path to the data file (.mat, .h5, .hdf5, .npy, .otb+)
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
    raw = _read_array(file_path, fmt, field_spec, field_name=field, layout=layout)

    # Slice channels if specified
    raw = _slice_channels(raw, field_spec.get("channels"))

    # Fix orientation → always (samples, channels) for 2D
    if raw.ndim == 2:
        raw = _fix_orientation(raw, field_spec.get("orientation", "auto"))

    return torch.from_numpy(raw).to(dtype=torch.float32)


def _read_array(
    file_path: Path,
    fmt: str,
    field_spec: Dict,
    field_name: str = None,
    layout: Dict = None,
) -> np.ndarray:
    """Read a raw numpy array from file using the field spec."""

    primary_path = field_spec["path"]
    fallbacks = field_spec.get("fallback_keys", [])

    if fmt == "h5":
        return _read_h5(file_path, primary_path, fallbacks)
    elif fmt == "mat":
        return _read_mat(file_path, primary_path, fallbacks)
    elif fmt == "npy":
        return np.load(str(file_path))
    elif fmt == "otb":
        return _read_otb(file_path, field_name)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")


def _read_h5(file_path: Path, dataset_path: str, fallbacks: List[str]) -> np.ndarray:
    """Read from HDF5 file."""
    import h5py

    with h5py.File(file_path, "r") as f:
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
    """Read from .mat file (v5/v7 via scipy, v7.3 via h5py).

    var_name supports dot notation for MATLAB struct fields, e.g. "signal.data"
    resolves as mat["signal"]["data"] with scipy's (1,1)-wrapped struct convention.
    """
    import scipy.io as sio

    try:
        mat = sio.loadmat(str(file_path))
    except NotImplementedError:
        # v7.3 .mat files are HDF5
        return _read_h5(file_path, var_name, fallbacks)

    def _traverse(path: str):
        """Resolve a dot-notation path through the loaded mat dict."""
        parts = path.split(".")
        obj = mat
        for part in parts:
            if isinstance(obj, dict):
                if part not in obj:
                    return None
                obj = obj[part]
            elif isinstance(obj, np.ndarray):
                # scipy wraps scalar MATLAB structs as shape-(1,1) arrays
                if obj.shape == (1, 1):
                    obj = obj[0, 0]
                if obj.dtype.names and part in obj.dtype.names:
                    obj = obj[part]
                    if isinstance(obj, np.ndarray) and obj.shape == (1, 1):
                        obj = obj[0, 0]
                else:
                    return None
            else:
                return None
        return np.asarray(obj, dtype=np.float64)

    # Try primary path (dot-notation aware)
    result = _traverse(var_name)
    if result is not None:
        return result

    # Try fallbacks
    for key in fallbacks:
        result = _traverse(key)
        if result is not None:
            return result

    available = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(
        f"Variable '{var_name}' not found in {file_path.name}. "
        f"Available: {available}"
    )


def _read_otb(file_path: Path, field: str) -> np.ndarray:
    """
    Read from OTB+ file (tar archive) without extracting to disk.

    Auto-detects adapters and channel counts from the device XML.
    Returns data in (samples, channels) orientation.

    Supported fields:
        emg        → EMG channels converted to mV
        aux/force  → auxiliary .sip channels (float64)
        timestamps → computed from sample count and sampling frequency
    """
    import tarfile
    import xml.etree.ElementTree as ET

    with tarfile.open(str(file_path), "r") as tar:
        members = {m.name: m for m in tar.getmembers()}

        # Find the .sig and matching .xml
        sig_name = next(
            (n for n in members if n.endswith(".sig")),
            None,
        )
        if sig_name is None:
            raise FileNotFoundError(f"No .sig file found in {file_path.name}")

        xml_name = sig_name.rsplit(".", 1)[0] + ".xml"
        if xml_name not in members:
            raise FileNotFoundError(f"No matching XML '{xml_name}' in {file_path.name}")

        # Parse device XML
        xml_bytes = tar.extractfile(members[xml_name]).read()
        xml_root = ET.fromstring(xml_bytes)
        device_info = xml_root.attrib

        nADbit = int(device_info["ad_bits"])
        nchans = int(device_info["DeviceTotalChannels"])
        fs = int(device_info["SampleFrequency"])

        if field == "emg":
            return _read_otb_emg(
                tar, members, sig_name, xml_root, device_info, nADbit, nchans
            )
        elif field in ("aux", "force"):
            return _read_otb_aux(tar, members)
        elif field == "timestamps":
            sig_bytes = tar.extractfile(members[sig_name]).read()
            n_samples = len(sig_bytes) // (nADbit // 8 * nchans)
            return np.arange(n_samples, dtype=np.float64) / fs
        else:
            raise KeyError(
                f"Unknown field '{field}' for OTB+ format. "
                f"Supported: emg, aux, force, timestamps"
            )


def _read_otb_emg(
    tar, members, sig_name, xml_root, device_info, nADbit, nchans
) -> np.ndarray:
    """Read and convert EMG channels from OTB+ to mV. Returns (samples, channels)."""
    # Device-specific power supply
    device_name = device_info["Name"].split(";")[0]
    if device_name != "QUATTROCENTO":
        raise ValueError(f"Unsupported OTB device: {device_name}")
    power_supply = 5  # volts

    # Read raw signal → (samples, total_device_channels)
    sig_bytes = tar.extractfile(members[sig_name]).read()
    raw = (
        np.frombuffer(sig_bytes, dtype=f"int{nADbit}")
        .reshape(-1, nchans)
        .astype(np.float64)
    )

    # Auto-detect active adapters from consecutive ChannelStartIndex values
    adapter_info = xml_root.findall(".//Adapter")
    active_adapters = []
    for i in range(len(adapter_info) - 1):
        start = int(adapter_info[i].attrib["ChannelStartIndex"])
        end = int(adapter_info[i + 1].attrib["ChannelStartIndex"])
        n_ch = end - start
        if n_ch > 0:
            active_adapters.append((i, start, n_ch))

    if not active_adapters:
        raise ValueError(f"No active adapters found in {sig_name}")

    total_emg_ch = sum(n_ch for _, _, n_ch in active_adapters)
    emg = np.zeros((raw.shape[0], total_emg_ch))

    col = 0
    for adapter_idx, raw_start, n_ch in active_adapters:
        gain = float(adapter_info[adapter_idx].attrib["Gain"])
        scale = (power_supply * 1000) / (2**nADbit * gain)
        emg[:, col : col + n_ch] = raw[:, raw_start : raw_start + n_ch] * scale
        col += n_ch

    return emg  # (samples, channels) in mV


def _read_otb_aux(tar, members) -> np.ndarray:
    """Read auxiliary .sip channels from OTB+. Returns (samples, channels)."""
    sip_names = sorted(n for n in members if n.endswith(".sip"))
    if not sip_names:
        raise KeyError("No auxiliary channels (.sip) found in OTB+ file")

    arrays = [
        np.frombuffer(tar.extractfile(members[n]).read(), dtype="float64")
        for n in sip_names
    ]
    min_len = min(len(a) for a in arrays)
    return np.column_stack([a[:min_len] for a in arrays])


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
        return data[ch[0] : ch[1], :]
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
        raise ValueError(
            f"Unknown orientation: '{orientation}'. "
            f"Use 'channels_first', 'samples_first', or 'auto'."
        )
