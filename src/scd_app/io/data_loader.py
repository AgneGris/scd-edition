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


def _read_array(
    file_path: Path,
    fmt: str,
    field_spec: Dict,
    field_name: str = None,
    layout: Dict = None,
) -> np.ndarray:
    """Read a raw numpy array from file using the field spec."""
    
    if fmt == "mat_dynamic":
        return _read_mat_dynamic(file_path, field_spec)
    
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


def _read_mat_dynamic(file_path: Path, field_spec: Dict) -> np.ndarray:
    """
    Read from .mat with auto-detection across multiple possible sources.
    
    Supports:
      - "struct" sources: MATLAB struct with sub-fields matching a pattern (e.g. grid1, grid5)
        → finds all matching sub-fields, concatenates along channel axis
      - "array" sources: plain MATLAB variable
        → loads directly
    """
    import scipy.io as sio

    sources = field_spec.get("sources", [])
    if not sources:
        raise ValueError("mat_dynamic format requires 'sources' list in field spec")

    try:
        mat = sio.loadmat(str(file_path), squeeze_me=True)
    except NotImplementedError:
        # v7.3 .mat → HDF5
        return _read_mat_dynamic_h5(file_path, sources)

    for source in sources:
        path = source["path"]
        if path not in mat:
            continue

        if source.get("type") == "struct":
            pattern = source.get("pattern", "")
            struct = mat[path]

            # scipy loads MATLAB structs as numpy void with named fields
            if not (hasattr(struct, 'dtype') and struct.dtype.names):
                continue

            names = sorted(
                [n for n in struct.dtype.names if n.startswith(pattern)],
                key=lambda n: int(''.join(filter(str.isdigit, n)) or 0)
            )
            if not names:
                continue

            arrays = []
            for name in names:
                arr = np.asarray(struct[name])
                if arr.ndim == 0:
                    arr = np.asarray(arr.item())
                arrays.append(arr)

            # Concatenate along channel axis (smaller dimension)
            ref = arrays[0]
            if ref.ndim == 2:
                ch_axis = 0 if ref.shape[0] < ref.shape[1] else 1
                result = np.concatenate(arrays, axis=ch_axis)
            else:
                result = np.concatenate(arrays)

            grid_sizes = [a.shape[ch_axis] if a.ndim == 2 else len(a) for a in arrays]
            print(f"  Loaded {path}: {len(names)} grids {list(zip(names, grid_sizes))} "
                  f"→ {result.shape}")
            return result

        else:  # plain array
            result = np.asarray(mat[path])
            print(f"  Loaded {path}: {result.shape}")
            return result

    available = [k for k in mat.keys() if not k.startswith('__')]
    raise KeyError(
        f"No matching source found in {file_path.name}. "
        f"Tried: {[s['path'] for s in sources]}. Available: {available}"
    )


def _read_mat_dynamic_h5(file_path: Path, sources: list) -> np.ndarray:
    """Handle v7.3 .mat files (HDF5) for mat_dynamic format."""
    import h5py

    with h5py.File(file_path, 'r') as f:
        for source in sources:
            path = source["path"]
            if path not in f:
                continue

            item = f[path]

            if source.get("type") == "struct" and isinstance(item, h5py.Group):
                pattern = source.get("pattern", "")
                names = sorted(
                    [n for n in item.keys() if n.startswith(pattern)],
                    key=lambda n: int(''.join(filter(str.isdigit, n)) or 0)
                )
                if not names:
                    continue

                arrays = [np.array(item[n]) for n in names]
                ref = arrays[0]
                if ref.ndim == 2:
                    ch_axis = 0 if ref.shape[0] < ref.shape[1] else 1
                    result = np.concatenate(arrays, axis=ch_axis)
                else:
                    result = np.concatenate(arrays)

                grid_sizes = [a.shape[ch_axis] if a.ndim == 2 else len(a) for a in arrays]
                print(f"  Loaded {path} (HDF5): {len(names)} grids "
                      f"{list(zip(names, grid_sizes))} → {result.shape}")
                return result

            else:  # dataset
                result = np.array(item)
                print(f"  Loaded {path} (HDF5): {result.shape}")
                return result

        available = list(f.keys())
        raise KeyError(
            f"No matching source found in {file_path.name}. "
            f"Tried: {[s['path'] for s in sources]}. Available: {available}"
        )
    
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

    with tarfile.open(str(file_path), 'r') as tar:
        members = {m.name: m for m in tar.getmembers()}

        # Find the .sig and matching .xml
        sig_name = next(
            (n for n in members if n.endswith('.sig')),
            None,
        )
        if sig_name is None:
            raise FileNotFoundError(f"No .sig file found in {file_path.name}")

        xml_name = sig_name.rsplit('.', 1)[0] + '.xml'
        if xml_name not in members:
            raise FileNotFoundError(f"No matching XML '{xml_name}' in {file_path.name}")

        # Parse device XML
        xml_bytes = tar.extractfile(members[xml_name]).read()
        xml_root = ET.fromstring(xml_bytes)
        device_info = xml_root.attrib

        nADbit = int(device_info['ad_bits'])
        nchans = int(device_info['DeviceTotalChannels'])
        fs = int(device_info['SampleFrequency'])

        if field == "emg":
            return _read_otb_emg(tar, members, sig_name, xml_root, device_info,
                                 nADbit, nchans)
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


def _read_otb_emg(tar, members, sig_name, xml_root, device_info,
                   nADbit, nchans) -> np.ndarray:
    """Read and convert EMG channels from OTB+ to mV. Returns (samples, channels)."""
    # Device-specific power supply
    device_name = device_info['Name'].split(';')[0]
    if device_name != 'QUATTROCENTO':
        raise ValueError(f"Unsupported OTB device: {device_name}")
    power_supply = 5  # volts

    # Read raw signal → (samples, total_device_channels)
    sig_bytes = tar.extractfile(members[sig_name]).read()
    raw = (np.frombuffer(sig_bytes, dtype=f'int{nADbit}')
           .reshape(-1, nchans)
           .astype(np.float64))

    # Auto-detect active adapters from consecutive ChannelStartIndex values
    adapter_info = xml_root.findall('.//Adapter')
    active_adapters = []
    for i in range(len(adapter_info) - 1):
        start = int(adapter_info[i].attrib['ChannelStartIndex'])
        end = int(adapter_info[i + 1].attrib['ChannelStartIndex'])
        n_ch = end - start
        if n_ch > 0:
            active_adapters.append((i, start, n_ch))

    if not active_adapters:
        raise ValueError(f"No active adapters found in {sig_name}")

    total_emg_ch = sum(n_ch for _, _, n_ch in active_adapters)
    emg = np.zeros((raw.shape[0], total_emg_ch))

    col = 0
    for adapter_idx, raw_start, n_ch in active_adapters:
        gain = float(adapter_info[adapter_idx].attrib['Gain'])
        scale = (power_supply * 1000) / (2 ** nADbit * gain)
        emg[:, col:col + n_ch] = raw[:, raw_start:raw_start + n_ch] * scale
        col += n_ch

    return emg  # (samples, channels) in mV


def _read_otb_aux(tar, members) -> np.ndarray:
    """Read auxiliary .sip channels from OTB+. Returns (samples, channels)."""
    sip_names = sorted(n for n in members if n.endswith('.sip'))
    if not sip_names:
        raise KeyError("No auxiliary channels (.sip) found in OTB+ file")

    arrays = [
        np.frombuffer(tar.extractfile(members[n]).read(), dtype='float64')
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
        raise ValueError(
            f"Unknown orientation: '{orientation}'. "
            f"Use 'channels_first', 'samples_first', or 'auto'."
        )