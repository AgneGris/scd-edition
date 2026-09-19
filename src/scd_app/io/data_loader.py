"""
EMG data loader.
Reads any EMG file given a YAML layout descriptor.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_layout(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML layout descriptor."""
    with open(yaml_path) as f:
        layout = yaml.safe_load(f)

    if "name" not in layout or "format" not in layout or "fields" not in layout:
        raise ValueError(
            f"Invalid layout file: must contain 'name', 'format', and 'fields' keys. "
            f"Got: {list(layout.keys())}"
        )
    return layout


def load_field(
    file_path: Path,
    layout: dict[str, Any],
    field: str,
) -> torch.Tensor:
    """
    Load a single field (emg, force, timestamps) from a data file.

    Parameters
    ----------
    file_path : Path
        Path to the data file (.mat, .h5, .hdf5, .npy, .csv, .txt,
        .otb+, .otb4, .rhs)
    layout : dict
        Parsed YAML layout descriptor (from load_layout)
    field : str
        Which field to load: "emg", "force", "timestamps", etc.

    Returns
    -------
    torch.Tensor
        For 2D fields: (samples, channels) — always this orientation.
        For 1D fields: (samples,)

    A top-level ``decimate: q`` in the layout reduces the sampling rate of
    every field by an integer factor ``q`` (anti-aliased for signals, plain
    subsampling for timestamps). Use it for recordings sampled far above the
    amplifier bandwidth so extension factors stay meaningful in milliseconds.
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

    expected_channels = field_spec.get("expected_channels")
    if expected_channels is not None:
        actual_channels = raw.shape[1] if raw.ndim == 2 else 1
        if actual_channels != int(expected_channels):
            raise ValueError(
                f"Field '{field}' in {file_path.name} contains {actual_channels} "
                f"channels, but layout '{layout['name']}' expects "
                f"{int(expected_channels)}"
            )

    q = decimation_factor(layout)
    if q > 1:
        raw = _decimate(raw, q, subsample_only=(field == "timestamps"))

    return torch.from_numpy(np.ascontiguousarray(raw)).to(dtype=torch.float32)


def decimation_factor(layout: dict[str, Any]) -> int:
    """Integer decimation factor declared by a layout (1 when absent)."""
    q = layout.get("decimate")
    if q is None:
        return 1
    try:
        q = int(q)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Layout 'decimate' must be an integer, got {q!r}") from err
    if q < 1:
        raise ValueError(f"Layout 'decimate' must be >= 1, got {q}")
    return q


def _decimate(data: np.ndarray, q: int, subsample_only: bool = False) -> np.ndarray:
    """Reduce the sample rate along axis 0 by an integer factor.

    Signals are low-pass filtered with a zero-phase FIR before subsampling so
    nothing above the new Nyquist folds back. Timestamps are only subsampled —
    filtering a time axis would be meaningless.
    """
    if subsample_only:
        return data[::q]
    from scipy.signal import decimate

    return decimate(
        np.asarray(data, dtype=np.float64), q, ftype="fir", axis=0, zero_phase=True
    ).astype(np.float32)


# File extensions each layout format can describe. Used when picking a preset
# for a file: an HDF5-backed format opens a .hdf5 file whatever its fields are
# called, so the format has to be ruled in by extension before its dataset
# paths are probed.
FORMAT_EXTENSIONS: dict[str, tuple] = {
    "h5": (".h5", ".hdf5"),
    "mat": (".mat",),
    "npy": (".npy",),
    "csv": (".csv", ".txt"),
    "otb": (".otb", ".otb+"),
    "otb4": (".otb4",),
    "rhs": (".rhs",),
}


def format_matches_extension(fmt: str, ext: str) -> bool:
    """True when a layout format can describe a file with this extension."""
    known = FORMAT_EXTENSIONS.get(fmt)
    # An unrecognised format is not evidence of a mismatch — leave it in play.
    return True if known is None else ext.lower() in known


def can_read_field(
    file_path: str | Path,
    layout: dict[str, Any],
    field: str = "emg",
) -> bool | None:
    """
    Check whether a layout's field resolves in a file, without reading the data.

    Several presets can describe the same extension (a generic ".hdf5" and a
    study-specific one, say), so callers need to tell them apart before loading
    hundreds of megabytes with the wrong dataset path.

    Returns
    -------
    True / False
        The field was found / not found in the file.
    None
        The format offers no cheap probe (scipy .mat, OTB, .npy). "Unknown" is
        not "unreadable" — the caller should fall back to its own heuristic.
    """
    file_path = Path(file_path)
    field_spec = layout.get("fields", {}).get(field)
    if field_spec is None:
        return False

    fmt = layout.get("format")
    if fmt not in ("h5", "mat"):
        return None

    keys = [field_spec.get("path"), *list(field_spec.get("fallback_keys", []))]
    keys = [k for k in keys if k]
    if not keys:
        return False

    import h5py

    try:
        with h5py.File(file_path, "r") as f:
            for name, expected in layout.get("required_attributes", {}).items():
                if name not in f.attrs:
                    return False
                actual = f.attrs[name]
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8", errors="replace")
                if expected is not None and str(actual) != str(expected):
                    return False
            return any(key in f and isinstance(f[key], h5py.Dataset) for key in keys)
    except OSError:
        # Not an HDF5 container. A "mat" layout then points at a v5/v7 file,
        # which scipy reads and h5py cannot probe — unknown, not unreadable.
        return None if fmt == "mat" else False


def load_metadata(file_path: str | Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Load acquisition metadata when the selected format provides it.

    When the layout decimates, the reported sampling frequency and sample
    count describe the data as ``load_field`` delivers it; the file's own
    values are kept under ``native_*`` keys.
    """
    file_path = Path(file_path)
    fmt = layout["format"]
    if fmt == "otb4":
        from scd_app.io.otb4_loader import read_otb4_metadata

        meta = read_otb4_metadata(file_path)
    elif fmt == "rhs":
        from scd_app.io.rhs_loader import read_rhs_metadata

        meta = read_rhs_metadata(file_path)
    elif fmt == "h5":
        meta = _read_h5_metadata(file_path, layout)
    else:
        return {"format": fmt}

    q = decimation_factor(layout)
    if q > 1 and "sampling_frequency" in meta:
        meta["native_sampling_frequency"] = meta["sampling_frequency"]
        meta["sampling_frequency"] = meta["sampling_frequency"] / q
        if "n_samples" in meta:
            meta["native_n_samples"] = meta["n_samples"]
            # scipy.signal.decimate yields ceil(n / q) samples.
            meta["n_samples"] = -(-int(meta["n_samples"]) // q)
        meta["decimate"] = q
    return meta


def _read_array(
    file_path: Path,
    fmt: str,
    field_spec: dict,
    field_name: str | None = None,
    layout: dict | None = None,
) -> np.ndarray:
    """Read a raw numpy array from file using the field spec."""

    primary_path = field_spec.get("path", "")
    fallbacks = field_spec.get("fallback_keys", [])

    if fmt == "h5":
        return _read_h5(file_path, primary_path, fallbacks)
    elif fmt == "mat":
        return _read_mat(file_path, primary_path, fallbacks)
    elif fmt == "npy":
        return np.load(str(file_path), allow_pickle=False)
    elif fmt == "csv":
        return _read_delimited(file_path, field_spec)
    elif fmt == "otb":
        return _read_otb(file_path, field_name)
    elif fmt == "otb4":
        from scd_app.io.otb4_loader import read_otb4

        return read_otb4(file_path, field_name)
    elif fmt == "rhs":
        from scd_app.io.rhs_loader import read_rhs

        return read_rhs(file_path, field_name)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")


def _read_delimited(file_path: Path, field_spec: dict) -> np.ndarray:
    """Read a numeric delimited-text matrix.

    ``delimiter`` defaults to a comma for CSV and to whitespace for ``.txt``.
    ``skip_header`` may be an integer or ``"auto"``; auto skips one leading
    row when it is not entirely numeric. Comment lines beginning with ``#``
    are ignored by NumPy.
    """
    delimiter = field_spec.get("delimiter")
    if delimiter is None:
        delimiter = None if file_path.suffix.lower() == ".txt" else ","

    skip_header = field_spec.get("skip_header", "auto")
    if skip_header == "auto":
        skip_header = _detect_text_header(file_path, delimiter)
    try:
        skip_header = int(skip_header)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "CSV 'skip_header' must be a non-negative integer or 'auto'"
        ) from err
    if skip_header < 0:
        raise ValueError("CSV 'skip_header' must be non-negative")

    data = np.genfromtxt(
        file_path,
        delimiter=delimiter,
        comments="#",
        dtype=np.float64,
        skip_header=skip_header,
    )
    data = np.asarray(data)
    if data.size == 0 or data.ndim == 0:
        raise ValueError(f"No numeric matrix found in {file_path.name}")
    if np.isnan(data).all():
        raise ValueError(f"No numeric values found in {file_path.name}")
    return data


def _detect_text_header(file_path: Path, delimiter: str | None) -> int:
    """Return 1 when the first non-comment text row contains non-numbers."""
    with open(file_path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split(delimiter) if delimiter else stripped.split()
            try:
                for value in values:
                    float(value.strip())
            except ValueError:
                # genfromtxt counts physical lines for skip_header, including
                # comments and blanks before this first data-like row.
                return line_number
            return 0
    return 0


def _read_h5_metadata(file_path: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Read shape and standard sampling-rate attributes without loading arrays."""
    import h5py

    meta: dict[str, Any] = {"format": "h5"}
    field_spec = layout.get("fields", {}).get("emg", {})
    paths = [field_spec.get("path"), *field_spec.get("fallback_keys", [])]
    paths = [path for path in paths if path]

    with h5py.File(file_path, "r") as file:
        dataset = next(
            (
                file[path]
                for path in paths
                if path in file and isinstance(file[path], h5py.Dataset)
            ),
            None,
        )
        if dataset is None:
            return meta

        if dataset.ndim == 2:
            first, second = map(int, dataset.shape)
            orientation = field_spec.get("orientation", "auto")
            if orientation == "channels_first" or (
                orientation == "auto" and second > first
            ):
                n_samples, n_channels = second, first
            else:
                n_samples, n_channels = first, second
            meta["n_samples"] = n_samples
            meta["emg_channel_count"] = n_channels

        attr_names = (
            "sampling_rate_hz",
            "sampling_frequency",
            "sampling_rate",
            "fsamp",
            "fs",
        )
        for attrs in (dataset.attrs, file.attrs):
            for name in attr_names:
                if name not in attrs:
                    continue
                value = np.asarray(attrs[name]).squeeze()
                if value.ndim == 0:
                    try:
                        frequency = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(frequency) and frequency > 0:
                        meta["sampling_frequency"] = frequency
                        return meta
    return meta


def _read_h5(file_path: Path, dataset_path: str, fallbacks: list[str]) -> np.ndarray:
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


def _read_mat(file_path: Path, var_name: str, fallbacks: list[str]) -> np.ndarray:
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
        # A path that stops on a struct (or on a non-numeric field) is not a
        # usable array — treat it as a miss so the next fallback gets a turn.
        if isinstance(obj, np.ndarray) and obj.dtype.names:
            return None
        try:
            return np.asarray(obj, dtype=np.float64)
        except (ValueError, TypeError):
            return None

    # Try primary path (dot-notation aware)
    result = _traverse(var_name)
    if result is not None:
        return result

    # Try fallbacks
    for key in fallbacks:
        result = _traverse(key)
        if result is not None:
            return result

    available = [k for k in mat if not k.startswith("__")]
    raise KeyError(
        f"Variable '{var_name}' not found in {file_path.name}. Available: {available}"
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
