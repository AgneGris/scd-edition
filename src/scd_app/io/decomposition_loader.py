"""Load decomposition pickles and adapt upstream SCD output for the editor."""

from __future__ import annotations

import io
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from scd_app.core.utils import to_numpy

GUI_FORMAT = "scd-edition"
UPSTREAM_SCD_FORMAT = "swarm-contrastive-decomposition"
CURRENT_SCHEMA_VERSION = 1
_GUI_REQUIRED_KEYS = {"ports", "discharge_times", "pulse_trains"}
_SCD_TIMESTAMP_KEYS = ("timestamps", "MUPulses")
_SCD_SOURCE_KEYS = ("source", "sources")
_SCD_METRIC_KEYS = ("silhouettes", "RoA", "fr", "cov", "best_exp")


class UnsupportedDecompositionFormat(ValueError):
    """Raised when a pickle is neither an SCD Edition file nor raw SCD output."""


def detect_decomposition_format(data: Any) -> str:
    """Return the supported format name for an unpickled object."""
    if not isinstance(data, dict):
        raise UnsupportedDecompositionFormat(
            f"Expected a dictionary, found {type(data).__name__}."
        )
    if _GUI_REQUIRED_KEYS.issubset(data):
        return GUI_FORMAT
    has_timestamps = any(key in data for key in _SCD_TIMESTAMP_KEYS)
    has_sources = any(key in data for key in _SCD_SOURCE_KEYS)
    if has_timestamps and has_sources:
        return UPSTREAM_SCD_FORMAT
    keys = ", ".join(sorted(str(key) for key in data)) or "<none>"
    raise UnsupportedDecompositionFormat(
        "The pickle is not a supported SCD Edition or raw "
        f"swarm-contrastive-decomposition result. Found keys: {keys}"
    )


def load_decomposition_file(path: Path) -> dict:
    """Load a trusted pickle and return a validated SCD Edition structure.

    Pickle files can execute code while loading. This function is intended only
    for decomposition files created by the user or another trusted source.
    """
    path = Path(path)
    with path.open("rb") as handle:
        data = _CPUCompatibleUnpickler(handle).load()
    file_format = detect_decomposition_format(data)
    if file_format == GUI_FORMAT:
        return migrate_and_validate_decomposition(data)
    return migrate_and_validate_decomposition(
        convert_scd_output(data, source_path=path)
    )


def migrate_and_validate_decomposition(data: dict) -> dict:
    """Upgrade a native session to the current schema and validate its shape."""
    migrated = dict(data)
    schema_version = migrated.get("schema_version")
    if schema_version is None:
        # Files written before a formal schema existed used the floating-point
        # ``version`` field. Their in-memory structure is the basis of schema 1.
        schema_version = CURRENT_SCHEMA_VERSION
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise UnsupportedDecompositionFormat(
            "SCD Edition schema_version must be an integer."
        )
    if schema_version < 1:
        raise UnsupportedDecompositionFormat(
            f"Unsupported SCD Edition schema version {schema_version}."
        )
    if schema_version > CURRENT_SCHEMA_VERSION:
        raise UnsupportedDecompositionFormat(
            "This file uses SCD Edition schema version "
            f"{schema_version}, but this application supports up to "
            f"version {CURRENT_SCHEMA_VERSION}. Please update SCD Edition."
        )

    migrated["format"] = GUI_FORMAT
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return _validate_native_decomposition(migrated)


def _validate_native_decomposition(data: dict) -> dict:
    """Return normalized data or raise before the Edition UI changes state."""
    ports = _outer_list(data.get("ports"), "ports")
    if not ports:
        raise UnsupportedDecompositionFormat("SCD Edition file contains no ports.")
    if any(not isinstance(port, str) or not port.strip() for port in ports):
        raise UnsupportedDecompositionFormat(
            "SCD Edition ports must be non-empty strings."
        )
    if len(set(ports)) != len(ports):
        raise UnsupportedDecompositionFormat("SCD Edition port names must be unique.")

    discharge_ports = _outer_list(data.get("discharge_times"), "discharge_times")
    source_ports = _outer_list(data.get("pulse_trains"), "pulse_trains")
    for field_name, values in (
        ("discharge_times", discharge_ports),
        ("pulse_trains", source_ports),
    ):
        if len(values) != len(ports):
            raise UnsupportedDecompositionFormat(
                f"SCD Edition {field_name} has {len(values)} port entries, "
                f"but ports has {len(ports)}."
            )

    sampling_rate = data.get("sampling_rate", data.get("fsamp"))
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDecompositionFormat(
            "SCD Edition file does not contain a valid sampling rate."
        ) from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise UnsupportedDecompositionFormat(
            "SCD Edition sampling rate must be a positive finite number."
        )

    plateau_start = 0
    plateau_coords = data.get("plateau_coords", data.get("selected_points"))
    if plateau_coords is not None:
        try:
            plateau = np.asarray(plateau_coords, dtype=float).flatten()
        except (TypeError, ValueError) as exc:
            raise UnsupportedDecompositionFormat(
                "SCD Edition plateau coordinates must be numeric."
            ) from exc
        if (
            plateau.size != 2
            or not np.all(np.isfinite(plateau))
            or plateau[0] < 0
            or plateau[1] <= plateau[0]
        ):
            raise UnsupportedDecompositionFormat(
                "SCD Edition plateau coordinates must be [start, end] with "
                "0 <= start < end."
            )
        plateau_start = int(plateau[0])
        data["plateau_coords"] = [plateau_start, int(plateau[1])]

    normalized_timestamps = []
    normalized_sources = []
    for port_index, port_name in enumerate(ports):
        timestamp_items = _unit_items(
            discharge_ports[port_index],
            f"discharge_times[{port_index}]",
        )
        source_items = _unit_items(
            source_ports[port_index],
            f"pulse_trains[{port_index}]",
        )
        if len(timestamp_items) != len(source_items):
            raise UnsupportedDecompositionFormat(
                f"Port {port_name!r} has {len(timestamp_items)} timestamp sets "
                f"but {len(source_items)} source signals."
            )

        port_timestamps = []
        port_sources = []
        for unit_index, (timestamps, source) in enumerate(
            zip(timestamp_items, source_items, strict=True)
        ):
            source_array = _numeric_array(
                source,
                f"pulse_trains[{port_index}][{unit_index}]",
            ).flatten()
            if source_array.size == 0:
                raise UnsupportedDecompositionFormat(
                    f"Port {port_name!r}, unit {unit_index} has an empty source."
                )

            timestamp_values = _numeric_array(
                timestamps,
                f"discharge_times[{port_index}][{unit_index}]",
            ).flatten()
            if not np.all(np.isfinite(timestamp_values)):
                raise UnsupportedDecompositionFormat(
                    f"Port {port_name!r}, unit {unit_index} has non-finite timestamps."
                )
            if not np.allclose(timestamp_values, np.rint(timestamp_values)):
                raise UnsupportedDecompositionFormat(
                    f"Port {port_name!r}, unit {unit_index} has non-integer timestamps."
                )
            timestamp_array = np.rint(timestamp_values).astype(np.int64)
            if timestamp_array.size and int(timestamp_array.min()) < 0:
                raise UnsupportedDecompositionFormat(
                    f"Port {port_name!r}, unit {unit_index} has negative timestamps."
                )
            if timestamp_array.size and int(timestamp_array.max()) >= source_array.size:
                # Some early files stored absolute timestamps beside a
                # plateau-local source. Migrate those coordinates explicitly.
                local = timestamp_array - plateau_start
                if (
                    plateau_start > 0
                    and int(local.min()) >= 0
                    and int(local.max()) < source_array.size
                ):
                    timestamp_array = local
                else:
                    raise UnsupportedDecompositionFormat(
                        f"Port {port_name!r}, unit {unit_index} has timestamps "
                        "outside its source signal."
                    )

            port_timestamps.append(timestamp_array)
            port_sources.append(source_array)

        normalized_timestamps.append(port_timestamps)
        normalized_sources.append(port_sources)

    raw_motor_unit_ids = data.get("motor_unit_ids")
    if raw_motor_unit_ids is None:
        normalized_motor_unit_ids = [
            list(range(len(port_timestamps)))
            for port_timestamps in normalized_timestamps
        ]
    else:
        motor_unit_id_ports = _outer_list(raw_motor_unit_ids, "motor_unit_ids")
        if len(motor_unit_id_ports) != len(ports):
            raise UnsupportedDecompositionFormat(
                "SCD Edition motor_unit_ids has "
                f"{len(motor_unit_id_ports)} port entries, but ports has "
                f"{len(ports)}."
            )

        normalized_motor_unit_ids = []
        for port_index, port_name in enumerate(ports):
            port_ids = _outer_list(
                motor_unit_id_ports[port_index],
                f"motor_unit_ids[{port_index}]",
            )
            expected_count = len(normalized_timestamps[port_index])
            if len(port_ids) != expected_count:
                raise UnsupportedDecompositionFormat(
                    f"Port {port_name!r} has {expected_count} motor units but "
                    f"{len(port_ids)} motor_unit_ids entries."
                )

            normalized_port_ids = []
            for unit_index, unit_id in enumerate(port_ids):
                if isinstance(unit_id, (bool, np.bool_)) or not isinstance(
                    unit_id, (int, np.integer)
                ):
                    raise UnsupportedDecompositionFormat(
                        "SCD Edition "
                        f"motor_unit_ids[{port_index}][{unit_index}] must be "
                        "a non-negative integer."
                    )
                normalized_id = int(unit_id)
                if normalized_id < 0:
                    raise UnsupportedDecompositionFormat(
                        "SCD Edition "
                        f"motor_unit_ids[{port_index}][{unit_index}] must be "
                        "a non-negative integer."
                    )
                normalized_port_ids.append(normalized_id)

            if len(set(normalized_port_ids)) != len(normalized_port_ids):
                raise UnsupportedDecompositionFormat(
                    f"SCD Edition motor_unit_ids for port {port_name!r} must be unique."
                )
            normalized_motor_unit_ids.append(normalized_port_ids)

    normalized = dict(data)
    normalized["ports"] = ports
    normalized["sampling_rate"] = sampling_rate
    normalized["discharge_times"] = normalized_timestamps
    normalized["pulse_trains"] = normalized_sources
    normalized["motor_unit_ids"] = normalized_motor_unit_ids
    if not isinstance(normalized.get("notes", []), list):
        normalized["notes"] = []
    if not isinstance(normalized.get("edit_history", []), list):
        normalized["edit_history"] = []
    return normalized


def _outer_list(value: Any, field_name: str) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return list(value)
    raise UnsupportedDecompositionFormat(f"SCD Edition {field_name} must be a list.")


def _unit_items(value: Any, field_name: str) -> list:
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(np.asarray(item).ndim == 0 for item in value):
            return [np.asarray(value)]
        return list(value)
    try:
        array = to_numpy(value)
    except Exception as exc:
        raise UnsupportedDecompositionFormat(
            f"SCD Edition {field_name} is not an array or list."
        ) from exc
    if array.ndim == 0 or array.size == 0:
        return []
    if array.ndim == 1:
        return [array]
    return [array[index] for index in range(array.shape[0])]


def _numeric_array(value: Any, field_name: str) -> np.ndarray:
    try:
        array = to_numpy(value)
    except Exception as exc:
        raise UnsupportedDecompositionFormat(
            f"SCD Edition {field_name} is not an array."
        ) from exc
    if not np.issubdtype(array.dtype, np.number):
        raise UnsupportedDecompositionFormat(
            f"SCD Edition {field_name} must contain numeric values."
        )
    return array


class _CPUCompatibleUnpickler(pickle.Unpickler):
    """Load torch-backed pickle values without requiring their original GPU."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            def load_torch_storage(value):
                return torch.load(
                    io.BytesIO(value),
                    map_location="cpu",
                    weights_only=False,
                )

            return load_torch_storage
        return super().find_class(module, name)


def convert_scd_output(data: dict, source_path: Path | None = None) -> dict:
    """Convert one raw upstream SCD dictionary into a one-port editor file."""
    if detect_decomposition_format(data) != UPSTREAM_SCD_FORMAT:
        raise UnsupportedDecompositionFormat(
            "The supplied dictionary is not raw SCD output."
        )

    timestamps_raw = _first_present(data, _SCD_TIMESTAMP_KEYS)
    timestamps = _timestamp_list(timestamps_raw)
    if not timestamps:
        raise UnsupportedDecompositionFormat("The SCD result contains no motor units.")

    n_units = len(timestamps)
    sources = [
        np.asarray(item).flatten()
        for item in _unit_list(
            _first_present(data, _SCD_SOURCE_KEYS), n_units, "source"
        )
    ]
    if any(source.size == 0 for source in sources):
        raise UnsupportedDecompositionFormat("One or more SCD sources are empty.")
    source_lengths = {int(source.size) for source in sources}
    if len(source_lengths) != 1:
        raise UnsupportedDecompositionFormat(
            f"SCD sources have inconsistent lengths: {sorted(source_lengths)}"
        )
    source_length = source_lengths.pop()

    for unit_idx, unit_timestamps in enumerate(timestamps):
        if unit_timestamps.size and (
            int(unit_timestamps.min()) < 0
            or int(unit_timestamps.max()) >= source_length
        ):
            raise UnsupportedDecompositionFormat(
                f"SCD timestamps for unit {unit_idx} fall outside its source."
            )

    filters = _filter_list(data.get("filters", data.get("mu_filters")), n_units)
    editor_filters = None if all(item is None for item in filters) else filters
    preprocessing_config = data.get("preprocessing_config") or {}
    if not isinstance(preprocessing_config, dict):
        raise UnsupportedDecompositionFormat(
            "SCD preprocessing_config must be a dictionary."
        )

    sampling_rate = data.get(
        "sampling_rate",
        data.get("fsamp", preprocessing_config.get("sampling_frequency")),
    )
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDecompositionFormat(
            "The SCD result does not record a valid sampling frequency."
        ) from exc
    if sampling_rate <= 0:
        raise UnsupportedDecompositionFormat(
            "The SCD sampling frequency must be positive."
        )

    w_mat_raw = data.get("w_mat")
    w_mat = to_numpy(w_mat_raw) if w_mat_raw is not None else None
    if w_mat is not None and w_mat.size == 0:
        w_mat = None
    n_channels = _infer_channel_count(filters, w_mat, preprocessing_config)
    port_name = _infer_port_name(source_path)

    # Signal as loaded by SCD (train(..., save_data=True) or
    # save_results(..., neural_data=...)); stored (channels, samples) like
    # decomp_worker does. It is the untrimmed recording with rejected channels
    # untouched, so the trimming and bad-channel fill that SCD's
    # preprocess_data applied are reproduced from preprocessing_config below.
    emg_data = _signal_array(data.get("data"))
    if emg_data is not None and not n_channels:
        n_channels = int(emg_data.shape[0])
    if emg_data is not None and n_channels and emg_data.shape[0] != n_channels:
        raise UnsupportedDecompositionFormat(
            f"SCD data has {emg_data.shape[0]} channels but the filters were "
            f"computed on {n_channels}."
        )

    emg_mask = _rejected_channel_mask(preprocessing_config, n_channels)
    start_sample = _window_start_sample(preprocessing_config, sampling_rate)

    provenance = {
        "format": UPSTREAM_SCD_FORMAT,
        "source_file": source_path.name if source_path is not None else None,
        "scd_commit": _read_scd_commit(source_path),
        "converted_by": "scd-edition",
    }
    provenance = {key: value for key, value in provenance.items() if value is not None}

    scd_metadata = {
        key: _portable_value(data[key]) for key in _SCD_METRIC_KEYS if key in data
    }

    converted = {
        "format": GUI_FORMAT,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "version": 1.1,
        "ports": [port_name],
        "sampling_rate": sampling_rate,
        "discharge_times": [timestamps],
        "pulse_trains": [sources],
        "mu_filters": [editor_filters],
        "w_mat": [w_mat],
        "peel_off_sequence": [_portable_value(data.get("peel_off_sequence", []))],
        "preprocessing_config": [_portable_value(preprocessing_config)],
        "plateau_coords": [start_sample, start_sample + source_length],
        "chans_per_electrode": [n_channels],
        "channel_indices": [list(range(n_channels))],
        "emg_mask": [emg_mask],
        "electrodes": [None],
        "aux_channels": [],
        "aux_configs": [],
        "decomposition_params": [{}],
        "acquisition_metadata": {"format": "scd-output"},
        "import_provenance": provenance,
        "scd_metadata": scd_metadata,
    }
    # Only present when SCD stored the signal: the full-source and filter
    # recalculation checks test for the key, and None would not survive them.
    if emg_data is not None:
        converted["data"] = emg_data
    return converted


def _signal_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = to_numpy(value)
    if arr.ndim != 2 or arr.size == 0:
        raise UnsupportedDecompositionFormat(
            f"SCD data must be a 2D signal, found shape {arr.shape}."
        )
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return np.ascontiguousarray(arr)


def _rejected_channel_mask(preprocessing_config: dict, n_channels: int) -> list[int]:
    """1 for channels SCD replaced with noise before decomposing, else 0."""
    mask = [0] * n_channels
    for channel in preprocessing_config.get("bad_channels") or []:
        try:
            channel = int(channel)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDecompositionFormat(
                f"SCD bad_channels contains a non-integer entry: {channel!r}."
            ) from exc
        if not 0 <= channel < n_channels:
            raise UnsupportedDecompositionFormat(
                f"SCD bad channel {channel} is outside the {n_channels} channels."
            )
        mask[channel] = 1
    return mask


def _window_start_sample(preprocessing_config: dict, sampling_rate: float) -> int:
    """First sample of the window SCD decomposed, in the signal as loaded."""
    start_time = preprocessing_config.get("start_time") or 0
    try:
        start_sample = int(round(float(start_time) * sampling_rate))
    except (TypeError, ValueError) as exc:
        raise UnsupportedDecompositionFormat(
            f"SCD start_time is not a number: {start_time!r}."
        ) from exc
    if start_sample < 0:
        raise UnsupportedDecompositionFormat(
            f"SCD start_time must not be negative, found {start_time!r}."
        )
    return start_sample


def _first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _timestamp_list(value: Any) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        arr = to_numpy(value)
        if arr.dtype == object:
            items = list(arr.flatten())
        elif arr.ndim <= 1:
            items = [arr]
        else:
            items = [arr[index] for index in range(arr.shape[0])]
    result = []
    for item in items:
        arr = to_numpy(item).flatten()
        if arr.size and not np.all(np.isfinite(arr)):
            raise UnsupportedDecompositionFormat(
                "SCD timestamps contain non-finite values."
            )
        result.append(arr.astype(np.int64, copy=False))
    return result


def _unit_list(value: Any, n_units: int, field_name: str) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        arr = to_numpy(value)
        if arr.dtype == object:
            items = list(arr.flatten())
        elif n_units == 1:
            items = [arr]
        elif arr.ndim >= 2 and arr.shape[0] == n_units:
            items = [arr[index] for index in range(n_units)]
        elif arr.ndim == 2 and arr.shape[1] == n_units:
            items = [arr[:, index] for index in range(n_units)]
        else:
            raise UnsupportedDecompositionFormat(
                f"Could not separate SCD {field_name} data into {n_units} units."
            )
    if len(items) != n_units:
        raise UnsupportedDecompositionFormat(
            f"SCD has {n_units} timestamp sets but {len(items)} {field_name} entries."
        )
    return [to_numpy(item) for item in items]


def _filter_list(value: Any, n_units: int) -> list[np.ndarray | None]:
    if value is None:
        return [None] * n_units
    return [np.asarray(item).flatten() for item in _unit_list(value, n_units, "filter")]


def _infer_channel_count(
    filters: list[np.ndarray | None],
    w_mat: np.ndarray | None,
    preprocessing_config: dict,
) -> int:
    feature_count = 0
    for mu_filter in filters:
        if mu_filter is not None and mu_filter.size:
            feature_count = int(mu_filter.size)
            break
    if not feature_count and w_mat is not None and w_mat.ndim >= 1:
        feature_count = int(w_mat.shape[0])
    try:
        extension_factor = int(preprocessing_config.get("extension_factor", 0))
    except (TypeError, ValueError):
        extension_factor = 0
    if extension_factor > 0 and feature_count % extension_factor == 0:
        return feature_count // extension_factor
    return 0


def _infer_port_name(source_path: Path | None) -> str:
    if source_path is None:
        return "SCD"
    match = re.search(r"(?:^|_)muscle-([^_]+)", source_path.stem, re.IGNORECASE)
    if match:
        return match.group(1)
    stem = re.sub(r"_scddict$", "", source_path.stem, flags=re.IGNORECASE)
    return stem or "SCD"


def _read_scd_commit(source_path: Path | None) -> str | None:
    if source_path is None:
        return None
    stem = re.sub(r"_scddict$", "", source_path.stem, flags=re.IGNORECASE)
    commit_path = source_path.with_name(f"{stem}_scdcommit.txt")
    if not commit_path.is_file():
        return None
    commit = commit_path.read_text(encoding="utf-8").strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) else None


def _portable_value(value: Any):
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "detach"):
        arr = to_numpy(value)
        return arr.item() if arr.ndim == 0 else arr
    return value
