"""Discover candidate arrays in user-supplied scientific data files."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scd_app.io.data_loader import _read_delimited


@dataclass(frozen=True)
class ArrayCandidate:
    """A numeric array that could contain EMG or an auxiliary signal."""

    path: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def is_matrix(self) -> bool:
        return len(self.shape) == 2 and all(size > 0 for size in self.shape)

    @property
    def display_name(self) -> str:
        dimensions = " × ".join(str(value) for value in self.shape)
        return f"{self.path} — {dimensions} [{self.dtype}]"


@dataclass(frozen=True)
class RecordingInspection:
    """Arrays and metadata discovered without executing file-supplied code."""

    file_path: Path
    arrays: tuple[ArrayCandidate, ...]
    sampling_rate_hz: float | None = None
    sampling_rate_source: str | None = None

    @property
    def suggested_array(self) -> ArrayCandidate | None:
        matrices = [candidate for candidate in self.arrays if candidate.is_matrix]
        return max(matrices, key=_candidate_score, default=None)


def inspect_recording(file_path: str | Path) -> RecordingInspection:
    """Inspect MATLAB, HDF5, NumPy, CSV, or delimited-text data."""
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension in {".h5", ".hdf5"}:
        return _inspect_hdf5(path)
    if extension == ".mat":
        try:
            return _inspect_hdf5(path)
        except OSError:
            return _inspect_mat(path)
    if extension == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        candidate = _candidate("array", array)
        return RecordingInspection(path, (candidate,) if candidate else ())
    if extension in {".csv", ".txt"}:
        array = _read_delimited(
            path,
            {"delimiter": None, "skip_header": "auto"},
        )
        candidate = _candidate("table", array)
        return RecordingInspection(path, (candidate,) if candidate else ())
    raise ValueError(
        f"Cannot inspect {extension or 'a file without an extension'} files. "
        "Use MATLAB, HDF5, NumPy, CSV, or delimited text."
    )


def _inspect_hdf5(path: Path) -> RecordingInspection:
    import h5py

    arrays: list[ArrayCandidate] = []
    sampling: tuple[float, str] | None = None
    with h5py.File(path, "r") as file:
        sampling = _sampling_from_mapping(file.attrs, "file attribute")

        def visit(name: str, item):
            nonlocal sampling
            if not isinstance(item, h5py.Dataset):
                return
            if sampling is None:
                sampling = _sampling_from_mapping(
                    item.attrs, f"{name} dataset attribute"
                )
            if item.ndim == 0 and _is_sampling_name(name):
                value = _positive_scalar(item[()])
                if value is not None and sampling is None:
                    sampling = value, f"{name} dataset"
            if item.ndim > 0 and np.issubdtype(item.dtype, np.number):
                arrays.append(
                    ArrayCandidate(name, tuple(map(int, item.shape)), str(item.dtype))
                )

        file.visititems(visit)

    arrays.sort(key=_candidate_score, reverse=True)
    return RecordingInspection(
        path,
        tuple(arrays),
        sampling[0] if sampling else None,
        sampling[1] if sampling else None,
    )


def _inspect_mat(path: Path) -> RecordingInspection:
    import scipy.io as sio

    arrays: list[ArrayCandidate] = []
    sampling: tuple[float, str] | None = None

    def walk(value: Any, name: str):
        nonlocal sampling
        if isinstance(value, Mapping):
            for child_name, child in value.items():
                if str(child_name).startswith("__"):
                    continue
                path_name = f"{name}.{child_name}" if name else str(child_name)
                walk(child, path_name)
            return

        if _is_sampling_name(name):
            scalar = _positive_scalar(value)
            if scalar is not None and sampling is None:
                sampling = scalar, f"{name} variable"

        candidate = _candidate(name, value)
        if candidate is not None:
            arrays.append(candidate)

    numeric_classes = {
        "double",
        "single",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    }
    for name, shape, matlab_class in sio.whosmat(path):
        if _is_sampling_name(name):
            value = sio.loadmat(path, variable_names=[name], simplify_cells=True)[name]
            scalar = _positive_scalar(value)
            if scalar is not None and sampling is None:
                sampling = scalar, f"{name} variable"
            continue
        if matlab_class in numeric_classes:
            arrays.append(
                ArrayCandidate(name, tuple(map(int, shape)), f"MATLAB {matlab_class}")
            )
        elif matlab_class == "struct":
            # Load structs one at a time because their nested field shapes are
            # not exposed by whosmat. Top-level numeric matrices stay on the
            # cheap metadata-only path, which matters for large recordings.
            value = sio.loadmat(path, variable_names=[name], simplify_cells=True)[name]
            walk(value, name)

    arrays.sort(key=_candidate_score, reverse=True)
    return RecordingInspection(
        path,
        tuple(arrays),
        sampling[0] if sampling else None,
        sampling[1] if sampling else None,
    )


def _candidate(path: str, value: Any) -> ArrayCandidate | None:
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
    if value.ndim == 0 or not np.issubdtype(value.dtype, np.number):
        return None
    return ArrayCandidate(path, tuple(map(int, value.shape)), str(value.dtype))


def _candidate_score(candidate: ArrayCandidate) -> tuple[int, int, int, int]:
    """Put plausible, explicitly named EMG matrices first."""
    normalised = _normalise_name(candidate.path)
    name_score = 0
    if "emg" in normalised:
        name_score = 3
    elif "signal" in normalised or "recording" in normalised:
        name_score = 2
    elif "data" in normalised:
        name_score = 1

    if not candidate.is_matrix:
        return (0, name_score, 0, int(np.prod(candidate.shape)))
    smaller, larger = sorted(candidate.shape)
    plausible_channels = int(2 <= smaller <= 2048 and larger > smaller)
    return (2, plausible_channels, name_score, int(np.prod(candidate.shape)))


def _sampling_from_mapping(
    values: Mapping[str, Any], source: str
) -> tuple[float, str] | None:
    for name in values:
        if not _is_sampling_name(str(name)):
            continue
        value = _positive_scalar(values[name])
        if value is not None:
            return value, f"{source} {name}"
    return None


def _is_sampling_name(name: str) -> bool:
    final_component = name.replace("/", ".").split(".")[-1]
    return _normalise_name(final_component) in {
        "samplingratehz",
        "samplingfrequency",
        "samplingrate",
        "fsamp",
        "fs",
    }


def _normalise_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _positive_scalar(value: Any) -> float | None:
    array = np.asarray(value).squeeze()
    if array.ndim != 0:
        return None
    try:
        result = float(array)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None
