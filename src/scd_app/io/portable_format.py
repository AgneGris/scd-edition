"""A small, documented interchange format for recordings SCD can consume."""

from pathlib import Path
from typing import Any

import numpy as np


def write_portable_recording(
    file_path: str | Path,
    emg: np.ndarray,
    sampling_rate_hz: float,
    *,
    aux: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    emg_unit: str = "mV",
    source_format: str = "",
    source_file: str = "",
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a recording as portable, sample-first HDF5.

    Parameters
    ----------
    file_path
        Destination ending in ``.h5`` or ``.hdf5``.
    emg
        Numeric array shaped ``(samples, channels)``.
    sampling_rate_hz
        Native sampling frequency before any decimation.
    aux
        Optional sample-first auxiliary matrix (a vector becomes one column).
    timestamps
        Optional vector with one timestamp per EMG sample.
    overwrite
        Existing files are protected unless this is explicitly true.
    """
    import h5py

    destination = Path(file_path)
    if destination.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Portable recordings must use a .h5 or .hdf5 extension")

    emg_array = _sample_first_matrix(emg, "emg")
    try:
        sampling_rate = float(sampling_rate_hz)
    except (TypeError, ValueError) as err:
        raise ValueError("sampling_rate_hz must be a positive number") from err
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate_hz must be a positive number")

    aux_array = None
    if aux is not None:
        aux_array = _sample_first_matrix(aux, "aux", allow_vector=True)
        if aux_array.shape[0] != emg_array.shape[0]:
            raise ValueError("aux and emg must contain the same number of samples")

    timestamp_array = None
    if timestamps is not None:
        timestamp_array = np.asarray(timestamps)
        if timestamp_array.ndim != 1:
            raise ValueError("timestamps must be a one-dimensional array")
        if timestamp_array.shape[0] != emg_array.shape[0]:
            raise ValueError(
                "timestamps and emg must contain the same number of samples"
            )

    extra_metadata = metadata or {}
    reserved = {
        "scd_recording_format",
        "sampling_rate_hz",
        "emg_unit",
        "source_format",
        "source_file",
    }
    for key, value in extra_metadata.items():
        if key in reserved:
            raise ValueError(f"Metadata key {key!r} is reserved")
        if not isinstance(value, (str, bytes, bool, int, float, np.number)):
            raise TypeError(f"Metadata value for {key!r} must be a scalar or string")

    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(destination, mode) as file:
        file.attrs["scd_recording_format"] = "1.0"
        file.attrs["sampling_rate_hz"] = sampling_rate
        file.attrs["emg_unit"] = str(emg_unit)
        if source_format:
            file.attrs["source_format"] = str(source_format)
        if source_file:
            file.attrs["source_file"] = str(source_file)
        for key, value in extra_metadata.items():
            file.attrs[key] = value

        file.create_dataset(
            "emg", data=emg_array, compression="gzip", shuffle=True, chunks=True
        )
        if aux_array is not None:
            file.create_dataset(
                "aux", data=aux_array, compression="gzip", shuffle=True, chunks=True
            )
        if timestamp_array is not None:
            file.create_dataset(
                "timestamps",
                data=timestamp_array,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
    return destination


def _sample_first_matrix(
    value: np.ndarray, name: str, *, allow_vector: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if allow_vector and array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    return array
