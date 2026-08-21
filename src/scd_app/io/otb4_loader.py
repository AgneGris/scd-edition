"""Reader for Novecento+ ``.otb4`` recording archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tarfile
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class _Track:
    title: str
    subtitle: str
    device: str
    is_control: bool
    n_channels: int
    total_channels: int
    channel_offset: int
    acquisition_channel: int
    sampling_frequency: int
    sample_size: int
    gain: float
    adc_bits: int
    adc_range: float
    unit: str
    unit_factor: float
    duration: float
    stream: str
    adapter: str
    sensor: str
    mode: str
    channel_description: str
    description_name: str
    rows: int | None
    columns: int | None
    ied_mm: float | None
    lowpass_filter: str
    highpass_filter: str
    start_date: str
    end_date: str

    @property
    def is_quaternion(self) -> bool:
        return "quaternion" in f"{self.title} {self.description_name}".lower()

    @property
    def is_emg(self) -> bool:
        return (
            not self.is_control
            and self.device.lower() == "novecento+"
            and re.fullmatch(r"IN\d+", self.title.strip(), re.IGNORECASE) is not None
            and self.adc_bits > 0
        )

    @property
    def is_aux(self) -> bool:
        return (
            not self.is_control
            and self.device.lower() == "novecento+"
            and not self.is_emg
            and not self.is_quaternion
            and self.adc_bits > 0
        )


def read_otb4(file_path: Path, field: str) -> np.ndarray:
    """Read a canonical field from a Novecento+ archive.

    Two-dimensional fields are returned as ``(samples, channels)``. EMG and
    auxiliary signals are converted from ADC counts using each track's gain,
    ADC range, bit depth, and unit factor.
    """
    file_path = Path(file_path)
    with tarfile.open(str(file_path), "r:*") as archive:
        members = {member.name: member for member in archive.getmembers()}
        device, tracks = _parse_archive_metadata(archive, members, file_path)
        emg_tracks = [track for track in tracks if track.is_emg]
        aux_tracks = [track for track in tracks if track.is_aux]
        _validate_track_groups(
            emg_tracks, aux_tracks, device["sampling_frequency"], file_path
        )
        _validate_sample_alignment(members, [*emg_tracks, *aux_tracks], file_path)

        if field == "emg":
            return _read_track_group(archive, members, emg_tracks, file_path)
        if field in ("aux", "force"):
            if not aux_tracks:
                raise KeyError(f"No auxiliary channels found in {file_path.name}")
            return _read_track_group(archive, members, aux_tracks, file_path)
        if field == "timestamps":
            n_samples = _track_sample_count(members, emg_tracks[0], file_path)
            return np.arange(n_samples, dtype=np.float64) / device["sampling_frequency"]
        raise KeyError(
            f"Unknown field '{field}' for Novecento+ format. "
            "Supported: emg, aux, force, timestamps"
        )


def read_otb4_metadata(file_path: Path) -> Dict:
    """Return JSON-serializable acquisition and channel metadata."""
    file_path = Path(file_path)
    with tarfile.open(str(file_path), "r:*") as archive:
        members = {member.name: member for member in archive.getmembers()}
        device, tracks = _parse_archive_metadata(archive, members, file_path)
        emg_tracks = [track for track in tracks if track.is_emg]
        aux_tracks = [track for track in tracks if track.is_aux]
        _validate_track_groups(
            emg_tracks, aux_tracks, device["sampling_frequency"], file_path
        )
        _validate_sample_alignment(members, [*emg_tracks, *aux_tracks], file_path)

        n_samples = _track_sample_count(members, emg_tracks[0], file_path)
        grids = _describe_tracks(emg_tracks, members, file_path)
        aux_channels = _describe_tracks(aux_tracks, members, file_path)
        return {
            "format": "otb4",
            "device": "Novecento+",
            "sampling_frequency": device["sampling_frequency"],
            "sample_size": device["sample_size"],
            "adc_bits": device["adc_bits"],
            "adc_range": device["adc_range"],
            "gain": device["gain"],
            "n_samples": n_samples,
            "duration_seconds": n_samples / device["sampling_frequency"],
            "emg_channel_count": sum(track.n_channels for track in emg_tracks),
            "aux_channel_count": sum(track.n_channels for track in aux_tracks),
            "grids": grids,
            "aux_channels": aux_channels,
            "start_date": emg_tracks[0].start_date,
            "end_date": emg_tracks[0].end_date,
        }


def _parse_archive_metadata(
    archive: tarfile.TarFile,
    members: Dict[str, tarfile.TarInfo],
    file_path: Path,
) -> Tuple[Dict, List[_Track]]:
    device_member = _find_member(members, "DeviceParameters.xml", file_path)
    tracks_member = _find_tracks_member(members, file_path)
    device_root = _read_xml(archive, device_member, file_path)
    tracks_root = _read_xml(archive, tracks_member, file_path)

    device_type = next(
        (value for key, value in device_root.attrib.items() if key.endswith("}type")),
        "",
    )
    if "novecento" not in device_type.lower():
        raise ValueError(
            f"{file_path.name} is not a Novecento+ recording "
            f"(DeviceParameters type is {device_type!r})"
        )

    device = {
        "adc_bits": _required_int(device_root, "AdBits", file_path),
        "sample_size": _required_int(device_root, "SampleSize", file_path),
        "sampling_frequency": _required_int(
            device_root, "SamplingFrequency", file_path
        ),
        "gain": _required_float(device_root, "Gain", file_path),
        "adc_range": _required_float(device_root, "ADC_Range", file_path),
    }
    tracks = [_parse_track(node, file_path) for node in tracks_root.findall("TrackInfo")]
    if not tracks:
        raise ValueError(f"No TrackInfo records found in {file_path.name}")
    return device, tracks


def _parse_track(node: ET.Element, file_path: Path) -> _Track:
    descriptions = node.find("StringsDescriptions")
    description = node.find("Description")

    def get_desc(key: str) -> str:
        return _text(descriptions, key)

    def get_geometry(key: str) -> str | None:
        return _optional_number(description, key)

    return _Track(
        title=_text(node, "Title"),
        subtitle=_text(node, "SubTitle"),
        device=_text(node, "Device"),
        is_control=_text(node, "IsControl").lower() == "true",
        n_channels=_required_int(node, "NumberOfChannels", file_path),
        total_channels=_required_int(node, "TotalChannelsInFile", file_path),
        channel_offset=_required_int(node, "ChannelOffsetInSubPacket", file_path),
        acquisition_channel=_required_int(node, "AcquisitionChannel", file_path),
        sampling_frequency=_required_int(node, "SamplingFrequency", file_path),
        sample_size=_required_int(node, "SampleSize", file_path),
        gain=_required_float(node, "Gain", file_path),
        adc_bits=_required_int(node, "ADC_Nbits", file_path),
        adc_range=_required_float(node, "ADC_Range", file_path),
        unit=_text(node, "UnitOfMeasurement"),
        unit_factor=_required_float(node, "UnitOfMeasurementFactor", file_path),
        duration=_required_float(node, "TimeDuration", file_path),
        stream=_text(node, "SignalStreamPath"),
        adapter=get_desc("OriginalAdapter"),
        sensor=get_desc("OriginalSensor"),
        mode=get_desc("Mode"),
        channel_description=get_desc("Channels"),
        description_name=_text(description, "Name"),
        rows=_as_int(get_geometry("NRow")),
        columns=_as_int(get_geometry("NColumn")),
        ied_mm=_as_float(get_geometry("IED")),
        lowpass_filter=get_desc("LowPassFilter"),
        highpass_filter=get_desc("HighPassFilter"),
        start_date=get_desc("StartDate"),
        end_date=get_desc("EndDate"),
    )


def _validate_track_groups(
    emg_tracks: List[_Track],
    aux_tracks: List[_Track],
    device_sampling_frequency: int,
    file_path: Path,
) -> None:
    if not emg_tracks:
        raise ValueError(f"No Novecento+ EMG input tracks found in {file_path.name}")

    emg_fs = {track.sampling_frequency for track in emg_tracks}
    if len(emg_fs) != 1:
        raise ValueError(
            f"EMG tracks in {file_path.name} have inconsistent sampling frequencies: "
            f"{sorted(emg_fs)}"
        )
    if next(iter(emg_fs)) != device_sampling_frequency:
        raise ValueError(
            f"EMG tracks in {file_path.name} use {next(iter(emg_fs))} Hz, "
            f"but DeviceParameters declares {device_sampling_frequency} Hz"
        )
    aux_fs = {track.sampling_frequency for track in aux_tracks}
    if aux_fs and aux_fs != emg_fs:
        raise ValueError(
            f"Auxiliary tracks in {file_path.name} use {sorted(aux_fs)} Hz, "
            f"but EMG uses {next(iter(emg_fs))} Hz; the streams cannot be aligned"
        )

    for track in [*emg_tracks, *aux_tracks]:
        if track.n_channels <= 0:
            raise ValueError(f"Track {track.title!r} declares no channels")
        if track.channel_offset < 0 or (
            track.channel_offset + track.n_channels > track.total_channels
        ):
            raise ValueError(
                f"Track {track.title!r} in {file_path.name} declares channels "
                f"[{track.channel_offset}, {track.channel_offset + track.n_channels}) "
                f"outside its {track.total_channels}-channel stream"
            )


def _validate_sample_alignment(
    members: Dict[str, tarfile.TarInfo], tracks: List[_Track], file_path: Path
) -> None:
    counts = {
        (track.title, track.subtitle, track.stream): _track_sample_count(
            members, track, file_path
        )
        for track in tracks
    }
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        details = ", ".join(
            f"{title or subtitle or stream}={count}"
            for (title, subtitle, stream), count in counts.items()
        )
        raise ValueError(
            f"Novecento+ streams in {file_path.name} do not align: {details}"
        )


def _read_track_group(
    archive: tarfile.TarFile,
    members: Dict[str, tarfile.TarInfo],
    tracks: Iterable[_Track],
    file_path: Path,
) -> np.ndarray:
    arrays: List[np.ndarray] = []
    stream_cache: Dict[str, np.ndarray] = {}
    expected_samples: int | None = None

    for track in tracks:
        raw = stream_cache.get(track.stream)
        if raw is None:
            member = _find_stream_member(members, track.stream, file_path)
            n_samples = _track_sample_count(members, track, file_path)
            dtype = _integer_dtype(track.sample_size, track, file_path)
            stream = archive.extractfile(member)
            if stream is None:
                raise FileNotFoundError(
                    f"Could not read stream {track.stream!r} in {file_path.name}"
                )
            raw = np.frombuffer(stream.read(), dtype=dtype).reshape(
                n_samples, track.total_channels
            )
            stream_cache[track.stream] = raw

        selected = raw[
            :, track.channel_offset: track.channel_offset + track.n_channels
        ]
        scale = _track_scale(track, file_path)
        arrays.append(selected.astype(np.float32) * np.float32(scale))
        if expected_samples is None:
            expected_samples = selected.shape[0]
        elif selected.shape[0] != expected_samples:
            raise ValueError(
                f"Track {track.title!r} in {file_path.name} has "
                f"{selected.shape[0]} samples; expected {expected_samples}"
            )

    return np.concatenate(arrays, axis=1)


def _track_sample_count(
    members: Dict[str, tarfile.TarInfo], track: _Track, file_path: Path
) -> int:
    member = _find_stream_member(members, track.stream, file_path)
    bytes_per_sample = track.sample_size * track.total_channels
    if bytes_per_sample <= 0 or member.size % bytes_per_sample:
        raise ValueError(
            f"Stream {track.stream!r} in {file_path.name} is {member.size} bytes, "
            f"not a whole number of {track.total_channels}-channel packets "
            f"with {track.sample_size}-byte samples"
        )
    return member.size // bytes_per_sample


def _track_scale(track: _Track, file_path: Path) -> float:
    if track.adc_bits <= 0 or track.gain == 0:
        raise ValueError(
            f"Track {track.title!r} in {file_path.name} has invalid ADC metadata "
            f"(bits={track.adc_bits}, gain={track.gain})"
        )
    return track.adc_range * track.unit_factor / (2**track.adc_bits * track.gain)


def _integer_dtype(sample_size: int, track: _Track, file_path: Path) -> np.dtype:
    try:
        return np.dtype(f"<i{sample_size}")
    except TypeError as exc:
        raise ValueError(
            f"Track {track.title!r} in {file_path.name} uses unsupported "
            f"sample size {sample_size}"
        ) from exc


def _describe_tracks(
    tracks: Iterable[_Track],
    members: Dict[str, tarfile.TarInfo],
    file_path: Path,
) -> List[Dict]:
    result = []
    output_start = 0
    for index, track in enumerate(tracks):
        result.append(
            {
                "index": index,
                "name": track.title,
                "subtitle": track.subtitle,
                "channel_description": track.channel_description,
                "output_channel_start": output_start,
                "output_channel_end": output_start + track.n_channels,
                "channel_count": track.n_channels,
                "source_stream": track.stream,
                "source_channel_start": track.channel_offset,
                "source_channel_end": track.channel_offset + track.n_channels,
                "acquisition_channel": track.acquisition_channel,
                "sampling_frequency": track.sampling_frequency,
                "n_samples": _track_sample_count(members, track, file_path),
                "unit": track.unit,
                "unit_factor": track.unit_factor,
                "gain": track.gain,
                "adc_bits": track.adc_bits,
                "adc_range": track.adc_range,
                "adapter": track.adapter,
                "sensor": track.sensor,
                "mode": track.mode,
                "rows": track.rows,
                "columns": track.columns,
                "ied_mm": track.ied_mm,
                "lowpass_filter": track.lowpass_filter,
                "highpass_filter": track.highpass_filter,
            }
        )
        output_start += track.n_channels
    return result


def _read_xml(
    archive: tarfile.TarFile, member: tarfile.TarInfo, file_path: Path
) -> ET.Element:
    stream = archive.extractfile(member)
    if stream is None:
        raise FileNotFoundError(f"Could not read {member.name!r} in {file_path.name}")
    try:
        return ET.fromstring(stream.read())
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in {member.name!r}: {exc}") from exc


def _find_member(
    members: Dict[str, tarfile.TarInfo], basename: str, file_path: Path
) -> tarfile.TarInfo:
    match = next(
        (member for name, member in members.items() if Path(name).name == basename),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"No {basename!r} found in {file_path.name}")
    return match


def _find_tracks_member(
    members: Dict[str, tarfile.TarInfo], file_path: Path
) -> tarfile.TarInfo:
    matches = sorted(
        (
            member
            for name, member in members.items()
            if re.fullmatch(r"Tracks_\d+\.xml", Path(name).name, re.IGNORECASE)
        ),
        key=lambda member: member.name,
    )
    if not matches:
        raise FileNotFoundError(f"No Tracks_*.xml found in {file_path.name}")
    return matches[0]


def _find_stream_member(
    members: Dict[str, tarfile.TarInfo], stream_name: str, file_path: Path
) -> tarfile.TarInfo:
    if stream_name in members:
        return members[stream_name]
    return _find_member(members, Path(stream_name).name, file_path)


def _text(node: ET.Element | None, name: str) -> str:
    if node is None:
        return ""
    value = node.findtext(name)
    return value.strip() if value else ""


def _required_int(node: ET.Element, name: str, file_path: Path) -> int:
    value = _text(node, name)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid or missing {name!r} metadata in {file_path.name}: {value!r}"
        ) from exc


def _required_float(node: ET.Element, name: str, file_path: Path) -> float:
    value = _text(node, name)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid or missing {name!r} metadata in {file_path.name}: {value!r}"
        ) from exc


def _optional_number(node: ET.Element | None, name: str) -> str | None:
    value = _text(node, name)
    return value or None


def _as_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _as_float(value: str | None) -> float | None:
    return float(value) if value is not None else None
