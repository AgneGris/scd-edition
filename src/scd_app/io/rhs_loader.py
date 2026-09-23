"""Reader for Intan RHS2000 ``.rhs`` recordings (Intan Stim/Recording Controller).

File layout follows Intan's "RHS2000 Data File Formats" note: a little-endian
header describing the amplifier settings and every signal channel, followed by
fixed-size blocks of 128 samples. Within a block each stream is stored
channel-major, i.e. all 128 samples of channel 0, then channel 1, and so on.

Only the streams SCD Edition needs are decoded:

    emg        → enabled amplifier channels, converted to mV
    aux        → enabled board ADC inputs, converted to V
    timestamps → seconds from the first sample

DC-amplifier and stimulation streams are skipped over but never decoded.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

RHS_MAGIC = 0xD69127AC
SAMPLES_PER_BLOCK = 128

# ADC step sizes from the Intan format note.
AMPLIFIER_UV_PER_STEP = 0.195  # µV per bit, offset binary around 32768
BOARD_ADC_V_PER_STEP = 312.5e-6  # V per bit, offset binary around 32768

# ``signal_type`` values in the channel records. Types 1 and 2 (aux input,
# supply voltage) exist only in RHD files and never appear in RHS.
_SIGNAL_AMPLIFIER = 0
_SIGNAL_BOARD_ADC = 3
_SIGNAL_BOARD_DAC = 4
_SIGNAL_DIGITAL_IN = 5
_SIGNAL_DIGITAL_OUT = 6

# Notch setting recorded in the header. The controller software applies this
# only to its display; saved data are unfiltered.
_NOTCH_MODES = {0: None, 1: 50.0, 2: 60.0}


@dataclass(frozen=True)
class _Channel:
    native_name: str
    custom_name: str
    native_order: int
    signal_type: int
    port_name: str
    port_prefix: str
    impedance_magnitude: float
    impedance_phase: float


@dataclass
class _Header:
    version: tuple
    sampling_frequency: float
    dsp_enabled: bool
    dsp_cutoff_hz: float
    lower_bandwidth_hz: float
    upper_bandwidth_hz: float
    notch_hz: float | None
    notes: list[str]
    dc_amplifier_saved: bool
    reference_channel: str
    header_bytes: int
    amplifier: list[_Channel] = field(default_factory=list)
    board_adc: list[_Channel] = field(default_factory=list)
    board_dac: list[_Channel] = field(default_factory=list)
    digital_in: list[_Channel] = field(default_factory=list)
    digital_out: list[_Channel] = field(default_factory=list)

    @property
    def block_dtype(self) -> np.dtype:
        """Structured dtype of one 128-sample data block."""
        n_amp = len(self.amplifier)
        fields = [("t", "<i4", (SAMPLES_PER_BLOCK,))]
        if n_amp:
            fields.append(("amp", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
            if self.dc_amplifier_saved:
                fields.append(("dc", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
            fields.append(("stim", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
        if self.board_adc:
            fields.append(("adc", "<u2", (len(self.board_adc), SAMPLES_PER_BLOCK)))
        if self.board_dac:
            fields.append(("dac", "<u2", (len(self.board_dac), SAMPLES_PER_BLOCK)))
        if self.digital_in:
            fields.append(("din", "<u2", (SAMPLES_PER_BLOCK,)))
        if self.digital_out:
            fields.append(("dout", "<u2", (SAMPLES_PER_BLOCK,)))
        return np.dtype(fields)


def read_rhs(file_path: Path, field_name: str) -> np.ndarray:
    """Read a canonical field from an RHS file.

    Two-dimensional fields are returned as ``(samples, channels)``.
    """
    file_path = Path(file_path)
    with open(file_path, "rb") as fid:
        header = _read_header(fid, file_path)
        n_blocks = _count_blocks(file_path, header)
        if field_name == "timestamps":
            n_samples = n_blocks * SAMPLES_PER_BLOCK
            return np.arange(n_samples, dtype=np.float64) / header.sampling_frequency

        if field_name == "emg":
            if not header.amplifier:
                raise ValueError(f"No amplifier channels enabled in {file_path.name}")
            stream, offset, scale = "amp", 32768.0, AMPLIFIER_UV_PER_STEP / 1000.0
        elif field_name in ("aux", "force"):
            if not header.board_adc:
                raise KeyError(
                    f"No board ADC (aux) channels enabled in {file_path.name}"
                )
            stream, offset, scale = "adc", 32768.0, BOARD_ADC_V_PER_STEP
        else:
            raise KeyError(
                f"Unknown field '{field_name}' for RHS format. "
                "Supported: emg, aux, force, timestamps"
            )

        blocks = np.fromfile(fid, dtype=header.block_dtype, count=n_blocks, offset=0)

    raw = blocks[stream]  # (n_blocks, n_channels, 128)
    n_channels = raw.shape[1]
    # Block-major → contiguous per-channel → (samples, channels)
    data = raw.transpose(1, 0, 2).reshape(n_channels, -1).T
    return (data.astype(np.float32) - np.float32(offset)) * np.float32(scale)


def read_rhs_metadata(file_path: Path) -> dict:
    """Return JSON-serializable acquisition and channel metadata."""
    file_path = Path(file_path)
    with open(file_path, "rb") as fid:
        header = _read_header(fid, file_path)
    n_blocks = _count_blocks(file_path, header)
    n_samples = n_blocks * SAMPLES_PER_BLOCK
    fs = header.sampling_frequency

    grids = []
    if header.amplifier:
        # One "grid" per headstage port so the config tab can report them,
        # mirroring the Novecento+ metadata shape.
        start = 0
        for port in _ordered_unique(ch.port_name for ch in header.amplifier):
            chans = [ch for ch in header.amplifier if ch.port_name == port]
            grids.append(
                {
                    "index": len(grids),
                    "name": port,
                    "output_channel_start": start,
                    "output_channel_end": start + len(chans),
                    "channel_count": len(chans),
                    "channel_names": [ch.native_name for ch in chans],
                    "custom_channel_names": [ch.custom_name for ch in chans],
                    "impedance_kohm": [
                        round(ch.impedance_magnitude / 1e3, 1) for ch in chans
                    ],
                    "sampling_frequency": fs,
                    "unit": "mV",
                }
            )
            start += len(chans)

    return {
        "format": "rhs",
        "device": "Intan RHS2000",
        "file_version": f"{header.version[0]}.{header.version[1]}",
        "sampling_frequency": fs,
        "n_samples": n_samples,
        "duration_seconds": n_samples / fs,
        "emg_channel_count": len(header.amplifier),
        "aux_channel_count": len(header.board_adc),
        "grids": grids,
        "aux_channels": [
            {
                "index": i,
                "name": ch.native_name,
                "custom_name": ch.custom_name,
                "unit": "V",
            }
            for i, ch in enumerate(header.board_adc)
        ],
        "amplifier_bandwidth_hz": [
            header.lower_bandwidth_hz,
            header.upper_bandwidth_hz,
        ],
        "dsp_offset_removal_hz": header.dsp_cutoff_hz if header.dsp_enabled else None,
        "notch_display_hz": header.notch_hz,
        "dc_amplifier_saved": header.dc_amplifier_saved,
        "reference_channel": header.reference_channel,
        "notes": [n for n in header.notes if n],
    }


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _read_header(fid: BinaryIO, file_path: Path) -> _Header:
    def read(fmt: str):
        size = struct.calcsize("<" + fmt)
        chunk = fid.read(size)
        if len(chunk) != size:
            raise ValueError(f"Truncated RHS header in {file_path.name}")
        return struct.unpack("<" + fmt, chunk)

    (magic,) = read("I")
    if magic != RHS_MAGIC:
        raise ValueError(
            f"{file_path.name} is not an Intan RHS file (magic 0x{magic:08X})"
        )
    version = read("hh")
    (fs,) = read("f")

    (dsp_enabled,) = read("h")
    dsp_cutoff, lower_bw, _lower_settle_bw, upper_bw = read("ffff")
    read("ffff")  # desired dsp/lower/lower-settle/upper bandwidths
    (notch_mode,) = read("h")
    read("ff")  # desired / actual impedance test frequency
    read("hh")  # amp settle mode, charge recovery mode
    read("fff")  # stim step size, charge recovery current limit / target voltage

    notes = [_read_qstring(fid, file_path) for _ in range(3)]
    (dc_saved,) = read("h")
    read("h")  # eval board mode
    reference = _read_qstring(fid, file_path)

    header = _Header(
        version=version,
        sampling_frequency=float(fs),
        dsp_enabled=bool(dsp_enabled),
        dsp_cutoff_hz=float(dsp_cutoff),
        lower_bandwidth_hz=float(lower_bw),
        upper_bandwidth_hz=float(upper_bw),
        notch_hz=_NOTCH_MODES.get(int(notch_mode)),
        notes=notes,
        dc_amplifier_saved=bool(dc_saved),
        reference_channel=reference,
        header_bytes=0,
    )

    (n_groups,) = read("h")
    for _ in range(n_groups):
        group_name = _read_qstring(fid, file_path)
        group_prefix = _read_qstring(fid, file_path)
        group_enabled, n_channels, _n_amp = read("hhh")
        for _ in range(n_channels):
            native = _read_qstring(fid, file_path)
            custom = _read_qstring(fid, file_path)
            (
                native_order,
                _custom_order,
                signal_type,
                channel_enabled,
                _chip_channel,
                _command_stream,
                _board_stream,
            ) = read("hhhhhhh")
            read("hhhh")  # voltage trigger mode/threshold, digital trigger, edge
            imp_mag, imp_phase = read("ff")
            if not (group_enabled and channel_enabled):
                continue
            channel = _Channel(
                native_name=native,
                custom_name=custom,
                native_order=int(native_order),
                signal_type=int(signal_type),
                port_name=group_name,
                port_prefix=group_prefix,
                impedance_magnitude=float(imp_mag),
                impedance_phase=float(imp_phase),
            )
            target = {
                _SIGNAL_AMPLIFIER: header.amplifier,
                _SIGNAL_BOARD_ADC: header.board_adc,
                _SIGNAL_BOARD_DAC: header.board_dac,
                _SIGNAL_DIGITAL_IN: header.digital_in,
                _SIGNAL_DIGITAL_OUT: header.digital_out,
            }.get(channel.signal_type)
            if target is None:
                raise ValueError(
                    f"Unsupported signal type {signal_type} for channel "
                    f"{native!r} in {file_path.name}"
                )
            target.append(channel)

    header.header_bytes = fid.tell()
    return header


def _read_qstring(fid: BinaryIO, file_path: Path) -> str:
    """Read a Qt-serialised string: uint32 byte length then UTF-16LE text."""
    chunk = fid.read(4)
    if len(chunk) != 4:
        raise ValueError(f"Truncated RHS header in {file_path.name}")
    (length,) = struct.unpack("<I", chunk)
    if length == 0xFFFFFFFF:
        return ""
    if length % 2:
        raise ValueError(f"Malformed string in RHS header of {file_path.name}")
    data = fid.read(length)
    if len(data) != length:
        raise ValueError(f"Truncated RHS header in {file_path.name}")
    return data.decode("utf-16-le")


def _count_blocks(file_path: Path, header: _Header) -> int:
    payload = file_path.stat().st_size - header.header_bytes
    block_bytes = header.block_dtype.itemsize
    n_blocks, remainder = divmod(payload, block_bytes)
    if remainder:
        # A recording stopped mid-block leaves a partial trailer; keep the
        # complete blocks rather than refusing the whole file.
        logger.warning(
            "%s: ignoring %s trailing bytes from an incomplete final data block",
            file_path.name,
            remainder,
        )
    if n_blocks == 0:
        raise ValueError(f"{file_path.name} contains no data blocks")
    return int(n_blocks)


def _ordered_unique(values) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)
