"""Small, human-readable provenance reports for decomposition output files."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

AUDIT_FORMAT = "scd-edition-audit"
AUDIT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "development checkout"


def _basename(value: Any) -> str:
    """Strip both POSIX and Windows-style parents from a stored name."""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _software_versions() -> dict[str, str]:
    return {
        "scd_edition": _distribution_version("scd-edition"),
        "swarm_contrastive_decomposition": _distribution_version(
            "swarm-contrastive-decomposition"
        ),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
    }


def _compute_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device = None
    if cuda_available:
        with suppress(Exception):
            device = torch.cuda.get_device_name(0)
    return {
        "backend": "cuda" if cuda_available else "cpu",
        "device": device,
        "pytorch_cuda_build": torch.version.cuda,
    }


def _portable(value: Any) -> Any:
    """Convert small configuration values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        with suppress(Exception):
            return _portable(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        with suppress(Exception):
            return _portable(tolist())
    return str(value)


def _safe_index(values: Any, index: int, default: Any) -> Any:
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return default
    return values[index]


def _file_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    identity: dict[str, Any] = {"file_name": source.name}
    try:
        stat = source.stat()
    except OSError:
        return identity
    identity["size_bytes"] = stat.st_size
    identity["modified_at_utc"] = (
        datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return identity


def create_decomposition_provenance(
    source_path: Path | None,
    *,
    status: str,
    started_at_utc: str,
    duration_seconds: float,
    motor_units_by_port: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Capture run metadata that must survive subsequent edition saves."""
    return {
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "duration_seconds": round(max(0.0, duration_seconds), 3),
        "input_recording": _file_identity(source_path),
        "motor_units_detected": {
            "total": sum(motor_units_by_port.values()),
            "by_port": dict(motor_units_by_port),
        }
        if motor_units_by_port is not None
        else None,
        "software": _software_versions(),
        "compute": _compute_environment(),
    }


def _sanitise_provenance(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        return {}

    result = {
        key: _portable(provenance.get(key))
        for key in (
            "status",
            "started_at_utc",
            "completed_at_utc",
            "duration_seconds",
        )
        if provenance.get(key) is not None
    }
    recording = provenance.get("input_recording")
    if isinstance(recording, dict) and recording.get("file_name"):
        result["input_recording"] = {
            key: _portable(recording.get(key))
            for key in ("file_name", "size_bytes", "modified_at_utc", "sha256")
            if recording.get(key) is not None
        }
        result["input_recording"]["file_name"] = _basename(
            result["input_recording"]["file_name"]
        )

    detected = provenance.get("motor_units_detected")
    if isinstance(detected, dict):
        by_port = detected.get("by_port")
        if isinstance(by_port, dict):
            result["motor_units_detected"] = {
                "total": _portable(detected.get("total")),
                "by_port": {
                    str(port): _portable(count) for port, count in by_port.items()
                },
            }

    for section in ("software", "compute"):
        value = provenance.get(section)
        if isinstance(value, dict):
            result[section] = _portable(value)
    return result


def _port_summaries(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    ports = data.get("ports") or []
    discharge_times = data.get("discharge_times") or []
    pulse_trains = data.get("pulse_trains") or []
    summaries = []
    total_units = 0

    for index, port_name in enumerate(ports):
        units = _safe_index(discharge_times, index, None)
        if units is None:
            units = _safe_index(pulse_trains, index, [])
        try:
            unit_count = len(units)
        except TypeError:
            unit_count = 0
        total_units += unit_count

        channel_indices = _safe_index(data.get("channel_indices"), index, None)
        channel_count = _safe_index(data.get("chans_per_electrode"), index, None)
        if channel_count is None and isinstance(channel_indices, (list, tuple)):
            channel_count = len(channel_indices)

        rejection_mask = _safe_index(data.get("emg_mask"), index, [])
        rejected_channels = []
        if isinstance(rejection_mask, (list, tuple)):
            rejected_channels = [
                position
                for position, rejected in enumerate(rejection_mask)
                if bool(rejected)
            ]

        summary = {
            "name": str(port_name),
            "motor_units": unit_count,
            "channel_count": _portable(channel_count),
            "channel_indices": _portable(channel_indices),
            "rejected_channel_positions": rejected_channels,
            "electrode": _portable(_safe_index(data.get("electrodes"), index, None)),
            "decomposition_parameters": _portable(
                _safe_index(data.get("decomposition_params"), index, {})
            ),
            "preprocessing": _portable(
                _safe_index(data.get("preprocessing_config"), index, {})
            ),
        }
        summaries.append(summary)

    return summaries, total_units


def _auxiliary_summaries(configs: Any) -> list[dict[str, Any]]:
    if not isinstance(configs, list):
        return []
    allowed = (
        "name",
        "unit",
        "source",
        "start_chan",
        "end_chan",
        "field_path",
        "mvc",
    )
    return [
        {
            key: _portable(config.get(key))
            for key in allowed
            if config.get(key) is not None
        }
        for config in configs
        if isinstance(config, dict)
    ]


def _editing_summary(data: dict[str, Any]) -> dict[str, Any]:
    history = data.get("edit_history")
    history = history if isinstance(history, list) else []
    event_types = Counter(
        str(event.get("event_type", "unknown"))
        for event in history
        if isinstance(event, dict)
    )
    last_event_at = next(
        (
            event.get("datetime")
            for event in reversed(history)
            if isinstance(event, dict) and event.get("datetime")
        ),
        None,
    )
    notes = data.get("notes")
    flagged = data.get("flagged_mus")
    reliability_overrides = data.get("reliability_overrides")
    return {
        "history_events": len(history),
        "event_counts": dict(sorted(event_types.items())),
        "last_event_at": _portable(last_event_at),
        "notes_count": len(notes) if isinstance(notes, list) else 0,
        "units_flagged_for_deletion": _nested_item_count(flagged),
        "reliability_overrides": _nested_item_count(reliability_overrides),
    }


def _nested_item_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(
        len(items)
        for items in value.values()
        if isinstance(items, (list, tuple, dict, set))
    )


def build_audit_report(
    decomposition_path: Path,
    data: dict[str, Any],
    *,
    operation: str,
    derived_from: Path | None = None,
) -> dict[str, Any]:
    """Build a report without copying raw signals, spikes, notes, or paths."""
    ports, total_units = _port_summaries(data)
    provenance = _sanitise_provenance(data.get("audit_provenance"))
    detected = provenance.get("motor_units_detected", {})
    acquisition = data.get("acquisition_metadata")
    acquisition_format = (
        acquisition.get("format") if isinstance(acquisition, dict) else None
    )
    plateau = data.get("plateau_coords", data.get("selected_points"))

    output = {"file_name": Path(decomposition_path).name}
    if derived_from is not None:
        output["derived_from"] = Path(derived_from).name

    return {
        "format": AUDIT_FORMAT,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "operation": operation,
        "report_environment": {
            "software": _software_versions(),
            "compute": _compute_environment(),
        },
        "output": output,
        "provenance": provenance,
        "decomposition": {
            "sampling_rate_hz": _portable(data.get("sampling_rate", data.get("fsamp"))),
            "plateau_samples": _portable(plateau),
            "acquisition_format": _portable(acquisition_format),
            "motor_units_detected": detected.get("total"),
            "motor_units_retained": total_units,
            "ports": ports,
            "auxiliary_channels": _auxiliary_summaries(data.get("aux_configs")),
        },
        "editing": _editing_summary(data),
    }


def audit_path_for(decomposition_path: Path) -> Path:
    """Return the sidecar name for a decomposition output path."""
    return Path(decomposition_path).with_suffix(".audit.json")


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def write_audit_report(
    decomposition_path: Path,
    data: dict[str, Any],
    *,
    operation: str,
    derived_from: Path | None = None,
) -> Path:
    """Atomically write and return the decomposition's audit sidecar."""
    destination = audit_path_for(decomposition_path)
    report = build_audit_report(
        decomposition_path,
        data,
        operation=operation,
        derived_from=derived_from,
    )
    _atomic_json_dump(report, destination)
    return destination
