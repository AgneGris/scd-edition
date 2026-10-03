"""Pure loading and persistence helpers for Edition sessions.

This module owns transformations between Edition's in-memory motor-unit model
and its persisted decomposition dictionary. It intentionally has no Qt imports
so the data contract can be tested without constructing the GUI.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scd_app.core.electrode_layouts import get_grid_config
from scd_app.core.mu_model import MotorUnit
from scd_app.core.mu_properties import compute_port_properties
from scd_app.core.utils import to_numpy
from scd_app.io.decomposition_loader import CURRENT_SCHEMA_VERSION, GUI_FORMAT

logger = logging.getLogger(__name__)

_PASSTHROUGH_KEYS = (
    "data",
    "aux_channels",
    "plateau_coords",
    "chans_per_electrode",
    "channel_indices",
    "emg_mask",
    "electrodes",
    "dewhitened_filters",
    "version",
    "preprocessing_config",
    "w_mat",
    "selected_points",
    "import_provenance",
    "scd_metadata",
    "audit_provenance",
    "decomposition_params",
    "aux_configs",
    "acquisition_metadata",
)


@dataclass(frozen=True)
class EditionSaveState:
    """Qt-independent snapshot of the state needed for an Edition save."""

    ports: Mapping[str, Sequence[MotorUnit]]
    sampling_rate: float
    full_source_mode: bool
    start_sample: int
    end_sample: int
    edit_history: list
    notes: list[str]
    original_decomposition: Mapping[str, Any] | None = None
    emg_data: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class LoadedEditionPort:
    """Parsed, GUI-independent state for one electrode port."""

    channel_count: int
    motor_units: list[MotorUnit]
    emg: np.ndarray | None
    raw_channels: np.ndarray | None
    grid_config: dict | None
    active_grid_positions: dict[int, tuple[int, int]] | None
    rejected_channel_positions: set[tuple[int, int]]
    migrated_notes: list[str]


def normalise_notes(notes: object) -> list[str]:
    """Return non-empty text entries from a persisted or edited note list."""
    if not isinstance(notes, list):
        return []
    return [note.strip() for note in notes if isinstance(note, str) and note.strip()]


_NOTE_TAG = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\((?P<port>.+?), MU (?P<mu_id>\d+)\): "
)


def note_tag(note: str) -> tuple[str, int] | None:
    """Return the (port, MU id) a note is tagged with, or None if untagged."""
    match = _NOTE_TAG.match(note)
    if match is None:
        return None
    return match["port"], int(match["mu_id"])


def notes_for_unit(notes: Sequence[str], port_name: str, mu_id: int) -> list[str]:
    """Return the notes tagged with one motor unit."""
    return [note for note in notes if note_tag(note) == (port_name, mu_id)]


def remap_note_tags(
    notes: Sequence[str], id_maps: Mapping[str, Mapping[int, int]]
) -> list[str]:
    """Rewrite MU tags after deleted units are removed and the rest renumbered.

    ``id_maps`` maps each affected port's old unit ids to their new ids. Notes
    for units missing from their port's map are kept but tagged as deleted, so
    they no longer match any remaining unit.
    """
    remapped = []
    for note in notes:
        match = _NOTE_TAG.match(note)
        if match is None or match["port"] not in id_maps:
            remapped.append(note)
            continue
        port, old_id = match["port"], int(match["mu_id"])
        new_id = id_maps[port].get(old_id)
        tag = f"MU {old_id}, deleted" if new_id is None else f"MU {new_id}"
        remapped.append(f"{match['stamp']} ({port}, {tag}): {note[match.end() :]}")
    return remapped


def ensure_list_of_arrays(data: object) -> list[np.ndarray]:
    """Normalise legacy tensor, array, and list containers per motor unit."""
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        return []
    if isinstance(data, list):
        if not data:
            return []
        first = data[0]
        if isinstance(first, (np.ndarray, list)) or hasattr(first, "detach"):
            return [to_numpy(item) for item in data]
        return [to_numpy(data)]
    array = to_numpy(data)
    if array.ndim == 0 or array.size == 0:
        return []
    if array.ndim == 1:
        return [array]
    return [array[index] for index in range(array.shape[0])]


def normalise_aux_channels(
    aux_channels: object,
    *,
    acquisition_format: str | None,
    configured_channels: Sequence[Mapping[str, Any]] = (),
) -> list[dict]:
    """Upgrade persisted auxiliary-channel metadata in place.

    Older files nested channel attributes under ``meta`` and sometimes stored
    MVC values in volts while the corresponding signal was represented in mV.
    Missing MVC values may be filled from the active application configuration.
    """
    if not isinstance(aux_channels, list):
        return []

    channels = [channel for channel in aux_channels if isinstance(channel, dict)]
    for channel in channels:
        metadata = channel.pop("meta", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                channel.setdefault(key, value)

        mvc = channel.get("mvc")
        if (
            acquisition_format not in ("otb4", "rhs")
            and mvc is not None
            and 0 < float(mvc) < 1.0
        ):
            corrected = float(mvc) * 1000.0
            logger.info(
                "Corrected legacy MVC unit for %s: %s V -> %s mV",
                channel.get("unit", "?"),
                mvc,
                corrected,
            )
            channel["mvc"] = corrected

    configured_by_key = {}
    for configured in configured_channels:
        for key in (configured.get("name", ""), configured.get("unit", "")):
            if key:
                configured_by_key[key] = configured

    for channel in channels:
        if channel.get("mvc") is not None:
            continue
        match = configured_by_key.get(channel.get("name", "")) or configured_by_key.get(
            channel.get("unit", "")
        )
        if match and match.get("mvc") is not None:
            channel["mvc"] = match["mvc"]
            logger.info(
                "Filled MVC from configuration for %s: %s mV",
                channel.get("unit", "?"),
                channel["mvc"],
            )

    return channels


def load_edition_port(
    *,
    port_index: int,
    port_name: str,
    decomposition: Mapping[str, Any],
    emg_full: np.ndarray | None,
    start_sample: int,
    end_sample: int,
    full_port_results: Mapping[int, Sequence],
    channel_offset: int,
    full_source_mode: bool,
    sampling_rate: float,
    existing_notes: Sequence[str] = (),
    property_computer: Callable[..., list] = compute_port_properties,
) -> LoadedEditionPort:
    """Parse one persisted port without accessing any Qt widget state."""
    channels_per_electrode = decomposition.get("chans_per_electrode", [])
    channel_indices_all = decomposition.get("channel_indices")
    mask_list = decomposition.get("emg_mask", [])
    electrode_list = decomposition.get("electrodes", [])

    channel_count = (
        int(channels_per_electrode[port_index])
        if port_index < len(channels_per_electrode)
        else 64
    )

    if (
        channel_indices_all is not None
        and port_index < len(channel_indices_all)
        and channel_indices_all[port_index] is not None
    ):
        port_channel_indices = np.asarray(channel_indices_all[port_index], dtype=int)
    else:
        port_channel_indices = np.arange(
            channel_offset, channel_offset + channel_count, dtype=int
        )

    if port_index < len(mask_list) and mask_list[port_index] is not None:
        local_active = np.where(
            to_numpy(np.asarray(mask_list[port_index])).flatten() == 0
        )[0]
    else:
        local_active = np.arange(channel_count)
    global_active = port_channel_indices[
        local_active[local_active < len(port_channel_indices)]
    ]

    emg_port = None
    raw_channels = None
    if emg_full is not None:
        valid_channels = global_active[global_active < emg_full.shape[0]]
        if len(valid_channels) > 0:
            if full_source_mode:
                emg_port = emg_full[valid_channels, :]
            else:
                emg_port = emg_full[
                    valid_channels,
                    max(0, start_sample) : min(end_sample, emg_full.shape[1]),
                ]
            valid_port_channels = port_channel_indices[
                port_channel_indices < emg_full.shape[0]
            ]
            raw_channels = emg_full[valid_port_channels, :]

    discharge_times = decomposition["discharge_times"]
    pulse_trains = decomposition["pulse_trains"]
    port_discharge = (
        discharge_times[port_index] if port_index < len(discharge_times) else []
    )
    port_sources = pulse_trains[port_index] if port_index < len(pulse_trains) else []
    all_filters = decomposition.get("mu_filters", [])
    port_filters = all_filters[port_index] if port_index < len(all_filters) else None

    timestamp_arrays = ensure_list_of_arrays(port_discharge)
    source_arrays = ensure_list_of_arrays(port_sources)
    all_motor_unit_ids = decomposition.get("motor_unit_ids", [])
    motor_unit_ids = (
        list(all_motor_unit_ids[port_index])
        if port_index < len(all_motor_unit_ids)
        else list(range(len(timestamp_arrays)))
    )
    all_unit_lineage = decomposition.get("unit_lineage", [])
    port_lineage = (
        list(all_unit_lineage[port_index])
        if isinstance(all_unit_lineage, (list, tuple))
        and port_index < len(all_unit_lineage)
        and isinstance(all_unit_lineage[port_index], (list, tuple))
        else []
    )
    filter_arrays = (
        ensure_list_of_arrays(port_filters)
        if port_filters is not None
        else [None] * len(timestamp_arrays)
    )
    full_results = full_port_results.get(port_index, [])

    motor_units = []
    for motor_unit_index in range(len(timestamp_arrays)):
        lineage = (
            port_lineage[motor_unit_index]
            if motor_unit_index < len(port_lineage)
            and isinstance(port_lineage[motor_unit_index], Mapping)
            else {}
        )
        peel_group_id = lineage.get("peel_group_id", motor_unit_index)
        split_parent_id = lineage.get("split_parent_id")
        split_label = lineage.get("split_label")
        try:
            peel_group_id = int(peel_group_id)
        except (TypeError, ValueError):
            peel_group_id = motor_unit_index
        try:
            split_parent_id = (
                int(split_parent_id) if split_parent_id is not None else None
            )
        except (TypeError, ValueError):
            split_parent_id = None
        if split_label not in ("A", "B"):
            split_label = None

        motor_unit_filter = (
            to_numpy(filter_arrays[motor_unit_index])
            if motor_unit_index < len(filter_arrays)
            and filter_arrays[motor_unit_index] is not None
            else None
        )

        if (
            full_source_mode
            and motor_unit_index < len(full_results)
            and full_results[motor_unit_index][0] is not None
        ):
            full_result = full_results[motor_unit_index]
            source = full_result[0]
            absolute_timestamps = full_result[1]
            if len(full_result) > 2 and full_result[2] is not None:
                motor_unit_filter = full_result[2]
        else:
            source = (
                to_numpy(source_arrays[motor_unit_index]).flatten()
                if motor_unit_index < len(source_arrays)
                else np.zeros(1)
            )
            plateau_timestamps = (
                to_numpy(timestamp_arrays[motor_unit_index]).flatten().astype(np.int64)
            )
            absolute_timestamps = (
                plateau_timestamps + start_sample
                if full_source_mode
                else plateau_timestamps
            )

        motor_units.append(
            MotorUnit(
                id=int(motor_unit_ids[motor_unit_index]),
                timestamps=absolute_timestamps,
                source=source,
                port_name=port_name,
                mu_filter=motor_unit_filter,
                peel_group_id=peel_group_id,
                split_parent_id=split_parent_id,
                split_label=split_label,
            )
        )

    electrode_type = (
        electrode_list[port_index] if port_index < len(electrode_list) else None
    )
    grid_config = get_grid_config(electrode_type)

    corrected_positions = None
    rejected_positions: set[tuple[int, int]] = set()
    if grid_config is not None:
        mapping = grid_config.get("muap_mapping", {})
        raw_positions = grid_config["positions"]
        corrected_positions = {}
        for new_index, original_index in enumerate(local_active):
            key = mapping.get(int(original_index), int(original_index))
            position = raw_positions.get(key)
            if position is not None:
                corrected_positions[new_index] = position

        active_indices = set(local_active)
        for original_index in range(channel_count):
            if original_index in active_indices:
                continue
            key = mapping.get(original_index, original_index)
            position = raw_positions.get(key)
            if position is not None:
                rejected_positions.add(position)

    if motor_units:
        properties = property_computer(
            all_timestamps=[motor_unit.timestamps for motor_unit in motor_units],
            all_sources=[motor_unit.source for motor_unit in motor_units],
            emg_port=emg_port,
            grid_positions=(
                corrected_positions
                if corrected_positions is not None
                else (grid_config["positions"] if grid_config else None)
            ),
            grid_shape=grid_config["grid_shape"] if grid_config else None,
            fsamp=sampling_rate,
        )
        for motor_unit, unit_properties in zip(motor_units, properties, strict=True):
            motor_unit.props = unit_properties

    for index in decomposition.get("flagged_mus", {}).get(port_name, []):
        if 0 <= index < len(motor_units):
            motor_units[index].flagged_duplicate = True

    for index in decomposition.get("reviewed_mus", {}).get(port_name, []):
        if 0 <= index < len(motor_units):
            motor_units[index].reviewed = True

    reliability_overrides = decomposition.get("reliability_overrides", {}).get(
        port_name, {}
    )
    for index, value in reliability_overrides.items():
        if 0 <= index < len(motor_units) and motor_units[index].props is not None:
            motor_units[index].props.reliability_override = bool(value)

    migrated_notes = []
    known_notes = set(existing_notes)
    port_notes = decomposition.get("mu_notes", [])
    if port_index < len(port_notes):
        for motor_unit, note in zip(motor_units, port_notes[port_index], strict=False):
            if not isinstance(note, str) or not note.strip():
                continue
            flattened_note = " ".join(note.splitlines()).strip()
            full_note = (
                f"0000-00-00 00:00:00 ({port_name}, MU {motor_unit.id}): "
                f"{flattened_note}"
            )
            if full_note not in known_notes:
                migrated_notes.append(full_note)
                known_notes.add(full_note)

    logger.info(
        "Port '%s': %d MUs, %d spikes%s",
        port_name,
        len(motor_units),
        sum(len(motor_unit.timestamps) for motor_unit in motor_units),
        " (full)" if full_source_mode else "",
    )
    return LoadedEditionPort(
        channel_count=channel_count,
        motor_units=motor_units,
        emg=emg_port,
        raw_channels=raw_channels,
        grid_config=grid_config,
        active_grid_positions=corrected_positions,
        rejected_channel_positions=rejected_positions,
        migrated_notes=migrated_notes,
    )


def build_edition_save_data(state: EditionSaveState) -> dict:
    """Build the versioned decomposition dictionary written by Edition."""
    port_names = list(state.ports)
    discharge_times = []
    pulse_trains = []
    mu_filters = []
    mu_properties = []
    motor_unit_ids = []
    unit_lineage = []
    flagged_mus_per_port = {}
    reviewed_mus_per_port = {}
    reliability_overrides_per_port = {}

    for port_name in port_names:
        motor_units = state.ports[port_name]
        motor_unit_ids.append([motor_unit.id for motor_unit in motor_units])
        unit_lineage.append(
            [
                {
                    "peel_group_id": (
                        motor_unit.peel_group_id
                        if motor_unit.peel_group_id is not None
                        else index
                    ),
                    "split_parent_id": motor_unit.split_parent_id,
                    "split_label": motor_unit.split_label,
                }
                for index, motor_unit in enumerate(motor_units)
            ]
        )
        flagged_mus_per_port[port_name] = [
            index
            for index, motor_unit in enumerate(motor_units)
            if motor_unit.flagged_for_deletion
        ]
        reviewed_mus_per_port[port_name] = [
            index for index, motor_unit in enumerate(motor_units) if motor_unit.reviewed
        ]
        reliability_overrides_per_port[port_name] = {
            index: motor_unit.props.reliability_override
            for index, motor_unit in enumerate(motor_units)
            if motor_unit.props is not None
            and motor_unit.props.reliability_is_overridden
        }

        if state.full_source_mode:
            saved_timestamps = [
                motor_unit.timestamps - state.start_sample for motor_unit in motor_units
            ]
            plateau_length = state.end_sample - state.start_sample
            saved_sources = [
                (
                    motor_unit.source[state.start_sample : state.end_sample]
                    if len(motor_unit.source) > plateau_length
                    else motor_unit.source
                )
                for motor_unit in motor_units
            ]
        else:
            saved_timestamps = [motor_unit.timestamps for motor_unit in motor_units]
            saved_sources = [motor_unit.source for motor_unit in motor_units]

        discharge_times.append(saved_timestamps)
        pulse_trains.append(saved_sources)
        mu_filters.append([motor_unit.mu_filter for motor_unit in motor_units] or None)

        port_properties = []
        for motor_unit in motor_units:
            if motor_unit.props is None:
                port_properties.append({})
                continue
            properties = asdict(motor_unit.props)
            properties.pop("muap_grid", None)
            properties["is_reliable"] = motor_unit.props.is_reliable
            properties["auto_reliable"] = motor_unit.props.auto_reliable
            port_properties.append(properties)
        mu_properties.append(port_properties)

    save_data = {
        "format": GUI_FORMAT,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "ports": port_names,
        "sampling_rate": state.sampling_rate,
        "discharge_times": discharge_times,
        "pulse_trains": pulse_trains,
        "motor_unit_ids": motor_unit_ids,
        "unit_lineage": unit_lineage,
        "mu_filters": mu_filters,
        "skip_filter_recalc": True,
        "mu_properties": mu_properties,
        "flagged_mus": flagged_mus_per_port,
        "reviewed_mus": reviewed_mus_per_port,
        "reliability_overrides": reliability_overrides_per_port,
        "edit_history": state.edit_history,
        "notes": state.notes,
    }

    if state.original_decomposition is not None:
        for key in _PASSTHROUGH_KEYS:
            value = state.original_decomposition.get(key)
            if value is not None and key not in save_data:
                save_data[key] = value

        original_peel_sequence = state.original_decomposition.get("peel_off_sequence")
        if original_peel_sequence is not None:
            save_data["peel_off_sequence"] = _sync_peel_sequence_timestamps(
                original_peel_sequence,
                discharge_times,
            )

        original_filters = state.original_decomposition.get("mu_filters")
        if original_filters is not None:
            save_data["mu_filters_original"] = original_filters

    if state.emg_data:
        save_data["emg_per_port"] = dict(state.emg_data)

    return save_data


def sync_peel_sequence_to_discharge_times(decomposition: dict) -> None:
    """Copy the saved discharge trains into the peel sequence, in place.

    The replay on load shows each unit at its peel sequence timestamps, and
    older files still have the original trains there.
    """
    peel_sequence = decomposition.get("peel_off_sequence")
    if peel_sequence is not None:
        decomposition["peel_off_sequence"] = _sync_peel_sequence_timestamps(
            peel_sequence,
            decomposition["discharge_times"],
        )


def ensure_split_peel_steps(decomposition: dict) -> None:
    """Expand legacy one-step split groups before a full replay.

    Current Edition files store split A and B as consecutive accepted peel
    entries. Early split sessions could retain only one entry for the shared
    parent stage. Full-source replay needs one entry per child so A can be
    peeled before B is estimated.
    """
    peel_sequence = decomposition.get("peel_off_sequence")
    discharge_times = decomposition.get("discharge_times")
    unit_lineage = decomposition.get("unit_lineage")
    if not all(
        isinstance(value, list)
        for value in (peel_sequence, discharge_times, unit_lineage)
    ):
        return

    per_port = bool(peel_sequence) and isinstance(peel_sequence[0], list)
    if per_port:
        rebuilt = []
        for port_index, entries in enumerate(peel_sequence):
            timestamps = (
                discharge_times[port_index]
                if port_index < len(discharge_times)
                and isinstance(discharge_times[port_index], list)
                else []
            )
            lineage = (
                unit_lineage[port_index]
                if port_index < len(unit_lineage)
                and isinstance(unit_lineage[port_index], list)
                else []
            )
            units = _split_replay_units(
                port_index,
                timestamps,
                lineage,
                accepted_offset=0,
            )
            rebuilt.append(_expand_split_peel_entries(entries, units))
        decomposition["peel_off_sequence"] = rebuilt
        return

    units = []
    accepted_offset = 0
    for port_index, timestamps in enumerate(discharge_times):
        port_timestamps = timestamps if isinstance(timestamps, list) else []
        lineage = (
            unit_lineage[port_index]
            if port_index < len(unit_lineage)
            and isinstance(unit_lineage[port_index], list)
            else []
        )
        units.extend(
            _split_replay_units(
                port_index,
                port_timestamps,
                lineage,
                accepted_offset=accepted_offset,
            )
        )
        accepted_offset += len(port_timestamps)
    decomposition["peel_off_sequence"] = _expand_split_peel_entries(
        peel_sequence,
        units,
    )


def _split_replay_units(
    port_index: int,
    timestamps: list,
    lineage: list,
    *,
    accepted_offset: int,
) -> list[dict]:
    """Describe persisted split children in peel-replay coordinates."""
    units = []
    for local_index, unit_timestamps in enumerate(timestamps):
        item = lineage[local_index] if local_index < len(lineage) else None
        if not isinstance(item, Mapping) or item.get("split_parent_id") is None:
            continue
        label = item.get("split_label")
        if label not in ("A", "B"):
            continue
        units.append(
            {
                "accepted_unit_idx": accepted_offset + local_index,
                "group": (
                    port_index,
                    item.get("peel_group_id", local_index),
                    item["split_parent_id"],
                ),
                "label": label,
                "timestamps": np.asarray(unit_timestamps, dtype=np.int64).copy(),
            }
        )
    return units


def _expand_split_peel_entries(entries: list, units: list[dict]) -> list:
    """Return ``entries`` with every legacy split group expanded A then B."""
    groups: dict[tuple, list[dict]] = {}
    for unit in units:
        groups.setdefault(unit["group"], []).append(unit)

    replacements: dict[int, list[dict]] = {}
    replaced_positions: set[int] = set()
    for children in groups.values():
        if len(children) < 2:
            continue
        ordered = sorted(
            children,
            key=lambda child: (child["label"] not in ("A", "B"), child["label"]),
        )
        child_indices = {child["accepted_unit_idx"] for child in ordered}
        positions = [
            position
            for position, entry in enumerate(entries)
            if isinstance(entry, Mapping)
            and entry.get("accepted_unit_idx") in child_indices
        ]
        present_indices = {
            int(entries[position]["accepted_unit_idx"]) for position in positions
        }
        if child_indices <= present_indices or not positions:
            continue

        insert_at = min(positions)
        template = dict(entries[insert_at])
        replacements[insert_at] = [
            {
                **template,
                "accepted_unit_idx": child["accepted_unit_idx"],
                "timestamps": child["timestamps"],
            }
            for child in ordered
        ]
        replaced_positions.update(positions)

    if not replacements:
        return entries

    rebuilt = []
    for position, entry in enumerate(entries):
        if position in replacements:
            rebuilt.extend(replacements[position])
        elif position not in replaced_positions:
            rebuilt.append(entry)
    return rebuilt


def _sync_peel_sequence_timestamps(
    peel_sequence,
    discharge_times: list[list[np.ndarray]],
):
    """Copy a peel sequence with accepted steps using current discharge trains."""
    per_port = (
        isinstance(peel_sequence, list)
        and bool(peel_sequence)
        and isinstance(peel_sequence[0], list)
    )

    def updated_entry(entry, timestamps_by_index):
        if not isinstance(entry, Mapping):
            return entry
        updated = dict(entry)
        unit_index = entry.get("accepted_unit_idx")
        if isinstance(unit_index, (int, np.integer)) and 0 <= int(unit_index) < len(
            timestamps_by_index
        ):
            updated["timestamps"] = np.asarray(
                timestamps_by_index[int(unit_index)], dtype=np.int64
            ).copy()
        return updated

    if per_port:
        return [
            [
                updated_entry(
                    entry,
                    discharge_times[port_index]
                    if port_index < len(discharge_times)
                    else [],
                )
                for entry in entries
            ]
            for port_index, entries in enumerate(peel_sequence)
        ]

    flattened_timestamps = [
        timestamps
        for port_timestamps in discharge_times
        for timestamps in port_timestamps
    ]
    return [updated_entry(entry, flattened_timestamps) for entry in peel_sequence]
