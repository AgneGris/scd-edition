"""Characterization tests for Edition's persisted session format."""

import numpy as np

from scd_app.core.mu_model import MotorUnit
from scd_app.core.mu_properties import MUProperties
from scd_app.io.decomposition_loader import CURRENT_SCHEMA_VERSION, GUI_FORMAT
from scd_app.io.edition_session import (
    EditionSaveState,
    build_edition_save_data,
    ensure_list_of_arrays,
    load_edition_port,
    normalise_aux_channels,
)


def test_save_dict_preserves_full_source_and_provenance_contract():
    properties = MUProperties(n_spikes=2, reliability_override=False)
    properties.muap_grid = np.ones((1, 1, 3))
    properties.duplicate_candidates = {2: 0.91}
    unit = MotorUnit(
        id=0,
        timestamps=np.array([110, 125], dtype=np.int64),
        source=np.arange(200, dtype=float),
        port_name="Grid 1",
        mu_filter=np.array([0.25, 0.75]),
        flagged_duplicate=True,
        reviewed=True,
        props=properties,
    )

    original_data = np.arange(20, dtype=float).reshape(2, 10)
    original_filters = [[np.array([9.0, 8.0])]]
    state = EditionSaveState(
        ports={"Grid 1": [unit]},
        sampling_rate=2048.0,
        start_sample=100,
        end_sample=150,
        full_source_mode=True,
        edit_history=[{"event": "delete_spike"}],
        notes=["reviewed"],
        original_decomposition={
            "data": original_data,
            "aux_channels": [{"name": "force"}],
            "peel_off_sequence": [[0]],
            "mu_filters": original_filters,
            "import_provenance": {"format": "upstream"},
        },
        emg_data={"Grid 1": np.ones((2, 50))},
    )

    saved = build_edition_save_data(state)

    assert saved["format"] == GUI_FORMAT
    assert saved["schema_version"] == CURRENT_SCHEMA_VERSION
    assert saved["ports"] == ["Grid 1"]
    assert saved["sampling_rate"] == 2048.0
    assert saved["skip_filter_recalc"] is True
    np.testing.assert_array_equal(saved["discharge_times"][0][0], [10, 25])
    np.testing.assert_array_equal(saved["pulse_trains"][0][0], np.arange(100, 150))
    np.testing.assert_array_equal(saved["mu_filters"][0][0], [0.25, 0.75])
    assert saved["flagged_mus"] == {"Grid 1": [0]}
    assert saved["reviewed_mus"] == {"Grid 1": [0]}
    assert saved["reliability_overrides"] == {"Grid 1": {0: False}}
    assert "muap_grid" not in saved["mu_properties"][0][0]
    assert "duplicate_candidates" not in saved["mu_properties"][0][0]
    assert saved["edit_history"] == [{"event": "delete_spike"}]
    assert saved["notes"] == ["reviewed"]
    assert saved["data"] is original_data
    assert saved["mu_filters_original"] is original_filters
    assert saved["peel_off_sequence"] == [[0]]
    assert saved["import_provenance"] == {"format": "upstream"}
    assert set(saved["emg_per_port"]) == {"Grid 1"}


def test_list_of_arrays_normalisation_contract():
    assert ensure_list_of_arrays(None) == []
    assert ensure_list_of_arrays(np.array([])) == []

    one_dimensional = ensure_list_of_arrays(np.array([1, 2, 3]))
    assert len(one_dimensional) == 1
    np.testing.assert_array_equal(one_dimensional[0], [1, 2, 3])

    two_dimensional = ensure_list_of_arrays(np.array([[1, 2], [3, 4]]))
    assert len(two_dimensional) == 2
    np.testing.assert_array_equal(two_dimensional[0], [1, 2])
    np.testing.assert_array_equal(two_dimensional[1], [3, 4])

    scalar_list = ensure_list_of_arrays([1, 2, 3])
    assert len(scalar_list) == 1
    np.testing.assert_array_equal(scalar_list[0], [1, 2, 3])


def test_aux_channel_normalisation_upgrades_legacy_metadata_and_mvc():
    channels = [
        {
            "name": "force",
            "unit": "mV",
            "meta": {"mvc": 0.125, "name": "nested name"},
        },
        {"name": "torque", "unit": "Nm"},
    ]

    result = normalise_aux_channels(
        channels,
        acquisition_format=None,
        configured_channels=[{"name": "torque", "unit": "Nm", "mvc": 42.0}],
    )

    assert result is not channels
    assert result[0] is channels[0]
    assert result[0]["name"] == "force"
    assert result[0]["mvc"] == 125.0
    assert "meta" not in result[0]
    assert result[1]["mvc"] == 42.0


def test_aux_channel_mvc_in_native_volts_is_not_rescaled():
    channels = [{"name": "force", "unit": "V", "mvc": 0.125}]

    normalise_aux_channels(channels, acquisition_format="otb4")

    assert channels[0]["mvc"] == 0.125


def test_load_port_restores_channels_quality_flags_and_legacy_notes():
    emg = np.arange(60, dtype=float).reshape(3, 20)
    decomposition = {
        "chans_per_electrode": [3],
        "channel_indices": [np.array([0, 1, 2])],
        "emg_mask": [np.array([0, 1, 0])],
        "electrodes": ["unsupported"],
        "discharge_times": [[np.array([1, 4])]],
        "pulse_trains": [[np.arange(5, dtype=float)]],
        "mu_filters": [[np.array([0.2, 0.8])]],
        "flagged_mus": {"Grid 1": [0]},
        "reviewed_mus": {"Grid 1": [0]},
        "reliability_overrides": {"Grid 1": {0: False}},
        "mu_notes": [["legacy\nnote"]],
    }

    loaded = load_edition_port(
        port_index=0,
        port_name="Grid 1",
        decomposition=decomposition,
        emg_full=emg,
        start_sample=10,
        end_sample=15,
        full_port_results={},
        channel_offset=0,
        full_source_mode=False,
        sampling_rate=1000.0,
        property_computer=lambda **_kwargs: [MUProperties(n_spikes=2)],
    )

    assert loaded.channel_count == 3
    np.testing.assert_array_equal(loaded.emg, emg[[0, 2], 10:15])
    np.testing.assert_array_equal(loaded.raw_channels, emg)
    assert loaded.grid_config is None
    assert loaded.rejected_channel_positions == set()
    assert len(loaded.motor_units) == 1
    motor_unit = loaded.motor_units[0]
    np.testing.assert_array_equal(motor_unit.timestamps, [1, 4])
    np.testing.assert_array_equal(motor_unit.source, np.arange(5))
    np.testing.assert_array_equal(motor_unit.mu_filter, [0.2, 0.8])
    assert motor_unit.flagged_duplicate is True
    assert motor_unit.reviewed is True
    assert motor_unit.props.reliability_override is False
    assert loaded.migrated_notes == ["0000-00-00 00:00:00 (Grid 1, MU 0): legacy note"]


def test_load_port_defaults_to_unreviewed_for_older_files():
    loaded = load_edition_port(
        port_index=0,
        port_name="Grid 1",
        decomposition={
            "chans_per_electrode": [1],
            "discharge_times": [[np.array([1, 4])]],
            "pulse_trains": [[np.arange(5, dtype=float)]],
        },
        emg_full=None,
        start_sample=0,
        end_sample=5,
        full_port_results={},
        channel_offset=0,
        full_source_mode=False,
        sampling_rate=1000.0,
        property_computer=lambda **_kwargs: [MUProperties(n_spikes=2)],
    )

    assert loaded.motor_units[0].reviewed is False
