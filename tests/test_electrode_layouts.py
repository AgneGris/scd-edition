"""Characterization tests for the supported electrode layouts."""

import pytest

from scd_app.core.electrode_layouts import ELECTRODE_GRIDS, get_grid_config


@pytest.mark.parametrize("name", sorted(ELECTRODE_GRIDS))
def test_electrode_layout_mapping_is_complete_and_in_bounds(name):
    config = ELECTRODE_GRIDS[name]
    rows, columns = config["grid_shape"]
    positions = config["positions"]

    assert len(positions) == config["n_channels"]
    assert len(config["muap_mapping"]) == config["n_channels"]
    assert set(config["muap_mapping"].values()) == set(positions)
    assert len(set(positions.values())) == config["n_channels"]
    assert all(
        0 <= row < rows and 0 <= column < columns for row, column in positions.values()
    )


@pytest.mark.parametrize(
    ("electrode_type", "canonical_name"),
    [
        ("gr04mm1305", "GR04MM1305"),
        ("recording_GR08MM1305_port1", "GR08MM1305"),
        ("hd08mm1606, channels 17-96", "HD08MM1606, CHANNELS 17-96"),
        ("ultrahd 4x4", "ULTRAHD 4X4"),
    ],
)
def test_get_grid_config_accepts_case_insensitive_metadata(
    electrode_type, canonical_name
):
    assert get_grid_config(electrode_type) is ELECTRODE_GRIDS[canonical_name]


@pytest.mark.parametrize("electrode_type", [None, "", "unsupported-array"])
def test_get_grid_config_returns_none_for_unknown_layout(electrode_type):
    assert get_grid_config(electrode_type) is None
