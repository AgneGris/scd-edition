"""Loader-preset selection and the channel count it feeds."""

import os

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scd_app.io.data_loader import (
    can_read_field,
    format_matches_extension,
)

GENERIC_H5 = {"name": ".hdf5", "format": "h5", "fields": {"emg": {"path": "emg/data"}}}
SIM_H5 = {
    "name": "Simulated grid (.hdf5)",
    "format": "h5",
    "fields": {"emg": {"path": "emg", "orientation": "samples_first"}},
}
MAT = {"name": ".mat", "format": "mat", "fields": {"emg": {"path": "emg"}}}


def _sim_file(tmp_path, name="sim.hdf5", n_channels=320):
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_dataset("emg", data=np.zeros((512, n_channels), dtype=np.float64))
        f.create_dataset("noise", data=np.zeros((512, n_channels), dtype=np.float64))
    return path


def _generic_file(tmp_path, name="generic.hdf5"):
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_group("emg").create_dataset("data", data=np.zeros((64, 512)))
    return path


def test_can_read_field_distinguishes_presets_sharing_an_extension(tmp_path):
    sim = _sim_file(tmp_path)
    generic = _generic_file(tmp_path)

    assert can_read_field(sim, SIM_H5) is True
    assert can_read_field(sim, GENERIC_H5) is False
    assert can_read_field(generic, GENERIC_H5) is True
    assert can_read_field(generic, SIM_H5) is False


def test_can_read_field_reports_unknown_when_there_is_no_cheap_probe(tmp_path):
    scipy_mat = tmp_path / "recording.mat"
    scipy_mat.write_bytes(b"MATLAB 5.0 MAT-file, not HDF5")

    # A v5/v7 .mat is scipy's to read; h5py cannot answer, so neither do we.
    assert can_read_field(scipy_mat, MAT) is None
    # ...but a non-HDF5 file under an "h5" layout is genuinely unreadable.
    assert can_read_field(scipy_mat, GENERIC_H5) is False


def test_can_read_field_rejects_a_path_that_lands_on_a_group(tmp_path):
    generic = _generic_file(tmp_path)
    group_layout = {"name": "g", "format": "h5", "fields": {"emg": {"path": "emg"}}}

    assert can_read_field(generic, group_layout) is False


def test_can_read_field_accepts_a_fallback_key(tmp_path):
    sim = _sim_file(tmp_path)
    layout = {
        "name": "fb",
        "format": "h5",
        "fields": {"emg": {"path": "missing", "fallback_keys": ["emg"]}},
    }

    assert can_read_field(sim, layout) is True


def test_format_matches_extension():
    assert format_matches_extension("h5", ".hdf5")
    assert format_matches_extension("h5", ".H5")
    assert not format_matches_extension("mat", ".hdf5")
    assert not format_matches_extension("otb4", ".hdf5")
    # An unrecognised format is not evidence of a mismatch.
    assert format_matches_extension("something_new", ".hdf5")


def _config_tab():
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    return app, ConfigTab()


def test_browsing_keeps_a_loader_that_can_read_the_file(tmp_path):
    """The bug: browsing a .hdf5 file used to snap the combo to the generic
    ".hdf5" preset, whose emg/data path is absent from simulated files."""
    app, tab = _config_tab()
    try:
        if "Simulated grid (.hdf5)" not in tab._loader_layouts:
            pytest.skip("simulated-grid preset not installed")

        tab._select_loader("Simulated grid (.hdf5)")
        tab._auto_select_loader(_sim_file(tmp_path))

        assert tab.loader_combo.currentText() == "Simulated grid (.hdf5)"
    finally:
        tab.close()
        app.processEvents()


def test_browsing_switches_to_the_preset_that_resolves_the_file(tmp_path):
    app, tab = _config_tab()
    try:
        if "Simulated grid (.hdf5)" not in tab._loader_layouts:
            pytest.skip("simulated-grid preset not installed")

        tab._select_loader(".hdf5")
        tab._auto_select_loader(_sim_file(tmp_path))
        assert tab.loader_combo.currentText() == "Simulated grid (.hdf5)"

        tab._auto_select_loader(_generic_file(tmp_path))
        assert tab.loader_combo.currentText() == ".hdf5"
    finally:
        tab.close()
        app.processEvents()


def test_browsing_falls_back_to_the_extension_when_no_preset_resolves(tmp_path):
    app, tab = _config_tab()
    try:
        unreadable = tmp_path / "unknown.hdf5"
        with h5py.File(unreadable, "w") as f:
            f.create_dataset("mystery", data=np.zeros((8, 8)))

        tab._select_loader(".mat")
        tab._auto_select_loader(unreadable)

        assert tab.loader_combo.currentText() == ".hdf5"
    finally:
        tab.close()
        app.processEvents()


def test_unreadable_file_is_reported_as_unknown_not_as_too_few_channels(tmp_path):
    """A failed probe must not fall back to a 256-channel placeholder and then
    fail a 320-channel grid with "exceeds available channels"."""
    app, tab = _config_tab()
    try:
        tab.emg_path = _sim_file(tmp_path)
        tab.emg_paths = [tab.emg_path]
        tab._select_loader(".hdf5")  # wrong preset: emg/data is absent
        tab._refresh_file_metadata()
        tab._update_file_info()

        assert tab._channel_count_known is False
        assert "Load failed" in tab.file_info_label.text()

        tab._add_grid()
        card = tab.grid_cards[0]
        card.start_spin.setValue(0)
        card.end_spin.setValue(320)

        is_valid, warnings = tab._validate_configuration()

        assert not is_valid
        assert any("Channel count unknown" in w for w in warnings)
        assert not any("exceed file" in w for w in warnings)
    finally:
        tab.close()
        app.processEvents()


def test_readable_file_reports_its_channel_count(tmp_path):
    app, tab = _config_tab()
    try:
        if "Simulated grid (.hdf5)" not in tab._loader_layouts:
            pytest.skip("simulated-grid preset not installed")

        tab.emg_path = _sim_file(tmp_path)
        tab.emg_paths = [tab.emg_path]
        tab._select_loader("Simulated grid (.hdf5)")
        tab._refresh_file_metadata()
        tab._update_file_info()

        assert tab._channel_count_known is True
        assert tab.max_channels == 320
        assert "320 channels" in tab.file_info_label.text()
    finally:
        tab.close()
        app.processEvents()
