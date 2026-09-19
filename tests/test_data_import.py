import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scd_app.io.data_inspector import inspect_recording
from scd_app.io.data_loader import load_field, load_layout, load_metadata
from scd_app.io.portable_format import write_portable_recording

PRESETS = (
    Path(__file__).parents[1] / "src" / "scd_app" / "resources" / "loaders_configs"
)


def test_generic_npy_preset_loads_without_pickle_and_fixes_orientation(tmp_path):
    path = tmp_path / "recording.npy"
    source = np.arange(30, dtype=np.float32).reshape(3, 10)
    np.save(path, source)

    loaded = load_field(path, load_layout(PRESETS / "loader_npy.yaml"), "emg")

    assert loaded.shape == (10, 3)
    np.testing.assert_array_equal(loaded.numpy(), source.T)


def test_generic_csv_preset_accepts_a_header(tmp_path):
    path = tmp_path / "recording.csv"
    path.write_text(
        "# exported by acquisition software\nch1,ch2\n1,2\n3,4\n5,6\n",
        encoding="utf-8",
    )

    loaded = load_field(path, load_layout(PRESETS / "loader_csv.yaml"), "emg")

    assert loaded.shape == (3, 2)
    np.testing.assert_array_equal(
        loaded.numpy(), np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    )


def test_generic_text_preset_uses_whitespace(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("1 2\n3 4\n", encoding="utf-8")

    loaded = load_field(path, load_layout(PRESETS / "loader_csv.yaml"), "emg")

    np.testing.assert_array_equal(
        loaded.numpy(), np.array([[1, 2], [3, 4]], dtype=np.float32)
    )


def test_hdf5_inspector_discovers_nested_arrays_and_sampling_rate(tmp_path):
    path = tmp_path / "unknown.h5"
    with h5py.File(path, "w") as file:
        file.attrs["sampling_rate_hz"] = 4096
        file.create_dataset("recording/emg_signal", shape=(100, 64), dtype="f4")
        file.create_dataset("recording/force", shape=(100,), dtype="f4")

    inspection = inspect_recording(path)

    assert inspection.sampling_rate_hz == 4096
    assert inspection.suggested_array is not None
    assert inspection.suggested_array.path == "recording/emg_signal"
    assert inspection.suggested_array.shape == (100, 64)


def test_mat_inspector_uses_dot_paths_for_nested_structs(tmp_path):
    path = tmp_path / "unknown.mat"
    savemat(
        path,
        {
            "recording": {"signal": np.zeros((32, 200), dtype=np.float32)},
            "fsamp": 2048,
        },
    )

    inspection = inspect_recording(path)

    assert inspection.sampling_rate_hz == 2048
    assert inspection.suggested_array is not None
    assert inspection.suggested_array.path == "recording.signal"
    assert inspection.suggested_array.shape == (32, 200)
    layout = load_layout(PRESETS / "loader_mat.yaml")
    layout["fields"]["emg"]["path"] = inspection.suggested_array.path
    assert load_field(path, layout, "emg").shape == (200, 32)


def test_portable_recording_round_trip_and_metadata(tmp_path):
    path = tmp_path / "portable.h5"
    emg = np.arange(40, dtype=np.float32).reshape(10, 4)
    aux = np.linspace(0, 1, 10)
    timestamps = np.arange(10) / 2000
    write_portable_recording(
        path,
        emg,
        2000,
        aux=aux,
        timestamps=timestamps,
        source_format="vendor-x",
    )
    layout = load_layout(PRESETS / "loader_scd_h5.yaml")

    np.testing.assert_array_equal(load_field(path, layout, "emg").numpy(), emg)
    assert load_field(path, layout, "aux").shape == (10, 1)
    metadata = load_metadata(path, layout)
    assert metadata["sampling_frequency"] == 2000
    assert metadata["n_samples"] == 10
    assert metadata["emg_channel_count"] == 4

    with pytest.raises(FileExistsError):
        write_portable_recording(path, emg, 2000)


def test_portable_metadata_tracks_decimation(tmp_path):
    path = tmp_path / "portable.h5"
    write_portable_recording(path, np.zeros((11, 2)), 2000)
    layout = load_layout(PRESETS / "loader_scd_h5.yaml")
    layout["decimate"] = 2

    metadata = load_metadata(path, layout)

    assert metadata["native_sampling_frequency"] == 2000
    assert metadata["sampling_frequency"] == 1000
    assert metadata["native_n_samples"] == 11
    assert metadata["n_samples"] == 6


def test_config_tab_applies_an_inspected_hdf5_selection(tmp_path):
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import ConfigTab

    path = tmp_path / "unknown.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("vendor/raw", data=np.zeros((100, 8)))

    app = QApplication.instance() or QApplication([])
    tab = ConfigTab()
    try:
        assert ".npy" in tab._loader_layouts
        assert ".csv" in tab._loader_layouts
        tab._apply_import_selection(
            path,
            field_path="vendor/raw",
            orientation="samples_first",
            sampling_rate=4000,
        )

        layout = tab._get_layout_with_overrides()
        assert layout is not None
        assert layout["format"] == "h5"
        assert layout["fields"]["emg"]["path"] == "vendor/raw"
        assert layout["fields"]["emg"]["orientation"] == "samples_first"
        assert tab._channel_count_known
        assert tab.max_channels == 8
        assert tab.fsamp_edit.text() == "4000"

        npy_path = tmp_path / "unusual.npy"
        np.save(npy_path, np.zeros((8, 100)))
        tab._apply_import_selection(
            npy_path,
            field_path="array",
            orientation="samples_first",
            sampling_rate=1000,
        )
        layout = tab._get_layout_with_overrides()
        assert layout is not None
        assert layout["format"] == "npy"
        assert layout["fields"]["emg"]["orientation"] == "samples_first"
        assert load_field(npy_path, layout, "emg").shape == (8, 100)
    finally:
        tab.close()
        app.processEvents()


def test_import_dialog_does_not_guess_a_missing_sampling_rate(tmp_path):
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.widgets.import_data_dialog import ImportDataDialog

    path = tmp_path / "recording.npy"
    np.save(path, np.zeros((100, 8)))
    inspection = inspect_recording(path)
    app = QApplication.instance() or QApplication([])
    dialog = ImportDataDialog(inspection)
    try:
        assert dialog.sampling_rate_edit.text() == ""
        assert dialog.sampling_rate_edit.placeholderText() == "required"
    finally:
        dialog.close()
        app.processEvents()
