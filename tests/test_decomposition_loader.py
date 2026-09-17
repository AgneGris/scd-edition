import os
import pickle

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scd_app.io.decomposition_loader import (
    GUI_FORMAT,
    UPSTREAM_SCD_FORMAT,
    UnsupportedDecompositionFormat,
    convert_scd_output,
    detect_decomposition_format,
    load_decomposition_file,
)


def _raw_scd_result():
    source_1 = np.zeros((200, 1), dtype=np.float32)
    source_1[[20, 80, 140], 0] = 1
    source_2 = np.zeros((200, 1), dtype=np.float32)
    source_2[[40, 100, 160], 0] = 1
    return {
        "silhouettes": [np.float32(0.91), np.float32(0.87)],
        "timestamps": [np.array([20, 80, 140]), np.array([40, 100, 160])],
        "source": [source_1, source_2],
        "filters": [np.ones((6, 1)), np.full((6, 1), 2)],
        "peel_off_sequence": [
            {"timestamps": np.array([20, 80, 140]), "accepted_unit_idx": 0},
            {"timestamps": np.array([40, 100, 160]), "accepted_unit_idx": 1},
        ],
        "preprocessing_config": {
            "sampling_frequency": 1000,
            "extension_factor": 2,
        },
        "w_mat": np.eye(6, dtype=np.float32),
    }


def test_detects_native_and_upstream_formats():
    native = {"ports": ["Grid"], "discharge_times": [[]], "pulse_trains": [[]]}
    assert detect_decomposition_format(native) == GUI_FORMAT
    assert detect_decomposition_format(_raw_scd_result()) == UPSTREAM_SCD_FORMAT

    with pytest.raises(UnsupportedDecompositionFormat, match="Found keys"):
        detect_decomposition_format({"unknown": []})


def test_converts_upstream_scd_output_and_preserves_provenance(tmp_path):
    source_path = tmp_path / "sub-05_task-pull10_muscle-FD_raw_run-00_scddict.pkl"
    commit = "910f36d1274f832e74992fae266cf91d46b2d94d"
    with source_path.open("wb") as handle:
        pickle.dump(_raw_scd_result(), handle)
    source_path.with_name(
        "sub-05_task-pull10_muscle-FD_raw_run-00_scdcommit.txt"
    ).write_text(commit, encoding="utf-8")

    converted = load_decomposition_file(source_path)

    assert converted["ports"] == ["FD"]
    assert converted["sampling_rate"] == 1000
    assert converted["plateau_coords"] == [0, 200]
    assert converted["chans_per_electrode"] == [3]
    assert len(converted["discharge_times"][0]) == 2
    assert len(converted["pulse_trains"][0]) == 2
    assert converted["pulse_trains"][0][0].shape == (200,)
    assert converted["mu_filters"][0][0].shape == (6,)
    assert converted["import_provenance"]["scd_commit"] == commit
    assert converted["scd_metadata"]["silhouettes"] == pytest.approx([0.91, 0.87])


def test_torch_storage_is_remapped_to_cpu(tmp_path, monkeypatch):
    raw = _raw_scd_result()
    raw["source"] = [torch.as_tensor(source) for source in raw["source"]]
    source_path = tmp_path / "cuda-portable_scddict.pkl"
    with source_path.open("wb") as handle:
        pickle.dump(raw, handle)

    map_locations = []
    original_torch_load = torch.load

    def recording_torch_load(*args, **kwargs):
        map_locations.append(kwargs.get("map_location"))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_torch_load)
    converted = load_decomposition_file(source_path)

    assert map_locations
    assert set(map_locations) == {"cpu"}
    assert isinstance(converted["pulse_trains"][0][0], np.ndarray)


def test_rejects_mismatched_scd_unit_counts():
    raw = _raw_scd_result()
    raw["source"] = raw["source"][:1]
    with pytest.raises(UnsupportedDecompositionFormat, match="source entries"):
        convert_scd_output(raw)


def test_edition_loads_and_edits_upstream_scd_output(tmp_path):
    from PySide6.QtWidgets import QApplication

    from scd_app.core.mu_model import EditMode
    from scd_app.gui.tabs.edition_tab import EditionTab

    source_path = tmp_path / "subject_muscle-FD_scddict.pkl"
    with source_path.open("wb") as handle:
        pickle.dump(_raw_scd_result(), handle)

    app = QApplication.instance() or QApplication([])
    tab = EditionTab()
    tab.load_from_path(source_path)

    assert list(tab._ports) == ["FD"]
    assert len(tab._ports["FD"]) == 2
    motor_unit = tab._ports["FD"][0]
    original_timestamps = motor_unit.timestamps.copy()

    tab._set_mode(EditMode.DELETE)
    tab._handle_delete_click(int(original_timestamps[0]))
    assert len(motor_unit.timestamps) == len(original_timestamps) - 1
    tab._undo()
    np.testing.assert_array_equal(motor_unit.timestamps, original_timestamps)

    saved = tab._build_save_dict()
    assert saved["import_provenance"]["format"] == UPSTREAM_SCD_FORMAT
    assert saved["skip_filter_recalc"] is True

    tab.close()
    app.processEvents()
