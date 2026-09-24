import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scd_app.io.decomposition_loader import (
    CURRENT_SCHEMA_VERSION,
    GUI_FORMAT,
    UPSTREAM_SCD_FORMAT,
    UnsupportedDecompositionFormat,
    convert_scd_output,
    detect_decomposition_format,
    load_decomposition_file,
    migrate_and_validate_decomposition,
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


def _native_result():
    return {
        "ports": ["Grid 1"],
        "sampling_rate": 1000,
        "discharge_times": [[np.array([2, 6])]],
        "pulse_trains": [[np.arange(10, dtype=float)]],
    }


def test_migrates_legacy_native_session_to_versioned_schema():
    legacy = _native_result()

    migrated = migrate_and_validate_decomposition(legacy)

    assert "schema_version" not in legacy
    assert migrated["format"] == GUI_FORMAT
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["sampling_rate"] == 1000.0
    assert migrated["motor_unit_ids"] == [[0]]
    np.testing.assert_array_equal(migrated["discharge_times"][0][0], [2, 6])


def test_preserves_sparse_motor_unit_ids_per_port():
    native = _native_result()
    native["discharge_times"][0].append(np.array([1, 8]))
    native["pulse_trains"][0].append(np.arange(10, dtype=float))
    native["motor_unit_ids"] = [[0, 3]]

    migrated = migrate_and_validate_decomposition(native)

    assert migrated["motor_unit_ids"] == [[0, 3]]


@pytest.mark.parametrize(
    ("motor_unit_ids", "message"),
    [
        ([[0, 1]], "motor units but 2 motor_unit_ids"),
        ([[2.0]], "non-negative integer"),
        ([[-1]], "non-negative integer"),
    ],
)
def test_rejects_invalid_motor_unit_ids(motor_unit_ids, message):
    malformed = _native_result()
    malformed["motor_unit_ids"] = motor_unit_ids

    with pytest.raises(UnsupportedDecompositionFormat, match=message):
        migrate_and_validate_decomposition(malformed)


def test_rejects_duplicate_motor_unit_ids_within_a_port():
    malformed = _native_result()
    malformed["discharge_times"][0].append(np.array([1, 8]))
    malformed["pulse_trains"][0].append(np.arange(10, dtype=float))
    malformed["motor_unit_ids"] = [[1, 1]]

    with pytest.raises(UnsupportedDecompositionFormat, match="must be unique"):
        migrate_and_validate_decomposition(malformed)


def test_rejects_native_session_with_misaligned_port_data():
    malformed = _native_result()
    malformed["ports"].append("Grid 2")

    with pytest.raises(UnsupportedDecompositionFormat, match="port entries"):
        migrate_and_validate_decomposition(malformed)


def test_rejects_session_from_a_newer_schema():
    future = _native_result()
    future["schema_version"] = CURRENT_SCHEMA_VERSION + 1

    with pytest.raises(UnsupportedDecompositionFormat, match="Please update"):
        migrate_and_validate_decomposition(future)


def test_migrates_legacy_absolute_timestamps_to_plateau_local():
    legacy = _native_result()
    legacy["plateau_coords"] = [100, 110]
    legacy["discharge_times"] = [[np.array([102, 106])]]

    migrated = migrate_and_validate_decomposition(legacy)

    np.testing.assert_array_equal(migrated["discharge_times"][0][0], [2, 6])


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
    assert converted["format"] == GUI_FORMAT
    assert converted["schema_version"] == CURRENT_SCHEMA_VERSION
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
    assert saved["format"] == GUI_FORMAT
    assert saved["schema_version"] == CURRENT_SCHEMA_VERSION
    assert saved["import_provenance"]["format"] == UPSTREAM_SCD_FORMAT
    assert saved["skip_filter_recalc"] is True

    tab.close()
    app.processEvents()


def _raw_scd_result_with_signal():
    """Raw SCD output as written by train(..., save_data=True) (scd >= 0.2.3)."""
    raw = _raw_scd_result()
    # 3 channels x 300 samples as loaded; SCD decomposed samples 50..250
    # (start_time 0.05 s at 1 kHz, 200 samples of source) with channel 1 rejected.
    raw["data"] = np.arange(900, dtype=np.float32).reshape(3, 300)
    raw["preprocessing_config"].update(
        {"bad_channels": [1], "start_time": 0.05, "end_time": 0.25}
    )
    return raw


def test_converts_signal_rejection_mask_and_window_from_upstream_output():
    from scd_app.core.filter_recalculation import (
        supports_filter_recalculation,
        supports_full_source_computation,
    )

    converted = convert_scd_output(_raw_scd_result_with_signal())

    assert converted["data"].shape == (3, 300)
    assert converted["emg_mask"] == [[0, 1, 0]]
    assert converted["plateau_coords"] == [50, 250]
    assert supports_full_source_computation(converted) == (True, "")
    assert supports_filter_recalculation(converted) == (True, "")


def test_upstream_signal_stored_time_major_is_transposed():
    raw = _raw_scd_result_with_signal()
    raw["data"] = raw["data"].T.copy()  # (samples, channels)

    converted = convert_scd_output(raw)

    assert converted["data"].shape == (3, 300)


def test_upstream_output_without_signal_omits_data_key():
    converted = convert_scd_output(_raw_scd_result())

    assert "data" not in converted
    assert converted["emg_mask"] == [[0, 0, 0]]
    assert converted["plateau_coords"] == [0, 200]


def test_rejects_upstream_signal_with_wrong_channel_count():
    raw = _raw_scd_result_with_signal()
    raw["data"] = np.zeros((4, 300), dtype=np.float32)

    with pytest.raises(UnsupportedDecompositionFormat, match="4 channels"):
        convert_scd_output(raw)


def test_rejects_upstream_bad_channel_outside_signal():
    raw = _raw_scd_result_with_signal()
    raw["preprocessing_config"]["bad_channels"] = [3]

    with pytest.raises(UnsupportedDecompositionFormat, match="outside"):
        convert_scd_output(raw)


def test_redetection_ignores_filter_transient_inside_edge_mask():
    from scd_app.core.filter_recalculation import (
        _extract_timestamps,
        _get_scd_modules,
    )

    fn = _get_scd_modules()
    rng = np.random.default_rng(0)
    # Noise floor with small peaks, real spikes at ~5, and a band-pass
    # transient at the start of the recording that dwarfs them.
    source = torch.from_numpy(rng.normal(0.0, 0.3, 2000).astype(np.float32))
    spikes = list(range(100, 2000, 100))
    source[spikes] = torch.from_numpy(
        rng.uniform(4.5, 5.5, len(spikes)).astype(np.float32)
    )
    source[1] = 60.0

    unmasked = _extract_timestamps(source, fn, min_peak_sep=10, edge_mask=0)
    masked = _extract_timestamps(source, fn, min_peak_sep=10, edge_mask=50)

    assert 1 in unmasked
    assert 1 not in masked
    assert set(spikes) <= set(masked.tolist())


def test_edge_mask_samples_reads_snapshot_and_tolerates_missing_values():
    from scd_app.core.filter_recalculation import _edge_mask_samples

    assert _edge_mask_samples({"edge_mask_size": 200}) == 200
    assert _edge_mask_samples({}) == 0
    assert _edge_mask_samples({"edge_mask_size": None}) == 0
    assert _edge_mask_samples({"edge_mask_size": "bad"}) == 0


_SWARM_TEST_DATA = (
    Path(__file__).resolve().parents[2]
    / "swarm-contrastive-decomposition"
    / "data"
    / "input"
    / "emg.mat"
)


@pytest.mark.slow
def test_swarm_output_replays_to_its_own_timestamps(tmp_path):
    """Decompose with swarm-contrastive-decomposition, load here, replay.

    The regression this guards: without the rejection mask the replay whitens
    the real rejected channel with a w_mat computed on noise, and the
    re-detected spikes bear no relation to the saved ones.
    """
    import scd

    if tuple(int(p) for p in scd.__version__.split(".")[:3]) < (0, 2, 3):
        pytest.skip("needs swarm-contrastive-decomposition >= 0.2.3 (save_data)")
    if not _SWARM_TEST_DATA.is_file():
        pytest.skip(f"test recording not found at {_SWARM_TEST_DATA}")

    from scd_app.core.filter_recalculation import compute_all_full_sources

    dictionary, _ = scd.train(
        _SWARM_TEST_DATA,
        config_name="surface",
        max_iterations=2,
        verbose_mode=False,
        output_final_source_plot=False,
    )
    pkl_path = tmp_path / "emg_surface.pkl"
    scd.save_results(pkl_path, dictionary)

    converted = load_decomposition_file(pkl_path)
    assert converted["emg_mask"][0][56] == 1
    assert converted["data"].shape[0] == 64

    saved = converted["discharge_times"][0]
    for redetect in (False, True):
        results, _, _, err = compute_all_full_sources(
            converted, device=torch.device("cpu"), redetect_timestamps=redetect
        )
        assert err == ""
        for unit_idx, (_source, timestamps, _filt) in enumerate(results[0]):
            if redetect:
                # The STA-recalculated filter can shift a peak by a sample or two
                near = sum(np.min(np.abs(timestamps - t)) <= 2 for t in saved[unit_idx])
                assert near >= 0.95 * len(saved[unit_idx])
            else:
                np.testing.assert_array_equal(timestamps, saved[unit_idx])
