import importlib
import json
import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_visualisation_offsets_plateau_local_timestamps():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.visualisation_tab import VisualisationTab

    app = _application()
    tab = VisualisationTab()
    tab._fsamp = 1000.0
    tab._start_sample = 5000
    mu = MotorUnit(
        id=0,
        timestamps=np.array([10, 20], dtype=np.int64),
        source=np.zeros(100),
        port_name="Grid 1",
    )

    tab._timestamps_are_absolute = False
    np.testing.assert_array_equal(tab._absolute_timestamps(mu), [5010, 5020])
    matrix, time_axis, display_fs = tab._build_idr_matrix([("Grid 1", mu)])
    assert display_fs == 1000.0
    assert matrix.shape == (11, 1)
    assert time_axis[0] == pytest.approx(5.01)

    tab._timestamps_are_absolute = True
    np.testing.assert_array_equal(tab._absolute_timestamps(mu), [10, 20])

    tab.close()
    app.processEvents()


def test_atomic_pickle_failure_preserves_existing_file(tmp_path):
    atomic_pickle = importlib.import_module("scd_app.io.atomic_pickle")
    destination = tmp_path / "result.pkl"
    destination.write_bytes(b"existing result")

    with (
        patch.object(atomic_pickle.pickle, "dump", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        atomic_pickle.atomic_pickle_dump({"new": "data"}, destination)

    assert destination.read_bytes() == b"existing result"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_pickle_writes_loadable_data(tmp_path):
    from scd_app.io.atomic_pickle import atomic_pickle_dump

    destination = tmp_path / "result.pkl"
    expected = {"ports": ["Grid 1"], "value": np.arange(3)}

    atomic_pickle_dump(expected, destination)

    with destination.open("rb") as handle:
        actual = pickle.load(handle)
    assert actual["ports"] == expected["ports"]
    np.testing.assert_array_equal(actual["value"], expected["value"])


def test_edition_save_writes_reproducibility_report(tmp_path):
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab(fsamp=1000.0)
    tab._ports = {
        "Grid 1": [
            MotorUnit(
                id=0,
                timestamps=np.array([2, 6], dtype=np.int64),
                source=np.arange(10, dtype=float),
                port_name="Grid 1",
            )
        ]
    }
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0
    tab._output_path = tmp_path / "edited.pkl"

    assert tab._save_file() is True
    assert tab._output_path.exists()
    report_path = tab._output_path.with_suffix(".audit.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["operation"] == "edition"
    assert report["decomposition"]["motor_units_detected"] is None
    assert report["decomposition"]["motor_units_retained"] == 1

    tab.close()
    app.processEvents()


def test_invalid_pickle_reports_a_load_error(tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    path = tmp_path / "invalid.pkl"
    path.write_bytes(b"not a pickle")
    tab = EditionTab()

    with patch.object(QMessageBox, "critical") as critical:
        loaded = tab.load_from_path(path)

    assert loaded is False
    critical.assert_called_once()
    tab.close()
    app.processEvents()


def test_failed_parse_restores_the_current_edition_session(tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab(fsamp=1000.0)
    original_unit = MotorUnit(
        id=0,
        timestamps=np.array([2, 6], dtype=np.int64),
        source=np.arange(10, dtype=float),
        port_name="Original",
    )
    original_ports = {"Original": [original_unit]}
    original_path = tmp_path / "original.pkl"
    replacement_path = tmp_path / "replacement.pkl"
    with replacement_path.open("wb") as handle:
        pickle.dump(
            {
                "ports": ["Replacement"],
                "sampling_rate": 1000,
                "discharge_times": [[np.array([1])]],
                "pulse_trains": [[np.arange(5, dtype=float)]],
            },
            handle,
        )

    tab._ports = original_ports
    tab._current_port = "Original"
    tab._current_mu_idx = 0
    tab._loaded_path = original_path
    tab._set_dirty(True)
    tab._refresh_port_combo()
    tab._refresh_mu_combo()

    def fail_after_mutating_state(_data):
        tab._ports = {}
        tab._current_port = None
        raise RuntimeError("synthetic parse failure")

    with (
        patch.object(tab, "confirm_save_changes", return_value=True),
        patch.object(
            tab,
            "_load_decomposition_data",
            side_effect=fail_after_mutating_state,
        ),
        patch.object(QMessageBox, "critical"),
    ):
        loaded = tab.load_from_path(replacement_path)

    assert loaded is False
    assert tab._ports is original_ports
    assert tab._current_port == "Original"
    assert tab._current_mu_idx == 0
    assert tab._loaded_path == original_path
    assert tab.is_dirty is True

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_cancelled_save_prevents_window_close():
    from PySide6.QtWidgets import QMessageBox

    from scd_app.gui.main_window import MainWindow

    app = _application()
    window = MainWindow()
    window.edition_tab._set_dirty(True)
    event = MagicMock()

    with (
        patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Save,
        ),
        patch.object(window.edition_tab, "_save_file", return_value=False),
    ):
        window.closeEvent(event)

    event.ignore.assert_called_once_with()
    event.accept.assert_not_called()

    window.edition_tab._set_dirty(False)
    window.close()
    app.processEvents()


def test_window_close_requests_a_safe_worker_shutdown():
    from PySide6.QtWidgets import QMessageBox

    from scd_app.gui.main_window import MainWindow

    app = _application()
    window = MainWindow()
    worker = MagicMock()
    event = MagicMock()
    window.decomp_tab.worker = worker

    with (
        patch.object(window.decomp_tab, "has_running_worker", return_value=True),
        patch.object(window.decomp_tab, "request_shutdown") as request_shutdown,
        patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
    ):
        window.closeEvent(event)

    request_shutdown.assert_called_once_with()
    worker.finished.connect.assert_called_once_with(window._finish_pending_close)
    event.ignore.assert_called_once_with()
    event.accept.assert_not_called()
    assert window._close_pending is True

    window._close_pending = False
    window.decomp_tab.worker = None
    window.close()
    app.processEvents()


def test_flag_change_marks_edition_dirty():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    unit = MotorUnit(
        id=0,
        timestamps=np.array([10, 20], dtype=np.int64),
        source=np.zeros(100),
        port_name="Grid 1",
    )
    tab._ports = {"Grid 1": [unit]}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0

    tab._toggle_flag_delete()

    assert unit.flagged_duplicate is True
    assert tab.is_dirty is True

    tab.close()
    app.processEvents()


def test_selection_shortcuts_toggle_switch_and_respect_disabled_buttons():
    from PySide6.QtGui import QShortcut

    from scd_app.gui.tabs.edition_tab import EditionTab
    from scd_app.gui.widgets.source_plot_widget import SelectionArm

    app = _application()
    tab = EditionTab()
    tab.btn_sel_add.setEnabled(True)
    tab.btn_sel_delete.setEnabled(True)
    shortcuts = {
        shortcut.key().toString(): shortcut for shortcut in tab.findChildren(QShortcut)
    }

    shortcuts["A"].activated.emit()
    assert tab.btn_sel_add.isChecked() is True
    assert tab.btn_sel_delete.isChecked() is False
    assert tab._sel_arm == SelectionArm.ADD
    assert tab.source_plot._sel_arm == SelectionArm.ADD

    shortcuts["D"].activated.emit()
    assert tab.btn_sel_add.isChecked() is False
    assert tab.btn_sel_delete.isChecked() is True
    assert tab._sel_arm == SelectionArm.DELETE
    assert tab.source_plot._sel_arm == SelectionArm.DELETE

    shortcuts["A"].activated.emit()
    assert tab.btn_sel_add.isChecked() is True
    assert tab.btn_sel_delete.isChecked() is False
    assert tab._sel_arm == SelectionArm.ADD

    shortcuts["A"].activated.emit()
    assert tab.btn_sel_add.isChecked() is False
    assert tab._sel_arm == SelectionArm.NONE
    assert tab.source_plot._sel_arm == SelectionArm.NONE

    tab.btn_sel_delete.setEnabled(False)
    shortcuts["D"].activated.emit()
    assert tab.btn_sel_delete.isChecked() is False
    assert tab._sel_arm == SelectionArm.NONE

    tab.close()
    app.processEvents()


def test_reset_view_uses_local_source_in_plateau_only_mode():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    unit = MotorUnit(
        id=0,
        timestamps=np.array([10, 20], dtype=np.int64),
        source=np.array([1.0, 2.0, 3.0]),
        port_name="Grid 1",
    )
    tab._ports = {"Grid 1": [unit]}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0
    tab._fsamp = 1000.0
    tab._start_sample = 5000
    tab._end_sample = 5003
    tab._full_source_mode = False

    view_box = tab.source_plot.getViewBox()
    with (
        patch.object(view_box, "setRange") as set_range,
        patch.object(tab.fr_plot, "reset_y_range"),
    ):
        tab._reset_view_full()

    assert set_range.call_args.kwargs["yRange"] == pytest.approx((-0.45, 9.45))

    tab.close()
    app.processEvents()


def test_reset_view_uses_plateau_window_in_full_source_mode():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    unit = MotorUnit(
        id=0,
        timestamps=np.array([2, 3], dtype=np.int64),
        source=np.array([100.0, 1.0, 2.0, 3.0, 1.0]),
        port_name="Grid 1",
    )
    tab._ports = {"Grid 1": [unit]}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0
    tab._fsamp = 1000.0
    tab._start_sample = 1
    tab._end_sample = 4
    tab._full_source_mode = True

    view_box = tab.source_plot.getViewBox()
    with (
        patch.object(view_box, "setRange") as set_range,
        patch.object(tab.fr_plot, "reset_y_range"),
    ):
        tab._reset_view_full()

    assert set_range.call_args.kwargs["yRange"] == pytest.approx((-0.45, 9.45))

    tab.close()
    app.processEvents()


def test_reset_view_empty_full_source_window_uses_safe_fallback():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    unit = MotorUnit(
        id=0,
        timestamps=np.array([], dtype=np.int64),
        source=np.array([1.0, 2.0, 3.0]),
        port_name="Grid 1",
    )
    tab._ports = {"Grid 1": [unit]}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0
    tab._fsamp = 1000.0
    tab._start_sample = 10
    tab._end_sample = 20
    tab._full_source_mode = True

    view_box = tab.source_plot.getViewBox()
    with (
        patch.object(view_box, "setRange") as set_range,
        patch.object(tab.fr_plot, "reset_y_range"),
    ):
        tab._reset_view_full()

    assert set_range.call_args.kwargs["yRange"] == pytest.approx((-0.05, 1.05))

    tab.close()
    app.processEvents()


def test_saved_plateau_session_reopens_with_same_muap_template():
    from scd_app.core.mu_properties import MUProperties
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    full_emg = np.zeros((2, 200), dtype=float)
    waveform = np.array([[1.0, 4.0, 1.0], [-2.0, 6.0, -2.0]])
    for absolute_sample in (110, 130):
        full_emg[:, absolute_sample - 1 : absolute_sample + 2] = waveform

    source = np.zeros(50, dtype=float)
    source[[10, 30]] = 1.0
    original = {
        "ports": ["Grid 1"],
        "sampling_rate": 1000.0,
        "plateau_coords": [100, 150],
        "discharge_times": [[np.array([10, 30], dtype=np.int64)]],
        "pulse_trains": [[source]],
        "data": full_emg,
        "chans_per_electrode": [2],
        "channel_indices": [np.array([0, 1])],
        "emg_mask": [np.zeros(2, dtype=np.int8)],
        "electrodes": ["unsupported"],
    }

    def template_properties(**kwargs):
        emg_port = kwargs["emg_port"]
        properties = []
        for timestamps in kwargs["all_timestamps"]:
            snippets = np.stack(
                [emg_port[:, sample - 1 : sample + 2] for sample in timestamps]
            )
            unit_properties = MUProperties(n_spikes=len(timestamps))
            unit_properties.muap_grid = snippets.mean(axis=0)[:, np.newaxis, :]
            properties.append(unit_properties)
        return properties

    tab = EditionTab()
    with patch(
        "scd_app.gui.tabs.edition_tab.compute_port_properties",
        side_effect=template_properties,
    ):
        tab._load_decomposition_data(original)
        template_before = tab._ports["Grid 1"][0].props.muap_grid.copy()
        saved = tab._build_save_dict()

        assert saved["skip_filter_recalc"] is True
        assert saved["plateau_coords"] == [100, 150]

        tab._load_decomposition_data(saved)

    assert tab._start_sample == 100
    assert tab._end_sample == 150
    assert tab._full_source_mode is False
    np.testing.assert_array_equal(tab._emg_data["Grid 1"], full_emg[:, 100:150])
    np.testing.assert_array_equal(
        tab._ports["Grid 1"][0].timestamps, np.array([10, 30])
    )
    np.testing.assert_array_equal(
        tab._ports["Grid 1"][0].props.muap_grid, template_before
    )

    tab.close()
    app.processEvents()


def test_file_note_normalisation_rejects_malformed_and_blank_entries():
    from scd_app.io.edition_session import normalise_notes

    assert normalise_notes(None) == []
    assert normalise_notes("not a list") == []
    assert normalise_notes([" first ", "", " \n ", 3, "second"]) == [
        "first",
        "second",
    ]


def test_legacy_note_migration_skips_blank_notes():
    from scd_app.core.mu_model import MUProperties
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    tab._fsamp = 1000.0
    tab._full_source_mode = False
    decomp_data = {
        "chans_per_electrode": [1],
        "discharge_times": [[np.array([1]), np.array([2]), np.array([3])]],
        "pulse_trains": [[np.zeros(5), np.zeros(5), np.zeros(5)]],
        "mu_notes": [["", "legacy\nnote", " \n "]],
    }

    with patch(
        "scd_app.gui.tabs.edition_tab.compute_port_properties",
        return_value=[MUProperties(), MUProperties(), MUProperties()],
    ):
        tab._load_single_port(
            port_idx=0,
            port_name="Grid 1",
            decomp_data=decomp_data,
            emg_full=None,
            start_sample=0,
            end_sample=5,
            full_port_results={},
            ch_offset=0,
        )

    assert tab._notes == ["0000-00-00 00:00:00 (Grid 1, MU 1): legacy note"]

    tab.close()
    app.processEvents()


def test_debounced_properties_update_the_edited_unit_after_selection_changes():
    from scd_app.core.mu_model import MotorUnit
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    units = [
        MotorUnit(
            id=index,
            timestamps=np.array([10, 20], dtype=np.int64),
            source=np.zeros(100),
            port_name="Grid 1",
        )
        for index in range(2)
    ]
    tab._ports = {"Grid 1": units}
    tab._emg_data = {"Grid 1": np.zeros((2, 100))}
    tab._grid_info = {"Grid 1": None}
    tab._pending_props_key = ("Grid 1", 0)
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 1
    recomputed = MagicMock()

    with patch(
        "scd_app.gui.tabs.edition_tab.recompute_unit_properties",
        return_value=recomputed,
    ) as recompute:
        tab._flush_props_update()

    recompute.assert_called_once()
    assert units[0].props is recomputed
    assert units[1].props is None

    tab.close()
    app.processEvents()
