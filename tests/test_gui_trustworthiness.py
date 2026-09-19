import importlib
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
