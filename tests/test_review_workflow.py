"""Tests for the persisted motor-unit review workflow."""

import os
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _motor_unit(unit_id: int, port_name: str, *, reviewed: bool = False):
    from scd_app.core.mu_model import MotorUnit

    return MotorUnit(
        id=unit_id,
        timestamps=np.array([10, 30], dtype=np.int64),
        source=np.zeros(64),
        port_name=port_name,
        reviewed=reviewed,
    )


def _set_current(tab, port_name: str, unit_index: int) -> None:
    tab._current_port = port_name
    tab._current_mu_idx = unit_index
    tab._refresh_port_combo()
    tab.port_combo.blockSignals(True)
    tab.port_combo.setCurrentText(port_name)
    tab.port_combo.blockSignals(False)
    tab._refresh_mu_combo()
    tab.mu_combo.blockSignals(True)
    tab.mu_combo.setCurrentIndex(unit_index)
    tab.mu_combo.blockSignals(False)
    tab._update_review_controls()


def test_review_toggle_updates_progress_and_edits_reset_review_state():
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    reviewed = _motor_unit(0, "Grid A", reviewed=True)
    current = _motor_unit(1, "Grid A")
    tab._ports = {"Grid A": [reviewed, current]}
    _set_current(tab, "Grid A", 1)

    with patch.object(tab, "_update_plots"):
        tab._toggle_reviewed()

    assert current.reviewed is True
    assert tab.btn_reviewed.isChecked() is True
    assert tab.btn_reviewed.text() == "☑ Reviewed"
    assert tab.review_progress_label.text() == "Reviewed 2/2"
    assert "☑ reviewed" in tab.mu_combo.itemText(1)
    assert tab.is_dirty is True

    with (
        patch.object(tab.fr_plot, "set_data"),
        patch.object(tab.source_plot, "update_timestamps"),
    ):
        tab._on_data_changed("Spike added")
    tab._props_timer.stop()

    assert current.reviewed is False
    assert tab.review_progress_label.text() == "Reviewed 1/2"
    assert "review reset" in tab.status_bar.currentMessage()

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_next_unreviewed_wraps_across_ports_and_reports_completion():
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    first = _motor_unit(0, "Grid A", reviewed=True)
    second = _motor_unit(1, "Grid A")
    third = _motor_unit(0, "Grid B")
    tab._ports = {"Grid A": [first, second], "Grid B": [third]}
    _set_current(tab, "Grid A", 0)

    with patch.object(tab, "_update_plots"):
        tab._select_next_unreviewed()
        assert (tab._current_port, tab._current_mu_idx) == ("Grid A", 1)

        second.reviewed = True
        tab._update_review_controls()
        tab._select_next_unreviewed()
        assert (tab._current_port, tab._current_mu_idx) == ("Grid B", 0)

    third.reviewed = True
    tab._update_review_controls()
    assert tab.review_progress_label.text() == "Reviewed 3/3"
    assert tab.btn_next_unreviewed.isEnabled() is False

    tab._select_next_unreviewed()
    assert tab.status_bar.currentMessage() == "All motor units have been reviewed"

    tab.close()
    app.processEvents()


def test_deleting_flagged_units_preserves_review_state_of_retained_units():
    from PySide6.QtWidgets import QMessageBox

    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    deleted = _motor_unit(0, "Grid A", reviewed=True)
    deleted.flagged_duplicate = True
    retained = _motor_unit(1, "Grid A", reviewed=True)
    tab._ports = {"Grid A": [deleted, retained]}
    _set_current(tab, "Grid A", 0)

    with (
        patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
        patch.object(tab, "_update_plots"),
    ):
        tab._delete_all_flagged()

    assert tab._ports == {"Grid A": [retained]}
    assert retained.id == 0
    assert retained.reviewed is True
    assert tab.review_progress_label.text() == "Reviewed 1/1"

    tab._set_dirty(False)
    tab.close()
    app.processEvents()
