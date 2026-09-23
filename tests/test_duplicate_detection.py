"""Characterization tests for Edition duplicate scans."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from scd_app.core.duplicate_detection import (
    clear_duplicate_roles,
    scan_cross_port_duplicates,
    scan_within_port_duplicates,
)
from scd_app.core.mu_model import MotorUnit
from scd_app.core.mu_properties import MUProperties
from scd_app.gui.tabs.edition_tab import EditionTab


def _application():
    return QApplication.instance() or QApplication([])


def _motor_unit(unit_id: int, *, sil: float, stability: float) -> MotorUnit:
    unit = MotorUnit(
        id=unit_id,
        timestamps=np.array([10 + unit_id, 30 + unit_id]),
        source=np.zeros(64),
    )
    unit.props = MUProperties(
        n_spikes=2,
        sil=sil,
        muap_template_stability=stability,
    )
    return unit


def _patched_duplicate_ui(tab):
    return (
        patch.object(tab, "_log_event"),
        patch.object(tab, "_refresh_mu_combo"),
        patch.object(tab, "_update_quality_panel"),
        patch.object(tab, "_update_status"),
        patch.object(tab, "_show_duplicate_report"),
        patch.object(tab, "_mark_modified"),
    )


def test_within_port_scan_flags_only_the_lower_quality_units():
    app = _application()
    tab = EditionTab()
    high = _motor_unit(0, sil=0.95, stability=0.9)
    low = _motor_unit(1, sil=0.70, stability=0.7)
    middle = _motor_unit(2, sil=0.85, stability=0.8)
    tab._ports = {"Grid 1": [high, low, middle]}
    tab._current_port = "Grid 1"

    agreement = np.array(
        [
            [1.0, 0.40, 0.10],
            [0.40, 1.0, 0.35],
            [0.10, 0.35, 1.0],
        ]
    )
    patches = _patched_duplicate_ui(tab)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "scd_app.core.duplicate_detection._tb_spike_comp.rate_of_agreement_full",
                return_value=(agreement, None),
            )
        )
        ui_mocks = [stack.enter_context(patcher) for patcher in patches]
        tab._flag_within_duplicates()

    assert high.within_duplicate_role == "keep"
    assert low.within_duplicate_role == "delete"
    assert middle.within_duplicate_role == "keep"
    assert high.flagged_for_deletion is False
    assert low.flagged_for_deletion is True
    assert middle.flagged_for_deletion is False
    assert low.flagged_duplicate is False
    assert [partner[1] for partner in low.within_duplicate_partners] == [0, 2]
    assert ui_mocks[4].call_args.kwargs["flagged_by_port"] == {"Grid 1": [1]}
    assert ui_mocks[4].call_args.kwargs["n_compared"] == 3

    tab.close()
    app.processEvents()


def test_cross_port_scan_flags_the_lower_quality_unit():
    app = _application()
    tab = EditionTab()
    high = _motor_unit(0, sil=0.95, stability=0.9)
    low = _motor_unit(0, sil=0.70, stability=0.7)
    tab._ports = {"Grid A": [high], "Grid B": [low]}

    patches = _patched_duplicate_ui(tab)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "scd_app.core.duplicate_detection._tb_spike_comp.rate_of_agreement_full",
                return_value=(np.array([[0.42]]), None),
            )
        )
        ui_mocks = [stack.enter_context(patcher) for patcher in patches]
        tab._flag_cross_duplicates()

    assert high.cross_duplicate_role == "keep"
    assert low.cross_duplicate_role == "delete"
    assert high.cross_duplicate_partners == [("Grid B", 0, 0.42)]
    assert low.cross_duplicate_partners == [("Grid A", 0, 0.42)]
    assert high.flagged_for_deletion is False
    assert low.flagged_for_deletion is True
    assert low.flagged_duplicate is False
    assert ui_mocks[4].call_args.kwargs["flagged_by_port"] == {
        "Grid A": [],
        "Grid B": [0],
    }

    tab.close()
    app.processEvents()


def test_clearing_duplicate_roles_preserves_manual_flags():
    unit = _motor_unit(0, sil=0.9, stability=0.8)
    unit.flagged_duplicate = True
    unit.within_duplicate_role = "delete"
    unit.within_duplicate_partners = [("Grid 1", 1, 0.4)]
    unit.cross_duplicate_role = "delete"
    unit.cross_duplicate_partners = [("Grid 2", 2, 0.5)]
    ports = {"Grid 1": [unit]}

    clear_duplicate_roles(ports, "within")

    assert unit.within_duplicate_role is None
    assert unit.within_duplicate_partners == []
    assert unit.cross_duplicate_role == "delete"
    assert unit.flagged_for_deletion is True

    clear_duplicate_roles(ports, "cross")

    assert unit.cross_duplicate_role is None
    assert unit.cross_duplicate_partners == []
    assert unit.flagged_duplicate is True
    assert unit.flagged_for_deletion is True


def test_clearing_duplicate_roles_removes_an_unpersisted_suggestion():
    unit = _motor_unit(0, sil=0.9, stability=0.8)
    unit.within_duplicate_role = "delete"
    unit.within_duplicate_partners = [("Grid 1", 1, 0.4)]

    clear_duplicate_roles({"Grid 1": [unit]}, "within")

    assert unit.flagged_duplicate is False
    assert unit.flagged_for_deletion is False


def test_flagged_unit_is_compared_but_cannot_eliminate_unflagged_partner():
    flagged_high = _motor_unit(0, sil=0.99, stability=0.99)
    flagged_high.flagged_duplicate = True
    unflagged_low = _motor_unit(1, sil=0.60, stability=0.60)

    result = scan_within_port_duplicates(
        {"Grid 1": [flagged_high, unflagged_low]},
        2048.0,
        agreement_computer=lambda **_kwargs: (
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            None,
        ),
    )

    assert result.pairs == [("Grid 1", 0, "Grid 1", 1, 0.5)]
    assert flagged_high.within_duplicate_role == "delete"
    assert unflagged_low.within_duplicate_role == "keep"
    assert flagged_high.flagged_for_deletion is True
    assert unflagged_low.flagged_for_deletion is False


def test_cross_port_scan_also_prefers_an_unflagged_keeper():
    flagged_high = _motor_unit(0, sil=0.99, stability=0.99)
    flagged_high.flagged_duplicate = True
    unflagged_low = _motor_unit(0, sil=0.60, stability=0.60)

    result = scan_cross_port_duplicates(
        {"Grid A": [flagged_high], "Grid B": [unflagged_low]},
        2048.0,
        agreement_computer=lambda **_kwargs: (np.array([[0.5]]), None),
    )

    assert result.pairs == [("Grid A", 0, "Grid B", 0, 0.5)]
    assert flagged_high.cross_duplicate_role == "delete"
    assert unflagged_low.cross_duplicate_role == "keep"
    assert flagged_high.flagged_for_deletion is True
    assert unflagged_low.flagged_for_deletion is False


def test_within_port_button_does_not_replace_other_ports_scan_results():
    app = _application()
    tab = EditionTab()
    first = _motor_unit(0, sil=0.95, stability=0.9)
    second = _motor_unit(1, sil=0.70, stability=0.7)
    other = _motor_unit(0, sil=0.8, stability=0.8)
    other.within_duplicate_role = "keep"
    other.within_duplicate_partners = [("Grid B", 1, 0.4)]
    tab._ports = {"Grid A": [first, second], "Grid B": [other]}
    tab._current_port = "Grid A"

    patches = _patched_duplicate_ui(tab)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "scd_app.core.duplicate_detection._tb_spike_comp.rate_of_agreement_full",
                return_value=(np.array([[1.0, 0.4], [0.4, 1.0]]), None),
            )
        )
        ui_mocks = [stack.enter_context(patcher) for patcher in patches]
        tab._flag_within_duplicates()

    assert second.within_duplicate_role == "delete"
    assert other.within_duplicate_role == "keep"
    assert other.within_duplicate_partners == [("Grid B", 1, 0.4)]
    assert ui_mocks[4].call_args.kwargs["flagged_by_port"] == {"Grid A": [1]}

    tab.close()
    app.processEvents()


def test_duplicate_controls_follow_the_review_workflow_order():
    app = _application()
    tab = EditionTab()
    layout = tab.btn_delete_flagged.parentWidget().layout()

    assert layout.indexOf(tab.btn_delete_flagged) < layout.indexOf(
        tab.btn_flag_within_dups
    )
    assert layout.indexOf(tab.btn_flag_within_dups) < layout.indexOf(
        tab.btn_flag_cross_dups
    )

    tab.close()
    app.processEvents()


def test_manual_unflag_can_override_a_duplicate_suggestion():
    app = _application()
    tab = EditionTab()
    unit = _motor_unit(0, sil=0.8, stability=0.8)
    unit.within_duplicate_role = "delete"
    unit.within_duplicate_partners = [("Grid A", 1, 0.4)]
    tab._ports = {"Grid A": [unit]}
    tab._current_port = "Grid A"
    tab._current_mu_idx = 0

    tab._toggle_flag_delete()

    assert unit.flagged_duplicate is False
    assert unit.within_duplicate_role == "keep"
    assert unit.flagged_for_deletion is False

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_within_scan_reports_skipped_and_failed_ports():
    single = _motor_unit(0, sil=0.9, stability=0.8)
    first = _motor_unit(0, sil=0.9, stability=0.8)
    second = _motor_unit(1, sil=0.8, stability=0.7)

    def fail_agreement(**_kwargs):
        raise RuntimeError("comparison failed")

    result = scan_within_port_duplicates(
        {"Single": [single], "Pair": [first, second]},
        2048.0,
        agreement_computer=fail_agreement,
    )

    assert result.pairs == []
    assert result.flagged_by_port == {"Single": [], "Pair": []}
    assert result.n_flagged == 0
    assert result.n_compared == 2
    assert result.skipped_ports == ["Single"]
    assert result.failed_ports == ["Pair"]
