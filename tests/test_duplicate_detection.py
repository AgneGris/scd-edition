"""Characterization tests for Edition duplicate scans."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from scd_app.core.duplicate_detection import (
    clear_duplicate_roles,
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
    assert high.flagged_duplicate is False
    assert low.flagged_duplicate is True
    assert middle.flagged_duplicate is False
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
    assert high.flagged_duplicate is False
    assert low.flagged_duplicate is True
    assert ui_mocks[4].call_args.kwargs["flagged_by_port"] == {
        "Grid A": [],
        "Grid B": [0],
    }

    tab.close()
    app.processEvents()


def test_clearing_one_duplicate_kind_preserves_flags_owned_by_the_other():
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
    assert unit.flagged_duplicate is True

    clear_duplicate_roles(ports, "cross")

    assert unit.cross_duplicate_role is None
    assert unit.cross_duplicate_partners == []
    assert unit.flagged_duplicate is False


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
