"""Regression tests for post-decomposition motor-unit splitting."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from scd_app.core.unit_splitting import suggest_split_by_peak_height

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _unit(unit_id: int, timestamps: list[int], *, peel_group_id: int):
    from scd_app.core.mu_model import MotorUnit
    from scd_app.core.mu_properties import MUProperties

    source = np.zeros(100, dtype=float)
    spike_index = np.arange(len(timestamps))
    source[timestamps] = np.where(
        spike_index % 2 == 0,
        4.0 + 0.1 * spike_index,
        1.0 + 0.1 * spike_index,
    )
    return MotorUnit(
        id=unit_id,
        timestamps=np.asarray(timestamps, dtype=np.int64),
        source=source,
        port_name="Grid A",
        mu_filter=np.array([0.25, 0.75]),
        peel_group_id=peel_group_id,
        reviewed=True,
        props=MUProperties(n_spikes=len(timestamps)),
    )


def _tab_with_two_units():
    from scd_app.gui.tabs.edition_tab import EditionTab

    tab = EditionTab(fsamp=1000.0)
    merged = _unit(0, [10, 20, 30, 40, 50, 60], peel_group_id=0)
    later = _unit(1, [15, 35, 55], peel_group_id=1)
    tab._ports = {"Grid A": [merged, later]}
    tab._current_port = "Grid A"
    tab._current_mu_idx = 0
    tab._original_decomp_data = {
        "peel_off_sequence": [
            [
                {"accepted_unit_idx": 0, "timestamps": merged.timestamps.copy()},
                {"accepted_unit_idx": 1, "timestamps": later.timestamps.copy()},
            ]
        ]
    }
    tab._filter_recalc_available = True
    tab._refresh_port_combo()
    tab.port_combo.blockSignals(True)
    tab.port_combo.setCurrentText("Grid A")
    tab.port_combo.blockSignals(False)
    tab._refresh_mu_combo()
    tab.mu_combo.blockSignals(True)
    tab.mu_combo.setCurrentIndex(0)
    tab.mu_combo.blockSignals(False)
    tab._update_plots(reset_view=True)
    tab._update_split_button_state()
    return tab, merged, later


def _set_preview_group_b(tab, timestamps: set[int]):
    """Correct the automatic preview until group B matches ``timestamps``."""
    for timestamp in sorted(tab._split_group_b.symmetric_difference(timestamps)):
        tab._toggle_split_spike(timestamp)


def test_peak_height_suggestion_finds_two_source_distributions():
    timestamps = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    source = np.zeros(100, dtype=float)
    source[[10, 30, 50]] = [1.0, 1.1, 0.9]
    source[[20, 40, 60]] = [4.0, 4.1, 3.9]

    suggestion = suggest_split_by_peak_height(timestamps, source)

    assert set(suggestion.group_a) == {20, 40, 60}
    assert set(suggestion.group_b) == {10, 30, 50}
    assert suggestion.separation_score > 0.99
    assert 1.1**2 < suggestion.threshold < 3.9**2
    assert set(suggestion.group_a).isdisjoint(suggestion.group_b)
    assert set(np.concatenate((suggestion.group_a, suggestion.group_b))) == set(
        timestamps
    )


def test_peak_height_suggestion_rejects_a_flat_distribution():
    timestamps = np.array([10, 20, 30, 40], dtype=np.int64)
    source = np.zeros(50, dtype=float)
    source[timestamps] = 2.0

    with pytest.raises(ValueError, match="two separable levels"):
        suggest_split_by_peak_height(timestamps, source)


def test_split_preview_is_non_destructive_and_cancelable():
    app = _application()
    tab, merged, _later = _tab_with_two_units()
    original = merged.timestamps.copy()

    tab.btn_split_unit.click()
    assert tab._split_preview_active() is True
    assert tab.btn_split_unit.text() == "Cancel Split"
    assert tab.btn_split_unit.isHidden() is False
    assert tab.btn_confirm_split.isHidden() is False
    assert tab._confirm_split_action.isVisible() is True
    assert 0 < len(tab._split_group_b) < len(original)
    assert tab.btn_confirm_split.isEnabled() is True
    _set_preview_group_b(tab, {20, 40})

    np.testing.assert_array_equal(merged.timestamps, original)
    assert tab.is_dirty is False
    assert tab.btn_confirm_split.isEnabled() is True
    assert tab._split_preview_manually_adjusted is True
    primary = [point.data() for point in tab.source_plot._spike_scatter.points()]
    secondary = [
        point.data() for point in tab.source_plot._secondary_spike_scatter.points()
    ]
    assert primary == [10, 30, 50, 60]
    assert secondary == [20, 40]

    tab.btn_split_unit.click()

    np.testing.assert_array_equal(merged.timestamps, original)
    assert len(tab._ports["Grid A"]) == 2
    assert tab.is_dirty is False
    assert tab._split_preview_active() is False
    assert tab.btn_split_unit.text() == "Split Unit"
    assert tab.btn_confirm_split.isHidden() is True
    assert tab._confirm_split_action.isVisible() is False
    assert tab.btn_flag_delete.isEnabled() is True
    tab.close()
    app.processEvents()


def test_confirm_split_creates_two_units_and_two_grouped_peel_steps():
    from scd_app.core.mu_properties import MUProperties

    app = _application()
    tab, merged, later = _tab_with_two_units()
    tab._undo_stack = {("Grid A", 0): [object()]}
    tab._redo_stack = {("Grid A", 0): [object()]}

    with patch.object(
        tab,
        "_recompute_split_properties",
        side_effect=lambda _port, unit: MUProperties(n_spikes=len(unit.timestamps)),
    ):
        tab._start_split_preview()
        # Deliberately invert the preview; confirmation must still orient the
        # higher-amplitude population as split A.
        _set_preview_group_b(tab, {10, 30, 50})
        tab._confirm_split_unit()

    first, second, retained_later = tab._ports["Grid A"]
    assert first is merged
    assert retained_later is later
    assert [unit.id for unit in tab._ports["Grid A"]] == [0, 2, 1]
    np.testing.assert_array_equal(first.timestamps, [10, 30, 50])
    np.testing.assert_array_equal(second.timestamps, [20, 40, 60])
    assert first.split_label == "A"
    assert second.split_label == "B"
    assert first.split_parent_id == second.split_parent_id == 0
    assert first.peel_group_id == second.peel_group_id == 0
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.mu_filter, second.mu_filter)
    assert first.source is not second.source
    assert first.mu_filter is not second.mu_filter
    assert first.reviewed is False
    assert second.reviewed is False
    assert tab._undo_stack == {}
    assert tab._redo_stack == {}

    peel = tab._original_decomp_data["peel_off_sequence"][0]
    assert [entry["accepted_unit_idx"] for entry in peel] == [0, 1, 2]
    np.testing.assert_array_equal(peel[0]["timestamps"], [10, 30, 50])
    np.testing.assert_array_equal(peel[1]["timestamps"], [20, 40, 60])
    assert tab.btn_recalc_filter.isEnabled() is True
    assert tab._edit_history[-1]["event_type"] == "split_unit"
    assert tab._edit_history[-1]["split_method"] == "source_peak_height_distribution"
    assert tab._edit_history[-1]["manually_adjusted"] is True
    assert tab.is_dirty is True

    primary = [point.data() for point in tab.source_plot._spike_scatter.points()]
    secondary = [
        point.data() for point in tab.source_plot._secondary_spike_scatter.points()
    ]
    assert primary == [10, 30, 50]
    assert secondary == []

    tab._current_mu_idx = 1
    tab._update_plots(reset_view=False)
    primary = [point.data() for point in tab.source_plot._spike_scatter.points()]
    secondary = [
        point.data() for point in tab.source_plot._secondary_spike_scatter.points()
    ]
    assert primary == [20, 40, 60]
    assert secondary == []

    saved = tab._build_save_dict()
    assert saved["motor_unit_ids"] == [[0, 2, 1]]
    assert saved["unit_lineage"][0][0] == {
        "peel_group_id": 0,
        "split_parent_id": 0,
        "split_label": "A",
    }
    assert saved["unit_lineage"][0][1] == {
        "peel_group_id": 0,
        "split_parent_id": 0,
        "split_label": "B",
    }
    saved_peel = saved["peel_off_sequence"][0]
    assert [entry["accepted_unit_idx"] for entry in saved_peel] == [0, 1, 2]
    np.testing.assert_array_equal(saved_peel[0]["timestamps"], [10, 30, 50])
    np.testing.assert_array_equal(saved_peel[1]["timestamps"], [20, 40, 60])

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_split_b_recalculation_peels_curated_split_a_first():
    from scd_app.core.mu_properties import MUProperties

    app = _application()
    tab, _merged, _later = _tab_with_two_units()
    with patch.object(
        tab,
        "_recompute_split_properties",
        side_effect=lambda _port, unit: MUProperties(n_spikes=len(unit.timestamps)),
    ):
        tab._start_split_preview()
        _set_preview_group_b(tab, {20, 40, 60})
        tab._confirm_split_unit()

    tab._current_mu_idx = 1
    tab.mu_combo.setCurrentIndex(1)
    tab._raw_port_channels["Grid A"] = np.zeros((2, 100), dtype=float)
    tab._update_recalc_control()
    assert tab.btn_recalc_filter.isEnabled() is True

    new_filter = np.array([[0.8, 0.2]])
    new_source = np.linspace(0.0, 1.0, 100)
    new_timestamps = np.array([20, 40, 60], dtype=np.int64)
    with patch(
        "scd_app.gui.tabs.edition_tab.recalculate_unit_filter",
        return_value=(new_filter, new_source, new_timestamps),
    ) as recalculate:
        tab._recalculate_filter()

    kwargs = recalculate.call_args.kwargs
    assert kwargs["local_mu_idx"] == 1
    assert kwargs["replay_stop_before_local_idx"] == 1
    replay_timestamps = kwargs["current_port_timestamps_abs"]
    np.testing.assert_array_equal(replay_timestamps[0], [10, 30, 50])
    np.testing.assert_array_equal(replay_timestamps[1], [20, 40, 60])
    np.testing.assert_array_equal(replay_timestamps[2], [15, 35, 55])
    np.testing.assert_array_equal(tab._ports["Grid A"][1].mu_filter, new_filter)
    np.testing.assert_array_equal(tab._ports["Grid A"][1].source, new_source)

    tab._props_timer.stop()
    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_peel_replay_uses_current_timestamp_trains_before_later_units():
    import torch

    from scd_app.core.filter_recalculation import _replay_peel_off_for_port

    peeled_timestamp_trains = []

    def peel_off_source(emg, timestamps, _window_size):
        peeled_timestamp_trains.append(timestamps.cpu().numpy().copy())
        return emg

    peel_sequence = [
        {"accepted_unit_idx": 0, "timestamps": np.array([1, 2])},
        {"accepted_unit_idx": 1, "timestamps": np.array([3, 4])},
        {"accepted_unit_idx": 2, "timestamps": np.array([5, 6])},
    ]
    current_timestamps = [
        np.array([10, 11]),
        np.array([30, 31]),
        np.array([50, 51]),
    ]
    with patch(
        "scd_app.core.filter_recalculation._get_scd_modules",
        return_value={"peel_off_source": peel_off_source},
    ):
        _replay_peel_off_for_port(
            torch.zeros((100, 2)),
            peel_sequence,
            [None, None, None],
            global_offset=0,
            start_sample=0,
            end_sample=100,
            window_size=5,
            min_peak_sep=3,
            device=torch.device("cpu"),
            stop_before_local_idx=2,
            current_timestamps_abs=current_timestamps,
        )

    assert len(peeled_timestamp_trains) == 2
    np.testing.assert_array_equal(peeled_timestamp_trains[0], [10, 11])
    np.testing.assert_array_equal(peeled_timestamp_trains[1], [30, 31])


def test_legacy_single_step_split_is_upgraded_for_independent_replay():
    from scd_app.core.mu_properties import MUProperties

    app = _application()
    tab, _merged, _later = _tab_with_two_units()
    with patch.object(
        tab,
        "_recompute_split_properties",
        side_effect=lambda _port, unit: MUProperties(n_spikes=len(unit.timestamps)),
    ):
        tab._start_split_preview()
        _set_preview_group_b(tab, {20, 40, 60})
        tab._confirm_split_unit()

    peel = tab._original_decomp_data["peel_off_sequence"][0]
    del peel[1]
    assert [entry["accepted_unit_idx"] for entry in peel] == [0, 2]

    tab._ensure_split_peel_steps()

    peel = tab._original_decomp_data["peel_off_sequence"][0]
    assert [entry["accepted_unit_idx"] for entry in peel] == [0, 1, 2]
    np.testing.assert_array_equal(peel[0]["timestamps"], [10, 30, 50])
    np.testing.assert_array_equal(peel[1]["timestamps"], [20, 40, 60])

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


def test_deleting_one_split_child_keeps_shared_peel_step():
    from PySide6.QtWidgets import QMessageBox

    from scd_app.core.mu_properties import MUProperties

    app = _application()
    tab, _merged, _later = _tab_with_two_units()
    with patch.object(
        tab,
        "_recompute_split_properties",
        side_effect=lambda _port, unit: MUProperties(n_spikes=len(unit.timestamps)),
    ):
        tab._start_split_preview()
        _set_preview_group_b(tab, {20, 40, 60})
        tab._confirm_split_unit()

    tab._ports["Grid A"][0].flagged_duplicate = True
    with (
        patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
        patch.object(tab, "_update_plots"),
    ):
        tab._delete_all_flagged()

    assert [unit.id for unit in tab._ports["Grid A"]] == [2, 1]
    peel = tab._original_decomp_data["peel_off_sequence"][0]
    assert [entry["accepted_unit_idx"] for entry in peel] == [0, 1]
    assert tab._ports["Grid A"][0].peel_group_id == 0

    tab._set_dirty(False)
    tab.close()
    app.processEvents()


@pytest.mark.parametrize("has_split_units", [True, False])
def test_loading_edited_file_explains_or_asks_about_filter_recalc(
    tmp_path, has_split_units
):
    from PySide6.QtWidgets import QMessageBox

    from scd_app.gui.tabs.edition_tab import EditionTab

    app = _application()
    tab = EditionTab()
    path = tmp_path / "edited.pkl"
    path.write_bytes(b"placeholder")
    split_parent_id = 0 if has_split_units else None
    data = {
        "skip_filter_recalc": True,
        "unit_lineage": [
            [
                {
                    "peel_group_id": 0,
                    "split_parent_id": split_parent_id,
                    "split_label": "A" if has_split_units else None,
                }
            ]
        ],
    }

    with (
        patch(
            "scd_app.gui.tabs.edition_tab.load_decomposition_file",
            return_value=data,
        ),
        patch.object(tab, "_load_decomposition_data"),
        patch.object(QMessageBox, "information") as information,
        patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ) as question,
    ):
        assert tab.load_from_path(path) is True

    assert data["skip_filter_recalc"] is True
    if has_split_units:
        information.assert_called_once()
        question.assert_not_called()
    else:
        information.assert_not_called()
        question.assert_called_once()

    tab.close()
    app.processEvents()


def test_split_lineage_round_trips_through_session_loader():
    from scd_app.core.mu_properties import MUProperties
    from scd_app.io.decomposition_loader import migrate_and_validate_decomposition
    from scd_app.io.edition_session import load_edition_port

    app = _application()
    tab, _merged, _later = _tab_with_two_units()
    with patch.object(
        tab,
        "_recompute_split_properties",
        side_effect=lambda _port, unit: MUProperties(n_spikes=len(unit.timestamps)),
    ):
        tab._start_split_preview()
        _set_preview_group_b(tab, {20, 40, 60})
        tab._confirm_split_unit()
    saved = migrate_and_validate_decomposition(tab._build_save_dict())

    loaded = load_edition_port(
        port_index=0,
        port_name="Grid A",
        decomposition=saved,
        emg_full=None,
        start_sample=0,
        end_sample=100,
        full_port_results={},
        channel_offset=0,
        full_source_mode=False,
        sampling_rate=1000.0,
        property_computer=lambda **_kwargs: [
            MUProperties(),
            MUProperties(),
            MUProperties(),
        ],
    )

    assert [unit.peel_group_id for unit in loaded.motor_units] == [0, 0, 1]
    assert [unit.split_parent_id for unit in loaded.motor_units] == [0, 0, None]
    assert [unit.split_label for unit in loaded.motor_units] == ["A", "B", None]

    tab._set_dirty(False)
    tab.close()
    app.processEvents()
