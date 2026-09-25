from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scd_app.core.spike_muap import (
    SpikeMUAPInspection,
    SpikeMUAPUnavailable,
    inspect_spike_muap,
)


def _emg_with_events(
    timestamps: list[int],
    reference: np.ndarray,
    *,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    emg = np.zeros((reference.shape[0], 100), dtype=float)
    half_window = reference.shape[1] // 2
    for index, timestamp in enumerate(timestamps):
        event = (
            selected
            if selected is not None and index == len(timestamps) - 1
            else reference
        )
        emg[:, timestamp - half_window : timestamp + half_window] = event
    return emg


def test_identical_scaled_discharge_has_unit_similarity_and_separate_amplitude():
    template = np.array(
        [
            [0.0, -1.0, 0.5, 3.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.25, 1.5, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    timestamps = np.array([20, 40, 60])
    emg = _emg_with_events(timestamps.tolist(), template, selected=2.0 * template)

    result = inspect_spike_muap(
        emg,
        timestamps,
        selected_sample=60,
        fsamp=1000.0,
        win_ms=10,
        max_lag_ms=0,
    )

    assert result.similarity == pytest.approx(1.0)
    assert result.amplitude_ratio == pytest.approx(2.0)
    assert result.n_reference_spikes == 2
    assert result.n_informative_channels == 2
    assert result.reference_grid.shape[:2] == (2, 1)
    assert result.selected_grid.shape == result.reference_grid.shape


def test_selected_discharge_is_excluded_from_reference_template():
    template = np.vstack(
        [
            np.array([0.0, 0.0, -1.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(10),
        ]
    )
    outlier = -template
    timestamps = np.array([20, 40, 60])
    emg = _emg_with_events(timestamps.tolist(), template, selected=outlier)

    result = inspect_spike_muap(
        emg,
        timestamps,
        selected_sample=60,
        fsamp=1000.0,
        win_ms=10,
        max_lag_ms=0,
    )

    assert result.similarity == pytest.approx(-1.0)
    reference = result.reference_grid[0, 0]
    selected = result.selected_grid[0, 0]
    finite = np.isfinite(reference) & np.isfinite(selected)
    np.testing.assert_allclose(selected[finite], -reference[finite])


def test_small_timing_jitter_is_aligned_and_reported():
    template = np.array([[0.0, 0.0, -1.0, 0.5, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    delayed = np.zeros_like(template)
    delayed[:, 2:] = template[:, :-2]
    timestamps = np.array([20, 40, 60])
    emg = _emg_with_events(timestamps.tolist(), template, selected=delayed)

    result = inspect_spike_muap(
        emg,
        timestamps,
        selected_sample=60,
        fsamp=1000.0,
        win_ms=10,
        max_lag_ms=2,
    )

    assert result.similarity == pytest.approx(1.0)
    assert result.lag_ms == pytest.approx(2.0)
    assert result.reference_grid.shape[-1] > template.shape[-1]
    reference_display = result.reference_grid[0, 0]
    selected_display = result.selected_grid[0, 0]
    raw_selected_display = result.raw_selected_grid[0, 0]
    assert np.count_nonzero(np.isfinite(reference_display)) == 10
    assert np.count_nonzero(np.isfinite(selected_display)) == 10
    assert np.count_nonzero(np.isfinite(raw_selected_display)) == 10
    np.testing.assert_allclose(
        reference_display[np.isfinite(reference_display)],
        (template - np.mean(template, axis=1, keepdims=True)).ravel(),
    )
    expected_selected = delayed - np.mean(delayed, axis=1, keepdims=True)
    np.testing.assert_allclose(
        selected_display[np.isfinite(selected_display)], expected_selected.ravel()
    )
    np.testing.assert_allclose(
        raw_selected_display[np.isfinite(raw_selected_display)],
        expected_selected.ravel(),
    )


def test_other_unit_contribution_is_removed_before_comparison():
    target = np.array([[0.0, -1.0, 0.5, 3.0, 1.0, -0.5, -2.0, -1.0, 0.0, 0.0]])
    interfering = np.array([[0.0, 2.0, 1.0, -1.0, -3.0, -1.0, 0.0, 1.0, 1.0, 0.0]])
    target_timestamps = np.array([40, 80, 120])
    other_timestamps = np.array([20, 80, 160])
    emg = np.zeros((1, 190), dtype=float)
    half_window = target.shape[1] // 2
    for timestamp in target_timestamps:
        emg[:, timestamp - half_window : timestamp + half_window] += target
    for timestamp in other_timestamps:
        emg[:, timestamp - half_window : timestamp + half_window] += interfering

    raw_result = inspect_spike_muap(
        emg,
        target_timestamps,
        selected_sample=80,
        fsamp=1000.0,
        win_ms=10,
        max_lag_ms=0,
    )
    reduced_result = inspect_spike_muap(
        emg,
        target_timestamps,
        selected_sample=80,
        fsamp=1000.0,
        other_unit_timestamps=[other_timestamps],
        win_ms=10,
        max_lag_ms=0,
    )

    assert raw_result.similarity < 0.9
    assert reduced_result.similarity == pytest.approx(1.0)
    assert reduced_result.amplitude_ratio == pytest.approx(1.0)
    assert reduced_result.raw_similarity == pytest.approx(raw_result.similarity)
    assert reduced_result.raw_amplitude_ratio == pytest.approx(
        raw_result.amplitude_ratio
    )
    assert reduced_result.n_subtracted_units == 1
    assert reduced_result.n_subtracted_events == 1
    np.testing.assert_allclose(
        reduced_result.reference_grid, raw_result.reference_grid, equal_nan=True
    )
    reference = reduced_result.reference_grid[0, 0]
    selected = reduced_result.selected_view(remove_other_units=True)[0][0, 0]
    finite = np.isfinite(reference) & np.isfinite(selected)
    np.testing.assert_allclose(selected[finite], reference[finite])


def test_active_channel_mapping_is_preserved_in_grid_output():
    template = np.array(
        [
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 2.0, 0.0, -2.0],
        ]
    )
    timestamps = np.array([10, 20])
    emg = _emg_with_events(timestamps.tolist(), template)

    result = inspect_spike_muap(
        emg,
        timestamps,
        selected_sample=20,
        fsamp=1000.0,
        win_ms=4,
        max_lag_ms=0,
        grid_positions={0: (0, 1), 1: (1, 0)},
        grid_shape=(2, 2),
    )

    assert result.reference_grid.shape[:2] == (2, 2)
    assert result.reference_grid.shape[-1] >= 4
    assert np.any(result.reference_grid[0, 1] != 0)
    assert np.any(result.reference_grid[1, 0] != 0)
    assert np.all(result.reference_grid[0, 0] == 0)


def test_edge_spike_and_missing_reference_are_reported():
    emg = np.ones((2, 50))

    with pytest.raises(SpikeMUAPUnavailable, match="recording edge"):
        inspect_spike_muap(
            emg,
            np.array([2, 20]),
            selected_sample=2,
            fsamp=1000.0,
            win_ms=10,
        )

    with pytest.raises(SpikeMUAPUnavailable, match="one other"):
        inspect_spike_muap(
            emg,
            np.array([20]),
            selected_sample=20,
            fsamp=1000.0,
            win_ms=10,
        )


def test_source_marker_right_click_requests_non_destructive_inspection():
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.widgets.source_plot_widget import SourcePlotWidget

    app = QApplication.instance() or QApplication([])
    widget = SourcePlotWidget()
    widget.set_fsamp(1000.0)
    widget.set_data(np.ones(20), np.array([12], dtype=np.int64))
    requested = []
    widget.spike_inspect_requested.connect(requested.append)
    point = widget._spike_scatter.points()[0]
    assert point.data() == 12
    event = MagicMock()
    event.button.return_value = Qt.MouseButton.RightButton

    widget._on_spike_marker_clicked(None, [point], event)

    assert requested == [12]
    event.accept.assert_called_once_with()
    widget.set_inspected_spike(12)
    highlight_x, _ = widget._inspected_spike_scatter.getData()
    np.testing.assert_allclose(highlight_x, [0.012])

    requested.clear()
    press_event = MagicMock()
    press_event.button.return_value = Qt.MouseButton.RightButton
    press_event.pos.return_value = QPoint(1, 1)
    release_event = MagicMock()
    release_event.button.return_value = Qt.MouseButton.RightButton
    with patch.object(widget, "_spike_sample_near_position", return_value=12):
        widget.mousePressEvent(press_event)
        widget.mouseReleaseEvent(release_event)

    assert requested == [12]
    press_event.accept.assert_called_once_with()
    release_event.accept.assert_called_once_with()
    assert widget._spike_inspection_press is False
    widget.close()
    app.processEvents()


def test_edition_inspection_does_not_modify_the_motor_unit():
    from PySide6.QtWidgets import QApplication

    from scd_app.core.mu_model import MotorUnit
    from scd_app.core.mu_properties import MUProperties
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = QApplication.instance() or QApplication([])
    tab = EditionTab(fsamp=1000.0)
    timestamps = np.array([20, 40, 60], dtype=np.int64)
    unit = MotorUnit(
        id=0,
        timestamps=timestamps.copy(),
        source=np.zeros(100),
        port_name="Grid 1",
        props=MUProperties(muap_grid=np.ones((1, 1, 10))),
    )
    other_timestamps = np.array([30, 50, 70], dtype=np.int64)
    other_unit = MotorUnit(
        id=1,
        timestamps=other_timestamps,
        source=np.zeros(100),
        port_name="Grid 1",
        props=MUProperties(muap_grid=np.ones((1, 1, 10))),
    )
    result = SpikeMUAPInspection(
        selected_sample=60,
        reference_grid=np.ones((1, 1, 10)),
        selected_grid=np.ones((1, 1, 10)),
        similarity=1.0,
        amplitude_ratio=1.0,
        lag_ms=0.0,
        n_reference_spikes=2,
        n_informative_channels=1,
    )
    tab._ports = {"Grid 1": [unit, other_unit]}
    tab._emg_data = {"Grid 1": np.zeros((1, 100))}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0

    with (
        patch(
            "scd_app.gui.tabs.edition_tab.inspect_spike_muap", return_value=result
        ) as inspect,
        patch.object(tab, "_plot_muap") as plot_muap,
    ):
        tab._inspect_spike_muap(60)

    np.testing.assert_array_equal(unit.timestamps, timestamps)
    assert tab.is_dirty is False
    assert tab._spike_muap_inspection is result
    assert tab._spike_muap_inspection_key == ("Grid 1", 0)
    passed_other_units = inspect.call_args.kwargs["other_unit_timestamps"]
    assert passed_other_units == []
    plot_muap.assert_called_once_with()

    # The next unit sees MU 0's current timestamps, matching accepted-unit
    # peel-off order; later units are never subtracted from earlier ones.
    tab._spike_muap_inspection = None
    tab._spike_muap_inspection_key = None
    tab._current_mu_idx = 1
    with (
        patch(
            "scd_app.gui.tabs.edition_tab.inspect_spike_muap", return_value=result
        ) as inspect,
        patch.object(tab, "_plot_muap"),
    ):
        tab._inspect_spike_muap(70)
    passed_other_units = inspect.call_args.kwargs["other_unit_timestamps"]
    assert len(passed_other_units) == 1
    np.testing.assert_array_equal(passed_other_units[0], timestamps)

    tab.close()
    app.processEvents()


def test_inspected_spike_can_be_hidden_and_navigated():
    from PySide6.QtWidgets import QApplication

    from scd_app.core.mu_model import MotorUnit
    from scd_app.core.mu_properties import MUProperties
    from scd_app.gui.tabs.edition_tab import EditionTab

    app = QApplication.instance() or QApplication([])
    tab = EditionTab(fsamp=1000.0)
    unit = MotorUnit(
        id=0,
        timestamps=np.array([20, 40, 60], dtype=np.int64),
        source=np.zeros(100),
        port_name="Grid 1",
        props=MUProperties(muap_grid=np.ones((1, 1, 10))),
    )
    tab._ports = {"Grid 1": [unit]}
    tab._current_port = "Grid 1"
    tab._current_mu_idx = 0
    tab._spike_muap_inspection = SpikeMUAPInspection(
        selected_sample=40,
        reference_grid=np.ones((1, 1, 10)),
        selected_grid=np.ones((1, 1, 10)),
        similarity=1.0,
        amplitude_ratio=1.0,
        lag_ms=0.0,
        n_reference_spikes=2,
        n_informative_channels=1,
        raw_selected_grid=np.full((1, 1, 10), 2.0),
        raw_similarity=0.25,
        raw_amplitude_ratio=2.0,
        raw_lag_ms=1.0,
    )
    tab._spike_muap_inspection_key = ("Grid 1", 0)
    tab._update_muap_inspection_controls()

    assert tab.muap_spike_position_label.text() == "Spike 2/3"
    assert tab.btn_prev_inspected_spike.isEnabled()
    assert tab.btn_next_inspected_spike.isEnabled()
    assert tab.spike_signal_combo.currentText() == "Raw EMG"
    assert tab._selected_spike_view(tab._spike_muap_inspection)[1:] == (
        0.25,
        2.0,
        1.0,
    )

    with patch.object(tab, "_inspect_spike_muap") as inspect:
        tab._navigate_inspected_spike(1)
    inspect.assert_called_once_with(60)

    with patch.object(tab, "_navigate_inspected_spike") as navigate:
        tab._handle_horizontal_arrow(-1)
    navigate.assert_called_once_with(-1)

    with patch.object(tab, "_plot_muap") as plot_muap:
        tab.btn_show_selected_spike.setChecked(False)
    assert tab._show_selected_spike is False
    plot_muap.assert_called_once_with()

    with (
        patch.object(tab, "_plot_muap") as plot_muap,
        patch.object(tab, "_update_status"),
    ):
        tab.spike_signal_combo.setCurrentIndex(1)
    assert tab._remove_other_units_from_selected_spike is True
    assert tab._selected_spike_view(tab._spike_muap_inspection)[1:] == (
        1.0,
        1.0,
        0.0,
    )
    plot_muap.assert_called_once_with()

    with (
        patch("scd_app.gui.tabs.edition_tab.inspect_spike_muap") as inspect,
        patch.object(tab, "_plot_muap") as plot_muap,
        patch.object(tab, "_update_status") as update_status,
    ):
        tab._inspect_spike_muap(40)
    inspect.assert_not_called()
    assert tab._spike_muap_inspection is None
    assert tab._spike_muap_inspection_key is None
    plot_muap.assert_called_once_with()
    assert update_status.call_args_list[-1].args == ("Spike MUAP inspection cleared",)

    with patch.object(tab, "_pan_source") as pan_source:
        tab._handle_horizontal_arrow(1)
    pan_source.assert_called_once_with(0.02)

    tab.close()
    app.processEvents()


def test_muap_grid_renders_and_clears_selected_spike_overlay():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.edition_tab import EditionTab

    app = QApplication.instance() or QApplication([])
    tab = EditionTab(fsamp=1000.0)
    tab._current_mu_idx = 0
    ordinary = np.arange(10, dtype=float).reshape(1, 1, 10)
    reference = (ordinary + 1.0).copy()
    selected = (reference * 2.0).copy()
    result = SpikeMUAPInspection(
        selected_sample=60,
        reference_grid=reference,
        selected_grid=selected,
        similarity=1.0,
        amplitude_ratio=2.0,
        lag_ms=0.0,
        n_reference_spikes=2,
        n_informative_channels=1,
    )
    grid_config = {"grid_shape": (1, 1), "positions": {0: (0, 0)}}

    tab._render_muap_grid(ordinary, grid_config, inspection=result)

    _, reference_y = tab._muap_waveform_items[(0, 0)].getData()
    _, selected_y = tab._muap_inspection_items[(0, 0)].getData()
    np.testing.assert_allclose(reference_y, reference.ravel())
    np.testing.assert_allclose(selected_y, selected.ravel())
    reference_pen = tab._muap_waveform_items[(0, 0)].opts["pen"]
    selected_pen = tab._muap_inspection_items[(0, 0)].opts["pen"]
    assert reference_pen.widthF() == pytest.approx(3.0)
    assert reference_pen.style() == Qt.PenStyle.SolidLine
    assert selected_pen.widthF() == pytest.approx(1.5)
    assert selected_pen.style() == Qt.PenStyle.SolidLine
    assert selected_pen.color().alpha() == 210

    tab._show_selected_spike = False
    tab._render_muap_grid(ordinary, grid_config, inspection=result)
    _, hidden_y = tab._muap_inspection_items[(0, 0)].getData()
    assert hidden_y is None or hidden_y.size == 0

    tab._render_muap_grid(ordinary, grid_config)

    _, ordinary_y = tab._muap_waveform_items[(0, 0)].getData()
    _, cleared_y = tab._muap_inspection_items[(0, 0)].getData()
    np.testing.assert_allclose(ordinary_y, ordinary.ravel())
    assert cleared_y is None or cleared_y.size == 0

    tab.close()
    app.processEvents()
