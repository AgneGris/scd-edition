import numpy as np

from scd_app.core import mu_properties


def _emg_with_templates(
    timestamps: np.ndarray,
    template_a: np.ndarray,
    template_b: np.ndarray | None = None,
) -> np.ndarray:
    """Build two-channel EMG with alternating synthetic discharge templates."""
    template_b = template_a if template_b is None else template_b
    half_window = len(template_a) // 2
    emg = np.zeros((2, int(np.max(timestamps)) + half_window + 10))
    for index, timestamp in enumerate(timestamps):
        template = template_a if index % 2 == 0 else template_b
        window = slice(timestamp - half_window, timestamp - half_window + len(template))
        emg[0, window] += template
        emg[1, window] += 0.5 * template
    return emg


def test_muap_template_stability_is_one_for_matching_split_half_templates():
    timestamps = np.arange(20, 220, 20, dtype=np.int64)
    template = np.array([-1.0, 0.5, 3.0, -1.0])
    emg = _emg_with_templates(timestamps, template)

    stability = mu_properties._compute_muap_template_stability(
        emg, timestamps, fsamp=1000.0, win_ms=4
    )

    np.testing.assert_allclose(stability, 1.0)


def test_muap_template_stability_clips_opposite_templates_to_zero():
    timestamps = np.arange(20, 220, 20, dtype=np.int64)
    template = np.array([-1.0, 0.5, 3.0, -1.0])
    emg = _emg_with_templates(timestamps, template, -template)

    stability = mu_properties._compute_muap_template_stability(
        emg, timestamps, fsamp=1000.0, win_ms=4
    )

    np.testing.assert_allclose(stability, 0.0)


def test_muap_template_stability_requires_five_valid_events_per_half():
    timestamps = np.arange(20, 200, 20, dtype=np.int64)
    template = np.array([-1.0, 0.5, 3.0, -1.0])
    emg = _emg_with_templates(timestamps, template)

    stability = mu_properties._compute_muap_template_stability(
        emg, timestamps, fsamp=1000.0, win_ms=4
    )

    assert np.isnan(stability)


def test_muap_template_stability_is_descriptive_not_a_reliability_threshold():
    props = mu_properties.MUProperties(
        n_spikes=20,
        discharge_rate_hz=10.0,
        cov_pct=20.0,
        sil=0.95,
        muap_template_stability=0.1,
    )

    assert "pnr" not in props.quality_flags
    assert not hasattr(props, "pnr_db")
    assert "muap_template_stability" not in props.quality_flags
    assert props.auto_reliable


def test_port_properties_populates_stability_without_toolbox(monkeypatch):
    timestamps = np.arange(20, 220, 20, dtype=np.int64)
    template = np.array([-1.0, 0.5, 3.0, -1.0])
    emg = _emg_with_templates(timestamps, template)
    source = np.zeros(emg.shape[1])
    source[timestamps] = 1.0
    monkeypatch.setattr(mu_properties, "_TOOLBOX_AVAILABLE", False)

    props = mu_properties.compute_port_properties(
        all_timestamps=[timestamps],
        all_sources=[source],
        emg_port=emg,
        grid_positions=None,
        grid_shape=None,
        fsamp=1000.0,
        win_ms=4,
    )

    np.testing.assert_allclose(props[0].muap_template_stability, 1.0)
