"""Parity tests: filter recalculation must preprocess EMG exactly as SCD did."""

import pytest
import torch

FS = 2048
EXTENSION_FACTOR = 8
CPU = torch.device("cpu")


def _emg(n_samples: int, n_channels: int = 16, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    drift = torch.randn(n_samples, n_channels, generator=gen).cumsum(0) * 0.01
    return drift + torch.randn(n_samples, n_channels, generator=gen)


def _scd_preprocess(emg: torch.Tensor, **config):
    """Run SCD's own preprocessing: (whitened emg, w_mat, preprocessing_config)."""
    from scd.config.structures import Config
    from scd.models.scd import SwarmContrastiveDecomposition

    model = SwarmContrastiveDecomposition()
    model.config = Config(
        device="cpu",
        sampling_frequency=FS,
        extension_factor=EXTENSION_FACTOR,
        verbose_mode=False,
        **config,
    )
    whitened = model.preprocess_emg(emg.clone())
    # The config is read back through SCD's own snapshot so that a key renamed
    # upstream breaks this test instead of silently skipping a step.
    model.decomp = {}
    model._capture_preprocessing_config()
    return whitened, model.w_mat.numpy(), model.decomp["preprocessing_config"]


@pytest.mark.parametrize("whitening_method", ["zca", "pca_cor"])
@pytest.mark.parametrize("high_pass_cutoff", [20, None])
@pytest.mark.parametrize("autocorrelation_whiten", [False, True])
def test_preprocess_emg_matches_scd(
    autocorrelation_whiten, high_pass_cutoff, whitening_method
):
    from scd_app.core.filter_recalculation import preprocess_emg

    emg = _emg(FS * 4)
    expected, w_mat, config = _scd_preprocess(
        emg,
        low_pass_cutoff=500,
        high_pass_cutoff=high_pass_cutoff,
        autocorrelation_whiten=autocorrelation_whiten,
        whitening_method=whitening_method,
    )

    # Saved w_mat, and the refit used for files that have none
    for saved_w_mat in (w_mat, None):
        got = preprocess_emg(emg, config, CPU, w_mat=saved_w_mat)
        torch.testing.assert_close(got, expected, atol=1e-3, rtol=0)


@pytest.mark.parametrize("high_pass_cutoff", [20, None])
def test_full_signal_replay_matches_scd_on_the_plateau(high_pass_cutoff):
    """SCD decomposed only the plateau; the editor replays the whole signal."""
    from scd_app.core.filter_recalculation import preprocess_emg

    start, end = FS * 2, FS * 4
    emg = _emg(FS * 6)
    expected, w_mat, config = _scd_preprocess(
        emg[start:end], low_pass_cutoff=500, high_pass_cutoff=high_pass_cutoff
    )

    got = preprocess_emg(emg, config, CPU, w_mat=w_mat, plateau=slice(start, end))

    # The filters settle differently on the cropped and the full signal, so
    # only the plateau interior is expected to match.  Autocorrelation
    # whitening is left out: it amplifies those edge differences ~70x, so its
    # replay is checked by the parity and invariance tests instead.
    margin = FS // 4
    torch.testing.assert_close(
        got[start + margin : end - margin],
        expected[margin:-margin],
        atol=0.05,
        rtol=0,
    )


@pytest.mark.parametrize("high_pass_cutoff", [20, None])
@pytest.mark.parametrize("autocorrelation_whiten", [False, True])
def test_whitening_statistics_come_from_the_plateau(
    autocorrelation_whiten, high_pass_cutoff
):
    """Changing the signal well outside the plateau must not move the fit."""
    from scd_app.core.filter_recalculation import preprocess_emg

    start, end = FS * 2, FS * 4
    guard = FS // 2  # beyond the reach of the filters and the extension
    quiet = _emg(FS * 6)
    loud = quiet.clone()
    loud[: start - guard] = loud[: start - guard] * 5 + 3
    loud[end + guard :] = loud[end + guard :] * 5 - 3
    _, w_mat, config = _scd_preprocess(
        quiet[start:end],
        low_pass_cutoff=500,
        high_pass_cutoff=high_pass_cutoff,
        autocorrelation_whiten=autocorrelation_whiten,
    )

    plateau = slice(start, end)
    for saved_w_mat in (w_mat, None):
        got_quiet = preprocess_emg(
            quiet, config, CPU, w_mat=saved_w_mat, plateau=plateau
        )
        got_loud = preprocess_emg(loud, config, CPU, w_mat=saved_w_mat, plateau=plateau)
        torch.testing.assert_close(
            got_loud[plateau], got_quiet[plateau], atol=1e-5, rtol=0
        )


@pytest.mark.parametrize("low_pass_cutoff", [500, None])
def test_preprocess_emg_leaves_input_untouched(low_pass_cutoff):
    # SCD's filters and scd.whiten both modify their argument in place; with no
    # filter and no extension the raw tensor itself would reach whitening.
    from scd_app.core.filter_recalculation import preprocess_emg

    emg = _emg(FS) + 3
    before = emg.clone()
    config = {
        "sampling_frequency": FS,
        "extension_factor": 1,
        "low_pass_cutoff": low_pass_cutoff,
    }
    preprocess_emg(emg, config, CPU, plateau=slice(FS // 4, FS // 2))
    assert torch.equal(emg, before)
