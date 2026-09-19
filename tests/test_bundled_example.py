import json

from scipy.io import whosmat

from scd_app.examples import bundled_example_config, bundled_example_path


def test_bundled_example_matches_its_configuration():
    sample_path = bundled_example_path()
    variables = {name: shape for name, shape, _dtype in whosmat(sample_path)}
    config = bundled_example_config()

    assert variables["emg"] == (102401, 64)
    assert config["loader"] == ".mat"
    assert config["sampling_rate"] == 10240
    assert config["emg_path"] == "emg"
    assert config["grids"][0]["start_chan"] == 0
    assert config["grids"][0]["end_chan"] == 64


def test_bundled_example_config_returns_a_fresh_copy():
    first = bundled_example_config()
    first["grids"].clear()

    assert len(bundled_example_config()["grids"]) == 1
    json.dumps(bundled_example_config())
