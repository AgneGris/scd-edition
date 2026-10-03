import json
from pathlib import Path

from scipy.io import whosmat

from scd_app.examples import bundled_example_config, bundled_example_path
from scd_app.io.decomposition_loader import load_decomposition_file


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


def test_published_demo_decomposition_matches_bundled_recording():
    repository_root = Path(__file__).resolve().parents[1]
    raw_path = repository_root / "examples/scd-demo/emg.mat"
    output_path = repository_root / "examples/scd-demo/emg_decomp_output.pkl"

    variables = {name: shape for name, shape, _dtype in whosmat(raw_path)}
    decomposition = load_decomposition_file(output_path)

    assert variables["emg"] == (102401, 64)
    assert decomposition["sampling_rate"] == 10240
    assert decomposition["ports"] == ["emg_surface"]
    assert len(decomposition["discharge_times"][0]) == 10
    assert {source.size for source in decomposition["pulse_trains"][0]} == {102401}
    assert decomposition["import_provenance"]["raw_recording_sha256"] == (
        "ef0283ca7e541268c929d943e3a23df630fa9fb9566a714f82c80b36e615f17a"
    )
