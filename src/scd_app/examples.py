"""Access to the example recording distributed with SCD Edition."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


def _example_resource(name: str) -> Path:
    resource = files("scd_app").joinpath("resources", "examples", name)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"Bundled example resource is missing: {name}")
    return path


def bundled_example_path() -> Path:
    """Return the installed path of the bundled 64-channel MATLAB recording."""
    return _example_resource("emg.mat")


def bundled_example_config() -> dict:
    """Return a fresh copy of the matching GUI configuration."""
    with _example_resource("example_config.json").open(encoding="utf-8") as handle:
        return json.load(handle)
