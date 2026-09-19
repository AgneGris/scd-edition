"""Validate release artifacts without installing them."""

from __future__ import annotations

import email
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib

REQUIRED_WHEEL_SUFFIXES = {
    "scd_app/resources/examples/emg.mat",
    "scd_app/resources/examples/example_config.json",
    "scd_app/resources/loaders_configs/loader_csv.yaml",
    "scd_app/resources/loaders_configs/loader_npy.yaml",
    "scd_app/resources/loaders_configs/loader_scd_h5.yaml",
    "scd_app/gui/widgets/import_data_dialog.py",
    "scd_app/io/portable_format.py",
    "scd_app/_vendor/motor_unit_toolbox/props.py",
    "scd_app/_vendor/motor_unit_toolbox/spike_comp.py",
}
PROJECT = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def _only(pattern: str) -> Path:
    matches = list(Path("dist").glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern!r} artifact, found {matches}")
    return matches[0]


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for suffix in REQUIRED_WHEEL_SUFFIXES:
            if not any(name.endswith(suffix) for name in names):
                raise AssertionError(f"Wheel is missing {suffix}")

        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        if any("git+" in requirement for requirement in requirements):
            raise AssertionError("Wheel metadata contains a direct Git dependency")
        if not any(requirement.startswith("h5py>=") for requirement in requirements):
            raise AssertionError("Wheel metadata is missing the h5py dependency")
        if (
            metadata["Name"] != PROJECT["name"]
            or metadata["Version"] != PROJECT["version"]
        ):
            raise AssertionError(
                f"Unexpected wheel identity: {metadata['Name']} {metadata['Version']}"
            )
        if not any(name.endswith("licenses/THIRD_PARTY_NOTICES.md") for name in names):
            raise AssertionError("Wheel is missing THIRD_PARTY_NOTICES.md")


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        if any("/paper/" in name for name in names):
            raise AssertionError("Source distribution still contains the removed draft")
        for suffix in (
            "CITATION.cff",
            "CONTRIBUTING.md",
            "docs/importing-data.md",
            "docs/quickstart.md",
            "src/scd_app/resources/examples/emg.mat",
        ):
            if not any(name.endswith(suffix) for name in names):
                raise AssertionError(f"Source distribution is missing {suffix}")


def main() -> int:
    check_wheel(_only("*.whl"))
    check_sdist(_only("*.tar.gz"))
    print("Distribution contents and metadata are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
