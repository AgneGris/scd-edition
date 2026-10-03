"""Build the published SCD demo decomposition from the upstream demo output.

The upstream file is a pickle and must only be supplied from a trusted source.
Its numerical decomposition content is preserved; only the missing run metadata
is added before conversion to the native SCD Edition schema.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scd_app.io.atomic_pickle import atomic_pickle_dump
from scd_app.io.decomposition_loader import (
    _CPUCompatibleUnpickler,
    convert_scd_output,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "examples/scd-demo/emg_decomp_output.pkl"
RAW_RECORDING = REPOSITORY_ROOT / "examples/scd-demo/emg.mat"

SURFACE_PREPROCESSING = {
    "sampling_frequency": 10240,
    "start_time": 0,
    "end_time": -1,
    "low_pass_cutoff": 500,
    "high_pass_cutoff": 20,
    "extension_factor": 5,
    "whitening_method": "zca",
    "autocorrelation_whiten": False,
    "bad_channels": [56],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_example(source: Path, output: Path) -> None:
    """Convert *source* and atomically write the publishable example."""
    source = source.resolve()
    with source.open("rb") as handle:
        upstream = _CPUCompatibleUnpickler(handle).load()

    upstream = dict(upstream)
    upstream["sampling_rate"] = SURFACE_PREPROCESSING["sampling_frequency"]
    upstream["preprocessing_config"] = dict(SURFACE_PREPROCESSING)

    converted = convert_scd_output(upstream, source_path=source)
    converted["skip_filter_recalc"] = True
    converted["electrodes"] = ["GR10MM0808"]
    converted["acquisition_metadata"] = {
        "format": "scd-demo",
        "raw_recording": "examples/scd-demo/emg.mat",
    }
    converted["import_provenance"].update(
        {
            "upstream_output_sha256": _sha256(source),
            "raw_recording_sha256": _sha256(RAW_RECORDING),
        }
    )

    atomic_pickle_dump(converted, output)
    print(f"Wrote {output}")
    print(f"SHA-256: {_sha256(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="trusted upstream data/output/emg_surface.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    build_example(args.source, args.output)


if __name__ == "__main__":
    main()
