"""Run the development checks without changing the selected PyTorch backend."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _commands(backend: str) -> list[list[str]]:
    """Build commands that sync once, then leave the environment untouched."""
    run_without_sync = ["uv", "run", "--no-sync"]
    checked_paths = ["src", "tests", "scripts", "docs"]
    return [
        ["uv", "sync", "--locked", "--extra", backend],
        [*run_without_sync, "ruff", "check", *checked_paths],
        [*run_without_sync, "ruff", "format", "--check", *checked_paths],
        [*run_without_sync, "pytest"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run linting, formatting checks, and tests while preserving an "
            "explicit CPU or CUDA PyTorch installation."
        )
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("cpu", "cuda"),
        help="PyTorch backend to sync before running the checks.",
    )
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("error: uv is required but was not found on PATH", file=sys.stderr)
        return 127

    for command in _commands(args.backend):
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
