"""Crash-safe pickle writes for decomposition and edition output files."""

from __future__ import annotations

import os
import pickle
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def atomic_pickle_dump(value: Any, path: Path) -> None:
    """Write *value* to *path* without exposing a partially written file.

    The temporary file is created beside the destination so ``os.replace`` stays
    on the same filesystem and is atomic on supported platforms.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
