"""Persistent application logging and shareable runtime diagnostics."""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

import torch

APP_LOGGER_NAME = "scd_app"
LOG_FILE_NAME = "scd-edition.log"
_HANDLER_MARKER = "_scd_edition_handler"


def get_log_directory() -> Path:
    """Return the platform-appropriate directory for SCD Edition logs."""
    override = os.environ.get("SCD_LOG_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (
            Path(base) / "SCD Edition" / "logs"
            if base
            else Path.home() / "AppData" / "Local" / "SCD Edition" / "logs"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "SCD Edition"

    state_home = os.environ.get("XDG_STATE_HOME")
    base = (
        Path(state_home).expanduser()
        if state_home
        else Path.home() / ".local" / "state"
    )
    return base / "scd-edition" / "logs"


def _create_log_directory(log_directory: Path | None = None) -> Path:
    directory = (
        Path(log_directory) if log_directory is not None else get_log_directory()
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "scd-edition" / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def configure_logging(log_directory: Path | None = None) -> Path:
    """Configure rotating file logging once and return the active log path."""
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    for handler in app_logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False) and hasattr(
            handler, "baseFilename"
        ):
            return Path(handler.baseFilename)

    directory = _create_log_directory(log_directory)
    log_path = directory / LOG_FILE_NAME
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARKER, True)
    app_logger.addHandler(file_handler)

    if sys.stderr is not None:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _HANDLER_MARKER, True)
        app_logger.addHandler(console_handler)

    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    return log_path


def shutdown_logging() -> None:
    """Close application handlers; primarily useful for clean test teardown."""
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    for handler in list(app_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            app_logger.removeHandler(handler)
            handler.close()


def runtime_diagnostics(log_path: Path | None = None) -> str:
    """Return non-recording-specific runtime information for support reports."""
    try:
        app_version = version("scd-edition")
    except PackageNotFoundError:
        app_version = "development checkout"

    cuda_available = torch.cuda.is_available()
    lines = [
        f"SCD Edition: {app_version}",
        f"Python: {platform.python_version()}",
        f"Operating system: {platform.platform()}",
        f"Architecture: {platform.machine()}",
        f"PyTorch: {torch.__version__}",
        f"PyTorch CUDA build: {torch.version.cuda or 'none'}",
        f"CUDA available: {cuda_available}",
    ]
    if cuda_available:
        try:
            lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            lines.append(f"CUDA device: unavailable ({exc})")
    if log_path is not None:
        lines.append(f"Log file: {Path(log_path)}")
    return "\n".join(lines)


def log_startup_diagnostics(log_path: Path) -> None:
    logging.getLogger(__name__).info(
        "Application starting\n%s", runtime_diagnostics(log_path)
    )


def install_exception_hook() -> None:
    """Log otherwise-uncaught Python exceptions before normal handling."""
    if getattr(sys.excepthook, "_scd_edition_hook", False):
        return

    previous_hook = sys.excepthook

    def log_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, traceback)
            return
        logging.getLogger(f"{APP_LOGGER_NAME}.uncaught").critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, traceback),
        )
        if sys.stderr is not None:
            previous_hook(exc_type, exc_value, traceback)

    log_exception._scd_edition_hook = True
    sys.excepthook = log_exception
