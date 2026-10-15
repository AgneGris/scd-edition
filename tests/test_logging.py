import logging
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from scd_app.core.logging_config import (
    configure_logging,
    runtime_diagnostics,
    shutdown_logging,
)
from scd_app.core.package_metadata import installed_app_version


def test_rotating_application_log_records_messages(tmp_path):
    try:
        log_path = configure_logging(tmp_path)

        logging.getLogger("scd_app.test").warning("diagnostic test message")
        for handler in logging.getLogger("scd_app").handlers:
            handler.flush()

        assert log_path == tmp_path / "scd-studio.log"
        assert "diagnostic test message" in log_path.read_text(encoding="utf-8")
    finally:
        shutdown_logging()


def test_runtime_diagnostics_include_compute_backend(tmp_path):
    diagnostics = runtime_diagnostics(tmp_path / "scd-studio.log")

    assert "SCD Studio:" in diagnostics
    assert "Python:" in diagnostics
    assert "PyTorch:" in diagnostics
    assert "CUDA available:" in diagnostics
    assert "Log file:" in diagnostics


def test_installed_version_prefers_scd_studio_distribution():
    with patch("scd_app.core.package_metadata.version", return_value="0.2.0") as lookup:
        assert installed_app_version() == "0.2.0"

    lookup.assert_called_once_with("scd-studio")


def test_installed_version_accepts_legacy_distribution():
    with patch(
        "scd_app.core.package_metadata.version",
        side_effect=(PackageNotFoundError("scd-studio"), "0.1.0"),
    ) as lookup:
        assert installed_app_version() == "0.1.0"

    assert [call.args[0] for call in lookup.call_args_list] == [
        "scd-studio",
        "scd-edition",
    ]
