import logging

from scd_app.core.logging_config import (
    configure_logging,
    runtime_diagnostics,
    shutdown_logging,
)


def test_rotating_application_log_records_messages(tmp_path):
    try:
        log_path = configure_logging(tmp_path)

        logging.getLogger("scd_app.test").warning("diagnostic test message")
        for handler in logging.getLogger("scd_app").handlers:
            handler.flush()

        assert log_path == tmp_path / "scd-edition.log"
        assert "diagnostic test message" in log_path.read_text(encoding="utf-8")
    finally:
        shutdown_logging()


def test_runtime_diagnostics_include_compute_backend(tmp_path):
    diagnostics = runtime_diagnostics(tmp_path / "scd-edition.log")

    assert "SCD Edition:" in diagnostics
    assert "Python:" in diagnostics
    assert "PyTorch:" in diagnostics
    assert "CUDA available:" in diagnostics
    assert "Log file:" in diagnostics
