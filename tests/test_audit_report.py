import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scd_app.io.audit_report import (
    AUDIT_FORMAT,
    audit_path_for,
    build_audit_report,
    create_decomposition_provenance,
    write_audit_report,
)


def _decomposition_data(source_path: Path) -> dict:
    return {
        "ports": ["Grid 1"],
        "sampling_rate": 2048,
        "plateau_coords": [100, 500],
        "discharge_times": [[np.array([10, 20, 30])]],
        "pulse_trains": [[np.arange(400, dtype=float)]],
        "data": np.full((2, 600), 987654.321),
        "chans_per_electrode": [2],
        "channel_indices": [[4, 5]],
        "emg_mask": [[0, 1]],
        "electrodes": ["Grid 8x8"],
        "decomposition_params": [{"sil_threshold": 0.9, "iterations": 100}],
        "preprocessing_config": [{"extension_factor": 16}],
        "aux_configs": [{"name": "Force", "unit": "N", "source": "signal"}],
        "notes": ["participant-specific note text"],
        "reviewed_mus": {"Grid 1": [0]},
        "edit_history": [
            {
                "datetime": "2026-09-23T12:00:00",
                "event_type": "edit",
                "timestamps_after": [10, 20, 30],
            }
        ],
        "audit_provenance": create_decomposition_provenance(
            source_path,
            status="complete",
            started_at_utc="2026-09-23T11:59:00Z",
            duration_seconds=60.125,
        ),
    }


def test_audit_report_summarises_run_without_signal_or_note_content(tmp_path):
    source_path = tmp_path / "participant-01.otb4"
    source_path.write_bytes(b"recording")
    output_path = tmp_path / "participant-01_decomp_output.pkl"

    report = build_audit_report(
        output_path,
        _decomposition_data(source_path),
        operation="decomposition",
    )
    encoded = json.dumps(report)

    assert report["format"] == AUDIT_FORMAT
    assert "scd_edition" in report["report_environment"]["software"]
    assert report["report_environment"]["compute"]["backend"] in {"cpu", "cuda"}
    assert report["output"]["file_name"] == output_path.name
    assert report["provenance"]["input_recording"]["file_name"] == source_path.name
    assert report["decomposition"]["motor_units_detected"] is None
    assert report["decomposition"]["motor_units_retained"] == 1
    assert report["decomposition"]["ports"][0]["motor_units"] == 1
    assert report["decomposition"]["ports"][0]["rejected_channel_positions"] == [1]
    assert report["editing"]["history_events"] == 1
    assert report["editing"]["notes_count"] == 1
    assert report["editing"]["units_reviewed"] == 1
    assert str(tmp_path) not in encoded
    assert "participant-specific note text" not in encoded
    assert "987654.321" not in encoded
    assert "timestamps_after" not in encoded


def test_writes_audit_report_atomically_beside_decomposition(tmp_path):
    source_path = tmp_path / "recording.mat"
    source_path.write_bytes(b"recording")
    output_path = tmp_path / "result.pkl"

    audit_path = write_audit_report(
        output_path,
        _decomposition_data(source_path),
        operation="edition",
        derived_from=tmp_path / "original.pkl",
    )

    assert audit_path == audit_path_for(output_path)
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["operation"] == "edition"
    assert report["output"]["derived_from"] == "original.pkl"
    assert list(tmp_path.glob("*.tmp")) == []


def test_stored_cross_platform_source_path_is_reduced_to_filename(tmp_path):
    data = _decomposition_data(tmp_path / "recording.mat")
    # Synthetic text only: this path is never opened on the test machine.
    data["audit_provenance"]["input_recording"]["file_name"] = (
        r"C:\sensitive\participant-02.mat"
    )

    report = build_audit_report(
        tmp_path / "result.pkl",
        data,
        operation="edition",
    )

    assert report["provenance"]["input_recording"]["file_name"] == "participant-02.mat"


def test_failed_audit_write_preserves_existing_report(tmp_path):
    source_path = tmp_path / "recording.mat"
    source_path.write_bytes(b"recording")
    output_path = tmp_path / "result.pkl"
    audit_path = audit_path_for(output_path)
    audit_path.write_text("existing report", encoding="utf-8")

    with (
        patch("scd_app.io.audit_report.json.dump", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        write_audit_report(
            output_path,
            _decomposition_data(source_path),
            operation="decomposition",
        )

    assert audit_path.read_text(encoding="utf-8") == "existing report"
    assert list(tmp_path.glob("*.tmp")) == []


def test_decomposition_worker_writes_embedded_and_sidecar_provenance(tmp_path):
    from scd_app.core.decomp_worker import DecompositionWorker

    source_path = tmp_path / "recording.mat"
    source_path.write_bytes(b"recording")
    output_path = tmp_path / "result.pkl"
    worker = DecompositionWorker(
        emg_data=torch.zeros((10, 2)),
        grid_configs={
            "Grid 1": {
                "channels": [0, 1],
                "electrode_type": "Grid 8x8",
                "params": {"sil_threshold": 0.9},
            }
        },
        rejected_channels=[np.array([0, 1])],
        plateau_coords=np.array([0, 10]),
        sampling_rate=1000,
        save_path=output_path,
        emg_file_path=source_path,
    )
    results = {
        "pulse_trains": [[np.arange(10, dtype=float)]],
        "discharge_times": [[np.array([2, 6])]],
        "mu_filters": [np.array([])],
        "ports": ["Grid 1"],
        "w_mat": [None],
        "peel_off_sequence": [[]],
        "preprocessing_config": [{}],
    }

    worker._save_results(results)

    import pickle

    with output_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["audit_provenance"]["status"] == "complete"
    assert saved["audit_provenance"]["input_recording"]["file_name"] == source_path.name
    report = json.loads(audit_path_for(output_path).read_text(encoding="utf-8"))
    assert report["decomposition"]["motor_units_detected"] == 1
    assert report["decomposition"]["motor_units_retained"] == 1
