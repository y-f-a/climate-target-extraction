from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cte.dashboard.app import create_app


def test_dashboard_index_route_renders_without_request_query_param(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    app = create_app(experiments_root=experiments_root)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Climate Target Extraction Dashboard" in response.text
    assert "Active Parse-Cache Jobs" in response.text


def test_run_detail_route_renders_diagnostics_sections(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_render"
    experiment_dir = experiments_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_path = experiment_dir / "results" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "micro_f1": 0.6,
                    "micro_precision": 0.7,
                    "micro_recall": 0.5,
                    "field_acc_macro": {"target_year": 0.3},
                },
                "per_doc": [
                    {
                        "doc": "aapl.2024.targets.v1.json",
                        "tp": 1,
                        "fp": 1,
                        "fn": 1,
                        "field_accuracy": {"target_year": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest_payload = {
        "run_id": run_id,
        "run_label": run_id,
        "timestamp_utc": "2026-02-10T12:00:00+00:00",
        "started_at_utc": "2026-02-10T12:00:00+00:00",
        "finished_at_utc": "2026-02-10T12:10:00+00:00",
        "run_status": "completed",
        "pipeline": "no_rag",
        "pipeline_version": "no_rag.v1",
        "model_name": "gpt-5.2-2025-12-11",
        "judge_model_name": "gpt-5-mini-2025-08-07",
        "runs": [
            {
                "counter": 1,
                "report_path": str(report_path),
                "aggregate": {
                    "micro_f1": 0.6,
                    "micro_precision": 0.7,
                    "micro_recall": 0.5,
                },
            }
        ],
        "artifacts": {"experiment_dir": str(experiment_dir)},
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    app = create_app(experiments_root=experiments_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert "Live Progress" in response.text
    assert "Diagnostics Source" in response.text
    assert "Run Counter" in response.text
    assert str(report_path) in response.text
    assert "Company-Year Diagnostics" in response.text
    assert "Retrieved Chunks" in response.text
    assert "Evaluation Item" not in response.text
    assert "AAPL 2024" in response.text
    assert "source PDF mapping unavailable for non-RAG runs" in response.text
    assert "retrieved chunk mapping unavailable for non-RAG runs" in response.text
    assert "Field Summary Diagnostics" in response.text


def test_run_detail_route_auto_refreshes_for_active_runs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "active_live_refresh"
    experiment_dir = experiments_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_label": run_id,
                "timestamp_utc": "2026-02-10T12:00:00+00:00",
                "started_at_utc": "2026-02-10T12:00:00+00:00",
                "finished_at_utc": None,
                "run_status": "running",
                "pipeline": "no_rag",
                "pipeline_version": "no_rag.v1",
                "model_name": "gpt-5.2-2025-12-11",
                "judge_model_name": "gpt-5-mini-2025-08-07",
                "runs": [],
                "artifacts": {"experiment_dir": str(experiment_dir)},
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "live_status.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "job_kind": "experiment_run",
                "run_id": run_id,
                "status": "running",
                "updated_at_utc": "2026-02-10T12:01:00+00:00",
                "stage": "extract",
                "run_counter_current": 1,
                "run_count_total": 2,
                "extract_progress": {"completed": 3, "total": 14, "current_doc_name": "meta.2024.targets.v1.json"},
                "evaluate_progress": {"completed": 0, "total": 14, "current_doc_name": None},
                "stalled_after_seconds": 1200,
            }
        ),
        encoding="utf-8",
    )

    app = create_app(experiments_root=experiments_root)
    client = TestClient(app)
    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert 'http-equiv="refresh"' in response.text
    assert "Live Progress" in response.text
