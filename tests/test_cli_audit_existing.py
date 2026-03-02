from __future__ import annotations

import json
from pathlib import Path

from cte import cli


def _write_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {
            "micro_f1": 0.5,
            "micro_precision": 0.8,
            "micro_recall": 0.4,
            "hallucination_rate": 0.2,
            "field_acc_macro": {},
        },
        "per_doc": [
            {
                "doc": "aapl.2023.targets.v1.json",
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "field_accuracy": {},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(experiments_root: Path, report_path: Path) -> None:
    run_id = "20260210T000000Z-audit-test-aaaa1111"
    experiment_dir = experiments_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "run_label": "audit_test",
        "timestamp_utc": "2026-02-10T00:00:00+00:00",
        "pipeline": "no_rag",
        "pipeline_version": "no_rag.v1",
        "model_name": "gpt-5.2-2025-12-11",
        "baseline_run_id": None,
        "parent_run_id": None,
        "changed_components": [],
        "change_reason": None,
        "runs": [
            {
                "counter": 1,
                "prediction_dir": "unused",
                "prediction_files": 1,
                "report_path": str(report_path),
                "aggregate": {
                    "micro_f1": 0.5,
                    "micro_precision": 0.8,
                    "micro_recall": 0.4,
                    "hallucination_rate": 0.2,
                    "field_acc_macro": {},
                },
            }
        ],
        "artifacts": {
            "experiment_dir": str(experiment_dir),
            "results_dir": str(experiment_dir / "results" / "no_rag"),
            "analysis_dir": str(experiment_dir / "analysis"),
            "generated_targets_dir": str(experiment_dir / "generated_targets"),
            "indexes_root": str(experiment_dir / "indexes"),
        },
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_audit_existing_command_writes_new_outputs(tmp_path: Path, capsys) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    report_path = _write_report(tmp_path / "reports" / "single.json")

    experiments_root.mkdir()
    results_root.mkdir()
    _write_manifest(experiments_root, report_path)

    exit_code = cli.main(
        [
            "audit-existing",
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
            "--experiments-root",
            str(experiments_root),
            "--run-notes-file",
            str(experiments_root / "run_notes.toml"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "framework_runs=1" in captured.out
    assert "linked_comparisons=0" in captured.out
    assert "legacy_reports=0" in captured.out
    assert "summary_markdown=" in captured.out
    assert "experiment_run_log_csv=" in captured.out
    assert "linked_comparisons_csv=" in captured.out
    assert "legacy_results_log_csv=" in captured.out
    assert "experiment_run_log_jsonl=" in captured.out

    assert (out_dir / "existing_results_audit.md").exists()
    assert (out_dir / "experiment_run_log.csv").exists()
    assert (out_dir / "linked_comparisons.csv").exists()
    assert (out_dir / "legacy_results_log.csv").exists()
    assert (out_dir / "experiment_run_log.jsonl").exists()
