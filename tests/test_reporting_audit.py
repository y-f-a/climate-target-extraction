from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cte.reporting import audit_existing_results


def _write_report(
    path: Path,
    *,
    doc_names: list[str],
    micro_f1: float,
    micro_precision: float,
    micro_recall: float,
    hallucination_rate: float,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "hallucination_rate": hallucination_rate,
            "field_acc_macro": {},
        },
        "per_doc": [
            {
                "doc": doc_name,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "field_accuracy": {},
            }
            for doc_name in doc_names
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_record(
    *,
    counter: int,
    report_path: Path,
    prediction_files: int,
    micro_f1: float,
    micro_precision: float,
    micro_recall: float,
    hallucination_rate: float,
) -> dict:
    return {
        "counter": counter,
        "prediction_dir": "unused",
        "prediction_files": prediction_files,
        "report_path": str(report_path),
        "aggregate": {
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "hallucination_rate": hallucination_rate,
            "field_acc_macro": {},
        },
    }


def _write_manifest(
    experiments_root: Path,
    *,
    run_id: str,
    run_label: str,
    timestamp_utc: str,
    runs: list[dict],
    pipeline: str = "no_rag",
    pipeline_version: str = "no_rag.v1",
    baseline_run_id: str | None = None,
    parent_run_id: str | None = None,
    changed_components: list[str] | None = None,
    change_reason: str | None = None,
    run_status: str | None = None,
    failed_stage: str | None = None,
    error_message: str | None = None,
) -> dict:
    experiment_dir = experiments_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "timestamp_utc": timestamp_utc,
        "pipeline": pipeline,
        "pipeline_version": pipeline_version,
        "model_name": "gpt-5.2-2025-12-11",
        "baseline_run_id": baseline_run_id,
        "parent_run_id": parent_run_id,
        "changed_components": changed_components or [],
        "change_reason": change_reason,
        "run_status": run_status,
        "failed_stage": failed_stage,
        "error_message": error_message,
        "runs": runs,
        "artifacts": {
            "experiment_dir": str(experiment_dir),
            "results_dir": str(experiment_dir / "results" / pipeline),
            "analysis_dir": str(experiment_dir / "analysis"),
            "generated_targets_dir": str(experiment_dir / "generated_targets"),
            "indexes_root": str(experiment_dir / "indexes"),
        },
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_experiment_log(experiments_root: Path, payloads: list[dict]) -> None:
    lines = [json.dumps(payload, sort_keys=True) for payload in payloads]
    (experiments_root / "experiment_log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_audit_builds_framework_rows_with_notes_and_repeat_stats(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    results_root.mkdir()

    docs = [
        "aapl.2023.targets.v1.json",
        "aapl.2024.targets.v1.json",
        "msft.2023.targets.v1.json",
        "msft.2024.targets.v1.json",
    ]
    report_1 = _write_report(
        tmp_path / "reports" / "run1.json",
        doc_names=docs,
        micro_f1=0.4,
        micro_precision=0.9,
        micro_recall=0.3,
        hallucination_rate=0.1,
    )
    report_2 = _write_report(
        tmp_path / "reports" / "run2.json",
        doc_names=docs,
        micro_f1=0.6,
        micro_precision=0.95,
        micro_recall=0.5,
        hallucination_rate=0.05,
    )

    payload = _write_manifest(
        experiments_root,
        run_id="run_base",
        run_label="gate1_no_rag",
        timestamp_utc="2026-02-10T10:00:00+00:00",
        runs=[
            _run_record(
                counter=1,
                report_path=report_1,
                prediction_files=4,
                micro_f1=0.4,
                micro_precision=0.9,
                micro_recall=0.3,
                hallucination_rate=0.1,
            ),
            _run_record(
                counter=2,
                report_path=report_2,
                prediction_files=4,
                micro_f1=0.6,
                micro_precision=0.95,
                micro_recall=0.5,
                hallucination_rate=0.05,
            ),
        ],
        changed_components=["retrieval"],
    )
    _write_experiment_log(experiments_root, [payload])

    notes_file = experiments_root / "run_notes.toml"
    notes_file.write_text(
        (
            "[aliases]\n"
            '"run_base" = "baseline no-rag"\n\n'
            "[descriptions]\n"
            '"run_base" = "Parity baseline before refactor"\n'
        ),
        encoding="utf-8",
    )

    summary = audit_existing_results(
        results_root,
        out_dir,
        experiments_root=experiments_root,
        run_notes_file=notes_file,
    )

    assert summary["framework_runs"] == 1
    rows = _read_csv(out_dir / "experiment_run_log.csv")
    assert len(rows) == 1

    row = rows[0]
    assert row["label"] == "baseline no-rag"
    assert row["change_description"] == "Parity baseline before refactor"
    assert row["n_runs"] == "2"
    assert row["n_docs"] == "4"
    assert row["n_companies"] == "2"
    assert row["n_years"] == "2"
    assert row["run_status"] == "completed"
    assert row["status"] == "complete"
    assert float(row["micro_f1_mean"]) == pytest.approx(0.5)
    assert float(row["micro_f1_std"]) == pytest.approx(0.1)

    jsonl_lines = (out_dir / "experiment_run_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 1


def test_audit_linked_comparisons_are_explicit_and_strict(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    results_root.mkdir()

    shared_docs = ["aapl.2023.targets.v1.json", "aapl.2024.targets.v1.json"]
    mismatched_docs = ["meta.2023.targets.v1.json"]

    baseline_report = _write_report(
        tmp_path / "reports" / "baseline.json",
        doc_names=shared_docs,
        micro_f1=0.4,
        micro_precision=0.8,
        micro_recall=0.3,
        hallucination_rate=0.2,
    )
    candidate_report = _write_report(
        tmp_path / "reports" / "candidate.json",
        doc_names=shared_docs,
        micro_f1=0.6,
        micro_precision=0.9,
        micro_recall=0.5,
        hallucination_rate=0.1,
    )
    mismatch_report = _write_report(
        tmp_path / "reports" / "mismatch.json",
        doc_names=mismatched_docs,
        micro_f1=0.55,
        micro_precision=0.85,
        micro_recall=0.45,
        hallucination_rate=0.12,
    )
    unlinked_report = _write_report(
        tmp_path / "reports" / "unlinked.json",
        doc_names=shared_docs,
        micro_f1=0.52,
        micro_precision=0.84,
        micro_recall=0.41,
        hallucination_rate=0.15,
    )

    _write_manifest(
        experiments_root,
        run_id="baseline_run",
        run_label="baseline",
        timestamp_utc="2026-02-10T10:00:00+00:00",
        runs=[
            _run_record(
                counter=1,
                report_path=baseline_report,
                prediction_files=2,
                micro_f1=0.4,
                micro_precision=0.8,
                micro_recall=0.3,
                hallucination_rate=0.2,
            )
        ],
    )
    _write_manifest(
        experiments_root,
        run_id="candidate_run",
        run_label="candidate",
        timestamp_utc="2026-02-10T11:00:00+00:00",
        baseline_run_id="baseline_run",
        runs=[
            _run_record(
                counter=1,
                report_path=candidate_report,
                prediction_files=2,
                micro_f1=0.6,
                micro_precision=0.9,
                micro_recall=0.5,
                hallucination_rate=0.1,
            )
        ],
    )
    _write_manifest(
        experiments_root,
        run_id="mismatch_run",
        run_label="mismatch",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        baseline_run_id="baseline_run",
        runs=[
            _run_record(
                counter=1,
                report_path=mismatch_report,
                prediction_files=1,
                micro_f1=0.55,
                micro_precision=0.85,
                micro_recall=0.45,
                hallucination_rate=0.12,
            )
        ],
    )
    _write_manifest(
        experiments_root,
        run_id="unlinked_run",
        run_label="unlinked",
        timestamp_utc="2026-02-10T13:00:00+00:00",
        runs=[
            _run_record(
                counter=1,
                report_path=unlinked_report,
                prediction_files=2,
                micro_f1=0.52,
                micro_precision=0.84,
                micro_recall=0.41,
                hallucination_rate=0.15,
            )
        ],
    )

    audit_existing_results(results_root, out_dir, experiments_root=experiments_root, run_notes_file=None)
    rows = _read_csv(out_dir / "linked_comparisons.csv")

    assert len(rows) == 2
    comparable = next(row for row in rows if row["candidate_run_id"] == "candidate_run")
    assert comparable["status"] == "comparable"
    assert float(comparable["delta_micro_f1"]) == pytest.approx(0.2)

    mismatch = next(row for row in rows if row["candidate_run_id"] == "mismatch_run")
    assert mismatch["status"] == "not_comparable"
    assert mismatch["reason"] == "doc_set_mismatch"
    assert all(row["candidate_run_id"] != "unlinked_run" for row in rows)


def test_audit_keeps_legacy_results_separate(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    (results_root / "rag").mkdir(parents=True)
    (results_root / "llm_only_no_rag").mkdir(parents=True)

    _write_report(
        results_root / "rag" / "a.json",
        doc_names=["aapl.2023.targets.v1.json"],
        micro_f1=0.7,
        micro_precision=0.8,
        micro_recall=0.6,
        hallucination_rate=0.2,
    )
    _write_report(
        results_root / "llm_only_no_rag" / "b.json",
        doc_names=["msft.2024.targets.v1.json"],
        micro_f1=0.5,
        micro_precision=0.9,
        micro_recall=0.4,
        hallucination_rate=0.1,
    )

    audit_existing_results(results_root, out_dir, experiments_root=experiments_root, run_notes_file=None)

    legacy_rows = _read_csv(out_dir / "legacy_results_log.csv")
    assert len(legacy_rows) == 2
    assert all(row["n_runs"] == "n/a" for row in legacy_rows)
    assert all(row["metadata_note"] == "legacy report (limited metadata)" for row in legacy_rows)

    markdown = (out_dir / "existing_results_audit.md").read_text(encoding="utf-8")
    assert "## Experiment Runs (Framework)" in markdown
    assert "No framework runs found." in markdown
    assert "## Legacy Historical Results (Separate Semantics)" in markdown


def test_audit_marks_incomplete_runs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    results_root.mkdir()

    # Missing report_path and prediction_files to simulate an aborted/incomplete run.
    _write_manifest(
        experiments_root,
        run_id="incomplete_run",
        run_label="incomplete",
        timestamp_utc="2026-02-10T09:30:00+00:00",
        runs=[
            {
                "counter": 1,
                "prediction_dir": "unused",
                "prediction_files": None,
                "report_path": None,
                "aggregate": None,
            }
        ],
    )

    audit_existing_results(results_root, out_dir, experiments_root=experiments_root, run_notes_file=None)
    rows = _read_csv(out_dir / "experiment_run_log.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_status"] == "incomplete"
    assert row["status"] == "incomplete"
    assert "missing_report_path" in row["status_reason"]


def test_audit_marks_aborted_runs_from_manifest_status(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    results_root.mkdir()

    _write_manifest(
        experiments_root,
        run_id="aborted_run",
        run_label="aborted",
        timestamp_utc="2026-02-10T09:45:00+00:00",
        runs=[
            {
                "counter": 1,
                "prediction_dir": "unused",
                "prediction_files": 1,
                "report_path": None,
                "aggregate": None,
            }
        ],
        run_status="aborted",
        failed_stage="evaluate",
        error_message="RuntimeError: judge API timeout",
    )

    audit_existing_results(results_root, out_dir, experiments_root=experiments_root, run_notes_file=None)
    rows = _read_csv(out_dir / "experiment_run_log.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_status"] == "aborted"
    assert row["status"] == "incomplete"
    assert "run_status:aborted" in row["status_reason"]
    assert "failed_stage:evaluate" in row["status_reason"]


def test_audit_uses_manifest_fallback_when_experiment_log_missing(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "audit"
    experiments_root.mkdir()
    results_root.mkdir()

    report = _write_report(
        tmp_path / "reports" / "fallback.json",
        doc_names=["aapl.2023.targets.v1.json"],
        micro_f1=0.3,
        micro_precision=0.7,
        micro_recall=0.2,
        hallucination_rate=0.3,
    )
    _write_manifest(
        experiments_root,
        run_id="fallback_run",
        run_label="fallback",
        timestamp_utc="2026-02-10T09:00:00+00:00",
        runs=[
            _run_record(
                counter=1,
                report_path=report,
                prediction_files=1,
                micro_f1=0.3,
                micro_precision=0.7,
                micro_recall=0.2,
                hallucination_rate=0.3,
            )
        ],
    )

    summary = audit_existing_results(results_root, out_dir, experiments_root=experiments_root, run_notes_file=None)

    assert summary["framework_runs"] == 1
    rows = _read_csv(out_dir / "experiment_run_log.csv")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "fallback_run"
