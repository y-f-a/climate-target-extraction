from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cte.dashboard.data import load_dashboard_snapshot


def _write_report(
    path: Path,
    *,
    micro_f1: float,
    micro_precision: float,
    micro_recall: float,
    field_acc_macro: dict[str, float] | None = None,
    per_doc: list[dict] | None = None,
    usage: dict | None = None,
    cost: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "field_acc_macro": field_acc_macro or {},
        },
        "per_doc": per_doc or [],
    }
    if usage is not None:
        payload["usage"] = usage
    if cost is not None:
        payload["cost"] = cost
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(
    experiments_root: Path,
    *,
    run_id: str,
    run_status: str | None,
    timestamp_utc: str,
    runs: list[dict],
    pipeline: str = "no_rag",
    pipeline_version: str = "no_rag.v1",
    index_events: list[dict] | None = None,
) -> Path:
    experiment_dir = experiments_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "run_label": run_id,
        "timestamp_utc": timestamp_utc,
        "started_at_utc": timestamp_utc,
        "finished_at_utc": timestamp_utc if run_status == "completed" else None,
        "run_status": run_status,
        "pipeline": pipeline,
        "pipeline_version": pipeline_version,
        "model_name": "gpt-5.2-2025-12-11",
        "judge_model_name": "gpt-5-mini-2025-08-07",
        "index_events": index_events or [],
        "runs": runs,
        "artifacts": {"experiment_dir": str(experiment_dir)},
    }
    path = experiment_dir / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_index_manifest(
    index_dir: Path,
    *,
    source_root: str,
    source_manifest: list[dict],
) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_id": index_dir.name,
        "index_action": "reused",
        "index_fingerprint": "fake",
        "source_root": source_root,
        "source_manifest": source_manifest,
    }
    path = index_dir / "index_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dashboard_snapshot_sorts_completed_runs_by_f1_and_marks_stalled_active(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    high_report = _write_report(
        tmp_path / "reports" / "high.json",
        micro_f1=0.9,
        micro_precision=0.95,
        micro_recall=0.8,
    )
    low_report = _write_report(
        tmp_path / "reports" / "low.json",
        micro_f1=0.3,
        micro_precision=0.5,
        micro_recall=0.2,
    )

    _write_manifest(
        experiments_root,
        run_id="completed_high",
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {
                "counter": 1,
                "report_path": str(high_report),
                "aggregate": {
                    "micro_f1": 0.9,
                    "micro_precision": 0.95,
                    "micro_recall": 0.8,
                },
            }
        ],
    )
    _write_manifest(
        experiments_root,
        run_id="completed_low",
        run_status="completed",
        timestamp_utc="2026-02-10T11:00:00+00:00",
        runs=[
            {
                "counter": 1,
                "report_path": str(low_report),
                "aggregate": {
                    "micro_f1": 0.3,
                    "micro_precision": 0.5,
                    "micro_recall": 0.2,
                },
            }
        ],
    )
    active_manifest = _write_manifest(
        experiments_root,
        run_id="active_run",
        run_status="running",
        timestamp_utc="2026-02-10T13:00:00+00:00",
        runs=[],
    )

    now_utc = datetime(2026, 2, 10, 13, 30, tzinfo=UTC)
    stale_time = (now_utc - timedelta(minutes=61)).timestamp()
    os.utime(active_manifest, (stale_time, stale_time))

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=now_utc)

    assert len(snapshot.active_runs) == 1
    assert snapshot.active_runs[0].run_id == "active_run"
    assert snapshot.active_runs[0].stalled is True
    assert "stalled >20m without progress update" in snapshot.active_runs[0].warnings
    assert "manifest has no run entries" not in snapshot.active_runs[0].warnings
    assert "missing metrics" not in snapshot.active_runs[0].warnings
    assert snapshot.active_runs[0].metrics_pending is True
    assert snapshot.active_runs[0].metric_runs_display == "-"

    completed_ids = [row.run_id for row in snapshot.completed_runs]
    assert completed_ids[:2] == ["completed_high", "completed_low"]
    assert snapshot.completed_runs[0].metric_runs_display == "1"


def test_dashboard_snapshot_uses_report_fallback_and_handles_invalid_manifest(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "fallback_run"
    experiment_dir = experiments_root / run_id
    relative_report = Path("results/no_rag/report.json")
    report_path = _write_report(
        experiment_dir / relative_report,
        micro_f1=0.7,
        micro_precision=0.8,
        micro_recall=0.65,
        usage={"total_tokens": 180, "prompt_tokens": 120, "completion_tokens": 60},
        cost={"total_cost_usd": 0.0134},
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {
                "counter": 1,
                "report_path": str(relative_report),
                "aggregate": {},
            }
        ],
    )

    invalid_dir = experiments_root / "invalid_run"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "manifest.json").write_text("{not-valid-json", encoding="utf-8")

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))

    run = snapshot.runs_by_id[run_id]
    assert run.f1 == pytest.approx(0.7)
    assert run.precision == pytest.approx(0.8)
    assert run.recall == pytest.approx(0.65)
    assert run.tokens_text == "total=180, prompt=120, completion=60"
    assert run.cost_text == "$0.0134"
    assert run.report_links[0].exists is True
    assert Path(run.report_links[0].link_path) == report_path.resolve()
    assert "missing metrics" not in run.warnings
    assert run.metric_runs_display == "1"

    invalid = snapshot.runs_by_id["invalid_run"]
    assert invalid.run_status == "invalid_manifest"
    assert "invalid manifest JSON" in invalid.warnings


def test_dashboard_snapshot_overlays_live_status_and_uses_live_update_for_stalled(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "active_with_live_status"
    active_manifest = _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="running",
        timestamp_utc="2026-02-10T13:00:00+00:00",
        runs=[],
    )

    now_utc = datetime(2026, 2, 10, 13, 30, tzinfo=UTC)
    stale_manifest_time = (now_utc - timedelta(minutes=61)).timestamp()
    os.utime(active_manifest, (stale_manifest_time, stale_manifest_time))

    live_status_path = experiments_root / run_id / "live_status.json"
    live_status_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "job_kind": "experiment_run",
                "run_id": run_id,
                "status": "running",
                "started_at_utc": "2026-02-10T13:00:00+00:00",
                "updated_at_utc": "2026-02-10T13:25:00+00:00",
                "stalled_after_seconds": 1200,
                "stage": "extract",
                "run_counter_current": 1,
                "run_count_total": 2,
                "extract_progress": {
                    "completed": 6,
                    "total": 14,
                    "current_doc_name": "meta.2024.targets.v1.json",
                },
                "evaluate_progress": {
                    "completed": 0,
                    "total": 14,
                    "current_doc_name": None,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=now_utc)
    run = snapshot.runs_by_id[run_id]
    assert run.stalled is False
    assert run.live_stage == "extract"
    assert run.live_run_counter_current == 1
    assert run.live_run_counter_total == 2
    assert run.live_extract_completed == 6
    assert run.live_extract_total == 14
    assert run.live_extract_current_doc_name == "meta.2024.targets.v1.json"
    assert run.live_evaluate_completed == 0
    assert run.live_evaluate_total == 14

    live_status_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "job_kind": "experiment_run",
                "run_id": run_id,
                "status": "running",
                "started_at_utc": "2026-02-10T13:00:00+00:00",
                "updated_at_utc": "2026-02-10T13:05:00+00:00",
                "stalled_after_seconds": 1200,
            }
        ),
        encoding="utf-8",
    )
    stale_snapshot = load_dashboard_snapshot(experiments_root, now_utc=now_utc)
    stale_run = stale_snapshot.runs_by_id[run_id]
    assert stale_run.stalled is True
    assert "stalled >20m without progress update" in stale_run.warnings


def test_dashboard_snapshot_includes_active_parse_cache_jobs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    parsed_docs_root = tmp_path / "parsed_docs"
    run_root = parsed_docs_root / "_runs"
    run_root.mkdir(parents=True)

    active_dir = run_root / "20260223T000000000000Z-parse_cache_build"
    active_dir.mkdir(parents=True)
    (active_dir / "live_status.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "job_kind": "parse_cache_build",
                "run_id": "20260223T000000000000Z-parse_cache_build",
                "status": "running",
                "mode": "execute",
                "started_at_utc": "2026-02-23T00:00:00+00:00",
                "updated_at_utc": "2026-02-23T00:02:00+00:00",
                "processed": 3,
                "total": 10,
                "current_source_relative_path": "nvda/2024/FY2024-NVIDIA-Corporate-Sustainability-Report.pdf",
                "hits": 1,
                "planned_new": 0,
                "parsed": 2,
                "failed": 0,
                "stalled_after_seconds": 1200,
            }
        ),
        encoding="utf-8",
    )

    completed_dir = run_root / "20260223T000100000000Z-parse_cache_build"
    completed_dir.mkdir(parents=True)
    (completed_dir / "live_status.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "job_kind": "parse_cache_build",
                "run_id": "20260223T000100000000Z-parse_cache_build",
                "status": "completed",
                "mode": "execute",
                "processed": 10,
                "total": 10,
                "stalled_after_seconds": 1200,
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_dashboard_snapshot(
        experiments_root,
        parsed_docs_root=parsed_docs_root,
        now_utc=datetime(2026, 2, 23, 0, 3, tzinfo=UTC),
    )
    assert len(snapshot.parse_cache_active_runs) == 1
    active = snapshot.parse_cache_active_runs[0]
    assert active.run_id == "20260223T000000000000Z-parse_cache_build"
    assert active.status == "running"
    assert active.mode == "execute"
    assert active.processed == 3
    assert active.total == 10
    assert active.current_source_relative_path is not None


def test_dashboard_snapshot_displays_standard_error_for_multi_run_metrics(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    _write_manifest(
        experiments_root,
        run_id="two_run_metrics",
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {
                "counter": 1,
                "report_path": None,
                "aggregate": {
                    "micro_f1": 0.6,
                    "micro_precision": 0.8,
                    "micro_recall": 0.4,
                },
            },
            {
                "counter": 2,
                "report_path": None,
                "aggregate": {
                    "micro_f1": 0.8,
                    "micro_precision": 1.0,
                    "micro_recall": 0.6,
                },
            },
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id["two_run_metrics"]

    assert run.f1 == pytest.approx(0.7)
    assert run.f1_se == pytest.approx(0.1)
    assert run.f1_display == "0.7000 ± 0.1000"
    assert run.recall_display == "0.5000 ± 0.1000"
    assert run.precision_display == "0.9000 ± 0.1000"
    assert run.metric_runs_display == "2"


def test_dashboard_diagnostics_chooses_highest_counter_report_when_readable(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_highest_counter"
    experiment_dir = experiments_root / run_id
    low_report_rel = Path("results/counter_1.json")
    high_report_rel = Path("results/counter_2.json")
    _write_report(
        experiment_dir / low_report_rel,
        micro_f1=0.5,
        micro_precision=0.5,
        micro_recall=0.5,
        per_doc=[
            {"doc": "aapl.2024.targets.v1.json", "tp": 1, "fp": 1, "fn": 1},
        ],
    )
    _write_report(
        experiment_dir / high_report_rel,
        micro_f1=0.7,
        micro_precision=0.8,
        micro_recall=0.6,
        per_doc=[
            {"doc": "msft.2024.targets.v1.json", "tp": 3, "fp": 1, "fn": 0},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {"counter": 1, "report_path": str(low_report_rel), "aggregate": {}},
            {"counter": 2, "report_path": str(high_report_rel), "aggregate": {}},
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]

    assert run.diagnostics_source_counter == 2
    assert run.diagnostics_source_report_path == str(high_report_rel)
    assert run.diagnostics_warnings == []
    assert [row.company_year for row in run.company_year_rows] == ["MSFT 2024"]


def test_dashboard_diagnostics_falls_back_to_lower_counter_report_with_warning(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_fallback"
    experiment_dir = experiments_root / run_id
    fallback_report_rel = Path("results/counter_1.json")
    _write_report(
        experiment_dir / fallback_report_rel,
        micro_f1=0.6,
        micro_precision=0.6,
        micro_recall=0.6,
        per_doc=[
            {"doc": "goog.2024.targets.v1.json", "tp": 2, "fp": 1, "fn": 1},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {"counter": 1, "report_path": str(fallback_report_rel), "aggregate": {}},
            {"counter": 2, "report_path": "results/missing_counter_2.json", "aggregate": {}},
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]

    assert run.diagnostics_source_counter == 1
    assert run.diagnostics_source_report_path == str(fallback_report_rel)
    assert "using fallback report because latest counter report was missing or unreadable" in run.diagnostics_warnings


def test_dashboard_diagnostics_company_year_aggregation_and_sorting(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_company_year"
    report_path = _write_report(
        experiments_root / run_id / "results/report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "aapl.2024.first.targets.json", "tp": 2, "fp": 1, "fn": 0},
            {"doc": "aapl.2024.second.targets.json", "tp": 1, "fp": 0, "fn": 1},
            {"doc": "msft.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 3},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[{"counter": 1, "report_path": str(report_path), "aggregate": {}}],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]

    assert [row.company_year for row in run.company_year_rows] == ["MSFT 2024", "AAPL 2024"]

    aapl_row = next(row for row in run.company_year_rows if row.company_year == "AAPL 2024")
    assert aapl_row.tp == 3
    assert aapl_row.fp == 1
    assert aapl_row.fn == 1
    assert aapl_row.recall == pytest.approx(0.75)
    assert aapl_row.precision == pytest.approx(0.75)
    assert aapl_row.f1 == pytest.approx(0.75)
    assert aapl_row.source_pdf_mapping_note == "source PDF mapping unavailable for non-RAG runs"
    assert aapl_row.source_pdf_rows == []
    assert aapl_row.source_pdf_warnings == []

    msft_row = next(row for row in run.company_year_rows if row.company_year == "MSFT 2024")
    assert msft_row.source_pdf_mapping_note == "source PDF mapping unavailable for non-RAG runs"
    assert msft_row.source_pdf_rows == []
    assert msft_row.source_pdf_warnings == []


def test_dashboard_diagnostics_company_year_zero_denominator_behavior(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_zero_denominator"
    report_path = _write_report(
        experiments_root / run_id / "results/report.json",
        micro_f1=1.0,
        micro_precision=1.0,
        micro_recall=1.0,
        per_doc=[
            {"doc": "meta.2024.targets.v1.json", "tp": 0, "fp": 0, "fn": 0},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[{"counter": 1, "report_path": str(report_path), "aggregate": {}}],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    row = run.company_year_rows[0]

    assert row.company_year == "META 2024"
    assert row.precision == pytest.approx(1.0)
    assert row.recall == pytest.approx(1.0)
    assert row.f1 == pytest.approx(1.0)


def test_dashboard_diagnostics_field_summary_uses_macro_accuracy_and_missing_counts(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_field_summary"
    report_path = _write_report(
        experiments_root / run_id / "results/report.json",
        micro_f1=0.6,
        micro_precision=0.6,
        micro_recall=0.6,
        field_acc_macro={"target_year": 0.4, "base_year": 0.9},
        per_doc=[
            {
                "doc": "aapl.2024.targets.v1.json",
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "field_accuracy": {
                    "target_year": 1.0,
                    "base_year": None,
                    "scope": 0.5,
                },
            },
            {
                "doc": "msft.2024.targets.v1.json",
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "field_accuracy": {
                    "target_year": None,
                    "base_year": 0.2,
                },
            },
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[{"counter": 1, "report_path": str(report_path), "aggregate": {}}],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    rows_by_field = {row.field: row for row in run.field_summary_rows}

    assert [row.field for row in run.field_summary_rows] == ["target_year", "base_year", "scope"]

    target_year_row = rows_by_field["target_year"]
    assert target_year_row.accuracy == pytest.approx(0.4)
    assert target_year_row.company_years_with_score == 1
    assert target_year_row.missing_count == 1

    base_year_row = rows_by_field["base_year"]
    assert base_year_row.accuracy == pytest.approx(0.9)
    assert base_year_row.company_years_with_score == 1
    assert base_year_row.missing_count == 1

    scope_row = rows_by_field["scope"]
    assert scope_row.accuracy is None
    assert scope_row.company_years_with_score == 1
    assert scope_row.missing_count == 0


def test_dashboard_diagnostics_missing_report_returns_empty_diagnostics(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_missing_report"
    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[
            {"counter": 2, "report_path": "results/missing_2.json", "aggregate": {}},
            {"counter": 1, "report_path": "results/missing_1.json", "aggregate": {}},
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]

    assert run.diagnostics_source_report_path is None
    assert run.diagnostics_source_report_link is None
    assert run.diagnostics_source_counter is None
    assert run.company_year_rows == []
    assert run.field_summary_rows == []
    assert "no readable report available for diagnostics" in run.diagnostics_warnings


def test_dashboard_diagnostics_rag_maps_evaluation_item_to_source_pdfs_and_citations(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_rag_mapping"
    experiment_dir = experiments_root / run_id
    report_path = _write_report(
        experiment_dir / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "nvda.2024.targets.v1.json", "tp": 2, "fp": 1, "fn": 1},
        ],
    )

    prediction_dir = experiment_dir / "generated_targets" / "rag" / "gpt5_2" / "run_1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "nvda.2024.targets.v1.json").write_text(
        json.dumps(
            {
                "company": "NVIDIA",
                "targets": [
                    {
                        "sources": [
                            "FY2024-NVIDIA-Corporate-Sustainability-Report.pdf#p8",
                            "FY2024-NVIDIA-Corporate-Sustainability-Report.pdf#p8",
                            "FY2024-NVIDIA-Corporate-Sustainability-Report.pdf#p12",
                            "not-in-index.pdf#p2",
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index_dir = tmp_path / "indexes" / "rag.v1" / "nvda_2024"
    _write_index_manifest(
        index_dir,
        source_root=str(tmp_path / "source_docs" / "nvda" / "2024"),
        source_manifest=[
            {"relative_path": "FY2024-NVIDIA-Corporate-Sustainability-Report.pdf"},
            {"relative_path": "NVIDIA-2024-Annual-Report.pdf"},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        index_events=[
            {
                "doc": "nvda.2024.targets.v1.json",
                "index_dir": str(index_dir),
            }
        ],
        runs=[
            {
                "counter": 1,
                "report_path": str(report_path),
                "prediction_dir": str(prediction_dir),
                "aggregate": {},
            }
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    company_row = run.company_year_rows[0]
    assert company_row.source_pdf_mapping_note is None
    assert len(company_row.source_pdf_rows) == 2

    pdf_rows = {row.filename: row for row in company_row.source_pdf_rows}
    cited_row = pdf_rows["FY2024-NVIDIA-Corporate-Sustainability-Report.pdf"]
    assert cited_row.cited is True
    assert cited_row.citation_count == 3
    assert cited_row.pages == [8, 12]
    assert cited_row.source_link is None

    uncited_row = pdf_rows["NVIDIA-2024-Annual-Report.pdf"]
    assert uncited_row.cited is False
    assert uncited_row.citation_count == 0
    assert uncited_row.pages == []

    assert any(
        "nvda.2024.targets.v1.json: citation sources not found in indexed PDFs: not-in-index.pdf"
        in warning
        for warning in company_row.source_pdf_warnings
    )
    assert any(
        "nvda.2024.targets.v1.json: citation sources not found in indexed PDFs: not-in-index.pdf"
        in warning
        for warning in run.diagnostics_warnings
    )


def test_dashboard_diagnostics_rag_includes_retrieved_chunks_from_sidecar(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_rag_retrieved_chunks"
    experiment_dir = experiments_root / run_id
    report_path = _write_report(
        experiment_dir / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "msft.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 1},
        ],
    )

    prediction_dir = experiment_dir / "generated_targets" / "rag" / "gpt5_2" / "run_1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "msft.2024.targets.v1.json").write_text(
        json.dumps({"company": "Microsoft", "targets": []}),
        encoding="utf-8",
    )
    (prediction_dir / "_retrieved_chunks.json").write_text(
        json.dumps(
            {
                "query_prompt": "extract targets",
                "retrieval_rerank_profile": "off",
                "documents": [
                    {
                        "doc": "msft.2024.targets.v1.json",
                        "retrieved_count": 2,
                        "retrieved_chunks": [
                            {
                                "rank": 1,
                                "score": 0.92,
                                "metadata": {
                                    "file_name": "Microsoft-Report.pdf",
                                    "source_relative_path": "msft/2024/Microsoft-Report.pdf",
                                    "page": 13,
                                },
                                "text": "Target text one",
                                "text_length": 15,
                                "text_sha256": "abc123",
                            },
                            {
                                "rank": 2,
                                "score": 0.77,
                                "metadata": {
                                    "source_relative_path": "msft/2024/Appendix.pdf",
                                    "page": 2,
                                },
                                "text": "Target text two",
                                "text_length": 15,
                                "text_sha256": "def456",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index_dir = tmp_path / "indexes" / "rag.v1" / "msft_2024"
    _write_index_manifest(
        index_dir,
        source_root=str(tmp_path / "source_docs" / "msft" / "2024"),
        source_manifest=[{"relative_path": "Microsoft-Report.pdf"}],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        index_events=[
            {
                "doc": "msft.2024.targets.v1.json",
                "index_dir": str(index_dir),
            }
        ],
        runs=[
            {
                "counter": 1,
                "report_path": str(report_path),
                "prediction_dir": str(prediction_dir),
                "aggregate": {},
            }
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    company_row = run.company_year_rows[0]

    assert company_row.retrieved_chunk_mapping_note is None
    assert company_row.retrieved_chunk_warnings == []
    assert company_row.retrieved_chunk_count == 2
    assert len(company_row.retrieved_chunk_rows) == 2

    first_chunk = company_row.retrieved_chunk_rows[0]
    assert first_chunk.evaluation_item_id == "msft.2024.targets.v1.json"
    assert first_chunk.rank == 1
    assert first_chunk.score == pytest.approx(0.92)
    assert first_chunk.file_name == "Microsoft-Report.pdf"
    assert first_chunk.source_relative_path == "msft/2024/Microsoft-Report.pdf"
    assert first_chunk.page == 13
    assert first_chunk.text == "Target text one"

    second_chunk = company_row.retrieved_chunk_rows[1]
    assert second_chunk.file_name == "Appendix.pdf"
    assert second_chunk.source_relative_path == "msft/2024/Appendix.pdf"
    assert second_chunk.page == 2


def test_dashboard_diagnostics_no_rag_shows_mapping_unavailable_note(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_no_rag_mapping_note"
    report_path = _write_report(
        experiments_root / run_id / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "aapl.2024.targets.v1.json", "tp": 1, "fp": 1, "fn": 1},
        ],
    )
    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        runs=[{"counter": 1, "report_path": str(report_path), "aggregate": {}}],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    company_row = snapshot.runs_by_id[run_id].company_year_rows[0]

    assert company_row.source_pdf_mapping_note == "source PDF mapping unavailable for non-RAG runs"
    assert company_row.source_pdf_rows == []


def test_dashboard_diagnostics_rag_missing_retrieved_chunks_sidecar_sets_mapping_note(
    tmp_path: Path,
) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_rag_missing_retrieved_chunks"
    report_path = _write_report(
        experiments_root / run_id / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "aapl.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 1},
        ],
    )

    prediction_dir = experiments_root / run_id / "generated_targets" / "rag" / "gpt5_2" / "run_1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "aapl.2024.targets.v1.json").write_text(
        json.dumps({"company": "Apple", "targets": []}),
        encoding="utf-8",
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        runs=[
            {
                "counter": 1,
                "report_path": str(report_path),
                "prediction_dir": str(prediction_dir),
                "aggregate": {},
            }
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    company_row = run.company_year_rows[0]

    assert company_row.retrieved_chunk_rows == []
    assert company_row.retrieved_chunk_mapping_note == "retrieved chunk mapping unavailable"
    assert any(
        "retrieved chunk mapping unavailable: _retrieved_chunks.json is missing" in warning
        for warning in run.diagnostics_warnings
    )


def test_dashboard_diagnostics_rag_missing_index_event_keeps_item_and_warns(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_rag_missing_index_event"
    report_path = _write_report(
        experiments_root / run_id / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "msft.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 1},
        ],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        runs=[{"counter": 1, "report_path": str(report_path), "prediction_dir": "predictions", "aggregate": {}}],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    company_row = run.company_year_rows[0]

    assert company_row.source_pdf_mapping_note == "source PDF mapping unavailable"
    assert company_row.source_pdf_rows == []
    assert any(
        "msft.2024.targets.v1.json: source PDF mapping unavailable: no index event found" in warning
        for warning in company_row.source_pdf_warnings
    )
    assert any(
        "msft.2024.targets.v1.json: source PDF mapping unavailable: no index event found"
        in warning
        for warning in run.diagnostics_warnings
    )


def test_dashboard_diagnostics_rag_multiple_index_dirs_uses_most_recent_and_warns(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_rag_multiple_index_dirs"
    experiment_dir = experiments_root / run_id
    report_path = _write_report(
        experiment_dir / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "msft.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 1},
        ],
    )

    prediction_dir = experiment_dir / "generated_targets" / "rag" / "gpt5_2" / "run_1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "msft.2024.targets.v1.json").write_text(
        json.dumps({"company": "Microsoft", "targets": [{"sources": ["B.pdf#p9"]}]}),
        encoding="utf-8",
    )

    index_dir_a = tmp_path / "indexes" / "rag.v1" / "a"
    index_dir_b = tmp_path / "indexes" / "rag.v1" / "b"
    _write_index_manifest(
        index_dir_a,
        source_root=str(tmp_path / "docs_a"),
        source_manifest=[{"relative_path": "A.pdf"}],
    )
    _write_index_manifest(
        index_dir_b,
        source_root=str(tmp_path / "docs_b"),
        source_manifest=[{"relative_path": "B.pdf"}],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        index_events=[
            {"doc": "msft.2024.targets.v1.json", "index_dir": str(index_dir_a)},
            {"doc": "msft.2024.targets.v1.json", "index_dir": str(index_dir_b)},
        ],
        runs=[
            {
                "counter": 1,
                "report_path": str(report_path),
                "prediction_dir": str(prediction_dir),
                "aggregate": {},
            }
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    run = snapshot.runs_by_id[run_id]
    company_row = run.company_year_rows[0]

    assert company_row.source_pdf_mapping_note is None
    assert [row.filename for row in company_row.source_pdf_rows] == ["B.pdf"]
    assert any(
        "msft.2024.targets.v1.json: multiple index mappings found; using most recent event" in warning
        for warning in company_row.source_pdf_warnings
    )
    assert any(
        "msft.2024.targets.v1.json: multiple index mappings found; using most recent event"
        in warning
        for warning in run.diagnostics_warnings
    )


def test_dashboard_diagnostics_source_pdf_link_enabled_only_inside_repo_root(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    run_id = "diagnostics_source_pdf_links"
    experiment_dir = experiments_root / run_id
    report_path = _write_report(
        experiment_dir / "results" / "report.json",
        micro_f1=0.6,
        micro_precision=0.7,
        micro_recall=0.5,
        per_doc=[
            {"doc": "aapl.2024.targets.v1.json", "tp": 1, "fp": 0, "fn": 1},
        ],
    )

    prediction_dir = experiment_dir / "generated_targets" / "rag" / "gpt5_2" / "run_1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "aapl.2024.targets.v1.json").write_text(
        json.dumps({"company": "Apple", "targets": []}),
        encoding="utf-8",
    )

    inside_index_dir = tmp_path / "indexes" / "rag.v1" / "inside_repo"
    outside_index_dir = tmp_path / "indexes" / "rag.v1" / "outside_repo"
    _write_index_manifest(
        inside_index_dir,
        source_root=str(Path.cwd()),
        source_manifest=[{"relative_path": "docs/fake-in-repo.pdf"}],
    )
    _write_index_manifest(
        outside_index_dir,
        source_root=str(tmp_path / "external_docs"),
        source_manifest=[{"relative_path": "outside.pdf"}],
    )

    _write_manifest(
        experiments_root,
        run_id=run_id,
        run_status="completed",
        timestamp_utc="2026-02-10T12:00:00+00:00",
        pipeline="rag",
        pipeline_version="rag.v1",
        index_events=[
            {"doc": "aapl.2024.targets.v1.json", "index_dir": str(outside_index_dir)},
            {"doc": "aapl.2024.targets.v1.json", "index_dir": str(inside_index_dir)},
        ],
        runs=[
            {
                "counter": 1,
                "report_path": str(report_path),
                "prediction_dir": str(prediction_dir),
                "aggregate": {},
            }
        ],
    )

    snapshot = load_dashboard_snapshot(experiments_root, now_utc=datetime(2026, 2, 10, 13, tzinfo=UTC))
    company_row = snapshot.runs_by_id[run_id].company_year_rows[0]
    source_pdf_row = company_row.source_pdf_rows[0]

    assert source_pdf_row.filename == "fake-in-repo.pdf"
    assert source_pdf_row.source_link is not None
