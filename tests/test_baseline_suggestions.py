from __future__ import annotations

import json
from pathlib import Path

from cte.baseline_suggestions import suggest_baselines


def _write_manifest(
    experiments_root: Path,
    *,
    run_id: str,
    pipeline_version: str,
    model_name: str,
    micro_f1: float,
    micro_precision: float,
    micro_recall: float,
    timestamp_utc: str,
) -> None:
    run_dir = experiments_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_label": run_id,
        "run_status": "completed",
        "pipeline": "rag",
        "pipeline_version": pipeline_version,
        "model_name": model_name,
        "judge_model_name": "gpt-5-mini-2025-08-07",
        "started_at_utc": timestamp_utc,
        "finished_at_utc": timestamp_utc,
        "timestamp_utc": timestamp_utc,
        "runs": [
            {
                "counter": 1,
                "prediction_dir": "unused",
                "prediction_files": 1,
                "report_path": None,
                "aggregate": {
                    "micro_f1": micro_f1,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                },
            }
        ],
        "artifacts": {"experiment_dir": str(run_dir)},
    }
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_suggest_baselines_uses_best_completed_run_when_no_existing_file(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir(parents=True)

    _write_manifest(
        experiments_root,
        run_id="run_low",
        pipeline_version="rag.v1",
        model_name="gpt-5.2-2025-12-11",
        micro_f1=0.4,
        micro_precision=0.8,
        micro_recall=0.3,
        timestamp_utc="2026-02-21T12:00:00+00:00",
    )
    _write_manifest(
        experiments_root,
        run_id="run_high",
        pipeline_version="rag.v1",
        model_name="gpt-5.2-2025-12-11",
        micro_f1=0.6,
        micro_precision=0.9,
        micro_recall=0.5,
        timestamp_utc="2026-02-21T12:10:00+00:00",
    )

    result = suggest_baselines(
        experiments_root=experiments_root,
        baselines_file=tmp_path / "docs" / "BASELINES.md",
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.action == "set"
    assert row.current_baseline_run_id is None
    assert row.suggested_run_id == "run_high"
    assert row.suggested_f1 == 0.6
    assert "candidate from completed runs" in row.notes


def test_suggest_baselines_flags_review_replace_when_candidate_changes(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir(parents=True)

    _write_manifest(
        experiments_root,
        run_id="old_baseline",
        pipeline_version="rag.v1",
        model_name="gpt-5.2-2025-12-11",
        micro_f1=0.45,
        micro_precision=0.7,
        micro_recall=0.35,
        timestamp_utc="2026-02-21T12:00:00+00:00",
    )
    _write_manifest(
        experiments_root,
        run_id="new_candidate",
        pipeline_version="rag.v1",
        model_name="gpt-5.2-2025-12-11",
        micro_f1=0.62,
        micro_precision=0.91,
        micro_recall=0.49,
        timestamp_utc="2026-02-21T12:20:00+00:00",
    )
    _write_manifest(
        experiments_root,
        run_id="keep_run",
        pipeline_version="no_rag.v1",
        model_name="gpt-5.2-2025-12-11",
        micro_f1=0.5,
        micro_precision=1.0,
        micro_recall=0.33,
        timestamp_utc="2026-02-21T12:05:00+00:00",
    )

    baselines_file = tmp_path / "docs" / "BASELINES.md"
    baselines_file.parent.mkdir(parents=True)
    baselines_file.write_text(
        "\n".join(
            [
                "# BASELINES",
                "",
                "| baseline_key | pipeline_version | model_generation | baseline_run_id | status | updated_at_utc | notes |",
                "|---|---|---|---|---|---|---|",
                "| rag_parity_baseline | rag.v1 | gpt-5.2-2025-12-11 | old_baseline | active | 2026-02-21T00:00:00+00:00 | existing row |",
                "| no_rag_anchor | no_rag.v1 | gpt-5.2-2025-12-11 | keep_run | active | 2026-02-21T00:00:00+00:00 | existing row |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = suggest_baselines(experiments_root=experiments_root, baselines_file=baselines_file)
    by_key = {(row.pipeline_version, row.model_generation): row for row in result.rows}

    rag_row = by_key[("rag.v1", "gpt-5.2-2025-12-11")]
    assert rag_row.action == "review_replace"
    assert rag_row.current_baseline_run_id == "old_baseline"
    assert rag_row.suggested_run_id == "new_candidate"

    no_rag_row = by_key[("no_rag.v1", "gpt-5.2-2025-12-11")]
    assert no_rag_row.action == "keep"
    assert no_rag_row.current_baseline_run_id == "keep_run"
    assert no_rag_row.suggested_run_id == "keep_run"
