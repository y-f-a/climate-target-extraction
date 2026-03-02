from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .io import ensure_dir, read_json

CORE_METRICS = ("micro_f1", "micro_precision", "micro_recall", "hallucination_rate")
TRACKS = ("rag", "llm_only_no_rag")
DOC_PATTERN = re.compile(r"^(?P<ticker>[a-z0-9]+)\.(?P<year>\d{4})\.targets\.v1\.json$")


def _safe_delta(candidate: Any, baseline: Any) -> float | None:
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return float(candidate) - float(baseline)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metric_stats(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return (mean(values), pstdev(values))


def _norm_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


def _doc_signature(doc_names: set[str]) -> str | None:
    if not doc_names:
        return None
    joined = "\n".join(sorted(doc_names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _parse_doc(doc_name: str) -> tuple[str, str] | None:
    match = DOC_PATTERN.match(doc_name)
    if match is None:
        return None
    return (match.group("ticker").upper(), match.group("year"))


def _resolve_report_path(path_like: Any, experiment_dir: Path | None) -> Path | None:
    raw = _norm_text(path_like)
    if raw is None:
        return None

    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if experiment_dir is not None and not path.is_absolute():
        candidates.append(experiment_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_run_notes(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}, {}

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    aliases_raw = payload.get("aliases", {})
    descriptions_raw = payload.get("descriptions", {})

    aliases = {
        str(key): str(value).strip()
        for key, value in aliases_raw.items()
        if str(value).strip()
    }
    descriptions = {
        str(key): str(value).strip()
        for key, value in descriptions_raw.items()
        if str(value).strip()
    }
    return aliases, descriptions


def _load_experiment_payloads(experiments_root: Path) -> list[dict[str, Any]]:
    payloads_by_run_id: dict[str, dict[str, Any]] = {}

    log_path = experiments_root / "experiment_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            run_id = _norm_text(payload.get("run_id"))
            if run_id is None:
                continue
            payloads_by_run_id[run_id] = payload

    for manifest_path in sorted(experiments_root.glob("*/manifest.json")):
        try:
            payload = read_json(manifest_path)
        except Exception:
            continue
        run_id = _norm_text(payload.get("run_id")) or manifest_path.parent.name
        if run_id not in payloads_by_run_id:
            payloads_by_run_id[run_id] = payload

    rows = list(payloads_by_run_id.values())
    rows.sort(key=lambda item: _norm_text(item.get("timestamp_utc")) or "", reverse=True)
    return rows


def _derive_change_description(
    payload: dict[str, Any],
    *,
    override: str | None,
) -> str:
    if override:
        return override

    change_reason = _norm_text(payload.get("change_reason"))
    if change_reason:
        return change_reason

    changed_components = payload.get("changed_components", [])
    if isinstance(changed_components, list):
        parts = [str(item).strip() for item in changed_components if str(item).strip()]
        if parts:
            return "changed components: " + ", ".join(parts)

    if _norm_text(payload.get("baseline_run_id")) or _norm_text(payload.get("parent_run_id")):
        return "lineage-linked run"

    return "parity/default configuration"


def _build_framework_rows(
    payloads: list[dict[str, Any]],
    *,
    aliases: dict[str, str],
    descriptions: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        run_id = _norm_text(payload.get("run_id"))
        if run_id is None:
            continue

        run_records = payload.get("runs", [])
        if not isinstance(run_records, list):
            run_records = []

        n_runs = len(run_records)
        prediction_counts: list[int] = []
        doc_names: set[str] = set()
        company_set: set[str] = set()
        year_set: set[str] = set()
        metric_values: dict[str, list[float]] = {metric: [] for metric in CORE_METRICS}
        status_reasons: list[str] = []
        explicit_run_status = _norm_text(payload.get("run_status"))
        explicit_failed_stage = _norm_text(payload.get("failed_stage"))
        explicit_error_message = _norm_text(payload.get("error_message"))

        experiment_dir_value = payload.get("artifacts", {}).get("experiment_dir")
        experiment_dir = Path(experiment_dir_value) if _norm_text(experiment_dir_value) else None

        for record in run_records:
            if not isinstance(record, dict):
                continue

            prediction_files = record.get("prediction_files")
            if isinstance(prediction_files, int):
                prediction_counts.append(prediction_files)
            else:
                status_reasons.append("missing_prediction_count")

            aggregate = record.get("aggregate", {})
            if isinstance(aggregate, dict):
                for metric in CORE_METRICS:
                    value = _as_float(aggregate.get(metric))
                    if value is not None:
                        metric_values[metric].append(value)
            else:
                status_reasons.append("missing_aggregate")

            report_path = _resolve_report_path(record.get("report_path"), experiment_dir)
            if report_path is None:
                status_reasons.append("missing_report_path")
                continue

            try:
                report_payload = read_json(report_path)
            except Exception:
                status_reasons.append("unreadable_report")
                continue

            per_doc = report_payload.get("per_doc", [])
            if not isinstance(per_doc, list):
                continue

            for item in per_doc:
                if not isinstance(item, dict):
                    continue
                doc_name = _norm_text(item.get("doc"))
                if doc_name is None:
                    continue
                doc_names.add(doc_name)
                parsed = _parse_doc(doc_name)
                if parsed is not None:
                    ticker, year = parsed
                    company_set.add(ticker)
                    year_set.add(year)

        n_docs: int | None = None
        n_docs_consistent: bool | None = None
        n_docs_values: str | None = None
        if prediction_counts:
            unique_counts = sorted(set(prediction_counts))
            n_docs_values = ",".join(str(value) for value in unique_counts)
            n_docs_consistent = len(unique_counts) == 1
            if n_docs_consistent:
                n_docs = unique_counts[0]
            else:
                status_reasons.append("inconsistent_prediction_counts")
        else:
            status_reasons.append("missing_prediction_counts")

        if not doc_names:
            status_reasons.append("missing_doc_set")
        inferred_complete = not status_reasons and bool(doc_names)

        if explicit_run_status in {"running", "completed", "aborted"}:
            run_status = explicit_run_status
        else:
            run_status = "completed" if inferred_complete else "incomplete"

        if run_status == "running":
            status_reasons.append("run_status:running")
        if run_status == "aborted":
            status_reasons.append("run_status:aborted")
            if explicit_failed_stage:
                status_reasons.append(f"failed_stage:{explicit_failed_stage}")
            if explicit_error_message:
                status_reasons.append(f"error:{explicit_error_message}")
        if run_status == "completed" and not inferred_complete:
            status_reasons.append("completed_with_missing_artifacts")

        status = "complete" if run_status == "completed" else "incomplete"
        row: dict[str, Any] = {
            "run_id": run_id,
            "label": aliases.get(run_id, _norm_text(payload.get("run_label")) or run_id),
            "pipeline": _norm_text(payload.get("pipeline")),
            "pipeline_version": _norm_text(payload.get("pipeline_version")),
            "model_name": _norm_text(payload.get("model_name")),
            "timestamp_utc": _norm_text(payload.get("timestamp_utc")),
            "n_runs": n_runs,
            "n_docs": n_docs,
            "n_docs_consistent": n_docs_consistent,
            "n_docs_values": n_docs_values,
            "n_companies": len(company_set) if company_set else None,
            "n_years": len(year_set) if year_set else None,
            "change_description": _derive_change_description(
                payload,
                override=descriptions.get(run_id),
            ),
            "baseline_run_id": _norm_text(payload.get("baseline_run_id")),
            "parent_run_id": _norm_text(payload.get("parent_run_id")),
            "doc_set_signature": _doc_signature(doc_names),
            "run_status": run_status,
            "status": status,
            "status_reason": ",".join(sorted(set(status_reasons))) if status_reasons else "",
        }

        for metric in CORE_METRICS:
            stats = _metric_stats(metric_values[metric])
            if stats is None:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
            else:
                row[f"{metric}_mean"], row[f"{metric}_std"] = stats

        rows.append(row)

    rows.sort(key=lambda item: item.get("timestamp_utc") or "", reverse=True)
    return rows


def _build_linked_comparisons(framework_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run_id = {row["run_id"]: row for row in framework_rows}
    linked: list[dict[str, Any]] = []

    for candidate in framework_rows:
        baseline_run_id = candidate.get("baseline_run_id")
        parent_run_id = candidate.get("parent_run_id")
        link_target = baseline_run_id or parent_run_id
        if not link_target:
            continue

        link_type = "baseline_run_id" if baseline_run_id else "parent_run_id"
        baseline = by_run_id.get(link_target)

        row: dict[str, Any] = {
            "link_type": link_type,
            "baseline_run_id": link_target,
            "baseline_label": baseline.get("label") if baseline else None,
            "candidate_run_id": candidate.get("run_id"),
            "candidate_label": candidate.get("label"),
            "candidate_timestamp_utc": candidate.get("timestamp_utc"),
            "status": "not_comparable",
            "reason": "",
            "delta_micro_f1": None,
            "delta_micro_precision": None,
            "delta_micro_recall": None,
            "delta_hallucination_rate": None,
        }

        if baseline is None:
            row["reason"] = "baseline_not_found"
            linked.append(row)
            continue

        if baseline.get("run_status") != "completed" or candidate.get("run_status") != "completed":
            row["reason"] = "incomplete_run"
            linked.append(row)
            continue

        candidate_signature = candidate.get("doc_set_signature")
        baseline_signature = baseline.get("doc_set_signature")
        if not candidate_signature or not baseline_signature:
            row["reason"] = "missing_doc_set_signature"
            linked.append(row)
            continue
        if candidate_signature != baseline_signature:
            row["reason"] = "doc_set_mismatch"
            linked.append(row)
            continue

        missing_metric = False
        for metric in CORE_METRICS:
            if candidate.get(f"{metric}_mean") is None or baseline.get(f"{metric}_mean") is None:
                missing_metric = True
                break
        if missing_metric:
            row["reason"] = "missing_core_metrics"
            linked.append(row)
            continue

        row["status"] = "comparable"
        row["reason"] = ""
        row["delta_micro_f1"] = _safe_delta(
            candidate.get("micro_f1_mean"),
            baseline.get("micro_f1_mean"),
        )
        row["delta_micro_precision"] = _safe_delta(
            candidate.get("micro_precision_mean"),
            baseline.get("micro_precision_mean"),
        )
        row["delta_micro_recall"] = _safe_delta(
            candidate.get("micro_recall_mean"),
            baseline.get("micro_recall_mean"),
        )
        row["delta_hallucination_rate"] = _safe_delta(
            candidate.get("hallucination_rate_mean"),
            baseline.get("hallucination_rate_mean"),
        )
        linked.append(row)

    linked.sort(key=lambda item: item.get("candidate_timestamp_utc") or "", reverse=True)
    return linked


def _build_legacy_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for track in TRACKS:
        for path in sorted((results_root / track).glob("*.json")):
            try:
                payload = read_json(path)
            except Exception:
                continue
            aggregate = payload.get("aggregate", {})
            if not isinstance(aggregate, dict):
                aggregate = {}

            rows.append(
                {
                    "track": track,
                    "report_file": path.name,
                    "report_path": str(path),
                    "micro_f1": _as_float(aggregate.get("micro_f1")),
                    "micro_precision": _as_float(aggregate.get("micro_precision")),
                    "micro_recall": _as_float(aggregate.get("micro_recall")),
                    "hallucination_rate": _as_float(aggregate.get("hallucination_rate")),
                    "n_runs": "n/a",
                    "n_docs": "n/a",
                    "n_companies": "n/a",
                    "n_years": "n/a",
                    "change_description": "n/a",
                    "metadata_note": "legacy report (limited metadata)",
                }
            )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def _coverage_text(row: dict[str, Any]) -> str:
    n_docs = row.get("n_docs")
    n_docs_values = _norm_text(row.get("n_docs_values"))
    n_companies = row.get("n_companies")
    n_years = row.get("n_years")

    docs_text = "n/a"
    if isinstance(n_docs, int):
        docs_text = str(n_docs)
    elif n_docs_values:
        docs_text = n_docs_values

    if isinstance(n_companies, int) and isinstance(n_years, int):
        return f"{n_companies}x{n_years} ({docs_text} docs)"
    if docs_text != "n/a":
        return f"n/a ({docs_text} docs)"
    return "n/a"


def _legacy_track_stats(legacy_rows: list[dict[str, Any]], track: str) -> dict[str, tuple[float, float, float, float]]:
    rows = [row for row in legacy_rows if row.get("track") == track]
    stats: dict[str, tuple[float, float, float, float]] = {}
    for metric in CORE_METRICS:
        values = [
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), (int, float))
        ]
        if values:
            stats[metric] = (mean(values), pstdev(values), min(values), max(values))
    return stats


def _build_markdown_report(
    *,
    framework_rows: list[dict[str, Any]],
    linked_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    results_root: Path,
    experiments_root: Path,
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Existing Results Audit",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- experiments_root: `{experiments_root}`",
        f"- results_root: `{results_root}`",
        "",
        "## Experiment Runs (Framework)",
        "",
    ]

    if not framework_rows:
        lines.append("No framework runs found.")
    else:
        lines.extend(
            [
                "| Date | Label | Run ID | Version | Coverage | Repeats | Run Status | Change | micro_f1 | precision | recall | hallucination |",
                "|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in framework_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("timestamp_utc") or "n/a",
                        row.get("label") or "n/a",
                        row.get("run_id") or "n/a",
                        row.get("pipeline_version") or "n/a",
                        _coverage_text(row),
                        str(row.get("n_runs") if isinstance(row.get("n_runs"), int) else "n/a"),
                        row.get("run_status") or row.get("status") or "n/a",
                        row.get("change_description") or "n/a",
                        _format_metric(row.get("micro_f1_mean")),
                        _format_metric(row.get("micro_precision_mean")),
                        _format_metric(row.get("micro_recall_mean")),
                        _format_metric(row.get("hallucination_rate_mean")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Explicit Linked Comparisons", ""])
    if not linked_rows:
        lines.append("No explicit baseline/parent links found.")
    else:
        lines.extend(
            [
                "| Candidate | Baseline | Status | dF1 | dPrecision | dRecall | dHallucination | Reason |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in linked_rows:
            candidate = f"{row.get('candidate_label') or 'n/a'} (`{row.get('candidate_run_id') or 'n/a'}`)"
            baseline = f"{row.get('baseline_label') or 'n/a'} (`{row.get('baseline_run_id') or 'n/a'}`)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        candidate,
                        baseline,
                        row.get("status") or "n/a",
                        _format_metric(row.get("delta_micro_f1")),
                        _format_metric(row.get("delta_micro_precision")),
                        _format_metric(row.get("delta_micro_recall")),
                        _format_metric(row.get("delta_hallucination_rate")),
                        row.get("reason") or "",
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Legacy Historical Results (Separate Semantics)", ""])
    if not legacy_rows:
        lines.append("No legacy result files found.")
    else:
        lines.extend(
            [
                "| Track | Report | micro_f1 | precision | recall | hallucination | Metadata |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in legacy_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("track") or "n/a",
                        row.get("report_file") or "n/a",
                        _format_metric(row.get("micro_f1")),
                        _format_metric(row.get("micro_precision")),
                        _format_metric(row.get("micro_recall")),
                        _format_metric(row.get("hallucination_rate")),
                        row.get("metadata_note") or "n/a",
                    ]
                )
                + " |"
            )

        lines.extend(["", "### Legacy Track Stats"])
        for track in TRACKS:
            stats = _legacy_track_stats(legacy_rows, track)
            lines.append("")
            lines.append(f"- {track}: reports={len([row for row in legacy_rows if row.get('track') == track])}")
            for metric in CORE_METRICS:
                item = stats.get(metric)
                if item is None:
                    continue
                avg, std, lo, hi = item
                lines.append(
                    f"  - {metric}: mean={avg:.4f}, std={std:.4f}, min={lo:.4f}, max={hi:.4f}"
                )

        lines.extend(
            [
                "",
                "## Caveats",
                "",
                "- Framework runs are sourced from `artifacts/experiments/experiment_log.jsonl` with manifest fallback.",
                "- Linked comparisons are only created from explicit lineage fields (`baseline_run_id` or `parent_run_id`).",
                "- No baseline auto-picking heuristics are used.",
                "- Linked deltas are only marked comparable when both runs have metrics and matching document-set signatures.",
                "- Runs with `run_status != completed` are not treated as comparable.",
                "- Legacy results from `data/results` are shown in a separate section due to limited metadata.",
            ]
        )
    return "\n".join(lines) + "\n"


def compare_reports(
    baseline_report_path: Path,
    candidate_report_path: Path,
    out_dir: Path,
) -> dict[str, Path]:
    baseline = read_json(baseline_report_path)
    candidate = read_json(candidate_report_path)
    ensure_dir(out_dir)

    per_doc_out = out_dir / "per_doc_delta.csv"
    per_field_out = out_dir / "per_field_delta.csv"
    markdown_out = out_dir / "comparison.md"

    baseline_docs = {item["doc"]: item for item in baseline.get("per_doc", [])}
    candidate_docs = {item["doc"]: item for item in candidate.get("per_doc", [])}
    docs = sorted(set(baseline_docs) | set(candidate_docs))

    with per_doc_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doc",
                "baseline_tp",
                "candidate_tp",
                "delta_tp",
                "baseline_fp",
                "candidate_fp",
                "delta_fp",
                "baseline_fn",
                "candidate_fn",
                "delta_fn",
            ],
        )
        writer.writeheader()
        for doc in docs:
            b = baseline_docs.get(doc, {})
            c = candidate_docs.get(doc, {})
            writer.writerow(
                {
                    "doc": doc,
                    "baseline_tp": b.get("tp", 0),
                    "candidate_tp": c.get("tp", 0),
                    "delta_tp": c.get("tp", 0) - b.get("tp", 0),
                    "baseline_fp": b.get("fp", 0),
                    "candidate_fp": c.get("fp", 0),
                    "delta_fp": c.get("fp", 0) - b.get("fp", 0),
                    "baseline_fn": b.get("fn", 0),
                    "candidate_fn": c.get("fn", 0),
                    "delta_fn": c.get("fn", 0) - b.get("fn", 0),
                }
            )

    baseline_fields = baseline.get("aggregate", {}).get("field_acc_macro", {})
    candidate_fields = candidate.get("aggregate", {}).get("field_acc_macro", {})
    fields = sorted(set(baseline_fields) | set(candidate_fields))

    with per_field_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["field", "baseline", "candidate", "delta"],
        )
        writer.writeheader()
        for field in fields:
            b = baseline_fields.get(field)
            c = candidate_fields.get(field)
            writer.writerow(
                {
                    "field": field,
                    "baseline": b,
                    "candidate": c,
                    "delta": _safe_delta(c, b),
                }
            )

    agg_keys = ["micro_precision", "micro_recall", "micro_f1", "hallucination_rate"]
    agg_lines = []
    for key in agg_keys:
        b = baseline.get("aggregate", {}).get(key)
        c = candidate.get("aggregate", {}).get(key)
        d = _safe_delta(c, b)
        agg_lines.append((key, b, c, d))

    doc_regressions = []
    for doc in docs:
        b = baseline_docs.get(doc, {})
        c = candidate_docs.get(doc, {})
        score = (c.get("fn", 0) - b.get("fn", 0)) + (c.get("fp", 0) - b.get("fp", 0))
        doc_regressions.append((score, doc, b, c))
    doc_regressions.sort(reverse=True)

    lines = [
        "# Comparison Summary",
        "",
        f"- baseline: `{baseline_report_path}`",
        f"- candidate: `{candidate_report_path}`",
        "",
        "## Aggregate Deltas",
    ]
    for key, b, c, d in agg_lines:
        lines.append(f"- {key}: baseline={b} candidate={c} delta={d}")

    lines.append("")
    lines.append("## Top Regression Docs")
    for score, doc, b, c in doc_regressions[:5]:
        lines.append(
            "- "
            f"{doc}: score={score} "
            f"(fn {b.get('fn', 0)}->{c.get('fn', 0)}, fp {b.get('fp', 0)}->{c.get('fp', 0)})"
        )

    markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "comparison_markdown": markdown_out,
        "per_doc_csv": per_doc_out,
        "per_field_csv": per_field_out,
    }


def audit_existing_results(
    results_root: Path,
    out_dir: Path,
    *,
    experiments_root: Path = Path("artifacts/experiments"),
    run_notes_file: Path | None = Path("artifacts/experiments/run_notes.toml"),
) -> dict[str, Any]:
    ensure_dir(out_dir)
    summary_path = out_dir / "existing_results_audit.md"
    framework_csv_path = out_dir / "experiment_run_log.csv"
    framework_jsonl_path = out_dir / "experiment_run_log.jsonl"
    linked_csv_path = out_dir / "linked_comparisons.csv"
    legacy_csv_path = out_dir / "legacy_results_log.csv"

    aliases, descriptions = _load_run_notes(run_notes_file)
    payloads = _load_experiment_payloads(experiments_root)
    framework_rows = _build_framework_rows(payloads, aliases=aliases, descriptions=descriptions)
    linked_rows = _build_linked_comparisons(framework_rows)
    legacy_rows = _build_legacy_rows(results_root)

    framework_fields = [
        "run_id",
        "label",
        "pipeline",
        "pipeline_version",
        "model_name",
        "timestamp_utc",
        "n_runs",
        "n_docs",
        "n_docs_consistent",
        "n_docs_values",
        "n_companies",
        "n_years",
        "run_status",
        "status",
        "status_reason",
        "change_description",
        "baseline_run_id",
        "parent_run_id",
        "doc_set_signature",
        "micro_f1_mean",
        "micro_f1_std",
        "micro_precision_mean",
        "micro_precision_std",
        "micro_recall_mean",
        "micro_recall_std",
        "hallucination_rate_mean",
        "hallucination_rate_std",
    ]
    linked_fields = [
        "link_type",
        "baseline_run_id",
        "baseline_label",
        "candidate_run_id",
        "candidate_label",
        "candidate_timestamp_utc",
        "status",
        "reason",
        "delta_micro_f1",
        "delta_micro_precision",
        "delta_micro_recall",
        "delta_hallucination_rate",
    ]
    legacy_fields = [
        "track",
        "report_file",
        "report_path",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "hallucination_rate",
        "n_runs",
        "n_docs",
        "n_companies",
        "n_years",
        "change_description",
        "metadata_note",
    ]

    _write_csv(framework_csv_path, framework_rows, framework_fields)
    _write_jsonl(framework_jsonl_path, framework_rows)
    _write_csv(linked_csv_path, linked_rows, linked_fields)
    _write_csv(legacy_csv_path, legacy_rows, legacy_fields)

    markdown = _build_markdown_report(
        framework_rows=framework_rows,
        linked_rows=linked_rows,
        legacy_rows=legacy_rows,
        results_root=results_root,
        experiments_root=experiments_root,
    )
    summary_path.write_text(markdown, encoding="utf-8")

    comparable_count = len([row for row in linked_rows if row.get("status") == "comparable"])
    not_comparable_count = len([row for row in linked_rows if row.get("status") != "comparable"])

    return {
        "summary_markdown": summary_path,
        "experiment_run_log_csv": framework_csv_path,
        "linked_comparisons_csv": linked_csv_path,
        "legacy_results_log_csv": legacy_csv_path,
        "experiment_run_log_jsonl": framework_jsonl_path,
        "framework_runs": len(framework_rows),
        "linked_comparisons_total": len(linked_rows),
        "linked_comparable": comparable_count,
        "linked_not_comparable": not_comparable_count,
        "legacy_reports": len(legacy_rows),
        "recent_framework_rows": framework_rows[:5],
    }
