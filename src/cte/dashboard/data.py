from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ..io import read_json
from ..live_status import LIVE_STATUS_STALLED_AFTER_SECONDS, load_live_status

ACTIVE_RUN_STATUSES = {"running", "queued"}
STALL_AFTER = timedelta(seconds=LIVE_STATUS_STALLED_AFTER_SECONDS)
DOC_COMPANY_YEAR_PATTERN = re.compile(r"^(?P<ticker>[a-z0-9]+)\.(?P<year>\d{4})\..+\.json$")
SOURCE_CITATION_PATTERN = re.compile(r"^(?P<source>.+?)(?:#p(?P<page>\d+))?$", flags=re.IGNORECASE)

_TOKEN_KEYS = (
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
)
_COST_KEYS = (
    "total_cost_usd",
    "cost_usd",
    "estimated_cost_usd",
    "input_cost_usd",
    "output_cost_usd",
)

_TOKEN_LABELS = {
    "total_tokens": "total",
    "prompt_tokens": "prompt",
    "completion_tokens": "completion",
    "input_tokens": "input",
    "output_tokens": "output",
}
_COST_LABELS = {
    "total_cost_usd": "total",
    "cost_usd": "total",
    "estimated_cost_usd": "estimated",
    "input_cost_usd": "input",
    "output_cost_usd": "output",
}


@dataclass(slots=True)
class ArtifactLink:
    label: str
    display_path: str
    link_path: str
    exists: bool


@dataclass(slots=True)
class SourcePdfDiagnosticRow:
    filename: str
    source_path: str | None
    source_link: str | None
    cited: bool
    citation_count: int
    pages: list[int]

    @property
    def cited_display(self) -> str:
        return "yes" if self.cited else "no"

    @property
    def pages_display(self) -> str:
        if not self.pages:
            return "-"
        return ", ".join(str(page) for page in self.pages)


@dataclass(slots=True)
class RetrievedChunkDiagnosticRow:
    evaluation_item_id: str
    rank: int | None
    score: float | None
    file_name: str | None
    source_relative_path: str | None
    page: int | None
    text: str
    text_length: int | None
    text_sha256: str | None

    @property
    def score_display(self) -> str:
        return _format_metric(self.score)

    @property
    def page_display(self) -> str:
        if self.page is None:
            return "-"
        return str(self.page)

    @property
    def text_preview(self) -> str:
        if not self.text:
            return "-"
        compact = " ".join(self.text.split())
        if len(compact) <= 220:
            return compact
        return f"{compact[:217].rstrip()}..."


@dataclass(slots=True)
class CompanyYearDiagnosticRow:
    company_year: str
    recall: float
    precision: float
    f1: float
    tp: int
    fp: int
    fn: int
    source_pdf_rows: list[SourcePdfDiagnosticRow] = field(default_factory=list)
    source_pdf_mapping_note: str | None = None
    source_pdf_warnings: list[str] = field(default_factory=list)
    retrieved_chunk_rows: list[RetrievedChunkDiagnosticRow] = field(default_factory=list)
    retrieved_chunk_mapping_note: str | None = None
    retrieved_chunk_warnings: list[str] = field(default_factory=list)

    @property
    def recall_display(self) -> str:
        return _format_metric(self.recall)

    @property
    def precision_display(self) -> str:
        return _format_metric(self.precision)

    @property
    def f1_display(self) -> str:
        return _format_metric(self.f1)

    @property
    def retrieved_chunk_count(self) -> int:
        return len(self.retrieved_chunk_rows)


@dataclass(slots=True)
class FieldSummaryDiagnosticRow:
    field: str
    accuracy: float | None
    company_years_with_score: int
    missing_count: int

    @property
    def accuracy_display(self) -> str:
        return _format_metric(self.accuracy)


@dataclass(slots=True)
class RunView:
    run_id: str
    run_label: str
    run_status: str
    pipeline: str | None
    pipeline_version: str | None
    model_generation: str | None
    judge_model_name: str | None
    started_at_utc: str | None
    finished_at_utc: str | None
    timestamp_utc: str | None
    f1: float | None
    recall: float | None
    precision: float | None
    f1_se: float | None
    recall_se: float | None
    precision_se: float | None
    f1_runs: int
    recall_runs: int
    precision_runs: int
    tokens_text: str | None
    cost_text: str | None
    stalled: bool
    warnings: list[str]
    is_active: bool
    baseline_run_id: str | None
    parent_run_id: str | None
    manifest_link: ArtifactLink
    report_links: list[ArtifactLink]
    diagnostics_source_report_path: str | None
    diagnostics_source_report_link: str | None
    diagnostics_source_counter: int | None
    diagnostics_warnings: list[str]
    company_year_rows: list[CompanyYearDiagnosticRow]
    field_summary_rows: list[FieldSummaryDiagnosticRow]
    live_stage: str | None
    live_run_counter_current: int | None
    live_run_counter_total: int | None
    live_extract_completed: int | None
    live_extract_total: int | None
    live_extract_current_doc_name: str | None
    live_evaluate_completed: int | None
    live_evaluate_total: int | None
    live_evaluate_current_doc_name: str | None
    live_updated_at_utc: str | None
    live_stalled_after_seconds: int | None

    @property
    def has_warning(self) -> bool:
        return bool(self.warnings)

    @property
    def f1_display(self) -> str:
        return _format_metric_with_se(self.f1, self.f1_se)

    @property
    def recall_display(self) -> str:
        return _format_metric_with_se(self.recall, self.recall_se)

    @property
    def precision_display(self) -> str:
        return _format_metric_with_se(self.precision, self.precision_se)

    @property
    def tokens_display(self) -> str:
        return self.tokens_text or "-"

    @property
    def cost_display(self) -> str:
        return self.cost_text or "-"

    @property
    def metrics_pending(self) -> bool:
        return self.is_active and self.f1 is None and self.recall is None and self.precision is None

    @property
    def metric_runs_display(self) -> str:
        if self.f1_runs == 0 and self.recall_runs == 0 and self.precision_runs == 0:
            return "-"
        if self.f1_runs == self.recall_runs == self.precision_runs:
            return str(self.f1_runs)
        return f"f1={self.f1_runs}, r={self.recall_runs}, p={self.precision_runs}"

    @property
    def diagnostics_source_counter_display(self) -> str:
        if self.diagnostics_source_counter is None:
            return "-"
        return str(self.diagnostics_source_counter)


@dataclass(slots=True)
class DashboardSnapshot:
    generated_at_utc: str
    experiments_root: str
    active_runs: list[RunView]
    completed_runs: list[RunView]
    parse_cache_active_runs: list["ParseCacheRunView"]
    runs_by_id: dict[str, RunView]
    baseline_mapping_path: str
    baseline_mapping_exists: bool
    parsed_docs_root: str

    @property
    def total_runs(self) -> int:
        return len(self.active_runs) + len(self.completed_runs)


@dataclass(slots=True)
class ParseCacheRunView:
    run_id: str
    status: str
    mode: str | None
    config_path: str | None
    started_at_utc: str | None
    updated_at_utc: str | None
    finished_at_utc: str | None
    stalled: bool
    processed: int | None
    total: int | None
    current_source_relative_path: str | None
    hits: int
    planned_new: int
    parsed: int
    failed: int


def load_dashboard_snapshot(
    experiments_root: Path,
    *,
    parsed_docs_root: Path | None = None,
    now_utc: datetime | None = None,
) -> DashboardSnapshot:
    current_time = now_utc or datetime.now(UTC)
    manifests = sorted(Path(experiments_root).glob("*/manifest.json"))
    rows = [_build_run_view(manifest_path, current_time) for manifest_path in manifests]

    active_runs = [row for row in rows if row.is_active]
    completed_runs = [row for row in rows if not row.is_active]

    active_runs.sort(key=_sort_timestamp_key, reverse=True)
    completed_runs.sort(key=_completed_sort_key)
    parse_cache_root = (
        Path(parsed_docs_root) if parsed_docs_root is not None else Path("artifacts/parsed_docs")
    )
    parse_cache_active_runs = _load_parse_cache_active_runs(parse_cache_root, now_utc=current_time)

    runs_by_id: dict[str, RunView] = {}
    for row in rows:
        runs_by_id[row.run_id] = row

    baseline_mapping = Path("docs/BASELINES.md")
    return DashboardSnapshot(
        generated_at_utc=current_time.isoformat(),
        experiments_root=str(Path(experiments_root)),
        active_runs=active_runs,
        completed_runs=completed_runs,
        parse_cache_active_runs=parse_cache_active_runs,
        runs_by_id=runs_by_id,
        baseline_mapping_path=str(baseline_mapping),
        baseline_mapping_exists=baseline_mapping.exists(),
        parsed_docs_root=str(parse_cache_root),
    )


def _build_run_view(manifest_path: Path, now_utc: datetime) -> RunView:
    warnings: set[str] = set()
    manifest_link = ArtifactLink(
        label="manifest",
        display_path=str(manifest_path),
        link_path=str(manifest_path.resolve()),
        exists=manifest_path.exists(),
    )

    try:
        payload = read_json(manifest_path)
    except Exception:
        return RunView(
            run_id=manifest_path.parent.name,
            run_label=manifest_path.parent.name,
            run_status="invalid_manifest",
            pipeline=None,
            pipeline_version=None,
            model_generation=None,
            judge_model_name=None,
            started_at_utc=None,
            finished_at_utc=None,
            timestamp_utc=None,
            f1=None,
            recall=None,
            precision=None,
            f1_se=None,
            recall_se=None,
            precision_se=None,
            f1_runs=0,
            recall_runs=0,
            precision_runs=0,
            tokens_text=None,
            cost_text=None,
            stalled=False,
            warnings=["invalid manifest JSON"],
            is_active=False,
            baseline_run_id=None,
            parent_run_id=None,
            manifest_link=manifest_link,
            report_links=[],
            diagnostics_source_report_path=None,
            diagnostics_source_report_link=None,
            diagnostics_source_counter=None,
            diagnostics_warnings=["no diagnostics available: invalid manifest"],
            company_year_rows=[],
            field_summary_rows=[],
            live_stage=None,
            live_run_counter_current=None,
            live_run_counter_total=None,
            live_extract_completed=None,
            live_extract_total=None,
            live_extract_current_doc_name=None,
            live_evaluate_completed=None,
            live_evaluate_total=None,
            live_evaluate_current_doc_name=None,
            live_updated_at_utc=None,
            live_stalled_after_seconds=None,
        )

    run_id = _norm_text(payload.get("run_id")) or manifest_path.parent.name
    run_status = (_norm_text(payload.get("run_status")) or "").strip().lower()
    if not run_status:
        run_status = "completed" if _has_non_empty_runs(payload) else "unknown"
        warnings.add("missing run_status")

    experiment_dir = _resolve_experiment_dir(payload, manifest_path)
    live_status_path = experiment_dir / "live_status.json"
    live_payload = _load_experiment_live_status(live_status_path)
    live_stage = _norm_text(live_payload.get("stage")) if live_payload is not None else None
    live_run_counter_current = (
        _as_optional_int(live_payload.get("run_counter_current")) if live_payload else None
    )
    live_run_counter_total = (
        _as_optional_int(live_payload.get("run_count_total")) if live_payload is not None else None
    )
    live_extract_payload = (
        live_payload.get("extract_progress")
        if isinstance(live_payload, dict)
        else None
    )
    live_evaluate_payload = (
        live_payload.get("evaluate_progress")
        if isinstance(live_payload, dict)
        else None
    )
    if not isinstance(live_extract_payload, dict):
        live_extract_payload = {}
    if not isinstance(live_evaluate_payload, dict):
        live_evaluate_payload = {}
    live_extract_completed = _as_optional_int(live_extract_payload.get("completed"))
    live_extract_total = _as_optional_int(live_extract_payload.get("total"))
    live_extract_current_doc_name = _norm_text(live_extract_payload.get("current_doc_name"))
    live_evaluate_completed = _as_optional_int(live_evaluate_payload.get("completed"))
    live_evaluate_total = _as_optional_int(live_evaluate_payload.get("total"))
    live_evaluate_current_doc_name = _norm_text(live_evaluate_payload.get("current_doc_name"))
    live_updated_at_utc = _norm_text(live_payload.get("updated_at_utc")) if live_payload else None
    live_stalled_after_seconds = (
        _as_optional_int(live_payload.get("stalled_after_seconds")) if live_payload is not None else None
    )

    is_active = run_status in ACTIVE_RUN_STATUSES
    stalled = (
        _is_stalled(
            manifest_path=manifest_path,
            now_utc=now_utc,
            live_updated_at_utc=live_updated_at_utc,
            stalled_after_seconds=live_stalled_after_seconds,
        )
        if is_active
        else False
    )
    if stalled:
        warnings.add("stalled >20m without progress update")

    metric_summary = _extract_metrics_and_artifacts(
        payload,
        experiment_dir=experiment_dir,
        warning_set=warnings,
    )

    if is_active:
        warnings.discard("manifest has no run entries")
        warnings.discard("missing metrics")

    diagnostics = _build_run_diagnostics(payload, experiment_dir=experiment_dir)

    return RunView(
        run_id=run_id,
        run_label=_norm_text(payload.get("run_label")) or run_id,
        run_status=run_status,
        pipeline=_norm_text(payload.get("pipeline")),
        pipeline_version=_norm_text(payload.get("pipeline_version")),
        model_generation=_norm_text(payload.get("model_name")),
        judge_model_name=_norm_text(payload.get("judge_model_name")),
        started_at_utc=_norm_text(payload.get("started_at_utc")),
        finished_at_utc=_norm_text(payload.get("finished_at_utc")),
        timestamp_utc=_norm_text(payload.get("timestamp_utc")),
        f1=metric_summary["f1"],
        recall=metric_summary["recall"],
        precision=metric_summary["precision"],
        f1_se=metric_summary["f1_se"],
        recall_se=metric_summary["recall_se"],
        precision_se=metric_summary["precision_se"],
        f1_runs=metric_summary["f1_runs"],
        recall_runs=metric_summary["recall_runs"],
        precision_runs=metric_summary["precision_runs"],
        tokens_text=metric_summary["tokens_text"],
        cost_text=metric_summary["cost_text"],
        stalled=stalled,
        warnings=sorted(warnings),
        is_active=is_active,
        baseline_run_id=_norm_text(payload.get("baseline_run_id")),
        parent_run_id=_norm_text(payload.get("parent_run_id")),
        manifest_link=manifest_link,
        report_links=metric_summary["report_links"],
        diagnostics_source_report_path=diagnostics["source_report_path"],
        diagnostics_source_report_link=diagnostics["source_report_link"],
        diagnostics_source_counter=diagnostics["source_counter"],
        diagnostics_warnings=diagnostics["warnings"],
        company_year_rows=diagnostics["company_year_rows"],
        field_summary_rows=diagnostics["field_summary_rows"],
        live_stage=live_stage,
        live_run_counter_current=live_run_counter_current,
        live_run_counter_total=live_run_counter_total,
        live_extract_completed=live_extract_completed,
        live_extract_total=live_extract_total,
        live_extract_current_doc_name=live_extract_current_doc_name,
        live_evaluate_completed=live_evaluate_completed,
        live_evaluate_total=live_evaluate_total,
        live_evaluate_current_doc_name=live_evaluate_current_doc_name,
        live_updated_at_utc=live_updated_at_utc,
        live_stalled_after_seconds=live_stalled_after_seconds,
    )


def _extract_metrics_and_artifacts(
    payload: dict[str, Any],
    *,
    experiment_dir: Path,
    warning_set: set[str],
) -> dict[str, Any]:
    f1_values: list[float] = []
    recall_values: list[float] = []
    precision_values: list[float] = []
    report_links: list[ArtifactLink] = []
    tokens_text = _extract_token_text([payload])
    cost_text = _extract_cost_text([payload])

    run_records = payload.get("runs")
    if not isinstance(run_records, list):
        warning_set.add("invalid runs payload")
        run_records = []
    if not run_records:
        warning_set.add("manifest has no run entries")

    for index, run_record in enumerate(run_records, start=1):
        if not isinstance(run_record, dict):
            warning_set.add("invalid run entry")
            continue

        report_payload: dict[str, Any] | None = None
        report_aggregate: dict[str, Any] = {}

        report_path_raw = _norm_text(run_record.get("report_path"))
        if report_path_raw:
            link = _build_artifact_link(f"report #{index}", report_path_raw, experiment_dir)
            report_links.append(link)
            if link.exists:
                try:
                    report_payload = read_json(Path(link.link_path))
                except Exception:
                    warning_set.add("unreadable report file")
            else:
                warning_set.add("missing report file")
        else:
            warning_set.add("missing report path")

        if isinstance(report_payload, dict):
            aggregate = report_payload.get("aggregate")
            if isinstance(aggregate, dict):
                report_aggregate = aggregate

        run_aggregate_raw = run_record.get("aggregate")
        run_aggregate = run_aggregate_raw if isinstance(run_aggregate_raw, dict) else {}

        f1 = _first_float(run_aggregate.get("micro_f1"), report_aggregate.get("micro_f1"))
        recall = _first_float(run_aggregate.get("micro_recall"), report_aggregate.get("micro_recall"))
        precision = _first_float(
            run_aggregate.get("micro_precision"),
            report_aggregate.get("micro_precision"),
        )
        if f1 is not None:
            f1_values.append(f1)
        if recall is not None:
            recall_values.append(recall)
        if precision is not None:
            precision_values.append(precision)

        if tokens_text is None:
            tokens_text = _extract_token_text([run_record, run_aggregate, report_aggregate, report_payload])
        if cost_text is None:
            cost_text = _extract_cost_text([run_record, run_aggregate, report_aggregate, report_payload])

    f1_mean = _mean_or_none(f1_values)
    recall_mean = _mean_or_none(recall_values)
    precision_mean = _mean_or_none(precision_values)
    f1_se = _standard_error(f1_values)
    recall_se = _standard_error(recall_values)
    precision_se = _standard_error(precision_values)
    if f1_mean is None or recall_mean is None or precision_mean is None:
        warning_set.add("missing metrics")

    return {
        "f1": f1_mean,
        "recall": recall_mean,
        "precision": precision_mean,
        "f1_se": f1_se,
        "recall_se": recall_se,
        "precision_se": precision_se,
        "f1_runs": len(f1_values),
        "recall_runs": len(recall_values),
        "precision_runs": len(precision_values),
        "tokens_text": tokens_text,
        "cost_text": cost_text,
        "report_links": report_links,
    }


def _has_non_empty_runs(payload: dict[str, Any]) -> bool:
    runs = payload.get("runs")
    return isinstance(runs, list) and len(runs) > 0


def _build_run_diagnostics(
    payload: dict[str, Any],
    *,
    experiment_dir: Path,
) -> dict[str, Any]:
    run_records_raw = payload.get("runs")
    if not isinstance(run_records_raw, list) or not run_records_raw:
        return {
            "source_report_path": None,
            "source_report_link": None,
            "source_counter": None,
            "warnings": ["no run entries available for diagnostics"],
            "company_year_rows": [],
            "field_summary_rows": [],
        }

    candidates = _report_candidates(run_records_raw, experiment_dir=experiment_dir)
    if not candidates:
        return {
            "source_report_path": None,
            "source_report_link": None,
            "source_counter": None,
            "warnings": ["no report paths available for diagnostics"],
            "company_year_rows": [],
            "field_summary_rows": [],
        }

    primary_candidate = candidates[0]
    chosen_payload: dict[str, Any] | None = None
    chosen_candidate: dict[str, Any] | None = None

    for candidate in candidates:
        link = candidate["link"]
        if not link.exists:
            continue
        try:
            maybe_payload = read_json(Path(link.link_path))
        except Exception:
            continue
        if isinstance(maybe_payload, dict):
            chosen_payload = maybe_payload
            chosen_candidate = candidate
            break

    if chosen_payload is None or chosen_candidate is None:
        return {
            "source_report_path": None,
            "source_report_link": None,
            "source_counter": None,
            "warnings": ["no readable report available for diagnostics"],
            "company_year_rows": [],
            "field_summary_rows": [],
        }

    warnings: list[str] = []
    if chosen_candidate is not primary_candidate:
        warnings.append(
            "using fallback report because latest counter report was missing or unreadable"
        )

    repo_root = Path.cwd().resolve()
    company_year_rows, field_status_by_company_year, mapping_warnings = _build_company_year_rows(
        chosen_payload,
        manifest_payload=payload,
        chosen_counter=chosen_candidate["counter"],
        chosen_run_record=chosen_candidate["run_record"],
        experiment_dir=experiment_dir,
        repo_root=repo_root,
    )
    field_summary_rows = _build_field_summary_rows(chosen_payload, field_status_by_company_year)
    warnings.extend(mapping_warnings)
    if not company_year_rows:
        warnings.append("diagnostics source report has no per_doc rows")

    source_link: ArtifactLink = chosen_candidate["link"]
    return {
        "source_report_path": source_link.display_path,
        "source_report_link": source_link.link_path,
        "source_counter": chosen_candidate["counter"],
        "warnings": warnings,
        "company_year_rows": company_year_rows,
        "field_summary_rows": field_summary_rows,
    }


def _report_candidates(run_records: list[Any], *, experiment_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for index, record in enumerate(run_records):
        if not isinstance(record, dict):
            continue
        report_path = _norm_text(record.get("report_path"))
        if report_path is None:
            continue
        counter_raw = record.get("counter")
        counter = counter_raw if isinstance(counter_raw, int) else None
        candidates.append(
            {
                "counter": counter,
                "index": index,
                "run_record": record,
                "link": _build_artifact_link(
                    f"report #{counter if counter is not None else (index + 1)}",
                    report_path,
                    experiment_dir,
                ),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            candidate["counter"] if candidate["counter"] is not None else -1,
            candidate["index"],
        ),
        reverse=True,
    )
    return candidates

def _build_company_year_rows(
    report_payload: dict[str, Any],
    *,
    manifest_payload: dict[str, Any],
    chosen_counter: int | None,
    chosen_run_record: dict[str, Any] | None,
    experiment_dir: Path,
    repo_root: Path,
) -> tuple[list[CompanyYearDiagnosticRow], dict[str, dict[str, int]], list[str]]:
    per_doc = report_payload.get("per_doc")
    if not isinstance(per_doc, list):
        return [], {}, []

    totals: dict[str, dict[str, int]] = {}
    evaluation_items_by_company_year: dict[str, dict[str, dict[str, int]]] = {}
    field_status_by_company_year: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    mapping_cache: dict[str, Any] = {
        "index_manifest": {},
        "prediction_payload": {},
        "retrieved_chunk_trace": {},
    }
    retrieval_trace = _load_retrieved_chunk_trace_for_run(
        manifest_payload=manifest_payload,
        chosen_counter=chosen_counter,
        chosen_run_record=chosen_run_record,
        experiment_dir=experiment_dir,
        cache=mapping_cache,
    )
    for retrieval_warning in retrieval_trace["warnings"]:
        if retrieval_warning not in warnings:
            warnings.append(retrieval_warning)

    for item in per_doc:
        if not isinstance(item, dict):
            continue
        evaluation_item_id = _norm_text(item.get("doc"))
        if evaluation_item_id is None:
            continue
        company_year = _company_year_from_doc(evaluation_item_id)

        tp = _as_int(item.get("tp"))
        fp = _as_int(item.get("fp"))
        fn = _as_int(item.get("fn"))
        stats = totals.setdefault(company_year, {"tp": 0, "fp": 0, "fn": 0})
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn

        evaluation_items_for_company_year = evaluation_items_by_company_year.setdefault(company_year, {})
        item_stats = evaluation_items_for_company_year.setdefault(
            evaluation_item_id,
            {"tp": 0, "fp": 0, "fn": 0},
        )
        item_stats["tp"] += tp
        item_stats["fp"] += fp
        item_stats["fn"] += fn

        field_accuracy = item.get("field_accuracy")
        if isinstance(field_accuracy, dict):
            company_field_status = field_status_by_company_year.setdefault(company_year, {})
            for field, value in field_accuracy.items():
                existing_state = company_field_status.get(field, 0)
                candidate_state = 2 if _as_float(value) is not None else 1
                company_field_status[field] = max(existing_state, candidate_state)

    rows: list[CompanyYearDiagnosticRow] = []
    for company_year, stats in totals.items():
        precision = _safe_precision(stats["tp"], stats["fp"])
        recall = _safe_recall(stats["tp"], stats["fn"])
        f1 = _safe_f1(precision, recall)

        aggregated_source_pdf_rows: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        company_source_pdf_warnings: list[str] = []
        company_source_pdf_warning_seen: set[str] = set()
        company_retrieved_chunk_warnings: list[str] = []
        company_retrieved_chunk_warning_seen: set[str] = set()
        mapping_notes: list[str] = []
        retrieved_chunk_rows: list[RetrievedChunkDiagnosticRow] = []
        retrieved_chunk_mapping_note = retrieval_trace["mapping_note"]
        evaluation_items_for_company_year = evaluation_items_by_company_year.get(company_year, {})
        for evaluation_item_id in evaluation_items_for_company_year:
            mapping = _build_source_pdf_mapping_for_evaluation_item(
                manifest_payload=manifest_payload,
                chosen_counter=chosen_counter,
                chosen_run_record=chosen_run_record,
                evaluation_item_id=evaluation_item_id,
                experiment_dir=experiment_dir,
                repo_root=repo_root,
                cache=mapping_cache,
            )
            for mapping_warning in mapping["warnings"]:
                warnings.append(f"{evaluation_item_id}: {mapping_warning}")
                company_warning = f"{evaluation_item_id}: {mapping_warning}"
                if company_warning not in company_source_pdf_warning_seen:
                    company_source_pdf_warning_seen.add(company_warning)
                    company_source_pdf_warnings.append(company_warning)

            mapping_note = _norm_text(mapping.get("mapping_note"))
            if mapping_note is not None:
                mapping_notes.append(mapping_note)

            for source_pdf_row in mapping["source_pdf_rows"]:
                key = (
                    source_pdf_row.filename.lower(),
                    source_pdf_row.source_path,
                    source_pdf_row.source_link,
                )
                aggregate = aggregated_source_pdf_rows.setdefault(
                    key,
                    {
                        "filename": source_pdf_row.filename,
                        "source_path": source_pdf_row.source_path,
                        "source_link": source_pdf_row.source_link,
                        "cited": False,
                        "citation_count": 0,
                        "pages": set(),
                    },
                )
                aggregate["cited"] = aggregate["cited"] or source_pdf_row.cited
                aggregate["citation_count"] += source_pdf_row.citation_count
                aggregate["pages"].update(source_pdf_row.pages)

            if retrieval_trace["mapping_note"] is not None:
                continue
            chunk_entries = retrieval_trace["documents"].get(evaluation_item_id)
            if chunk_entries is None:
                chunk_warning = (
                    f"{evaluation_item_id}: retrieved chunk trace unavailable for evaluation item"
                )
                if chunk_warning not in company_retrieved_chunk_warning_seen:
                    company_retrieved_chunk_warning_seen.add(chunk_warning)
                    company_retrieved_chunk_warnings.append(chunk_warning)
                if chunk_warning not in warnings:
                    warnings.append(chunk_warning)
                continue

            parsed_chunk_rows, chunk_parse_warnings = _parse_retrieved_chunk_rows(
                evaluation_item_id=evaluation_item_id,
                chunk_entries=chunk_entries,
            )
            retrieved_chunk_rows.extend(parsed_chunk_rows)
            for chunk_warning in chunk_parse_warnings:
                warning_text = f"{evaluation_item_id}: {chunk_warning}"
                if warning_text not in company_retrieved_chunk_warning_seen:
                    company_retrieved_chunk_warning_seen.add(warning_text)
                    company_retrieved_chunk_warnings.append(warning_text)
                if warning_text not in warnings:
                    warnings.append(warning_text)

        source_pdf_rows: list[SourcePdfDiagnosticRow] = []
        for aggregate in aggregated_source_pdf_rows.values():
            source_pdf_rows.append(
                SourcePdfDiagnosticRow(
                    filename=aggregate["filename"],
                    source_path=aggregate["source_path"],
                    source_link=aggregate["source_link"],
                    cited=aggregate["cited"],
                    citation_count=aggregate["citation_count"],
                    pages=sorted(aggregate["pages"]),
                )
            )
        source_pdf_rows.sort(key=lambda row: row.filename.lower())

        source_pdf_mapping_note: str | None = None
        if not source_pdf_rows and mapping_notes:
            unique_mapping_notes = list(dict.fromkeys(mapping_notes))
            source_pdf_mapping_note = (
                unique_mapping_notes[0]
                if len(unique_mapping_notes) == 1
                else "source PDF mapping unavailable for one or more entries"
            )

        retrieved_chunk_rows.sort(
            key=lambda row: (
                row.evaluation_item_id.lower(),
                row.rank if row.rank is not None else 10**9,
                row.file_name.lower() if row.file_name is not None else "",
            )
        )
        if (
            retrieval_trace["mapping_note"] is None
            and not retrieved_chunk_rows
            and evaluation_items_for_company_year
        ):
            retrieved_chunk_mapping_note = "no retrieved chunks found for this company-year"

        rows.append(
            CompanyYearDiagnosticRow(
                company_year=company_year,
                recall=recall,
                precision=precision,
                f1=f1,
                tp=stats["tp"],
                fp=stats["fp"],
                fn=stats["fn"],
                source_pdf_rows=source_pdf_rows,
                source_pdf_mapping_note=source_pdf_mapping_note,
                source_pdf_warnings=company_source_pdf_warnings,
                retrieved_chunk_rows=retrieved_chunk_rows,
                retrieved_chunk_mapping_note=retrieved_chunk_mapping_note,
                retrieved_chunk_warnings=company_retrieved_chunk_warnings,
            )
        )

    rows.sort(key=lambda row: (row.recall, -row.fn, row.company_year.lower()))
    return rows, field_status_by_company_year, warnings


def _build_field_summary_rows(
    report_payload: dict[str, Any],
    field_status_by_company_year: dict[str, dict[str, int]],
) -> list[FieldSummaryDiagnosticRow]:
    aggregate = report_payload.get("aggregate")
    field_acc_macro: dict[str, Any] = {}
    if isinstance(aggregate, dict):
        maybe_field_acc = aggregate.get("field_acc_macro")
        if isinstance(maybe_field_acc, dict):
            field_acc_macro = maybe_field_acc

    known_fields = set(field_acc_macro.keys())
    for per_company_year in field_status_by_company_year.values():
        known_fields.update(per_company_year.keys())

    rows: list[FieldSummaryDiagnosticRow] = []
    for field in known_fields:
        with_score = 0
        missing = 0
        for per_company_year in field_status_by_company_year.values():
            state = per_company_year.get(field, 0)
            if state == 2:
                with_score += 1
            elif state == 1:
                missing += 1

        rows.append(
            FieldSummaryDiagnosticRow(
                field=field,
                accuracy=_as_float(field_acc_macro.get(field)),
                company_years_with_score=with_score,
                missing_count=missing,
            )
        )

    rows.sort(
        key=lambda row: (
            row.accuracy is None,
            row.accuracy if row.accuracy is not None else 0.0,
            row.field.lower(),
        )
    )
    return rows


def _build_source_pdf_mapping_for_evaluation_item(
    *,
    manifest_payload: dict[str, Any],
    chosen_counter: int | None,
    chosen_run_record: dict[str, Any] | None,
    evaluation_item_id: str,
    experiment_dir: Path,
    repo_root: Path,
    cache: dict[str, Any],
) -> dict[str, Any]:
    pipeline = _norm_text(manifest_payload.get("pipeline"))
    if pipeline != "rag":
        return {
            "mapping_available": False,
            "mapping_note": "source PDF mapping unavailable for non-RAG runs",
            "source_pdf_rows": [],
            "warnings": [],
        }

    warnings: list[str] = []
    index_events = manifest_payload.get("index_events")
    if not isinstance(index_events, list):
        warnings.append("source PDF mapping unavailable: missing index events")
        return {
            "mapping_available": False,
            "mapping_note": "source PDF mapping unavailable",
            "source_pdf_rows": [],
            "warnings": warnings,
        }

    matching_index_dirs: list[str] = []
    for event in index_events:
        if not isinstance(event, dict):
            continue
        event_doc = _norm_text(event.get("doc"))
        event_index_dir = _norm_text(event.get("index_dir"))
        if event_doc == evaluation_item_id and event_index_dir is not None:
            matching_index_dirs.append(event_index_dir)

    if not matching_index_dirs:
        warnings.append("source PDF mapping unavailable: no index event found")
        return {
            "mapping_available": False,
            "mapping_note": "source PDF mapping unavailable",
            "source_pdf_rows": [],
            "warnings": warnings,
        }

    if len(set(matching_index_dirs)) > 1:
        warnings.append("multiple index mappings found; using most recent event")
    selected_index_dir = matching_index_dirs[-1]

    index_manifest_payload = _load_index_manifest_payload(
        index_dir_raw=selected_index_dir,
        experiment_dir=experiment_dir,
        cache=cache,
    )
    if index_manifest_payload is None:
        warnings.append("source PDF mapping unavailable: index manifest missing or unreadable")
        return {
            "mapping_available": False,
            "mapping_note": "source PDF mapping unavailable",
            "source_pdf_rows": [],
            "warnings": warnings,
        }

    source_pdf_entries, source_pdf_warnings = _build_source_pdf_entries_from_index_manifest(
        index_manifest_payload=index_manifest_payload,
        repo_root=repo_root,
    )
    warnings.extend(source_pdf_warnings)
    if not source_pdf_entries:
        warnings.append("source PDF mapping unavailable: index manifest has no source files")
        return {
            "mapping_available": False,
            "mapping_note": "source PDF mapping unavailable",
            "source_pdf_rows": [],
            "warnings": warnings,
        }

    citation_summary, citation_warnings = _extract_citation_summary(
        manifest_payload=manifest_payload,
        chosen_counter=chosen_counter,
        chosen_run_record=chosen_run_record,
        evaluation_item_id=evaluation_item_id,
        experiment_dir=experiment_dir,
        cache=cache,
    )
    warnings.extend(citation_warnings)

    matched_citation_keys: set[str] = set()
    source_pdf_rows: list[SourcePdfDiagnosticRow] = []
    for entry in source_pdf_entries:
        citation = citation_summary.get(entry["filename"].lower())
        if citation is not None:
            matched_citation_keys.add(entry["filename"].lower())
        pages = citation["pages"] if citation is not None else []
        source_pdf_rows.append(
            SourcePdfDiagnosticRow(
                filename=entry["filename"],
                source_path=entry["source_path"],
                source_link=entry["source_link"],
                cited=citation is not None,
                citation_count=citation["count"] if citation is not None else 0,
                pages=pages,
            )
        )

    unmatched_citations = sorted(
        citation["filename"]
        for key, citation in citation_summary.items()
        if key not in matched_citation_keys
    )
    if unmatched_citations:
        warnings.append(
            "citation sources not found in indexed PDFs: " + ", ".join(unmatched_citations)
        )

    source_pdf_rows.sort(key=lambda row: row.filename.lower())
    return {
        "mapping_available": True,
        "mapping_note": None,
        "source_pdf_rows": source_pdf_rows,
        "warnings": warnings,
    }


def _load_index_manifest_payload(
    *,
    index_dir_raw: str,
    experiment_dir: Path,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    index_manifest_cache = cache.setdefault("index_manifest", {})
    if index_dir_raw in index_manifest_cache:
        cached = index_manifest_cache[index_dir_raw]
        return cached if isinstance(cached, dict) else None

    index_dir = _resolve_existing_path(index_dir_raw, experiment_dir=experiment_dir)
    if index_dir is None:
        index_manifest_cache[index_dir_raw] = None
        return None

    index_manifest_path = index_dir / "index_manifest.json"
    if not index_manifest_path.exists():
        index_manifest_cache[index_dir_raw] = None
        return None

    try:
        payload = read_json(index_manifest_path)
    except Exception:
        index_manifest_cache[index_dir_raw] = None
        return None
    if not isinstance(payload, dict):
        index_manifest_cache[index_dir_raw] = None
        return None

    index_manifest_cache[index_dir_raw] = payload
    return payload


def _build_source_pdf_entries_from_index_manifest(
    *,
    index_manifest_payload: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, str | None]], list[str]]:
    warnings: list[str] = []
    source_manifest = index_manifest_payload.get("source_manifest")
    if not isinstance(source_manifest, list):
        return [], []

    source_root_raw = _norm_text(index_manifest_payload.get("source_root"))
    source_root: Path | None = None
    if source_root_raw is not None:
        source_root = Path(source_root_raw)
        if not source_root.is_absolute():
            source_root = (Path.cwd() / source_root).resolve(strict=False)

    rows: list[dict[str, str | None]] = []
    for entry in source_manifest:
        if not isinstance(entry, dict):
            continue
        relative_path = _norm_text(entry.get("relative_path"))
        if relative_path is None:
            continue
        filename = Path(relative_path).name or relative_path

        source_path: str | None = None
        source_link: str | None = None
        if source_root is not None:
            absolute_path = (source_root / relative_path).resolve(strict=False)
            source_path = str(absolute_path)
            if _is_path_within_root(absolute_path, repo_root):
                source_link = str(absolute_path)
        else:
            source_path = relative_path

        rows.append(
            {
                "filename": filename,
                "source_path": source_path,
                "source_link": source_link,
            }
        )

    if not rows:
        warnings.append("index manifest has no parseable source files")
    return rows, warnings


def _extract_citation_summary(
    *,
    manifest_payload: dict[str, Any],
    chosen_counter: int | None,
    chosen_run_record: dict[str, Any] | None,
    evaluation_item_id: str,
    experiment_dir: Path,
    cache: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    prediction_payload = _load_prediction_payload_for_evaluation_item(
        manifest_payload=manifest_payload,
        chosen_counter=chosen_counter,
        chosen_run_record=chosen_run_record,
        evaluation_item_id=evaluation_item_id,
        experiment_dir=experiment_dir,
        cache=cache,
    )
    if prediction_payload is None:
        warnings.append("prediction file unavailable; citation summary unavailable")
        return {}, warnings

    targets = prediction_payload.get("targets")
    if not isinstance(targets, list):
        return {}, warnings

    summary: dict[str, dict[str, Any]] = {}
    for target in targets:
        if not isinstance(target, dict):
            continue
        sources = target.get("sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            source_text = _norm_text(source)
            if source_text is None:
                continue
            parsed = _parse_source_citation(source_text)
            if parsed is None:
                continue
            filename, page = parsed
            key = filename.lower()
            stats = summary.setdefault(
                key,
                {"filename": filename, "count": 0, "pages": set()},
            )
            stats["count"] += 1
            if page is not None:
                stats["pages"].add(page)

    for stats in summary.values():
        stats["pages"] = sorted(stats["pages"])
    return summary, warnings


def _load_prediction_payload_for_evaluation_item(
    *,
    manifest_payload: dict[str, Any],
    chosen_counter: int | None,
    chosen_run_record: dict[str, Any] | None,
    evaluation_item_id: str,
    experiment_dir: Path,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    run_record = chosen_run_record
    if not isinstance(run_record, dict):
        run_record = _resolve_run_record_for_counter(
            manifest_payload=manifest_payload,
            counter=chosen_counter,
        )
    if not isinstance(run_record, dict):
        return None

    prediction_dir_raw = _norm_text(run_record.get("prediction_dir"))
    if prediction_dir_raw is None:
        return None
    prediction_dir = _resolve_existing_path(prediction_dir_raw, experiment_dir=experiment_dir)
    if prediction_dir is None:
        return None

    prediction_path = (prediction_dir / evaluation_item_id).resolve(strict=False)
    prediction_payload_cache = cache.setdefault("prediction_payload", {})
    cache_key = str(prediction_path)
    if cache_key in prediction_payload_cache:
        cached = prediction_payload_cache[cache_key]
        return cached if isinstance(cached, dict) else None

    if not prediction_path.exists():
        prediction_payload_cache[cache_key] = None
        return None
    try:
        payload = read_json(prediction_path)
    except Exception:
        prediction_payload_cache[cache_key] = None
        return None
    if not isinstance(payload, dict):
        prediction_payload_cache[cache_key] = None
        return None

    prediction_payload_cache[cache_key] = payload
    return payload


def _load_retrieved_chunk_trace_for_run(
    *,
    manifest_payload: dict[str, Any],
    chosen_counter: int | None,
    chosen_run_record: dict[str, Any] | None,
    experiment_dir: Path,
    cache: dict[str, Any],
) -> dict[str, Any]:
    pipeline = _norm_text(manifest_payload.get("pipeline"))
    if pipeline != "rag":
        return {
            "mapping_note": "retrieved chunk mapping unavailable for non-RAG runs",
            "documents": {},
            "warnings": [],
        }

    run_record = chosen_run_record
    if not isinstance(run_record, dict):
        run_record = _resolve_run_record_for_counter(
            manifest_payload=manifest_payload,
            counter=chosen_counter,
        )
    if not isinstance(run_record, dict):
        warning = "retrieved chunk mapping unavailable: run record missing"
        return {
            "mapping_note": "retrieved chunk mapping unavailable",
            "documents": {},
            "warnings": [warning],
        }

    prediction_dir_raw = _norm_text(run_record.get("prediction_dir"))
    if prediction_dir_raw is None:
        warning = "retrieved chunk mapping unavailable: prediction directory missing"
        return {
            "mapping_note": "retrieved chunk mapping unavailable",
            "documents": {},
            "warnings": [warning],
        }

    prediction_dir = _resolve_existing_path(prediction_dir_raw, experiment_dir=experiment_dir)
    if prediction_dir is None:
        warning = "retrieved chunk mapping unavailable: prediction directory not found"
        return {
            "mapping_note": "retrieved chunk mapping unavailable",
            "documents": {},
            "warnings": [warning],
        }

    sidecar_path = (prediction_dir / "_retrieved_chunks.json").resolve(strict=False)
    cache_key = str(sidecar_path)
    trace_cache = cache.setdefault("retrieved_chunk_trace", {})

    payload: dict[str, Any] | None
    if cache_key in trace_cache:
        cached = trace_cache[cache_key]
        payload = cached if isinstance(cached, dict) else None
    else:
        if not sidecar_path.exists():
            trace_cache[cache_key] = None
            warning = "retrieved chunk mapping unavailable: _retrieved_chunks.json is missing"
            return {
                "mapping_note": "retrieved chunk mapping unavailable",
                "documents": {},
                "warnings": [warning],
            }
        try:
            maybe_payload = read_json(sidecar_path)
        except Exception:
            trace_cache[cache_key] = None
            warning = "retrieved chunk mapping unavailable: could not read _retrieved_chunks.json"
            return {
                "mapping_note": "retrieved chunk mapping unavailable",
                "documents": {},
                "warnings": [warning],
            }
        if not isinstance(maybe_payload, dict):
            trace_cache[cache_key] = None
            warning = "retrieved chunk mapping unavailable: invalid _retrieved_chunks.json payload"
            return {
                "mapping_note": "retrieved chunk mapping unavailable",
                "documents": {},
                "warnings": [warning],
            }
        trace_cache[cache_key] = maybe_payload
        payload = maybe_payload

    if payload is None:
        warning = "retrieved chunk mapping unavailable"
        return {
            "mapping_note": "retrieved chunk mapping unavailable",
            "documents": {},
            "warnings": [warning],
        }

    documents_raw = payload.get("documents")
    if not isinstance(documents_raw, list):
        warning = "retrieved chunk mapping unavailable: sidecar has no documents list"
        return {
            "mapping_note": "retrieved chunk mapping unavailable",
            "documents": {},
            "warnings": [warning],
        }

    documents: dict[str, list[Any]] = {}
    for item in documents_raw:
        if not isinstance(item, dict):
            continue
        doc_id = _norm_text(item.get("doc"))
        if doc_id is None:
            continue
        retrieved_chunks = item.get("retrieved_chunks")
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []
        documents[doc_id] = retrieved_chunks

    return {
        "mapping_note": None,
        "documents": documents,
        "warnings": [],
    }


def _parse_retrieved_chunk_rows(
    *,
    evaluation_item_id: str,
    chunk_entries: list[Any],
) -> tuple[list[RetrievedChunkDiagnosticRow], list[str]]:
    rows: list[RetrievedChunkDiagnosticRow] = []
    warnings: list[str] = []

    for chunk_entry in chunk_entries:
        if not isinstance(chunk_entry, dict):
            warnings.append("retrieved chunk entry is not an object")
            continue

        metadata_raw = chunk_entry.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

        source_relative_path = _norm_text(metadata.get("source_relative_path"))
        file_name = _norm_text(metadata.get("file_name"))
        if file_name is None and source_relative_path is not None:
            file_name = Path(source_relative_path).name or source_relative_path

        text_raw = chunk_entry.get("text")
        text = str(text_raw) if text_raw is not None else ""
        rows.append(
            RetrievedChunkDiagnosticRow(
                evaluation_item_id=evaluation_item_id,
                rank=_as_optional_int(chunk_entry.get("rank")),
                score=_as_float(chunk_entry.get("score")),
                file_name=file_name,
                source_relative_path=source_relative_path,
                page=_as_optional_int(metadata.get("page")),
                text=text,
                text_length=_as_optional_int(chunk_entry.get("text_length")),
                text_sha256=_norm_text(chunk_entry.get("text_sha256")),
            )
        )

    return rows, warnings


def _resolve_run_record_for_counter(
    *,
    manifest_payload: dict[str, Any],
    counter: int | None,
) -> dict[str, Any] | None:
    run_records = manifest_payload.get("runs")
    if not isinstance(run_records, list):
        return None

    for run_record in run_records:
        if not isinstance(run_record, dict):
            continue
        run_counter = run_record.get("counter")
        if isinstance(run_counter, int) and run_counter == counter:
            return run_record
    return None


def _parse_source_citation(source_text: str) -> tuple[str, int | None] | None:
    match = SOURCE_CITATION_PATTERN.match(source_text)
    if match is None:
        return None
    source_part = _norm_text(match.group("source"))
    if source_part is None:
        return None
    filename = Path(source_part).name or source_part
    if ".pdf" not in filename.lower():
        return None
    page_text = _norm_text(match.group("page"))
    page = int(page_text) if page_text is not None and page_text.isdigit() else None
    return filename, page


def _resolve_existing_path(raw_path: str, *, experiment_dir: Path) -> Path | None:
    for candidate in _path_candidates(raw_path, experiment_dir):
        if candidate.exists():
            return candidate
    return None


def _is_path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _resolve_experiment_dir(payload: dict[str, Any], manifest_path: Path) -> Path:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        experiment_dir = _norm_text(artifacts.get("experiment_dir"))
        if experiment_dir:
            return Path(experiment_dir)
    return manifest_path.parent


def _build_artifact_link(label: str, raw_path: str, experiment_dir: Path) -> ArtifactLink:
    candidates = _path_candidates(raw_path, experiment_dir)
    for candidate in candidates:
        if candidate.exists():
            return ArtifactLink(
                label=label,
                display_path=raw_path,
                link_path=str(candidate.resolve()),
                exists=True,
            )

    fallback = candidates[0].resolve(strict=False)
    return ArtifactLink(
        label=label,
        display_path=raw_path,
        link_path=str(fallback),
        exists=False,
    )


def _path_candidates(raw_path: str, experiment_dir: Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        return [path]

    candidates = [Path.cwd() / path]
    candidates.append(experiment_dir / path)

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _extract_token_text(sources: list[dict[str, Any] | None]) -> str | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        values = _extract_numeric_values(source, _TOKEN_KEYS, nested_keys=("usage", "token_counts", "tokens"))
        if values:
            parts = [
                f"{_TOKEN_LABELS[key]}={_format_number(values[key])}"
                for key in _TOKEN_KEYS
                if key in values
            ]
            return ", ".join(parts)
    return None


def _extract_cost_text(sources: list[dict[str, Any] | None]) -> str | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        values = _extract_numeric_values(source, _COST_KEYS, nested_keys=("cost", "pricing", "usage"))
        if values:
            total_value = values.get("total_cost_usd")
            if total_value is None:
                total_value = values.get("cost_usd")
            if total_value is not None:
                return f"${total_value:.4f}"

            parts = [
                f"{_COST_LABELS[key]}=${values[key]:.4f}"
                for key in _COST_KEYS
                if key in values
            ]
            return ", ".join(parts)
    return None


def _extract_numeric_values(
    source: dict[str, Any],
    keys: tuple[str, ...],
    *,
    nested_keys: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}

    for key in keys:
        value = _as_float(source.get(key))
        if value is not None:
            values[key] = value

    for nested_key in nested_keys:
        nested = source.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            if key in values:
                continue
            value = _as_float(nested.get(key))
            if value is not None:
                values[key] = value

    return values


def _load_experiment_live_status(path: Path) -> dict[str, Any] | None:
    payload = load_live_status(path)
    if payload is None:
        return None
    job_kind = _norm_text(payload.get("job_kind"))
    if job_kind is not None and job_kind != "experiment_run":
        return None
    return payload


def _load_parse_cache_active_runs(parsed_docs_root: Path, *, now_utc: datetime) -> list[ParseCacheRunView]:
    run_root = parsed_docs_root / "_runs"
    if not run_root.exists():
        return []

    rows: list[ParseCacheRunView] = []
    for live_status_path in sorted(run_root.glob("*/live_status.json")):
        payload = load_live_status(live_status_path)
        if payload is None:
            continue
        job_kind = _norm_text(payload.get("job_kind"))
        if job_kind is not None and job_kind != "parse_cache_build":
            continue
        status = (_norm_text(payload.get("status")) or "").lower()
        if status not in ACTIVE_RUN_STATUSES:
            continue

        updated_at_utc = _norm_text(payload.get("updated_at_utc"))
        stalled_after_seconds = _as_optional_int(payload.get("stalled_after_seconds"))
        stalled = _is_stalled(
            manifest_path=live_status_path,
            now_utc=now_utc,
            live_updated_at_utc=updated_at_utc,
            stalled_after_seconds=stalled_after_seconds,
        )

        rows.append(
            ParseCacheRunView(
                run_id=_norm_text(payload.get("run_id")) or live_status_path.parent.name,
                status=status,
                mode=_norm_text(payload.get("mode")),
                config_path=_norm_text(payload.get("config_path")),
                started_at_utc=_norm_text(payload.get("started_at_utc")),
                updated_at_utc=updated_at_utc,
                finished_at_utc=_norm_text(payload.get("finished_at_utc")),
                stalled=stalled,
                processed=_as_optional_int(payload.get("processed")),
                total=_as_optional_int(payload.get("total")),
                current_source_relative_path=_norm_text(payload.get("current_source_relative_path")),
                hits=_as_int(payload.get("hits")),
                planned_new=_as_int(payload.get("planned_new")),
                parsed=_as_int(payload.get("parsed")),
                failed=_as_int(payload.get("failed")),
            )
        )

    rows.sort(
        key=lambda row: (
            _parse_timestamp(row.updated_at_utc) or _parse_timestamp(row.started_at_utc) or datetime.fromtimestamp(0, tz=UTC)
        ),
        reverse=True,
    )
    return rows


def _sort_timestamp_key(row: RunView) -> float:
    timestamp = _parse_timestamp(row.started_at_utc)
    if timestamp is None:
        timestamp = _parse_timestamp(row.timestamp_utc)
    if timestamp is None:
        timestamp = _parse_timestamp(row.finished_at_utc)
    return timestamp.timestamp() if timestamp is not None else 0.0


def _completed_sort_key(row: RunView) -> tuple[bool, float, float, str]:
    return (
        row.f1 is None,
        -(row.f1 or 0.0),
        -_sort_timestamp_key(row),
        row.run_id,
    )


def _is_stalled(
    *,
    manifest_path: Path,
    now_utc: datetime,
    live_updated_at_utc: str | None = None,
    stalled_after_seconds: int | None = None,
) -> bool:
    threshold = (
        timedelta(seconds=stalled_after_seconds)
        if isinstance(stalled_after_seconds, int) and stalled_after_seconds > 0
        else STALL_AFTER
    )

    live_updated = _parse_timestamp(live_updated_at_utc)
    if live_updated is not None:
        return (now_utc - live_updated) > threshold

    try:
        changed_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=UTC)
    except FileNotFoundError:
        return False
    return (now_utc - changed_at) > threshold


def _parse_timestamp(value: str | None) -> datetime | None:
    text = _norm_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(stdev(values) / sqrt(len(values)))


def _first_float(*values: Any) -> float | None:
    for value in values:
        converted = _as_float(value)
        if converted is not None:
            return converted
    return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _format_metric_with_se(value: float | None, se: float | None) -> str:
    mean_text = _format_metric(value)
    if value is None or se is None:
        return mean_text
    return f"{mean_text} ± {se:.4f}"


def _format_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.2f}"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _norm_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


def _safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0:
        return 0.0
    return (2 * precision * recall) / denom


def _company_year_from_doc(doc_name: str) -> str:
    match = DOC_COMPANY_YEAR_PATTERN.match(doc_name.lower())
    if match is None:
        return doc_name
    return f"{match.group('ticker').upper()} {match.group('year')}"
