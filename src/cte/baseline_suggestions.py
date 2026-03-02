from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .dashboard.data import RunView, load_dashboard_snapshot


@dataclass(slots=True)
class ExistingBaselineRow:
    baseline_key: str
    pipeline_version: str
    model_generation: str
    baseline_run_id: str | None
    status: str | None
    updated_at_utc: str | None
    notes: str | None


@dataclass(slots=True)
class BaselineSuggestionRow:
    baseline_key: str
    pipeline_version: str
    model_generation: str
    current_baseline_run_id: str | None
    suggested_run_id: str | None
    suggested_f1: float | None
    suggested_recall: float | None
    suggested_precision: float | None
    action: str
    notes: str


@dataclass(slots=True)
class BaselineSuggestionResult:
    rows: list[BaselineSuggestionRow]
    markdown_table: str


def suggest_baselines(
    *,
    experiments_root: Path,
    baselines_file: Path,
) -> BaselineSuggestionResult:
    snapshot = load_dashboard_snapshot(experiments_root)
    existing = _load_existing_baselines(baselines_file)
    candidates = _best_completed_candidates(snapshot.completed_runs)

    keys = sorted(set(existing.keys()) | set(candidates.keys()))
    rows: list[BaselineSuggestionRow] = []
    for key in keys:
        pipeline_version, model_generation = key
        existing_row = existing.get(key)
        candidate = candidates.get(key)
        current_run_id = existing_row.baseline_run_id if existing_row is not None else None
        suggested_run_id = candidate.run_id if candidate is not None else None
        action = _classify_action(current_run_id, suggested_run_id)

        notes_parts: list[str] = []
        if candidate is not None:
            notes_parts.append(
                f"candidate from completed runs ({candidate.run_id}, f1={candidate.f1_display})"
            )
        else:
            notes_parts.append("no completed run with metrics found")
        if existing_row is not None and existing_row.status:
            notes_parts.append(f"current status={existing_row.status}")
        if existing_row is not None and existing_row.notes:
            notes_parts.append(f"current note: {existing_row.notes}")

        rows.append(
            BaselineSuggestionRow(
                baseline_key=(
                    existing_row.baseline_key
                    if existing_row is not None
                    else _default_baseline_key(pipeline_version, model_generation)
                ),
                pipeline_version=pipeline_version,
                model_generation=model_generation,
                current_baseline_run_id=current_run_id,
                suggested_run_id=suggested_run_id,
                suggested_f1=candidate.f1 if candidate is not None else None,
                suggested_recall=candidate.recall if candidate is not None else None,
                suggested_precision=candidate.precision if candidate is not None else None,
                action=action,
                notes="; ".join(notes_parts),
            )
        )

    rows.sort(
        key=lambda row: (
            row.pipeline_version,
            row.model_generation,
            _action_rank(row.action),
            row.baseline_key,
        )
    )
    return BaselineSuggestionResult(rows=rows, markdown_table=_format_markdown_table(rows))


def _action_rank(action: str) -> int:
    order = {
        "review_replace": 0,
        "set": 1,
        "no_candidate": 2,
        "keep": 3,
    }
    return order.get(action, 9)


def _classify_action(current_run_id: str | None, suggested_run_id: str | None) -> str:
    if suggested_run_id is None:
        return "no_candidate"
    if current_run_id is None:
        return "set"
    if current_run_id == suggested_run_id:
        return "keep"
    return "review_replace"


def _best_completed_candidates(rows: list[RunView]) -> dict[tuple[str, str], RunView]:
    candidates: dict[tuple[str, str], RunView] = {}

    for row in rows:
        if row.run_status != "completed":
            continue
        if row.f1 is None:
            continue
        if row.pipeline_version is None or row.model_generation is None:
            continue
        key = (row.pipeline_version, row.model_generation)
        best = candidates.get(key)
        if best is None or _candidate_sort_tuple(row) > _candidate_sort_tuple(best):
            candidates[key] = row

    return candidates


def _candidate_sort_tuple(row: RunView) -> tuple[float, float, float, float]:
    return (
        row.f1 or -1.0,
        row.recall or -1.0,
        row.precision or -1.0,
        _parse_timestamp(row.timestamp_utc or row.finished_at_utc or row.started_at_utc),
    )


def _parse_timestamp(text: str | None) -> float:
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _load_existing_baselines(path: Path) -> dict[tuple[str, str], ExistingBaselineRow]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_index: int | None = None
    headers: list[str] = []

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        candidate_headers = [cell.strip() for cell in stripped.strip("|").split("|")]
        if {"baseline_key", "pipeline_version", "model_generation"}.issubset(candidate_headers):
            header_index = index
            headers = candidate_headers
            break

    if header_index is None:
        return {}

    rows: dict[tuple[str, str], ExistingBaselineRow] = {}
    for raw_line in lines[header_index + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            break
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        if all(cell and set(cell) <= {"-"} for cell in cells):
            continue

        row = dict(zip(headers, cells, strict=True))
        pipeline_version = row.get("pipeline_version", "").strip()
        model_generation = row.get("model_generation", "").strip()
        if not pipeline_version or not model_generation:
            continue

        baseline_run_id = _normalize_run_id(row.get("baseline_run_id"))
        existing = ExistingBaselineRow(
            baseline_key=row.get("baseline_key", "").strip() or _default_baseline_key(
                pipeline_version, model_generation
            ),
            pipeline_version=pipeline_version,
            model_generation=model_generation,
            baseline_run_id=baseline_run_id,
            status=_normalize_text(row.get("status")),
            updated_at_utc=_normalize_text(row.get("updated_at_utc")),
            notes=_normalize_text(row.get("notes")),
        )
        rows[(pipeline_version, model_generation)] = existing

    return rows


def _normalize_run_id(value: str | None) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    if text in {"-", "n/a", "none", "null"}:
        return None
    return text


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text if text else None


def _default_baseline_key(pipeline_version: str, model_generation: str) -> str:
    text = f"{pipeline_version}_{model_generation}".lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "baseline_key"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _format_text(value: str | None) -> str:
    return value if value else "-"


def _format_markdown_table(rows: list[BaselineSuggestionRow]) -> str:
    lines = [
        "| baseline_key | pipeline_version | model_generation | current_baseline_run_id | suggested_run_id | f1 | recall | precision | action | notes |",
        "|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.baseline_key,
                    row.pipeline_version,
                    row.model_generation,
                    _format_text(row.current_baseline_run_id),
                    _format_text(row.suggested_run_id),
                    _format_metric(row.suggested_f1),
                    _format_metric(row.suggested_recall),
                    _format_metric(row.suggested_precision),
                    row.action,
                    row.notes.replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)
