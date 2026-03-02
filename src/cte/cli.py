from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    apply_overrides,
    load_run_config,
    maybe_load_env_file,
    resolve_source_docs_root,
)
from .eval import evaluate_from_dirs, evaluate_from_doc_names
from .experiment import (
    append_experiment_log,
    build_run_paths,
    ensure_gate_lineage,
    git_metadata,
    make_run_id,
    now_utc,
    sha256_file,
    write_manifest,
)
from .index_registry import IndexRegistry
from .io import ensure_dir, legacy_report_name, target_doc_name, write_json
from .live_status import LiveStatusTracker
from .prompts import (
    get_eval_prompt_path,
    get_extract_prompt_path,
    load_eval_prompt,
    load_extract_prompt,
)
from .progress import ProgressReporter
from .parse_cache import parse_settings_from_component_settings, run_parse_cache_build
from .prompt_cache import manifest_prompt_cache_fields
from .baseline_suggestions import suggest_baselines
from .reporting import audit_existing_results, compare_reports


DOC_PATTERN = re.compile(r"^(?P<ticker>[a-z0-9]+)\.(?P<year>\d{4})\.targets\.v1\.json$")


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _discover_pairs(pred_dir: Path) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for path in pred_dir.glob("*.json"):
        match = DOC_PATTERN.match(path.name)
        if match:
            entries.append((match.group("ticker").upper(), match.group("year"), path.name))
    if not entries:
        raise ValueError(f"No prediction files matching expected pattern found in {pred_dir}")
    return sorted(entries)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prediction_doc_names(pred_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in pred_dir.glob("*.json"):
        if DOC_PATTERN.match(path.name):
            names.add(path.name)
    return names


def _expected_doc_names(*, company_tickers: list[str], years: list[str]) -> set[str]:
    names: set[str] = set()
    for company_ticker in company_tickers:
        for year in years:
            names.add(target_doc_name(company_ticker, str(year)))
    return names


def _normalized_text_list(value: Any, *, uppercase: bool = False) -> list[str] | None:
    if not isinstance(value, list):
        return None
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text.upper() if uppercase else text)
    return sorted(normalized)


def _report_doc_names(report_payload: dict[str, Any]) -> set[str]:
    per_doc = report_payload.get("per_doc")
    if not isinstance(per_doc, list):
        return set()
    names: set[str] = set()
    for row in per_doc:
        if not isinstance(row, dict):
            continue
        doc_name = row.get("doc")
        if not isinstance(doc_name, str):
            continue
        normalized = doc_name.strip()
        if normalized:
            names.add(normalized)
    return names


def _upsert_run_record(run_records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    counter = int(record.get("counter", 0) or 0)
    for idx, existing in enumerate(run_records):
        if int(existing.get("counter", 0) or 0) == counter:
            run_records[idx] = record
            return
    run_records.append(record)
    run_records.sort(key=lambda row: int(row.get("counter", 0) or 0))


def _validate_resume_manifest(
    *,
    manifest: dict[str, Any],
    config: Any,
    parse_settings: Any,
    run_id: str,
    run_label: str,
) -> None:
    prompt_cache_fields = manifest_prompt_cache_fields(config)
    expected_company_tickers = _normalized_text_list(config.company_tickers, uppercase=True)
    expected_years = _normalized_text_list(config.years)
    expected_doc_names = sorted(
        _expected_doc_names(company_tickers=config.company_tickers, years=config.years)
    )
    if str(manifest.get("run_id", "")) != run_id:
        raise ValueError(
            f"Resume manifest run_id mismatch: expected '{run_id}', got '{manifest.get('run_id')}'."
        )
    if str(manifest.get("run_label", "")) != run_label:
        raise ValueError(
            f"Resume manifest run_label mismatch: expected '{run_label}', got '{manifest.get('run_label')}'."
        )
    comparisons: list[tuple[str, Any, Any]] = [
        ("pipeline", manifest.get("pipeline"), config.pipeline),
        ("pipeline_version", manifest.get("pipeline_version"), config.pipeline_version),
        ("model_alias", manifest.get("model_alias"), config.model_alias),
        ("model_name", manifest.get("model_name"), config.model_name),
        ("judge_model_name", manifest.get("judge_model_name"), config.judge_model_name),
        ("run_count", manifest.get("run_count"), config.run_count),
        (
            "company_tickers",
            _normalized_text_list(manifest.get("company_tickers"), uppercase=True),
            expected_company_tickers,
        ),
        ("years", _normalized_text_list(manifest.get("years")), expected_years),
        (
            "expected_doc_names",
            _normalized_text_list(manifest.get("expected_doc_names")),
            expected_doc_names,
        ),
        ("parent_run_id", manifest.get("parent_run_id"), config.parent_run_id),
        ("baseline_run_id", manifest.get("baseline_run_id"), config.baseline_run_id),
        ("prompt_versions", manifest.get("prompt_versions"), config.prompt_versions),
        ("component_versions", manifest.get("component_versions"), config.component_versions),
        ("component_settings", manifest.get("component_settings"), config.component_settings),
        (
            "target_postprocess_profile",
            manifest.get("target_postprocess_profile", "off"),
            config.target_postprocess_profile,
        ),
        (
            "retrieval_rerank_profile",
            manifest.get("retrieval_rerank_profile", "off"),
            config.retrieval_rerank_profile,
        ),
        ("pdf_source_mode", manifest.get("pdf_source_mode"), parse_settings.pdf_source_mode),
        (
            "prompt_cache_enabled",
            manifest.get("prompt_cache_enabled", False),
            prompt_cache_fields["prompt_cache_enabled"],
        ),
        (
            "prompt_cache_retention",
            manifest.get("prompt_cache_retention"),
            prompt_cache_fields["prompt_cache_retention"],
        ),
        (
            "prompt_cache_scope",
            manifest.get("prompt_cache_scope"),
            prompt_cache_fields["prompt_cache_scope"],
        ),
    ]
    mismatches: list[str] = []
    for key, actual, expected in comparisons:
        if actual != expected:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "Resume manifest config mismatch for fields: "
            + ", ".join(mismatches)
            + ". Use the same config or start a new run."
        )


def _run_command(args: argparse.Namespace) -> int:
    progress = ProgressReporter(enabled=not bool(getattr(args, "quiet", False)))

    config = load_run_config(Path(args.config))
    loaded_env_file = maybe_load_env_file(config.env_file)
    parse_settings = parse_settings_from_component_settings(config.component_settings)
    prompt_cache_fields = manifest_prompt_cache_fields(config)
    config = apply_overrides(
        config,
        index_policy=args.index_policy,
    )

    ensure_gate_lineage(config)

    extract_prompt_version = config.prompt_versions["extract"]
    eval_prompt_version = config.prompt_versions["eval"]
    extract_prompt_path = get_extract_prompt_path(config.pipeline, extract_prompt_version)
    eval_prompt_path = get_eval_prompt_path(eval_prompt_version)
    extract_prompt = load_extract_prompt(config.pipeline, extract_prompt_version)
    eval_prompt = load_eval_prompt(eval_prompt_version)
    schema_path = Path(__file__).with_name("schemas.py")
    schema_sha256 = sha256_file(schema_path)

    expected_doc_names = _expected_doc_names(
        company_tickers=config.company_tickers,
        years=config.years,
    )
    docs_per_run = len(config.company_tickers) * len(config.years)
    progress.stage(
        "setup",
        f"pipeline={config.pipeline_version} runs={config.run_count} docs_per_run={docs_per_run}",
    )

    requested_resume_run_id = str(args.resume_run_id).strip() if args.resume_run_id else None
    run_id = requested_resume_run_id or make_run_id(args.run_label)
    paths = build_run_paths(config, run_id)
    live_status_path = paths.experiment_dir / "live_status.json"

    git_meta = git_metadata(Path.cwd())
    started_at_utc = now_utc()
    if requested_resume_run_id:
        manifest_path = paths.experiment_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Resume requested but manifest not found for run_id={run_id}: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Resume manifest is not valid JSON: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Resume manifest has invalid payload type: {manifest_path}")

        _validate_resume_manifest(
            manifest=manifest,
            config=config,
            parse_settings=parse_settings,
            run_id=run_id,
            run_label=args.run_label,
        )
        previous_status = str(manifest.get("run_status", "unknown"))
        progress.stage("setup", f"resuming run_id={run_id} previous_status={previous_status}")

        existing_runs = manifest.get("runs")
        run_records = [dict(row) for row in existing_runs if isinstance(row, dict)] if isinstance(existing_runs, list) else []
        existing_index_events = manifest.get("index_events")
        all_index_events = (
            [dict(row) for row in existing_index_events if isinstance(row, dict)]
            if isinstance(existing_index_events, list)
            else []
        )
        manifest["runs"] = run_records
        manifest["index_events"] = all_index_events
        manifest["timestamp_utc"] = started_at_utc
        manifest["run_status"] = "running"
        manifest["failed_stage"] = None
        manifest["error_message"] = None
        manifest["finished_at_utc"] = None
        manifest["resumed_at_utc"] = started_at_utc
        resume_history = manifest.get("resume_history")
        history_rows: list[dict[str, Any]] = (
            [dict(row) for row in resume_history if isinstance(row, dict)]
            if isinstance(resume_history, list)
            else []
        )
        history_rows.append({"timestamp_utc": started_at_utc, "previous_status": previous_status})
        manifest["resume_history"] = history_rows
        manifest["env_file_loaded"] = str(loaded_env_file) if loaded_env_file else None
        manifest.update(prompt_cache_fields)
        manifest["model_alias"] = config.model_alias
        manifest["run_count"] = config.run_count
        manifest["company_tickers"] = list(config.company_tickers)
        manifest["years"] = list(config.years)
        manifest["expected_doc_names"] = sorted(expected_doc_names)
        manifest["target_postprocess_profile"] = config.target_postprocess_profile
        manifest["retrieval_rerank_profile"] = config.retrieval_rerank_profile
    else:
        run_records = []
        all_index_events = []
        manifest = {
            "run_id": run_id,
            "run_label": args.run_label,
            "timestamp_utc": started_at_utc,
            "started_at_utc": started_at_utc,
            "finished_at_utc": None,
            "run_status": "running",
            "failed_stage": None,
            "error_message": None,
            "pipeline": config.pipeline,
            "pipeline_version": config.pipeline_version,
            "model_alias": config.model_alias,
            "model_name": config.model_name,
            "judge_model_name": config.judge_model_name,
            "run_count": config.run_count,
            "company_tickers": list(config.company_tickers),
            "years": list(config.years),
            "expected_doc_names": sorted(expected_doc_names),
            "prompt_versions": config.prompt_versions,
            "component_versions": config.component_versions,
            "component_settings": config.component_settings,
            "target_postprocess_profile": config.target_postprocess_profile,
            "retrieval_rerank_profile": config.retrieval_rerank_profile,
            "parent_run_id": config.parent_run_id,
            "changed_components": config.changed_components,
            "change_reason": config.change_reason,
            "baseline_run_id": config.baseline_run_id,
            "index_id": None,
            "index_fingerprint": None,
            "index_action": None,
            "pdf_source_mode": parse_settings.pdf_source_mode,
            "parsed_cache_provider": parse_settings.provider if parse_settings.pdf_source_mode == "cache_only" else None,
            "parsed_cache_profile_key": parse_settings.profile_key()
            if parse_settings.pdf_source_mode == "cache_only"
            else None,
            "parsed_cache_fingerprint": None,
            "parsed_cache_hit_count": None,
            "parsed_cache_miss_count": None,
            "prompt_cache_enabled": prompt_cache_fields["prompt_cache_enabled"],
            "prompt_cache_retention": prompt_cache_fields["prompt_cache_retention"],
            "prompt_cache_scope": prompt_cache_fields["prompt_cache_scope"],
            "index_events": all_index_events,
            "git_branch": git_meta["git_branch"],
            "git_commit": git_meta["git_commit"],
            "git_dirty": git_meta.get("git_dirty"),
            "provenance": {
                "extract_prompt_path": str(extract_prompt_path),
                "extract_prompt_sha256": _sha256_text(extract_prompt),
                "eval_prompt_path": str(eval_prompt_path),
                "eval_prompt_sha256": _sha256_text(eval_prompt),
                "schema_path": str(schema_path),
                "schema_sha256": schema_sha256,
            },
            "runs": run_records,
            "artifacts": {
                "experiment_dir": str(paths.experiment_dir),
                "generated_targets_dir": str(paths.generated_targets_dir),
                "results_dir": str(paths.results_dir),
                "analysis_dir": str(paths.analysis_dir),
                "indexes_root": str(paths.indexes_root),
            },
            "env_file_loaded": str(loaded_env_file) if loaded_env_file else None,
        }

    manifest["live_status_path"] = str(live_status_path)
    live_started_at_utc = str(manifest.get("started_at_utc") or started_at_utc)
    live_status = LiveStatusTracker(
        path=live_status_path,
        job_kind="experiment_run",
        run_id=run_id,
        started_at_utc=live_started_at_utc,
        initial={
            "run_label": args.run_label,
            "pipeline": config.pipeline,
            "pipeline_version": config.pipeline_version,
            "model_name": config.model_name,
            "judge_model_name": config.judge_model_name,
            "run_count_total": config.run_count,
            "run_counter_current": 0,
            "stage": "setup",
            "extract_progress": {
                "completed": 0,
                "total": docs_per_run,
                "current_doc_name": None,
            },
            "evaluate_progress": {
                "completed": 0,
                "total": docs_per_run,
                "current_doc_name": None,
            },
        },
    )

    manifest_path = write_manifest(paths.experiment_dir, manifest)
    current_stage = "extract"

    try:
        for counter in range(1, config.run_count + 1):
            existing_record = next(
                (
                    row
                    for row in run_records
                    if isinstance(row, dict) and int(row.get("counter", 0) or 0) == counter
                ),
                None,
            )
            current_stage = "extract"
            progress.stage("extract", f"run {counter}/{config.run_count} start")
            live_status.update(
                {
                    "stage": "extract",
                    "run_counter_current": counter,
                    "extract_progress": {
                        "completed": 0,
                        "total": docs_per_run,
                        "current_doc_name": None,
                    },
                    "evaluate_progress": {
                        "completed": 0,
                        "total": docs_per_run,
                        "current_doc_name": None,
                    },
                }
            )
            pred_dir = ensure_dir(paths.generated_targets_dir / f"run_{counter}")
            existing_doc_names = _prediction_doc_names(pred_dir)
            existing_expected_doc_names = existing_doc_names & expected_doc_names
            missing_expected_doc_names = expected_doc_names - existing_doc_names
            skip_doc_names = (
                existing_expected_doc_names
                if requested_resume_run_id and existing_expected_doc_names
                else None
            )

            def _extract_progress(doc_name: str, current: int, total: int) -> None:
                progress.doc("extract", current, total, doc_name)
                live_status.update(
                    {
                        "stage": "extract",
                        "run_counter_current": counter,
                        "extract_progress": {
                            "completed": current,
                            "total": total,
                            "current_doc_name": doc_name,
                        },
                    }
                )

            if requested_resume_run_id and not missing_expected_doc_names:
                written_paths = []
                index_events = []
                progress.stage(
                    "extract",
                    (
                        f"run {counter}/{config.run_count} reused existing predictions "
                        f"({len(existing_expected_doc_names)}/{len(expected_doc_names)})"
                    ),
                )
                live_status.update(
                    {
                        "stage": "extract",
                        "run_counter_current": counter,
                        "extract_progress": {
                            "completed": len(existing_expected_doc_names),
                            "total": len(expected_doc_names),
                            "current_doc_name": None,
                        },
                    }
                )
            else:
                if config.pipeline == "no_rag":
                    from .pipelines.no_rag.v1 import run_batch

                    run_batch_kwargs: dict[str, Any] = {
                        "config": config,
                        "output_dir": pred_dir,
                        "system_prompt": extract_prompt,
                        "progress_fn": _extract_progress,
                    }
                    if skip_doc_names:
                        run_batch_kwargs["skip_doc_names"] = skip_doc_names
                    written_paths = run_batch(**run_batch_kwargs)
                    index_events = []
                else:
                    from .pipelines.rag.v1 import run_batch

                    source_docs_root = resolve_source_docs_root(config)
                    registry = IndexRegistry(paths.indexes_root)
                    run_batch_kwargs = {
                        "config": config,
                        "config_path": Path(args.config),
                        "output_dir": pred_dir,
                        "system_prompt": extract_prompt,
                        "source_docs_root": source_docs_root,
                        "index_registry": registry,
                        "progress_fn": _extract_progress,
                    }
                    if skip_doc_names:
                        run_batch_kwargs["skip_doc_names"] = skip_doc_names
                    written_paths, index_events = run_batch(**run_batch_kwargs)

            all_index_events.extend(index_events)
            prediction_doc_names = _prediction_doc_names(pred_dir)
            prediction_file_count = len(prediction_doc_names & expected_doc_names)
            missing_prediction_doc_names = sorted(expected_doc_names - prediction_doc_names)
            if missing_prediction_doc_names:
                missing_preview = ", ".join(missing_prediction_doc_names[:5])
                if len(missing_prediction_doc_names) > 5:
                    missing_preview += ", ..."
                raise ValueError(
                    f"Missing expected prediction docs for run {counter}: {missing_preview}"
                )
            progress.stage(
                "extract",
                (
                    f"run {counter}/{config.run_count} complete "
                    f"(new={len(written_paths)} total={prediction_file_count} docs)"
                ),
            )
            live_status.update(
                {
                    "stage": "extract",
                    "run_counter_current": counter,
                    "extract_progress": {
                        "completed": prediction_file_count,
                        "total": len(expected_doc_names),
                        "current_doc_name": None,
                    },
                }
            )

            report_path: Path | None = None
            report_payload: dict[str, Any] | None = None
            if not args.skip_eval:
                live_status.update({"stage": "evaluate", "run_counter_current": counter})
                existing_report_path = (
                    str(existing_record.get("report_path", "")).strip()
                    if existing_record is not None
                    else ""
                )
                can_reuse_report = False
                if (
                    requested_resume_run_id
                    and len(written_paths) == 0
                    and prediction_file_count == len(expected_doc_names)
                    and existing_report_path
                    and Path(existing_report_path).exists()
                ):
                    candidate_report_path = Path(existing_report_path)
                    try:
                        candidate_payload = json.loads(
                            candidate_report_path.read_text(encoding="utf-8")
                        )
                    except json.JSONDecodeError:
                        candidate_payload = None
                    if (
                        isinstance(candidate_payload, dict)
                        and _report_doc_names(candidate_payload) == expected_doc_names
                    ):
                        report_path = candidate_report_path
                        report_payload = candidate_payload
                        can_reuse_report = True
                if can_reuse_report:
                    progress.stage(
                        "evaluate",
                        f"run {counter}/{config.run_count} reused existing report",
                    )
                    live_status.update(
                        {
                            "stage": "evaluate",
                            "run_counter_current": counter,
                            "evaluate_progress": {
                                "completed": len(expected_doc_names),
                                "total": len(expected_doc_names),
                                "current_doc_name": None,
                            },
                        }
                    )
                else:
                    current_stage = "evaluate"
                    progress.stage("evaluate", f"run {counter}/{config.run_count} start")

                    def _evaluate_progress(doc_name: str, current: int, total: int) -> None:
                        progress.doc("evaluate", current, total, doc_name)
                        live_status.update(
                            {
                                "stage": "evaluate",
                                "run_counter_current": counter,
                                "evaluate_progress": {
                                    "completed": current,
                                    "total": total,
                                    "current_doc_name": doc_name,
                                },
                            }
                        )

                    report_payload = evaluate_from_dirs(
                        pred_dir,
                        config.reference_targets_dir,
                        company_tickers=config.company_tickers,
                        years=config.years,
                        eval_system_prompt=eval_prompt,
                        judge_model_name=config.judge_model_name,
                        progress_fn=_evaluate_progress,
                    )
                    report_file_name = legacy_report_name(config.pipeline, config.model_alias, counter)
                    report_path = write_json(paths.results_dir / report_file_name, report_payload)
                    progress.stage("evaluate", f"run {counter}/{config.run_count} complete")
                    live_status.update(
                        {
                            "stage": "evaluate",
                            "run_counter_current": counter,
                            "evaluate_progress": {
                                "completed": docs_per_run,
                                "total": docs_per_run,
                                "current_doc_name": None,
                            },
                        }
                    )
            else:
                progress.stage("evaluate", f"run {counter}/{config.run_count} skipped (--skip-eval)")
                live_status.update(
                    {
                        "stage": "evaluate",
                        "run_counter_current": counter,
                        "evaluate_progress": {
                            "completed": 0,
                            "total": docs_per_run,
                            "current_doc_name": None,
                            "skipped": True,
                        },
                    }
                )

            existing_aggregate = existing_record.get("aggregate") if existing_record else None
            existing_report_value = (
                str(existing_record.get("report_path", "")).strip() if existing_record else ""
            )
            record = {
                "counter": counter,
                "prediction_dir": str(pred_dir),
                "prediction_files": prediction_file_count,
                "report_path": (
                    str(report_path)
                    if report_path
                    else (existing_report_value or None)
                ),
                "aggregate": (
                    report_payload.get("aggregate")
                    if report_payload
                    else (existing_aggregate if isinstance(existing_aggregate, dict) else None)
                ),
            }
            _upsert_run_record(run_records, record)
            manifest["timestamp_utc"] = now_utc()
            write_manifest(paths.experiment_dir, manifest)

        current_stage = "finalize"
        live_status.update({"stage": "finalize", "run_counter_current": config.run_count})
        index_ids = sorted({event["index_id"] for event in all_index_events})
        index_fingerprints = sorted({event["index_fingerprint"] for event in all_index_events})
        index_actions = sorted({event["index_action"] for event in all_index_events})
        manifest["index_id"] = ",".join(index_ids) if index_ids else None
        manifest["index_fingerprint"] = ",".join(index_fingerprints) if index_fingerprints else None
        manifest["index_action"] = (
            "mixed" if len(index_actions) > 1 else (index_actions[0] if index_actions else None)
        )
        parsed_cache_fingerprints = sorted(
            {
                event.get("parsed_cache_fingerprint")
                for event in all_index_events
                if event.get("parsed_cache_fingerprint")
            }
        )
        if parsed_cache_fingerprints:
            manifest["parsed_cache_fingerprint"] = ",".join(parsed_cache_fingerprints)
            manifest["parsed_cache_hit_count"] = len(parsed_cache_fingerprints)
            manifest["parsed_cache_miss_count"] = 0

        finished_at_utc = now_utc()
        manifest["timestamp_utc"] = finished_at_utc
        manifest["finished_at_utc"] = finished_at_utc
        manifest["run_status"] = "completed"
        manifest["failed_stage"] = None
        manifest["error_message"] = None

        progress.stage("finalize", "writing manifest and experiment log")
        manifest_path = write_manifest(paths.experiment_dir, manifest)
        live_status.finalize(
            status="completed",
            extra={
                "stage": "finalize",
                "run_counter_current": config.run_count,
            },
        )
        log_path = append_experiment_log(config.artifacts_root, manifest)

        print(f"run_id={run_id}")
        print(f"manifest={manifest_path}")
        print(f"experiment_log={log_path}")
        return 0
    except BaseException as exc:
        index_ids = sorted({event["index_id"] for event in all_index_events})
        index_fingerprints = sorted({event["index_fingerprint"] for event in all_index_events})
        index_actions = sorted({event["index_action"] for event in all_index_events})
        manifest["index_id"] = ",".join(index_ids) if index_ids else None
        manifest["index_fingerprint"] = ",".join(index_fingerprints) if index_fingerprints else None
        manifest["index_action"] = (
            "mixed" if len(index_actions) > 1 else (index_actions[0] if index_actions else None)
        )
        parsed_cache_fingerprints = sorted(
            {
                event.get("parsed_cache_fingerprint")
                for event in all_index_events
                if event.get("parsed_cache_fingerprint")
            }
        )
        if parsed_cache_fingerprints:
            manifest["parsed_cache_fingerprint"] = ",".join(parsed_cache_fingerprints)
            manifest["parsed_cache_hit_count"] = len(parsed_cache_fingerprints)
            manifest["parsed_cache_miss_count"] = 0

        finished_at_utc = now_utc()
        manifest["timestamp_utc"] = finished_at_utc
        manifest["finished_at_utc"] = finished_at_utc
        manifest["run_status"] = "aborted"
        manifest["failed_stage"] = current_stage
        manifest["error_message"] = f"{type(exc).__name__}: {exc}"

        manifest_path = write_manifest(paths.experiment_dir, manifest)
        live_status.finalize(
            status="aborted",
            error_message=f"{type(exc).__name__}: {exc}",
            extra={
                "stage": "abort",
                "run_counter_current": int(
                    max((int(row.get("counter", 0) or 0) for row in run_records), default=0)
                ),
            },
        )
        log_path = append_experiment_log(config.artifacts_root, manifest)

        progress.stage(
            "abort",
            f"run_status=aborted stage={current_stage} error={type(exc).__name__}",
        )
        progress.stage("abort", f"manifest={manifest_path}")
        progress.stage("abort", f"experiment_log={log_path}")
        raise


def _evaluate_command(args: argparse.Namespace) -> int:
    progress = ProgressReporter(enabled=not bool(getattr(args, "quiet", False)))

    maybe_load_env_file(Path(args.env_file) if args.env_file else None)
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    tickers = _parse_csv(args.company_tickers)
    years = _parse_csv(args.years)
    eval_prompt = load_eval_prompt(args.eval_prompt_version)
    progress.stage("setup", f"pred_dir={pred_dir} ref_dir={ref_dir}")

    if tickers and years:
        progress.stage(
            "discover",
            f"using explicit filters: {len(tickers)} tickers x {len(years)} years",
        )
        progress.stage("evaluate", f"scoring {len(tickers) * len(years)} docs")
        report = evaluate_from_dirs(
            pred_dir,
            ref_dir,
            company_tickers=tickers,
            years=years,
            eval_system_prompt=eval_prompt,
            judge_model_name=args.judge_model,
            progress_fn=lambda doc_name, current, total: progress.doc(
                "evaluate", current, total, doc_name
            ),
        )
    else:
        discovered = _discover_pairs(pred_dir)
        progress.stage("discover", f"discovered {len(discovered)} docs from prediction dir")
        ticker_filter = {ticker.upper() for ticker in tickers} if tickers else None
        year_filter = set(years) if years else None
        doc_names = [
            doc_name
            for ticker, year, doc_name in discovered
            if (ticker_filter is None or ticker in ticker_filter)
            and (year_filter is None or year in year_filter)
        ]
        if not doc_names:
            raise ValueError("No prediction files matched the requested company/year filters")
        progress.stage("discover", f"selected {len(doc_names)} docs after filters")
        progress.stage("evaluate", f"scoring {len(doc_names)} docs")
        report = evaluate_from_doc_names(
            pred_dir,
            ref_dir,
            doc_names=doc_names,
            eval_system_prompt=eval_prompt,
            judge_model_name=args.judge_model,
            progress_fn=lambda doc_name, current, total: progress.doc(
                "evaluate", current, total, doc_name
            ),
        )

    out_path = write_json(Path(args.out), report)
    print(f"wrote {out_path}")
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    outputs = compare_reports(
        Path(args.baseline_report),
        Path(args.candidate_report),
        Path(args.out_dir),
    )
    for key, path in outputs.items():
        print(f"{key}={path}")
    return 0


def _audit_command(args: argparse.Namespace) -> int:
    notes_file = Path(args.run_notes_file) if args.run_notes_file else None
    summary = audit_existing_results(
        Path(args.results_root),
        Path(args.out_dir),
        experiments_root=Path(args.experiments_root),
        run_notes_file=notes_file,
    )

    print(f"framework_runs={summary['framework_runs']}")
    print(
        "linked_comparisons="
        f"{summary['linked_comparisons_total']} "
        f"(comparable={summary['linked_comparable']}, "
        f"not_comparable={summary['linked_not_comparable']})"
    )
    print(f"legacy_reports={summary['legacy_reports']}")

    print("recent_framework_runs:")
    for row in summary["recent_framework_rows"]:
        micro_f1 = row.get("micro_f1_mean")
        micro_f1_text = f"{micro_f1:.4f}" if isinstance(micro_f1, (int, float)) else "n/a"
        print(
            "- "
            f"{row.get('timestamp_utc') or 'n/a'} | "
            f"{row.get('label') or row.get('run_id') or 'n/a'} | "
            f"{row.get('pipeline_version') or 'n/a'} | "
            f"micro_f1={micro_f1_text} | "
            f"run_status={row.get('run_status') or row.get('status') or 'n/a'}"
        )

    print(f"summary_markdown={summary['summary_markdown']}")
    print(f"experiment_run_log_csv={summary['experiment_run_log_csv']}")
    print(f"linked_comparisons_csv={summary['linked_comparisons_csv']}")
    print(f"legacy_results_log_csv={summary['legacy_results_log_csv']}")
    print(f"experiment_run_log_jsonl={summary['experiment_run_log_jsonl']}")
    return 0


def _dashboard_command(args: argparse.Namespace) -> int:
    from .dashboard.app import run_dashboard

    try:
        run_dashboard(
            experiments_root=Path(args.experiments_root),
            parsed_docs_root=Path(args.parsed_docs_root),
            host=args.host,
            port=args.port,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _suggest_baselines_command(args: argparse.Namespace) -> int:
    result = suggest_baselines(
        experiments_root=Path(args.experiments_root),
        baselines_file=Path(args.baselines_file),
    )

    print(f"experiments_root={args.experiments_root}")
    print(f"baselines_file={args.baselines_file}")
    print(f"suggested_rows={len(result.rows)}")
    print("mode=suggest-only (no files modified)")
    print("")
    print(result.markdown_table)
    return 0


def _parse_cache_command(args: argparse.Namespace) -> int:
    if args.parse_cache_command == "build":
        return run_parse_cache_build(args)
    raise ValueError(f"Unknown parse-cache command: {args.parse_cache_command}")


def _load_status_snapshot(*, experiments_root: Path, parsed_docs_root: Path) -> Any:
    from .dashboard.data import load_dashboard_snapshot

    return load_dashboard_snapshot(
        experiments_root,
        parsed_docs_root=parsed_docs_root,
    )


def _format_progress(completed: int | None, total: int | None) -> str:
    if not isinstance(total, int) or total <= 0:
        return "-"
    if not isinstance(completed, int):
        return f"0/{total}"
    return f"{completed}/{total}"


def _print_status_snapshot(snapshot: Any) -> None:
    print(f"generated_at_utc={getattr(snapshot, 'generated_at_utc', '-')}")

    active_runs = list(getattr(snapshot, "active_runs", []))
    parse_cache_active_runs = list(getattr(snapshot, "parse_cache_active_runs", []))

    print(f"active_experiment_runs={len(active_runs)}")
    if not active_runs:
        print("- none")
    for run in active_runs:
        stalled_marker = " stalled" if getattr(run, "stalled", False) else ""
        print(
            "- "
            f"{run.run_id} [{run.run_status}{stalled_marker}] "
            f"stage={getattr(run, 'live_stage', None) or '-'} "
            f"counter={_format_progress(getattr(run, 'live_run_counter_current', None), getattr(run, 'live_run_counter_total', None))} "
            f"extract={_format_progress(getattr(run, 'live_extract_completed', None), getattr(run, 'live_extract_total', None))} "
            f"evaluate={_format_progress(getattr(run, 'live_evaluate_completed', None), getattr(run, 'live_evaluate_total', None))} "
            f"updated={getattr(run, 'live_updated_at_utc', None) or getattr(run, 'timestamp_utc', None) or '-'}"
        )

    print(f"active_parse_cache_jobs={len(parse_cache_active_runs)}")
    if not parse_cache_active_runs:
        print("- none")
    for job in parse_cache_active_runs:
        stalled_marker = " stalled" if getattr(job, "stalled", False) else ""
        print(
            "- "
            f"{job.run_id} [{job.status}{stalled_marker}] "
            f"mode={job.mode or '-'} "
            f"progress={_format_progress(job.processed, job.total)} "
            f"counts=hits:{job.hits} planned:{job.planned_new} parsed:{job.parsed} failed:{job.failed} "
            f"current={job.current_source_relative_path or '-'} "
            f"updated={job.updated_at_utc or '-'}"
        )


def _status_command(args: argparse.Namespace) -> int:
    experiments_root = Path(args.experiments_root)
    parsed_docs_root = Path(args.parsed_docs_root)
    interval_sec = max(1.0, float(args.interval_sec))

    try:
        while True:
            snapshot = _load_status_snapshot(
                experiments_root=experiments_root,
                parsed_docs_root=parsed_docs_root,
            )
            _print_status_snapshot(snapshot)
            if not args.watch:
                break
            time.sleep(interval_sec)
            print("")
    except KeyboardInterrupt:
        print("status watch stopped")
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cte")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one configured experiment")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--run-label", required=True)
    run_parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Resume an existing aborted/running run by run_id using the same config and run label.",
    )
    run_parser.add_argument(
        "--index-policy",
        choices=["reuse_or_build", "reuse_only", "rebuild"],
        default=None,
    )
    run_parser.add_argument("--skip-eval", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate prediction JSONs")
    eval_parser.add_argument("--pred-dir", required=True)
    eval_parser.add_argument("--ref-dir", required=True)
    eval_parser.add_argument("--out", required=True)
    eval_parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    eval_parser.add_argument("--eval-prompt-version", default="v001")
    eval_parser.add_argument("--company-tickers", default=None)
    eval_parser.add_argument("--years", default=None)
    eval_parser.add_argument("--env-file", default=".env.local")
    eval_parser.add_argument("--quiet", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation reports")
    compare_parser.add_argument("--baseline-report", required=True)
    compare_parser.add_argument("--candidate-report", required=True)
    compare_parser.add_argument("--out-dir", required=True)

    audit_parser = subparsers.add_parser("audit-existing", help="Summarize existing result reports")
    audit_parser.add_argument("--results-root", default="data/results")
    audit_parser.add_argument("--out-dir", default="artifacts/audit")
    audit_parser.add_argument("--experiments-root", default="artifacts/experiments")
    audit_parser.add_argument("--run-notes-file", default="artifacts/experiments/run_notes.toml")

    dashboard_parser = subparsers.add_parser("dashboard", help="Run local read-only dashboard")
    dashboard_parser.add_argument("--experiments-root", default="artifacts/experiments")
    dashboard_parser.add_argument("--parsed-docs-root", default="artifacts/parsed_docs")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8000)

    status_parser = subparsers.add_parser(
        "status",
        help="Show active run/build status from live dashboard artifacts",
    )
    status_parser.add_argument("--experiments-root", default="artifacts/experiments")
    status_parser.add_argument("--parsed-docs-root", default="artifacts/parsed_docs")
    status_parser.add_argument("--watch", action="store_true")
    status_parser.add_argument("--interval-sec", type=float, default=5.0)

    suggest_baselines_parser = subparsers.add_parser(
        "suggest-baselines",
        help="Suggest baseline mappings from completed runs (manual review only)",
    )
    suggest_baselines_parser.add_argument("--experiments-root", default="artifacts/experiments")
    suggest_baselines_parser.add_argument("--baselines-file", default="docs/BASELINES.md")

    parse_cache_parser = subparsers.add_parser(
        "parse-cache",
        help="Build/review reusable paid PDF parse cache artifacts",
    )
    parse_cache_subparsers = parse_cache_parser.add_subparsers(
        dest="parse_cache_command", required=True
    )

    parse_cache_build_parser = parse_cache_subparsers.add_parser(
        "build",
        help="Plan or execute parse-cache generation for configured scope",
    )
    parse_cache_build_parser.add_argument("--config", required=True)
    parse_cache_build_parser.add_argument("--company-tickers", default=None)
    parse_cache_build_parser.add_argument("--years", default=None)
    parse_cache_build_parser.add_argument("--execute", action="store_true")
    parse_cache_build_parser.add_argument("--max-new-pdfs", type=int, default=None)
    parse_cache_build_parser.add_argument("--quiet", action="store_true")
    parse_cache_build_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "compare":
        return _compare_command(args)
    if args.command == "audit-existing":
        return _audit_command(args)
    if args.command == "dashboard":
        return _dashboard_command(args)
    if args.command == "status":
        return _status_command(args)
    if args.command == "suggest-baselines":
        return _suggest_baselines_command(args)
    if args.command == "parse-cache":
        return _parse_cache_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
