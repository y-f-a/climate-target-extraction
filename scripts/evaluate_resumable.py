from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cte.config import maybe_load_env_file
from cte.eval import (
    FIELDS_TO_SCORE,
    build_test_cases_from_doc_names,
    judge_align_and_score_openai,
)
from cte.io import write_json
from cte.prompts import load_eval_prompt

DOC_PATTERN = re.compile(r"^(?P<ticker>[a-z0-9]+)\.(?P<year>\d{4})\.targets\.v1\.json$")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


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


def _grade_to_score(value: str) -> float:
    if value == "EXACT":
        return 1.0
    if value == "PARTIAL":
        return 0.5
    return 0.0


def _build_doc_row(doc_name: str, judged: dict[str, Any]) -> dict[str, Any]:
    matches = judged.get("matches", [])
    unmatched_gold = judged.get("unmatched_gold", [])
    unmatched_pred = judged.get("unmatched_pred", [])

    field_accuracy = {field: None for field in FIELDS_TO_SCORE}
    for field in FIELDS_TO_SCORE:
        grades = [
            _grade_to_score(match.get("field_scores", {}).get(field, {}).get("grade", "WRONG"))
            for match in matches
            if field in match.get("field_scores", {})
        ]
        if grades:
            field_accuracy[field] = sum(grades) / len(grades)

    return {
        "doc": doc_name,
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gold),
        "field_accuracy": field_accuracy,
    }


def _aggregate_from_rows(per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = sum(int(row.get("tp", 0)) for row in per_doc)
    false_positive = sum(int(row.get("fp", 0)) for row in per_doc)
    false_negative = sum(int(row.get("fn", 0)) for row in per_doc)

    sums = {field: 0.0 for field in FIELDS_TO_SCORE}
    counts = {field: 0 for field in FIELDS_TO_SCORE}
    for row in per_doc:
        fa = row.get("field_accuracy", {})
        for field in FIELDS_TO_SCORE:
            value = fa.get(field)
            if value is not None:
                sums[field] += float(value)
                counts[field] += 1

    micro_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 1.0
    )
    micro_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )
    micro_f1 = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    hallucination_rate = (
        false_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    field_macro = {
        field: (sums[field] / counts[field] if counts[field] else None)
        for field in FIELDS_TO_SCORE
    }

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "hallucination_rate": hallucination_rate,
        "field_acc_macro": field_macro,
    }


def _write_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp_path.replace(checkpoint_path)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    return json.loads(checkpoint_path.read_text())


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pred_dir: Path,
    ref_dir: Path,
    judge_model: str,
    eval_prompt_version: str,
    doc_names: list[str],
) -> None:
    expected = {
        "pred_dir": str(pred_dir),
        "ref_dir": str(ref_dir),
        "judge_model_name": judge_model,
        "eval_prompt_version": eval_prompt_version,
        "doc_names": doc_names,
    }
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint mismatch for '{key}'. "
                f"expected={expected_value!r}, actual={actual_value!r}. "
                "Use --reset-checkpoint to restart."
            )


def _build_doc_names(
    pred_dir: Path,
    *,
    company_tickers: list[str],
    years: list[str],
) -> list[str]:
    discovered = _discover_pairs(pred_dir)
    ticker_filter = {ticker.upper() for ticker in company_tickers} if company_tickers else None
    year_filter = set(years) if years else None
    doc_names = [
        doc_name
        for ticker, year, doc_name in discovered
        if (ticker_filter is None or ticker in ticker_filter)
        and (year_filter is None or year in year_filter)
    ]
    if not doc_names:
        raise ValueError("No prediction files matched the requested company/year filters")
    return doc_names


def _run(args: argparse.Namespace) -> int:
    maybe_load_env_file(Path(args.env_file) if args.env_file else None)

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    out_path = Path(args.out)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(f"{args.out}.checkpoint.json")

    eval_prompt = load_eval_prompt(args.eval_prompt_version)
    company_tickers = _parse_csv(args.company_tickers)
    years = _parse_csv(args.years)
    doc_names = _build_doc_names(pred_dir, company_tickers=company_tickers, years=years)
    test_cases = build_test_cases_from_doc_names(pred_dir, ref_dir, doc_names=doc_names)

    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None:
        checkpoint = {
            "version": 1,
            "pred_dir": str(pred_dir),
            "ref_dir": str(ref_dir),
            "judge_model_name": args.judge_model,
            "eval_prompt_version": args.eval_prompt_version,
            "doc_names": doc_names,
            "completed": {},
            "updated_at_utc": _now_utc(),
        }
        _write_checkpoint(checkpoint_path, checkpoint)
    else:
        _validate_checkpoint(
            checkpoint,
            pred_dir=pred_dir,
            ref_dir=ref_dir,
            judge_model=args.judge_model,
            eval_prompt_version=args.eval_prompt_version,
            doc_names=doc_names,
        )

    completed: dict[str, dict[str, Any]] = checkpoint.get("completed", {})
    total_docs = len(test_cases)

    for idx, test_case in enumerate(test_cases, start=1):
        doc_name = test_case["doc_name"]
        if doc_name in completed:
            if not args.quiet:
                print(f"[resume] {idx}/{total_docs} {doc_name}")
            continue

        gold = json.loads(test_case["reference"])
        pred = json.loads(test_case["response"])
        judged = judge_align_and_score_openai(gold, pred, eval_prompt, args.judge_model)
        completed[doc_name] = _build_doc_row(doc_name, judged)

        checkpoint["completed"] = completed
        checkpoint["updated_at_utc"] = _now_utc()
        checkpoint["completed_count"] = len(completed)
        checkpoint["last_completed_doc"] = doc_name
        _write_checkpoint(checkpoint_path, checkpoint)

        if not args.quiet:
            print(f"[evaluate] {len(completed)}/{total_docs} {doc_name}")

    per_doc = [completed[doc_name] for doc_name in doc_names]
    report = {
        "aggregate": _aggregate_from_rows(per_doc),
        "per_doc": per_doc,
    }
    write_json(out_path, report)

    if not args.quiet:
        print(f"wrote {out_path}")
        print(f"checkpoint={checkpoint_path}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_resumable",
        description="Evaluate predictions with per-document checkpointing and resume support.",
    )
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--eval-prompt-version", default="v001")
    parser.add_argument("--company-tickers", default=None)
    parser.add_argument("--years", default=None)
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
