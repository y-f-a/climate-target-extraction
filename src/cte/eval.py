from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .io import read_json
from .schemas import FIELDS_TO_SCORE, ScoreOutput

USER_TEMPLATE = """
Gold JSON for one document:
{gold}

Predicted JSON for the same document:
{pred}

Remember:
- Ignore targets where target_type == "non_target_claim" on BOTH sides.
- Only score these fields: {fields}.
- Return STRICT JSON as specified.
""".strip()


JudgeFn = Callable[[dict[str, Any], dict[str, Any], str, str], dict[str, Any]]
ProgressFn = Callable[[str, int, int], None]


def _project(obj: dict[str, Any]) -> dict[str, Any]:
    keep = set(FIELDS_TO_SCORE + ["target_type"])
    return {
        "company": obj.get("company"),
        "targets": [
            {k: target.get(k) for k in keep if k in target}
            for target in obj.get("targets", [])
            if target.get("target_type") != "non_target_claim"
        ],
    }


def _grade_to_score(value: str) -> float:
    if value == "EXACT":
        return 1.0
    if value == "PARTIAL":
        return 0.5
    return 0.0


def judge_align_and_score_openai(
    gold: dict[str, Any],
    pred: dict[str, Any],
    eval_system_prompt: str,
    model_name: str,
) -> dict[str, Any]:
    from openai import OpenAI as OpenAIClient

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = USER_TEMPLATE.format(
        gold=json.dumps(_project(gold), ensure_ascii=False),
        pred=json.dumps(_project(pred), ensure_ascii=False),
        fields=", ".join(FIELDS_TO_SCORE),
    )

    response = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": eval_system_prompt},
            {"role": "user", "content": prompt},
        ],
        text_format=ScoreOutput,
        reasoning={"effort": "high"},
    )

    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return parsed.model_dump()

    text = (getattr(response, "output_text", None) or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])


def build_test_cases(
    pred_dir: Path,
    reference_dir: Path,
    *,
    company_tickers: list[str],
    years: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for ticker in company_tickers:
        for year in years:
            doc_name = f"{ticker.lower()}.{year}.targets.v1.json"
            pred_path = pred_dir / doc_name
            ref_path = reference_dir / doc_name
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction JSON: {pred_path}")
            if not ref_path.exists():
                raise FileNotFoundError(f"Missing reference JSON: {ref_path}")
            cases.append(
                {
                    "doc_name": doc_name,
                    "response": json.dumps(read_json(pred_path), ensure_ascii=False),
                    "reference": json.dumps(read_json(ref_path), ensure_ascii=False),
                }
            )
    return cases


def build_test_cases_from_doc_names(
    pred_dir: Path,
    reference_dir: Path,
    *,
    doc_names: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for doc_name in doc_names:
        pred_path = pred_dir / doc_name
        ref_path = reference_dir / doc_name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction JSON: {pred_path}")
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference JSON: {ref_path}")
        cases.append(
            {
                "doc_name": doc_name,
                "response": json.dumps(read_json(pred_path), ensure_ascii=False),
                "reference": json.dumps(read_json(ref_path), ensure_ascii=False),
            }
        )
    return cases


def evaluate_dataset(
    test_cases: list[dict[str, Any]],
    *,
    eval_system_prompt: str,
    judge_model_name: str,
    judge_fn: JudgeFn | None = None,
    progress_fn: ProgressFn | None = None,
) -> dict[str, Any]:
    judge = judge_fn or judge_align_and_score_openai

    per_doc = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    sums = {field: 0.0 for field in FIELDS_TO_SCORE}
    counts = {field: 0 for field in FIELDS_TO_SCORE}
    total_docs = len(test_cases)

    for idx, test_case in enumerate(test_cases, start=1):
        doc_name = test_case["doc_name"]
        gold = json.loads(test_case["reference"])
        pred = json.loads(test_case["response"])

        judged = judge(gold, pred, eval_system_prompt, judge_model_name)

        matches = judged.get("matches", [])
        unmatched_gold = judged.get("unmatched_gold", [])
        unmatched_pred = judged.get("unmatched_pred", [])

        true_positive += len(matches)
        false_positive += len(unmatched_pred)
        false_negative += len(unmatched_gold)

        field_accuracy = {field: None for field in FIELDS_TO_SCORE}
        for field in FIELDS_TO_SCORE:
            grades = [
                _grade_to_score(match.get("field_scores", {}).get(field, {}).get("grade", "WRONG"))
                for match in matches
                if field in match.get("field_scores", {})
            ]
            if grades:
                value = sum(grades) / len(grades)
                field_accuracy[field] = value
                sums[field] += value
                counts[field] += 1

        per_doc.append(
            {
                "doc": doc_name,
                "tp": len(matches),
                "fp": len(unmatched_pred),
                "fn": len(unmatched_gold),
                "field_accuracy": field_accuracy,
            }
        )
        if progress_fn is not None:
            progress_fn(doc_name, idx, total_docs)

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
        "aggregate": {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "hallucination_rate": hallucination_rate,
            "field_acc_macro": field_macro,
        },
        "per_doc": per_doc,
    }


def evaluate_from_dirs(
    pred_dir: Path,
    reference_dir: Path,
    *,
    company_tickers: list[str],
    years: list[str],
    eval_system_prompt: str,
    judge_model_name: str,
    judge_fn: JudgeFn | None = None,
    progress_fn: ProgressFn | None = None,
) -> dict[str, Any]:
    test_cases = build_test_cases(
        pred_dir,
        reference_dir,
        company_tickers=company_tickers,
        years=years,
    )
    return evaluate_dataset(
        test_cases,
        eval_system_prompt=eval_system_prompt,
        judge_model_name=judge_model_name,
        judge_fn=judge_fn,
        progress_fn=progress_fn,
    )


def evaluate_from_doc_names(
    pred_dir: Path,
    reference_dir: Path,
    *,
    doc_names: list[str],
    eval_system_prompt: str,
    judge_model_name: str,
    judge_fn: JudgeFn | None = None,
    progress_fn: ProgressFn | None = None,
) -> dict[str, Any]:
    test_cases = build_test_cases_from_doc_names(
        pred_dir,
        reference_dir,
        doc_names=doc_names,
    )
    return evaluate_dataset(
        test_cases,
        eval_system_prompt=eval_system_prompt,
        judge_model_name=judge_model_name,
        judge_fn=judge_fn,
        progress_fn=progress_fn,
    )
