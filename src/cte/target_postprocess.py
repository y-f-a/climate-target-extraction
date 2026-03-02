from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal

TargetPostprocessProfile = Literal["off", "fp_dedupe_conservative_v1"]

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_QUANTIZE = Decimal("0.000001")
_NEAR_DUPLICATE_FIELDS: tuple[str, ...] = (
    "target_type",
    "horizon",
    "metric_type",
    "scopes_covered",
    "scope3_categories",
    "ambition",
    "status",
    "base_year",
    "target_year",
    "reduction_pct",
    "target_value",
    "unit",
)


def apply_target_postprocess(
    payload: dict[str, Any],
    profile: TargetPostprocessProfile | str,
) -> tuple[dict[str, Any], dict[str, int]]:
    targets = payload.get("targets")
    target_count = len(targets) if isinstance(targets, list) else 0
    noop_summary = {
        "input_targets": target_count,
        "output_targets": target_count,
        "removed_targets": 0,
        "merged_groups": 0,
    }
    if profile == "off":
        return payload, noop_summary
    if profile != "fp_dedupe_conservative_v1":
        raise ValueError(f"Unsupported target_postprocess_profile: {profile}")
    if not isinstance(targets, list):
        return payload, noop_summary

    exact_output: list[dict[str, Any]] = []
    exact_seen: dict[str, int] = {}
    exact_group_sizes: dict[str, int] = {}

    for item in targets:
        target = _as_target_dict(item)
        exact_key = _stable_hash(_normalized_target_payload(target))
        if exact_key not in exact_seen:
            exact_seen[exact_key] = len(exact_output)
            exact_output.append(target)
            exact_group_sizes[exact_key] = 1
            continue

        keep_idx = exact_seen[exact_key]
        exact_group_sizes[exact_key] = exact_group_sizes[exact_key] + 1
        exact_output[keep_idx]["sources"] = _merge_sources(
            exact_output[keep_idx].get("sources"),
            target.get("sources"),
        )

    near_group_order: list[tuple[Any, ...]] = []
    near_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    near_group_sizes: dict[tuple[Any, ...], int] = {}
    for target in exact_output:
        group_key = _near_duplicate_key(target)
        if group_key not in near_groups:
            near_group_order.append(group_key)
            near_groups[group_key] = {
                "target": deepcopy(target),
                "score": _completeness_score(target),
                "richness": _metadata_richness(target),
                "hash": _stable_hash(_normalized_target_payload(target)),
            }
            near_group_sizes[group_key] = 1
            continue

        near_group_sizes[group_key] = near_group_sizes[group_key] + 1
        existing = near_groups[group_key]
        merged_sources = _merge_sources(
            existing["target"].get("sources"),
            target.get("sources"),
        )
        candidate_score = _completeness_score(target)
        candidate_richness = _metadata_richness(target)
        candidate_hash = _stable_hash(_normalized_target_payload(target))
        existing_score = int(existing["score"])
        existing_richness = tuple(existing["richness"])
        existing_hash = str(existing["hash"])
        if _is_better_target(
            candidate_score=candidate_score,
            existing_score=existing_score,
            candidate_richness=candidate_richness,
            existing_richness=existing_richness,
            candidate_hash=candidate_hash,
            existing_hash=existing_hash,
        ):
            replacement = deepcopy(target)
            replacement["sources"] = merged_sources
            near_groups[group_key] = {
                "target": replacement,
                "score": candidate_score,
                "richness": candidate_richness,
                "hash": candidate_hash,
            }
        else:
            existing["target"]["sources"] = merged_sources

    final_targets = [near_groups[key]["target"] for key in near_group_order]
    exact_merged_groups = sum(1 for size in exact_group_sizes.values() if size > 1)
    near_merged_groups = sum(1 for size in near_group_sizes.values() if size > 1)
    summary = {
        "input_targets": target_count,
        "output_targets": len(final_targets),
        "removed_targets": target_count - len(final_targets),
        "merged_groups": exact_merged_groups + near_merged_groups,
    }
    out_payload = dict(payload)
    out_payload["targets"] = final_targets
    return out_payload, summary


def _as_target_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {"notes": str(value)}


def _normalized_target_payload(target: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(target)
    normalized["scopes_covered"] = _normalize_scopes(target.get("scopes_covered"))
    normalized["scope3_categories"] = _normalize_scope3_categories(target.get("scope3_categories"))
    normalized["unit"] = _normalize_unit(target.get("unit"))
    normalized["reduction_pct"] = _normalize_number(target.get("reduction_pct"))
    normalized["target_value"] = _normalize_number(target.get("target_value"))
    normalized["target_type"] = _normalize_text(target.get("target_type"))
    normalized["horizon"] = _normalize_text(target.get("horizon"))
    normalized["metric_type"] = _normalize_text(target.get("metric_type"))
    normalized["ambition"] = _normalize_text(target.get("ambition"))
    normalized["status"] = _normalize_text(target.get("status"))
    normalized["base_year"] = _normalize_int(target.get("base_year"))
    normalized["target_year"] = _normalize_int(target.get("target_year"))
    normalized["sources"] = _normalize_string_list(target.get("sources"))
    normalized["title"] = _normalize_text(target.get("title"), collapse=True)
    normalized["notes"] = _normalize_text(target.get("notes"), collapse=True)
    return normalized


def _near_duplicate_key(target: dict[str, Any]) -> tuple[Any, ...]:
    normalized = _normalized_target_payload(target)
    key_parts: list[Any] = []
    for field in _NEAR_DUPLICATE_FIELDS:
        value = normalized.get(field)
        if isinstance(value, list):
            key_parts.append(tuple(value))
        else:
            key_parts.append(value)
    return tuple(key_parts)


def _completeness_score(target: dict[str, Any]) -> int:
    normalized = _normalized_target_payload(target)
    score = 0
    for field in _NEAR_DUPLICATE_FIELDS:
        value = normalized.get(field)
        if _has_value(value):
            score += 1
    if normalized.get("scopes_covered"):
        score += 1
    return score


def _metadata_richness(target: dict[str, Any]) -> tuple[int, int, int]:
    sources = _normalize_string_list(target.get("sources"))
    notes = _normalize_text(target.get("notes"), collapse=True)
    title = _normalize_text(target.get("title"), collapse=True)
    return (len(sources), 1 if notes else 0, 1 if title else 0)


def _is_better_target(
    *,
    candidate_score: int,
    existing_score: int,
    candidate_richness: tuple[int, int, int],
    existing_richness: tuple[int, int, int],
    candidate_hash: str,
    existing_hash: str,
) -> bool:
    if candidate_score != existing_score:
        return candidate_score > existing_score
    if candidate_richness != existing_richness:
        return candidate_richness > existing_richness
    return candidate_hash < existing_hash


def _normalize_scopes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    scopes: set[str] = set()
    for item in value:
        text = str(item).strip().upper()
        if text:
            scopes.add(text)
    return sorted(scopes)


def _normalize_scope3_categories(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    categories: set[int] = set()
    for item in value:
        normalized = _normalize_int(item)
        if normalized is not None:
            categories.add(normalized)
    return sorted(categories)


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    values: set[str] = set()
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            values.add(text)
    return sorted(values)


def _normalize_number(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    quantized = numeric.quantize(_NUMBER_QUANTIZE, rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text


def _normalize_unit(value: Any) -> str | None:
    return _normalize_text(value, lower=True, collapse=True)


def _normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(
    value: Any,
    *,
    lower: bool = False,
    collapse: bool = False,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if collapse:
        text = _WHITESPACE_RE.sub(" ", text)
    if lower:
        text = text.lower()
    return text


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merge_sources(left: Any, right: Any) -> list[str]:
    return sorted(set(_normalize_string_list(left)) | set(_normalize_string_list(right)))
