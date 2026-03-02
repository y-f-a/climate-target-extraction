from __future__ import annotations

from copy import deepcopy

from cte.target_postprocess import apply_target_postprocess


def _base_target() -> dict[str, object]:
    return {
        "title": "Near-term reduction target",
        "target_type": "sbti_near_term",
        "horizon": "near_term",
        "metric_type": "absolute",
        "scopes_covered": ["S2", "S1"],
        "scope3_categories": [2, 1],
        "ambition": "1.5C",
        "base_year": 2020,
        "target_year": 2030,
        "reduction_pct": 42.0,
        "target_value": None,
        "unit": " tCO2e / revenue ",
        "status": "approved",
        "notes": "Board approved",
        "sources": ["page-3", "page-1"],
    }


def test_exact_duplicate_targets_collapse_to_one() -> None:
    target_a = _base_target()
    target_b = deepcopy(target_a)
    payload = {"company": "Example Corp", "targets": [target_a, target_b]}

    processed, summary = apply_target_postprocess(payload, "fp_dedupe_conservative_v1")

    assert len(processed["targets"]) == 1
    assert summary == {
        "input_targets": 2,
        "output_targets": 1,
        "removed_targets": 1,
        "merged_groups": 1,
    }


def test_near_duplicate_merges_sources_and_collapses() -> None:
    richer = _base_target()
    richer["sources"] = ["page-1", "page-3"]
    richer["notes"] = "Board approved by committee"

    sparse = _base_target()
    sparse["notes"] = None
    sparse["sources"] = ["page-2"]
    sparse["title"] = None
    sparse["unit"] = "tCO2e / revenue"

    payload = {"company": "Example Corp", "targets": [sparse, richer]}

    processed, summary = apply_target_postprocess(payload, "fp_dedupe_conservative_v1")

    assert len(processed["targets"]) == 1
    kept = processed["targets"][0]
    assert kept["notes"] == "Board approved by committee"
    assert kept["sources"] == ["page-1", "page-2", "page-3"]
    assert summary["removed_targets"] == 1
    assert summary["merged_groups"] == 1


def test_distinct_targets_are_preserved() -> None:
    target_a = _base_target()
    target_b = _base_target()
    target_b["target_year"] = 2035
    target_b["reduction_pct"] = 60.0
    payload = {"company": "Example Corp", "targets": [target_a, target_b]}

    processed, summary = apply_target_postprocess(payload, "fp_dedupe_conservative_v1")

    assert len(processed["targets"]) == 2
    assert summary["removed_targets"] == 0
    assert summary["merged_groups"] == 0


def test_postprocess_is_deterministic() -> None:
    target_a = _base_target()
    target_b = _base_target()
    target_b["notes"] = "alternative note"
    target_b["sources"] = ["page-9"]
    payload = {"company": "Example Corp", "targets": [target_a, target_b]}

    processed_a, summary_a = apply_target_postprocess(payload, "fp_dedupe_conservative_v1")
    processed_b, summary_b = apply_target_postprocess(payload, "fp_dedupe_conservative_v1")

    assert processed_a == processed_b
    assert summary_a == summary_b


def test_profile_off_is_noop() -> None:
    payload = {"company": "Example Corp", "targets": [_base_target()]}

    processed, summary = apply_target_postprocess(payload, "off")

    assert processed is payload
    assert summary == {
        "input_targets": 1,
        "output_targets": 1,
        "removed_targets": 0,
        "merged_groups": 0,
    }
