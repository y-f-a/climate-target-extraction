from pathlib import Path

from cte.io import legacy_report_name, target_doc_name
from cte.schemas import ExtractedTargets


def test_target_doc_name_contract() -> None:
    assert target_doc_name("AAPL", "2024") == "aapl.2024.targets.v1.json"


def test_legacy_report_name_contract() -> None:
    assert legacy_report_name("rag", "gpt5_2", 1) == "v2_gpt5_2_1.json"
    assert legacy_report_name("no_rag", "gpt5_2", 1) == "full_report_gpt5_2_1.json"


def test_reference_json_parses_with_schema() -> None:
    sample = Path("data/evaluation_set/reference_targets/aapl.2024.targets.v1.json")
    payload = sample.read_text(encoding="utf-8")
    parsed = ExtractedTargets.model_validate_json(payload)
    assert parsed.targets
