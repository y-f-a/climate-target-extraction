import pytest

from cte.config import RunConfig, validate_lineage_requirements


def test_parity_lineage_requires_no_parent() -> None:
    cfg = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
    )
    validate_lineage_requirements(cfg)


def test_non_parity_requires_parent_and_changes() -> None:
    cfg = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v2",
        parent_run_id=None,
        changed_components=[],
    )
    with pytest.raises(ValueError):
        validate_lineage_requirements(cfg)


def test_non_parity_with_parent_and_changes_passes() -> None:
    cfg = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v2",
        parent_run_id="run_123",
        changed_components=["retrieval"],
    )
    validate_lineage_requirements(cfg)
