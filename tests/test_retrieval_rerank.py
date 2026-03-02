from __future__ import annotations

import pytest

from cte.retrieval_rerank import resolve_retrieval_rerank_spec


def test_resolve_retrieval_rerank_off_returns_none() -> None:
    assert resolve_retrieval_rerank_spec(profile="off", fusion_top_k=5) is None


def test_resolve_retrieval_rerank_sentence_transformer_defaults() -> None:
    spec = resolve_retrieval_rerank_spec(
        profile="sentence_transformer_default_v1",
        fusion_top_k=5,
    )
    assert spec is not None
    assert spec.profile == "sentence_transformer_default_v1"
    assert spec.model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert spec.device == "cpu"
    assert spec.candidate_top_k == 20
    assert spec.top_n == 5


def test_resolve_retrieval_rerank_uses_fusion_top_k_when_larger() -> None:
    spec = resolve_retrieval_rerank_spec(
        profile="sentence_transformer_default_v1",
        fusion_top_k=24,
    )
    assert spec is not None
    assert spec.candidate_top_k == 24
    assert spec.top_n == 24


def test_resolve_retrieval_rerank_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval_rerank_profile"):
        resolve_retrieval_rerank_spec(profile="unknown_profile", fusion_top_k=5)
