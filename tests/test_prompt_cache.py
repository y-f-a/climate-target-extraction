from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from cte.config import RunConfig, load_run_config
from cte.pipelines.rag.v1 import _build_query_engine, _collect_retrieved_chunks
from cte.prompt_cache import (
    build_rag_extract_prompt_cache_key,
    build_rag_extract_prompt_cache_options,
    manifest_prompt_cache_fields,
)


def test_run_config_prompt_cache_defaults() -> None:
    config = RunConfig(pipeline="rag", pipeline_version="rag.v1")
    assert config.openai_prompt_cache_enabled is False
    assert config.openai_prompt_cache_retention == "in-memory"
    assert config.target_postprocess_profile == "off"
    assert config.retrieval_rerank_profile == "off"


def test_run_config_rejects_invalid_prompt_cache_retention() -> None:
    with pytest.raises(ValidationError):
        RunConfig(
            pipeline="rag",
            pipeline_version="rag.v1",
            openai_prompt_cache_retention="7d",  # type: ignore[arg-type]
        )


def test_load_run_config_reads_prompt_cache_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-cache.toml"
    config_path.write_text(
        """
[run]
pipeline = "rag"
pipeline_version = "rag.v1"
openai_prompt_cache_enabled = true
openai_prompt_cache_retention = "24h"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    assert config.openai_prompt_cache_enabled is True
    assert config.openai_prompt_cache_retention == "24h"


def test_load_run_config_reads_target_postprocess_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "target-postprocess.toml"
    config_path.write_text(
        """
[run]
pipeline = "rag"
pipeline_version = "rag.v1"
target_postprocess_profile = "fp_dedupe_conservative_v1"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    assert config.target_postprocess_profile == "fp_dedupe_conservative_v1"


def test_load_run_config_reads_retrieval_rerank_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieval-rerank.toml"
    config_path.write_text(
        """
[run]
pipeline = "rag"
pipeline_version = "rag.v1"
retrieval_rerank_profile = "sentence_transformer_default_v1"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    assert config.retrieval_rerank_profile == "sentence_transformer_default_v1"


def test_run_config_rejects_invalid_retrieval_rerank_profile() -> None:
    with pytest.raises(ValidationError):
        RunConfig(
            pipeline="rag",
            pipeline_version="rag.v1",
            retrieval_rerank_profile="invalid",  # type: ignore[arg-type]
        )


def test_build_prompt_cache_key_is_stable_and_prompt_dependent() -> None:
    config = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
        openai_prompt_cache_enabled=True,
    )
    key_a1 = build_rag_extract_prompt_cache_key(
        config=config,
        extract_system_prompt="extract prompt a",
    )
    key_a2 = build_rag_extract_prompt_cache_key(
        config=config,
        extract_system_prompt="extract prompt a",
    )
    key_b = build_rag_extract_prompt_cache_key(
        config=config,
        extract_system_prompt="extract prompt b",
    )

    assert key_a1 is not None
    assert key_a1 == key_a2
    assert key_a1 != key_b


def test_build_prompt_cache_options_returns_empty_when_disabled() -> None:
    config = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
        openai_prompt_cache_enabled=False,
    )
    options = build_rag_extract_prompt_cache_options(
        config=config,
        extract_system_prompt="extract prompt",
    )
    assert options == {}


def test_manifest_prompt_cache_fields_are_effective_only_for_rag() -> None:
    rag_config = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
        openai_prompt_cache_enabled=True,
        openai_prompt_cache_retention="in-memory",
    )
    rag_fields = manifest_prompt_cache_fields(rag_config)
    assert rag_fields["prompt_cache_enabled"] is True
    assert rag_fields["prompt_cache_retention"] == "in-memory"
    assert rag_fields["prompt_cache_scope"] == "rag_extract_only"

    no_rag_config = RunConfig(
        pipeline="no_rag",
        pipeline_version="no_rag.v1",
        openai_prompt_cache_enabled=True,
        openai_prompt_cache_retention="24h",
    )
    no_rag_fields = manifest_prompt_cache_fields(no_rag_config)
    assert no_rag_fields["prompt_cache_enabled"] is False
    assert no_rag_fields["prompt_cache_retention"] is None
    assert no_rag_fields["prompt_cache_scope"] is None


class _FakeOpenAI:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        _FakeOpenAI.last_kwargs = kwargs

    def as_structured_llm(self, _schema: Any) -> str:
        return "structured-llm"


class _FakeBM25Retriever:
    @classmethod
    def from_defaults(cls, *, nodes: list[Any], similarity_top_k: int) -> dict[str, Any]:
        return {"nodes": nodes, "k": similarity_top_k}


class _FakeQueryFusionRetriever:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSentenceTransformerRerank:
    raise_import_error = False
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        if _FakeSentenceTransformerRerank.raise_import_error:
            raise ImportError("missing sentence-transformers")
        _FakeSentenceTransformerRerank.last_kwargs = kwargs
        self.kwargs = kwargs


class _FakeRetrieverQueryEngine:
    def __init__(
        self,
        *,
        retriever: Any,
        response_synthesizer: Any,
        node_postprocessors: list[Any] | None = None,
    ) -> None:
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors


class _FakeVectorIndex:
    def as_retriever(self, *, similarity_top_k: int) -> dict[str, Any]:
        return {"k": similarity_top_k}


def _deps() -> dict[str, Any]:
    return {
        "OpenAI": _FakeOpenAI,
        "BM25Retriever": _FakeBM25Retriever,
        "QueryFusionRetriever": _FakeQueryFusionRetriever,
        "SentenceTransformerRerank": _FakeSentenceTransformerRerank,
        "RetrieverQueryEngine": _FakeRetrieverQueryEngine,
        "get_response_synthesizer": lambda *, llm, response_mode: {
            "llm": llm,
            "response_mode": response_mode,
        },
    }


def test_build_query_engine_passes_prompt_cache_options() -> None:
    _FakeOpenAI.last_kwargs = None
    _FakeSentenceTransformerRerank.raise_import_error = False
    _FakeSentenceTransformerRerank.last_kwargs = None
    engine = _build_query_engine(
        _deps(),
        vector_index=_FakeVectorIndex(),
        nodes=[],
        model_name="gpt-5.2-2025-12-11",
        system_prompt="extract prompt",
        temperature=0.0,
        max_tokens=16390,
        similarity_top_k=10,
        fusion_top_k=5,
        retrieval_rerank_profile="off",
        prompt_cache_options={
            "prompt_cache_key": "cte:rag_extract:test",
            "prompt_cache_retention": "in-memory",
        },
    )

    assert isinstance(engine, _FakeRetrieverQueryEngine)
    assert _FakeOpenAI.last_kwargs is not None
    assert _FakeOpenAI.last_kwargs["additional_kwargs"] == {
        "prompt_cache_key": "cte:rag_extract:test",
        "prompt_cache_retention": "in-memory",
    }
    assert engine.node_postprocessors is None


def test_build_query_engine_omits_prompt_cache_options_when_disabled() -> None:
    _FakeOpenAI.last_kwargs = None
    _FakeSentenceTransformerRerank.raise_import_error = False
    _FakeSentenceTransformerRerank.last_kwargs = None
    engine = _build_query_engine(
        _deps(),
        vector_index=_FakeVectorIndex(),
        nodes=[],
        model_name="gpt-5.2-2025-12-11",
        system_prompt="extract prompt",
        temperature=0.0,
        max_tokens=16390,
        similarity_top_k=10,
        fusion_top_k=5,
        retrieval_rerank_profile="off",
        prompt_cache_options=None,
    )

    assert isinstance(engine, _FakeRetrieverQueryEngine)
    assert _FakeOpenAI.last_kwargs is not None
    assert "additional_kwargs" not in _FakeOpenAI.last_kwargs
    assert engine.node_postprocessors is None


def test_build_query_engine_enables_sentence_transformer_rerank() -> None:
    _FakeOpenAI.last_kwargs = None
    _FakeSentenceTransformerRerank.raise_import_error = False
    _FakeSentenceTransformerRerank.last_kwargs = None

    engine = _build_query_engine(
        _deps(),
        vector_index=_FakeVectorIndex(),
        nodes=[],
        model_name="gpt-5.2-2025-12-11",
        system_prompt="extract prompt",
        temperature=0.0,
        max_tokens=16390,
        similarity_top_k=10,
        fusion_top_k=5,
        retrieval_rerank_profile="sentence_transformer_default_v1",
        prompt_cache_options=None,
    )

    assert isinstance(engine, _FakeRetrieverQueryEngine)
    assert isinstance(engine.retriever, _FakeQueryFusionRetriever)
    assert engine.retriever.kwargs["similarity_top_k"] == 20
    assert isinstance(engine.node_postprocessors, list)
    assert len(engine.node_postprocessors) == 1
    assert _FakeSentenceTransformerRerank.last_kwargs == {
        "top_n": 5,
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "device": "cpu",
        "keep_retrieval_score": True,
        "trust_remote_code": False,
    }


def test_build_query_engine_rerank_missing_dependencies_raises_runtime_error() -> None:
    _FakeSentenceTransformerRerank.raise_import_error = True
    _FakeSentenceTransformerRerank.last_kwargs = None

    with pytest.raises(RuntimeError, match="Reranking requires sentence-transformer dependencies"):
        _build_query_engine(
            _deps(),
            vector_index=_FakeVectorIndex(),
            nodes=[],
            model_name="gpt-5.2-2025-12-11",
            system_prompt="extract prompt",
            temperature=0.0,
            max_tokens=16390,
            similarity_top_k=10,
            fusion_top_k=5,
            retrieval_rerank_profile="sentence_transformer_default_v1",
            prompt_cache_options=None,
        )


def test_collect_retrieved_chunks_extracts_rank_score_and_text() -> None:
    class _Node:
        def __init__(self, text: str, metadata: dict[str, Any]) -> None:
            self.node_id = "node-123"
            self.id_ = "node-id-123"
            self.metadata = metadata
            self._text = text

        def get_content(self, *, metadata_mode: str | None = None) -> str:
            assert metadata_mode == "none"
            return self._text

    response = SimpleNamespace(
        source_nodes=[
            SimpleNamespace(
                score=0.98,
                node=_Node(
                    "chunk body",
                    {"file_name": "report.pdf", "page": 3, "tags": {"a", "b"}},
                ),
            )
        ]
    )

    rows = _collect_retrieved_chunks(response)
    assert len(rows) == 1
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] == 0.98
    assert rows[0]["node_id"] == "node-123"
    assert rows[0]["id_"] == "node-id-123"
    assert rows[0]["metadata"]["file_name"] == "report.pdf"
    assert rows[0]["metadata"]["page"] == 3
    assert sorted(rows[0]["metadata"]["tags"]) == ["a", "b"]
    assert rows[0]["text"] == "chunk body"
    assert rows[0]["text_length"] == len("chunk body")
    assert rows[0]["text_sha256"] == hashlib.sha256("chunk body".encode("utf-8")).hexdigest()


def test_collect_retrieved_chunks_handles_missing_or_non_list_source_nodes() -> None:
    assert _collect_retrieved_chunks(SimpleNamespace(source_nodes=None)) == []
    assert _collect_retrieved_chunks(SimpleNamespace(source_nodes="invalid")) == []
