from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ...config import RunConfig
from ...index_registry import (
    IndexRegistry,
    build_source_manifest,
    compute_index_fingerprint,
)
from ...io import target_doc_name, write_json
from ...parse_cache import (
    CLEANUP_DEFAULT_PREVIEW_CHARS,
    ScopedPdf,
    _cleanup_page_cache_root,
    _cleanup_page_rows_with_llm,
    build_cache_content_fingerprint,
    load_cached_pages_for_source_manifest,
    parse_settings_from_component_settings,
)
from ...prompt_cache import build_rag_extract_prompt_cache_options
from ...retrieval_rerank import resolve_retrieval_rerank_spec
from ...schemas import ExtractedTargets
from ...target_postprocess import apply_target_postprocess

QUERY_PROMPT = (
    "Extract climate emission target data that follows SBTi-style definitions "
    "(near-term, long-term/net-zero; absolute or intensity; Scopes 1/2/3), "
    "These could be in text form, covering any of the areas of Scope 1, 2 or 3 "
    "from electricity to value chain emissions. "
    "STRICTLY from the provided context. If a field isn’t stated, leave it null."
)


def _import_rag_dependencies() -> dict[str, Any]:
    try:
        import pymupdf4llm
        from llama_index.core import (
            Settings,
            SimpleDirectoryReader,
            VectorStoreIndex,
            load_index_from_storage,
        )
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.response_synthesizers import get_response_synthesizer
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.schema import Document
        from llama_index.core.storage import StorageContext
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as exc:
        raise RuntimeError(
            "Missing RAG dependencies. Install with: `uv sync --extra rag`"
        ) from exc

    return {
        "pymupdf4llm": pymupdf4llm,
        "Settings": Settings,
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "VectorStoreIndex": VectorStoreIndex,
        "load_index_from_storage": load_index_from_storage,
        "IngestionPipeline": IngestionPipeline,
        "SentenceSplitter": SentenceSplitter,
        "SentenceTransformerRerank": SentenceTransformerRerank,
        "RetrieverQueryEngine": RetrieverQueryEngine,
        "get_response_synthesizer": get_response_synthesizer,
        "QueryFusionRetriever": QueryFusionRetriever,
        "Document": Document,
        "StorageContext": StorageContext,
        "OpenAIEmbedding": OpenAIEmbedding,
        "OpenAI": OpenAI,
        "BM25Retriever": BM25Retriever,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _extract_node_text(node: Any) -> str:
    if node is None:
        return ""
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            return str(get_content(metadata_mode="none"))
        except TypeError:
            try:
                return str(get_content())
            except Exception:
                pass
        except Exception:
            pass
    text = getattr(node, "text", None)
    if text is None:
        return ""
    return str(text)


def _collect_retrieved_chunks(response: Any) -> list[dict[str, Any]]:
    source_nodes = getattr(response, "source_nodes", None)
    if not isinstance(source_nodes, list):
        return []

    rows: list[dict[str, Any]] = []
    for rank, source_node in enumerate(source_nodes, start=1):
        node = getattr(source_node, "node", None)
        text = _extract_node_text(node)
        metadata = getattr(node, "metadata", {}) if node is not None else {}
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "rank": rank,
                "score": getattr(source_node, "score", None),
                "node_id": getattr(node, "node_id", None) if node is not None else None,
                "id_": getattr(node, "id_", None) if node is not None else None,
                "metadata": _json_safe_value(metadata),
                "text": text,
                "text_sha256": _sha256_text(text),
                "text_length": len(text),
            }
        )
    return rows


def _local_cleanup_cache_root(parsed_docs_root: Path) -> Path:
    return parsed_docs_root / "pymupdf_cleanup"


def _local_cleanup_profile_payload(parse_settings: Any) -> dict[str, Any]:
    return {
        "cleanup_mode": parse_settings.cleanup_mode,
        "cleanup_model": parse_settings.cleanup_model,
        "cleanup_doc_scope": parse_settings.cleanup_doc_scope,
        "cleanup_enabled_doc_pairs": list(parse_settings.cleanup_enabled_doc_pairs),
        "cleanup_max_pages_per_pdf": parse_settings.cleanup_max_pages_per_pdf,
        "cleanup_score_threshold": parse_settings.cleanup_score_threshold,
        "cleanup_use_score_threshold": parse_settings.cleanup_use_score_threshold,
        "cleanup_extra_keywords": list(parse_settings.cleanup_extra_keywords),
        "cleanup_low_text_rescue": parse_settings.cleanup_low_text_rescue,
        "cleanup_low_text_max_chars": parse_settings.cleanup_low_text_max_chars,
        "cleanup_image_dpi": parse_settings.cleanup_image_dpi,
        "cleanup_timeout_sec": parse_settings.cleanup_timeout_sec,
        "cleanup_length_ratio_max": parse_settings.cleanup_length_ratio_max,
        "cleanup_numeric_guardrail_enabled": parse_settings.cleanup_numeric_guardrail_enabled,
        "cleanup_page_cache_miss_policy": parse_settings.cleanup_page_cache_miss_policy,
        "cleanup_profile_version": parse_settings.cleanup_profile_version,
    }


def _local_cleanup_profile_key(parse_settings: Any) -> tuple[str, str, dict[str, Any]]:
    payload = _local_cleanup_profile_payload(parse_settings)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = _sha256_text(serialized)
    token = f"{parse_settings.cleanup_mode}-{parse_settings.cleanup_profile_version}-{digest[:12]}"
    key = re.sub(r"[^a-z0-9._-]+", "-", token.lower()).strip("-")
    return key, digest, payload


def _local_cleanup_entry_dir(
    *,
    parsed_docs_root: Path,
    cleanup_profile_key: str,
    source_sha256: str,
) -> Path:
    return _local_cleanup_cache_root(parsed_docs_root) / cleanup_profile_key / source_sha256


def _extract_pdf_page_rows(
    *,
    pdf_reader: Any,
    absolute_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    raw_docs = pdf_reader.load_data(str(absolute_path))
    page_map: dict[int, list[str]] = {}
    total_pages = 0
    for doc in raw_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        try:
            page = int(metadata.get("page", 0) or 0)
        except (TypeError, ValueError):
            continue
        if page <= 0:
            continue
        total_pages = max(total_pages, page)
        try:
            page_total = int(metadata.get("total_pages", 0) or 0)
        except (TypeError, ValueError):
            page_total = 0
        if page_total > 0:
            total_pages = max(total_pages, page_total)
        text = str(getattr(doc, "text", "") or "")
        if text.strip():
            page_map.setdefault(page, []).append(text)

    if total_pages <= 0:
        try:
            import fitz

            with fitz.open(absolute_path) as handle:
                total_pages = int(handle.page_count)
        except Exception:
            total_pages = 0
    if total_pages <= 0:
        return [], 0

    page_rows: list[dict[str, Any]] = []
    for page in range(1, total_pages + 1):
        chunks = [chunk.strip() for chunk in page_map.get(page, []) if str(chunk).strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk in seen:
                continue
            seen.add(chunk)
            deduped.append(chunk)
        page_rows.append({"page": page, "text_markdown": "\n\n".join(deduped)})
    return page_rows, total_pages


def _load_local_cleanup_cache_entry(
    *,
    entry_dir: Path,
    cleanup_profile_key: str,
    cleanup_profile_sha256: str,
    source_relative_path: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    manifest_path = entry_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("cache_kind") != "pymupdf_cleanup":
        return None
    if manifest.get("schema_version") != "v1":
        return None
    if manifest.get("status") != "success":
        return None
    if str(manifest.get("cleanup_profile_key", "")) != cleanup_profile_key:
        return None
    if str(manifest.get("cleanup_profile_sha256", "")) != cleanup_profile_sha256:
        return None
    if str(manifest.get("source_sha256", "")) != source_sha256:
        return None
    if str(manifest.get("source_relative_path", "")) != source_relative_path:
        return None
    try:
        page_count = int(manifest.get("page_count", 0) or 0)
    except (TypeError, ValueError):
        return None
    if page_count <= 0:
        return None
    pages_dir = entry_dir / str(manifest.get("pages_dir", "pages"))
    if not pages_dir.exists():
        return None

    page_rows: list[dict[str, Any]] = []
    for page in range(1, page_count + 1):
        page_path = pages_dir / f"{page:04d}.md"
        if not page_path.exists():
            return None
        page_rows.append({"page": page, "text_markdown": page_path.read_text(encoding="utf-8")})
    return page_rows, manifest


def _write_local_cleanup_cache_entry(
    *,
    entry_dir: Path,
    cleanup_profile_key: str,
    cleanup_profile_sha256: str,
    cleanup_profile_payload: dict[str, Any],
    scoped_pdf: ScopedPdf,
    page_rows: list[dict[str, Any]],
    cleanup_artifacts: Any,
) -> dict[str, Any]:
    entry_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = entry_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for old_path in sorted(pages_dir.glob("*.md")):
        old_path.unlink()
    for row in page_rows:
        page = int(row.get("page", 0) or 0)
        if page <= 0:
            continue
        text = str(row.get("text_markdown", ""))
        (pages_dir / f"{page:04d}.md").write_text(text, encoding="utf-8")

    content_payload = [
        {
            "page": int(row.get("page", 0) or 0),
            "text_sha256": _sha256_text(str(row.get("text_markdown", ""))),
        }
        for row in page_rows
    ]
    content_sha256 = _sha256_text(json.dumps(content_payload, ensure_ascii=False, sort_keys=True))
    cleanup_summary = dict(cleanup_artifacts.summary or {})
    manifest = {
        "cache_kind": "pymupdf_cleanup",
        "schema_version": "v1",
        "status": "success",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_relative_path": scoped_pdf.source_relative_path,
        "source_file_name": scoped_pdf.source_file_name,
        "source_sha256": scoped_pdf.source_sha256,
        "source_size_bytes": scoped_pdf.source_size_bytes,
        "page_count": scoped_pdf.page_count,
        "cleanup_profile_key": cleanup_profile_key,
        "cleanup_profile_sha256": cleanup_profile_sha256,
        "cleanup_profile_payload": cleanup_profile_payload,
        "cleanup_mode": cleanup_summary.get("mode"),
        "cleanup_profile_version": cleanup_summary.get("profile_version"),
        "cleanup_model": cleanup_summary.get("model"),
        "cleanup_attempted_pages": int(cleanup_summary.get("attempted_pages", 0) or 0),
        "cleanup_accepted_pages": int(cleanup_summary.get("accepted_pages", 0) or 0),
        "cleanup_rejected_pages": int(cleanup_summary.get("rejected_pages", 0) or 0),
        "cleanup_failed_pages": int(cleanup_summary.get("failed_pages", 0) or 0),
        "cleanup_page_cache_hits": int(cleanup_summary.get("page_cache_hits", 0) or 0),
        "cleanup_page_cache_misses": int(cleanup_summary.get("page_cache_misses", 0) or 0),
        "cleanup_page_cache_write_errors": int(cleanup_summary.get("page_cache_write_errors", 0) or 0),
        "cleanup_enabled_for_doc": bool(cleanup_summary.get("enabled_for_doc", False)),
        "cleanup_selected_pages": [int(page) for page in cleanup_summary.get("selected_pages", [])],
        "cleanup_summary": cleanup_summary,
        "cleanup_audit_path": "cleanup_audit.jsonl",
        "pages_dir": "pages",
        "page_file_pattern": "{page:04d}.md",
        "content_sha256": content_sha256,
    }
    write_json(entry_dir / "manifest.json", manifest)
    _write_jsonl(entry_dir / "cleanup_audit.jsonl", cleanup_artifacts.audit_rows)
    return manifest


def _build_documents_from_page_rows(
    *,
    deps: dict[str, Any],
    page_rows: list[dict[str, Any]],
    source_relative_path: str,
    source_file_name: str,
    total_pages: int,
    absolute_path: Path,
) -> list[Any]:
    docs: list[Any] = []
    for row in sorted(page_rows, key=lambda item: int(item.get("page", 0) or 0)):
        page = int(row.get("page", 0) or 0)
        if page <= 0:
            continue
        text_markdown = str(row.get("text_markdown", ""))
        docs.append(
            deps["Document"](
                text=text_markdown,
                metadata={
                    "file_name": source_file_name,
                    "file_path": str(absolute_path),
                    "source_relative_path": source_relative_path,
                    "page": page,
                    "total_pages": total_pages,
                },
            )
        )
    return docs


def _load_or_build_local_parser_docs_with_cleanup_cache(
    *,
    deps: dict[str, Any],
    parse_settings: Any,
    source_dir: Path,
    source_manifest: list[Any],
    parsed_docs_root: Path,
    ticker: str,
    year: str,
) -> tuple[list[Any], dict[str, Any]]:
    details: dict[str, Any] = {
        "mode": parse_settings.cleanup_mode,
        "model": parse_settings.cleanup_model if parse_settings.cleanup_mode != "off" else None,
        "attempted_files": 0,
        "attempted_pages": 0,
        "accepted_pages": 0,
        "rejected_pages": 0,
        "failed_pages": 0,
        "page_cache_hits": 0,
        "page_cache_misses": 0,
        "page_cache_write_errors": 0,
        "artifact_dir": None,
        "cache_profile_key": None,
        "cache_profile_sha256": None,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_root": None,
        "file_summaries": [],
    }
    if parse_settings.cleanup_mode == "off":
        return [], details

    cleanup_profile_key, cleanup_profile_sha256, cleanup_profile_payload = _local_cleanup_profile_key(
        parse_settings
    )
    cache_root = _local_cleanup_cache_root(parsed_docs_root)
    details["artifact_dir"] = str(cache_root)
    details["cache_root"] = str(cache_root)
    details["cache_profile_key"] = cleanup_profile_key
    details["cache_profile_sha256"] = cleanup_profile_sha256

    pdf_reader = deps["pymupdf4llm"].LlamaMarkdownReader()
    docs: list[Any] = []
    for item in sorted(source_manifest, key=lambda row: str(row.relative_path)):
        relative_path = Path(str(item.relative_path)).as_posix()
        if not relative_path.lower().endswith(".pdf"):
            continue
        absolute_path = source_dir / relative_path
        source_relative_path = str((Path(ticker) / year / relative_path).as_posix())
        source_sha256 = str(item.sha256)
        source_size_bytes = int(item.size_bytes)
        source_file_name = Path(relative_path).name

        entry_dir = _local_cleanup_entry_dir(
            parsed_docs_root=parsed_docs_root,
            cleanup_profile_key=cleanup_profile_key,
            source_sha256=source_sha256,
        )
        cached = _load_local_cleanup_cache_entry(
            entry_dir=entry_dir,
            cleanup_profile_key=cleanup_profile_key,
            cleanup_profile_sha256=cleanup_profile_sha256,
            source_relative_path=source_relative_path,
            source_sha256=source_sha256,
        )

        cache_status = "hit"
        if cached is not None:
            page_rows, manifest = cached
            details["cache_hits"] += 1
            cleanup_summary = dict(manifest.get("cleanup_summary", {}))
            page_count = int(manifest.get("page_count", len(page_rows)) or len(page_rows))
        else:
            cache_status = "miss"
            details["cache_misses"] += 1
            extracted_rows, page_count = _extract_pdf_page_rows(
                pdf_reader=pdf_reader,
                absolute_path=absolute_path,
            )
            if page_count <= 0:
                continue
            scoped_pdf = ScopedPdf(
                ticker=ticker,
                year=year,
                absolute_path=absolute_path,
                source_relative_path=source_relative_path,
                source_file_name=source_file_name,
                source_sha256=source_sha256,
                source_size_bytes=source_size_bytes,
                page_count=page_count,
            )
            page_rows, cleanup_artifacts = _cleanup_page_rows_with_llm(
                scoped_pdf=scoped_pdf,
                page_rows=extracted_rows,
                settings=parse_settings,
                page_cache_root=_cleanup_page_cache_root(parsed_docs_root),
            )
            for row in cleanup_artifacts.audit_rows:
                row.setdefault("original_preview", str(row.get("original_preview", ""))[:CLEANUP_DEFAULT_PREVIEW_CHARS])
                row.setdefault("cleaned_preview", str(row.get("cleaned_preview", ""))[:CLEANUP_DEFAULT_PREVIEW_CHARS])
            manifest = _write_local_cleanup_cache_entry(
                entry_dir=entry_dir,
                cleanup_profile_key=cleanup_profile_key,
                cleanup_profile_sha256=cleanup_profile_sha256,
                cleanup_profile_payload=cleanup_profile_payload,
                scoped_pdf=scoped_pdf,
                page_rows=page_rows,
                cleanup_artifacts=cleanup_artifacts,
            )
            cleanup_summary = dict(cleanup_artifacts.summary)

        enabled_for_doc = bool(cleanup_summary.get("enabled_for_doc", False))
        if enabled_for_doc:
            details["attempted_files"] += 1
        details["attempted_pages"] += int(cleanup_summary.get("attempted_pages", 0) or 0)
        details["accepted_pages"] += int(cleanup_summary.get("accepted_pages", 0) or 0)
        details["rejected_pages"] += int(cleanup_summary.get("rejected_pages", 0) or 0)
        details["failed_pages"] += int(cleanup_summary.get("failed_pages", 0) or 0)
        details["page_cache_hits"] += int(cleanup_summary.get("page_cache_hits", 0) or 0)
        details["page_cache_misses"] += int(cleanup_summary.get("page_cache_misses", 0) or 0)
        details["page_cache_write_errors"] += int(
            cleanup_summary.get("page_cache_write_errors", 0) or 0
        )
        details["file_summaries"].append(
            {
                "source_relative_path": source_relative_path,
                "source_sha256": source_sha256,
                "cache_status": cache_status,
                "summary": cleanup_summary,
            }
        )
        docs.extend(
            _build_documents_from_page_rows(
                deps=deps,
                page_rows=page_rows,
                source_relative_path=source_relative_path,
                source_file_name=source_file_name,
                total_pages=page_count,
                absolute_path=absolute_path,
            )
        )

    return docs, details


def _build_or_load_vector_index(
    deps: dict[str, Any],
    *,
    source_dir: Path,
    source_manifest: list[Any],
    parsed_docs_root: Path,
    persist_dir: Path,
    rebuild: bool,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    pdf_source_mode: str,
    parse_settings: Any,
    ticker: str,
    year: str,
    cached_documents: list[Any] | None = None,
):
    settings = deps["Settings"]
    settings.embed_model = deps["OpenAIEmbedding"](model=embed_model, timeout=60, max_retries=5)
    settings.text_splitter = deps["SentenceSplitter"](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    storage_ready = any(persist_dir.iterdir()) and not rebuild

    if storage_ready:
        storage_ctx = deps["StorageContext"].from_defaults(persist_dir=str(persist_dir))
        vector_index = deps["load_index_from_storage"](storage_ctx)
        docstore = vector_index.storage_context.docstore
        nodes = list(docstore.docs.values())
        return vector_index, nodes, None

    if pdf_source_mode == "cache_only":
        if cached_documents is None:
            raise RuntimeError("cache_only mode requires preloaded cached_documents")
        docs = cached_documents
        local_parser_cleanup = None
    else:
        if parse_settings.cleanup_mode != "off":
            docs, local_parser_cleanup = _load_or_build_local_parser_docs_with_cleanup_cache(
                deps=deps,
                parse_settings=parse_settings,
                source_dir=source_dir,
                source_manifest=source_manifest,
                parsed_docs_root=parsed_docs_root,
                ticker=ticker,
                year=year,
            )
        else:
            pdf_reader = deps["pymupdf4llm"].LlamaMarkdownReader()
            extractors = {".pdf": pdf_reader}
            docs = deps["SimpleDirectoryReader"](
                input_dir=str(source_dir),
                file_extractor=extractors,
            ).load_data()
            local_parser_cleanup = None

    pipeline = deps["IngestionPipeline"](transformations=[settings.text_splitter])
    nodes = pipeline.run(documents=docs)

    vector_index = deps["VectorStoreIndex"](nodes)
    vector_index.storage_context.persist(persist_dir=str(persist_dir))
    return vector_index, nodes, local_parser_cleanup


def _build_query_engine(
    deps: dict[str, Any],
    *,
    vector_index: Any,
    nodes: list[Any],
    model_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    similarity_top_k: int,
    fusion_top_k: int,
    retrieval_rerank_profile: str,
    prompt_cache_options: dict[str, str] | None = None,
):
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 300,
        "strict": True,
        "max_retries": 5,
        "system_prompt": system_prompt,
    }
    if prompt_cache_options:
        llm_kwargs["additional_kwargs"] = dict(prompt_cache_options)

    llm = deps["OpenAI"](
        **llm_kwargs,
    )

    structured_llm = llm.as_structured_llm(ExtractedTargets)
    synthesizer = deps["get_response_synthesizer"](llm=structured_llm, response_mode="compact")

    rerank_spec = resolve_retrieval_rerank_spec(
        profile=retrieval_rerank_profile,
        fusion_top_k=fusion_top_k,
    )
    fusion_candidate_top_k = rerank_spec.candidate_top_k if rerank_spec else fusion_top_k

    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    bm25 = deps["BM25Retriever"].from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)

    hybrid = deps["QueryFusionRetriever"](
        retrievers=[vector_retriever, bm25],
        num_queries=1,
        similarity_top_k=fusion_candidate_top_k,
        mode="reciprocal_rerank",
        use_async=False,
    )
    node_postprocessors: list[Any] | None = None
    if rerank_spec is not None:
        try:
            reranker = deps["SentenceTransformerRerank"](
                top_n=rerank_spec.top_n,
                model=rerank_spec.model,
                device=rerank_spec.device,
                keep_retrieval_score=True,
                trust_remote_code=False,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Reranking requires sentence-transformer dependencies. "
                "Install with: `uv sync --extra rag --extra rerank --extra dev`."
            ) from exc
        node_postprocessors = [reranker]

    return deps["RetrieverQueryEngine"](
        retriever=hybrid,
        response_synthesizer=synthesizer,
        node_postprocessors=node_postprocessors,
    )


def run_batch(
    *,
    config: RunConfig,
    config_path: Path | None,
    output_dir: Path,
    system_prompt: str,
    source_docs_root: Path,
    index_registry: IndexRegistry,
    skip_doc_names: set[str] | None = None,
    progress_fn: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    deps = _import_rag_dependencies()

    settings = config.component_settings
    embed_model = settings.get("embedding_model", "text-embedding-3-large")
    chunk_size = int(settings.get("chunk_size", 8192))
    chunk_overlap = int(settings.get("chunk_overlap", 512))
    similarity_top_k = int(settings.get("similarity_top_k", 10))
    fusion_top_k = int(settings.get("fusion_top_k", 5))
    max_tokens = int(settings.get("max_tokens", 16390))
    temperature = float(settings.get("temperature", 0.0))
    parse_settings = parse_settings_from_component_settings(config.component_settings)
    parsed_docs_root = config.parsed_docs_root
    if not parsed_docs_root.is_absolute():
        parsed_docs_root = (Path.cwd() / parsed_docs_root).resolve()
    prompt_cache_options = build_rag_extract_prompt_cache_options(
        config=config,
        extract_system_prompt=system_prompt,
    )

    written: list[Path] = []
    index_events: list[dict[str, Any]] = []
    postprocess_summary_rows: list[dict[str, Any]] = []
    retrieval_trace_rows: list[dict[str, Any]] = []
    postprocess_profile = config.target_postprocess_profile
    retrieval_rerank_spec = resolve_retrieval_rerank_spec(
        profile=config.retrieval_rerank_profile,
        fusion_top_k=fusion_top_k,
    )
    total_docs = len(config.company_tickers) * len(config.years)
    current_doc = 0

    for company_ticker in config.company_tickers:
        for year in config.years:
            doc_name = target_doc_name(company_ticker.lower(), year)
            if skip_doc_names and doc_name in skip_doc_names:
                current_doc += 1
                if progress_fn is not None:
                    progress_fn(doc_name, current_doc, total_docs)
                continue
            ticker = company_ticker.lower()
            source_dir = source_docs_root / ticker / year

            source_manifest = build_source_manifest(source_dir)
            parsed_cache_fingerprint: str | None = None
            parsed_cache_details: dict[str, Any] | None = None
            cached_documents: list[Any] | None = None

            if parse_settings.pdf_source_mode == "cache_only":
                parsed_cache_fingerprint, parsed_cache_details = build_cache_content_fingerprint(
                    source_docs_root=source_docs_root,
                    source_dir=source_dir,
                    source_manifest=source_manifest,
                    parsed_docs_root=parsed_docs_root,
                    settings=parse_settings,
                    config_path=config_path,
                )

            fingerprint = compute_index_fingerprint(
                pipeline_version=config.pipeline_version,
                source_manifest=source_manifest,
                component_versions=config.component_versions,
                component_settings=config.component_settings,
                embedding_model=embed_model,
                parsed_cache_content_fingerprint=parsed_cache_fingerprint,
            )
            selection = index_registry.select(
                pipeline_version=config.pipeline_version,
                fingerprint=fingerprint,
                index_policy=config.index_policy,
            )

            persist_store = selection.index_dir / "store"
            should_rebuild = selection.action in {"built", "rebuilt"}

            if parse_settings.pdf_source_mode == "cache_only" and should_rebuild:
                cached_pages = load_cached_pages_for_source_manifest(
                    source_docs_root=source_docs_root,
                    source_dir=source_dir,
                    source_manifest=source_manifest,
                    parsed_docs_root=parsed_docs_root,
                    settings=parse_settings,
                    config_path=config_path,
                )
                docs: list[Any] = []
                for cached in cached_pages:
                    docs.append(
                        deps["Document"](
                            text=cached.text_markdown,
                            metadata={
                                "file_name": cached.file_name,
                                "source_relative_path": cached.source_relative_path,
                                "page": cached.page,
                                "total_pages": cached.total_pages,
                            },
                        )
                    )
                cached_documents = docs

            vector_index, nodes, local_parser_cleanup = _build_or_load_vector_index(
                deps,
                source_dir=source_dir,
                source_manifest=source_manifest,
                parsed_docs_root=parsed_docs_root,
                persist_dir=persist_store,
                rebuild=should_rebuild,
                embed_model=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pdf_source_mode=parse_settings.pdf_source_mode,
                parse_settings=parse_settings,
                ticker=ticker,
                year=year,
                cached_documents=cached_documents,
            )

            index_registry.write_manifest(
                selection,
                source_root=source_dir,
                source_manifest=source_manifest,
                component_versions=config.component_versions,
                component_settings=config.component_settings,
                embedding_model=embed_model,
                parsed_cache=parsed_cache_details,
            )

            qe = _build_query_engine(
                deps,
                vector_index=vector_index,
                nodes=nodes,
                model_name=config.model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                similarity_top_k=similarity_top_k,
                fusion_top_k=fusion_top_k,
                retrieval_rerank_profile=config.retrieval_rerank_profile,
                prompt_cache_options=prompt_cache_options,
            )

            response = qe.query(QUERY_PROMPT)
            model_obj = response.response
            retrieved_chunks = _collect_retrieved_chunks(response)
            retrieval_trace_rows.append(
                {
                    "doc": doc_name,
                    "retrieved_count": len(retrieved_chunks),
                    "retrieved_chunks": retrieved_chunks,
                }
            )

            out_path = output_dir / doc_name
            payload = model_obj.model_dump(mode="json")
            if postprocess_profile != "off":
                payload, postprocess_summary = apply_target_postprocess(
                    payload=payload,
                    profile=postprocess_profile,
                )
                postprocess_summary_rows.append({"doc": doc_name, **postprocess_summary})
            write_json(out_path, payload)
            written.append(out_path)
            current_doc += 1
            if progress_fn is not None:
                progress_fn(doc_name, current_doc, total_docs)

            index_events.append(
                {
                    "doc": doc_name,
                    "index_id": selection.index_id,
                    "index_fingerprint": selection.index_fingerprint,
                    "index_action": selection.action,
                    "index_dir": str(selection.index_dir),
                    "pdf_source_mode": parse_settings.pdf_source_mode,
                    "parsed_cache_fingerprint": parsed_cache_fingerprint,
                    "parsed_cache_provider": (
                        parsed_cache_details.get("provider") if parsed_cache_details else None
                    ),
                    "parsed_cache_profile_key": (
                        parsed_cache_details.get("profile_key") if parsed_cache_details else None
                    ),
                    "parsed_cache_entries": (
                        len(parsed_cache_details.get("entries", [])) if parsed_cache_details else None
                    ),
                    "local_parser_cleanup_mode": (
                        local_parser_cleanup.get("mode") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_model": (
                        local_parser_cleanup.get("model") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_cache_profile_key": (
                        local_parser_cleanup.get("cache_profile_key") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_cache_profile_sha256": (
                        local_parser_cleanup.get("cache_profile_sha256") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_cache_hits": (
                        local_parser_cleanup.get("cache_hits") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_cache_misses": (
                        local_parser_cleanup.get("cache_misses") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_cache_root": (
                        local_parser_cleanup.get("cache_root") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_attempted_files": (
                        local_parser_cleanup.get("attempted_files") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_attempted_pages": (
                        local_parser_cleanup.get("attempted_pages") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_accepted_pages": (
                        local_parser_cleanup.get("accepted_pages") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_rejected_pages": (
                        local_parser_cleanup.get("rejected_pages") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_failed_pages": (
                        local_parser_cleanup.get("failed_pages") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_page_cache_hits": (
                        local_parser_cleanup.get("page_cache_hits") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_page_cache_misses": (
                        local_parser_cleanup.get("page_cache_misses") if local_parser_cleanup else None
                    ),
                    "local_parser_cleanup_page_cache_write_errors": (
                        local_parser_cleanup.get("page_cache_write_errors")
                        if local_parser_cleanup
                        else None
                    ),
                    "local_parser_cleanup_artifact_dir": (
                        local_parser_cleanup.get("artifact_dir") if local_parser_cleanup else None
                    ),
                    "prompt_cache_enabled": bool(prompt_cache_options),
                    "prompt_cache_retention": prompt_cache_options.get("prompt_cache_retention")
                    if prompt_cache_options
                    else None,
                    "retrieval_rerank_profile": config.retrieval_rerank_profile,
                    "retrieval_rerank_model": (
                        retrieval_rerank_spec.model if retrieval_rerank_spec else None
                    ),
                    "retrieval_rerank_device": (
                        retrieval_rerank_spec.device if retrieval_rerank_spec else None
                    ),
                    "retrieval_rerank_candidate_top_k": (
                        retrieval_rerank_spec.candidate_top_k if retrieval_rerank_spec else None
                    ),
                    "retrieval_rerank_top_n": (
                        retrieval_rerank_spec.top_n if retrieval_rerank_spec else None
                    ),
                    "retrieved_chunks_count": len(retrieved_chunks),
                }
            )

    if postprocess_profile != "off":
        write_json(
            output_dir / "_target_postprocess_summary.json",
            {
                "profile": postprocess_profile,
                "documents": postprocess_summary_rows,
            },
        )

    if retrieval_trace_rows:
        write_json(
            output_dir / "_retrieved_chunks.json",
            {
                "query_prompt": QUERY_PROMPT,
                "retrieval_rerank_profile": config.retrieval_rerank_profile,
                "documents": retrieval_trace_rows,
            },
        )

    return written, index_events
