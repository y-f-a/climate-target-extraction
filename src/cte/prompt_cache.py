from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from .config import RunConfig
from .schemas import ExtractedTargets

PROMPT_CACHE_SCOPE_RAG_EXTRACT_ONLY = "rag_extract_only"
PROMPT_CACHE_SCHEMA_IDENTIFIER = "cte.schemas.ExtractedTargets"
PROMPT_CACHE_SCHEMA_VERSION = "v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _extracted_targets_schema_sha256() -> str:
    schema_payload = json.dumps(
        ExtractedTargets.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(schema_payload)


def rag_extract_prompt_cache_enabled(config: RunConfig) -> bool:
    return bool(config.pipeline == "rag" and config.openai_prompt_cache_enabled)


def manifest_prompt_cache_fields(config: RunConfig) -> dict[str, Any]:
    enabled = rag_extract_prompt_cache_enabled(config)
    return {
        "prompt_cache_enabled": enabled,
        "prompt_cache_retention": (
            config.openai_prompt_cache_retention if enabled else None
        ),
        "prompt_cache_scope": (
            PROMPT_CACHE_SCOPE_RAG_EXTRACT_ONLY if enabled else None
        ),
    }


def build_rag_extract_prompt_cache_key(
    *,
    config: RunConfig,
    extract_system_prompt: str,
) -> str | None:
    if not rag_extract_prompt_cache_enabled(config):
        return None

    payload = {
        "scope": PROMPT_CACHE_SCOPE_RAG_EXTRACT_ONLY,
        "pipeline_version": config.pipeline_version,
        "model_name": config.model_name,
        "extract_prompt_sha256": _sha256_text(extract_system_prompt),
        "schema_identifier": PROMPT_CACHE_SCHEMA_IDENTIFIER,
        "schema_version": PROMPT_CACHE_SCHEMA_VERSION,
        "schema_sha256": _extracted_targets_schema_sha256(),
    }
    digest = _sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return f"cte:rag_extract:{digest}"


def build_rag_extract_prompt_cache_options(
    *,
    config: RunConfig,
    extract_system_prompt: str,
) -> dict[str, str]:
    prompt_cache_key = build_rag_extract_prompt_cache_key(
        config=config,
        extract_system_prompt=extract_system_prompt,
    )
    if prompt_cache_key is None:
        return {}
    return {
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_retention": config.openai_prompt_cache_retention,
    }
