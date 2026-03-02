from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import RunConfig, load_run_config, maybe_load_env_file, resolve_source_docs_root
from .index_registry import SourceFileInfo, build_source_manifest
from .io import canonical_json_dumps, ensure_dir
from .live_status import LiveStatusTracker

CLIMATE_AUTO_TRIGGER_REGEX = (
    r"net\\s*zero|carbon\\s*neutral|scope\\s*1|scope\\s*2|scope\\s*3|"
    r"ghg|greenhouse\\s*gas|emissions\\s*target|science\\s*based\\s*target|sbti"
)

VALID_PDF_SOURCE_MODES = {"local_parser", "cache_only"}
VALID_OCR_MODES = {"auto", "off", "force"}
VALID_CLEANUP_MODES = {"off", "llm_faithful_v1"}
VALID_CLEANUP_DOC_SCOPES = {"doc_pairs", "all"}
VALID_CLEANUP_PAGE_CACHE_MISS_POLICIES = {"call_llm", "fail_open", "error"}
CLEANUP_DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
CLEANUP_DEFAULT_LENGTH_RATIO_MIN = 0.5
CLEANUP_DEFAULT_LENGTH_RATIO_MAX = 2.0
CLEANUP_DEFAULT_PREVIEW_CHARS = 500
CLEANUP_PAGE_CACHE_SCHEMA_VERSION = "v1"
CLEANUP_SYSTEM_PROMPT = (
    "You are a faithful page text cleanup assistant. Transcribe/clean page text exactly from the page image "
    "and provided extracted text. Preserve factual content, numbers, dates, units, and wording intent. "
    "Do not summarize. Do not add facts. Return only valid JSON with fields cleaned_text and notes."
)
CLEANUP_KEYWORD_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("net_zero", r"net[-\s]*zero", 2),
    ("carbon_neutral", r"carbon[-\s]*neutral|climate[-\s]*neutral|carbon[-\s]*negative", 2),
    ("scope", r"\bscope\s*[123]\b", 1),
    ("target", r"\btargets?\b", 1),
    ("emissions", r"\bemissions?\b|\bghg\b|\bgreenhouse\s*gas\b", 1),
    ("reduction", r"\breduction\b|\breduce\b|\bdecarbon", 1),
)
NUMERIC_TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)?%?")
WORD_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")


class CleanupResponse(BaseModel):
    cleaned_text: str
    notes: str | None = None


class ParseCacheError(RuntimeError):
    """Base parse cache error."""


class ParseCacheMissingError(ParseCacheError):
    """Raised when required cache entries are missing."""

    def __init__(self, message: str, *, missing_paths: list[str]) -> None:
        super().__init__(message)
        self.missing_paths = missing_paths


@dataclass(frozen=True)
class ParseSettings:
    pdf_source_mode: str
    provider: str
    profile: str
    settings_version: str
    result_type: str
    language: str
    ocr_mode: str
    premium_mode: bool
    extract_layout: bool
    adaptive_long_table: bool
    high_res_ocr: bool
    auto_mode: bool
    auto_trigger_tables: bool
    auto_trigger_images: bool
    auto_trigger_regex: str
    estimate_standard_credits_per_page: float
    estimate_premium_credits_per_page: float
    estimate_currency_label: str
    cleanup_mode: str
    cleanup_model: str
    cleanup_max_pages_per_pdf: int
    cleanup_score_threshold: int
    cleanup_use_score_threshold: bool
    cleanup_doc_scope: str
    cleanup_enabled_doc_pairs: tuple[str, ...]
    cleanup_extra_keywords: tuple[str, ...]
    cleanup_low_text_rescue: bool
    cleanup_low_text_max_chars: int
    cleanup_image_dpi: int
    cleanup_timeout_sec: int
    cleanup_profile_version: str
    cleanup_length_ratio_max: float
    cleanup_numeric_guardrail_enabled: bool
    cleanup_page_cache_miss_policy: str

    def profile_key(self) -> str:
        token = f"{self.profile}-{self.settings_version}".strip().lower()
        if self.cleanup_mode == "off":
            return re.sub(r"[^a-z0-9._-]+", "-", token).strip("-")
        cleanup_payload = {
            "cleanup_mode": self.cleanup_mode,
            "cleanup_model": self.cleanup_model,
            "cleanup_max_pages_per_pdf": self.cleanup_max_pages_per_pdf,
            "cleanup_score_threshold": self.cleanup_score_threshold,
            "cleanup_use_score_threshold": self.cleanup_use_score_threshold,
            "cleanup_doc_scope": self.cleanup_doc_scope,
            "cleanup_enabled_doc_pairs": list(self.cleanup_enabled_doc_pairs),
            "cleanup_extra_keywords": list(self.cleanup_extra_keywords),
            "cleanup_low_text_rescue": self.cleanup_low_text_rescue,
            "cleanup_low_text_max_chars": self.cleanup_low_text_max_chars,
            "cleanup_image_dpi": self.cleanup_image_dpi,
            "cleanup_timeout_sec": self.cleanup_timeout_sec,
            "cleanup_profile_version": self.cleanup_profile_version,
            "cleanup_length_ratio_max": self.cleanup_length_ratio_max,
            "cleanup_numeric_guardrail_enabled": self.cleanup_numeric_guardrail_enabled,
            "cleanup_page_cache_miss_policy": self.cleanup_page_cache_miss_policy,
        }
        digest = _sha256_text(json.dumps(cleanup_payload, ensure_ascii=False, sort_keys=True))[:12]
        cleanup_suffix = f"cleanup-{self.cleanup_profile_version}-{digest}"
        return re.sub(r"[^a-z0-9._-]+", "-", f"{token}-{cleanup_suffix}".lower()).strip("-")

    def provider_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "profile": self.profile,
            "settings_version": self.settings_version,
            "result_type": self.result_type,
            "language": self.language,
            "ocr_mode": self.ocr_mode,
            "premium_mode": self.premium_mode,
            "extract_layout": self.extract_layout,
            "adaptive_long_table": self.adaptive_long_table,
            "high_res_ocr": self.high_res_ocr,
            "auto_mode": self.auto_mode,
            "auto_trigger_tables": self.auto_trigger_tables,
            "auto_trigger_images": self.auto_trigger_images,
            "auto_trigger_regex": self.auto_trigger_regex,
            "cleanup_mode": self.cleanup_mode,
            "cleanup_model": self.cleanup_model,
            "cleanup_max_pages_per_pdf": self.cleanup_max_pages_per_pdf,
            "cleanup_score_threshold": self.cleanup_score_threshold,
            "cleanup_use_score_threshold": self.cleanup_use_score_threshold,
            "cleanup_doc_scope": self.cleanup_doc_scope,
            "cleanup_enabled_doc_pairs": list(self.cleanup_enabled_doc_pairs),
            "cleanup_extra_keywords": list(self.cleanup_extra_keywords),
            "cleanup_low_text_rescue": self.cleanup_low_text_rescue,
            "cleanup_low_text_max_chars": self.cleanup_low_text_max_chars,
            "cleanup_image_dpi": self.cleanup_image_dpi,
            "cleanup_timeout_sec": self.cleanup_timeout_sec,
            "cleanup_profile_version": self.cleanup_profile_version,
            "cleanup_length_ratio_max": self.cleanup_length_ratio_max,
            "cleanup_numeric_guardrail_enabled": self.cleanup_numeric_guardrail_enabled,
            "cleanup_page_cache_miss_policy": self.cleanup_page_cache_miss_policy,
        }


@dataclass(frozen=True)
class ScopedPdf:
    ticker: str
    year: str
    absolute_path: Path
    source_relative_path: str
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    page_count: int


@dataclass(frozen=True)
class CachePlanItem:
    scoped_pdf: ScopedPdf
    status: str
    reason: str
    entry_dir: Path
    manifest_path: Path
    manifest: dict[str, Any] | None


@dataclass(frozen=True)
class CreditBreakdown:
    standard_pages: int
    premium_pages: int
    estimated_credits: float


@dataclass(frozen=True)
class CachedPageRow:
    file_name: str
    source_relative_path: str
    page: int
    total_pages: int
    text_markdown: str


@dataclass(frozen=True)
class CleanupRunArtifacts:
    summary: dict[str, Any]
    audit_rows: list[dict[str, Any]]
    request_rows: list[dict[str, Any]]
    output_rows: list[dict[str, Any]]


def parse_settings_from_component_settings(component_settings: dict[str, Any]) -> ParseSettings:
    mode = str(component_settings.get("pdf_source_mode", "local_parser")).strip().lower()
    if mode not in VALID_PDF_SOURCE_MODES:
        raise ValueError(f"Invalid pdf_source_mode='{mode}'. Expected one of {sorted(VALID_PDF_SOURCE_MODES)}")

    ocr_mode = str(component_settings.get("pdf_parse_ocr_mode", "auto")).strip().lower()
    if ocr_mode not in VALID_OCR_MODES:
        raise ValueError(f"Invalid pdf_parse_ocr_mode='{ocr_mode}'. Expected one of {sorted(VALID_OCR_MODES)}")
    cleanup_mode = str(component_settings.get("pdf_cleanup_mode", "off")).strip().lower()
    if cleanup_mode not in VALID_CLEANUP_MODES:
        raise ValueError(
            f"Invalid pdf_cleanup_mode='{cleanup_mode}'. Expected one of {sorted(VALID_CLEANUP_MODES)}"
        )
    cleanup_doc_scope_default = "all" if mode == "local_parser" else "doc_pairs"
    cleanup_doc_scope = (
        str(component_settings.get("pdf_cleanup_doc_scope", cleanup_doc_scope_default)).strip().lower()
    )
    if cleanup_doc_scope not in VALID_CLEANUP_DOC_SCOPES:
        raise ValueError(
            "Invalid pdf_cleanup_doc_scope="
            f"'{cleanup_doc_scope}'. Expected one of {sorted(VALID_CLEANUP_DOC_SCOPES)}"
        )
    cleanup_page_cache_miss_policy = (
        str(component_settings.get("pdf_cleanup_page_cache_miss_policy", "call_llm")).strip().lower()
    )
    if cleanup_page_cache_miss_policy not in VALID_CLEANUP_PAGE_CACHE_MISS_POLICIES:
        raise ValueError(
            "Invalid pdf_cleanup_page_cache_miss_policy="
            f"'{cleanup_page_cache_miss_policy}'. Expected one of "
            f"{sorted(VALID_CLEANUP_PAGE_CACHE_MISS_POLICIES)}"
        )

    provider = str(component_settings.get("pdf_parse_provider", "llamaparse")).strip().lower()
    profile = str(component_settings.get("pdf_parse_profile", "hq_auto_v1")).strip() or "hq_auto_v1"
    settings_version = (
        str(component_settings.get("pdf_parse_settings_version", "v1")).strip() or "v1"
    )
    result_type = str(component_settings.get("pdf_parse_result_type", "markdown")).strip().lower()
    language = str(component_settings.get("pdf_parse_language", "en")).strip().lower() or "en"

    def _as_bool(key: str, default: bool) -> bool:
        raw = component_settings.get(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _as_float(key: str, default: float) -> float:
        raw = component_settings.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _as_int(key: str, default: int) -> int:
        raw = component_settings.get(key, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _as_doc_pairs(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = component_settings.get(key)
        if raw is None:
            return default
        if isinstance(raw, str):
            items = [item.strip() for item in raw.split(",") if item.strip()]
        elif isinstance(raw, (list, tuple, set)):
            items = [str(item).strip() for item in raw if str(item).strip()]
        else:
            raise ValueError(
                f"Invalid {key} type: {type(raw).__name__}. Expected comma-separated string or list."
            )

        normalized: set[str] = set()
        for token in items:
            value = token.lower()
            if ":" not in value:
                raise ValueError(
                    f"Invalid doc pair '{token}' in {key}. Expected format 'ticker:year' (e.g. tsla:2023)."
                )
            ticker_raw, year_raw = value.split(":", 1)
            ticker = re.sub(r"[^a-z0-9]+", "", ticker_raw)
            year = year_raw.strip()
            if not ticker or not re.fullmatch(r"\d{4}", year):
                raise ValueError(
                    f"Invalid doc pair '{token}' in {key}. Expected format 'ticker:year' with 4-digit year."
                )
            normalized.add(f"{ticker}:{year}")
        return tuple(sorted(normalized))

    def _as_string_list(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = component_settings.get(key)
        if raw is None:
            return default
        if isinstance(raw, str):
            items = [item.strip() for item in raw.split(",") if item.strip()]
        elif isinstance(raw, (list, tuple, set)):
            items = [str(item).strip() for item in raw if str(item).strip()]
        else:
            raise ValueError(
                f"Invalid {key} type: {type(raw).__name__}. Expected comma-separated string or list."
            )
        normalized = tuple(sorted({item.lower() for item in items if item}))
        return normalized

    auto_trigger_regex = (
        str(component_settings.get("pdf_parse_auto_trigger_regex", CLIMATE_AUTO_TRIGGER_REGEX)).strip()
        or CLIMATE_AUTO_TRIGGER_REGEX
    )
    cleanup_max_pages_per_pdf = max(1, _as_int("pdf_cleanup_max_pages_per_pdf", 3))
    cleanup_score_threshold = _as_int("pdf_cleanup_score_threshold", 3)
    cleanup_low_text_max_chars = max(1, _as_int("pdf_cleanup_low_text_max_chars", 60))
    cleanup_image_dpi = max(72, _as_int("pdf_cleanup_image_dpi", 200))
    cleanup_timeout_sec = max(1, _as_int("pdf_cleanup_timeout_sec", 90))
    cleanup_profile_version = (
        str(component_settings.get("pdf_cleanup_profile_version", "v1")).strip().lower() or "v1"
    )
    cleanup_model = (
        str(component_settings.get("pdf_cleanup_model", CLEANUP_DEFAULT_MODEL)).strip()
        or CLEANUP_DEFAULT_MODEL
    )
    cleanup_enabled_doc_pairs = _as_doc_pairs("pdf_cleanup_enabled_doc_pairs", tuple())
    cleanup_extra_keywords = _as_string_list("pdf_cleanup_extra_keywords", tuple())
    cleanup_length_ratio_max = max(
        CLEANUP_DEFAULT_LENGTH_RATIO_MIN,
        _as_float("pdf_cleanup_length_ratio_max", CLEANUP_DEFAULT_LENGTH_RATIO_MAX),
    )

    return ParseSettings(
        pdf_source_mode=mode,
        provider=provider,
        profile=profile,
        settings_version=settings_version,
        result_type=result_type,
        language=language,
        ocr_mode=ocr_mode,
        premium_mode=_as_bool("pdf_parse_premium_mode", True),
        extract_layout=_as_bool("pdf_parse_extract_layout", True),
        adaptive_long_table=_as_bool("pdf_parse_adaptive_long_table", True),
        high_res_ocr=_as_bool("pdf_parse_high_res_ocr", True),
        auto_mode=_as_bool("pdf_parse_auto_mode", True),
        auto_trigger_tables=_as_bool("pdf_parse_auto_trigger_tables", True),
        auto_trigger_images=_as_bool("pdf_parse_auto_trigger_images", True),
        auto_trigger_regex=auto_trigger_regex,
        estimate_standard_credits_per_page=_as_float(
            "pdf_parse_estimate_standard_credits_per_page", 15.0
        ),
        estimate_premium_credits_per_page=_as_float(
            "pdf_parse_estimate_premium_credits_per_page", 45.0
        ),
        estimate_currency_label=str(
            component_settings.get("pdf_parse_estimate_currency_label", "credits")
        ).strip()
        or "credits",
        cleanup_mode=cleanup_mode,
        cleanup_model=cleanup_model,
        cleanup_max_pages_per_pdf=cleanup_max_pages_per_pdf,
        cleanup_score_threshold=cleanup_score_threshold,
        cleanup_use_score_threshold=_as_bool("pdf_cleanup_use_score_threshold", True),
        cleanup_doc_scope=cleanup_doc_scope,
        cleanup_enabled_doc_pairs=cleanup_enabled_doc_pairs,
        cleanup_extra_keywords=cleanup_extra_keywords,
        cleanup_low_text_rescue=_as_bool("pdf_cleanup_low_text_rescue", True),
        cleanup_low_text_max_chars=cleanup_low_text_max_chars,
        cleanup_image_dpi=cleanup_image_dpi,
        cleanup_timeout_sec=cleanup_timeout_sec,
        cleanup_profile_version=cleanup_profile_version,
        cleanup_length_ratio_max=cleanup_length_ratio_max,
        cleanup_numeric_guardrail_enabled=_as_bool("pdf_cleanup_numeric_guardrail_enabled", True),
        cleanup_page_cache_miss_policy=cleanup_page_cache_miss_policy,
    )


def cache_entry_dir(parsed_docs_root: Path, settings: ParseSettings, source_sha256: str) -> Path:
    return parsed_docs_root / settings.provider / settings.profile_key() / source_sha256


def _manifest_path(entry_dir: Path) -> Path:
    return entry_dir / "manifest.json"


def _pages_path(entry_dir: Path) -> Path:
    return entry_dir / "pages.jsonl"


def _raw_path(entry_dir: Path) -> Path:
    return entry_dir / "raw_response.json.gz"


def _cleanup_audit_path(entry_dir: Path) -> Path:
    return entry_dir / "cleanup_audit.jsonl"


def _cleanup_requests_path(entry_dir: Path) -> Path:
    return entry_dir / "cleanup_requests.jsonl"


def _cleanup_outputs_path(entry_dir: Path) -> Path:
    return entry_dir / "cleanup_outputs.jsonl"


def _cleanup_page_cache_root(parsed_docs_root: Path) -> Path:
    return parsed_docs_root / "pymupdf_cleanup_page_cache"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_page_count(path: Path) -> int:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for parse-cache planning. Install with `uv sync --extra rag`."
        ) from exc

    with fitz.open(path) as doc:
        return int(doc.page_count)


def _build_scope(
    *,
    source_docs_root: Path,
    company_tickers: list[str],
    years: list[str],
) -> list[ScopedPdf]:
    scoped: list[ScopedPdf] = []
    for ticker in company_tickers:
        ticker_lc = ticker.lower()
        for year in years:
            source_dir = source_docs_root / ticker_lc / year
            if not source_dir.exists():
                raise FileNotFoundError(f"Source path not found: {source_dir}")

            source_manifest = build_source_manifest(source_dir)
            pdf_files = [item for item in source_manifest if item.relative_path.lower().endswith(".pdf")]
            if not pdf_files:
                raise FileNotFoundError(f"No PDF source files found in {source_dir}")

            for item in pdf_files:
                absolute_path = source_dir / item.relative_path
                scoped.append(
                    ScopedPdf(
                        ticker=ticker_lc,
                        year=year,
                        absolute_path=absolute_path,
                        source_relative_path=str((Path(ticker_lc) / year / item.relative_path).as_posix()),
                        source_file_name=Path(item.relative_path).name,
                        source_sha256=item.sha256,
                        source_size_bytes=item.size_bytes,
                        page_count=_safe_page_count(absolute_path),
                    )
                )

    scoped.sort(key=lambda row: row.source_relative_path)
    return scoped


def _cleanup_doc_key(scoped_pdf: ScopedPdf) -> str:
    return f"{scoped_pdf.ticker.lower()}:{scoped_pdf.year}"


def _cleanup_enabled_for_doc(*, settings: ParseSettings, scoped_pdf: ScopedPdf) -> bool:
    if settings.cleanup_mode == "off":
        return False
    if settings.cleanup_doc_scope == "all":
        return True
    enabled_pairs = set(settings.cleanup_enabled_doc_pairs)
    if not enabled_pairs:
        return False
    return _cleanup_doc_key(scoped_pdf) in enabled_pairs


def _cleanup_keyword_patterns(settings: ParseSettings) -> tuple[tuple[str, str, int], ...]:
    if not settings.cleanup_extra_keywords:
        return CLEANUP_KEYWORD_PATTERNS
    extra: list[tuple[str, str, int]] = []
    for idx, keyword in enumerate(settings.cleanup_extra_keywords, start=1):
        token = keyword.strip().lower()
        if not token:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", token).strip("_") or f"extra_{idx}"
        escaped = re.escape(token).replace(r"\ ", r"\s+")
        extra.append((f"extra_{slug}_{idx}", escaped, 1))
    return CLEANUP_KEYWORD_PATTERNS + tuple(extra)


def _word_tokens(text: str) -> list[str]:
    return WORD_TOKEN_PATTERN.findall(text.lower())


def _numeric_tokens(text: str) -> list[str]:
    return NUMERIC_TOKEN_PATTERN.findall(text.lower())


def _numeric_tokens_for_fidelity(
    *,
    original_text: str,
) -> tuple[list[str], list[str]]:
    original_numeric = _numeric_tokens(original_text)
    if not original_numeric:
        return [], []

    # Heuristic: repeated 4+ digit tokens near the header are often OCR merge noise
    # (e.g., concatenated running headers/page markers like "3825"), not factual values.
    header_numeric = _numeric_tokens(str(original_text)[:160])
    header_set = set(header_numeric)
    counts: dict[str, int] = {}
    for token in original_numeric:
        counts[token] = counts.get(token, 0) + 1

    ignored: set[str] = set()
    for token, count in counts.items():
        compact = token.replace(",", "").replace(".", "")
        if token not in header_set:
            continue
        if count < 2:
            continue
        if len(compact) < 4:
            continue
        if compact.startswith("20"):
            continue
        ignored.add(token)

    filtered = [token for token in original_numeric if token not in ignored]
    if not filtered:
        filtered = list(original_numeric)

    # Use unique tokens for retention so repeated OCR duplicates do not over-penalize.
    return sorted(set(filtered)), sorted(ignored)


def _render_page_image_data_url(
    *,
    scoped_pdf: ScopedPdf,
    page_no: int,
    image_dpi: int,
) -> str:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for cleanup page rendering. Install with `uv sync --extra rag`."
        ) from exc

    scale = max(float(image_dpi), 72.0) / 72.0
    with fitz.open(scoped_pdf.absolute_path) as doc:
        if page_no < 1 or page_no > int(doc.page_count):
            raise ValueError(
                f"Invalid page {page_no} for {scoped_pdf.source_relative_path} (max={doc.page_count})."
            )
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_bytes = pix.tobytes("png")
    image_b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def _score_cleanup_page(
    *,
    text_markdown: str,
    low_text_max_chars: int,
    settings: ParseSettings,
) -> dict[str, Any]:
    text = str(text_markdown or "")
    normalized = text.strip()
    text_lc = normalized.lower()

    keyword_hits: dict[str, int] = {}
    keyword_score = 0
    for key, pattern, weight in _cleanup_keyword_patterns(settings):
        count = len(re.findall(pattern, text_lc))
        keyword_hits[key] = count
        if count > 0:
            keyword_score += weight

    year_hits = len(set(YEAR_PATTERN.findall(text_lc)))
    numeric_hits = len(_numeric_tokens(text_lc))
    anchor_score = (1 if year_hits > 0 else 0) + (1 if numeric_hits > 0 else 0)

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    repeated_line_ratio = 0.0
    if lines:
        repeated_line_ratio = (len(lines) - len(set(lines))) / len(lines)
    noise_penalty = 1 if repeated_line_ratio >= 0.35 else 0

    char_count = len(normalized)
    short_text_penalty = 1 if char_count < low_text_max_chars else 0

    score = keyword_score + anchor_score - noise_penalty - short_text_penalty
    return {
        "score": int(score),
        "keyword_hits": keyword_hits,
        "year_hits": year_hits,
        "numeric_hits": numeric_hits,
        "repeated_line_ratio": repeated_line_ratio,
        "noise_penalty": noise_penalty,
        "short_text_penalty": short_text_penalty,
        "char_count": char_count,
    }


def _select_cleanup_candidates(
    *,
    page_rows: list[dict[str, Any]],
    settings: ParseSettings,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row in page_rows:
        page = int(row.get("page", 0) or 0)
        text_markdown = str(row.get("text_markdown", ""))
        scoring = _score_cleanup_page(
            text_markdown=text_markdown,
            low_text_max_chars=settings.cleanup_low_text_max_chars,
            settings=settings,
        )
        scored_rows.append(
            {
                "page": page,
                "text_markdown": text_markdown,
                "score": int(scoring["score"]),
                "scoring": scoring,
            }
        )

    primary: list[dict[str, Any]] = []
    for row in scored_rows:
        if row["page"] <= 0:
            continue
        if settings.cleanup_use_score_threshold and int(row["score"]) < settings.cleanup_score_threshold:
            continue
        primary.append(row)
    primary.sort(key=lambda row: (-int(row["score"]), int(row["page"])))
    selected = primary[: settings.cleanup_max_pages_per_pdf]
    selected_pages = {int(row["page"]) for row in selected}
    selected_score_by_page = {int(row["page"]): int(row["score"]) for row in selected}

    if settings.cleanup_low_text_rescue and selected and len(selected) < settings.cleanup_max_pages_per_pdf:
        rescue_candidates = []
        for row in scored_rows:
            page = int(row["page"])
            if page <= 0 or page in selected_pages:
                continue
            char_count = int(row.get("scoring", {}).get("char_count", 0))
            if char_count > settings.cleanup_low_text_max_chars:
                continue
            adjacent_scores = [
                selected_score_by_page[selected_page]
                for selected_page in selected_pages
                if abs(page - selected_page) == 1
            ]
            if adjacent_scores:
                rescue_candidates.append(
                    {
                        **row,
                        "adjacent_count": len(adjacent_scores),
                        "adjacent_max_score": max(adjacent_scores),
                    }
                )

        rescue_candidates.sort(
            key=lambda row: (
                -int(row.get("adjacent_count", 0)),
                -int(row.get("adjacent_max_score", 0)),
                -int(row["score"]),
                int(row["page"]),
            )
        )
        if rescue_candidates:
            selected.append(rescue_candidates[0])

    selected.sort(key=lambda row: int(row["page"]))
    return selected


def _validate_cleanup_fidelity(
    *,
    original_text: str,
    cleaned_text: str,
    settings: ParseSettings | None = None,
) -> dict[str, Any]:
    length_ratio_max = (
        settings.cleanup_length_ratio_max
        if settings is not None
        else CLEANUP_DEFAULT_LENGTH_RATIO_MAX
    )
    numeric_guardrail_enabled = (
        bool(settings.cleanup_numeric_guardrail_enabled) if settings is not None else True
    )
    original = str(original_text or "")
    cleaned = str(cleaned_text or "")
    cleaned_stripped = cleaned.strip()
    checks: dict[str, Any] = {
        "accepted": False,
        "reason": "unknown",
        "numeric_token_retention": 0.0,
        "original_token_coverage": 0.0,
        "length_ratio": 0.0,
        "non_empty": bool(cleaned_stripped),
        "parseable": ("\x00" not in cleaned),
        "numeric_guardrail_enabled": numeric_guardrail_enabled,
        "thresholds": {
            "numeric_token_retention_min": 0.95,
            "original_token_coverage_min": 0.75,
            "length_ratio_min": CLEANUP_DEFAULT_LENGTH_RATIO_MIN,
            "length_ratio_max": length_ratio_max,
        },
        "numeric_tokens_ignored": [],
    }

    if not checks["non_empty"]:
        checks["reason"] = "empty_cleaned_text"
        return checks
    if not checks["parseable"]:
        checks["reason"] = "invalid_characters"
        return checks

    original_numeric, ignored_numeric = _numeric_tokens_for_fidelity(original_text=original)
    cleaned_numeric_set = set(_numeric_tokens(cleaned))
    checks["numeric_tokens_ignored"] = ignored_numeric
    if not original_numeric:
        numeric_retention = 1.0
    else:
        numeric_retention = sum(1 for token in original_numeric if token in cleaned_numeric_set) / len(
            original_numeric
        )
    checks["numeric_token_retention"] = numeric_retention

    original_tokens = _word_tokens(original)
    cleaned_token_set = set(_word_tokens(cleaned))
    if not original_tokens:
        token_coverage = 1.0
    else:
        token_coverage = sum(1 for token in original_tokens if token in cleaned_token_set) / len(
            original_tokens
        )
    checks["original_token_coverage"] = token_coverage

    original_len = len(original.strip())
    cleaned_len = len(cleaned_stripped)
    if original_len == 0:
        length_ratio = 1.0 if cleaned_len > 0 else 0.0
    else:
        length_ratio = cleaned_len / original_len
    checks["length_ratio"] = length_ratio

    thresholds = checks["thresholds"]
    if numeric_guardrail_enabled and numeric_retention < thresholds["numeric_token_retention_min"]:
        checks["reason"] = "numeric_token_retention_below_threshold"
        return checks
    if token_coverage < thresholds["original_token_coverage_min"]:
        checks["reason"] = "original_token_coverage_below_threshold"
        return checks
    if length_ratio < thresholds["length_ratio_min"] or length_ratio > thresholds["length_ratio_max"]:
        checks["reason"] = "length_ratio_out_of_bounds"
        return checks

    checks["accepted"] = True
    checks["reason"] = (
        "accepted_numeric_guardrail_disabled" if not numeric_guardrail_enabled else "accepted"
    )
    return checks


def _cleanup_page_cache_model_token(model: str) -> str:
    token = re.sub(r"[^a-z0-9._-]+", "-", str(model).strip().lower()).strip("-")
    return token or "unknown-model"


def _cleanup_page_cache_key(
    *,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_no: int,
    original_text: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": CLEANUP_PAGE_CACHE_SCHEMA_VERSION,
        "source_sha256": scoped_pdf.source_sha256,
        "source_relative_path": scoped_pdf.source_relative_path,
        "page": int(page_no),
        "cleanup_model": settings.cleanup_model,
        "image_dpi": int(settings.cleanup_image_dpi),
        "original_text_sha256": _sha256_text(original_text),
        "system_prompt_sha256": _sha256_text(CLEANUP_SYSTEM_PROMPT),
    }
    digest = _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return digest, payload


def _cleanup_page_cache_path(
    *,
    page_cache_root: Path,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_no: int,
    cache_key_sha256: str,
) -> Path:
    model_token = _cleanup_page_cache_model_token(settings.cleanup_model)
    return (
        page_cache_root
        / CLEANUP_PAGE_CACHE_SCHEMA_VERSION
        / model_token
        / scoped_pdf.source_sha256
        / f"{int(page_no):04d}-{cache_key_sha256}.json"
    )


def _load_cleanup_page_cache_entry(
    *,
    page_cache_root: Path | None,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_no: int,
    cache_key_sha256: str,
) -> dict[str, Any] | None:
    if page_cache_root is None:
        return None
    path = _cleanup_page_cache_path(
        page_cache_root=page_cache_root,
        settings=settings,
        scoped_pdf=scoped_pdf,
        page_no=page_no,
        cache_key_sha256=cache_key_sha256,
    )
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("cache_key_sha256", "")) != cache_key_sha256:
        return None
    cleaned_text = payload.get("cleaned_text")
    if not isinstance(cleaned_text, str):
        return None
    notes_raw = payload.get("notes")
    notes = notes_raw if isinstance(notes_raw, str) else None
    return {
        "cleaned_text": cleaned_text,
        "notes": notes,
        "cache_path": str(path),
        "generated_at_utc": payload.get("generated_at_utc"),
    }


def _write_cleanup_page_cache_entry(
    *,
    page_cache_root: Path | None,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_no: int,
    cache_key_sha256: str,
    cache_key_payload: dict[str, Any],
    cleaned_text: str,
    notes: str | None,
) -> str | None:
    if page_cache_root is None:
        return None
    path = _cleanup_page_cache_path(
        page_cache_root=page_cache_root,
        settings=settings,
        scoped_pdf=scoped_pdf,
        page_no=page_no,
        cache_key_sha256=cache_key_sha256,
    )
    ensure_dir(path.parent)
    payload = {
        "schema_version": CLEANUP_PAGE_CACHE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cache_key_sha256": cache_key_sha256,
        "cache_key_payload": cache_key_payload,
        "source_relative_path": scoped_pdf.source_relative_path,
        "source_sha256": scoped_pdf.source_sha256,
        "page": int(page_no),
        "cleanup_model": settings.cleanup_model,
        "image_dpi": int(settings.cleanup_image_dpi),
        "cleaned_text": str(cleaned_text),
        "notes": notes,
    }
    path.write_text(canonical_json_dumps(payload), encoding="utf-8")
    return str(path)


def _call_cleanup_llm(
    *,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_no: int,
    original_text: str,
    image_data_url: str,
) -> CleanupResponse:
    from openai import OpenAI as OpenAIClient

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY")).with_options(
        timeout=float(settings.cleanup_timeout_sec)
    )
    user_prompt = (
        "Clean/transcribe the page text faithfully from the image and existing extracted text.\n"
        "Rules:\n"
        "- Preserve facts, numbers, years, units, and meaning.\n"
        "- Keep text in the same language.\n"
        "- Do not summarize and do not infer missing content.\n"
        f"- Source: {scoped_pdf.source_relative_path} page {page_no}.\n\n"
        "Existing extracted page text:\n"
        f"{original_text}"
    )
    response = client.responses.parse(
        model=settings.cleanup_model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": CLEANUP_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
        text_format=CleanupResponse,
        reasoning={"effort": "minimal"},
        max_output_tokens=2048,
    )

    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, CleanupResponse):
        return parsed
    if parsed is not None:
        return CleanupResponse.model_validate(parsed)

    output_text = str(getattr(response, "output_text", "") or "").strip()
    if not output_text:
        raise RuntimeError("Cleanup model returned empty output.")
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(output_text[start : end + 1])
        else:
            raise RuntimeError("Cleanup model output is not valid JSON.")
    return CleanupResponse.model_validate(payload)


def _cleanup_page_rows_with_llm(
    *,
    scoped_pdf: ScopedPdf,
    page_rows: list[dict[str, Any]],
    settings: ParseSettings,
    page_cache_root: Path | None = None,
) -> tuple[list[dict[str, Any]], CleanupRunArtifacts]:
    default_summary = {
        "mode": settings.cleanup_mode,
        "profile_version": settings.cleanup_profile_version,
        "numeric_guardrail_enabled": settings.cleanup_numeric_guardrail_enabled,
        "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
        "enabled_for_doc": False,
        "selected_pages": [],
        "attempted_pages": 0,
        "accepted_pages": 0,
        "rejected_pages": 0,
        "failed_pages": 0,
        "page_cache_hits": 0,
        "page_cache_misses": 0,
        "page_cache_write_errors": 0,
        "model": settings.cleanup_model if settings.cleanup_mode != "off" else None,
    }
    if not _cleanup_enabled_for_doc(settings=settings, scoped_pdf=scoped_pdf):
        return list(page_rows), CleanupRunArtifacts(
            summary=default_summary,
            audit_rows=[],
            request_rows=[],
            output_rows=[],
        )

    updated_by_page = {int(row.get("page", 0)): dict(row) for row in page_rows}
    candidates = _select_cleanup_candidates(page_rows=page_rows, settings=settings)
    selected_pages = [int(row["page"]) for row in candidates]

    attempts = 0
    accepted = 0
    rejected = 0
    failed = 0
    page_cache_hits = 0
    page_cache_misses = 0
    page_cache_write_errors = 0
    audit_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    for candidate in candidates:
        page_no = int(candidate["page"])
        original_text = str(candidate.get("text_markdown", ""))
        cache_key_sha256, cache_key_payload = _cleanup_page_cache_key(
            settings=settings,
            scoped_pdf=scoped_pdf,
            page_no=page_no,
            original_text=original_text,
        )
        request_row = {
            "page": page_no,
            "model": settings.cleanup_model,
            "timeout_sec": settings.cleanup_timeout_sec,
            "image_dpi": settings.cleanup_image_dpi,
            "input_text_sha256": _sha256_text(original_text),
            "input_chars": len(original_text),
            "page_cache_key_sha256": cache_key_sha256,
        }
        request_rows.append(request_row)

        attempts += 1
        started = time.perf_counter()
        decision = "failed"
        error_message: str | None = None
        cleaned_text = original_text
        checks = {
            "accepted": False,
            "reason": "cleanup_not_attempted",
            "numeric_token_retention": 0.0,
            "original_token_coverage": 0.0,
            "length_ratio": 0.0,
            "numeric_guardrail_enabled": settings.cleanup_numeric_guardrail_enabled,
        }
        notes: str | None = None
        page_cache_status = "disabled" if page_cache_root is None else "miss"
        page_cache_path: str | None = None
        llm_invoked = False

        cached_page = _load_cleanup_page_cache_entry(
            page_cache_root=page_cache_root,
            settings=settings,
            scoped_pdf=scoped_pdf,
            page_no=page_no,
            cache_key_sha256=cache_key_sha256,
        )
        if cached_page is not None:
            page_cache_status = "hit"
            page_cache_hits += 1
            page_cache_path = str(cached_page.get("cache_path", "") or "")
            notes = cached_page.get("notes")
            cleaned_text = str(cached_page.get("cleaned_text", ""))
            checks = _validate_cleanup_fidelity(
                original_text=original_text,
                cleaned_text=cleaned_text,
                settings=settings,
            )
            if checks.get("accepted"):
                decision = "accepted"
                accepted += 1
                target_row = updated_by_page.get(page_no, {"page": page_no, "text_markdown": ""})
                target_row["text_markdown"] = cleaned_text
                updated_by_page[page_no] = target_row
            else:
                decision = "rejected_guardrail"
                rejected += 1
        else:
            if page_cache_root is not None:
                page_cache_misses += 1
            miss_policy = settings.cleanup_page_cache_miss_policy
            if miss_policy == "error":
                raise RuntimeError(
                    "Cleanup page cache miss under policy=error: "
                    f"{scoped_pdf.source_relative_path} page {page_no}"
                )
            if miss_policy == "fail_open":
                failed += 1
                decision = "failed_open_page_cache_miss"
                checks["reason"] = "page_cache_miss_fail_open"
                error_message = "Cleanup page cache miss under fail_open policy."
            else:
                try:
                    if not openai_key:
                        failed += 1
                        decision = "failed_open_missing_openai_api_key"
                        checks["reason"] = "missing_openai_api_key"
                        error_message = "Missing OPENAI_API_KEY."
                    else:
                        image_data_url = _render_page_image_data_url(
                            scoped_pdf=scoped_pdf,
                            page_no=page_no,
                            image_dpi=settings.cleanup_image_dpi,
                        )
                        llm_invoked = True
                        llm_output = _call_cleanup_llm(
                            settings=settings,
                            scoped_pdf=scoped_pdf,
                            page_no=page_no,
                            original_text=original_text,
                            image_data_url=image_data_url,
                        )
                        notes = llm_output.notes
                        cleaned_text = llm_output.cleaned_text
                        try:
                            page_cache_path = _write_cleanup_page_cache_entry(
                                page_cache_root=page_cache_root,
                                settings=settings,
                                scoped_pdf=scoped_pdf,
                                page_no=page_no,
                                cache_key_sha256=cache_key_sha256,
                                cache_key_payload=cache_key_payload,
                                cleaned_text=cleaned_text,
                                notes=notes,
                            )
                        except Exception:
                            page_cache_write_errors += 1
                            page_cache_status = "write_error"

                        checks = _validate_cleanup_fidelity(
                            original_text=original_text,
                            cleaned_text=cleaned_text,
                            settings=settings,
                        )
                        if checks.get("accepted"):
                            decision = "accepted"
                            accepted += 1
                            target_row = updated_by_page.get(page_no, {"page": page_no, "text_markdown": ""})
                            target_row["text_markdown"] = cleaned_text
                            updated_by_page[page_no] = target_row
                        else:
                            decision = "rejected_guardrail"
                            rejected += 1
                except Exception as exc:  # pragma: no cover - network/runtime dependent
                    failed += 1
                    decision = "failed_open_runtime_error"
                    error_message = f"{type(exc).__name__}: {exc}"
                    checks["reason"] = "cleanup_runtime_error"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        audit_rows.append(
            {
                "page": page_no,
                "selection_score": int(candidate["score"]),
                "selection_details": candidate.get("scoring", {}),
                "decision": decision,
                "accepted": decision == "accepted",
                "error": error_message,
                "checks": checks,
                "llm_invoked": llm_invoked,
                "page_cache_status": page_cache_status,
                "page_cache_path": page_cache_path,
                "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
                "model": settings.cleanup_model,
                "elapsed_ms": elapsed_ms,
                "notes": notes,
                "original_chars": len(original_text),
                "cleaned_chars": len(cleaned_text),
                "original_preview": original_text[:CLEANUP_DEFAULT_PREVIEW_CHARS],
                "cleaned_preview": cleaned_text[:CLEANUP_DEFAULT_PREVIEW_CHARS],
            }
        )
        output_rows.append(
            {
                "page": page_no,
                "decision": decision,
                "accepted": decision == "accepted",
                "original_text_sha256": _sha256_text(original_text),
                "cleaned_text_sha256": _sha256_text(cleaned_text),
                "llm_invoked": llm_invoked,
                "page_cache_status": page_cache_status,
                "page_cache_path": page_cache_path,
                "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
            }
        )

    summary = {
        "mode": settings.cleanup_mode,
        "profile_version": settings.cleanup_profile_version,
        "numeric_guardrail_enabled": settings.cleanup_numeric_guardrail_enabled,
        "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
        "enabled_for_doc": True,
        "selected_pages": selected_pages,
        "attempted_pages": attempts,
        "accepted_pages": accepted,
        "rejected_pages": rejected,
        "failed_pages": failed,
        "page_cache_hits": page_cache_hits,
        "page_cache_misses": page_cache_misses,
        "page_cache_write_errors": page_cache_write_errors,
        "model": settings.cleanup_model,
    }
    updated_rows = [updated_by_page[key] for key in sorted(updated_by_page)]
    return updated_rows, CleanupRunArtifacts(
        summary=summary,
        audit_rows=audit_rows,
        request_rows=request_rows,
        output_rows=output_rows,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _estimated_cleanup_calls_for_item(*, settings: ParseSettings, scoped_pdf: ScopedPdf) -> int:
    if not _cleanup_enabled_for_doc(settings=settings, scoped_pdf=scoped_pdf):
        return 0
    return min(settings.cleanup_max_pages_per_pdf, max(int(scoped_pdf.page_count), 0))


def _validate_cache_manifest(
    *,
    manifest: dict[str, Any],
    scoped_pdf: ScopedPdf,
    settings: ParseSettings,
    entry_dir: Path,
) -> tuple[bool, str]:
    expected_profile_key = settings.profile_key()
    if str(manifest.get("provider", "")).strip().lower() != settings.provider:
        return False, "provider_mismatch"
    if str(manifest.get("profile_key", "")).strip().lower() != expected_profile_key:
        return False, "profile_mismatch"
    if str(manifest.get("parser_settings_version", "")).strip() != settings.settings_version:
        return False, "settings_version_mismatch"
    if str(manifest.get("source_sha256", "")).strip().lower() != scoped_pdf.source_sha256:
        return False, "source_sha_mismatch"
    if str(manifest.get("source_relative_path", "")) != scoped_pdf.source_relative_path:
        return False, "source_relative_path_mismatch"

    pages_rel = str(manifest.get("pages_path", "pages.jsonl"))
    raw_rel = str(manifest.get("raw_response_path", "raw_response.json.gz"))
    pages_path = entry_dir / pages_rel
    raw_path = entry_dir / raw_rel
    if not pages_path.exists():
        return False, "missing_pages_file"
    if not raw_path.exists():
        return False, "missing_raw_file"
    if not str(manifest.get("content_sha256", "")).strip():
        return False, "missing_content_sha"

    return True, "ok"


def _build_cache_plan(
    *,
    scoped_pdfs: list[ScopedPdf],
    parsed_docs_root: Path,
    settings: ParseSettings,
) -> list[CachePlanItem]:
    planned: list[CachePlanItem] = []
    for scoped_pdf in scoped_pdfs:
        entry_dir = cache_entry_dir(parsed_docs_root, settings, scoped_pdf.source_sha256)
        manifest_path = _manifest_path(entry_dir)
        if not manifest_path.exists():
            planned.append(
                CachePlanItem(
                    scoped_pdf=scoped_pdf,
                    status="missing",
                    reason="manifest_not_found",
                    entry_dir=entry_dir,
                    manifest_path=manifest_path,
                    manifest=None,
                )
            )
            continue

        try:
            manifest = _read_json(manifest_path)
        except Exception:
            planned.append(
                CachePlanItem(
                    scoped_pdf=scoped_pdf,
                    status="invalid",
                    reason="manifest_invalid_json",
                    entry_dir=entry_dir,
                    manifest_path=manifest_path,
                    manifest=None,
                )
            )
            continue

        ok, reason = _validate_cache_manifest(
            manifest=manifest,
            scoped_pdf=scoped_pdf,
            settings=settings,
            entry_dir=entry_dir,
        )
        planned.append(
            CachePlanItem(
                scoped_pdf=scoped_pdf,
                status="hit" if ok else "invalid",
                reason=reason,
                entry_dir=entry_dir,
                manifest_path=manifest_path,
                manifest=manifest if ok else None,
            )
        )

    return planned


def _credits_from_usage(
    *,
    job_pages: int,
    auto_mode_triggered_pages: int,
    settings: ParseSettings,
) -> CreditBreakdown:
    job_pages = max(0, int(job_pages))
    premium_pages = max(0, min(job_pages, int(auto_mode_triggered_pages)))
    standard_pages = max(0, job_pages - premium_pages)
    estimated_credits = (
        standard_pages * settings.estimate_standard_credits_per_page
        + premium_pages * settings.estimate_premium_credits_per_page
    )
    return CreditBreakdown(
        standard_pages=standard_pages,
        premium_pages=premium_pages,
        estimated_credits=estimated_credits,
    )


def _default_parse_command(
    *,
    config_path: Path | None,
    company_tickers: list[str] | None = None,
    years: list[str] | None = None,
) -> str:
    config_token = str(config_path) if config_path is not None else "<config.toml>"
    command = [
        "uv run cte parse-cache build",
        f"--config {config_token}",
        "--execute",
    ]
    if company_tickers:
        command.append(f"--company-tickers {','.join(company_tickers)}")
    if years:
        command.append(f"--years {','.join(years)}")
    return " ".join(command)


def _source_relative_path(
    *,
    source_docs_root: Path,
    source_dir: Path,
    relative_path: str,
) -> str:
    prefix = source_dir.relative_to(source_docs_root)
    return str((prefix / relative_path).as_posix())


def build_cache_content_fingerprint(
    *,
    source_docs_root: Path,
    source_dir: Path,
    source_manifest: list[SourceFileInfo],
    parsed_docs_root: Path,
    settings: ParseSettings,
    config_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for source_file in source_manifest:
        if not source_file.relative_path.lower().endswith(".pdf"):
            continue
        rel_path = _source_relative_path(
            source_docs_root=source_docs_root,
            source_dir=source_dir,
            relative_path=source_file.relative_path,
        )
        entry_dir = cache_entry_dir(parsed_docs_root, settings, source_file.sha256)
        manifest_path = _manifest_path(entry_dir)
        if not manifest_path.exists():
            missing.append(rel_path)
            continue

        try:
            manifest = _read_json(manifest_path)
        except Exception:
            missing.append(rel_path)
            continue

        scoped_pdf = ScopedPdf(
            ticker=Path(rel_path).parts[0],
            year=Path(rel_path).parts[1],
            absolute_path=source_docs_root / rel_path,
            source_relative_path=rel_path,
            source_file_name=Path(source_file.relative_path).name,
            source_sha256=source_file.sha256,
            source_size_bytes=source_file.size_bytes,
            page_count=0,
        )
        ok, _reason = _validate_cache_manifest(
            manifest=manifest,
            scoped_pdf=scoped_pdf,
            settings=settings,
            entry_dir=entry_dir,
        )
        if not ok:
            missing.append(rel_path)
            continue

        rows.append(
            {
                "source_relative_path": rel_path,
                "source_sha256": source_file.sha256,
                "cache_content_sha256": str(manifest.get("content_sha256", "")),
                "provider": settings.provider,
                "profile_key": settings.profile_key(),
                "settings_version": settings.settings_version,
            }
        )

    rows.sort(key=lambda row: row["source_relative_path"])

    if missing:
        cmd = _default_parse_command(config_path=config_path)
        message = (
            "Missing parse-cache entries for cache_only mode. "
            f"Build cache first: {cmd}. Missing: {', '.join(sorted(missing))}"
        )
        raise ParseCacheMissingError(message, missing_paths=sorted(missing))

    fingerprint = _sha256_text(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    details = {
        "provider": settings.provider,
        "profile_key": settings.profile_key(),
        "settings_version": settings.settings_version,
        "parsed_docs_root": str(parsed_docs_root),
        "entries": rows,
    }
    return fingerprint, details


def load_cached_pages_for_source_manifest(
    *,
    source_docs_root: Path,
    source_dir: Path,
    source_manifest: list[SourceFileInfo],
    parsed_docs_root: Path,
    settings: ParseSettings,
    config_path: Path | None = None,
) -> list[CachedPageRow]:
    rows: list[CachedPageRow] = []
    missing: list[str] = []

    for source_file in source_manifest:
        if not source_file.relative_path.lower().endswith(".pdf"):
            continue

        rel_path = _source_relative_path(
            source_docs_root=source_docs_root,
            source_dir=source_dir,
            relative_path=source_file.relative_path,
        )
        entry_dir = cache_entry_dir(parsed_docs_root, settings, source_file.sha256)
        manifest_path = _manifest_path(entry_dir)
        if not manifest_path.exists():
            missing.append(rel_path)
            continue

        try:
            manifest = _read_json(manifest_path)
        except Exception:
            missing.append(rel_path)
            continue

        scoped_pdf = ScopedPdf(
            ticker=Path(rel_path).parts[0],
            year=Path(rel_path).parts[1],
            absolute_path=source_docs_root / rel_path,
            source_relative_path=rel_path,
            source_file_name=Path(source_file.relative_path).name,
            source_sha256=source_file.sha256,
            source_size_bytes=source_file.size_bytes,
            page_count=0,
        )
        ok, _ = _validate_cache_manifest(
            manifest=manifest,
            scoped_pdf=scoped_pdf,
            settings=settings,
            entry_dir=entry_dir,
        )
        if not ok:
            missing.append(rel_path)
            continue

        pages_path = entry_dir / str(manifest.get("pages_path", "pages.jsonl"))
        total_pages = int(manifest.get("page_count", 0))
        page_rows: list[dict[str, Any]] = []
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            page_rows.append(json.loads(line))
        page_rows.sort(key=lambda row: int(row.get("page", 0)))

        for row in page_rows:
            page = int(row.get("page", 0))
            text_markdown = str(row.get("text_markdown", ""))
            rows.append(
                CachedPageRow(
                    file_name=Path(source_file.relative_path).name,
                    source_relative_path=rel_path,
                    page=page,
                    total_pages=total_pages,
                    text_markdown=text_markdown,
                )
            )

    if missing:
        cmd = _default_parse_command(config_path=config_path)
        message = (
            "Missing parse-cache entries for cache_only mode. "
            f"Build cache first: {cmd}. Missing: {', '.join(sorted(missing))}"
        )
        raise ParseCacheMissingError(message, missing_paths=sorted(missing))

    rows.sort(key=lambda row: (row.source_relative_path, row.page))
    return rows


def _llamaparse_kwargs(settings: ParseSettings) -> dict[str, Any]:
    disable_ocr = settings.ocr_mode == "off"
    # LlamaParse only allows one primary parsing mode at a time.
    # In our default HQ profile, auto_mode is the intended cost optimizer path.
    premium_mode = settings.premium_mode and not settings.auto_mode
    kwargs: dict[str, Any] = {
        "result_type": settings.result_type,
        "language": settings.language,
        "premium_mode": premium_mode,
        "extract_layout": settings.extract_layout,
        "adaptive_long_table": settings.adaptive_long_table,
        "high_res_ocr": settings.high_res_ocr,
        "auto_mode": settings.auto_mode,
        "auto_mode_trigger_on_table_in_page": settings.auto_trigger_tables,
        "auto_mode_trigger_on_image_in_page": settings.auto_trigger_images,
        "auto_mode_trigger_on_regexp_in_page": settings.auto_trigger_regex,
        "disable_ocr": disable_ocr,
        "split_by_page": True,
        "verbose": False,
        "show_progress": False,
        "ignore_errors": False,
    }
    return kwargs


def _parse_with_llamaparse(scoped_pdf: ScopedPdf, settings: ParseSettings) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        from llama_parse import LlamaParse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LlamaParse is required for execute mode. Install with `uv sync --extra rag` and use Python 3.12 worker mode."
        ) from exc

    api_key = os.getenv("LLAMA_CLOUD_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing LLAMA_CLOUD_API_KEY for execute mode.")

    parser = LlamaParse(api_key=api_key, **_llamaparse_kwargs(settings))
    json_results = parser.get_json_result(str(scoped_pdf.absolute_path))
    if not json_results:
        raise RuntimeError(f"No parse results returned for {scoped_pdf.source_relative_path}")

    if len(json_results) > 1:
        # Should not happen for default settings, but merge deterministically if partitions occur.
        merged_pages: list[dict[str, Any]] = []
        merged_job_pages = 0
        merged_auto_pages = 0
        cache_hit = False
        merged_payload: dict[str, Any] = {
            "file_path": str(scoped_pdf.absolute_path),
            "job_results": json_results,
            "pages": [],
            "job_metadata": {},
        }
        for item in json_results:
            pages = item.get("pages", []) if isinstance(item, dict) else []
            merged_pages.extend(pages)
            metadata = item.get("job_metadata", {}) if isinstance(item, dict) else {}
            merged_job_pages += int(metadata.get("job_pages", 0) or 0)
            merged_auto_pages += int(metadata.get("job_auto_mode_triggered_pages", 0) or 0)
            cache_hit = cache_hit or bool(metadata.get("job_is_cache_hit", False))
        merged_payload["pages"] = merged_pages
        merged_payload["job_metadata"] = {
            "job_pages": merged_job_pages,
            "job_auto_mode_triggered_pages": merged_auto_pages,
            "job_is_cache_hit": cache_hit,
        }
        payload = merged_payload
    else:
        payload = dict(json_results[0])

    pages_payload = payload.get("pages", []) if isinstance(payload, dict) else []
    page_rows: list[dict[str, Any]] = []
    for fallback_page, page in enumerate(pages_payload, start=1):
        page_no = int(page.get("page", fallback_page)) if isinstance(page, dict) else fallback_page
        text_markdown = ""
        if isinstance(page, dict):
            text_markdown = str(page.get("md") or page.get("text") or "")
        page_rows.append({"page": page_no, "text_markdown": text_markdown})
    page_rows.sort(key=lambda row: row["page"])

    job_metadata = payload.get("job_metadata", {}) if isinstance(payload, dict) else {}
    return page_rows, payload, job_metadata if isinstance(job_metadata, dict) else {}


def _write_cache_entry(
    *,
    parsed_docs_root: Path,
    settings: ParseSettings,
    scoped_pdf: ScopedPdf,
    page_rows: list[dict[str, Any]],
    raw_payload: dict[str, Any],
    job_metadata: dict[str, Any],
    cleanup_artifacts: CleanupRunArtifacts | None = None,
) -> dict[str, Any]:
    entry_dir = ensure_dir(cache_entry_dir(parsed_docs_root, settings, scoped_pdf.source_sha256))

    sorted_rows = sorted(page_rows, key=lambda row: int(row.get("page", 0)))

    pages_path = _pages_path(entry_dir)
    with pages_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            payload = {
                "page": int(row.get("page", 0)),
                "text_markdown": str(row.get("text_markdown", "")),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    raw_path = _raw_path(entry_dir)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(raw_payload))

    cleanup_summary = (
        cleanup_artifacts.summary
        if cleanup_artifacts is not None
        else {
            "mode": settings.cleanup_mode,
            "profile_version": settings.cleanup_profile_version,
            "numeric_guardrail_enabled": settings.cleanup_numeric_guardrail_enabled,
            "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
            "enabled_for_doc": False,
            "selected_pages": [],
            "attempted_pages": 0,
            "accepted_pages": 0,
            "rejected_pages": 0,
            "failed_pages": 0,
            "page_cache_hits": 0,
            "page_cache_misses": 0,
            "page_cache_write_errors": 0,
            "model": settings.cleanup_model if settings.cleanup_mode != "off" else None,
        }
    )
    if cleanup_artifacts is not None:
        _write_jsonl(_cleanup_audit_path(entry_dir), cleanup_artifacts.audit_rows)
        _write_jsonl(_cleanup_requests_path(entry_dir), cleanup_artifacts.request_rows)
        _write_jsonl(_cleanup_outputs_path(entry_dir), cleanup_artifacts.output_rows)

    cleanup_audit_rel = _cleanup_audit_path(entry_dir).name
    cleanup_requests_rel = _cleanup_requests_path(entry_dir).name
    cleanup_outputs_rel = _cleanup_outputs_path(entry_dir).name
    cleanup_audit_exists = (_cleanup_audit_path(entry_dir)).exists()
    cleanup_requests_exists = (_cleanup_requests_path(entry_dir)).exists()
    cleanup_outputs_exists = (_cleanup_outputs_path(entry_dir)).exists()

    content_payload = json.dumps(sorted_rows, ensure_ascii=False, sort_keys=True)
    content_sha256 = _sha256_text(content_payload)

    page_count = len(sorted_rows)
    job_pages = int(job_metadata.get("job_pages", page_count) or page_count)
    auto_pages = int(job_metadata.get("job_auto_mode_triggered_pages", 0) or 0)
    job_is_cache_hit = bool(job_metadata.get("job_is_cache_hit", False))

    manifest = {
        "provider": settings.provider,
        "profile_key": settings.profile_key(),
        "parser_settings": settings.provider_payload(),
        "parser_settings_version": settings.settings_version,
        "source_relative_path": scoped_pdf.source_relative_path,
        "source_file_name": scoped_pdf.source_file_name,
        "source_sha256": scoped_pdf.source_sha256,
        "source_size_bytes": scoped_pdf.source_size_bytes,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "page_count": page_count,
        "job_pages": job_pages,
        "job_auto_mode_triggered_pages": auto_pages,
        "job_is_cache_hit": job_is_cache_hit,
        "content_sha256": content_sha256,
        "pages_path": pages_path.name,
        "raw_response_path": raw_path.name,
        "cleanup_mode": cleanup_summary.get("mode"),
        "cleanup_profile_version": cleanup_summary.get("profile_version"),
        "cleanup_numeric_guardrail_enabled": bool(
            cleanup_summary.get(
                "numeric_guardrail_enabled", settings.cleanup_numeric_guardrail_enabled
            )
        ),
        "cleanup_page_cache_miss_policy": cleanup_summary.get(
            "page_cache_miss_policy", settings.cleanup_page_cache_miss_policy
        ),
        "cleanup_enabled_for_doc": bool(cleanup_summary.get("enabled_for_doc", False)),
        "cleanup_selected_pages": cleanup_summary.get("selected_pages", []),
        "cleanup_attempted_pages": int(cleanup_summary.get("attempted_pages", 0) or 0),
        "cleanup_accepted_pages": int(cleanup_summary.get("accepted_pages", 0) or 0),
        "cleanup_rejected_pages": int(cleanup_summary.get("rejected_pages", 0) or 0),
        "cleanup_failed_pages": int(cleanup_summary.get("failed_pages", 0) or 0),
        "cleanup_page_cache_hits": int(cleanup_summary.get("page_cache_hits", 0) or 0),
        "cleanup_page_cache_misses": int(cleanup_summary.get("page_cache_misses", 0) or 0),
        "cleanup_page_cache_write_errors": int(
            cleanup_summary.get("page_cache_write_errors", 0) or 0
        ),
        "cleanup_model": cleanup_summary.get("model"),
        "cleanup_audit_path": cleanup_audit_rel if cleanup_audit_exists else None,
        "cleanup_requests_path": cleanup_requests_rel if cleanup_requests_exists else None,
        "cleanup_outputs_path": cleanup_outputs_rel if cleanup_outputs_exists else None,
        "status": "success",
    }
    _manifest_path(entry_dir).write_text(canonical_json_dumps(manifest), encoding="utf-8")
    return manifest


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _delegate_execute_to_py312(args: Any) -> int:
    command: list[str] = [
        "uv",
        "run",
        "--python",
        "3.12",
        "--extra",
        "rag",
        "cte",
        "parse-cache",
        "build",
        "--config",
        str(args.config),
        "--execute",
        "--worker",
    ]
    if args.company_tickers:
        command.extend(["--company-tickers", str(args.company_tickers)])
    if args.years:
        command.extend(["--years", str(args.years)])
    if args.max_new_pdfs is not None:
        command.extend(["--max-new-pdfs", str(args.max_new_pdfs)])
    if getattr(args, "quiet", False):
        command.append("--quiet")

    print("Delegating execute mode to Python 3.12 worker:")
    print(" ".join(command))

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to launch Python 3.12 worker via uv. Ensure `uv` is installed and available."
        ) from exc

    return int(completed.returncode)


def run_parse_cache_build(args: Any) -> int:
    config_path = Path(args.config)
    config = load_run_config(config_path)
    maybe_load_env_file(config.env_file)
    settings = parse_settings_from_component_settings(config.component_settings)

    if settings.provider != "llamaparse":
        raise ValueError(
            f"Unsupported pdf_parse_provider='{settings.provider}'. Supported: llamaparse"
        )

    if args.execute and not getattr(args, "worker", False) and sys.version_info >= (3, 14):
        return _delegate_execute_to_py312(args)

    if args.execute and not os.getenv("LLAMA_CLOUD_API_KEY", "").strip():
        raise EnvironmentError("Missing LLAMA_CLOUD_API_KEY for execute mode.")

    source_docs_root = resolve_source_docs_root(config)
    parsed_docs_root = config.parsed_docs_root
    if not parsed_docs_root.is_absolute():
        parsed_docs_root = (Path.cwd() / parsed_docs_root).resolve()

    tickers = _parse_csv(args.company_tickers) or list(config.company_tickers)
    tickers = [ticker.upper() for ticker in tickers]
    years = _parse_csv(args.years) or list(config.years)

    scoped_pdfs = _build_scope(
        source_docs_root=source_docs_root,
        company_tickers=tickers,
        years=years,
    )
    plan = _build_cache_plan(
        scoped_pdfs=scoped_pdfs,
        parsed_docs_root=parsed_docs_root,
        settings=settings,
    )

    work_items = [row for row in plan if row.status != "hit"]
    if args.execute and args.max_new_pdfs is not None and len(work_items) > args.max_new_pdfs:
        raise RuntimeError(
            f"Planned new parses ({len(work_items)}) exceed --max-new-pdfs ({args.max_new_pdfs})."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-parse_cache_build"
    run_dir = ensure_dir(parsed_docs_root / "_runs" / run_id)
    live_status_path = run_dir / "live_status.json"
    live_status = LiveStatusTracker(
        path=live_status_path,
        job_kind="parse_cache_build",
        run_id=run_id,
        initial={
            "mode": "execute" if args.execute else "dry_run",
            "config_path": str(config_path),
            "source_docs_root": str(source_docs_root),
            "parsed_docs_root": str(parsed_docs_root),
            "company_tickers": tickers,
            "years": years,
            "pdf_total": len(plan),
            "stage": "build",
            "processed": 0,
            "total": len(plan),
            "current_source_relative_path": None,
            "hits": 0,
            "planned_new": 0,
            "parsed": 0,
            "failed": 0,
            "cleanup_attempted_pages": 0,
            "cleanup_accepted_pages": 0,
            "cleanup_rejected_pages": 0,
            "cleanup_failed_pages": 0,
            "cleanup_page_cache_hits": 0,
            "cleanup_page_cache_misses": 0,
            "cleanup_page_cache_write_errors": 0,
        },
    )

    failures: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    processed_count = 0
    live_hits = 0
    live_planned_new = 0
    live_parsed = 0
    live_failed = 0

    total_standard_pages = 0
    total_premium_pages = 0
    total_estimated_credits = 0.0
    total_cleanup_estimated_calls = 0
    total_cleanup_attempted_pages = 0
    total_cleanup_accepted_pages = 0
    total_cleanup_rejected_pages = 0
    total_cleanup_failed_pages = 0
    total_cleanup_page_cache_hits = 0
    total_cleanup_page_cache_misses = 0
    total_cleanup_page_cache_write_errors = 0
    cleanup_eligible_docs = 0

    def _update_live_progress(current_source_relative_path: str | None) -> None:
        live_status.update(
            {
                "stage": "build",
                "processed": processed_count,
                "total": len(plan),
                "current_source_relative_path": current_source_relative_path,
                "hits": live_hits,
                "planned_new": live_planned_new,
                "parsed": live_parsed,
                "failed": live_failed,
                "cleanup_attempted_pages": total_cleanup_attempted_pages,
                "cleanup_accepted_pages": total_cleanup_accepted_pages,
                "cleanup_rejected_pages": total_cleanup_rejected_pages,
                "cleanup_failed_pages": total_cleanup_failed_pages,
                "cleanup_page_cache_hits": total_cleanup_page_cache_hits,
                "cleanup_page_cache_misses": total_cleanup_page_cache_misses,
                "cleanup_page_cache_write_errors": total_cleanup_page_cache_write_errors,
            }
        )

    for item in plan:
        row: dict[str, Any] = {
            "source_relative_path": item.scoped_pdf.source_relative_path,
            "source_sha256": item.scoped_pdf.source_sha256,
            "source_size_bytes": item.scoped_pdf.source_size_bytes,
            "page_count": item.scoped_pdf.page_count,
            "status": item.status,
            "reason": item.reason,
            "entry_dir": str(item.entry_dir),
            "manifest_path": str(item.manifest_path),
        }
        cleanup_enabled_for_doc = _cleanup_enabled_for_doc(
            settings=settings,
            scoped_pdf=item.scoped_pdf,
        )
        if cleanup_enabled_for_doc:
            cleanup_eligible_docs += 1

        if item.status == "hit":
            row["estimated"] = {
                "standard_pages": 0,
                "premium_pages": 0,
                "estimated_credits": 0.0,
            }
            row["cleanup"] = {
                "enabled_for_doc": cleanup_enabled_for_doc,
                "estimated_calls": 0,
                "attempted_pages": 0,
                "accepted_pages": 0,
                "rejected_pages": 0,
                "failed_pages": 0,
                "page_cache_hits": 0,
                "page_cache_misses": 0,
                "page_cache_write_errors": 0,
            }
            file_rows.append(row)
            processed_count += 1
            live_hits += 1
            _update_live_progress(item.scoped_pdf.source_relative_path)
            continue

        if not args.execute:
            estimate = _credits_from_usage(
                job_pages=item.scoped_pdf.page_count,
                auto_mode_triggered_pages=0,
                settings=settings,
            )
            row["status"] = "planned_new"
            row["estimated"] = asdict(estimate)
            total_standard_pages += estimate.standard_pages
            total_premium_pages += estimate.premium_pages
            total_estimated_credits += estimate.estimated_credits
            estimated_cleanup_calls = _estimated_cleanup_calls_for_item(
                settings=settings,
                scoped_pdf=item.scoped_pdf,
            )
            total_cleanup_estimated_calls += estimated_cleanup_calls
            row["cleanup"] = {
                "enabled_for_doc": cleanup_enabled_for_doc,
                "estimated_calls": estimated_cleanup_calls,
                "attempted_pages": 0,
                "accepted_pages": 0,
                "rejected_pages": 0,
                "failed_pages": 0,
                "page_cache_hits": 0,
                "page_cache_misses": 0,
                "page_cache_write_errors": 0,
            }
            file_rows.append(row)
            processed_count += 1
            live_planned_new += 1
            _update_live_progress(item.scoped_pdf.source_relative_path)
            continue

        try:
            page_rows, raw_payload, job_metadata = _parse_with_llamaparse(item.scoped_pdf, settings)
            cleaned_rows, cleanup_artifacts = _cleanup_page_rows_with_llm(
                scoped_pdf=item.scoped_pdf,
                page_rows=page_rows,
                settings=settings,
                page_cache_root=_cleanup_page_cache_root(parsed_docs_root),
            )
            manifest = _write_cache_entry(
                parsed_docs_root=parsed_docs_root,
                settings=settings,
                scoped_pdf=item.scoped_pdf,
                page_rows=cleaned_rows,
                raw_payload=raw_payload,
                job_metadata=job_metadata,
                cleanup_artifacts=cleanup_artifacts,
            )
            estimate = _credits_from_usage(
                job_pages=int(manifest.get("job_pages", len(page_rows)) or len(page_rows)),
                auto_mode_triggered_pages=int(manifest.get("job_auto_mode_triggered_pages", 0) or 0),
                settings=settings,
            )
            row["status"] = "parsed"
            row["manifest_path"] = str(item.entry_dir / "manifest.json")
            row["job_pages"] = int(manifest.get("job_pages", len(page_rows)) or len(page_rows))
            row["job_auto_mode_triggered_pages"] = int(
                manifest.get("job_auto_mode_triggered_pages", 0) or 0
            )
            row["job_is_cache_hit"] = bool(manifest.get("job_is_cache_hit", False))
            row["cleanup"] = {
                "enabled_for_doc": bool(manifest.get("cleanup_enabled_for_doc", False)),
                "estimated_calls": len(manifest.get("cleanup_selected_pages", [])),
                "attempted_pages": int(manifest.get("cleanup_attempted_pages", 0) or 0),
                "accepted_pages": int(manifest.get("cleanup_accepted_pages", 0) or 0),
                "rejected_pages": int(manifest.get("cleanup_rejected_pages", 0) or 0),
                "failed_pages": int(manifest.get("cleanup_failed_pages", 0) or 0),
                "page_cache_hits": int(manifest.get("cleanup_page_cache_hits", 0) or 0),
                "page_cache_misses": int(manifest.get("cleanup_page_cache_misses", 0) or 0),
                "page_cache_write_errors": int(
                    manifest.get("cleanup_page_cache_write_errors", 0) or 0
                ),
                "mode": manifest.get("cleanup_mode"),
                "profile_version": manifest.get("cleanup_profile_version"),
                "model": manifest.get("cleanup_model"),
                "selected_pages": manifest.get("cleanup_selected_pages", []),
            }
            total_cleanup_estimated_calls += int(len(manifest.get("cleanup_selected_pages", [])))
            total_cleanup_attempted_pages += int(manifest.get("cleanup_attempted_pages", 0) or 0)
            total_cleanup_accepted_pages += int(manifest.get("cleanup_accepted_pages", 0) or 0)
            total_cleanup_rejected_pages += int(manifest.get("cleanup_rejected_pages", 0) or 0)
            total_cleanup_failed_pages += int(manifest.get("cleanup_failed_pages", 0) or 0)
            total_cleanup_page_cache_hits += int(manifest.get("cleanup_page_cache_hits", 0) or 0)
            total_cleanup_page_cache_misses += int(manifest.get("cleanup_page_cache_misses", 0) or 0)
            total_cleanup_page_cache_write_errors += int(
                manifest.get("cleanup_page_cache_write_errors", 0) or 0
            )
            row["estimated"] = asdict(estimate)
            total_standard_pages += estimate.standard_pages
            total_premium_pages += estimate.premium_pages
            total_estimated_credits += estimate.estimated_credits
            live_parsed += 1
        except Exception as exc:  # pragma: no cover - network/provider failure path
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["cleanup"] = {
                "enabled_for_doc": cleanup_enabled_for_doc,
                "estimated_calls": 0,
                "attempted_pages": 0,
                "accepted_pages": 0,
                "rejected_pages": 0,
                "failed_pages": 0,
                "page_cache_hits": 0,
                "page_cache_misses": 0,
                "page_cache_write_errors": 0,
            }
            failures.append(
                {
                    "source_relative_path": item.scoped_pdf.source_relative_path,
                    "error": row["error"],
                }
            )
            live_failed += 1

        file_rows.append(row)
        processed_count += 1
        _update_live_progress(item.scoped_pdf.source_relative_path)

    hit_count = sum(1 for row in file_rows if row["status"] == "hit")
    planned_new_count = sum(1 for row in file_rows if row["status"] == "planned_new")
    parsed_count = sum(1 for row in file_rows if row["status"] == "parsed")
    failed_count = sum(1 for row in file_rows if row["status"] == "failed")

    summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "live_status_path": str(live_status_path),
        "mode": "execute" if args.execute else "dry_run",
        "config_path": str(config_path),
        "source_docs_root": str(source_docs_root),
        "parsed_docs_root": str(parsed_docs_root),
        "scope": {
            "company_tickers": tickers,
            "years": years,
            "pdf_count": len(file_rows),
        },
        "parse_settings": settings.provider_payload(),
        "cost_estimate_rates": {
            "standard_credits_per_page": settings.estimate_standard_credits_per_page,
            "premium_credits_per_page": settings.estimate_premium_credits_per_page,
            "currency_label": settings.estimate_currency_label,
        },
        "counts": {
            "hits": hit_count,
            "planned_new": planned_new_count,
            "parsed": parsed_count,
            "failed": failed_count,
        },
        "cleanup": {
            "mode": settings.cleanup_mode,
            "model": settings.cleanup_model if settings.cleanup_mode != "off" else None,
            "numeric_guardrail_enabled": settings.cleanup_numeric_guardrail_enabled,
            "page_cache_miss_policy": settings.cleanup_page_cache_miss_policy,
            "eligible_docs": cleanup_eligible_docs,
            "estimated_calls": total_cleanup_estimated_calls,
            "attempted_pages": total_cleanup_attempted_pages,
            "accepted_pages": total_cleanup_accepted_pages,
            "rejected_pages": total_cleanup_rejected_pages,
            "failed_pages": total_cleanup_failed_pages,
            "page_cache_hits": total_cleanup_page_cache_hits,
            "page_cache_misses": total_cleanup_page_cache_misses,
            "page_cache_write_errors": total_cleanup_page_cache_write_errors,
        },
        "estimate_totals": {
            "standard_pages": total_standard_pages,
            "premium_pages": total_premium_pages,
            "estimated_credits": total_estimated_credits,
            "currency_label": settings.estimate_currency_label,
        },
        "files": file_rows,
        "failures_path": str(run_dir / "failures.json") if failures else None,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(canonical_json_dumps(summary), encoding="utf-8")

    if failures:
        (run_dir / "failures.json").write_text(canonical_json_dumps({"failures": failures}), encoding="utf-8")

    final_status = "failed" if args.execute and failures else "completed"
    live_status.finalize(
        status=final_status,
        extra={
            "stage": "finalize",
            "processed": processed_count,
            "total": len(plan),
            "current_source_relative_path": None,
            "hits": live_hits,
            "planned_new": live_planned_new,
            "parsed": live_parsed,
            "failed": live_failed,
            "cleanup_attempted_pages": total_cleanup_attempted_pages,
            "cleanup_accepted_pages": total_cleanup_accepted_pages,
            "cleanup_rejected_pages": total_cleanup_rejected_pages,
            "cleanup_failed_pages": total_cleanup_failed_pages,
            "cleanup_page_cache_hits": total_cleanup_page_cache_hits,
            "cleanup_page_cache_misses": total_cleanup_page_cache_misses,
            "cleanup_page_cache_write_errors": total_cleanup_page_cache_write_errors,
            "summary_path": str(summary_path),
        },
        error_message=None if final_status == "completed" else f"{len(failures)} parse failures",
    )

    if not getattr(args, "quiet", False):
        print(f"parse-cache mode={summary['mode']}")
        print(f"scope pdf_count={len(file_rows)}")
        print(
            "counts "
            f"hits={hit_count} planned_new={planned_new_count} parsed={parsed_count} failed={failed_count}"
        )
        print(
            "cleanup "
            f"mode={settings.cleanup_mode} eligible_docs={cleanup_eligible_docs} "
            f"estimated_calls={total_cleanup_estimated_calls} attempted={total_cleanup_attempted_pages} "
            f"accepted={total_cleanup_accepted_pages} rejected={total_cleanup_rejected_pages} "
            f"failed={total_cleanup_failed_pages} page_cache_hits={total_cleanup_page_cache_hits} "
            f"page_cache_misses={total_cleanup_page_cache_misses} "
            f"page_cache_write_errors={total_cleanup_page_cache_write_errors}"
        )
        print(
            "estimated "
            f"standard_pages={total_standard_pages} premium_pages={total_premium_pages} "
            f"total={total_estimated_credits:.4f} {settings.estimate_currency_label}"
        )
        print(f"summary={summary_path}")

    if args.execute and failures:
        return 1
    return 0
