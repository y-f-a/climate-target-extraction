from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cte.parse_cache as parse_cache_module
from cte.config import RunConfig
from cte.index_registry import build_source_manifest
from cte.parse_cache import (
    CachePlanItem,
    CleanupResponse,
    CleanupRunArtifacts,
    ParseCacheMissingError,
    ScopedPdf,
    _cleanup_page_rows_with_llm,
    _cleanup_enabled_for_doc,
    _llamaparse_kwargs,
    _select_cleanup_candidates,
    _validate_cleanup_fidelity,
    _write_cache_entry as write_cache_entry_impl,
    build_cache_content_fingerprint,
    load_cached_pages_for_source_manifest,
    parse_settings_from_component_settings,
    run_parse_cache_build,
)


def _write_cache_entry(
    *,
    parsed_docs_root: Path,
    provider: str,
    profile_key: str,
    source_sha256: str,
    source_relative_path: str,
    settings_version: str,
    content_sha256: str,
) -> None:
    entry_dir = parsed_docs_root / provider / profile_key / source_sha256
    entry_dir.mkdir(parents=True, exist_ok=True)

    pages_path = entry_dir / "pages.jsonl"
    pages_path.write_text(
        '{"page": 1, "text_markdown": "page one"}\n'
        '{"page": 2, "text_markdown": "page two"}\n',
        encoding="utf-8",
    )

    raw_path = entry_dir / "raw_response.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"ok": True}, sort_keys=True))

    manifest = {
        "provider": provider,
        "profile_key": profile_key,
        "parser_settings": {},
        "parser_settings_version": settings_version,
        "source_relative_path": source_relative_path,
        "source_file_name": Path(source_relative_path).name,
        "source_sha256": source_sha256,
        "source_size_bytes": 1,
        "generated_at_utc": "2026-02-22T00:00:00+00:00",
        "page_count": 2,
        "job_pages": 2,
        "job_auto_mode_triggered_pages": 0,
        "job_is_cache_hit": False,
        "content_sha256": content_sha256,
        "pages_path": "pages.jsonl",
        "raw_response_path": "raw_response.json.gz",
        "status": "success",
    }
    (entry_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_parse_settings_defaults_and_validation() -> None:
    settings = parse_settings_from_component_settings({})
    assert settings.pdf_source_mode == "local_parser"
    assert settings.provider == "llamaparse"
    assert settings.profile_key() == "hq_auto_v1-v1"
    assert settings.cleanup_mode == "off"
    assert settings.cleanup_model == "gpt-5-mini-2025-08-07"
    assert settings.cleanup_doc_scope == "all"
    assert settings.cleanup_use_score_threshold is True
    assert settings.cleanup_extra_keywords == tuple()
    assert settings.cleanup_length_ratio_max == 2.0
    assert settings.cleanup_numeric_guardrail_enabled is True
    assert settings.cleanup_page_cache_miss_policy == "call_llm"
    assert settings.cleanup_enabled_doc_pairs == tuple()

    with pytest.raises(ValueError):
        parse_settings_from_component_settings({"pdf_source_mode": "bad"})

    with pytest.raises(ValueError):
        parse_settings_from_component_settings({"pdf_parse_ocr_mode": "bad"})

    with pytest.raises(ValueError):
        parse_settings_from_component_settings({"pdf_cleanup_mode": "bad"})

    with pytest.raises(ValueError):
        parse_settings_from_component_settings({"pdf_cleanup_doc_scope": "bad"})

    with pytest.raises(ValueError):
        parse_settings_from_component_settings({"pdf_cleanup_page_cache_miss_policy": "bad"})

    with pytest.raises(ValueError):
        parse_settings_from_component_settings(
            {
                "pdf_cleanup_mode": "llm_faithful_v1",
                "pdf_cleanup_enabled_doc_pairs": ["not_a_pair"],
            }
        )


def test_cleanup_doc_scope_defaults_for_local_parser_vs_cache_only() -> None:
    local_settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
        }
    )
    cache_settings = parse_settings_from_component_settings(
        {
            "pdf_source_mode": "cache_only",
            "pdf_cleanup_mode": "llm_faithful_v1",
        }
    )

    assert local_settings.cleanup_doc_scope == "all"
    assert cache_settings.cleanup_doc_scope == "doc_pairs"


def test_build_cache_content_fingerprint_raises_when_missing(tmp_path: Path) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_dir = source_docs_root / "aapl" / "2024"
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "a.pdf"
    pdf_path.write_bytes(b"not-a-real-pdf")

    source_manifest = build_source_manifest(source_dir)
    settings = parse_settings_from_component_settings({"pdf_source_mode": "cache_only"})

    with pytest.raises(ParseCacheMissingError) as exc:
        build_cache_content_fingerprint(
            source_docs_root=source_docs_root,
            source_dir=source_dir,
            source_manifest=source_manifest,
            parsed_docs_root=tmp_path / "parsed_docs",
            settings=settings,
            config_path=tmp_path / "cfg.toml",
        )

    assert "aapl/2024/a.pdf" in exc.value.missing_paths


def test_build_cache_content_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_dir = source_docs_root / "aapl" / "2024"
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "a.pdf"
    pdf_path.write_bytes(b"fake")

    parsed_docs_root = tmp_path / "parsed_docs"
    source_manifest = build_source_manifest(source_dir)
    source_file = source_manifest[0]

    settings = parse_settings_from_component_settings({"pdf_source_mode": "cache_only"})

    _write_cache_entry(
        parsed_docs_root=parsed_docs_root,
        provider=settings.provider,
        profile_key=settings.profile_key(),
        source_sha256=source_file.sha256,
        source_relative_path="aapl/2024/a.pdf",
        settings_version=settings.settings_version,
        content_sha256="aaa",
    )

    fp1, _details = build_cache_content_fingerprint(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings,
        config_path=tmp_path / "cfg.toml",
    )

    _write_cache_entry(
        parsed_docs_root=parsed_docs_root,
        provider=settings.provider,
        profile_key=settings.profile_key(),
        source_sha256=source_file.sha256,
        source_relative_path="aapl/2024/a.pdf",
        settings_version=settings.settings_version,
        content_sha256="bbb",
    )

    fp2, _details = build_cache_content_fingerprint(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings,
        config_path=tmp_path / "cfg.toml",
    )

    assert fp1 != fp2


def test_load_cached_pages_for_source_manifest_reads_rows(tmp_path: Path) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_dir = source_docs_root / "aapl" / "2024"
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "a.pdf"
    pdf_path.write_bytes(b"fake")

    parsed_docs_root = tmp_path / "parsed_docs"
    source_manifest = build_source_manifest(source_dir)
    source_file = source_manifest[0]

    settings = parse_settings_from_component_settings({"pdf_source_mode": "cache_only"})

    _write_cache_entry(
        parsed_docs_root=parsed_docs_root,
        provider=settings.provider,
        profile_key=settings.profile_key(),
        source_sha256=source_file.sha256,
        source_relative_path="aapl/2024/a.pdf",
        settings_version=settings.settings_version,
        content_sha256="ccc",
    )

    rows = load_cached_pages_for_source_manifest(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings,
        config_path=tmp_path / "cfg.toml",
    )

    assert len(rows) == 2
    assert rows[0].file_name == "a.pdf"
    assert rows[0].source_relative_path == "aapl/2024/a.pdf"
    assert rows[0].page == 1
    assert rows[0].text_markdown == "page one"


def test_llamaparse_kwargs_enforces_mutually_exclusive_modes() -> None:
    settings_auto_and_premium = parse_settings_from_component_settings(
        {
            "pdf_parse_auto_mode": True,
            "pdf_parse_premium_mode": True,
        }
    )
    kwargs_auto_and_premium = _llamaparse_kwargs(settings_auto_and_premium)
    assert kwargs_auto_and_premium["auto_mode"] is True
    assert kwargs_auto_and_premium["premium_mode"] is False

    settings_premium_only = parse_settings_from_component_settings(
        {
            "pdf_parse_auto_mode": False,
            "pdf_parse_premium_mode": True,
        }
    )
    kwargs_premium_only = _llamaparse_kwargs(settings_premium_only)
    assert kwargs_premium_only["auto_mode"] is False
    assert kwargs_premium_only["premium_mode"] is True


def test_cleanup_profile_key_changes_with_scope_and_mode() -> None:
    off = parse_settings_from_component_settings({})
    on_a = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["tsla:2023"],
        }
    )
    on_b = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["tsla:2024"],
        }
    )

    assert off.profile_key() == "hq_auto_v1-v1"
    assert on_a.profile_key() != off.profile_key()
    assert on_a.profile_key() != on_b.profile_key()


def test_cleanup_profile_key_changes_with_numeric_guardrail_and_miss_policy() -> None:
    base = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
        }
    )
    numeric_off = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_numeric_guardrail_enabled": False,
        }
    )
    miss_error = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_page_cache_miss_policy": "error",
        }
    )

    assert base.profile_key() != numeric_off.profile_key()
    assert base.profile_key() != miss_error.profile_key()


def test_cache_fingerprint_changes_when_cleanup_profile_changes(tmp_path: Path) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_dir = source_docs_root / "aapl" / "2024"
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "a.pdf"
    pdf_path.write_bytes(b"fake")

    parsed_docs_root = tmp_path / "parsed_docs"
    source_manifest = build_source_manifest(source_dir)
    source_file = source_manifest[0]

    settings_a = parse_settings_from_component_settings(
        {
            "pdf_source_mode": "cache_only",
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["aapl:2024"],
        }
    )
    settings_b = parse_settings_from_component_settings(
        {
            "pdf_source_mode": "cache_only",
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["aapl:2023"],
        }
    )

    _write_cache_entry(
        parsed_docs_root=parsed_docs_root,
        provider=settings_a.provider,
        profile_key=settings_a.profile_key(),
        source_sha256=source_file.sha256,
        source_relative_path="aapl/2024/a.pdf",
        settings_version=settings_a.settings_version,
        content_sha256="same-content",
    )
    _write_cache_entry(
        parsed_docs_root=parsed_docs_root,
        provider=settings_b.provider,
        profile_key=settings_b.profile_key(),
        source_sha256=source_file.sha256,
        source_relative_path="aapl/2024/a.pdf",
        settings_version=settings_b.settings_version,
        content_sha256="same-content",
    )

    fp_a, _ = build_cache_content_fingerprint(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings_a,
        config_path=tmp_path / "cfg.toml",
    )
    fp_b, _ = build_cache_content_fingerprint(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings_b,
        config_path=tmp_path / "cfg.toml",
    )

    assert fp_a != fp_b


def test_cleanup_candidate_selection_with_low_text_rescue() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["tsla:2023"],
            "pdf_cleanup_max_pages_per_pdf": 3,
            "pdf_cleanup_score_threshold": 3,
            "pdf_cleanup_low_text_rescue": True,
            "pdf_cleanup_low_text_max_chars": 30,
        }
    )
    page_rows = [
        {"page": 1, "text_markdown": "table of contents"},
        {
            "page": 2,
            "text_markdown": (
                "Tesla net zero target by 2030 with 50% emissions reduction across Scope 1 and Scope 2."
            ),
        },
        {"page": 3, "text_markdown": "N/A"},
        {
            "page": 4,
            "text_markdown": (
                "Our emissions target and reduction pathway includes Scope 3 by 2040 and net-zero intent."
            ),
        },
    ]

    selected = _select_cleanup_candidates(page_rows=page_rows, settings=settings)
    assert [row["page"] for row in selected] == [2, 3, 4]


def test_cleanup_candidate_selection_without_threshold_uses_top_k() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_max_pages_per_pdf": 2,
            "pdf_cleanup_use_score_threshold": False,
            "pdf_cleanup_score_threshold": 99,
        }
    )
    page_rows = [
        {"page": 1, "text_markdown": "plain text with little signal"},
        {"page": 2, "text_markdown": "Scope 1 target 50% by 2030"},
        {"page": 3, "text_markdown": "miscellaneous"},
    ]

    selected = _select_cleanup_candidates(page_rows=page_rows, settings=settings)
    assert len(selected) == 2
    assert [row["page"] for row in selected] == [1, 2]


def test_cleanup_scoring_supports_extra_keywords() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_max_pages_per_pdf": 1,
            "pdf_cleanup_score_threshold": 1,
            "pdf_cleanup_extra_keywords": ["supplier engagement"],
        }
    )
    page_rows = [
        {"page": 1, "text_markdown": "boilerplate"},
        {"page": 2, "text_markdown": "Supplier engagement target with timeline."},
    ]
    selected = _select_cleanup_candidates(page_rows=page_rows, settings=settings)
    assert [row["page"] for row in selected] == [2]


def test_cleanup_fidelity_checks_accept_and_reject() -> None:
    accepted = _validate_cleanup_fidelity(
        original_text="Reduce Scope 1 and 2 emissions 50% by 2030.",
        cleaned_text="Reduce Scope 1 and 2 emissions 50% by 2030.",
    )
    assert accepted["accepted"] is True

    rejected = _validate_cleanup_fidelity(
        original_text="Reduce Scope 1 and 2 emissions 50% by 2030.",
        cleaned_text="Reduce Scope 1 and 2 emissions 40% by 2030.",
    )
    assert rejected["accepted"] is False
    assert rejected["reason"] == "numeric_token_retention_below_threshold"


def test_cleanup_fidelity_skips_numeric_guardrail_when_disabled() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_numeric_guardrail_enabled": False,
        }
    )
    accepted = _validate_cleanup_fidelity(
        original_text="Reduce Scope 1 and 2 emissions 50% by 2030.",
        cleaned_text="Reduce Scope 1 and 2 emissions 40% by 2030.",
        settings=settings,
    )
    assert accepted["accepted"] is True
    assert accepted["numeric_guardrail_enabled"] is False
    assert accepted["reason"] == "accepted_numeric_guardrail_disabled"


def test_cleanup_fidelity_allows_loose_length_ratio_when_configured() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_length_ratio_max": 8.0,
        }
    )
    accepted = _validate_cleanup_fidelity(
        original_text="Impact Report 2023 10. 51 Tons of CO2e.",
        cleaned_text=(
            "Impact Report 2023 10. 51 Tons of CO2e. "
            "After 17 years this avoids emissions and the same figures are preserved."
        ),
        settings=settings,
    )
    assert accepted["accepted"] is True
    assert accepted["thresholds"]["length_ratio_max"] == 8.0


def test_cleanup_fidelity_ignores_repeated_header_numeric_noise() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_length_ratio_max": 8.0,
        }
    )
    accepted = _validate_cleanup_fidelity(
        original_text=(
            "3825 Impact Report 2023 3825 Tesla's Safety Score Incentivizes Safer Driving"
        ),
        cleaned_text=(
            "Impact Report 2023 25 Tesla's Safety Score Incentivizes Safer Driving and rewards safe behavior."
        ),
        settings=settings,
    )
    assert accepted["accepted"] is True
    assert accepted["numeric_tokens_ignored"] == ["3825"]


def test_cleanup_fidelity_allows_length_ratio_around_nine_when_relaxed() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_length_ratio_max": 10.0,
        }
    )
    accepted = _validate_cleanup_fidelity(
        original_text="Impact Report 2023 17 Model Y 3.8",
        cleaned_text=(
            "Impact Report 2023 17 Model Y 3.8 "
            + ("expanded narrative text " * 12).strip()
        ),
        settings=settings,
    )
    assert accepted["accepted"] is True
    assert accepted["length_ratio"] > 8.0


def test_cleanup_enabled_for_doc_scope_all() -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="msft",
        year="2024",
        absolute_path=Path("/tmp/placeholder.pdf"),
        source_relative_path="msft/2024/placeholder.pdf",
        source_file_name="placeholder.pdf",
        source_sha256="sha",
        source_size_bytes=1,
        page_count=1,
    )
    assert _cleanup_enabled_for_doc(settings=settings, scoped_pdf=scoped_pdf) is True


def test_cleanup_fail_open_without_openai_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["tsla:2023"],
            "pdf_cleanup_max_pages_per_pdf": 3,
            "pdf_cleanup_score_threshold": 1,
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="tsla",
        year="2023",
        absolute_path=tmp_path / "dummy.pdf",
        source_relative_path="tsla/2023/dummy.pdf",
        source_file_name="dummy.pdf",
        source_sha256="abc",
        source_size_bytes=1,
        page_count=1,
    )
    page_rows = [{"page": 1, "text_markdown": "Net zero target by 2030 with 50% reduction."}]

    cleaned_rows, artifacts = _cleanup_page_rows_with_llm(
        scoped_pdf=scoped_pdf,
        page_rows=page_rows,
        settings=settings,
    )

    assert cleaned_rows[0]["text_markdown"] == page_rows[0]["text_markdown"]
    assert artifacts.summary["attempted_pages"] == 1
    assert artifacts.summary["failed_pages"] == 1
    assert artifacts.summary["accepted_pages"] == 0
    assert artifacts.audit_rows[0]["decision"] == "failed_open_missing_openai_api_key"


def test_cleanup_fail_open_page_cache_miss_without_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_page_cache_miss_policy": "fail_open",
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="tsla",
        year="2023",
        absolute_path=tmp_path / "dummy.pdf",
        source_relative_path="tsla/2023/dummy.pdf",
        source_file_name="dummy.pdf",
        source_sha256="abc-miss",
        source_size_bytes=1,
        page_count=1,
    )
    page_rows = [{"page": 1, "text_markdown": "Net zero target by 2030 with 50% reduction."}]
    llm_calls = {"count": 0}

    def fail_call_cleanup_llm(**kwargs):  # noqa: ARG001
        llm_calls["count"] += 1
        raise AssertionError("LLM should not be called under fail_open miss policy")

    monkeypatch.setattr("cte.parse_cache._call_cleanup_llm", fail_call_cleanup_llm)

    cleaned_rows, artifacts = _cleanup_page_rows_with_llm(
        scoped_pdf=scoped_pdf,
        page_rows=page_rows,
        settings=settings,
        page_cache_root=tmp_path / "page_cache",
    )

    assert llm_calls["count"] == 0
    assert cleaned_rows[0]["text_markdown"] == page_rows[0]["text_markdown"]
    assert artifacts.summary["attempted_pages"] == 1
    assert artifacts.summary["failed_pages"] == 1
    assert artifacts.summary["accepted_pages"] == 0
    assert artifacts.summary["page_cache_misses"] == 1
    assert artifacts.summary["page_cache_miss_policy"] == "fail_open"
    assert artifacts.audit_rows[0]["decision"] == "failed_open_page_cache_miss"
    assert artifacts.audit_rows[0]["llm_invoked"] is False
    assert artifacts.audit_rows[0]["checks"]["reason"] == "page_cache_miss_fail_open"


def test_cleanup_error_page_cache_miss_raises_without_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_page_cache_miss_policy": "error",
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="tsla",
        year="2023",
        absolute_path=tmp_path / "dummy.pdf",
        source_relative_path="tsla/2023/dummy.pdf",
        source_file_name="dummy.pdf",
        source_sha256="abc-miss-err",
        source_size_bytes=1,
        page_count=1,
    )
    page_rows = [{"page": 1, "text_markdown": "Net zero target by 2030 with 50% reduction."}]
    llm_calls = {"count": 0}

    def fail_call_cleanup_llm(**kwargs):  # noqa: ARG001
        llm_calls["count"] += 1
        raise AssertionError("LLM should not be called under error miss policy")

    monkeypatch.setattr("cte.parse_cache._call_cleanup_llm", fail_call_cleanup_llm)

    with pytest.raises(RuntimeError, match="Cleanup page cache miss under policy=error"):
        _cleanup_page_rows_with_llm(
            scoped_pdf=scoped_pdf,
            page_rows=page_rows,
            settings=settings,
            page_cache_root=tmp_path / "page_cache",
        )

    assert llm_calls["count"] == 0


def test_cleanup_page_cache_reuses_cleaned_page_across_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_max_pages_per_pdf": 8,
            "pdf_cleanup_use_score_threshold": False,
        }
    )
    alt_settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
            "pdf_cleanup_max_pages_per_pdf": 5,
            "pdf_cleanup_use_score_threshold": True,
            "pdf_cleanup_score_threshold": 3,
            "pdf_cleanup_numeric_guardrail_enabled": False,
            "pdf_cleanup_page_cache_miss_policy": "error",
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="tsla",
        year="2023",
        absolute_path=tmp_path / "dummy.pdf",
        source_relative_path="tsla/2023/dummy.pdf",
        source_file_name="dummy.pdf",
        source_sha256="sha-cache-1",
        source_size_bytes=1,
        page_count=1,
    )
    page_rows = [{"page": 1, "text_markdown": "Net zero target Scope 1 reduction 50% by 2030."}]
    page_cache_root = tmp_path / "page_cache"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "cte.parse_cache._render_page_image_data_url",
        lambda *, scoped_pdf, page_no, image_dpi: "data:image/png;base64,AAAA",  # noqa: ARG005
    )
    llm_calls = {"count": 0}

    def fake_call_cleanup_llm(*, settings, scoped_pdf, page_no, original_text, image_data_url):  # noqa: ARG001
        llm_calls["count"] += 1
        return CleanupResponse(cleaned_text=original_text, notes="cached")

    monkeypatch.setattr("cte.parse_cache._call_cleanup_llm", fake_call_cleanup_llm)

    first_rows, first_artifacts = _cleanup_page_rows_with_llm(
        scoped_pdf=scoped_pdf,
        page_rows=page_rows,
        settings=base_settings,
        page_cache_root=page_cache_root,
    )
    assert llm_calls["count"] == 1
    assert first_artifacts.summary["page_cache_hits"] == 0
    assert first_artifacts.summary["page_cache_misses"] == 1
    assert first_rows[0]["text_markdown"] == page_rows[0]["text_markdown"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "cte.parse_cache._call_cleanup_llm",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called on cache hit")),
    )

    second_rows, second_artifacts = _cleanup_page_rows_with_llm(
        scoped_pdf=scoped_pdf,
        page_rows=page_rows,
        settings=alt_settings,
        page_cache_root=page_cache_root,
    )
    assert llm_calls["count"] == 1
    assert second_rows[0]["text_markdown"] == page_rows[0]["text_markdown"]
    assert second_artifacts.summary["attempted_pages"] == 1
    assert second_artifacts.summary["accepted_pages"] == 1
    assert second_artifacts.summary["failed_pages"] == 0
    assert second_artifacts.summary["page_cache_hits"] == 1
    assert second_artifacts.summary["page_cache_misses"] == 0
    assert second_artifacts.audit_rows[0]["page_cache_status"] == "hit"
    assert second_artifacts.audit_rows[0]["llm_invoked"] is False


def test_write_cache_entry_includes_cleanup_manifest_fields(tmp_path: Path) -> None:
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["aapl:2024"],
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="aapl",
        year="2024",
        absolute_path=tmp_path / "a.pdf",
        source_relative_path="aapl/2024/a.pdf",
        source_file_name="a.pdf",
        source_sha256="sha123",
        source_size_bytes=10,
        page_count=1,
    )
    artifacts = CleanupRunArtifacts(
        summary={
            "mode": "llm_faithful_v1",
            "profile_version": "v1",
            "enabled_for_doc": True,
            "selected_pages": [1],
            "attempted_pages": 1,
            "accepted_pages": 1,
            "rejected_pages": 0,
            "failed_pages": 0,
            "model": settings.cleanup_model,
        },
        audit_rows=[{"page": 1, "decision": "accepted"}],
        request_rows=[{"page": 1, "model": settings.cleanup_model}],
        output_rows=[{"page": 1, "decision": "accepted"}],
    )
    manifest = write_cache_entry_impl(
        parsed_docs_root=tmp_path / "parsed_docs",
        settings=settings,
        scoped_pdf=scoped_pdf,
        page_rows=[{"page": 1, "text_markdown": "clean text"}],
        raw_payload={"ok": True},
        job_metadata={"job_pages": 1},
        cleanup_artifacts=artifacts,
    )

    assert manifest["cleanup_mode"] == "llm_faithful_v1"
    assert manifest["cleanup_attempted_pages"] == 1
    assert manifest["cleanup_accepted_pages"] == 1
    assert manifest["cleanup_rejected_pages"] == 0
    assert manifest["cleanup_failed_pages"] == 0
    assert manifest["cleanup_page_cache_hits"] == 0
    assert manifest["cleanup_page_cache_misses"] == 0
    assert manifest["cleanup_page_cache_write_errors"] == 0
    assert manifest["cleanup_numeric_guardrail_enabled"] is True
    assert manifest["cleanup_page_cache_miss_policy"] == "call_llm"
    assert manifest["cleanup_model"] == settings.cleanup_model
    assert manifest["cleanup_audit_path"] == "cleanup_audit.jsonl"
    assert manifest["cleanup_requests_path"] == "cleanup_requests.jsonl"
    assert manifest["cleanup_outputs_path"] == "cleanup_outputs.jsonl"


def test_load_cached_pages_returns_cleaned_text_written_by_impl(tmp_path: Path) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_dir = source_docs_root / "aapl" / "2024"
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "a.pdf"
    pdf_path.write_bytes(b"fake")

    parsed_docs_root = tmp_path / "parsed_docs"
    source_manifest = build_source_manifest(source_dir)
    source_file = source_manifest[0]
    settings = parse_settings_from_component_settings(
        {
            "pdf_source_mode": "cache_only",
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_enabled_doc_pairs": ["aapl:2024"],
        }
    )
    scoped_pdf = ScopedPdf(
        ticker="aapl",
        year="2024",
        absolute_path=pdf_path,
        source_relative_path="aapl/2024/a.pdf",
        source_file_name="a.pdf",
        source_sha256=source_file.sha256,
        source_size_bytes=source_file.size_bytes,
        page_count=1,
    )
    artifacts = CleanupRunArtifacts(
        summary={
            "mode": "llm_faithful_v1",
            "profile_version": "v1",
            "enabled_for_doc": True,
            "selected_pages": [1],
            "attempted_pages": 1,
            "accepted_pages": 1,
            "rejected_pages": 0,
            "failed_pages": 0,
            "model": settings.cleanup_model,
        },
        audit_rows=[{"page": 1, "decision": "accepted"}],
        request_rows=[{"page": 1, "model": settings.cleanup_model}],
        output_rows=[{"page": 1, "decision": "accepted"}],
    )
    write_cache_entry_impl(
        parsed_docs_root=parsed_docs_root,
        settings=settings,
        scoped_pdf=scoped_pdf,
        page_rows=[{"page": 1, "text_markdown": "cleaned target text"}],
        raw_payload={"ok": True},
        job_metadata={"job_pages": 1},
        cleanup_artifacts=artifacts,
    )

    rows = load_cached_pages_for_source_manifest(
        source_docs_root=source_docs_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        settings=settings,
        config_path=tmp_path / "cfg.toml",
    )

    assert len(rows) == 1
    assert rows[0].text_markdown == "cleaned target text"


def test_run_parse_cache_build_writes_live_status_for_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_docs_root.mkdir(parents=True)
    parsed_docs_root = tmp_path / "parsed_docs"

    config = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
        company_tickers=["AAPL"],
        years=["2024"],
        artifacts_root=tmp_path / "artifacts",
        parsed_docs_root=parsed_docs_root,
        component_settings={},
    )
    scoped_hit = ScopedPdf(
        ticker="aapl",
        year="2024",
        absolute_path=source_docs_root / "aapl" / "2024" / "hit.pdf",
        source_relative_path="aapl/2024/hit.pdf",
        source_file_name="hit.pdf",
        source_sha256="sha-hit",
        source_size_bytes=1,
        page_count=4,
    )
    scoped_new = ScopedPdf(
        ticker="aapl",
        year="2024",
        absolute_path=source_docs_root / "aapl" / "2024" / "new.pdf",
        source_relative_path="aapl/2024/new.pdf",
        source_file_name="new.pdf",
        source_sha256="sha-new",
        source_size_bytes=1,
        page_count=5,
    )
    plan = [
        CachePlanItem(
            scoped_pdf=scoped_hit,
            status="hit",
            reason="cache_hit",
            entry_dir=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-hit",
            manifest_path=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-hit" / "manifest.json",
            manifest={"status": "success"},
        ),
        CachePlanItem(
            scoped_pdf=scoped_new,
            status="missing",
            reason="cache_missing",
            entry_dir=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-new",
            manifest_path=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-new" / "manifest.json",
            manifest=None,
        ),
    ]

    monkeypatch.setattr(parse_cache_module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(parse_cache_module, "maybe_load_env_file", lambda _path: None)
    monkeypatch.setattr(parse_cache_module, "resolve_source_docs_root", lambda _config: source_docs_root)
    monkeypatch.setattr(parse_cache_module, "_build_scope", lambda **kwargs: [scoped_hit, scoped_new])
    monkeypatch.setattr(parse_cache_module, "_build_cache_plan", lambda **kwargs: plan)

    args = SimpleNamespace(
        config="configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml",
        company_tickers=None,
        years=None,
        execute=False,
        max_new_pdfs=None,
        quiet=True,
        worker=False,
    )
    exit_code = run_parse_cache_build(args)
    assert exit_code == 0

    run_dirs = sorted((parsed_docs_root / "_runs").glob("*-parse_cache_build"))
    assert run_dirs
    run_dir = run_dirs[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    live_status = json.loads((run_dir / "live_status.json").read_text(encoding="utf-8"))

    assert summary["live_status_path"] == str(run_dir / "live_status.json")
    assert summary["counts"]["hits"] == 1
    assert summary["counts"]["planned_new"] == 1
    assert summary["counts"]["parsed"] == 0
    assert summary["counts"]["failed"] == 0

    assert live_status["job_kind"] == "parse_cache_build"
    assert live_status["status"] == "completed"
    assert live_status["processed"] == 2
    assert live_status["total"] == 2
    assert live_status["hits"] == 1
    assert live_status["planned_new"] == 1
    assert live_status["parsed"] == 0
    assert live_status["failed"] == 0
    assert live_status["summary_path"] == str(run_dir / "summary.json")
    assert live_status["finished_at_utc"] is not None


def test_run_parse_cache_build_marks_live_status_failed_when_execute_has_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_docs_root = tmp_path / "source_docs"
    source_docs_root.mkdir(parents=True)
    parsed_docs_root = tmp_path / "parsed_docs"

    config = RunConfig(
        pipeline="rag",
        pipeline_version="rag.v1",
        company_tickers=["AAPL"],
        years=["2024"],
        artifacts_root=tmp_path / "artifacts",
        parsed_docs_root=parsed_docs_root,
        component_settings={},
    )
    scoped_new = ScopedPdf(
        ticker="aapl",
        year="2024",
        absolute_path=source_docs_root / "aapl" / "2024" / "new.pdf",
        source_relative_path="aapl/2024/new.pdf",
        source_file_name="new.pdf",
        source_sha256="sha-new",
        source_size_bytes=1,
        page_count=5,
    )
    plan = [
        CachePlanItem(
            scoped_pdf=scoped_new,
            status="missing",
            reason="cache_missing",
            entry_dir=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-new",
            manifest_path=parsed_docs_root / "llamaparse" / "hq_auto_v1-v1" / "sha-new" / "manifest.json",
            manifest=None,
        ),
    ]

    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-key")
    monkeypatch.setattr(parse_cache_module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(parse_cache_module, "maybe_load_env_file", lambda _path: None)
    monkeypatch.setattr(parse_cache_module, "resolve_source_docs_root", lambda _config: source_docs_root)
    monkeypatch.setattr(parse_cache_module, "_build_scope", lambda **kwargs: [scoped_new])
    monkeypatch.setattr(parse_cache_module, "_build_cache_plan", lambda **kwargs: plan)

    def _fail_parse(**kwargs):  # noqa: ANN003,ARG001
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(parse_cache_module, "_parse_with_llamaparse", _fail_parse)

    args = SimpleNamespace(
        config="configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml",
        company_tickers=None,
        years=None,
        execute=True,
        max_new_pdfs=None,
        quiet=True,
        worker=False,
    )
    exit_code = run_parse_cache_build(args)
    assert exit_code == 1

    run_dirs = sorted((parsed_docs_root / "_runs").glob("*-parse_cache_build"))
    assert run_dirs
    run_dir = run_dirs[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    live_status = json.loads((run_dir / "live_status.json").read_text(encoding="utf-8"))

    assert summary["counts"]["failed"] == 1
    assert live_status["status"] == "failed"
    assert live_status["failed"] == 1
    assert live_status["processed"] == 1
    assert live_status["total"] == 1
    assert live_status["error_message"] == "1 parse failures"
    assert live_status["finished_at_utc"] is not None
