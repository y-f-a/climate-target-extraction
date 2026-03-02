from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cte.index_registry import SourceFileInfo
from cte.parse_cache import CleanupRunArtifacts, parse_settings_from_component_settings
from cte.pipelines.rag.v1 import _load_or_build_local_parser_docs_with_cleanup_cache


@dataclass
class DummyDoc:
    text: str
    metadata: dict


class DummyPyMuPdfModule:
    class LlamaMarkdownReader:  # pragma: no cover - behavior is monkeypatched in tests
        def load_data(self, _path: str):
            return []


def _deps() -> dict:
    return {
        "Document": DummyDoc,
        "pymupdf4llm": DummyPyMuPdfModule,
    }


def test_local_cleanup_cache_miss_writes_manifest_pages_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source" / "tsla" / "2023"
    source_dir.mkdir(parents=True)
    (source_dir / "a.pdf").write_bytes(b"fake-pdf")
    source_manifest = [SourceFileInfo(relative_path="a.pdf", size_bytes=8, sha256="sha-a")]
    parsed_docs_root = tmp_path / "parsed_docs"

    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
        }
    )

    def fake_extract(*, pdf_reader, absolute_path):  # noqa: ARG001
        assert absolute_path.name == "a.pdf"
        return (
            [
                {"page": 1, "text_markdown": "page1 original"},
                {"page": 2, "text_markdown": "page2 original"},
            ],
            2,
        )

    def fake_cleanup(*, scoped_pdf, page_rows, settings, page_cache_root):  # noqa: ARG001
        assert scoped_pdf.source_relative_path == "tsla/2023/a.pdf"
        assert [row["page"] for row in page_rows] == [1, 2]
        cleaned_rows = [
            {"page": 1, "text_markdown": "page1 cleaned"},
            {"page": 2, "text_markdown": "page2 cleaned"},
        ]
        artifacts = CleanupRunArtifacts(
            summary={
                "mode": "llm_faithful_v1",
                "profile_version": "v1",
                "enabled_for_doc": True,
                "selected_pages": [1, 2],
                "attempted_pages": 2,
                "accepted_pages": 2,
                "rejected_pages": 0,
                "failed_pages": 0,
                "model": settings.cleanup_model,
            },
            audit_rows=[
                {"page": 1, "decision": "accepted"},
                {"page": 2, "decision": "accepted"},
            ],
            request_rows=[],
            output_rows=[],
        )
        return cleaned_rows, artifacts

    monkeypatch.setattr("cte.pipelines.rag.v1._extract_pdf_page_rows", fake_extract)
    monkeypatch.setattr("cte.pipelines.rag.v1._cleanup_page_rows_with_llm", fake_cleanup)

    docs, details = _load_or_build_local_parser_docs_with_cleanup_cache(
        deps=_deps(),
        parse_settings=settings,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        ticker="tsla",
        year="2023",
    )

    assert [doc.text for doc in docs] == ["page1 cleaned", "page2 cleaned"]
    assert details["cache_hits"] == 0
    assert details["cache_misses"] == 1
    assert details["attempted_files"] == 1
    assert details["attempted_pages"] == 2
    assert details["accepted_pages"] == 2

    entry_dir = (
        Path(details["cache_root"])
        / str(details["cache_profile_key"])
        / "sha-a"
    )
    assert (entry_dir / "manifest.json").exists()
    assert (entry_dir / "cleanup_audit.jsonl").exists()
    assert (entry_dir / "pages" / "0001.md").exists()
    assert (entry_dir / "pages" / "0002.md").exists()


def test_local_cleanup_cache_hit_reuses_cached_pages_without_reprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source" / "tsla" / "2023"
    source_dir.mkdir(parents=True)
    (source_dir / "a.pdf").write_bytes(b"fake-pdf")
    source_manifest = [SourceFileInfo(relative_path="a.pdf", size_bytes=8, sha256="sha-a")]
    parsed_docs_root = tmp_path / "parsed_docs"
    settings = parse_settings_from_component_settings(
        {
            "pdf_cleanup_mode": "llm_faithful_v1",
            "pdf_cleanup_doc_scope": "all",
        }
    )

    def fake_extract(*, pdf_reader, absolute_path):  # noqa: ARG001
        return (
            [
                {"page": 1, "text_markdown": "page1 original"},
                {"page": 2, "text_markdown": "page2 original"},
            ],
            2,
        )

    def fake_cleanup(*, scoped_pdf, page_rows, settings, page_cache_root):  # noqa: ARG001
        artifacts = CleanupRunArtifacts(
            summary={
                "mode": "llm_faithful_v1",
                "profile_version": "v1",
                "enabled_for_doc": True,
                "selected_pages": [1, 2],
                "attempted_pages": 2,
                "accepted_pages": 2,
                "rejected_pages": 0,
                "failed_pages": 0,
                "model": settings.cleanup_model,
            },
            audit_rows=[{"page": 1, "decision": "accepted"}],
            request_rows=[],
            output_rows=[],
        )
        return [
            {"page": 1, "text_markdown": "page1 cleaned"},
            {"page": 2, "text_markdown": "page2 cleaned"},
        ], artifacts

    monkeypatch.setattr("cte.pipelines.rag.v1._extract_pdf_page_rows", fake_extract)
    monkeypatch.setattr("cte.pipelines.rag.v1._cleanup_page_rows_with_llm", fake_cleanup)
    first_docs, first_details = _load_or_build_local_parser_docs_with_cleanup_cache(
        deps=_deps(),
        parse_settings=settings,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        ticker="tsla",
        year="2023",
    )
    assert [doc.text for doc in first_docs] == ["page1 cleaned", "page2 cleaned"]
    assert first_details["cache_misses"] == 1

    def fail_extract(*, pdf_reader, absolute_path):  # noqa: ARG001
        raise AssertionError("extract should not run on cache hit")

    def fail_cleanup(*, scoped_pdf, page_rows, settings, page_cache_root):  # noqa: ARG001
        raise AssertionError("cleanup should not run on cache hit")

    monkeypatch.setattr("cte.pipelines.rag.v1._extract_pdf_page_rows", fail_extract)
    monkeypatch.setattr("cte.pipelines.rag.v1._cleanup_page_rows_with_llm", fail_cleanup)
    second_docs, second_details = _load_or_build_local_parser_docs_with_cleanup_cache(
        deps=_deps(),
        parse_settings=settings,
        source_dir=source_dir,
        source_manifest=source_manifest,
        parsed_docs_root=parsed_docs_root,
        ticker="tsla",
        year="2023",
    )

    assert [doc.text for doc in second_docs] == ["page1 cleaned", "page2 cleaned"]
    assert second_details["cache_hits"] == 1
    assert second_details["cache_misses"] == 0
