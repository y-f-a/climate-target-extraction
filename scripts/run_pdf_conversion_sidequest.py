from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import fitz
import pymupdf4llm


DATA_SCOPE: list[tuple[str, str]] = [
    ("tsla", "2023"),
    ("nvda", "2024"),
    ("aapl", "2024"),
]
BASELINE_MANIFEST = Path(
    "artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json"
)


@dataclass
class PageText:
    page: int
    text: str


def _doc_name(ticker: str, year: str) -> str:
    return f"{ticker}.{year}.targets.v1.json"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_current_converter_cache(
    *,
    baseline_manifest_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str], list[PageText]]]:
    payload = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    index_events = payload.get("index_events", [])

    lookup: dict[str, dict[str, str]] = {}
    for event in index_events:
        doc = str(event.get("doc", ""))
        if not doc:
            continue
        if doc not in lookup:
            lookup[doc] = {
                "index_id": str(event.get("index_id", "")),
                "index_dir": str(event.get("index_dir", "")),
            }

    cache: dict[tuple[str, str, str], list[PageText]] = {}
    for ticker, year in DATA_SCOPE:
        doc = _doc_name(ticker, year)
        selected = lookup.get(doc)
        if not selected:
            raise RuntimeError(f"Missing baseline index event for {doc}")
        index_dir = Path(selected["index_dir"])
        docstore_path = index_dir / "store" / "docstore.json"
        docstore = json.loads(docstore_path.read_text(encoding="utf-8"))
        ref_info = docstore.get("docstore/ref_doc_info", {})
        data = docstore.get("docstore/data", {})

        per_file_page_texts: dict[str, dict[int, list[str]]] = {}
        per_file_total_pages: dict[str, int] = {}

        for row in ref_info.values():
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            file_name = str(metadata.get("file_name", ""))
            if not file_name.lower().endswith(".pdf"):
                continue
            try:
                page = int(metadata.get("page", 0))
            except (TypeError, ValueError):
                continue
            if page <= 0:
                continue
            try:
                total_pages = int(metadata.get("total_pages", 0))
            except (TypeError, ValueError):
                total_pages = 0
            if total_pages > 0:
                per_file_total_pages[file_name] = max(per_file_total_pages.get(file_name, 0), total_pages)

            node_ids = row.get("node_ids", []) if isinstance(row, dict) else []
            page_bucket = per_file_page_texts.setdefault(file_name, {}).setdefault(page, [])
            for node_id in node_ids:
                node = data.get(node_id, {})
                raw = node.get("__data__", {}) if isinstance(node, dict) else {}
                text = str(raw.get("text", ""))
                normalized = _normalize_space(text)
                if normalized:
                    page_bucket.append(normalized)

        for file_name, page_map in per_file_page_texts.items():
            max_page = max(page_map.keys()) if page_map else 0
            page_count = max(per_file_total_pages.get(file_name, 0), max_page)
            rows: list[PageText] = []
            for page in range(1, page_count + 1):
                chunks = page_map.get(page, [])
                seen: set[str] = set()
                deduped: list[str] = []
                for chunk in chunks:
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    deduped.append(chunk)
                rows.append(PageText(page=page, text="\n\n".join(deduped)))
            cache[(ticker, year, file_name)] = rows

    return lookup, cache


def _extract_with_pymupdf_markdown(pdf_path: Path) -> list[PageText]:
    rows: list[PageText] = []
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        page_separators=False,
    )
    for fallback_page, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        page_value = metadata.get("page", fallback_page)
        try:
            page = int(page_value)
        except (TypeError, ValueError):
            page = fallback_page
        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        rows.append(PageText(page=page, text=text))
    rows.sort(key=lambda row: row.page)
    return rows


def _extract_with_pymupdf_blocks(pdf_path: Path) -> list[PageText]:
    rows: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks", sort=True)
            chunks = [str(block[4]).strip() for block in blocks if str(block[4]).strip()]
            text = "\n\n".join(chunks)
            rows.append(PageText(page=page_index, text=text))
    return rows


def _write_pages(out_file: Path, rows: list[PageText]) -> None:
    lines: list[str] = []
    for row in rows:
        lines.append(f"## PAGE {row.page}")
        lines.append("")
        lines.append(row.text.rstrip())
        lines.append("")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_stats(rows: list[PageText]) -> dict[str, float | int]:
    page_count = len(rows)
    non_empty_pages = sum(1 for row in rows if row.text.strip())
    char_count = sum(len(row.text) for row in rows)
    return {
        "pages": page_count,
        "non_empty_pages": non_empty_pages,
        "empty_pages": page_count - non_empty_pages,
        "empty_page_rate": (page_count - non_empty_pages) / page_count if page_count else 0.0,
        "char_count": char_count,
    }


def _collect_pdfs(source_docs_root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for ticker, year in DATA_SCOPE:
        root = source_docs_root / ticker / year
        if not root.exists():
            continue
        pdfs.extend(sorted(root.glob("*.pdf")))
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local PDF converter side-quest extraction and store outputs."
    )
    parser.add_argument("--source-docs-root", type=Path, default=Path("source_docs_local"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/analysis") / "pdf_conversion_sidequest",
    )
    parser.add_argument("--baseline-manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_docs_root = args.source_docs_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    current_lookup, current_cache = _load_current_converter_cache(
        baseline_manifest_path=args.baseline_manifest,
    )

    converters: dict[str, Callable[[Path], list[PageText]]] = {
        "pymupdf_markdown": _extract_with_pymupdf_markdown,
        "pymupdf_blocks": _extract_with_pymupdf_blocks,
    }

    pdfs = _collect_pdfs(source_docs_root)
    run_started = datetime.now(timezone.utc).isoformat()

    summary: dict[str, object] = {
        "generated_at_utc": run_started,
        "source_docs_root": str(source_docs_root.resolve()),
        "scope": [{"ticker": ticker, "year": year} for ticker, year in DATA_SCOPE],
        "converters": ["current_pymupdf4llm", *list(converters.keys())],
        "current_converter_source_manifest": str(args.baseline_manifest),
        "files": [],
    }

    for pdf_path in pdfs:
        rel_pdf = pdf_path.relative_to(source_docs_root)
        ticker = rel_pdf.parts[0]
        year = rel_pdf.parts[1]
        file_row: dict[str, object] = {
            "pdf": str(rel_pdf),
            "outputs": {},
        }
        current_out = out_dir / "outputs" / "current_pymupdf4llm" / ticker / year / f"{pdf_path.stem}.md"
        current_key = (ticker, year, pdf_path.name)
        if current_key not in current_cache:
            raise RuntimeError(f"Missing current-converter cache for {current_key}")
        if current_out.exists() and not args.overwrite:
            file_row["outputs"]["current_pymupdf4llm"] = {
                "path": str(current_out),
                "stats": None,
                "status": "reused_existing",
                "source_index_id": current_lookup[_doc_name(ticker, year)]["index_id"],
            }
            print(f"[skip] current_pymupdf4llm {rel_pdf}", flush=True)
        else:
            print(f"[run] current_pymupdf4llm {rel_pdf}", flush=True)
            _write_pages(current_out, current_cache[current_key])
            file_row["outputs"]["current_pymupdf4llm"] = {
                "path": str(current_out),
                "stats": _build_stats(current_cache[current_key]),
                "status": "generated",
                "source_index_id": current_lookup[_doc_name(ticker, year)]["index_id"],
            }

        for converter_name, fn in converters.items():
            ext = "md" if converter_name != "pymupdf_blocks" else "txt"
            out_file = out_dir / "outputs" / converter_name / ticker / year / f"{pdf_path.stem}.{ext}"
            if out_file.exists() and not args.overwrite:
                file_row["outputs"][converter_name] = {
                    "path": str(out_file),
                    "stats": None,
                    "status": "reused_existing",
                }
                print(f"[skip] {converter_name} {rel_pdf}", flush=True)
                continue
            print(f"[run] {converter_name} {rel_pdf}", flush=True)
            rows = fn(pdf_path)
            _write_pages(out_file, rows)
            file_row["outputs"][converter_name] = {
                "path": str(out_file),
                "stats": _build_stats(rows),
                "status": "generated",
            }
        summary["files"].append(file_row)

    config = {
        "description": (
            "PDF conversion side-quest: local 3-way extraction comparison over full PDFs for "
            "tsla/2023, nvda/2024, aapl/2024."
        ),
        "decision_rule": (
            "Approve full converter swap only if one converter is clearly better across all three sets. "
            "If mixed, keep current converter and run second side-quest pass."
        ),
        "checklist_criteria": [
            "key_target_statement_survival",
            "sentence_continuity",
            "structure_retention",
            "numeric_and_scope_anchor_clarity",
            "noise_level",
        ],
    }
    (out_dir / "sidequest_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "sidequest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
