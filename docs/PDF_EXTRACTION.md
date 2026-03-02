# PDF Extraction

Use this doc when you need the current PDF text path, cache modes, or artifact locations.

## Current Behavior

- `cte run` reads PDF text in one of two modes:
  - `local_parser`: parse locally during the run.
  - `cache_only`: read only from the parse cache and fail fast if a required cache entry is missing.
- Paid parsing is isolated to `cte parse-cache build --execute`.
- `cte run` does not make paid parse calls.

## Current Settings

- `settings.pdf_source_mode`
  - `local_parser`
  - `cache_only`
- `settings.pdf_cleanup_page_cache_miss_policy`
  - `call_llm`: allow page-level cleanup cache misses to be filled during the run.
  - `error`: fail if a selected cleanup page is missing from cache.

## Artifact Locations

Parse cache:

```text
artifacts/parsed_docs/<provider>/<profile_key>/<pdf_sha256>/
  manifest.json
  pages.jsonl
  raw_response.json.gz
```

Local cleanup cache:

```text
artifacts/parsed_docs/pymupdf_cleanup/<cleanup_profile_key>/<pdf_sha256>/
  manifest.json
  cleanup_audit.jsonl
  pages/*.md
```

Run artifacts:

```text
artifacts/experiments/<run_id>/
  manifest.json
  results/
  analysis/
```

## Common Commands

Dry-run parse-cache planning:

```bash
uv run cte parse-cache build \
  --config configs/experiments/parity_rag_v1_baseline.toml
```

Execute paid parse-cache build:

```bash
uv run cte parse-cache build \
  --config configs/experiments/parity_rag_v1_baseline.toml \
  --execute
```

## When To Use Which Mode

- Use `local_parser` when you are working on local parsing or page-cleanup behavior.
- Use `cache_only` when you want strict reuse of already-built parse-cache text.
- Use `pdf_cleanup_page_cache_miss_policy = "error"` only when you want strict cleanup-cache replay behavior.

## Related Docs

- Strategy: `docs/EXPERIMENTAL_STRATEGY.md`
- Tasks: `docs/TASKS.md`
- Cleanup repair: `docs/E2A_CLEANUP_REPAIR_RUNBOOK.md`
- Archive history: `docs/archive/PDF_EXTRACTION_HISTORY_2026-03-02.md`
