# Climate Target Extraction

This repository runs controlled experiments to extract SBTi-aligned climate emissions targets from corporate disclosures.

If you are new here, start with the `rag` pipeline.  
`no_rag` is kept as a reference baseline for comparisons, not as the target architecture.

## Quickstart (RAG first)

### 1) Install dependencies

```bash
uv sync --extra rag --extra dev
```

### 2) Set environment

Create `.env.local` in the repo root:

```bash
cat > .env.local <<'ENV'
OPENAI_API_KEY=sk-...
CTE_SOURCE_DOCS_DIR=/path/to/source_docs_local
ENV
```

`cte run` auto-loads `.env.local` by default.

### 3) Run a RAG smoke experiment

This is the fastest practical first run for visitors.

```bash
uv run cte run \
  --config configs/experiments/parity_rag_v1.toml \
  --run-label rag_smoke
```

### 4) Run the official 2-run RAG baseline

Use this when you want strategy-aligned baseline evidence.

```bash
uv run cte run \
  --config configs/experiments/parity_rag_v1_baseline.toml \
  --run-label rag_parity_baseline
```

By default, `cte run` and `cte evaluate` print live progress to `stderr`. Use `--quiet` to suppress progress lines.

### 5) Run tests (optional)

```bash
uv run pytest
```

## Strategy At A Glance

- The evaluation dataset is fixed and treated as immutable.
- Winner signal is recall.
- Precision guardrail is strict: precision drop must be `<= 0.01` vs comparator.
- Every setup must run twice, and decisions use the 2-run average.
- `no_rag` is a model-generation reference anchor only.

Full strategy: `docs/EXPERIMENTAL_STRATEGY.md`  
Execution board: `docs/TASKS.md`

## Baseline-Only no-RAG

Use `no_rag` when you need a reference anchor for comparisons:

```bash
uv run cte run \
  --config configs/experiments/parity_no_rag_v1.toml \
  --run-label no_rag_anchor
```

## Optional: Paid Parse Cache For PDF Quality

`cte run` does not call paid parsing. Paid parsing is isolated to `cte parse-cache build --execute`.

Required env var for execute mode:

```bash
export LLAMA_CLOUD_API_KEY=...
```

Dry-run first (no paid calls):

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

If you want strict cache reuse in RAG runs, set `settings.pdf_source_mode = "cache_only"` in your run config. Missing cache entries fail fast with remediation guidance.

Current parse-cache guide: `docs/PDF_EXTRACTION.md`

## Useful Local Commands

Run read-only dashboard:

```bash
uv run cte dashboard
```

Terminal status snapshot:

```bash
uv run cte status
```

Watch mode:

```bash
uv run cte status --watch --interval-sec 5
```

Suggest baseline mappings (suggest-only, no file edits):

```bash
uv run cte suggest-baselines \
  --experiments-root artifacts/experiments \
  --baselines-file docs/BASELINES.md
```

## Outputs And Artifacts

`cte run` writes outputs under:

```text
artifacts/experiments/<run_id>/
  generated_targets/<pipeline>/<model_alias>/run_1/*.targets.v1.json
  results/<pipeline>/*.json
  analysis/
```

RAG indexes are stored under:

```text
artifacts/indexes/<pipeline_version>/<index_id>/
  index_manifest.json
  store/
```

Parse-cache artifacts are stored under:

```text
artifacts/parsed_docs/
```

These `artifacts/` outputs are local runtime files and are not committed to the public repository.

## Repository Layout

- `src/cte`: CLI, pipelines, evaluation, config, and indexing logic.
- `configs/experiments`: experiment TOML configs.
- `docs`: active strategy and operator documentation.
- `artifacts`: run outputs, manifests, logs, indexes, and analysis.
- `data`: evaluation references and historical result artifacts.
- `notebooks`: historical reference notebooks (read-only in this workflow).

## Documentation Map

- Strategy: `docs/EXPERIMENTAL_STRATEGY.md`
- Execution board: `docs/TASKS.md`
- PDF extraction + parse cache: `docs/PDF_EXTRACTION.md`
- Cleanup repair + reuse runbook: `docs/E2A_CLEANUP_REPAIR_RUNBOOK.md`
- Monitoring and operations: `docs/MONITORING.md`
- Baseline registry: `docs/BASELINES.md`
- Archive index: `docs/archive/README.md`

## Data Notes

The evaluation set covers 7 companies across 2 reporting years. Source disclosure documents are not redistributed in this repo; provide them locally via `CTE_SOURCE_DOCS_DIR`.

## Write-Up

- Part 1: https://www.reyfarhan.com/posts/climate-targets-01/
- Part 2: https://www.reyfarhan.com/posts/climate-targets-02/
- Part 3: https://www.reyfarhan.com/posts/climate-targets-03/
