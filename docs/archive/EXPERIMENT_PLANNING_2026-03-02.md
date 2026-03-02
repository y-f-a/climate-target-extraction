# Experiment Planning

## TL;DR
- Comparator baseline (February 21, 2026): recall `0.7500`, precision `0.7788`, f1 `0.7639`
  - `artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json`
- Track A / D / B all failed precision guard and were dropped.
- Local converter side-quest was mixed (no single global winner).
- Paid parse-cache (LlamaParse) now has strong evidence on a controlled slice (`NVDA 2024` + `AAPL 2024`):
  - precision unchanged, recall and f1 improved vs local parser.
- Remaining biggest bottleneck is still extraction/classification behavior on hard docs (especially NVDA), not only parse quality.

## Fixed Rules
- Primary goal: improve recall on the fixed 14-document evaluation set.
- Guardrail: precision drop must be `<= 0.01` versus comparator.
- Comparator for this cycle:
  - `artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json`
  - average: recall `0.7500`, precision `0.7788`, f1 `0.7639`

## Current Cycle Recap (A / D / B)
- Setup C was removed from this cycle.
- Track A (A1): dropped.
- Track D (D1 with one-cycle union exception): dropped.
- Track B (B1): dropped.

| Item | Evidence | Recall | Precision | F1 | Decision | Why |
|---|---|---:|---:|---:|---|---|
| Comparator baseline | `artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json` | 0.7500 | 0.7788 | 0.7639 | baseline | shared comparator |
| Track A / A1 (prompt relax) | `artifacts/experiments/20260222T135452Z-track_a_setup_a1-37840b85/manifest.json` | 0.8036 | 0.7371 | 0.7684 | drop | precision drop `0.0417` > guardrail |
| Track D / D1 (union exception score) | `artifacts/experiments/20260222T143025Z-track_d_setup_d1-e5bb5704/analysis/track_d_d1_union/union_report.json` | 0.7857 | 0.4400 | 0.5641 | drop | severe precision collapse |
| Track B / B1 (retrieval breadth) | `artifacts/experiments/20260222T155010Z-track_b_setup_b1-288e04f1/manifest.json` | 0.8571 | 0.6774 | 0.7558 | drop | precision drop `0.1014` > guardrail |

## Failure Signatures
- `tsla.2023.targets.v1.json`:
  - persistent miss across baseline/A/D/B (`tp=0`, `fn=1`)
- `nvda.2024.targets.v1.json`:
  - unstable; partial recovery only in Track B run 2 (`tp=1`, `fn=1`)
- `aapl.2024.targets.v1.json`:
  - can hit full recall, but false positives increase in broader-retrieval setups

## Root-Cause Map (Current Confidence)
- `tsla/2023` conversion/representation problem: `high`
  - earlier evidence showed key text fragmentation in local parsing paths
- `nvda/2024` extraction/classification behavior problem: `medium-high`
  - key statements are present in parsed/indexed text, but conversion to scored predictions is inconsistent
- cross-cutting precision sensitivity: `high`
  - recall increases often came with precision guard violations in A/D/B

## PDF Extraction Track (Where We Are Now)

### Local Converter Side-Quest (February 22, 2026)
Run:
- `artifacts/analysis/20260222T173435Z-pdf_conversion_sidequest`

Result:
- mixed, no single winner
  - `tsla/2023`: `pymupdf_blocks`
  - `nvda/2024`: `current_pymupdf4llm`
  - `aapl/2024`: `pymupdf_markdown`

Decision:
- no full local-converter swap from this evidence alone.

### Paid Parse-Cache Rollout (LlamaParse)
Key execute evidence:
- `artifacts/parsed_docs/_runs/20260222T193759965901Z-parse_cache_build/summary.json`
  - `NVDA,AAPL` year `2024`, `parsed=7`, `failed=0`
- `artifacts/parsed_docs/_runs/20260222T201824417231Z-parse_cache_build/summary.json`
  - one-file TSLA execute, `parsed=1`, `failed=0`
- TSLA remaining coverage check:
  - `artifacts/parsed_docs/_runs/20260222T202535768782Z-parse_cache_build/summary.json`
  - `hits=2`, `planned_new=1` (large file still not cached)

### Strict Head-to-Head (Only Source Mode Changed)
Comparison artifacts:
- `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/comparison.md`
- `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/per_doc_delta.csv`

Runs compared:
- local parser:
  - `artifacts/experiments/20260222T220812Z-head_to_head_local_nvda_aapl_2024-8aea79e3/manifest.json`
- cache-only:
  - `artifacts/experiments/20260222T221614Z-head_to_head_cache_nvda_aapl_2024-c7c28322/manifest.json`

Result on `NVDA 2024` + `AAPL 2024`:
- precision: `1.0` -> `1.0` (no loss)
- recall: `0.4` -> `0.8` (`+0.4`)
- f1: `0.5714` -> `0.8889` (`+0.3175`)

Read:
- LlamaParse cache mode is materially better than local parser on this controlled hard-doc slice.
- This is strong directional evidence, not yet a full 14-doc board proof.

## Current Decision Position
- Keep precision guardrail unchanged.
- Treat parse-cache path as preferred for covered hard docs (evidence-backed).
- Next bottleneck to attack: extraction/classification behavior (NVDA-first).
- Continue TSLA parse completion only when credit budget allows.

## Active Track E Subseries (Selection + Guardrail Policy)
Official subseries for PyMuPDF cleanup policy iteration:
- comparator: `artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json`
- setups:
  - `E2a` (currently running): `configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml` (`max_pages=8`, threshold off, numeric guardrail on)
  - `E2d` (next after E2a): `configs/experiments/track_e_setup_e2d_pymupdf_cleanup_numeric_off.toml` (numeric guardrail off only, strict cleanup cache reuse with `pdf_cleanup_page_cache_miss_policy="error"`)
  - `E2b` (blocked this cycle for ROI): `configs/experiments/track_e_setup_e2b_pymupdf_cleanup.toml` (`max_pages=8`, threshold on)
  - `E2c` (blocked this cycle for ROI): `configs/experiments/track_e_setup_e2c_pymupdf_cleanup.toml` (`max_pages=5`, threshold on)
- scoring policy:
  - full fixed 14-doc scope
  - `run_count=2` for `E2a` and `E2d`
  - `E2a` then `E2d` is the active decision path for this cycle
  - `E2b`/`E2c` are deprioritized to blocked for ROI and can be revisited in a later cycle
- keep/drop gate:
  - keep only if recall average improves and precision average drop is `<= 0.01`
- required evidence:
  - experiment manifest + both run reports + page-audit cache evidence for `tsla.2023`, `nvda.2024`, `aapl.2024` from `E2a` and `E2d`
- E2d preflight policy:
  - verify cleanup cache readiness on the full 14-doc scope before launch
  - no new cleanup LLM calls allowed for E2d
  - hard fail on selected-page cleanup cache miss (`miss_policy=error`)

## Next-Step Branch Logic
- If cache-backed runs continue to improve recall with no precision loss:
  - expand cache-backed scope and run official comparator tests.
- If cache-backed gains do not hold on broader scope:
  - keep cache as optional path; return to extraction/retrieval tuning with tight guardrails.
- If NVDA-specific behavior improves via extraction prompt/classification changes:
  - test on fixed slice first, then promote to broader cycle.

## Non-Changes in This Planning Document
- No public CLI/API contract changes described here.
- No schema changes (`src/cte/schemas.py`) described here.
- No change to fixed dataset policy or precision guardrail policy.
