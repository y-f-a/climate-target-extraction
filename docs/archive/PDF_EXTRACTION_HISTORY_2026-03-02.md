# PDF Extraction

## TL;DR
- Local PyMuPDF-based converter variants gave mixed results and no single clear winner.
- Paid LlamaParse cache extraction worked reliably on the docs we parsed.
- Manual inspection shows key NVDA and AAPL target statements are present in cached text.
- On a strict apples-to-apples test (`NVDA 2024` + `AAPL 2024`), `cache_only` (LlamaParse) beat `local_parser`:
  - precision unchanged (`1.0` vs `1.0`)
  - recall improved (`0.4` -> `0.8`)
  - f1 improved (`0.5714` -> `0.8889`)
- Main remaining issue is not only parsing. It is still extraction/classification behavior on hard docs (especially NVDA).

## Why This Document Exists
This is the single readable record of:
- what we tested for PDF extraction quality,
- what worked and did not work,
- what those results mean for the next experiment decisions.

## What We Mean by "PDF Extraction Quality"
Two separate layers matter:
- Parse quality: did the converter preserve the key statements in readable form?
- End-to-end extraction quality: did the model output the right targets after retrieval + extraction + evaluation rules?

A parser can improve parse quality while end-to-end misses still happen due to extraction/classification behavior.

## Current Architecture (Simple View)
- Paid parsing happens only in:
  - `cte parse-cache build --execute`
- Normal experiment runs do not call paid parsing:
  - `cte run`
- To force reuse of cached parse text:
  - `settings.pdf_source_mode = "cache_only"`
- If cache is missing in `cache_only` mode, run fails fast with a remediation command.

Cache artifact layout:
- `artifacts/parsed_docs/<provider>/<profile_key>/<pdf_sha256>/`
- Files:
  - `manifest.json`
  - `pages.jsonl`
  - `raw_response.json.gz`

## Evidence Timeline

### 1) Local Converter Side-Quest (Pre-LlamaParse)
Run:
- `artifacts/analysis/20260222T173435Z-pdf_conversion_sidequest`

Compared:
- `current_pymupdf4llm`
- `pymupdf_markdown`
- `pymupdf_blocks`

Outcome:
- mixed (no global winner)
- `tsla/2023` winner: `pymupdf_blocks`
- `nvda/2024` winner: `current_pymupdf4llm`
- `aapl/2024` winner: `pymupdf_markdown`

Decision at that point:
- no full local-converter swap.

### 2) Paid Parse-Cache Rollout (LlamaParse)
Important run evidence:

| Run | Scope | Outcome |
|---|---|---|
| `20260222T191720Z-parse_cache_build` | Dry-run full configured scope | `43` PDFs, estimate `50880` credits |
| `20260222T193759965901Z-parse_cache_build` | Execute `NVDA,AAPL` `2024` | `parsed=7`, `failed=0` |
| `20260222T201824417231Z-parse_cache_build` | Execute one-file TSLA (`2023-tesla-impact-report.pdf`) | `parsed=1`, `failed=0` |
| `20260222T202535768782Z-parse_cache_build` | TSLA coverage check dry-run | `hits=2`, `planned_new=1` (large TSLA file remaining) |

Cost interpretation:
- command-side estimator is conservative and was higher than observed admin spend in practice.
- example: one execute run estimated `23985` credits while admin showed lower live usage.

### 3) Manual Inspection of Cache-Only Validation Outputs
Validation run:
- `artifacts/experiments/20260222T202819Z-cache_only_validation_nvda_aapl_2024-c09ab9e7/manifest.json`

#### NVDA 2024
Reference:
- `data/evaluation_set/reference_targets/nvda.2024.targets.v1.json`
- 4 total targets in file, with 2 scored after evaluator filtering rules.

Model output:
- `artifacts/experiments/20260222T202819Z-cache_only_validation_nvda_aapl_2024-c09ab9e7/generated_targets/rag/gpt5_2/run_1/nvda.2024.targets.v1.json`
- model emitted only 1 item and labeled it `non_target_claim`.

Evaluator behavior:
- evaluator drops `non_target_claim` before scoring (`src/cte/eval.py`).
- practical result: this run scored `tp=0`, `fn=2` for NVDA.

Cached text evidence (statements present):
- `nvda/2024/FY2024-NVIDIA-Corporate-Sustainability-Report.pdf` p11-12
- `nvda/2024/NVIDIA-2024-Annual-Report.pdf` p63
- `nvda/2024/NVIDIA_VCDMA_Disclosure.pdf` p1

Conclusion:
- NVDA miss here looks mainly like extraction/classification behavior, not parse text loss.

#### AAPL 2024
Reference:
- `data/evaluation_set/reference_targets/aapl.2024.targets.v1.json`

Model output:
- `artifacts/experiments/20260222T202819Z-cache_only_validation_nvda_aapl_2024-c09ab9e7/generated_targets/rag/gpt5_2/run_1/aapl.2024.targets.v1.json`
- emitted 3 targets, matched as `tp=3`, `fn=0`, `fp=0` in that run report.

Cached text evidence (statements present):
- `aapl/2024/Apple-Supply-Chain-2025-Progress-Report.pdf` p11 and p56
- `aapl/2024/Apple_Environmental_Progress_Report_2025.pdf` p5, p10, p11

Conclusion:
- parse quality appears sufficient for recall on AAPL in this slice.
- remaining differences are mostly label/field normalization behavior.

### 4) Strict Head-to-Head (Only Parser Mode Changed)
Goal:
- isolate parser/source mode effect.

Controls:
- same docs (`NVDA 2024`, `AAPL 2024`)
- same model, prompts, retrieval, settings
- both rebuilt with `--index-policy rebuild`
- only changed `pdf_source_mode`

Runs:
- local parser:
  - `artifacts/experiments/20260222T220812Z-head_to_head_local_nvda_aapl_2024-8aea79e3/manifest.json`
  - report: `artifacts/experiments/20260222T220812Z-head_to_head_local_nvda_aapl_2024-8aea79e3/results/rag/v2_gpt5_2_1.json`
- cache-only:
  - `artifacts/experiments/20260222T221614Z-head_to_head_cache_nvda_aapl_2024-c7c28322/manifest.json`
  - report: `artifacts/experiments/20260222T221614Z-head_to_head_cache_nvda_aapl_2024-c7c28322/results/rag/v2_gpt5_2_1.json`
- comparison outputs:
  - `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/comparison.md`
  - `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/per_doc_delta.csv`
  - `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/per_field_delta.csv`

Result summary:

| Metric | Local Parser | Cache-Only (LlamaParse) | Delta (Cache - Local) |
|---|---:|---:|---:|
| Precision | 1.0 | 1.0 | 0.0 |
| Recall | 0.4 | 0.8 | +0.4 |
| F1 | 0.5714 | 0.8889 | +0.3175 |

Per-doc deltas:
- `nvda.2024.targets.v1.json`: `tp 0 -> 1`, `fn 2 -> 1`
- `aapl.2024.targets.v1.json`: `tp 2 -> 3`, `fn 1 -> 0`

Interpretation:
- on this controlled slice, LlamaParse cache was materially better with no precision loss.
- this is strong directional evidence, but still a slice, not the full board.

## Quick FN-Cause Snapshot (Selected Runs)
Based on a quick breakdown across baseline + Track A + Track B + cache-only validation:
- `underprediction_scored_targets`: `42.9%` of FNs
- `no_predictions`: `40.0%`
- `non_target_only_predictions`: `11.4%`
- `matching_or_field_mismatch`: `5.7%`

For `nvda.2024` specifically in that sample:
- `no_predictions`: `61.5%`
- `non_target_only_predictions`: `30.8%`
- `underprediction_scored_targets`: `7.7%`

Read:
- `non_target_claim` over-labeling is important for NVDA, but not the whole FN story overall.

## Current Status
- LlamaParse cache has clear positive evidence on the hard-doc slice tested.
- `NVDA 2024` remains partly behavior-limited (classification/output choices).
- `TSLA 2023` cache is still partial (2 files cached, 1 large file remaining).

## Open Pilots (Already Configured)
- Track E / Setup E1 (cache cleanup):
  - `configs/experiments/track_e_setup_e1_cleanup.toml`
- Track E / Setup E2 (PyMuPDF-first cleanup cache):
  - `configs/experiments/track_e_setup_e2_pymupdf_cleanup.toml`

## Track E Subseries (E2a/E2b/E2c)
Goal:
- isolate cleanup page-selection policy impact on recall/precision while all other cleanup settings stay fixed.

Comparator:
- `artifacts/experiments/20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb/manifest.json`
- baseline averages: recall `0.7500`, precision `0.7788`

Setups:
- `E2a` (`8/off`): `configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml`
- `E2b` (`8/on`): `configs/experiments/track_e_setup_e2b_pymupdf_cleanup.toml`
- `E2c` (`5/on`): `configs/experiments/track_e_setup_e2c_pymupdf_cleanup.toml`

Run commands:
- `uv run cte run --config configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml --run-label track_e_setup_e2a_pymupdf_cleanup`
- `uv run cte run --config configs/experiments/track_e_setup_e2b_pymupdf_cleanup.toml --run-label track_e_setup_e2b_pymupdf_cleanup`
- `uv run cte run --config configs/experiments/track_e_setup_e2c_pymupdf_cleanup.toml --run-label track_e_setup_e2c_pymupdf_cleanup`

Required evidence per setup:
- `artifacts/experiments/<run_id>/manifest.json`
- both run reports listed in `manifest.runs[].report_path`
- cleanup audits/pages:
  - `artifacts/parsed_docs/pymupdf_cleanup/<cleanup_profile_key>/<pdf_sha256>/cleanup_audit.jsonl`
  - `artifacts/parsed_docs/pymupdf_cleanup/<cleanup_profile_key>/<pdf_sha256>/pages/*.md`
- hard-doc diagnostics must include `tsla.2023`, `nvda.2024`, `aapl.2024`

Decision gate:
- keep only if recall average improves and precision average drop is <= `0.01` vs comparator.

## Practical Next Focus
If we optimize for impact per effort:
1. Keep using `cache_only` where coverage exists.
2. Target NVDA extraction/classification behavior next.
3. Complete remaining TSLA parse when credits allow.
