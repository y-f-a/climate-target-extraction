# Project Learnings Snapshot

- Snapshot date: 2026-03-02
- Evidence window covered: 2026-02-21 to 2026-02-23
- Purpose: record what the project has learned so far, including negative findings that narrowed the search space.

## Current Best Proven Setup

- Best full-board result so far is E2d quality mode:
  - recall `0.8036`
  - precision `0.8052`
  - f1 `0.8041`
  - evidence: `artifacts/analysis/20260223T155453Z-t033_e2d_quality_decision/decision_summary.md`
- This setup beat both E2a and the shared RAG comparator on the fixed 14-document evaluation set.

## Durable Positive Learnings

- RAG is necessary for useful recall in this project.
  - `no_rag` anchor recall was `0.2857`
  - shared RAG comparator recall was `0.7500`
- Better PDF text can produce real quality gains.
  - On the controlled `NVDA 2024` + `AAPL 2024` slice, `cache_only` with LlamaParse improved recall from `0.4` to `0.8` with no precision loss.
  - evidence: `artifacts/analysis/20260222T221614Z-head_to_head_local_vs_cache/comparison.md`
- PyMuPDF cleanup policy tuning can help on the full board.
  - E2d quality mode improved both recall and precision over E2a.
- The repo now has a much better audit trail.
  - live status files, parse-cache manifests, cleanup audits, comparison summaries, and retrieved chunk traces make experiments easier to inspect and trust.

## Durable Negative Learnings

- Recall gains are easy to buy with bad precision.
  - Track A prompt relaxation improved recall but failed the precision guard.
  - Track B broader retrieval improved recall more, but precision dropped even harder.
  - Track D cross-run union also failed because precision collapsed.
- No single local PDF converter won across all hard docs.
  - `tsla/2023`, `nvda/2024`, and `aapl/2024` each preferred a different converter in the local side-quest.
  - Result: no evidence for a clean global local-parser swap.
- Conservative post-processing was not a useful fix.
  - G1 removed `0` targets in both runs and still hurt both recall and precision.
- Strong false-positive guardrails can overcorrect.
  - G2a reduced average false positives from `5.5` to `2.5` and improved precision, but recall dropped by `0.1607`.
- Trying to restore recall after tightening prompts can overshoot.
  - G2b recovered some recall, but precision dropped by `0.1076` and false positives rose to `10.0`.
- Retrieval reranking was not a win on the board.
  - H1 improved `meta.2024`, but average recall and precision both got worse.

## Problem-Specific Learnings

- `nvda.2024` is the clearest remaining hard case.
  - Key text is present in parsed and cached outputs.
  - The bigger issue looks like extraction/classification behavior, not plain text loss.
- `tsla.2023` has looked more like a conversion or representation problem.
  - It was a persistent miss early on and improved only after PDF-path changes.
- `aapl.2024` is reachable, but precision-sensitive.
  - The system can recover recall there, but looser retrieval or prompts can introduce false positives quickly.

## What The Failure Patterns Mean

- Most false negatives are not mainly caused by tiny field-matching issues.
- The dominant miss patterns were:
  - underpredicting scored targets
  - producing no predictions
  - labeling valid items as `non_target_claim`
- For `nvda.2024`, the strongest pattern was no prediction or wrong non-target labeling.

## Operational Learnings

- The precision guard is doing useful work.
  - It prevented the project from treating noisy recall gains as wins.
- Two-run averaging is worth the extra cost.
  - Several setups looked better in one run than the other; the average gave a more stable decision.
- Repairability matters in this workflow.
  - E2a cleanup failures were recoverable with targeted refill.
  - E2d strict preflight failures were recoverable with targeted warmup and cache materialization.
- Paid parse cost estimates are conservative.
  - Dry-run and command estimates were higher than observed live spend in admin.

## Current Boundary Of What Is Proven

- Proven on the full 14-doc board:
  - E2d quality mode is better than E2a and better than the shared comparator.
- Proven only on a controlled slice:
  - LlamaParse cache is better than local parsing for `NVDA 2024` + `AAPL 2024`.
- Not yet proven:
  - full-board gains from paid parse-cache
  - strict replay-mode validation for E2d
  - quality impact of prompt-cache experiments

## Best Short Reading Of The Project State

- The project has found one real full-board improvement: E2d quality mode.
- The project has ruled out several tempting but weak directions:
  - looser extraction prompts
  - broader retrieval
  - union scoring
  - simple deterministic dedupe
  - heavy FP-guardrail prompting
  - sentence-transformer reranking
- The next serious bottleneck is targeted extraction/classification behavior on hard docs, especially `nvda.2024`, not generic pipeline churn.
