# EXPERIMENTAL_STRATEGY

## Status
- This is the only active strategy document.
- Previous strategy drafts are deprecated and should be ignored.

## Purpose
Operational strategy for fast, low-jargon RAG experimentation on the fixed dataset.

## Scope
- Dataset stays fixed and immutable.
- Applies to RAG architecture experimentation and model-specific leaderboards.
- Keeps `no_rag` as a reference anchor, not a winner.

## Core Decision Rules
- Primary winner signal: higher recall.
- Precision guard: candidate precision must not drop by more than `0.01` vs current best comparator.
- Improvement threshold: recall gain must be `> 0.00`.
- Recall tie-breaker: higher precision wins.
- Cost policy: always report cost, but do not use cost as a winner gate.

## Experiment Unit
- One "setup" = one hypothesis with one main change.
- Every setup must run twice.
- Official score = average of run 1 and run 2 for recall and precision.
- A setup is "completed" only after both runs finish and the average is computed.

## Track Model
- Use tracks to organize ideas.
- Each track starts with an explicit baseline setup.
- Inside a track, every new setup changes exactly one main thing.
- New tracks are opened case-by-case for major shifts.
- Model change always starts a new track.
- Maximum active tracks at once: `3`.

## Comparator Rules
- In-track decisions compare against the current best setup in the same track.
- Cross-track final decision compares each track winner head-to-head.
- Final head-to-head guard compares challengers against the current leading RAG run in that model leaderboard (precision drop max `0.01`).

## Leaderboards and Reset Logic
- Keep separate leaderboards by model generation.
- `no_rag` anchor exists per model generation and is reference-only.
- If base model changes:
  - rerun the `no_rag` anchor for the new model generation,
  - keep prior RAG baselines as historical (no forced full RAG reset),
  - do not mix old/new model runs in one leaderboard.
- Evaluator setup (`judge model` + `eval prompt`) is fixed within a leaderboard.
- If evaluator setup changes, start a new leaderboard.

## Scheduling and Flow
- Run scheduling across active tracks: round-robin.
- If a track is blocked, skip it and continue round-robin on unblocked tracks.
- Revisit blocked tracks only after an explicit unblock note.
- Review checkpoint cadence: every `5` completed setups (or sooner if all active tracks are paused).

## Pause and Reopen Rules
- Pause a track after `2` consecutive non-improving completed setups, or precision-guard failures.
- Technical failures do not count as non-improving setups:
  - retry same setup once,
  - if it fails again, mark track/setup blocked.
- Reopen a paused track only with a clearly different hypothesis, written in one sentence.

## Evaluation Policy
- Evaluate every run on the full fixed 14-document set.
- No screening subset stage.
- No formal confidence math in decision gates.
- Stability comes from mandatory two-run averaging for every setup.

## Keep/Drop Decision Algorithm
For each completed setup in a track:

1. Compute:
   - `recall_avg = (run1_recall + run2_recall) / 2`
   - `precision_avg = (run1_precision + run2_precision) / 2`
2. Compare to current track best:
   - pass if `recall_avg > best_recall` and `precision_avg >= best_precision - 0.01`
3. If recall ties, pick higher precision.
4. If pass, update track best immediately.
5. Mark setup as `keep` or `drop` with reason.

## Run Card (Required for Every Setup)
Use this fixed short card:
- `what_changed`
- `why`
- `expected_effect`
- `actual_result`
- `keep_or_drop`

## Manual Review Checklist (Every 5 Completed Setups)
- Confirm top setup per active track.
- Confirm paused/blocked track states.
- Confirm model-specific leaderboard status.
- Confirm no-RAG anchor is up to date for current model generation.
- Decide whether to continue, pause, open new track, or close cycle.

## Notes on Tradeoffs
- Mandatory two-run scoring improves stability but increases cost.
- Cost is intentionally excluded from winner gating to maximize learning signal.
- Simplicity is preserved by using only recall + precision guard + fixed operational rules.
