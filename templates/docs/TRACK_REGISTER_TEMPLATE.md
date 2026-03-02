# Track Register Template

Use this file to track active and historical experiment tracks.

## Header
- model_generation:
- evaluator_setup:
  - judge_model_name:
  - eval_prompt_version:
- no_rag_anchor_run_id:
- no_rag_anchor_metrics:
  - micro_recall:
  - micro_precision:

## Tracks

| track_id | status | created_at_utc | baseline_run_id | baseline_setup | current_best_run_id | current_best_recall | current_best_precision | non_improving_count | blocked_reason | notes |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| rag_track_001 | active | - | - | - | - | - | - | 0 | - | - |

## Status Values
- `active`
- `paused`
- `blocked`
- `closed`

## Rules Reminder
- Max 3 active tracks.
- One main change per setup inside a track.
- Pause after 2 consecutive non-improving setups.
- Reopen only with a clearly different one-sentence hypothesis.
