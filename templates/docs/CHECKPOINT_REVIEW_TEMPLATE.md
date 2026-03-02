# 5-Setup Checkpoint Review Template

Run this review every 5 completed setups (or sooner if all active tracks are paused).

## Checkpoint Info
- checkpoint_id:
- timestamp_utc:
- model_generation:
- evaluator_setup:
  - judge_model_name:
  - eval_prompt_version:

## Required Review Items

### 1) Top setup per active track
| track_id | top_setup_id | top_run_id | recall_avg | precision_avg | note |
|---|---|---|---:|---:|---|
| - | - | - | - | - | - |

### 2) Paused/blocked states
| track_id | status | reason | reopen_condition |
|---|---|---|---|
| - | - | - | - |

### 3) Model-specific leaderboard status
| rank | track_id | run_id | recall_avg | precision_avg | precision_guard_reference |
|---:|---|---|---:|---:|---|
| 1 | - | - | - | - | - |

### 4) No-RAG anchor check
- no_rag_anchor_run_id:
- no_rag_anchor_current_for_model: `true|false`
- action_if_false:

### 5) Decisions
- continue_current_tracks: `yes|no`
- pause_tracks:
- open_new_track: `yes|no`
- close_cycle: `yes|no`
- decision_notes:
