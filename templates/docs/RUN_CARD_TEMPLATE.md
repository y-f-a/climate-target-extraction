# Run Card Template

Use one run card per completed setup (2-run average).

## Identity
- track_id:
- setup_id:
- model_generation:
- evaluator_setup:
  - judge_model_name:
  - eval_prompt_version:

## Required Fields
- what_changed:
- why:
- expected_effect:
- actual_result:
- keep_or_drop:

## Experiment Links
- config_path:
- run_command:
- manifest_path:
- report_run_1_path:
- report_run_2_path:
- comparison_path:

## Metrics
- baseline_recall:
- baseline_precision:
- run_1_recall:
- run_1_precision:
- run_2_recall:
- run_2_precision:
- recall_avg:
- precision_avg:

## Decision Check
- recall_improved: `true|false`
- precision_guard_pass: `true|false` (candidate precision >= baseline precision - 0.01)
- tie_break_used: `true|false`
- decision_note:
