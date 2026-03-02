# BASELINES

Manual baseline registry for dashboard and review workflows.

## Rules
- Maintain this file manually.
- Use explicit `baseline_run_id` values (do not infer from labels).
- Update `updated_at_utc` whenever a row changes.
- Keep one active row per `(pipeline_version, model_generation)` unless you intentionally split tracks.

## Baseline Registry

| baseline_key | pipeline_version | model_generation | baseline_run_id | status | updated_at_utc | notes |
|---|---|---|---|---|---|---|
| no_rag_anchor | no_rag.v1 | gpt-5.2-2025-12-11 | 20260210T095905Z-gate1_no_rag-080e1808 | active | 2026-03-02T00:00:00+00:00 | Reference no-RAG anchor. |
| rag_parity_baseline | rag.v1 | gpt-5.2-2025-12-11 | 20260221T134430Z-parity_rag_v1_baseline_retry1-af7f13bb | active | 2026-03-02T00:00:00+00:00 | Shared RAG comparator baseline. |
