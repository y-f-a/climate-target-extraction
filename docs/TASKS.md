# TASKS

- Canonical open task board: `docs/TASKS.md`
- Completed and superseded work: `docs/archive/TASKS_ARCHIVE_2026-02-23.md`
- Status values: `todo`, `in_progress`, `blocked`
- Required fields per task: `id`, `title`, `status`, `owner`, `target_date`, `evidence_link`, `notes`
- `target_date` is optional for all statuses
- `evidence_link` is required for `blocked`; optional otherwise
- Owner must be a single primary owner (`user`, `codex`, or named person)
- Tasks move manually between sections
- When a task is completed or superseded, move it to the archive board instead of keeping it here
- ID format is sequential: `T-001`, `T-002`, ...

## Next

| id | title | status | owner | target_date | evidence_link | notes |
|---|---|---|---|---|---|---|
| T-035 | Run replay-mode strict cache-only check for E2d (`miss_policy=error`) as operational validation | todo | user | - | - | Run `uv run cte run --config configs/experiments/track_e_setup_e2d_pymupdf_cleanup_numeric_off.toml --run-label track_e_setup_e2d_pymupdf_cleanup_numeric_off`; this is an ops reproducibility check (non-primary quality gate). |

## Later

| id | title | status | owner | target_date | evidence_link | notes |
|---|---|---|---|---|---|---|
| T-027 | Implement Track F0 prompt-cache toggle for RAG extraction only (`RunConfig`, prompt-cache key helper, RAG wiring, manifest provenance, docs/tests) | todo | codex | - | - | Deferred until post-Track-E/H cycle. |
| T-028 | Execute Track F0 paired canary runs (`off` vs `on`) on `NVDA,AAPL` 2024 and capture cache-usage evidence | todo | user | - | - | Deferred until post-Track-E/H cycle. |
| T-029 | Build Track F0 comparison summary (run_1 vs run_1, run_2 vs run_2) and declare pass/fail against quality+cache gate | todo | codex | - | - | Deferred until post-Track-E/H cycle. |
| T-030 | Conditionally launch full Track F (14-doc, run_count=2) only if T-029 passes | todo | user | - | - | Deferred; gated on T-029 pass. |
