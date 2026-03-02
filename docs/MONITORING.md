# Monitoring

This project uses artifact-based live monitoring for long-running jobs.

## Surfaces

- Dashboard (primary): `uv run cte dashboard`
- Terminal snapshot/watch: `uv run cte status` and `uv run cte status --watch`

Both surfaces read the same artifact files and are read-only.

## Live Status Artifacts

### Experiment runs (`cte run`)

Path: `artifacts/experiments/<run_id>/live_status.json`

Written throughout setup/extract/evaluate/finalize with:

- stage
- run counter (`run_counter_current`, `run_count_total`)
- extract/evaluate progress and current doc
- `updated_at_utc`
- final status (`completed` or `aborted`)

`manifest.json` also includes `live_status_path`.

### Parse-cache builds (`cte parse-cache build`)

Path: `artifacts/parsed_docs/_runs/<run_id>/live_status.json`

Written throughout the file loop with:

- processed/total
- counts (`hits`, `planned_new`, `parsed`, `failed`)
- current source PDF path
- cleanup aggregate counters
- final status (`completed` or `failed`)
- `summary_path` on completion

## Stalled Rule

- Stalled threshold is 20 minutes (`1200` seconds).
- Detection prefers `live_status.updated_at_utc`.
- If a live status timestamp is unavailable, dashboard falls back to manifest file mtime.

## Scope

- Applies to runs and parse-cache builds that write `live_status.json`.
- Existing historical runs without `live_status.json` still render in dashboard using manifest/report fallback logic.
