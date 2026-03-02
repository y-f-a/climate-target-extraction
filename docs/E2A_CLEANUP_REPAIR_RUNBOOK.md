# E2a Cleanup Repair Runbook

This procedure repairs `failed_open_runtime_error` cleanup pages for the E2a cleanup profile.

- profile key: `llm_faithful_v1-v1-d870e4019d28`
- health target: aggregate `failed_pages == 0` and `page_cache_write_errors == 0`

## Canonical Flow (Reproducible + Reusable)

Use this as the default cycle flow:

1. Build/refresh cleanup cache once (E2a build pass):

```bash
uv run cte run \
  --config configs/experiments/track_e_setup_e2a_pymupdf_cleanup.toml \
  --run-label track_e_setup_e2a_pymupdf_cleanup
```

2. Verify aggregate cleanup health (`cleanup_failed_pages == 0`, `cleanup_page_cache_write_errors == 0`).
3. If aggregate is not clean, run the Repair Steps in this runbook.
4. Run E2d quality mode (selection drift allowed):

```bash
uv run cte run \
  --config configs/experiments/track_e_setup_e2d_pymupdf_cleanup_numeric_off_quality.toml \
  --run-label track_e_setup_e2d_pymupdf_cleanup_numeric_off_quality
```

Reusability rule:

- once cache is healthy, reuse it for subsequent E2d-style runs instead of rebuilding from scratch.

Mode split:

- quality mode (primary): `pdf_cleanup_page_cache_miss_policy = "call_llm"` so selection drift is allowed.
- replay mode (secondary ops check): `pdf_cleanup_page_cache_miss_policy = "error"` for strict cache-only verification.

## Preconditions

1. Ensure the active E2a run has fully completed:

```bash
uv run python - <<'PY'
import json, pathlib
root = pathlib.Path("artifacts/experiments")
candidates = sorted(root.glob("*-track_e_setup_e2a_pymupdf_cleanup-*/manifest.json"))
if not candidates:
    raise SystemExit("No E2a manifests found for run label track_e_setup_e2a_pymupdf_cleanup.")
m = candidates[-1]
d = json.loads(m.read_text())
print("manifest:", m)
print("run_status:", d.get("run_status"))
print("finished_at_utc:", d.get("finished_at_utc"))
PY
```

Do not run repair while E2a is still `running`.

## Repair Steps

1. Detect failed per-PDF cleanup cache directories from current manifests (no fixed doc snapshot):

```bash
uv run python - <<'PY'
import json, pathlib
root = pathlib.Path("artifacts/parsed_docs/pymupdf_cleanup/llm_faithful_v1-v1-d870e4019d28")
out = pathlib.Path("/tmp/e2a_failed_cleanup_dirs.txt")
failed_dirs = []
for mp in sorted(root.glob("*/manifest.json")):
    d = json.loads(mp.read_text())
    failed = int(d.get("cleanup_failed_pages", 0) or 0)
    write_errors = int(d.get("cleanup_page_cache_write_errors", 0) or 0)
    if failed > 0 or write_errors > 0:
        failed_dirs.append(str(mp.parent))
print("failed_dirs:", len(failed_dirs))
for p in failed_dirs:
    print(p)
out.write_text("\\n".join(failed_dirs) + ("\\n" if failed_dirs else ""))
print("saved_list:", out)
PY
```

2. Remove only the detected failed per-PDF cleanup cache directories:

```bash
while IFS= read -r d; do
  [ -n "$d" ] || continue
  rm -rf "$d"
done < /tmp/e2a_failed_cleanup_dirs.txt
```

3. Run the targeted refill subset (GOOGL/NVDA/AMZN, 2023+2024):

```bash
uv run cte run \
  --config configs/experiments/track_e_setup_e2a_cleanup_refill_subset.toml \
  --run-label track_e_setup_e2a_cleanup_refill_subset \
  --skip-eval
```

If current failures include docs outside this subset, run an equivalent refill command covering those docs before continuing.

4. Verify aggregate cleanup failures are cleared for the full profile:

```bash
uv run python - <<'PY'
import json, pathlib
root = pathlib.Path("artifacts/parsed_docs/pymupdf_cleanup/llm_faithful_v1-v1-d870e4019d28")
failed = 0
write_errors = 0
docs = 0
for mp in root.glob("*/manifest.json"):
    d = json.loads(mp.read_text())
    docs += 1
    failed += int(d.get("cleanup_failed_pages", 0) or 0)
    write_errors += int(d.get("cleanup_page_cache_write_errors", 0) or 0)
print("docs:", docs)
print("cleanup_failed_pages:", failed)
print("cleanup_page_cache_write_errors:", write_errors)
PY
```

Success criteria:

- `cleanup_failed_pages == 0`
- `cleanup_page_cache_write_errors == 0`

After success, continue with the quality-mode run you actually want to score, then run the replay-mode strict check separately if you need cache-only verification.

## Notes

- Static failed-doc snapshots drift quickly. Detect failed docs/pages from current cleanup manifests and preflight outputs.
- Full rebuild retries are higher-risk for parser stalls. Default recovery path is targeted refill plus targeted warmup.
- Keep each recovery step as a separate artifact so the sequence can be audited and safely automated later.
- Identity fallback entries are acceptable only for replay readiness checks; they should not be used to score quality-mode experiments.
