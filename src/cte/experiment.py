from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunConfig, RunPaths, validate_lineage_requirements
from .io import ensure_dir, write_json


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(run_label: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_label)
    return f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(cwd: Path) -> dict[str, str | bool | None]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, cwd=cwd, text=True).strip()
            return out if out else None
        except Exception:
            return None

    try:
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
        )
        git_dirty: bool | None = bool(dirty_output.strip())
    except Exception:
        git_dirty = None

    return {
        "git_branch": _run(["git", "branch", "--show-current"]),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": git_dirty,
    }


def build_run_paths(config: RunConfig, run_id: str) -> RunPaths:
    experiment_dir = ensure_dir(config.artifacts_root / "experiments" / run_id)
    generated_targets_dir = ensure_dir(
        experiment_dir / "generated_targets" / config.pipeline / config.model_alias
    )
    results_dir = ensure_dir(experiment_dir / "results" / config.pipeline)
    analysis_dir = ensure_dir(experiment_dir / "analysis")
    indexes_root = ensure_dir(config.artifacts_root / "indexes")

    return RunPaths(
        experiment_dir=experiment_dir,
        generated_targets_dir=generated_targets_dir,
        results_dir=results_dir,
        analysis_dir=analysis_dir,
        indexes_root=indexes_root,
    )


def ensure_gate_lineage(config: RunConfig) -> None:
    validate_lineage_requirements(config)


def write_manifest(experiment_dir: Path, payload: dict[str, Any]) -> Path:
    return write_json(experiment_dir / "manifest.json", payload)


def append_experiment_log(artifacts_root: Path, payload: dict[str, Any]) -> Path:
    log_path = ensure_dir(artifacts_root / "experiments") / "experiment_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return log_path
