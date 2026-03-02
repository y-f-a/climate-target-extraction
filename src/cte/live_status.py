from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, write_json

LIVE_STATUS_SCHEMA_VERSION = "v1"
LIVE_STATUS_STALLED_AFTER_SECONDS = 20 * 60


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


class LiveStatusTracker:
    def __init__(
        self,
        *,
        path: Path,
        job_kind: str,
        run_id: str,
        started_at_utc: str | None = None,
        stalled_after_seconds: int = LIVE_STATUS_STALLED_AFTER_SECONDS,
        initial: dict[str, Any] | None = None,
    ) -> None:
        started = started_at_utc or _now_utc()
        payload: dict[str, Any] = {
            "schema_version": LIVE_STATUS_SCHEMA_VERSION,
            "job_kind": str(job_kind),
            "run_id": str(run_id),
            "status": "running",
            "started_at_utc": started,
            "updated_at_utc": started,
            "finished_at_utc": None,
            "stalled_after_seconds": int(stalled_after_seconds),
            "error_message": None,
        }
        if initial:
            payload.update(initial)
        payload["schema_version"] = LIVE_STATUS_SCHEMA_VERSION
        payload["job_kind"] = str(job_kind)
        payload["run_id"] = str(run_id)
        payload["stalled_after_seconds"] = int(stalled_after_seconds)
        payload.setdefault("status", "running")
        payload.setdefault("started_at_utc", started)
        payload.setdefault("updated_at_utc", started)
        payload.setdefault("finished_at_utc", None)
        payload.setdefault("error_message", None)

        self.path = path
        self._payload = payload
        self._write()

    @property
    def payload(self) -> dict[str, Any]:
        return dict(self._payload)

    def update(self, updates: dict[str, Any] | None = None) -> Path:
        if updates:
            self._payload.update(updates)
        self._payload["updated_at_utc"] = _now_utc()
        return self._write()

    def finalize(
        self,
        *,
        status: str,
        error_message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        if extra:
            self._payload.update(extra)
        self._payload["status"] = str(status)
        finished_at_utc = _now_utc()
        self._payload["finished_at_utc"] = finished_at_utc
        self._payload["updated_at_utc"] = finished_at_utc
        self._payload["error_message"] = error_message
        return self._write()

    def _write(self) -> Path:
        ensure_dir(self.path.parent)
        return write_json(self.path, self._payload)


def load_live_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
