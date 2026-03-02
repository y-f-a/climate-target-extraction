from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .io import ensure_dir, write_json

IndexAction = Literal["reused", "built", "rebuilt"]


@dataclass(frozen=True)
class IndexSelection:
    action: IndexAction
    index_id: str
    index_fingerprint: str
    index_dir: Path


@dataclass(frozen=True)
class SourceFileInfo:
    relative_path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(source_root: Path) -> list[SourceFileInfo]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source path not found: {source_root}")

    files: list[SourceFileInfo] = []
    for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        files.append(
            SourceFileInfo(
                relative_path=str(path.relative_to(source_root)),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    if not files:
        raise FileNotFoundError(f"No source files found in {source_root}")
    return files


def compute_index_fingerprint(
    *,
    pipeline_version: str,
    source_manifest: list[SourceFileInfo],
    component_versions: dict[str, str],
    component_settings: dict[str, Any],
    embedding_model: str,
    parsed_cache_content_fingerprint: str | None = None,
) -> str:
    payload = {
        "pipeline_version": pipeline_version,
        "source_manifest": [item.__dict__ for item in source_manifest],
        "component_versions": component_versions,
        "component_settings": component_settings,
        "embedding_model": embedding_model,
        "parsed_cache_content_fingerprint": parsed_cache_content_fingerprint,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class IndexRegistry:
    def __init__(self, indexes_root: Path) -> None:
        self.indexes_root = ensure_dir(indexes_root)

    def _pipeline_root(self, pipeline_version: str) -> Path:
        return ensure_dir(self.indexes_root / pipeline_version)

    @staticmethod
    def _has_materialized_store(index_dir: Path) -> bool:
        store_dir = index_dir / "store"
        return store_dir.is_dir() and any(store_dir.iterdir())

    def _find_existing(self, pipeline_version: str, fingerprint: str) -> list[Path]:
        root = self._pipeline_root(pipeline_version)
        hits: list[Path] = []
        for item in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest_path = item / "index_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if manifest.get("index_fingerprint") == fingerprint:
                hits.append(item)
        return hits

    def select(
        self,
        *,
        pipeline_version: str,
        fingerprint: str,
        index_policy: str,
    ) -> IndexSelection:
        matches = self._find_existing(pipeline_version, fingerprint)
        reusable_matches = [path for path in matches if self._has_materialized_store(path)]

        if index_policy == "reuse_only":
            if not reusable_matches:
                if matches:
                    raise FileNotFoundError(
                        "Matching index manifest found, but no materialized store exists under reuse_only policy"
                    )
                raise FileNotFoundError(
                    "No frozen index found for requested fingerprint under reuse_only policy"
                )
            index_dir = reusable_matches[-1]
            return IndexSelection(
                action="reused",
                index_id=index_dir.name,
                index_fingerprint=fingerprint,
                index_dir=index_dir,
            )

        if index_policy == "reuse_or_build" and reusable_matches:
            index_dir = reusable_matches[-1]
            return IndexSelection(
                action="reused",
                index_id=index_dir.name,
                index_fingerprint=fingerprint,
                index_dir=index_dir,
            )

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        if index_policy == "rebuild":
            index_id = f"{fingerprint[:16]}-{timestamp}"
            action: IndexAction = "rebuilt"
        else:
            index_id = f"{fingerprint[:16]}"
            action = "built"

        index_dir = self._pipeline_root(pipeline_version) / index_id
        ensure_dir(index_dir)

        return IndexSelection(
            action=action,
            index_id=index_id,
            index_fingerprint=fingerprint,
            index_dir=index_dir,
        )

    def write_manifest(
        self,
        selection: IndexSelection,
        *,
        source_root: Path,
        source_manifest: list[SourceFileInfo],
        component_versions: dict[str, str],
        component_settings: dict[str, Any],
        embedding_model: str,
        parsed_cache: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "index_id": selection.index_id,
            "index_action": selection.action,
            "index_fingerprint": selection.index_fingerprint,
            "source_root": str(source_root),
            "source_manifest": [item.__dict__ for item in source_manifest],
            "component_versions": component_versions,
            "component_settings": component_settings,
            "embedding_model": embedding_model,
            "parsed_cache": parsed_cache,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        return write_json(selection.index_dir / "index_manifest.json", payload)
