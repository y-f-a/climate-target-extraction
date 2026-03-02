from pathlib import Path

from cte.index_registry import (
    IndexRegistry,
    build_source_manifest,
    compute_index_fingerprint,
)


def test_index_reuse_and_rebuild(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "aapl" / "2024"
    source_root.mkdir(parents=True)
    (source_root / "sample.pdf").write_bytes(b"fake-pdf-content")

    source_manifest = build_source_manifest(source_root)
    fingerprint = compute_index_fingerprint(
        pipeline_version="rag.v1",
        source_manifest=source_manifest,
        component_versions={
            "pdf_conversion": "v1",
            "node_splitting": "v1",
            "indexing": "v1",
            "retrieval": "v1",
            "evaluator": "v1",
        },
        component_settings={"chunk_size": 8192},
        embedding_model="text-embedding-3-large",
    )

    registry = IndexRegistry(tmp_path / "indexes")

    first = registry.select(
        pipeline_version="rag.v1",
        fingerprint=fingerprint,
        index_policy="reuse_or_build",
    )
    assert first.action == "built"
    (first.index_dir / "store").mkdir(parents=True)
    (first.index_dir / "store" / "index_store.json").write_text("{}", encoding="utf-8")

    registry.write_manifest(
        first,
        source_root=source_root,
        source_manifest=source_manifest,
        component_versions={"indexing": "v1"},
        component_settings={"chunk_size": 8192},
        embedding_model="text-embedding-3-large",
    )

    second = registry.select(
        pipeline_version="rag.v1",
        fingerprint=fingerprint,
        index_policy="reuse_or_build",
    )
    assert second.action == "reused"
    assert second.index_id == first.index_id

    third = registry.select(
        pipeline_version="rag.v1",
        fingerprint=fingerprint,
        index_policy="rebuild",
    )
    assert third.action == "rebuilt"
    assert third.index_id != first.index_id


def test_reuse_only_requires_materialized_store(tmp_path: Path) -> None:
    registry = IndexRegistry(tmp_path / "indexes")
    index_dir = tmp_path / "indexes" / "rag.v1" / "deadbeefdeadbeef"
    index_dir.mkdir(parents=True)
    (index_dir / "index_manifest.json").write_text(
        '{"index_fingerprint":"abc123"}',
        encoding="utf-8",
    )

    try:
        registry.select(
            pipeline_version="rag.v1",
            fingerprint="abc123",
            index_policy="reuse_only",
        )
    except FileNotFoundError as exc:
        assert "no materialized store" in str(exc).lower()
    else:
        raise AssertionError("Expected reuse_only to fail when index store is missing")


def test_reuse_or_build_ignores_malformed_manifest(tmp_path: Path) -> None:
    registry = IndexRegistry(tmp_path / "indexes")
    index_dir = tmp_path / "indexes" / "rag.v1" / "bad-manifest"
    index_dir.mkdir(parents=True)
    (index_dir / "index_manifest.json").write_text("{invalid", encoding="utf-8")

    selection = registry.select(
        pipeline_version="rag.v1",
        fingerprint="abc123",
        index_policy="reuse_or_build",
    )
    assert selection.action == "built"
