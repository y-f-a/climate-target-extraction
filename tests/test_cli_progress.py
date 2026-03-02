from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cte import cli
from cte.config import RunConfig


def _make_run_config(tmp_path: Path) -> RunConfig:
    ref_dir = tmp_path / "reference_targets"
    ref_dir.mkdir(exist_ok=True)
    return RunConfig(
        pipeline="no_rag",
        pipeline_version="no_rag.v1",
        company_tickers=["AAPL"],
        years=["2024"],
        artifacts_root=tmp_path / "artifacts",
        reference_targets_dir=ref_dir,
    )


def _mock_run_dependencies(tmp_path: Path, monkeypatch) -> None:
    config = _make_run_config(tmp_path)
    monkeypatch.setattr(cli, "load_run_config", lambda _: config)
    monkeypatch.setattr(cli, "maybe_load_env_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "load_extract_prompt", lambda *_args, **_kwargs: "extract prompt")
    monkeypatch.setattr(cli, "load_eval_prompt", lambda *_args, **_kwargs: "eval prompt")
    monkeypatch.setattr(cli, "make_run_id", lambda *_args, **_kwargs: "20260210T120000Z-test-12345678")
    monkeypatch.setattr(
        cli,
        "git_metadata",
        lambda *_args, **_kwargs: {
            "git_branch": "main",
            "git_commit": "abc123",
            "git_dirty": False,
        },
    )

    from cte.pipelines.no_rag import v1 as no_rag_v1

    def fake_run_batch(
        *,
        config: RunConfig,
        output_dir: Path,
        system_prompt: str,
        progress_fn=None,
    ) -> list[Path]:
        del config, system_prompt
        doc_name = "aapl.2024.targets.v1.json"
        target_path = output_dir / doc_name
        target_path.write_text('{"company": null, "targets": []}', encoding="utf-8")
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return [target_path]

    monkeypatch.setattr(no_rag_v1, "run_batch", fake_run_batch)


def test_run_shows_progress_and_preserves_stdout_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def fake_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn
        if progress_fn is not None:
            progress_fn("aapl.2024.targets.v1.json", 1, 1)
        return {"aggregate": {}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", fake_evaluate_from_dirs)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "progress-test",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[cte][setup]" in captured.err
    assert "[cte][extract] run 1/1 start" in captured.err
    assert "[cte][extract] 1/1 aapl.2024.targets.v1.json" in captured.err
    assert "[cte][evaluate] 1/1 aapl.2024.targets.v1.json" in captured.err
    assert "[cte][finalize]" in captured.err

    out_lines = [line.strip() for line in captured.out.strip().splitlines()]
    assert len(out_lines) == 3
    assert out_lines[0].startswith("run_id=")
    assert out_lines[1].startswith("manifest=")
    assert out_lines[2].startswith("experiment_log=")

    run_id = "20260210T120000Z-test-12345678"
    manifest_path = tmp_path / "artifacts" / "experiments" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    live_status_path = Path(manifest["live_status_path"])
    assert live_status_path.exists()
    live_payload = json.loads(live_status_path.read_text(encoding="utf-8"))
    assert live_payload["job_kind"] == "experiment_run"
    assert live_payload["status"] == "completed"
    assert live_payload["stage"] == "finalize"
    assert live_payload["run_count_total"] == 1
    assert live_payload["run_counter_current"] == 1
    assert live_payload["extract_progress"]["total"] == 1
    assert live_payload["evaluate_progress"]["total"] == 1
    assert live_payload["finished_at_utc"] is not None


def test_run_writes_provenance_hashes_to_manifest_and_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def fake_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn
        if progress_fn is not None:
            progress_fn("aapl.2024.targets.v1.json", 1, 1)
        return {"aggregate": {}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", fake_evaluate_from_dirs)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "provenance-test",
        ]
    )
    capsys.readouterr()
    assert exit_code == 0

    run_id = "20260210T120000Z-test-12345678"
    manifest_path = tmp_path / "artifacts" / "experiments" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["provenance"]

    assert manifest["git_dirty"] is False
    assert manifest["prompt_cache_enabled"] is False
    assert manifest["prompt_cache_retention"] is None
    assert manifest["prompt_cache_scope"] is None
    assert manifest["target_postprocess_profile"] == "off"
    assert manifest["retrieval_rerank_profile"] == "off"
    assert provenance["extract_prompt_sha256"] == hashlib.sha256(
        b"extract prompt"
    ).hexdigest()
    assert provenance["eval_prompt_sha256"] == hashlib.sha256(b"eval prompt").hexdigest()
    assert provenance["extract_prompt_path"] == "templates/prompts/no_rag/extract/v001.txt"
    assert provenance["eval_prompt_path"] == "templates/prompts/eval/align_score/v001.txt"
    assert provenance["schema_path"].endswith("src/cte/schemas.py")
    assert provenance["schema_sha256"] == cli.sha256_file(Path(provenance["schema_path"]))

    log_path = tmp_path / "artifacts" / "experiments" / "experiment_log.jsonl"
    log_payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert log_payload["git_dirty"] is False
    assert log_payload["prompt_cache_enabled"] is False
    assert log_payload["prompt_cache_retention"] is None
    assert log_payload["prompt_cache_scope"] is None
    assert log_payload["provenance"] == provenance


def test_run_quiet_suppresses_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def fake_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn
        if progress_fn is not None:
            progress_fn("aapl.2024.targets.v1.json", 1, 1)
        return {"aggregate": {}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", fake_evaluate_from_dirs)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "progress-test",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""

    out_lines = [line.strip() for line in captured.out.strip().splitlines()]
    assert len(out_lines) == 3
    assert out_lines[0].startswith("run_id=")
    assert out_lines[1].startswith("manifest=")
    assert out_lines[2].startswith("experiment_log=")


def test_run_skip_eval_reports_skipped_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluate_from_dirs should not be called when --skip-eval is set")

    monkeypatch.setattr(cli, "evaluate_from_dirs", fail_if_called)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "progress-test",
            "--skip-eval",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[cte][evaluate] run 1/1 skipped (--skip-eval)" in captured.err
    assert "[cte][evaluate] 1/1 " not in captured.err

    out_lines = [line.strip() for line in captured.out.strip().splitlines()]
    assert len(out_lines) == 3
    assert out_lines[0].startswith("run_id=")
    assert out_lines[1].startswith("manifest=")
    assert out_lines[2].startswith("experiment_log=")


def _write_minimal_doc(path: Path) -> None:
    path.write_text('{"company": null, "targets": []}', encoding="utf-8")


def test_evaluate_shows_progress_and_preserves_stdout_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    out_path = tmp_path / "report.json"
    pred_dir.mkdir()
    ref_dir.mkdir()

    for doc_name in ("aapl.2023.targets.v1.json", "msft.2024.targets.v1.json"):
        _write_minimal_doc(pred_dir / doc_name)
        _write_minimal_doc(ref_dir / doc_name)

    monkeypatch.setattr(cli, "load_eval_prompt", lambda *_args, **_kwargs: "eval prompt")

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluate_from_dirs should not be called for auto-discovery mode")

    def fake_evaluate_from_doc_names(
        pred_dir: Path,
        reference_dir: Path,
        *,
        doc_names: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, eval_system_prompt, judge_model_name, judge_fn
        if progress_fn is not None:
            for idx, doc_name in enumerate(doc_names, start=1):
                progress_fn(doc_name, idx, len(doc_names))
        return {"aggregate": {}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", fail_if_called)
    monkeypatch.setattr(cli, "evaluate_from_doc_names", fake_evaluate_from_doc_names)

    exit_code = cli.main(
        [
            "evaluate",
            "--pred-dir",
            str(pred_dir),
            "--ref-dir",
            str(ref_dir),
            "--out",
            str(out_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[cte][setup]" in captured.err
    assert "[cte][discover] discovered 2 docs from prediction dir" in captured.err
    assert "[cte][evaluate] scoring 2 docs" in captured.err
    assert "[cte][evaluate] 1/2 aapl.2023.targets.v1.json" in captured.err
    assert "[cte][evaluate] 2/2 msft.2024.targets.v1.json" in captured.err

    out_lines = [line.strip() for line in captured.out.strip().splitlines()]
    assert len(out_lines) == 1
    assert out_lines[0] == f"wrote {out_path}"


def test_evaluate_quiet_suppresses_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    out_path = tmp_path / "report.json"
    pred_dir.mkdir()
    ref_dir.mkdir()
    _write_minimal_doc(pred_dir / "aapl.2023.targets.v1.json")
    _write_minimal_doc(ref_dir / "aapl.2023.targets.v1.json")

    monkeypatch.setattr(cli, "load_eval_prompt", lambda *_args, **_kwargs: "eval prompt")

    def fake_evaluate_from_doc_names(
        pred_dir: Path,
        reference_dir: Path,
        *,
        doc_names: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, eval_system_prompt, judge_model_name, judge_fn
        if progress_fn is not None:
            for idx, doc_name in enumerate(doc_names, start=1):
                progress_fn(doc_name, idx, len(doc_names))
        return {"aggregate": {}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_doc_names", fake_evaluate_from_doc_names)

    exit_code = cli.main(
        [
            "evaluate",
            "--pred-dir",
            str(pred_dir),
            "--ref-dir",
            str(ref_dir),
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"wrote {out_path}"


def test_run_failure_marks_manifest_aborted(tmp_path: Path, monkeypatch, capsys) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError as exc:
        assert str(exc) == "judge API timeout"
    else:
        raise AssertionError("Expected RuntimeError from evaluate step")

    run_id = "20260210T120000Z-test-12345678"
    manifest_path = tmp_path / "artifacts" / "experiments" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "aborted"
    assert manifest["failed_stage"] == "evaluate"
    assert manifest["error_message"] == "RuntimeError: judge API timeout"
    assert manifest["finished_at_utc"] is not None
    live_status_path = Path(manifest["live_status_path"])
    live_payload = json.loads(live_status_path.read_text(encoding="utf-8"))
    assert live_payload["status"] == "aborted"
    assert live_payload["stage"] == "abort"
    assert live_payload["error_message"] == "RuntimeError: judge API timeout"
    assert live_payload["finished_at_utc"] is not None

    log_path = tmp_path / "artifacts" / "experiments" / "experiment_log.jsonl"
    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 1
    log_payload = json.loads(log_lines[0])
    assert log_payload["run_status"] == "aborted"
    assert log_payload["failed_stage"] == "evaluate"

    captured = capsys.readouterr()
    assert "[cte][abort] run_status=aborted stage=evaluate error=RuntimeError" in captured.err


def test_run_resume_reuses_existing_predictions(tmp_path: Path, monkeypatch, capsys) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    from cte.pipelines.no_rag import v1 as no_rag_v1

    calls: list[set[str] | None] = []
    doc_name = "aapl.2024.targets.v1.json"

    def resumable_run_batch(
        *,
        config: RunConfig,  # noqa: ARG001
        output_dir: Path,
        system_prompt: str,  # noqa: ARG001
        skip_doc_names: set[str] | None = None,
        progress_fn=None,
    ) -> list[Path]:
        calls.append(set(skip_doc_names) if skip_doc_names else None)
        if skip_doc_names and doc_name in skip_doc_names:
            if progress_fn is not None:
                progress_fn(doc_name, 1, 1)
            return []
        _write_minimal_doc(output_dir / doc_name)
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return [output_dir / doc_name]

    monkeypatch.setattr(no_rag_v1, "run_batch", resumable_run_batch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    run_id = "20260210T120000Z-test-12345678"
    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected first run to abort during evaluate")

    def successful_evaluate_from_dirs(
        pred_dir: Path,  # noqa: ARG001
        reference_dir: Path,  # noqa: ARG001
        *,
        company_tickers: list[str],  # noqa: ARG001
        years: list[str],  # noqa: ARG001
        eval_system_prompt: str,  # noqa: ARG001
        judge_model_name: str,  # noqa: ARG001
        judge_fn=None,  # noqa: ARG001
        progress_fn=None,
    ) -> dict:
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return {"aggregate": {"micro_f1": 1.0}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", successful_evaluate_from_dirs)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "abort-test",
            "--resume-run-id",
            run_id,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "resuming run_id=20260210T120000Z-test-12345678 previous_status=aborted" in captured.err
    assert len(calls) == 1
    assert calls[0] is None
    assert "reused existing predictions (1/1)" in captured.err

    manifest_path = tmp_path / "artifacts" / "experiments" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["resumed_at_utc"] is not None
    assert isinstance(manifest.get("resume_history"), list)
    assert len(manifest["resume_history"]) >= 1
    assert manifest["runs"][0]["counter"] == 1
    assert manifest["runs"][0]["prediction_files"] == 1
    assert manifest["runs"][0]["report_path"]


def test_run_resume_rejects_scope_mismatch(tmp_path: Path, monkeypatch) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    run_id = "20260210T120000Z-test-12345678"
    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected first run to abort during evaluate")

    mismatch_payload = _make_run_config(tmp_path).model_dump()
    mismatch_payload["years"] = ["2023"]
    mismatch_config = RunConfig.model_validate(mismatch_payload)
    monkeypatch.setattr(cli, "load_run_config", lambda _: mismatch_config)

    with pytest.raises(ValueError, match="Resume manifest config mismatch"):
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
                "--resume-run-id",
                run_id,
            ]
        )


def test_run_resume_rejects_target_postprocess_profile_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    run_id = "20260210T120000Z-test-12345678"
    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected first run to abort during evaluate")

    mismatch_payload = _make_run_config(tmp_path).model_dump()
    mismatch_payload["target_postprocess_profile"] = "fp_dedupe_conservative_v1"
    mismatch_config = RunConfig.model_validate(mismatch_payload)
    monkeypatch.setattr(cli, "load_run_config", lambda _: mismatch_config)

    with pytest.raises(ValueError, match="Resume manifest config mismatch"):
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
                "--resume-run-id",
                run_id,
            ]
        )


def test_run_resume_rejects_retrieval_rerank_profile_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    run_id = "20260210T120000Z-test-12345678"
    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected first run to abort during evaluate")

    mismatch_payload = _make_run_config(tmp_path).model_dump()
    mismatch_payload["retrieval_rerank_profile"] = "sentence_transformer_default_v1"
    mismatch_config = RunConfig.model_validate(mismatch_payload)
    monkeypatch.setattr(cli, "load_run_config", lambda _: mismatch_config)

    with pytest.raises(ValueError, match="Resume manifest config mismatch"):
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
                "--resume-run-id",
                run_id,
            ]
        )


def test_run_resume_does_not_treat_unexpected_docs_as_complete(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    from cte.pipelines.no_rag import v1 as no_rag_v1

    calls: list[set[str] | None] = []
    doc_name = "aapl.2024.targets.v1.json"

    def resumable_run_batch(
        *,
        config: RunConfig,  # noqa: ARG001
        output_dir: Path,
        system_prompt: str,  # noqa: ARG001
        skip_doc_names: set[str] | None = None,
        progress_fn=None,
    ) -> list[Path]:
        calls.append(set(skip_doc_names) if skip_doc_names else None)
        if skip_doc_names and doc_name in skip_doc_names:
            if progress_fn is not None:
                progress_fn(doc_name, 1, 1)
            return []
        _write_minimal_doc(output_dir / doc_name)
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return [output_dir / doc_name]

    monkeypatch.setattr(no_rag_v1, "run_batch", resumable_run_batch)

    def failing_evaluate_from_dirs(
        pred_dir: Path,
        reference_dir: Path,
        *,
        company_tickers: list[str],
        years: list[str],
        eval_system_prompt: str,
        judge_model_name: str,
        judge_fn=None,
        progress_fn=None,
    ) -> dict:
        del pred_dir, reference_dir, company_tickers, years, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        raise RuntimeError("judge API timeout")

    monkeypatch.setattr(cli, "evaluate_from_dirs", failing_evaluate_from_dirs)

    run_id = "20260210T120000Z-test-12345678"
    try:
        cli.main(
            [
                "run",
                "--config",
                "configs/experiments/parity_no_rag_v1.toml",
                "--run-label",
                "abort-test",
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected first run to abort during evaluate")
    capsys.readouterr()

    pred_dir = (
        tmp_path
        / "artifacts"
        / "experiments"
        / run_id
        / "generated_targets"
        / "no_rag"
        / "gpt5_2"
        / "run_1"
    )
    expected_doc = pred_dir / doc_name
    expected_doc.unlink()
    _write_minimal_doc(pred_dir / "msft.2024.targets.v1.json")

    def successful_evaluate_from_dirs(
        pred_dir: Path,  # noqa: ARG001
        reference_dir: Path,  # noqa: ARG001
        *,
        company_tickers: list[str],  # noqa: ARG001
        years: list[str],  # noqa: ARG001
        eval_system_prompt: str,  # noqa: ARG001
        judge_model_name: str,  # noqa: ARG001
        judge_fn=None,  # noqa: ARG001
        progress_fn=None,
    ) -> dict:
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return {"aggregate": {"micro_f1": 1.0}, "per_doc": []}

    monkeypatch.setattr(cli, "evaluate_from_dirs", successful_evaluate_from_dirs)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "abort-test",
            "--resume-run-id",
            run_id,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is None
    assert expected_doc.exists()
    assert "reused existing predictions" not in captured.err


def test_run_resume_re_evaluates_when_report_doc_set_mismatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _mock_run_dependencies(tmp_path, monkeypatch)

    doc_name = "aapl.2024.targets.v1.json"
    evaluate_call_count = {"count": 0}

    def evaluate_with_single_doc(
        pred_dir: Path,  # noqa: ARG001
        reference_dir: Path,  # noqa: ARG001
        *,
        company_tickers: list[str],  # noqa: ARG001
        years: list[str],  # noqa: ARG001
        eval_system_prompt: str,  # noqa: ARG001
        judge_model_name: str,  # noqa: ARG001
        judge_fn=None,  # noqa: ARG001
        progress_fn=None,
    ) -> dict:
        evaluate_call_count["count"] += 1
        if progress_fn is not None:
            progress_fn(doc_name, 1, 1)
        return {
            "aggregate": {"micro_f1": 1.0},
            "per_doc": [
                {
                    "doc": doc_name,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "field_accuracy": {},
                }
            ],
        }

    monkeypatch.setattr(cli, "evaluate_from_dirs", evaluate_with_single_doc)

    run_id = "20260210T120000Z-test-12345678"
    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "report-reuse-test",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    report_path = (
        tmp_path
        / "artifacts"
        / "experiments"
        / run_id
        / "results"
        / "no_rag"
        / "full_report_gpt5_2_1.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "aggregate": {"micro_f1": 0.1},
                "per_doc": [
                    {
                        "doc": "msft.2024.targets.v1.json",
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "field_accuracy": {},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/experiments/parity_no_rag_v1.toml",
            "--run-label",
            "report-reuse-test",
            "--resume-run-id",
            run_id,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert evaluate_call_count["count"] == 2
    assert "reused existing report" not in captured.err

    refreshed_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert refreshed_report["per_doc"][0]["doc"] == doc_name
