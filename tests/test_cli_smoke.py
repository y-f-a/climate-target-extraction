import pytest

from cte.cli import build_parser


def test_cli_parses_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "configs/experiments/parity_rag_v1.toml",
            "--run-label",
            "smoke",
            "--resume-run-id",
            "20260222T000000Z-smoke-12345678",
            "--quiet",
        ]
    )
    assert args.command == "run"
    assert args.resume_run_id == "20260222T000000Z-smoke-12345678"
    assert args.quiet is True


def test_cli_parses_evaluate_command_with_quiet() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--pred-dir",
            "pred",
            "--ref-dir",
            "ref",
            "--out",
            "report.json",
            "--quiet",
        ]
    )
    assert args.command == "evaluate"
    assert args.quiet is True


def test_cli_parses_compare_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--baseline-report",
            "a.json",
            "--candidate-report",
            "b.json",
            "--out-dir",
            "out",
        ]
    )
    assert args.command == "compare"


def test_cli_run_rejects_removed_pipeline_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--pipeline",
                "rag",
                "--pipeline-version",
                "rag.v1",
                "--config",
                "configs/experiments/parity_rag_v1.toml",
                "--run-label",
                "smoke",
            ]
        )


def test_cli_parses_audit_existing_command_with_new_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "audit-existing",
            "--results-root",
            "data/results",
            "--out-dir",
            "artifacts/audit",
            "--experiments-root",
            "artifacts/experiments",
            "--run-notes-file",
            "artifacts/experiments/run_notes.toml",
        ]
    )
    assert args.command == "audit-existing"
    assert args.experiments_root == "artifacts/experiments"


def test_cli_parses_dashboard_command_with_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["dashboard"])
    assert args.command == "dashboard"
    assert args.experiments_root == "artifacts/experiments"
    assert args.parsed_docs_root == "artifacts/parsed_docs"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_cli_parses_status_command_with_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["status"])
    assert args.command == "status"
    assert args.experiments_root == "artifacts/experiments"
    assert args.parsed_docs_root == "artifacts/parsed_docs"
    assert args.watch is False
    assert args.interval_sec == 5.0


def test_cli_parses_suggest_baselines_command_with_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["suggest-baselines"])
    assert args.command == "suggest-baselines"
    assert args.experiments_root == "artifacts/experiments"
    assert args.baselines_file == "docs/BASELINES.md"


def test_cli_parses_parse_cache_build_command_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "parse-cache",
            "build",
            "--config",
            "configs/experiments/parity_rag_v1.toml",
        ]
    )
    assert args.command == "parse-cache"
    assert args.parse_cache_command == "build"
    assert args.execute is False
    assert args.max_new_pdfs is None
