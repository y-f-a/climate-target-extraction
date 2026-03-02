from __future__ import annotations

from types import SimpleNamespace

from cte import cli


def _snapshot_with_active_rows() -> SimpleNamespace:
    active_run = SimpleNamespace(
        run_id="20260223T000000Z-track_e_setup_e2a-12345678",
        run_status="running",
        stalled=False,
        live_stage="extract",
        live_run_counter_current=1,
        live_run_counter_total=2,
        live_extract_completed=7,
        live_extract_total=14,
        live_evaluate_completed=0,
        live_evaluate_total=14,
        live_updated_at_utc="2026-02-23T08:30:00+00:00",
        timestamp_utc="2026-02-23T08:30:00+00:00",
    )
    active_parse_cache = SimpleNamespace(
        run_id="20260223T083000000000Z-parse_cache_build",
        status="running",
        stalled=False,
        mode="execute",
        processed=4,
        total=12,
        hits=2,
        planned_new=0,
        parsed=2,
        failed=0,
        current_source_relative_path="nvda/2024/FY2024-NVIDIA-Corporate-Sustainability-Report.pdf",
        updated_at_utc="2026-02-23T08:30:01+00:00",
    )
    return SimpleNamespace(
        generated_at_utc="2026-02-23T08:30:02+00:00",
        active_runs=[active_run],
        parse_cache_active_runs=[active_parse_cache],
    )


def test_status_command_prints_active_experiment_and_parse_cache_rows(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_status_snapshot",
        lambda **kwargs: _snapshot_with_active_rows(),  # noqa: ARG005
    )

    exit_code = cli.main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "active_experiment_runs=1" in captured.out
    assert "active_parse_cache_jobs=1" in captured.out
    assert "stage=extract" in captured.out
    assert "progress=4/12" in captured.out


def test_status_watch_mode_handles_keyboard_interrupt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_load_status_snapshot",
        lambda **kwargs: _snapshot_with_active_rows(),  # noqa: ARG005
    )

    def interrupt_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)

    exit_code = cli.main(["status", "--watch", "--interval-sec", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status watch stopped" in captured.out
