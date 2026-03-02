from argparse import Namespace
from pathlib import Path

from cte import cli


def test_evaluate_uses_discovered_doc_names_for_sparse_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    for doc_name in ("aapl.2023.targets.v1.json", "msft.2024.targets.v1.json"):
        (pred_dir / doc_name).write_text('{"company": null, "targets": []}', encoding="utf-8")
        (ref_dir / doc_name).write_text('{"company": null, "targets": []}', encoding="utf-8")

    captured_doc_names: list[str] = []

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
        del pred_dir, reference_dir, eval_system_prompt, judge_model_name, judge_fn, progress_fn
        captured_doc_names.extend(doc_names)
        return {"aggregate": {}, "per_doc": []}

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluate_from_dirs should not be called for sparse auto-discovery")

    monkeypatch.setattr(cli, "evaluate_from_doc_names", fake_evaluate_from_doc_names)
    monkeypatch.setattr(cli, "evaluate_from_dirs", fail_if_called)
    monkeypatch.setattr(cli, "load_eval_prompt", lambda _: "prompt")

    args = Namespace(
        env_file=None,
        pred_dir=str(pred_dir),
        ref_dir=str(ref_dir),
        out=str(tmp_path / "report.json"),
        judge_model="judge-model",
        eval_prompt_version="v001",
        company_tickers=None,
        years=None,
    )

    exit_code = cli._evaluate_command(args)

    assert exit_code == 0
    assert captured_doc_names == ["aapl.2023.targets.v1.json", "msft.2024.targets.v1.json"]
