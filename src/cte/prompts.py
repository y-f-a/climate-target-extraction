from __future__ import annotations

from pathlib import Path

TEMPLATES_ROOT = Path("templates")
PROMPTS_ROOT = TEMPLATES_ROOT / "prompts"


def get_extract_prompt_path(
    pipeline: str,
    version: str,
    *,
    prompts_root: Path = PROMPTS_ROOT,
) -> Path:
    return prompts_root / pipeline / "extract" / f"{version}.txt"


def get_eval_prompt_path(version: str, *, prompts_root: Path = PROMPTS_ROOT) -> Path:
    return prompts_root / "eval" / "align_score" / f"{version}.txt"


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_extract_prompt(pipeline: str, version: str, *, prompts_root: Path = PROMPTS_ROOT) -> str:
    return load_prompt(get_extract_prompt_path(pipeline, version, prompts_root=prompts_root))


def load_eval_prompt(version: str, *, prompts_root: Path = PROMPTS_ROOT) -> str:
    return load_prompt(get_eval_prompt_path(version, prompts_root=prompts_root))
