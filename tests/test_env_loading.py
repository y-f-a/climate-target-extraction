from pathlib import Path

import os

from cte.config import maybe_load_env_file


def test_maybe_load_env_file_loads_values(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\nCTE_SOURCE_DOCS_DIR=/tmp/docs\n", encoding="utf-8"
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CTE_SOURCE_DOCS_DIR", raising=False)

    loaded = maybe_load_env_file(env_file)
    assert loaded == env_file
    assert os.environ.get("OPENAI_API_KEY") == "sk-test"
    assert os.environ.get("CTE_SOURCE_DOCS_DIR") == "/tmp/docs"


def test_maybe_load_env_file_does_not_override_by_default(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    maybe_load_env_file(env_file)

    assert os.environ.get("OPENAI_API_KEY") == "from-env"
