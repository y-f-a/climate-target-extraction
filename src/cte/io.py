from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_json_dumps(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(canonical_json_dumps(payload), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def target_doc_name(company_ticker: str, year: str, schema_version: str = "v1") -> str:
    return f"{company_ticker.lower()}.{year}.targets.{schema_version}.json"


def legacy_report_name(pipeline: str, model_alias: str, counter: int) -> str:
    if pipeline == "rag":
        return f"v2_{model_alias}_{counter}.json"
    return f"full_report_{model_alias}_{counter}.json"
