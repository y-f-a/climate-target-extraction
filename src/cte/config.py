from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from .target_postprocess import TargetPostprocessProfile
from .retrieval_rerank import RetrievalRerankProfile

PipelineName = Literal["rag", "no_rag"]
IndexPolicy = Literal["reuse_or_build", "reuse_only", "rebuild"]
PromptCacheRetention = Literal["in-memory", "24h"]

DEFAULT_COMPANY_INFO: dict[str, str] = {
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation",
    "AAPL": "Apple Inc",
    "GOOGL": "Alphabet Inc",
    "AMZN": "Amazon.com Inc",
    "META": "Meta Platforms Inc",
    "TSLA": "Tesla Inc",
}

DEFAULT_YEARS: list[str] = ["2023", "2024"]
DEFAULT_MODELS: dict[str, str] = {
    "gpt5": "gpt-5-2025-08-07",
    "gpt5_1": "gpt-5.1-2025-11-13",
    "gpt5_2": "gpt-5.2-2025-12-11",
}


class RunConfig(BaseModel):
    pipeline: PipelineName
    pipeline_version: str
    model_alias: str = "gpt5_2"
    model_name: str = DEFAULT_MODELS["gpt5_2"]
    judge_model_name: str = "gpt-5-mini-2025-08-07"
    openai_prompt_cache_enabled: bool = False
    openai_prompt_cache_retention: PromptCacheRetention = "in-memory"
    target_postprocess_profile: TargetPostprocessProfile = "off"
    retrieval_rerank_profile: RetrievalRerankProfile = "off"

    years: list[str] = Field(default_factory=lambda: list(DEFAULT_YEARS))
    company_tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_COMPANY_INFO.keys()))
    company_info: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_COMPANY_INFO))

    prompt_versions: dict[str, str] = Field(
        default_factory=lambda: {
            "extract": "v001",
            "eval": "v001",
        }
    )

    component_versions: dict[str, str] = Field(
        default_factory=lambda: {
            "pdf_conversion": "v1",
            "node_splitting": "v1",
            "indexing": "v1",
            "retrieval": "v1",
            "evaluator": "v1",
        }
    )

    component_settings: dict[str, Any] = Field(default_factory=dict)

    artifacts_root: Path = Path("artifacts")
    parsed_docs_root: Path = Path("artifacts/parsed_docs")
    reference_targets_dir: Path = Path("data/evaluation_set/reference_targets")
    source_docs_root_env: str = "CTE_SOURCE_DOCS_DIR"
    env_file: Path | None = Path(".env.local")

    run_count: int = 1
    index_policy: IndexPolicy = "reuse_or_build"

    parent_run_id: str | None = None
    changed_components: list[str] = Field(default_factory=list)
    change_reason: str | None = None
    baseline_run_id: str | None = None

    @model_validator(mode="after")
    def _validate_model(self) -> "RunConfig":
        self.company_tickers = [ticker.upper() for ticker in self.company_tickers]
        if self.run_count < 1:
            raise ValueError("run_count must be >= 1")
        if not self.pipeline_version.startswith(f"{self.pipeline}."):
            raise ValueError(
                f"pipeline_version '{self.pipeline_version}' must start with '{self.pipeline}.'"
            )
        missing_company_info = [
            ticker for ticker in self.company_tickers if ticker not in self.company_info
        ]
        if missing_company_info:
            raise ValueError(
                "company_info is missing entries for: " + ", ".join(sorted(missing_company_info))
            )
        return self


class RunPaths(BaseModel):
    experiment_dir: Path
    generated_targets_dir: Path
    results_dir: Path
    analysis_dir: Path
    indexes_root: Path


def load_run_config(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    paths = raw.get("paths", {})
    prompts = raw.get("prompts", {})
    versions = raw.get("versions", {})

    def _blank_to_none(value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    payload: dict[str, Any] = {
        "pipeline": run.get("pipeline"),
        "pipeline_version": run.get("pipeline_version"),
        "model_alias": run.get("model_alias", "gpt5_2"),
        "model_name": run.get("model_name", DEFAULT_MODELS["gpt5_2"]),
        "judge_model_name": run.get("judge_model_name", "gpt-5-mini-2025-08-07"),
        "openai_prompt_cache_enabled": run.get("openai_prompt_cache_enabled", False),
        "openai_prompt_cache_retention": run.get("openai_prompt_cache_retention", "in-memory"),
        "target_postprocess_profile": run.get("target_postprocess_profile", "off"),
        "retrieval_rerank_profile": run.get("retrieval_rerank_profile", "off"),
        "years": run.get("years", list(DEFAULT_YEARS)),
        "company_tickers": run.get("company_tickers", list(DEFAULT_COMPANY_INFO.keys())),
        "run_count": run.get("run_count", 1),
        "index_policy": run.get("index_policy", "reuse_or_build"),
        "parent_run_id": _blank_to_none(run.get("parent_run_id")),
        "changed_components": run.get("changed_components", []),
        "change_reason": _blank_to_none(run.get("change_reason")),
        "baseline_run_id": _blank_to_none(run.get("baseline_run_id")),
        "artifacts_root": Path(paths.get("artifacts_root", "artifacts")),
        "parsed_docs_root": Path(paths.get("parsed_docs_root", "artifacts/parsed_docs")),
        "reference_targets_dir": Path(
            paths.get("reference_targets_dir", "data/evaluation_set/reference_targets")
        ),
        "source_docs_root_env": paths.get("source_docs_root_env", "CTE_SOURCE_DOCS_DIR"),
        "env_file": _blank_to_none(paths.get("env_file", ".env.local")),
        "prompt_versions": {
            "extract": prompts.get("extract_version", "v001"),
            "eval": prompts.get("eval_version", "v001"),
        },
        "component_versions": {
            "pdf_conversion": versions.get("pdf_conversion", "v1"),
            "node_splitting": versions.get("node_splitting", "v1"),
            "indexing": versions.get("indexing", "v1"),
            "retrieval": versions.get("retrieval", "v1"),
            "evaluator": versions.get("evaluator", "v1"),
        },
        "component_settings": raw.get("settings", {}),
    }

    company_info = raw.get("company_info")
    if company_info:
        payload["company_info"] = company_info

    return RunConfig.model_validate(payload)


def apply_overrides(
    config: RunConfig,
    *,
    index_policy: IndexPolicy | None = None,
) -> RunConfig:
    payload = config.model_dump()
    if index_policy is not None:
        payload["index_policy"] = index_policy
    return RunConfig.model_validate(payload)


def resolve_source_docs_root(config: RunConfig) -> Path:
    value = os.getenv(config.source_docs_root_env)
    if not value:
        raise EnvironmentError(
            f"Missing source docs env var '{config.source_docs_root_env}'. Set it to the disclosure root path."
        )
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source docs root does not exist: {path}")
    return path


def maybe_load_env_file(env_file: Path | None, *, override: bool = False) -> Path | None:
    if env_file is None:
        return None

    candidate = env_file.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        return None

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value

    return candidate


def is_non_parity_run(config: RunConfig) -> bool:
    if config.pipeline_version not in {"rag.v1", "no_rag.v1"}:
        return True
    return bool(config.changed_components or config.parent_run_id)


def validate_lineage_requirements(config: RunConfig) -> None:
    if is_non_parity_run(config):
        if not config.parent_run_id:
            raise ValueError("Non-parity runs require parent_run_id")
        if not config.changed_components:
            raise ValueError("Non-parity runs require changed_components")
