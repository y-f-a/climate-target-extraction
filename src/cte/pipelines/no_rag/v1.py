from __future__ import annotations

from pathlib import Path
from typing import Callable

from openai import OpenAI as OpenAIClient

from ...config import RunConfig
from ...io import target_doc_name, write_json
from ...schemas import ExtractedTargets

QUERY_PROMPT = (
    "Using only your general world knowledge, populate the schema with this company's climate "
    "emissions targets as of disclosure year {disclosure_year} for {company}. "
    "Focus on SBTi-style target definitions (near-term and long-term/net-zero), "
    "and classify each target as absolute or intensity. Capture scope coverage (Scopes 1, 2, 3) "
    "only when you are confident it applies to that target. "
    '(or "unknown" where required by the schema).'
)


def run_structured(
    *,
    client: OpenAIClient,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_output_tokens: int = 16390,
    reasoning_effort: str = "high",
) -> ExtractedTargets:
    response = client.responses.parse(
        model=model_name,
        reasoning={"effort": reasoning_effort},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ExtractedTargets,
        max_output_tokens=max_output_tokens,
    )
    return response.output_parsed


def run_batch(
    *,
    config: RunConfig,
    output_dir: Path,
    system_prompt: str,
    skip_doc_names: set[str] | None = None,
    progress_fn: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    client = OpenAIClient()
    written: list[Path] = []
    total_docs = len(config.company_tickers) * len(config.years)
    current_doc = 0

    for company_ticker in config.company_tickers:
        for year in config.years:
            ticker = company_ticker.lower()
            doc_name = target_doc_name(ticker, year)
            if skip_doc_names and doc_name in skip_doc_names:
                current_doc += 1
                if progress_fn is not None:
                    progress_fn(doc_name, current_doc, total_docs)
                continue
            user_prompt = QUERY_PROMPT.format(company=ticker, disclosure_year=year)
            model_obj = run_structured(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=config.model_name,
            )
            out_path = output_dir / doc_name
            write_json(out_path, model_obj.model_dump(mode="json"))
            written.append(out_path)
            current_doc += 1
            if progress_fn is not None:
                progress_fn(doc_name, current_doc, total_docs)

    return written
