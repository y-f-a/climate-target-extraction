from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetrievalRerankProfile = Literal["off", "sentence_transformer_default_v1"]


@dataclass(frozen=True)
class RetrievalRerankSpec:
    profile: RetrievalRerankProfile
    model: str
    device: str
    candidate_top_k: int
    top_n: int


def resolve_retrieval_rerank_spec(
    *,
    profile: RetrievalRerankProfile | str,
    fusion_top_k: int,
) -> RetrievalRerankSpec | None:
    if profile == "off":
        return None
    if profile != "sentence_transformer_default_v1":
        raise ValueError(f"Unsupported retrieval_rerank_profile: {profile}")

    top_n = max(1, int(fusion_top_k))
    return RetrievalRerankSpec(
        profile="sentence_transformer_default_v1",
        model="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
        candidate_top_k=max(top_n, 20),
        top_n=top_n,
    )
