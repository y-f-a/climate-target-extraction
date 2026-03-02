from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, conint, confloat

TargetHorizon = Literal["near_term", "long_term", "net_zero"]
MetricType = Literal["absolute", "intensity"]
Ambition = Literal["1.5C", "well_below_2C", "2C", "unspecified"]
Status = Literal["approved", "committed", "in_validation", "expired", "unknown"]
TargetType = Literal["sbti_near_term", "sbti_net_zero", "non_target_claim"]


class Target(BaseModel):
    title: Optional[str] = Field(None, description="Human-friendly label if present in text")
    target_type: TargetType = Field(..., description="Classify each item")
    horizon: TargetHorizon = Field(..., description="near_term / long_term / net_zero (SBTi framing)")
    metric_type: MetricType = Field(..., description="Absolute or intensity target")
    scopes_covered: list[Literal["S1", "S2", "S3"]] = Field(
        ...,
        description="Which scopes this target covers",
    )
    scope3_categories: Optional[list[conint(ge=1, le=15)]] = None
    ambition: Ambition = Field(..., description="Declared temperature alignment (if stated)")
    coverage_pct: Optional[confloat(ge=0, le=100)] = None

    base_year: Optional[int] = None
    target_year: Optional[int] = None
    reduction_pct: Optional[confloat(ge=0, le=100)] = None
    base_value: Optional[float] = None
    target_value: Optional[float] = None
    unit: Optional[str] = None

    status: Status = Field(..., description="SBTi status if mentioned")
    boundary: Optional[str] = None
    notes: Optional[str] = None
    sources: list[Optional[str]] = Field(
        ...,
        description="Doc names/URLs/node IDs where this target was found",
    )


class ExtractedTargets(BaseModel):
    company: Optional[str] = None
    targets: list[Target] = Field(default_factory=list)


FIELDS_TO_SCORE: list[str] = [
    "ambition",
    "base_year",
    "horizon",
    "metric_type",
    "reduction_pct",
    "scopes_covered",
    "target_year",
    "target_value",
    "unit",
]

Grade = Literal["EXACT", "PARTIAL", "WRONG"]


class FieldScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Grade
    note: str


class FieldScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambition: FieldScore
    base_year: FieldScore
    horizon: FieldScore
    metric_type: FieldScore
    reduction_pct: FieldScore
    scopes_covered: FieldScore
    target_year: FieldScore
    target_value: FieldScore
    unit: FieldScore


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gold_index: int
    pred_index: int
    field_scores: FieldScores


class ScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[Match]
    unmatched_gold: list[int]
    unmatched_pred: list[int]


def target_payload(obj: ExtractedTargets | dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj, ExtractedTargets):
        return obj.model_dump(mode="json")
    return obj
