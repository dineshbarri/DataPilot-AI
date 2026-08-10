from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    classification = "classification"
    regression = "regression"


class Severity(StrEnum):
    info = "info"
    warning = "warning"
    critical = "critical"


class Evidence(BaseModel):
    evidence_id: str
    claim: str
    metric: str
    value: float | int | str
    source: str
    method: str


class QualityIssue(BaseModel):
    code: str
    severity: Severity
    column: str | None = None
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    duplicate_rows: int
    missing_cells: int
    missing_rate: float
    memory_mb: float
    target: str
    task_type: TaskType
    target_cardinality: int


class AnalysisPlan(BaseModel):
    objective: str
    target: str
    task_type: TaskType
    primary_metric: str
    validation_strategy: str
    candidate_models: list[str]
    risk_controls: list[str]


class ModelResult(BaseModel):
    name: str
    primary_metric: str
    primary_score: float
    metrics: dict[str, float]
    cross_validation_mean: float | None = None
    cross_validation_std: float | None = None
    training_seconds: float


class CriticDecision(BaseModel):
    approved: bool
    score: float
    threshold: float
    reasons: list[str]
    retry_number: int


class ExplainabilityResult(BaseModel):
    method: str
    feature_importance: dict[str, float]
    caveats: list[str]


class RunSummary(BaseModel):
    run_id: str
    status: str
    dataset_name: str
    profile: DatasetProfile
    plan: AnalysisPlan
    quality_issues: list[QualityIssue]
    evidence: list[Evidence]
    model_results: list[ModelResult]
    best_model: str
    critic: CriticDecision
    explainability: ExplainabilityResult
    executive_summary: list[str]
    recommendations: list[str]
    artifacts: dict[str, str]
    trace: list[dict[str, Any]]
