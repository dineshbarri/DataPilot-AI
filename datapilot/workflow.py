from __future__ import annotations

import time
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from datapilot.config import Settings, get_settings
from datapilot.data import duckdb_overview
from datapilot.insights import deterministic_insights, optional_llm_narrative
from datapilot.modeling import (
    TrainingBundle,
    critic_decision,
    explain_model,
    train_models,
)
from datapilot.observability import log_to_mlflow
from datapilot.persistence import ArtifactStore, RunStore
from datapilot.quality import audit_quality, build_profile
from datapilot.reports import export_artifacts
from datapilot.schemas import AnalysisPlan, RunSummary, TaskType


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_name: str
    frame: pd.DataFrame
    target: str
    settings: Settings
    profile: Any
    quality_issues: list[Any]
    evidence: list[Any]
    eda: dict[str, Any]
    statistics: dict[str, Any]
    plan: AnalysisPlan
    feature_plan: dict[str, Any]
    model_bundle: TrainingBundle
    critic: Any
    explainability: Any
    executive_summary: list[str]
    recommendations: list[str]
    summary_payload: dict[str, Any]
    artifacts: dict[str, str]
    trace: list[dict[str, Any]]


def _trace(state: AgentState, agent: str, started: float, detail: str) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append(
        {
            "agent": agent,
            "status": "completed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "detail": detail,
        }
    )
    return trace


def data_quality_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    profile = build_profile(state["frame"], state["target"])
    issues, evidence = audit_quality(state["frame"], profile)
    return {
        "profile": profile,
        "quality_issues": issues,
        "evidence": evidence,
        "trace": _trace(
            state, "Data Quality Agent", started, f"Recorded {len(issues)} quality observations."
        ),
    }


def eda_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    frame = state["frame"]
    overview = duckdb_overview(frame)
    overview["numeric_summary"] = (
        frame.select_dtypes(include=np.number).describe().round(4).to_dict()
    )
    overview["categorical_cardinality"] = {
        column: int(frame[column].nunique(dropna=True))
        for column in frame.select_dtypes(exclude=np.number).columns
    }
    return {
        "eda": overview,
        "trace": _trace(state, "EDA Agent", started, "Computed DuckDB-backed dataset overview."),
    }


def statistical_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    frame = state["frame"]
    target = state["target"]
    numeric = frame.select_dtypes(include=np.number)
    correlations: dict[str, float] = {}
    if target in numeric.columns and len(numeric.columns) > 1:
        correlations = (
            numeric.corr(numeric_only=True)[target]
            .drop(labels=[target])
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .round(4)
            .to_dict()
        )
    statistics = {
        "top_absolute_target_correlations": correlations,
        "target_distribution": frame[target].value_counts(dropna=False).head(20).to_dict(),
    }
    return {
        "statistics": statistics,
        "trace": _trace(
            state,
            "Statistical Analysis Agent",
            started,
            "Measured target distribution and associations.",
        ),
    }


def planning_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    profile = state["profile"]
    if profile.task_type == TaskType.classification:
        metric = "balanced_accuracy"
        candidates = [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
            "XGBoost (when installed)",
        ]
    else:
        metric = "r2"
        candidates = [
            "Linear Regression",
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
            "XGBoost (when installed)",
        ]
    plan = AnalysisPlan(
        objective=f"Predict '{profile.target}' and produce reproducible, evidence-backed insights.",
        target=profile.target,
        task_type=profile.task_type,
        primary_metric=metric,
        validation_strategy="Training-only stratified cross-validation; untouched final test evaluation"
        if profile.task_type == TaskType.classification
        else "Training-only cross-validation; untouched final test evaluation",
        candidate_models=candidates,
        risk_controls=[
            "Drop rows with missing target before split",
            "Fit imputers, encoders, and scalers on training folds only",
            "Flag leakage-like names and identifier cardinality",
            "Require critic quality gate before explanation",
        ],
    )
    return {
        "plan": plan,
        "trace": _trace(state, "Planning Agent", started, f"Selected {metric} as primary metric."),
    }


def feature_engineering_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    profile = state["profile"]
    feature_plan = {
        "numeric": "Median imputation followed by standard scaling",
        "categorical": "Most-frequent imputation followed by unknown-safe one-hot encoding",
        "fit_scope": "Preprocessing is fitted inside each sklearn Pipeline after splitting",
        "dropped": ["exact duplicate rows", "rows with missing target"],
        "feature_count": profile.columns - 1,
    }
    return {
        "feature_plan": feature_plan,
        "trace": _trace(
            state,
            "Feature Engineering Agent",
            started,
            "Created leakage-safe ColumnTransformer plan.",
        ),
    }


def modeling_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    retry = state.get("model_bundle").retry_number + 1 if state.get("model_bundle") else 0
    bundle = train_models(
        state["frame"],
        state["target"],
        state["profile"].task_type,
        state["settings"],
        retry_number=retry,
    )
    return {
        "model_bundle": bundle,
        "trace": _trace(
            state,
            "Modeling Agent",
            started,
            f"Compared {len(bundle.results)} models; {bundle.best_model} ranked first.",
        ),
    }


def evaluation_critic_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    decision = critic_decision(state["model_bundle"], state["profile"].task_type, state["settings"])
    detail = "Approved analysis." if decision.approved else "Rejected analysis and requested retry."
    return {
        "critic": decision,
        "trace": _trace(state, "Evaluation / Critic Agent", started, detail),
    }


def critic_route(state: AgentState) -> str:
    return "explainability" if state["critic"].approved else "retry_modeling"


def explainability_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    result = explain_model(state["model_bundle"])
    return {
        "explainability": result,
        "trace": _trace(
            state, "Explainability Agent", started, f"Generated {result.method} explanations."
        ),
    }


def executive_insights_agent(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    summary, recommendations = deterministic_insights(state)
    llm_summary = optional_llm_narrative(state, state["evidence"], state["settings"])
    if llm_summary:
        summary = llm_summary
    return {
        "executive_summary": summary,
        "recommendations": recommendations,
        "trace": _trace(
            state,
            "Executive Insights Agent",
            started,
            "Created evidence-grounded narrative with deterministic metric provenance.",
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("data_quality", data_quality_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("statistics", statistical_agent)
    graph.add_node("planning", planning_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("modeling", modeling_agent)
    graph.add_node("critic", evaluation_critic_agent)
    graph.add_node("explainability", explainability_agent)
    graph.add_node("executive_insights", executive_insights_agent)
    graph.add_edge(START, "data_quality")
    graph.add_edge("data_quality", "eda")
    graph.add_edge("eda", "statistics")
    graph.add_edge("statistics", "planning")
    graph.add_edge("planning", "feature_engineering")
    graph.add_edge("feature_engineering", "modeling")
    graph.add_edge("modeling", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_route,
        {"retry_modeling": "modeling", "explainability": "explainability"},
    )
    graph.add_edge("explainability", "executive_insights")
    graph.add_edge("executive_insights", END)
    return graph.compile()


def run_analysis(
    frame: pd.DataFrame,
    target: str,
    dataset_name: str,
    settings: Settings | None = None,
) -> RunSummary:
    settings = settings or get_settings()
    run_id = f"run_{uuid4().hex[:12]}"
    store = RunStore(settings)
    artifact_store = ArtifactStore(settings.artifact_root)
    store.save(run_id, dataset_name, "running", {"run_id": run_id, "status": "running"})
    initial: AgentState = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "frame": frame,
        "target": target,
        "settings": settings,
        "trace": [],
    }
    try:
        final = build_graph().invoke(initial)
        payload = _summary_payload(final, run_id, dataset_name)
        final["summary_payload"] = payload
        log_to_mlflow(payload, settings)
        artifacts = export_artifacts(run_id, final, artifact_store.run_directory(run_id))
        payload["artifacts"] = artifacts
        payload["status"] = "completed"
        store.save(run_id, dataset_name, "completed", payload)
        return RunSummary.model_validate(payload)
    except Exception as exc:
        store.save(
            run_id,
            dataset_name,
            "failed",
            {"run_id": run_id, "dataset_name": dataset_name, "status": "failed", "error": str(exc)},
        )
        raise
    finally:
        store.engine.dispose()


def _summary_payload(state: AgentState, run_id: str, dataset_name: str) -> dict[str, Any]:
    bundle = state["model_bundle"]
    return {
        "run_id": run_id,
        "status": "completed",
        "dataset_name": dataset_name,
        "profile": state["profile"].model_dump(mode="json"),
        "plan": state["plan"].model_dump(mode="json"),
        "quality_issues": [item.model_dump(mode="json") for item in state["quality_issues"]],
        "evidence": [item.model_dump(mode="json") for item in state["evidence"]],
        "model_results": [item.model_dump(mode="json") for item in bundle.results],
        "model_failures": [item.model_dump(mode="json") for item in bundle.failures],
        "best_model": bundle.best_model,
        "critic": state["critic"].model_dump(mode="json"),
        "explainability": state["explainability"].model_dump(mode="json"),
        "executive_summary": state["executive_summary"],
        "recommendations": state["recommendations"],
        "artifacts": {},
        "trace": state["trace"],
    }
