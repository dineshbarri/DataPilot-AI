from __future__ import annotations

import json
from typing import Any

from datapilot.config import Settings
from datapilot.schemas import Evidence


def deterministic_insights(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    profile = state["profile"]
    best = state["model_bundle"].results[0]
    quality = state["quality_issues"]
    explainability = state["explainability"]
    important = list(explainability.feature_importance)[:3]
    summary = [
        (
            f"The analysis used {profile.rows:,} rows and {profile.columns:,} columns for a "
            f"{profile.task_type.value} task targeting '{profile.target}'."
        ),
        (
            f"{best.name} ranked first with training CV {best.primary_metric} {best.primary_score:.3f}; "
            f"its one-time test score was {best.final_test_score:.3f}."
        ),
        (
            f"{len(quality)} data-quality observations were recorded; "
            f"{sum(issue.severity.value == 'critical' for issue in quality)} are critical."
        ),
    ]
    if important:
        summary.append(
            f"The strongest predictive signals were {', '.join(important)} "
            f"according to {explainability.method.lower()}."
        )
    recommendations = [
        "Validate performance on fresh, out-of-time data before production deployment.",
        "Review suspected leakage and identifier columns with a domain owner.",
        "Monitor input drift and the primary metric after deployment.",
    ]
    if profile.missing_rate > 0.1:
        recommendations.insert(0, "Investigate upstream causes of missing data before retraining.")
    return summary, recommendations


def optional_llm_narrative(
    state: dict[str, Any], evidence: list[Evidence], settings: Settings
) -> list[str] | None:
    """Generate narrative only from bounded evidence; calculations remain deterministic."""
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        payload = {
            "profile": state["profile"].model_dump(),
            "best_model": state["model_bundle"].results[0].model_dump(),
            "critic": state["critic"].model_dump(),
            "evidence": [item.model_dump() for item in evidence[:25]],
        }
        prompt = (
            "You are a senior data scientist. Return exactly four concise markdown bullet points. "
            "Use only the JSON evidence below. Cite supporting evidence IDs in square brackets. "
            "Do not add numbers, causal claims, or facts absent from the payload.\n"
            + json.dumps(payload, default=str)
        )
        response = client.models.generate_content(model=settings.gemini_model, contents=prompt)
        lines = [line.strip("- ").strip() for line in response.text.splitlines() if line.strip()]
        return lines[:4] or None
    except Exception:
        return None


def answer_follow_up(run: dict[str, Any], question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in {"best model", "which model", "winner"}):
        top = run["model_results"][0]
        return (
            f"The best model was **{top['name']}**, with {top['primary_metric']} "
            f"training-CV **{top['selection_score']:.3f}** and one-time test "
            f"**{top['final_test_score']:.3f}**."
        )
    if any(token in lowered for token in {"feature", "important", "driver"}):
        importance = run["explainability"]["feature_importance"]
        top = list(importance.items())[:5]
        return (
            "Top predictive features: "
            + ", ".join(f"**{name}** ({value:.4f})" for name, value in top)
            + ". These are associations, not causal effects."
        )
    if any(token in lowered for token in {"quality", "missing", "leak", "risk"}):
        issues = run["quality_issues"]
        if not issues:
            return "No material quality flags were detected by the configured checks."
        return "Quality observations: " + "; ".join(item["message"] for item in issues[:6])
    if any(token in lowered for token in {"metric", "performance", "score"}):
        top = run["model_results"][0]
        formatted = ", ".join(
            f"{key}={value:.3f}" for key, value in top["final_test_metrics"].items()
        )
        return f"Selected-model one-time test metrics: {formatted}."
    return (
        "I can answer evidence-backed questions about the best model, performance metrics, "
        "data quality, leakage risk, and feature importance for this run."
    )
