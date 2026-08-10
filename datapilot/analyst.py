from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET_WORDS = {
    "target": 1.0,
    "label": 1.0,
    "outcome": 0.95,
    "class": 0.9,
    "churn": 0.95,
    "fraud": 0.95,
    "default": 0.9,
    "price": 0.8,
    "revenue": 0.75,
    "sales": 0.75,
    "diagnosis": 0.9,
    "status": 0.7,
    "response": 0.8,
    "converted": 0.9,
}
PII_PATTERN = re.compile(r"(email|phone|mobile|address|ssn|passport|account|name)", re.I)


@dataclass
class DatasetBrief:
    fingerprint: str
    rows: int
    columns: int
    numeric: int
    categorical: int
    datetime: int
    missing_cells: int
    duplicate_rows: int
    memory_mb: float


def inspect_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute an immediate analyst profile without requiring a target."""
    numeric = list(frame.select_dtypes(include=np.number).columns)
    datetime = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical = [c for c in frame.columns if c not in numeric and c not in datetime]
    missing = frame.isna().sum()
    brief = DatasetBrief(
        fingerprint=hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest()[:12],
        rows=len(frame),
        columns=len(frame.columns),
        numeric=len(numeric),
        categorical=len(categorical),
        datetime=len(datetime),
        missing_cells=int(missing.sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        memory_mb=round(float(frame.memory_usage(deep=True).sum() / 1_048_576), 2),
    )
    dictionary = build_data_dictionary(frame)
    quality_score = max(
        0,
        round(
            100
            - (brief.missing_cells / max(1, frame.size)) * 45
            - (brief.duplicate_rows / max(1, brief.rows)) * 25
            - sum(dictionary["issue_count"].clip(upper=3)) / max(1, len(dictionary)) * 4
        ),
    )
    return {
        "brief": brief,
        "dictionary": dictionary,
        "targets": rank_target_candidates(frame),
        "quality_score": quality_score,
        "missing": missing.sort_values(ascending=False),
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime,
        "correlation": frame[numeric].corr(numeric_only=True)
        if len(numeric) > 1
        else pd.DataFrame(),
    }


def build_data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an evidence-based data dictionary for every column."""
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        unique, missing = int(series.nunique(dropna=True)), int(series.isna().sum())
        issues: list[str] = []
        if missing:
            issues.append("Missing values")
        if unique <= 1:
            issues.append("Constant")
        if len(frame) and unique / len(frame) > 0.98:
            issues.append("Identifier-like")
        if PII_PATTERN.search(str(column)):
            issues.append("Potential PII")
        role = "Numeric feature" if pd.api.types.is_numeric_dtype(series) else "Categorical feature"
        if pd.api.types.is_datetime64_any_dtype(series):
            role = "Datetime"
        elif unique == len(frame) and len(frame) > 10:
            role = "Identifier"
        rows.append(
            {
                "column": str(column),
                "type": str(series.dtype),
                "role": role,
                "unique": unique,
                "missing": missing,
                "missing_%": round(missing / max(1, len(frame)) * 100, 2),
                "example_values": ", ".join(map(str, series.dropna().astype(str).unique()[:3]))[
                    :90
                ],
                "issues": ", ".join(issues) or "None detected",
                "issue_count": len(issues),
            }
        )
    return pd.DataFrame(rows)


def rank_target_candidates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank targets while explicitly leaving confirmation to the user."""
    ranked: list[dict[str, Any]] = []
    for position, column in enumerate(frame.columns):
        series, name = frame[column], str(column).lower().strip()
        cardinality = int(series.nunique(dropna=True))
        score = max((v for word, v in TARGET_WORDS.items() if word in name), default=0.0)
        if 2 <= cardinality <= max(20, int(len(frame) * 0.1)):
            score += 0.22
        if position == len(frame.columns) - 1:
            score += 0.12
        if cardinality >= max(10, int(len(frame) * 0.95)):
            score -= 0.45
        if series.isna().mean() > 0.5 or cardinality < 2:
            score -= 0.5
        task = "classification" if cardinality <= max(20, int(len(frame) * 0.05)) else "regression"
        if pd.api.types.is_numeric_dtype(series) and cardinality > 20:
            task = "regression"
        ranked.append(
            {
                "column": str(column),
                "task": task,
                "confidence": round(max(0, min(score, 0.99)), 2),
                "reason": f"{cardinality:,} distinct values; "
                + ("name/position signals detected" if score > 0.3 else "weak heuristic evidence"),
            }
        )
    return sorted(ranked, key=lambda item: item["confidence"], reverse=True)[:5]


def ai_context(frame: pd.DataFrame, profile: dict[str, Any], excluded: list[str]) -> dict[str, Any]:
    """Build a bounded, redacted payload suitable for AI interpretation."""
    safe = frame.drop(columns=[c for c in excluded if c in frame], errors="ignore").copy()
    pii = [c for c in safe.columns if PII_PATTERN.search(str(c))]
    safe = safe.drop(columns=pii, errors="ignore")
    return {
        "shape": list(frame.shape),
        "columns": profile["dictionary"].drop(columns=["issue_count"]).to_dict(orient="records"),
        "numeric_summary": safe.select_dtypes(include=np.number).describe().round(3).to_dict(),
        "sample": safe.head(3).replace({np.nan: None}).to_dict(orient="records"),
        "excluded_columns": sorted(set(excluded + pii)),
        "quality_score": profile["quality_score"],
        "target_candidates": profile["targets"],
    }


def gemini_dataset_summary(
    frame: pd.DataFrame, profile: dict[str, Any], api_key: str, model: str, excluded: list[str]
) -> str:
    """Ask Gemini for an evidence-bounded analyst narrative."""
    if not api_key:
        raise ValueError("Enter a Gemini API key to generate AI interpretation.")
    try:
        from google import genai
    except ImportError as exc:
        raise ValueError("Install the AI extra: pip install google-genai") from exc
    prompt = """You are DataPilot, a rigorous senior data analyst. Use ONLY the supplied JSON.
Return concise Markdown with exactly these headings: Finding, Evidence, Interpretation,
Limitation, Recommendation. Explain likely row grain and useful business questions, but label
uncertain semantics as assumptions. Never invent values, origin, or causal claims.
DATA:\n""" + json.dumps(ai_context(frame, profile, excluded), default=str)
    try:
        response = genai.Client(api_key=api_key).models.generate_content(
            model=model, contents=prompt
        )
        return response.text
    except Exception as exc:
        raise ValueError(f"Gemini request failed: {str(exc)[:240]}") from exc


def dataframe_csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
