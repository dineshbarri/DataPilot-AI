from __future__ import annotations

import re
from uuid import uuid4

import numpy as np
import pandas as pd

from datapilot.schemas import (
    DatasetProfile,
    Evidence,
    QualityIssue,
    Severity,
    TaskType,
)

LEAKAGE_PATTERNS = re.compile(
    r"(target|label|outcome|result|prediction|predicted|probability|score)$",
    re.IGNORECASE,
)


def infer_task_type(target: pd.Series) -> TaskType:
    unique = int(target.nunique(dropna=True))
    if (
        not pd.api.types.is_numeric_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or unique <= 20
        or unique / max(len(target), 1) < 0.05
    ):
        return TaskType.classification
    return TaskType.regression


def build_profile(frame: pd.DataFrame, target: str) -> DatasetProfile:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' is not present.")
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime = frame.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    missing_cells = int(frame.isna().sum().sum())
    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetime,
        duplicate_rows=int(frame.duplicated().sum()),
        missing_cells=missing_cells,
        missing_rate=round(missing_cells / max(frame.size, 1), 4),
        memory_mb=round(frame.memory_usage(deep=True).sum() / 1_048_576, 3),
        target=target,
        task_type=infer_task_type(frame[target]),
        target_cardinality=int(frame[target].nunique(dropna=True)),
    )


def audit_quality(
    frame: pd.DataFrame, profile: DatasetProfile
) -> tuple[list[QualityIssue], list[Evidence]]:
    issues: list[QualityIssue] = []
    evidence: list[Evidence] = []

    def add_evidence(claim: str, metric: str, value: object, source: str, method: str) -> str:
        evidence_id = f"EV-{uuid4().hex[:8].upper()}"
        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                claim=claim,
                metric=metric,
                value=value,
                source=source,
                method=method,
            )
        )
        return evidence_id

    missing_id = add_evidence(
        "Dataset missingness was measured across all cells.",
        "missing_rate",
        profile.missing_rate,
        "uploaded_dataset",
        "pandas.isna",
    )
    if profile.missing_rate > 0.2:
        issues.append(
            QualityIssue(
                code="HIGH_MISSINGNESS",
                severity=Severity.critical,
                message=f"{profile.missing_rate:.1%} of dataset cells are missing.",
                evidence_ids=[missing_id],
            )
        )
    elif profile.missing_rate > 0:
        issues.append(
            QualityIssue(
                code="MISSING_VALUES",
                severity=Severity.warning,
                message=f"{profile.missing_rate:.1%} of dataset cells are missing.",
                evidence_ids=[missing_id],
            )
        )

    duplicate_id = add_evidence(
        "Exact duplicate rows were counted before splitting.",
        "duplicate_rows",
        profile.duplicate_rows,
        "uploaded_dataset",
        "pandas.duplicated",
    )
    if profile.duplicate_rows:
        issues.append(
            QualityIssue(
                code="DUPLICATE_ROWS",
                severity=Severity.warning,
                message=f"{profile.duplicate_rows:,} exact duplicate rows can bias validation.",
                evidence_ids=[duplicate_id],
            )
        )

    target = frame[profile.target]
    target_missing = int(target.isna().sum())
    target_missing_id = add_evidence(
        "Rows with missing labels cannot be used for supervised training.",
        "missing_target_rows",
        target_missing,
        f"column:{profile.target}",
        "pandas.isna",
    )
    if target_missing:
        issues.append(
            QualityIssue(
                code="MISSING_TARGET",
                severity=Severity.critical,
                column=profile.target,
                message=f"{target_missing:,} rows have no target value and will be excluded.",
                evidence_ids=[target_missing_id],
            )
        )

    if profile.task_type == TaskType.classification:
        distribution = target.value_counts(normalize=True, dropna=True)
        minority_share = float(distribution.min()) if not distribution.empty else 0.0
        imbalance_id = add_evidence(
            "Class imbalance was measured using the minority-class share.",
            "minority_class_share",
            round(minority_share, 4),
            f"column:{profile.target}",
            "normalized value counts",
        )
        if minority_share < 0.1:
            issues.append(
                QualityIssue(
                    code="CLASS_IMBALANCE",
                    severity=Severity.warning,
                    column=profile.target,
                    message=f"Minority class represents only {minority_share:.1%} of labeled rows.",
                    evidence_ids=[imbalance_id],
                )
            )

    feature_frame = frame.drop(columns=[profile.target])
    for column in feature_frame.columns:
        normalized = column.strip().lower()
        leakage_risk = bool(LEAKAGE_PATTERNS.search(normalized))
        if feature_frame[column].nunique(dropna=True) == len(feature_frame):
            leakage_risk = leakage_risk or normalized.endswith(("_id", "id"))
        if leakage_risk:
            evidence_id = add_evidence(
                "A feature name or cardinality pattern may reveal the target or row identity.",
                "suspected_leakage_feature",
                column,
                f"column:{column}",
                "name and cardinality heuristic",
            )
            issues.append(
                QualityIssue(
                    code="LEAKAGE_RISK",
                    severity=Severity.warning,
                    column=column,
                    message=f"'{column}' may leak target or row identity; review before deployment.",
                    evidence_ids=[evidence_id],
                )
            )

    numeric = feature_frame.select_dtypes(include=np.number)
    for column in numeric.columns:
        series = numeric[column].dropna()
        if len(series) < 8:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_rate = float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean())
        if outlier_rate > 0.05:
            evidence_id = add_evidence(
                "Potential outliers were detected with the 1.5×IQR rule.",
                "outlier_rate",
                round(outlier_rate, 4),
                f"column:{column}",
                "Tukey IQR",
            )
            issues.append(
                QualityIssue(
                    code="OUTLIER_RATE",
                    severity=Severity.info,
                    column=column,
                    message=f"'{column}' has {outlier_rate:.1%} potential outliers.",
                    evidence_ids=[evidence_id],
                )
            )

    return issues, evidence


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, object]]:
    """Population stability index for numeric columns shared by two datasets."""
    reports: list[dict[str, object]] = []
    shared = reference.select_dtypes(include=np.number).columns.intersection(
        current.select_dtypes(include=np.number).columns
    )
    for column in shared:
        baseline = reference[column].dropna()
        observed = current[column].dropna()
        if baseline.nunique() < 2 or observed.empty:
            continue
        edges = np.unique(baseline.quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(edges) < 3:
            continue
        expected_counts, _ = np.histogram(baseline, bins=edges)
        actual_counts, _ = np.histogram(observed, bins=edges)
        expected = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
        actual = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        reports.append(
            {
                "column": column,
                "psi": round(psi, 4),
                "status": "high" if psi >= 0.25 else "moderate" if psi >= 0.1 else "stable",
            }
        )
    return reports
