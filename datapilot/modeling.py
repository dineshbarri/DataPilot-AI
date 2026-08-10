from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datapilot.config import Settings
from datapilot.schemas import (
    CriticDecision,
    ExplainabilityResult,
    ModelFailure,
    ModelResult,
    TaskType,
)
from datapilot.tuning import tune_random_forest


@dataclass
class TrainingBundle:
    pipeline: Pipeline
    results: list[ModelResult]
    best_model: str
    test_features: pd.DataFrame
    test_target: pd.Series
    retry_number: int
    failures: list[ModelFailure]


logger = logging.getLogger(__name__)


def _preprocessor(features: pd.DataFrame, settings: Settings) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=2,
                    max_categories=settings.max_categories_per_feature,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_models(task: TaskType, settings: Settings, retry: int) -> dict[str, BaseEstimator]:
    if task == TaskType.classification:
        models: dict[str, BaseEstimator] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1_000, class_weight="balanced", random_state=settings.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=180 + retry * 80,
                min_samples_leaf=max(1, 2 - retry),
                class_weight="balanced",
                n_jobs=-1,
                random_state=settings.random_state,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=180 + retry * 80,
                class_weight="balanced",
                n_jobs=-1,
                random_state=settings.random_state,
            ),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=180 + retry * 80,
                min_samples_leaf=max(1, 2 - retry),
                n_jobs=-1,
                random_state=settings.random_state,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=180 + retry * 80,
                n_jobs=-1,
                random_state=settings.random_state,
            ),
        }
    try:
        if task == TaskType.classification:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=160 + retry * 60,
                max_depth=4 + retry,
                learning_rate=0.07,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=settings.random_state,
            )
        else:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=160 + retry * 60,
                max_depth=4 + retry,
                learning_rate=0.07,
                n_jobs=-1,
                random_state=settings.random_state,
            )
    except ImportError:
        pass
    return models


def train_models(
    frame: pd.DataFrame,
    target: str,
    task: TaskType,
    settings: Settings,
    retry_number: int = 0,
) -> TrainingBundle:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    clean = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    features = clean.drop(columns=[target]).copy()
    labels = clean[target].copy()
    if features.empty:
        raise ValueError("No usable feature columns remain after removing the target.")
    if labels.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two distinct non-null values.")
    categorical = features.select_dtypes(exclude=np.number)
    estimated_width = len(features.select_dtypes(include=np.number).columns) + sum(
        min(int(categorical[column].nunique(dropna=True)), settings.max_categories_per_feature)
        for column in categorical.columns
    )
    if estimated_width > settings.max_encoded_features:
        raise ValueError(
            f"Estimated encoded width {estimated_width:,} exceeds the safe limit of "
            f"{settings.max_encoded_features:,}. Reduce high-cardinality columns or increase "
            "MAX_ENCODED_FEATURES after reviewing memory capacity."
        )
    stratify = (
        labels if task == TaskType.classification and labels.value_counts().min() >= 2 else None
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=stratify,
    )
    results: list[ModelResult] = []
    failures: list[ModelFailure] = []
    cv = min(5, max(2, int(len(x_train) / 20)))
    if task == TaskType.classification:
        smallest_class = int(y_train.value_counts().min())
        cv = min(cv, smallest_class) if smallest_class >= 2 else 0
        scoring = "balanced_accuracy"
    else:
        scoring = "r2"
    if cv < 2:
        raise ValueError("The training partition is too small for reliable cross-validation.")

    candidates = _candidate_models(task, settings, retry_number)
    tuned_parameters = tune_random_forest(
        task,
        _preprocessor(x_train, settings),
        x_train,
        y_train,
        settings,
        cv,
    )
    if tuned_parameters:
        candidates["Random Forest"].set_params(**tuned_parameters)

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", _preprocessor(x_train, settings)), ("model", estimator)]
        )
        started = time.perf_counter()
        try:
            cv_scores = (
                cross_val_score(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
                if cv >= 2
                else np.array([])
            )
            result = ModelResult(
                name=name,
                primary_metric="balanced_accuracy" if task == TaskType.classification else "r2",
                primary_score=float(cv_scores.mean()),
                metrics={},
                cross_validation_mean=float(cv_scores.mean()) if len(cv_scores) else None,
                cross_validation_std=float(cv_scores.std()) if len(cv_scores) else None,
                training_seconds=round(time.perf_counter() - started, 3),
                selection_score=float(cv_scores.mean()) if len(cv_scores) else None,
            )
            results.append(result)
        except Exception as exc:
            logger.warning(
                "Candidate %s failed during cross-validation: %s", name, type(exc).__name__
            )
            failures.append(
                ModelFailure(
                    name=name,
                    stage="cross_validation",
                    exception_category=type(exc).__name__,
                    sanitized_error="Candidate failed during cross-validation; inspect structured logs.",
                    training_seconds=round(time.perf_counter() - started, 3),
                    expected=isinstance(exc, (ValueError, TypeError)),
                )
            )
    if not results:
        raise RuntimeError("Every candidate model failed; review the data types and target.")
    results.sort(key=lambda item: item.selection_score or float("-inf"), reverse=True)
    best = results[0]
    final_pipeline = Pipeline(
        [
            ("preprocessor", _preprocessor(x_train, settings)),
            ("model", candidates[best.name]),
        ]
    )
    final_pipeline.fit(x_train, y_train)
    predictions = final_pipeline.predict(x_test)
    final_metrics = _metrics(task, y_test, predictions, final_pipeline, x_test)
    final_score = (
        final_metrics["balanced_accuracy"]
        if task == TaskType.classification
        else final_metrics["r2"]
    )
    best.final_test_score = float(final_score)
    best.final_test_metrics = final_metrics
    best.metrics = final_metrics
    return TrainingBundle(
        pipeline=final_pipeline,
        results=results,
        best_model=best.name,
        test_features=x_test,
        test_target=y_test,
        retry_number=retry_number,
        failures=failures,
    )


def _metrics(
    task: TaskType,
    truth: pd.Series,
    predictions: np.ndarray,
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> dict[str, float]:
    if task == TaskType.classification:
        metrics = {
            "accuracy": round(float(accuracy_score(truth, predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, predictions)), 4),
            "f1_weighted": round(float(f1_score(truth, predictions, average="weighted")), 4),
        }
        if truth.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(features)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(truth, probabilities)), 4)
        return metrics
    return {
        "r2": round(float(r2_score(truth, predictions)), 4),
        "rmse": round(float(mean_squared_error(truth, predictions) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(truth, predictions)), 4),
    }


def critic_decision(bundle: TrainingBundle, task: TaskType, settings: Settings) -> CriticDecision:
    best = bundle.results[0]
    threshold = (
        settings.min_classification_score
        if task == TaskType.classification
        else settings.min_regression_score
    )
    reasons: list[str] = []
    quality_score = best.cross_validation_mean if best.cross_validation_mean is not None else -1.0
    if quality_score < threshold:
        reasons.append(
            f"Training CV {best.primary_metric} {quality_score:.3f} is below {threshold:.3f}."
        )
    if (
        best.cross_validation_mean is not None
        and best.final_test_score is not None
        and abs(best.final_test_score - best.cross_validation_mean) > 0.2
    ):
        reasons.append("Holdout and cross-validation scores diverge by more than 0.20.")
    approved = not reasons or bundle.retry_number >= settings.max_critic_retries
    if not reasons:
        reasons.append("Performance and validation consistency passed the configured quality gate.")
    elif approved:
        reasons.append("Retry budget exhausted; result is retained with an explicit limitation.")
    return CriticDecision(
        approved=approved,
        score=quality_score,
        threshold=threshold,
        reasons=reasons,
        retry_number=bundle.retry_number,
    )


def explain_model(bundle: TrainingBundle) -> ExplainabilityResult:
    sample_size = min(300, len(bundle.test_features))
    features = bundle.test_features.iloc[:sample_size]
    target = bundle.test_target.iloc[:sample_size]
    try:
        transformed = bundle.pipeline.named_steps["preprocessor"].transform(features)
        transformed_names = bundle.pipeline.named_steps["preprocessor"].get_feature_names_out()
        estimator = bundle.pipeline.named_steps["model"]
        import shap

        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        raw = np.asarray(values.values)
        if raw.ndim == 3:
            raw = np.abs(raw).mean(axis=(0, 2))
        else:
            raw = np.abs(raw).mean(axis=0)
        importance = _top_importance(transformed_names, raw)
        return ExplainabilityResult(
            method="SHAP",
            feature_importance=importance,
            caveats=["SHAP values explain this fitted model, not causal effects."],
        )
    except Exception:
        permutation = permutation_importance(
            bundle.pipeline,
            features,
            target,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importance = _top_importance(features.columns, np.abs(permutation.importances_mean))
        return ExplainabilityResult(
            method="Permutation importance",
            feature_importance=importance,
            caveats=[
                "Permutation importance can dilute importance among correlated features.",
                "Feature importance is predictive, not causal.",
            ],
        )


def _top_importance(names: Any, values: np.ndarray, limit: int = 15) -> dict[str, float]:
    pairs = sorted(
        zip([str(name) for name in names], values.tolist(), strict=False),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return {name: round(float(value), 6) for name, value in pairs}
