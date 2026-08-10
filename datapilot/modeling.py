from __future__ import annotations

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
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
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


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
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
                OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
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
            "Histogram Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=120 + retry * 60, random_state=settings.random_state
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
            "Histogram Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=120 + retry * 60, random_state=settings.random_state
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
    clean = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    features = clean.drop(columns=[target]).copy()
    labels = clean[target].copy()
    if features.empty:
        raise ValueError("No usable feature columns remain after removing the target.")
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
    fitted: dict[str, Pipeline] = {}
    cv = min(5, max(2, int(len(x_train) / 20)))
    if task == TaskType.classification:
        smallest_class = int(y_train.value_counts().min())
        cv = min(cv, smallest_class) if smallest_class >= 2 else 0
        scoring = "balanced_accuracy"
    else:
        scoring = "r2"

    candidates = _candidate_models(task, settings, retry_number)
    tuned_parameters = tune_random_forest(
        task,
        _preprocessor(x_train),
        x_train,
        y_train,
        settings,
        cv,
    )
    if tuned_parameters:
        candidates["Random Forest"].set_params(**tuned_parameters)

    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", _preprocessor(x_train)), ("model", estimator)])
        started = time.perf_counter()
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            metrics = _metrics(task, y_test, predictions, pipeline, x_test)
            cv_scores = (
                cross_val_score(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
                if cv >= 2
                else np.array([])
            )
            result = ModelResult(
                name=name,
                primary_metric="balanced_accuracy" if task == TaskType.classification else "r2",
                primary_score=float(
                    metrics["balanced_accuracy"]
                    if task == TaskType.classification
                    else metrics["r2"]
                ),
                metrics=metrics,
                cross_validation_mean=float(cv_scores.mean()) if len(cv_scores) else None,
                cross_validation_std=float(cv_scores.std()) if len(cv_scores) else None,
                training_seconds=round(time.perf_counter() - started, 3),
            )
            results.append(result)
            fitted[name] = pipeline
        except Exception:
            continue
    if not results:
        raise RuntimeError("Every candidate model failed; review the data types and target.")
    results.sort(key=lambda item: item.primary_score, reverse=True)
    best = results[0]
    return TrainingBundle(
        pipeline=fitted[best.name],
        results=results,
        best_model=best.name,
        test_features=x_test,
        test_target=y_test,
        retry_number=retry_number,
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
    if best.primary_score < threshold:
        reasons.append(
            f"Holdout {best.primary_metric} {best.primary_score:.3f} is below {threshold:.3f}."
        )
    if (
        best.cross_validation_mean is not None
        and abs(best.primary_score - best.cross_validation_mean) > 0.2
    ):
        reasons.append("Holdout and cross-validation scores diverge by more than 0.20.")
    approved = not reasons or bundle.retry_number >= settings.max_critic_retries
    if not reasons:
        reasons.append("Performance and validation consistency passed the configured quality gate.")
    elif approved:
        reasons.append("Retry budget exhausted; result is retained with an explicit limitation.")
    return CriticDecision(
        approved=approved,
        score=best.primary_score,
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
