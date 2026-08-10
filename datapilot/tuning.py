from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from datapilot.config import Settings
from datapilot.schemas import TaskType


def tune_random_forest(
    task: TaskType,
    preprocessor: Any,
    features: pd.DataFrame,
    target: pd.Series,
    settings: Settings,
    cv: int,
) -> dict[str, Any]:
    """Run a bounded Optuna study when the optional AutoML extra is installed."""
    if settings.optuna_trials <= 0 or cv < 2:
        return {}
    try:
        import optuna
    except ImportError:
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scoring = "balanced_accuracy" if task == TaskType.classification else "r2"

    def objective(trial):
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 360, step=60),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
        }
        common = {
            **parameters,
            "n_jobs": -1,
            "random_state": settings.random_state,
        }
        if task == TaskType.classification:
            estimator = RandomForestClassifier(**common, class_weight="balanced")
        else:
            estimator = RandomForestRegressor(**common)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_val_score(
            pipeline,
            features,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=settings.optuna_trials,
        timeout=90,
        show_progress_bar=False,
    )
    return study.best_params
