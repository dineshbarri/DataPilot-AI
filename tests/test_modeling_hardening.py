from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datapilot.config import Settings
from datapilot.modeling import train_models
from datapilot.schemas import TaskType


def settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        artifact_root=tmp_path / "artifacts",
        database_url=f"sqlite:///{(tmp_path / 'runs.db').as_posix()}",
        optuna_trials=0,
        max_critic_retries=0,
        **overrides,
    )


def test_unknown_target_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        train_models(
            pd.DataFrame({"x": range(30), "y": range(30)}),
            "missing",
            TaskType.regression,
            settings(tmp_path),
        )


def test_single_class_target_rejected(tmp_path):
    frame = pd.DataFrame({"x": range(30), "target": [1] * 30})
    with pytest.raises(ValueError, match="two distinct"):
        train_models(frame, "target", TaskType.classification, settings(tmp_path))


def test_high_cardinality_width_guard(tmp_path):
    frame = pd.DataFrame(
        {"category": [f"id-{i}" for i in range(150)], "target": [i % 2 for i in range(150)]}
    )
    with pytest.raises(ValueError, match="encoded width"):
        train_models(
            frame,
            "target",
            TaskType.classification,
            settings(tmp_path, max_categories_per_feature=200, max_encoded_features=100),
        )


def test_regression_reports_separate_cv_and_test_scores(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    frame = pd.DataFrame(
        {
            "x": x,
            "segment": np.where(x > 0, "a", "b"),
            "target": 3 * x + rng.normal(scale=0.2, size=120),
        }
    )
    bundle = train_models(frame, "target", TaskType.regression, settings(tmp_path))
    best = bundle.results[0]
    assert best.selection_score == best.cross_validation_mean
    assert best.final_test_score is not None
    assert best.final_test_metrics
