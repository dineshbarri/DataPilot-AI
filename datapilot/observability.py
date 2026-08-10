from __future__ import annotations

from typing import Any

from datapilot.config import Settings


def log_to_mlflow(payload: dict[str, Any], settings: Settings) -> bool:
    """Record a run when MLflow is enabled; local demos remain dependency-light."""
    if not settings.enable_mlflow:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("datapilot-ai")
        best = payload["model_results"][0]
        with mlflow.start_run(run_name=payload["run_id"]):
            mlflow.log_params(
                {
                    "dataset": payload["dataset_name"],
                    "task_type": payload["profile"]["task_type"],
                    "target": payload["profile"]["target"],
                    "selected_model": payload["best_model"],
                }
            )
            mlflow.log_metrics({key: float(value) for key, value in best["metrics"].items()})
        return True
    except Exception:
        return False
