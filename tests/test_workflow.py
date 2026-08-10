from pathlib import Path

from datapilot.config import Settings
from datapilot.data import load_sample
from datapilot.workflow import run_analysis


def test_end_to_end_classification_workflow(tmp_path: Path):
    frame, target, name = load_sample("iris")
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        database_url=f"sqlite:///{(tmp_path / 'runs.db').as_posix()}",
        max_critic_retries=0,
        optuna_trials=0,
    )
    result = run_analysis(frame, target, name, settings)
    assert result.status == "completed"
    assert result.best_model
    assert result.model_results[0].primary_score >= 0
    assert result.critic.approved
    assert Path(result.artifacts["pipeline"]).exists()
    assert Path(result.artifacts["model_card"]).exists()
    assert any(item["agent"] == "Evaluation / Critic Agent" for item in result.trace)
