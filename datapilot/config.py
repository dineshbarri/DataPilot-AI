from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DataPilot AI"
    environment: str = "development"
    artifact_root: Path = Path("artifacts")
    database_url: str = "sqlite:///artifacts/datapilot.db"
    max_upload_mb: int = Field(default=25, ge=1, le=250)
    max_rows: int = Field(default=100_000, ge=100)
    max_columns: int = Field(default=250, ge=2)
    max_categories_per_feature: int = Field(default=100, ge=10, le=10_000)
    max_encoded_features: int = Field(default=5_000, ge=100, le=100_000)
    api_key: str | None = None
    requests_per_minute: int = Field(default=30, ge=1, le=10_000)
    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    max_critic_retries: int = Field(default=1, ge=0, le=3)
    min_classification_score: float = 0.55
    min_regression_score: float = 0.15
    optuna_trials: int = Field(default=8, ge=0, le=50)
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    cors_origins: str = "http://localhost:8501"

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
