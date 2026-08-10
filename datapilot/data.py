from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import duckdb
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from datapilot.config import Settings

SAMPLE_DATASETS = {
    "Iris classification": "iris",
    "Breast cancer classification": "breast_cancer",
    "Diabetes progression regression": "diabetes",
}


def load_sample(name: str) -> tuple[pd.DataFrame, str, str]:
    loaders = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"Unknown sample dataset: {name}")
    bundle = loaders[name](as_frame=True)
    frame = bundle.frame.copy()
    frame.columns = [str(column).replace(" ", "_") for column in frame.columns]
    target = str(bundle.target.name).replace(" ", "_")
    return frame, target, name


def read_dataset(
    source: bytes | BinaryIO | str | Path,
    filename: str,
    settings: Settings,
) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("Only CSV and Parquet files are supported.")
    if isinstance(source, bytes):
        if len(source) > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"File exceeds the {settings.max_upload_mb} MB upload limit.")
        stream: BinaryIO | str | Path = io.BytesIO(source)
    else:
        stream = source
    frame = pd.read_csv(stream) if suffix == ".csv" else pd.read_parquet(stream)
    validate_shape(frame, settings)
    frame.columns = _unique_columns([str(column).strip() for column in frame.columns])
    return frame


def validate_shape(frame: pd.DataFrame, settings: Settings) -> None:
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if len(frame) > settings.max_rows:
        raise ValueError(f"Dataset has {len(frame):,} rows; limit is {settings.max_rows:,}.")
    if len(frame.columns) > settings.max_columns:
        raise ValueError(
            f"Dataset has {len(frame.columns):,} columns; limit is {settings.max_columns:,}."
        )
    if len(frame.columns) < 2:
        raise ValueError("At least one feature and one target column are required.")


def _unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in columns:
        name = raw or "unnamed"
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count}")
    return result


def duckdb_overview(frame: pd.DataFrame) -> dict[str, object]:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("dataset", frame)
        row = connection.execute(
            """
            SELECT
              COUNT(*) AS row_count
            FROM dataset
            """
        ).fetchone()
        numeric = frame.select_dtypes(include="number")
        correlations = (
            numeric.corr(numeric_only=True).round(4).fillna(0).to_dict()
            if len(numeric.columns) > 1
            else {}
        )
        return {
            "row_count": int(row[0]),
            "duplicate_rows": int(frame.duplicated().sum()),
            "correlations": correlations,
        }
    finally:
        connection.close()
