from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from datapilot.config import Settings


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload_json: Mapped[str] = mapped_column(Text)


class RunStore:
    """SQLite locally; set DATABASE_URL to PostgreSQL in production."""

    def __init__(self, settings: Settings):
        self.engine = create_engine(settings.database_url, future=True)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def save(self, run_id: str, dataset_name: str, status: str, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            record = session.get(AnalysisRun, run_id)
            if record is None:
                record = AnalysisRun(
                    run_id=run_id,
                    dataset_name=dataset_name,
                    status=status,
                    created_at=now,
                    updated_at=now,
                    payload_json=json.dumps(payload, default=str),
                )
                session.add(record)
            else:
                record.status = status
                record.updated_at = now
                record.payload_json = json.dumps(payload, default=str)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(AnalysisRun, run_id)
            return json.loads(record.payload_json) if record else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        from sqlalchemy import select

        with self.sessions() as session:
            records = session.scalars(
                select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(limit)
            )
            return [
                {
                    "run_id": record.run_id,
                    "dataset_name": record.dataset_name,
                    "status": record.status,
                    "created_at": record.created_at.isoformat(),
                }
                for record in records
            ]


class ArtifactStore:
    """Local artifact storage with an interface that can be replaced by S3/MinIO."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def run_directory(self, run_id: str) -> Path:
        directory = (self.root / run_id).resolve()
        if self.root.resolve() not in directory.parents:
            raise ValueError("Invalid run identifier.")
        directory.mkdir(parents=True, exist_ok=True)
        return directory
