"""Bounded asynchronous analysis jobs with a production-queue compatible contract."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    progress: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    result: Any | None = None
    error: str | None = None
    future: Future[Any] | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result if self.status == "completed" else None,
            "error": self.error,
        }


class JobManager:
    """In-process development queue; replaceable by ARQ/Celery behind the same API."""

    def __init__(self, workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="datapilot")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, operation: Callable[[], Any]) -> Job:
        job = Job(job_id=f"job_{uuid4().hex[:12]}")
        with self._lock:
            self._jobs[job.job_id] = job
        job.future = self._executor.submit(self._run, job, operation)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.future is None or job.status not in {"queued", "running"}:
            return False
        cancelled = job.future.cancel()
        if cancelled:
            job.status, job.progress = "cancelled", 0
            job.updated_at = datetime.now(UTC).isoformat()
        return cancelled

    @staticmethod
    def _run(job: Job, operation: Callable[[], Any]) -> None:
        job.status, job.progress = "running", 10
        job.updated_at = datetime.now(UTC).isoformat()
        try:
            value = operation()
            job.result = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            job.status, job.progress = "completed", 100
        except Exception:
            job.status, job.progress = "failed", 100
            job.error = "Analysis failed. Use the correlation ID in server logs for support."
        finally:
            job.updated_at = datetime.now(UTC).isoformat()
