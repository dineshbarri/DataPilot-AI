from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from datapilot.config import get_settings
from datapilot.data import load_sample, read_dataset
from datapilot.insights import answer_follow_up
from datapilot.jobs import JobManager
from datapilot.persistence import RunStore
from datapilot.workflow import run_analysis

settings = get_settings()
logger = logging.getLogger(__name__)
jobs = JobManager(workers=2)
request_windows: dict[str, deque[float]] = defaultdict(deque)
app = FastAPI(
    title="DataPilot AI API",
    version="1.0.0",
    description="Evidence-grounded autonomous data science with LangGraph.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in settings.cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or uuid4().hex
    request.state.correlation_id = correlation_id
    if request.url.path.startswith("/v1/"):
        if settings.api_key and request.headers.get("X-API-Key") != settings.api_key:
            return _error_response(401, "Authentication required.", correlation_id)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = request_windows[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= settings.requests_per_minute:
            return _error_response(429, "Rate limit exceeded.", correlation_id)
        window.append(now)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


def _error_response(status_code: int, message: str, correlation_id: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"detail": message, "correlation_id": correlation_id},
        headers={"X-Correlation-ID": correlation_id},
    )


def _safe_failure(exc: Exception, correlation_id: str) -> HTTPException:
    logger.exception(
        "Analysis failure correlation_id=%s category=%s", correlation_id, type(exc).__name__
    )
    return HTTPException(
        status_code=500,
        detail={
            "message": "Analysis failed. Use the correlation ID when contacting support.",
            "correlation_id": correlation_id,
        },
    )


class SampleRequest(BaseModel):
    sample: str


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root() -> dict[str, str]:
    return {"name": settings.app_name, "status": "ready", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "environment": settings.environment,
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_rows": settings.max_rows,
            "max_columns": settings.max_columns,
        },
    }


@app.post("/v1/analyze/sample", status_code=202)
def analyze_sample(request: SampleRequest, http_request: Request):
    try:
        frame, target, dataset_name = load_sample(request.sample)
        job = jobs.submit(lambda: run_analysis(frame, target, dataset_name, settings))
        return {"job_id": job.job_id, "status": job.status, "status_url": f"/v1/jobs/{job.job_id}"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise _safe_failure(exc, http_request.state.correlation_id) from exc


@app.post("/v1/analyze/upload")
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(...),
):
    try:
        content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        frame = read_dataset(content, file.filename or "dataset.csv", settings)
        job = jobs.submit(
            lambda: run_analysis(frame, target, file.filename or "uploaded_dataset", settings)
        )
        return {"job_id": job.job_id, "status": job.status, "status_url": f"/v1/jobs/{job.job_id}"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise _safe_failure(exc, request.state.correlation_id) from exc


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.public()


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str):
    if jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, "cancelled": jobs.cancel(job_id)}


@app.get("/v1/runs")
def recent_runs():
    return RunStore(settings).list_recent()


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str):
    run = RunStore(settings).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.post("/v1/runs/{run_id}/chat")
def chat_with_run(run_id: str, request: ChatRequest):
    run = RunStore(settings).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"answer": answer_follow_up(run, request.question)}


@app.get("/v1/runs/{run_id}/artifacts/{artifact_name}")
def download_artifact(run_id: str, artifact_name: str):
    run = RunStore(settings).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    path_string = run.get("artifacts", {}).get(artifact_name)
    if not path_string:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = Path(path_string).resolve()
    artifact_root = settings.artifact_root.resolve()
    if artifact_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path, filename=path.name)
