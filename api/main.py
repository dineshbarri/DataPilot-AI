from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from datapilot.config import get_settings
from datapilot.data import load_sample, read_dataset
from datapilot.insights import answer_follow_up
from datapilot.persistence import RunStore
from datapilot.workflow import run_analysis

settings = get_settings()
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


@app.post("/v1/analyze/sample")
def analyze_sample(request: SampleRequest):
    try:
        frame, target, dataset_name = load_sample(request.sample)
        return run_analysis(frame, target, dataset_name, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/v1/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    target: str = Form(...),
):
    try:
        content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        frame = read_dataset(content, file.filename or "dataset.csv", settings)
        return run_analysis(frame, target, file.filename or "uploaded_dataset", settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


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
