# Deployment Guide

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a Streamlit app with `streamlit_app.py` as the entrypoint.
3. Use Python 3.12.
4. Add `GEMINI_API_KEY` only if optional LLM narration is desired.
5. Keep the default SQLite/local artifact mode for a stateless public demo.

The app is fully functional without an LLM key.

## Render API

`render.yaml` defines a Docker-backed FastAPI service. Configure:

- `DATABASE_URL`: Neon, Supabase or another PostgreSQL URL.
- `GEMINI_API_KEY`: optional.
- persistent/object storage if generated artifacts must survive redeployments.

## Docker Compose

```bash
docker compose up --build
```

- Streamlit: <http://localhost:8501>
- FastAPI: <http://localhost:8000/docs>

The isolated worker has no exposed port and no network.

## Production topology

For heavier datasets, make `/v1/analyze/*` enqueue jobs to a separate worker and return
`202 Accepted`. Persist state in PostgreSQL, place artifacts in S3-compatible storage, and
stream progress through server-sent events or polling.

