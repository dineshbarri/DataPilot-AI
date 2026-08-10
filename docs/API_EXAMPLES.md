# API examples

Start the service with `uvicorn api.main:app --port 8000`. If `API_KEY` is configured, include
`X-API-Key: <key>` in every `/v1/` request.

## Submit a bundled dataset

```bash
curl -i -X POST http://localhost:8000/v1/analyze/sample \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: iris-demo-001" \
  -d '{"sample":"iris"}'
```

```json
{"job_id":"job_abc123","status":"queued"}
```

## Submit an upload

```bash
curl -i -X POST http://localhost:8000/v1/analyze/upload \
  -F "file=@dataset.csv" \
  -F "target=outcome"
```

## Poll or cancel

```bash
curl http://localhost:8000/v1/jobs/job_abc123
curl -X DELETE http://localhost:8000/v1/jobs/job_abc123
```

A completed job includes the validated run summary in `result`. Errors exposed to clients are
sanitized and carry an `X-Correlation-ID`; use that identifier to locate structured server logs.
The machine-readable contract is committed as [`openapi.json`](openapi.json).

