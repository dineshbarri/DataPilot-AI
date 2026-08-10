from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_unknown_job_is_not_found():
    response = TestClient(app).get("/v1/jobs/job_missing")
    assert response.status_code == 404
    assert response.headers["x-correlation-id"]


def test_sample_analysis_uses_job_contract():
    client = TestClient(app)
    response = client.post("/v1/analyze/sample", json={"sample": "iris"})
    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"].startswith("job_")
    assert payload["status_url"].endswith(payload["job_id"])
