# Threat model

## Assets

Uploaded datasets, API keys, database credentials, generated artifacts, fitted models, run
metadata, and operational logs.

## Trust boundaries

Client → authenticated API → bounded job queue → isolated worker → run database/object storage →
optional Gemini API. The local threaded queue is a development adapter; production deployments
should use a durable queue and isolated workers.

## Principal threats and controls

| Threat | Control |
|---|---|
| Malicious or oversized upload | extension/size/shape limits; content never executed |
| Category explosion | sparse encoding, per-feature cap, estimated-width guard |
| Prompt injection in data | metadata-first prompts; controlled tools; output evaluation |
| Secret/PII disclosure | masking, redaction, secret scanning, bounded AI payload |
| Path traversal | resolved artifact-root containment check |
| Resource exhaustion | rate limits, worker bounds, timeouts, quotas in production |
| Cross-tenant access | production authentication, tenant-scoped storage and authorization |
| Dependency compromise | Dependabot, lock file, pip-audit, Trivy, pinned Actions |

Residual risk and deployment responsibilities are documented in `docs/SECURITY.md`.
