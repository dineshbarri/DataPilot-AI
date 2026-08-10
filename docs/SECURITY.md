# Security Model

## Public demo boundaries

- CSV and Parquet only.
- Configurable upload size, row and column limits.
- No arbitrary package installation.
- No LLM-generated code is executed in Streamlit or FastAPI.
- Artifact download paths are resolved beneath the configured artifact root.
- Secrets are read from environment variables and are excluded from Git.

## Optional calculation worker

The worker is not required by the main analysis. If enabled, it accepts expressions—not
statements or programs—and applies AST validation before evaluation with empty built-ins.
The Docker Compose profile additionally applies:

- `network_mode: none`
- read-only root filesystem
- `no-new-privileges`
- all Linux capabilities dropped
- 256 MB memory limit
- 0.5 CPU limit
- PID limit
- `noexec` temporary filesystem

This is defense in depth, not a guarantee that Python is a perfect untrusted-code sandbox.
For high-risk multi-tenant operation, place jobs in short-lived microVMs or a managed sandbox.

## Production checklist

- Replace SQLite with PostgreSQL.
- Store artifacts in a private object store with expiring signed URLs.
- Add authentication, tenant isolation, rate limiting and audit retention.
- Scan images and dependencies in CI.
- Encrypt uploads at rest and define deletion/retention policies.
- Complete privacy, fairness and domain-specific legal review.

## Reporting vulnerabilities

Do not open a public issue for sensitive vulnerabilities. Contact the repository owner
privately with reproduction steps and affected versions.

