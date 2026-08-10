# Data privacy

DataPilot defaults to deterministic local analysis. Gemini is optional and receives a bounded
metadata package rather than the full dataset: schema, aggregates, target candidates, quality
signals, and a small redacted sample. Users can exclude columns, and likely PII columns are
removed automatically.

Production operators must define dataset retention, deletion SLAs, tenant isolation, regional
processing, encryption, access logging, data-subject request handling, and subprocessors. API
keys and database credentials belong in a secret manager and must never enter exports or logs.
