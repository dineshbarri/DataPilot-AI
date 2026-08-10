# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [2.0.0] - 2026-08-10

### Added

- Training-only cross-validation selection and one-time untouched test evaluation.
- Sparse, cardinality-bounded categorical encoding and encoded-width guardrails.
- Structured candidate failure records.
- Asynchronous API job contract, cancellation, authentication, rate limiting, and correlation IDs.
- CI, Dependabot, pre-commit, Docker context controls, templates, threat model, privacy, and evaluation docs.

### Changed

- Consolidated Streamlit deployment onto the `app.py` implementation.
- Sanitized API and job errors returned to clients.
