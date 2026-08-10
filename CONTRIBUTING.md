# Contributing

1. Create a feature branch.
2. Install `pip install -e ".[dev]"`.
3. Run `ruff check .`.
4. Run `pytest --cov=datapilot --cov=api`.
5. Keep computed metrics deterministic and source-backed.
6. Add a test for every new quality rule, workflow route or API behavior.
7. Never commit datasets containing personal data, credentials or generated artifacts.

Pull requests should explain the user problem, architecture impact, validation performed and
any new security or model-risk considerations.

