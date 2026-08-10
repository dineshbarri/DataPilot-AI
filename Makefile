.PHONY: install test lint format coverage build run api security

install:
	python -m pip install -e ".[dev]"
test:
	python -m pytest
lint:
	python -m ruff check .
format:
	python -m ruff format .
coverage:
	python -m pytest --cov=datapilot --cov=api --cov-fail-under=75
build:
	python -m build
run:
	python -m streamlit run app.py
api:
	python -m uvicorn api.main:app --reload
security:
	python -m pip_audit
