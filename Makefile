.PHONY: lint format check pipeline notebook setup

# Lint and format
lint:
	ruff check src/ notebooks/

format:
	ruff format src/ notebooks/
	ruff check --fix src/ notebooks/

check: lint typecheck
	ruff format --check src/ notebooks/

# Tests & Typechecking
typecheck:
	mypy src/

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Pipeline
pipeline:
	python -m src.pipeline

# Notebook
notebook:
	jupyter notebook notebooks/

# Setup
setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
	.venv/bin/pre-commit install
