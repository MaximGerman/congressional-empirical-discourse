.PHONY: lint format check pipeline notebook setup

# Lint and format
lint:
	ruff check src/ notebooks/

format:
	ruff format src/ notebooks/
	ruff check --fix src/ notebooks/

check: lint
	ruff format --check src/ notebooks/

# Pipeline
pipeline:
	python -m src.pipeline

# Notebook
notebook:
	jupyter notebook notebooks/

# Setup
setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install ruff
	.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
