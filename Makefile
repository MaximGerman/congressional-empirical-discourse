.PHONY: lint format check typecheck test pipeline notebook setup clean-data explorer optimize update-docs

# Lint and format
lint:
	.venv/bin/ruff check src/ notebooks/

format:
	.venv/bin/ruff format src/ notebooks/
	.venv/bin/ruff check --fix src/ notebooks/

check: lint typecheck
	.venv/bin/ruff format --check src/ notebooks/

# Tests & Typechecking
typecheck:
	.venv/bin/mypy src/

test:
	export PYTHONPATH=$PYTHONPATH:. && .venv/bin/python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Dashboard & Data
explorer:
	.venv/bin/streamlit run scripts/explorer.py

optimize:
	.venv/bin/python scripts/optimize_data.py

update-docs:
	export PYTHONPATH=$PYTHONPATH:. && .venv/bin/python scripts/update_data_dict.py

# Pipeline
pipeline:
	.venv/bin/python -m src.pipeline

clean-data:
	rm -f data/*.csv

# Notebook
notebook:
	jupyter notebook notebooks/

# Setup
setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
	.venv/bin/pre-commit install
