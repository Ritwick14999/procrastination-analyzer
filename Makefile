.PHONY: help install dev test cov lint format typecheck check app cli evaluate train clean

PYTHON ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	$(PYTHON) -m pip install -e .

dev:  ## Install with dev and app extras
	$(PYTHON) -m pip install -e ".[dev,app]"

test:  ## Run the test suite
	$(PYTHON) -m pytest

cov:  ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=procrastination_analyzer --cov-report=term-missing

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src tests

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check src tests --fix

typecheck:  ## Type-check with mypy
	$(PYTHON) -m mypy

check: lint typecheck test  ## Run every gate CI runs

app:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run src/procrastination_analyzer/ui/app.py

cli:  ## Analyse the packaged sample dataset
	procrastination-analyzer analyze src/procrastination_analyzer/data/sample_commits.csv

evaluate:  ## Cross-validate the models against the heuristic baseline
	procrastination-analyzer evaluate --patterns

train:  ## Train and persist a risk model
	procrastination-analyzer train -o artifacts/risk_model.joblib

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
