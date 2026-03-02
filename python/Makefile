.PHONY: install lint format-check import-check flake8-check test fix-format

install:
	uv sync --extra dev

lint: format-check import-check flake8-check

format-check:
	uv run black --check --diff .

import-check:
	uv run isort --profile black --check-only --diff .

flake8-check:
	uv run flake8 . --count --max-line-length=100 --ignore=E203,W503 \
		--exclude=.venv --per-file-ignores='milvus_storage/_ffi.py:E501' --statistics

test:
	uv run pytest tests/ -v --tb=short --cov=milvus_storage --cov-report=xml

fix-format:
	uv run black .
	uv run isort --profile black .
