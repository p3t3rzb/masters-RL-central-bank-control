.PHONY: install hooks lint format test test-all

install:
	uv sync

hooks:
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy
	uv run complexipy -q

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest tests/unit

test-all:
	uv run pytest
