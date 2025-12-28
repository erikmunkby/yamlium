fix:
	uv run ruff format && uv run ruff check --fix-only --unsafe-fixes && uv run ruff check

test:
	uv run --group dev pytest --cov=yamlium .

test-report: test
	coverage html && \
	open htmlcov/index.html

type-check:
	uv run --group dev mypy tests/test_type_checking.py --check-untyped-defs

test-all: test type-check