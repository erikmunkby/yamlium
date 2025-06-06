test:
	uv run --group dev pytest --cov=yamlium .

test-report: test
	coverage html && \
	open htmlcov/index.html