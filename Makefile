.PHONY: test format

help:
	@echo "make test   -- runs tests for project"
	@echo

test:
	poetry run pytest --cov-report term --cov=fabrik tests

format:
	poetry run black .
	poetry run autoflake -r . --in-place --expand-star-imports --remove-unused-variables --remove-all-unused-imports