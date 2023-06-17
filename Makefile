.PHONY: test

help:
	@echo "make test   -- runs tests for project"
	@echo

test:
	poetry run pytest --cov-report term --cov=fabrik tests
