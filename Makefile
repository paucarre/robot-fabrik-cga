.PHONY: test

help:
	@echo "make test   -- runs tests for project"
	@echo

test:
	 coverage run --source fabrik/  -m unittest discover -s .  -p 'Test*.py' && \
	 coverage report
