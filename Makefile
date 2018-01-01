help:
	@echo "make test   -- runs tests for project"
	@echo

test:
	cd src && \
	 coverage run  -m unittest discover -s .  -p 'Test*.py' && \
	 coverage report
