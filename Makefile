help:
	@echo "make test   -- runs tests for project"
	@echo

test:
	cd src && \
	 python  -m unittest discover -s .  -p 'Test*.py'
