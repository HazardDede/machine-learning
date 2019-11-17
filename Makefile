.PHONY: lint test

# Path stuff
SOURCE_PATH=./ml


help:
	@echo "Targets:"
	@echo "    clean"
	@echo "        Remove python and release artifacts"
	@echo "    lint"
	@echo "        Check code with flake8, pylint..."
	@echo "    test"
	@echo "        Run py.test"

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	# find . -name '*~' -exec rm --force  {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache

clean: clean-pyc clean-build

flake8:
	flake8 \
		--max-line-length 100 \
		--max-complexity 8 \
		--ignore=E402,W503,Q000 \
		$(SOURCE_PATH)

pylint:
	# Automatically uses the configuration: pylintrc
	pylint $(SOURCE_PATH)

lint: flake8 pylint
