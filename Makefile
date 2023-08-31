lint:
	python -m flake8
	python -m black --check --diff .
	python -m mypy .

lint-fix:
	python -m black .

test:
	py.test
