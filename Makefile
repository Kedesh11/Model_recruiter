PY=python3
PIP=pip

.PHONY: install fmt lint type test precommit

install:
	$(PIP) install -r requirements.txt
	pre-commit install || true

fmt:
	black .
	ruff check --fix .

lint:
	ruff check .
	mlc=0; mypy || mlc=$$?; exit $$mlc

type:
	mypy

test:
	pytest -q

precommit:
	pre-commit run --all-files -v
