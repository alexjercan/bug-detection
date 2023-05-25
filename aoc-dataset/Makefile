.DEFAULT_GOAL := help
.PHONY: install format


### QUICK
# ¯¯¯¯¯¯¯

install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

format: ## Format
	python -m isort . --skip .venv/
	python -m black . --exclude .venv/
