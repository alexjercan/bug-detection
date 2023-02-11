.DEFAULT_GOAL := help


### QUICK
# ¯¯¯¯¯¯¯

install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

format: ## Format
	python -m black ./codenetpy ./bugnet ./repair-pipeline ./codex --exclude .venv/
	python -m isort ./codenetpy ./bugnet ./repair-pipeline ./codex --skip .venv/

lint: ## Lint
	python -m flake8 ./codenetpy ./bugnet ./repair-pipeline ./codex

test.safety: ## Check for dependencies security breach with safety
	python -m safety check

export PYTHONPATH=$PYTHONPATH:src
