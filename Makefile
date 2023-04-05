.DEFAULT_GOAL := help
.PHONY: install format lint test.safety codenetpy bugnet codex codegen

### QUICK
# ¯¯¯¯¯¯¯

install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

format: ## Format
	python -m isort ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen --skip .venv/
	python -m black ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen --exclude .venv/

lint: ## Lint
	python -m flake8 ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen

codenetpy: ## CodeNetPy
	python codenetpy/main.py

bugnet: ## BugNet
	python bugnet/main.py --log info

codex: ## Codex Experiments
	python codex/main.py --log info

codegen: ## Codegen Experiments
	TOKENIZERS_PARALLELISM=false python codegen/main.py --log info

export PYTHONPATH=$PYTHONPATH:src
