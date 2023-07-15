.DEFAULT_GOAL := help
.PHONY: install fmt lint bugnet description repair

### QUICK
# ¯¯¯¯¯¯¯

help: ## Help
	echo "You need this because you suck as a developer."

install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

fmt: ## Format
	python -m isort ./bugnet/ ./aoc-dataset/ ./repair-pipeline/ ./description/ ./repair/  --skip .venv/
	python -m black ./bugnet/ ./aoc-dataset/ ./repair-pipeline/ ./description/ ./repair/  --exclude .venv/

lint: ## Lint
	python -m flake8 ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen

bugnet: ## BugNet
	python bugnet/main.py --log info

aoc: ## AoC
	python aoc-dataset/main.py --log info

description: ## Generate descriptions
	TOKENIZERS_PARALLELISM=false python description/main.py --log info

repair: ## Generate repairs
	TOKENIZERS_PARALLELISM=false python repair/main.py --log info
