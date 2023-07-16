.DEFAULT_GOAL := help
.PHONY: install fmt lint mypy pyright pylint checks bugnet description repair

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
	python -m flake8 ./bugnet/
	python -m flake8 ./aoc-dataset/
	python -m flake8 ./description/
	python -m flake8 ./repair/
	python -m flake8 ./repair-pipeline/

pylint: ## Check with pylint
	python -m pylint ./bugnet/
	python -m pylint ./aoc-dataset/
	python -m pylint ./description/
	python -m pylint ./repair/

mypy: ## Check with mypy
	python -m mypy ./bugnet/ --ignore-missing-imports
	python -m mypy ./aoc-dataset/*.py --ignore-missing-imports
	python -m mypy ./description/ --ignore-missing-imports
	python -m mypy ./repair/ --ignore-missing-imports
	python -m mypy ./repair-pipeline/docker-example/ --ignore-missing-imports
	python -m mypy ./repair-pipeline/rest-api/ --ignore-missing-imports

pyright: ## Check with pyright
	python -m pyright ./bugnet/
	python -m pyright ./aoc-dataset/
	python -m pyright ./description/
	python -m pyright ./repair/

checks: lint pylint mypy pyright

bugnet: ## BugNet
	python bugnet/main.py --log info

aoc: ## AoC
	python aoc-dataset/main.py --log info

description: ## Generate descriptions
	TOKENIZERS_PARALLELISM=false python description/main.py --log info

repair: ## Generate repairs
	TOKENIZERS_PARALLELISM=false python repair/main.py --log info
