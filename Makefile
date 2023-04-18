.DEFAULT_GOAL := help
.PHONY: install format lint test.safety codenetpy bugnet codex codegen description

### QUICK
# ¯¯¯¯¯¯¯

install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

format: ## Format
	python -m isort ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen ./description/ --skip .venv/
	python -m black ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen ./description/ --exclude .venv/

lint: ## Lint
	python -m flake8 ./codenetpy ./bugnet ./repair-pipeline ./codex ./codegen

codenetpy: ## CodeNetPy
	python codenetpy/main.py

bugnet: ## BugNet
	python bugnet/main.py --log info

codex: ## Codex Experiments
	python codex/main.py --log info

codegen: ## Codegen Experiments
	TOKENIZERS_PARALLELISM=false python codegen/main.py --model codegen --log info

codet5: ## CodeT5 Experiments
	TOKENIZERS_PARALLELISM=false python codegen/main.py --model codet5 --log info

description: ## Let ChatGPT generate descriptions
	TOKENIZERS_PARALLELISM=false python description/main.py --model openai-gpt --log info

export PYTHONPATH=$PYTHONPATH:src
