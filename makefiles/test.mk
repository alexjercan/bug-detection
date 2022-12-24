### TEST
# ¯¯¯¯¯¯¯¯

test.lint: ## Lint python files with flake8
	python -m flake8 ./src

test.safety: ## Check for dependencies security breach with safety
	python -m safety check
