### FORMAT
# ¯¯¯¯¯¯¯¯

format.black: ## Run black on every file
	python -m black src/ --exclude .venv/

format.isort: ## Sort imports
	python -m isort src/ --skip .venv/
