### Codenet
# ¯¯¯¯¯¯¯¯¯¯¯

codenet.install: ## Install dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

codenet.start: ## Start app
	python src/main.py
