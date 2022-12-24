.DEFAULT_GOAL := help


### QUICK
# ¯¯¯¯¯¯¯

install: codenet.install ## Install

start: codenet.start ## Train

format: format.isort format.black ## Format

lint: test.lint ## Lint

export PYTHONPATH=$PYTHONPATH:src

include makefiles/codenet.mk
include makefiles/test.mk
include makefiles/format.mk
include makefiles/help.mk
