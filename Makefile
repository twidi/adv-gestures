PROJECT_NAME := "adv_gestures"
PROJECT_ROOT := "src/$(PROJECT_NAME)"

BOLD := \033[1m
RESET := \033[0m

default: help

.PHONY : help
help:  ## Show this help
	@echo "$(BOLD)$(PROJECT_NAME) project Makefile $(RESET)"
	@echo "Please use 'make $(BOLD)target$(RESET)' where $(BOLD)target$(RESET) is one of:"
	@grep -h ':\s\+##' Makefile | column -t -s# | awk -F ":" '{ print "  $(BOLD)" $$1 "$(RESET)" $$2 }'

.PHONY: install
install:  ## Install the project in the current environment, with its dependencies
	@echo "$(BOLD)Installing $(PROJECT_NAME)$(RESET)"
	@uv pip install .

.PHONY: dev
dev:  ## Install the project in the current environment, with its dependencies, including the ones needed in a development environment
	@echo "$(BOLD)Installing (or upgrading) $(PROJECT_NAME) in dev mode (with all dependencies)$(RESET)"
	@pip install --upgrade uv
	@uv pip install --upgrade pip setuptools
	@uv pip install --upgrade -e .[dev]
	@$(MAKE) full-clean

.PHONY: dist
dist:  ## Build the package
dist: clean
	@echo "$(BOLD)Building package$(RESET)"
	@python setup.py sdist bdist_wheel

.PHONY: clean
clean:  ## Clean python build related directories and files
	@echo "$(BOLD)Cleaning$(RESET)"
	@rm -rf build dist $(PROJECT_ROOT).egg-info

.PHONY: full-clean
full-clean:  ## Like "clean" but will clean some other generated directories or files
full-clean: clean
	@echo "$(BOLD)Full cleaning$(RESET)"
	find ./ -type d  \( -name '__pycache__' -or -name '.mypy_cache'  \) -print0 | xargs -tr0 rm -r

.PHONY: lint
lint:  ## Run all linters (check-isort, check-black, mypy, ruff)
lint: check-isort check-black ruff mypy

.PHONY: mypy
mypy:  ## Run the mypy tool
	@echo "$(BOLD)Running mypy$(RESET)"
	@mypy $(PROJECT_ROOT)

.PHONY: check-isort
check-isort:  ## Run the isort tool in check mode only (won't modify files)
	@echo "$(BOLD)Checking isort(RESET)"
	@isort $(PROJECT_ROOT) --check-only 2>&1

.PHONY: check-black
check-black:  ## Run the black tool in check mode only (won't modify files)
	@echo "$(BOLD)Checking black$(RESET)"
	@black --check  $(PROJECT_ROOT) 2>&1

.PHONY: ruff
ruff:  ## Run the ruff tool
	@echo "$(BOLD)Running ruff$(RESET)"
	@ruff check $(PROJECT_ROOT)

.PHONY: pretty
pretty:  ## Run all code beautifiers (isort, black)
pretty: isort black

.PHONY: isort
isort:  ## Run the isort tool and update files that need to
	@echo "$(BOLD)Running isort$(RESET)"
	@isort $(PROJECT_ROOT) --atomic

.PHONY: black
black:  ## Run the black tool and update files that need to
	@echo "$(BOLD)Running black$(RESET)"
	@black $(PROJECT_ROOT)
