ifneq ("$(wildcard .env)","")
    include .env
endif
SHELL := /bin/bash
HUGGING_FACE_TOKEN ?= ""  # Provide a default if the variable isn't in .env
TF_ENABLE_ONEDNN_OPTS=0
export

all: hooks install-dev lint test

PHONY: model-install
model-install:
	pipenv run transformers-cli download -h
	pipenv run transformers-cli download --cache_dir .cache --trust-remote-code --token "$(HUGGING_FACE_TOKEN)" bigcode/starcoder2-7b
# pipenv run transformers-cli download bigcode/starcoder2-7b
# pipenv run transformers-cli login
# pipenv run transformers-cli download --cache-dir .cache --trust-remote-code bigcode/starcoder2-7b
PHONY: hooks
hooks:
	pip install pre-commit
	pip install pipenv
	pre-commit install

PHONY: clean
clean:
# pipenv clean || echo "no environment found to clean"
	pipenv run python -c "import os; os.remove('requirements.txt')" || echo "no lock file to remove"
	pipenv run python -c "import os; os.remove('Pipfile.lock')" || echo "no lock file to remove"
	pipenv --rm || echo "no environment found to remove"

PHONY: lint
lint:
	pipenv run pre-commit run --all-files

PHONY: install
install:
	pipenv install

PHONY: install-dev
install-dev:
	pipenv install --dev

PHONY: sync
sync:
	pipenv sync

PHONY: sync-dev
sync-dev:
	pipenv sync --dev

PHONY: build
build:
	pipenv run pip freeze > requirements.txt
	sam build -c --use-container
