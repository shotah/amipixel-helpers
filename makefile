ifneq ("$(wildcard .env)","")
	include .env
endif
SHELL := /bin/bash
HUGGING_FACE_TOKEN ?= ""  # Provide a default if the variable isn't in .env
TF_ENABLE_ONEDNN_OPTS=0
export

all: hooks install-dev lint test

# --- Model Installation ---
DIFFUSION_MODEL_ID := runwayml/stable-diffusion-v1-5
PYTORCH_CUDA_VERSION := cu126

PHONY: model-install
model-install: install-pytorch download-diffusion-model

PHONY: install-pytorch
install-pytorch:
	pipenv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$(PYTORCH_CUDA_VERSION)

PHONY: download-diffusion-model
download-diffusion-model:
	pipenv run transformers-cli download --cache_dir .cache --trust-remote-code --token "$(HUGGING_FACE_TOKEN)" $(DIFFUSION_MODEL_ID)

# --- Hooks ---
PHONY: hooks
hooks:
	pip install pre-commit
	pip install pipenv
	pre-commit install

# --- Cleaning ---
PHONY: clean
clean:
# pipenv clean || echo "no environment found to clean"
	pipenv run python -c "import os; os.remove('requirements.txt')" || echo "no lock file to remove"
	pipenv run python -c "import os; os.remove('Pipfile.lock')" || echo "no lock file to remove"
	pipenv --rm || echo "no environment found to remove"

# --- Linting ---
PHONY: lint
lint:
	pipenv run pre-commit run --all-files

# --- Installation ---
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

# --- Building ---
PHONY: build
build:
	pipenv run pip freeze > requirements.txt
	sam build -c --use-container

# --- Running Frame Generator ---
FRAME_GENERATOR_SCRIPT := frame_generation/generate_frames.py # Assuming your script is in this folder
# pipenv run python $(FRAME_GENERATOR_SCRIPT)

PHONY: run-frame-generator
run-frame-generator:	
	pipenv run python frame_generation/found_script.py

# --- Help ---
PHONY: help
help:
    @echo "Makefile for AmiPixel Helpers Project"
    @echo ""
    @echo "Targets:"
    @echo "  all                 - Runs hooks, installs dev dependencies, lints, and tests."
    @echo "  model-install       - Installs PyTorch and downloads the Stable Diffusion model."
    @echo "  install-pytorch     - Installs PyTorch with the specified CUDA version."
    @echo "  download-diffusion-model - Downloads the Stable Diffusion model from Hugging Face."
    @echo "  hooks               - Installs pre-commit and pipenv, then installs pre-commit hooks."
    @echo "  clean               - Removes virtual environment, lock files, and requirements.txt."
    @echo "  lint                - Runs the pre-commit hooks for linting."
    @echo "  install             - Installs the project dependencies from Pipfile."
    @echo "  install-dev         - Installs the project development dependencies."
    @echo "  sync                - Syncs the virtual environment with Pipfile.lock."
    @echo "  sync-dev            - Syncs the virtual environment with Pipfile.lock (including dev)."
    @echo "  build               - Creates requirements.txt and builds using SAM."
    @echo "  run-frame-generator - Runs the frame generation script."
    @echo ""
    @echo "Variables:"
    @echo "  HUGGING_FACE_TOKEN  - Your Hugging Face access token (set in .env)."
    @echo "  PYTORCH_CUDA_VERSION - The CUDA version for PyTorch (default: $(PYTORCH_CUDA_VERSION))."
    @echo ""
    @echo "Usage:"
    @echo "  make <target>"
    @echo ""
    @echo "Example:"
    @echo "  make run-frame-generator"