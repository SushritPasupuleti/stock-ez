UV     := uv
PYTHON := $(UV) run python

.PHONY: install venv sync add upgrade lock run run-fast run-no-stream \
        run-verbose run-model models pull-model pull-model-fast \
        search-fund ui \
        docker-build docker-build-arm64 docker-build-amd64 docker-push docker-run docker-clean \
        clean clean-all clean-venv clean-reports help

# ── Setup ───────────────────────────────────────────────────────────────────

## Create .venv and sync all dependencies from pyproject.toml
install:
	$(UV) sync

## Recreate the virtual environment from scratch
venv:
	$(UV) venv --python 3.11

## Sync dependencies (install/remove to match pyproject.toml exactly)
sync:
	$(UV) sync

## Add a package to pyproject.toml and install it
## Usage: make add PKG=httpx
add:
	$(UV) add $(PKG)

## Add a dev-only package
## Usage: make add-dev PKG=ruff
add-dev:
	$(UV) add --dev $(PKG)

## Upgrade all packages to latest allowed versions
upgrade:
	$(UV) sync --upgrade

## Export a requirements.txt snapshot (for Docker, CI, etc.)
lock-export:
	$(UV) export --no-hashes -o requirements.lock.txt

# ── Ollama models ────────────────────────────────────────────────────────────

## Pull the default model (qwen3:14b — hybrid thinking, ~9 GB)
pull-model:
	ollama pull qwen3:14b

## Pull a faster model (qwen3:8b — ~5 GB)
pull-model-fast:
	ollama pull qwen3:8b

## Pull Qwen3.5 9B (~6 GB)
pull-model-qwen35:
	ollama pull qwen3.5:9b

## Pull Gemma 3 27B (~17 GB, near 16 GB VRAM limit)
pull-model-gemma27b:
	ollama pull gemma3:27b

## List recommended models for 16 GB VRAM
models:
	$(PYTHON) main.py --list-models

# ── Run ──────────────────────────────────────────────────────────────────────

## Run a full analysis with the default model (streaming)
run:
	$(PYTHON) main.py

## Run with the fast 8B model
run-fast:
	$(PYTHON) main.py --model qwen3:8b

## Run without streaming (print full response at once)
run-no-stream:
	$(PYTHON) main.py --no-stream

## Run with verbose/debug logging
run-verbose:
	$(PYTHON) main.py --verbose

## Run with a custom model   Usage: make run-model MODEL=mistral-small3.2:24b
run-model:
	$(PYTHON) main.py --model $(MODEL)

# ── Streamlit UI ─────────────────────────────────────────────────────────────

## Launch the Streamlit web UI
ui:
	$(UV) run streamlit run app.py

# ── Docker ───────────────────────────────────────────────────────────────────

IMAGE  ?= stock-ez
TAG    ?= latest
BUILDX_NAME ?= stock-ez-builder
GHCR_IMAGE ?= ghcr.io/$(shell git remote get-url origin | sed -E 's#(https://github.com/|git@github.com:|git://github.com/)([^/]+)/.*#\2#' | tr '[:upper:]' '[:lower:]')/stock-ez
GHCR_TAG ?= latest

## Create a Buildx builder that supports multi-platform builds
buildx-init:
	docker buildx inspect $(BUILDX_NAME) >/dev/null 2>&1 || docker buildx create --name $(BUILDX_NAME) --driver docker-container --use
	docker buildx use $(BUILDX_NAME)
	docker buildx inspect --bootstrap >/dev/null 2>&1 || true

## Build a Docker image for the current machine's platform
docker-build:
	docker build -t $(IMAGE):$(TAG) .

## Build for Apple Silicon / arm64  (requires: docker buildx)
docker-build-arm64:
	docker buildx build --platform linux/arm64 --load -t $(IMAGE):$(TAG)-arm64 .

## Build for Intel / amd64  (requires: docker buildx)
docker-build-amd64:
	docker buildx build --platform linux/amd64 --load -t $(IMAGE):$(TAG)-amd64 .

## Build and push a multi-arch manifest (arm64 + amd64) to a registry
## Usage: make docker-push REGISTRY=docker.io/myuser
docker-push: buildx-init
	docker buildx build \
		--platform linux/arm64,linux/amd64 \
		-t $(REGISTRY)/$(IMAGE):$(TAG) \
		--push .

## Build and push a multi-arch manifest to GitHub Container Registry (GHCR)
## Usage: make docker-push-ghcr GHCR_IMAGE=ghcr.io/OWNER/stock-ez GHCR_TAG=latest
## Requires: docker login ghcr.io

docker-push-ghcr: buildx-init
	docker buildx build \
		--platform linux/arm64,linux/amd64 \
		-t $(GHCR_IMAGE):$(GHCR_TAG) \
		--push .

## Run the Streamlit UI in Docker
## Ollama must be running on the host; update config.yaml base_url to
##   http://host.docker.internal:11434  before running this target.
docker-run:
	docker run --rm -it \
		-p 8501:8501 \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/reports:/app/reports" \
		-v "$(PWD)/config.yaml:/app/config.yaml:ro" \
		--add-host=host.docker.internal:host-gateway \
		$(IMAGE):$(TAG)

## Stop and remove all containers built from this image
docker-clean:
	-docker ps -aq --filter "ancestor=$(IMAGE):$(TAG)"       | xargs docker rm -f 2>/dev/null; true
	-docker ps -aq --filter "ancestor=$(IMAGE):$(TAG)-arm64" | xargs docker rm -f 2>/dev/null; true
	-docker ps -aq --filter "ancestor=$(IMAGE):$(TAG)-amd64" | xargs docker rm -f 2>/dev/null; true

# ── Fund search ──────────────────────────────────────────────────────────────

## Search for a mutual fund scheme code   Usage: make search-fund QUERY="Axis Bluechip"
search-fund:
	$(PYTHON) main.py --search-fund "$(QUERY)"

# ── Cleanup ──────────────────────────────────────────────────────────────────

## Remove Python bytecode and __pycache__ directories
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null; true

## Remove generated reports
clean-reports:
	rm -rf reports/

## Remove the entire venv (re-run `make install` to recreate)
clean-venv:
	rm -rf .venv

## Remove everything — venv, cache, reports
clean-all: clean clean-reports clean-venv

# ── Help ─────────────────────────────────────────────────────────────────────

## Print this help message
help:
	@echo ""
	@echo "  Stock-EZ — available targets"
	@echo ""
	@grep -E '^##' Makefile | sed 's/^## /  /'
	@echo ""
