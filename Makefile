UV     := uv
PYTHON := $(UV) run python

.PHONY: install venv sync add upgrade lock run run-fast run-no-stream \
        run-verbose run-model models pull-model pull-model-fast \
        search-fund ui clean clean-all clean-reports help

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
