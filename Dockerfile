# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Install Python dependencies with uv
# This layer is cached unless pyproject.toml / uv.lock change.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

# Bring in the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency manifests first for maximum cache reuse
COPY pyproject.toml uv.lock ./

# Sync all runtime deps into /app/.venv (no dev extras, no project install)
RUN uv sync --frozen --no-install-project --no-dev

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Lean runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the pre-built virtual environment from the deps stage
COPY --from=deps /app/.venv /app/.venv

# Activate the venv for all subsequent RUN / CMD calls
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application source
COPY app.py main.py config.yaml ./
COPY src/ src/

# Create directories for persistent data (mount these as volumes in production)
RUN mkdir -p /app/data /app/reports

# Streamlit: run headless, disable usage-stats popup
RUN mkdir -p /root/.streamlit \
 && printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml \
 && printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\nfileWatcherType = "none"\n\n[browser]\ngatherUsageStats = false\n' \
        > /root/.streamlit/config.toml

EXPOSE 8501

# data/ holds portfolio.db and news_cache.db; reports/ holds Markdown/PDF exports
VOLUME ["/app/data", "/app/reports"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
        || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
