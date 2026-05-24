# Stock-EZ

AI-powered investment analysis for Indian markets — stocks, ETFs, mutual funds, and physical precious metals.  
Runs **entirely locally**: no cloud LLM costs, no paid data APIs, no accounts required.

---

## Features

### 📊 Live Market Data
- **47 NSE/BSE stocks** across Banking, IT, Infra, FMCG, Pharma, Auto, Energy, Telecom, Defence, and more
- **11 ETFs** — index (Nifty 50, Next 50, Bank), sector (IT, PSU Bank, Pharma, Infra), commodity (Gold BeES, Silver BeES), and international (FANG+)
- **Mutual fund NAVs** via AMFI / mfapi.in (no API key)
- **BSE fallback** — automatically retries `.BO` if an NSE symbol returns no data
- **NIFTY 50 / SENSEX indices**

### 💼 Portfolio Tracker
- Persistent SQLite-backed portfolio (`data/portfolio.db`)
- Supports **stocks, ETFs, mutual funds, physical gold, and physical silver**
- Live P&L enrichment on every analysis run
- For **physical gold/silver**: enter quantity in grams and buy price in ₹/gram — live prices are fetched directly from COMEX spot futures (GC=F, SI=F) converted to INR via live USD/INR rate (no ETF proxy)

### 📈 Gold & Silver Market Pulse
Always-visible buy/sell timing signal based on 1 year of COMEX history:
- **RSI(14)** — Wilder's smoothed (oversold < 30, overbought > 70)
- **vs 50-day MA** — short-term trend
- **vs 200-day MA** — long-term trend
- **52-week change** context
- **Gold/Silver ratio** — historical relative-value indicator (ratio > 85: silver cheap; < 65: gold cheap)
- Results are cached for 1 hour; signal is `BUY` / `HOLD` / `SELL`

### 🤖 Local LLM Analysis
- Streams analysis from any **Ollama** model running on your machine
- Structured output: market sentiment, stock picks, fund picks, caution list, risk factors, portfolio review
- Configurable model, context window, and temperature

### 📰 News Feed
- Aggregates financial news from RSS feeds (ET Markets, Moneycontrol, Business Standard, Mint, Yahoo Finance)
- **SQLite news cache** (`data/news_cache.db`) — avoids re-fetching duplicate articles
- Configurable lookback window and per-source limits

### 📄 Reports
- Markdown reports saved to `reports/` after each analysis run
- **PDF export** from the UI

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.11 | Via system or [Nix](https://nixos.org) |
| [Ollama](https://ollama.com) ≥ 0.3 | Must be running before launching the app |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager (recommended) |
| Docker + Buildx | Only for containerised deployment |

---

## Quick Start (local)

```bash
# 1. Clone
git clone https://github.com/SushritPasupuleti/stock-ez
cd stock-ez

# 2. Install dependencies
make install          # uses uv

# 3. Pull an LLM (see model table below)
make pull-model       # pulls qwen3:14b  (~9 GB)

# 4. Launch the Streamlit UI
make ui               # opens http://localhost:8501
```

---

## Quick Start (Docker)

> Ollama must be running on your **host** machine before starting the container.  
> Update `config.yaml` → `llm.base_url` to `http://host.docker.internal:11434`.

```bash
# Build for your current platform
make docker-build

# Run — mounts data/ and reports/ as volumes, config.yaml read-only
make docker-run       # opens http://localhost:8501
```

### Build for a specific platform

```bash
# Apple Silicon (M-series Macs)
make docker-build-arm64

# Intel / AMD x86-64
make docker-build-amd64

# Build and push a multi-arch manifest to a registry
make docker-push REGISTRY=docker.io/myuser
```

> **Linux note**: `--add-host=host.docker.internal:host-gateway` is included in  
> `docker-run` so Ollama is reachable from inside the container on Linux.  
> On macOS with Docker Desktop, `host.docker.internal` resolves automatically.

---

## Recommended Models

| Model | VRAM | Notes |
|---|---|---|
| **qwen3:14b** | ~9 GB | Hybrid thinking — **default** |
| mistral-small3.2:24b | ~15 GB | Strong reasoning |
| deepseek-r1:14b | ~9 GB | Excellent chain-of-thought |
| phi4:14b | ~9 GB | Microsoft analytical model |
| gemma3:27b | ~17 GB | Near 16 GB VRAM limit |
| gemma3:12b | ~8 GB | Balanced Google model |
| qwen3:8b | ~5 GB | Fast, lighter option |

```bash
make pull-model            # qwen3:14b (default)
make pull-model-fast       # qwen3:8b
make run-model MODEL=phi4:14b
```

---

## Configuration (`config.yaml`)

```yaml
llm:
  model: "qwen3:14b"
  base_url: "http://localhost:11434"   # use http://host.docker.internal:11434 in Docker
  temperature: 0.3
  num_ctx: 8192

region:
  country: "IN"
  market: "NSE"
  currency: "INR"
  timezone: "Asia/Kolkata"
  name: "Hyderabad, India"

watchlist:
  stocks:
    - symbol: "RELIANCE.NS"     # NSE:  SYMBOL.NS
      name: "Reliance Industries"
    - symbol: "RELIANCE.BO"     # BSE:  SYMBOL.BO (auto-fallback also works)
      name: "Reliance Industries"

  etfs:
    - symbol: "NIFTYBEES.NS"
      name: "Nippon Nifty 50 BeES ETF"
    - symbol: "MAFANG.NS"
      name: "Mirae Asset NYSE FANG+ ETF"

  funds:                        # AMFI scheme codes from mfapi.in
    - scheme_code: "119551"
      name: "HDFC Top 100 Fund - Direct Growth"

news:
  max_articles: 20
  lookback_hours: 24
```

### Finding mutual fund scheme codes

```bash
make search-fund QUERY="Parag Parikh"
# or: python main.py --search-fund "Parag Parikh"
```

---

## Portfolio Tracking

Open the **Portfolio** tab in the UI.

| Asset type | Symbol / Code | Quantity | Buy Price |
|---|---|---|---|
| `stock` | `HDFCBANK.NS` | shares | ₹/share |
| `etf` | `NIFTYBEES.NS` | units | ₹/unit |
| `fund` | `119551` (AMFI code) | units | ₹/unit |
| `gold` | any label, e.g. `GOLD_COINS` | **grams** | **₹/gram** |
| `silver` | any label, e.g. `SILVER_BARS` | **grams** | **₹/gram** |

Gold and silver live prices are fetched from **COMEX spot futures** × **USD/INR** and converted to ₹/gram automatically — no ETF proxy involved.

---

## Make Targets

```
make install              Install / sync dependencies (uv)
make ui                   Launch Streamlit UI  (localhost:8501)
make run                  CLI analysis with default model
make run-fast             CLI with qwen3:8b
make run-model MODEL=X    CLI with a custom model
make pull-model           Pull qwen3:14b
make pull-model-fast      Pull qwen3:8b
make search-fund Q="..."  Find an AMFI scheme code

make docker-build         Build Docker image (current platform)
make docker-build-arm64   Build for Apple Silicon (arm64)
make docker-build-amd64   Build for Intel x86-64 (amd64)
make docker-push          Push multi-arch manifest  (REGISTRY=... required)
make docker-run           Run container (mounts data/ reports/ config.yaml)
make docker-clean         Remove containers from this image

make clean                Remove __pycache__ and .pyc files
make clean-reports        Remove generated reports/
make clean-all            Remove venv, cache, and reports
```

---

## Data Sources

| Data | Source | API Key |
|---|---|---|
| Stock prices | Yahoo Finance via `yfinance` | None |
| Commodity spot (gold, silver) | COMEX futures via `yfinance` (GC=F, SI=F, USDINR=X) | None |
| Mutual fund NAV | [mfapi.in](https://mfapi.in) (AMFI) | None |
| Financial news | ET Markets, Moneycontrol, Business Standard, Mint, Yahoo Finance RSS | None |

---

## Project Structure

```
stock-ez/
├── app.py                  Streamlit UI (7-tab layout)
├── main.py                 CLI entry point
├── config.yaml             Watchlist, LLM, news configuration
├── Dockerfile              Multi-stage container build (uv + Python 3.11-slim)
├── Makefile                All dev + Docker commands
├── pyproject.toml          Python dependencies (uv)
├── src/
│   ├── agent/
│   │   ├── llm.py          Ollama client with streaming
│   │   └── prompts.py      Structured analysis prompt
│   ├── config/
│   │   └── settings.py     Config loader
│   ├── data_sources/
│   │   ├── stocks.py       yfinance fetcher with BSE fallback
│   │   ├── funds.py        mfapi.in NAV fetcher
│   │   ├── news.py         RSS feed parser
│   │   ├── news_cache.py   SQLite news cache
│   │   └── portfolio.py    Portfolio store + P&L + metal signals
│   └── utils/
│       └── pdf_export.py   PDF report generator
├── data/                   SQLite databases (gitignored, mounted as volume)
└── reports/                Markdown / PDF exports (gitignored, mounted as volume)
```

---

## Disclaimer

Stock-EZ is for **educational and informational purposes only**.  
It does not constitute SEBI-registered investment advice.  
Always consult a qualified, SEBI-registered Investment Adviser (RIA) before making investment decisions.  
Past performance is not indicative of future results.


---

## What it does

1. **Fetches live market data** — NIFTY 50, SENSEX, NSE/BSE watchlist stocks, and mutual fund NAVs from free public sources.
2. **Reads recent financial news** — parsed from free RSS feeds (ET Markets, Moneycontrol, Business Standard, Mint, Yahoo Finance).
3. **Runs analysis locally** — sends all context to an Ollama LLM running on your machine.
4. **Produces structured recommendations** — market sentiment, stock picks (BUY/HOLD/SELL), fund picks, caution list, and top risks.
5. **Saves a Markdown report** to `reports/` after each run.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| [Ollama](https://ollama.com) | ≥ 0.3 |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/SushritPasupuleti/stock-ez
cd stock-ez

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull a recommended model (see model table below)
ollama pull qwen2.5:14b

# 4. Run
python main.py
```

---

## Recommended Models (16 GB VRAM)

| Model | VRAM | Notes |
|---|---|---|
| **qwen2.5:14b** | ~9 GB | Best financial reasoning — **default** |
| deepseek-r1:14b | ~9 GB | Excellent chain-of-thought |
| phi4:14b | ~9 GB | Strong analytical reasoning (Microsoft) |
| gemma3:12b | ~8 GB | Balanced (Google) |
| llama3.1:8b | ~5 GB | Fast, lighter option |
| mistral:7b | ~4 GB | Very fast for quick runs |

```bash
python main.py --list-models   # prints this table in the terminal
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --config PATH           Path to config file (default: config.yaml)
  --model MODEL           Override Ollama model
  --region "CITY, COUNTRY" Override region shown in analysis
  --market {NSE,BSE}      Override market context
  --no-stream             Print full response at once (no token streaming)
  --skip-preflight        Skip Ollama connection checks
  --list-models           Show recommended models and exit
  --search-fund QUERY     Search for a mutual fund scheme code by name
  -v, --verbose           Enable debug logging
```

**Examples:**

```bash
# Use a lighter model for a faster run
python main.py --model llama3.1:8b

# Non-streaming output (useful when piping to a file)
python main.py --no-stream > report.txt

# Find the scheme code for a fund you want to track
python main.py --search-fund "Parag Parikh"

# Custom config file
python main.py --config my_watchlist.yaml
```

---

## Configuration (`config.yaml`)

```yaml
llm:
  model: "qwen2.5:14b"          # Ollama model name
  base_url: "http://localhost:11434"
  temperature: 0.3
  num_ctx: 8192                  # Context window size

region:
  country: "IN"
  market: "NSE"                  # NSE or BSE
  currency: "INR"
  timezone: "Asia/Kolkata"
  name: "Hyderabad, India"       # Shown in analysis context

watchlist:
  stocks:                        # Yahoo Finance symbols (.NS = NSE, .BO = BSE)
    - symbol: "RELIANCE.NS"
      name: "Reliance Industries"
    # ... add more
  funds:                         # AMFI scheme codes from mfapi.in
    - scheme_code: "119551"
      name: "HDFC Top 100 Fund - Direct Growth"
    # ... add more

market_indices:
  - symbol: "^NSEI"
    name: "NIFTY 50"

news:
  max_articles: 20
  lookback_hours: 24
  sources:
    - name: "Economic Times Markets"
      url: "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
      type: "rss"
    # ... add more RSS feeds
```

### Finding mutual fund scheme codes

```bash
# Search directly from the CLI
python main.py --search-fund "Axis Bluechip"

# Or via browser
# https://api.mfapi.in/mf/search?q=Axis+Bluechip
```

### Adding more stocks

Use the Yahoo Finance symbol format:
- NSE stocks: `SYMBOL.NS` (e.g. `RELIANCE.NS`)
- BSE stocks: `SYMBOL.BO` (e.g. `RELIANCE.BO`)

---

## Free Data Sources

| Data | Source | API Key |
|---|---|---|
| Stock prices & fundamentals | [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | None |
| Mutual fund NAV & returns | [mfapi.in](https://mfapi.in) (AMFI data) | None |
| Financial news | ET Markets, Moneycontrol, Business Standard, Mint, Yahoo Finance RSS | None |

---

## Output

Each run streams the analysis to the terminal and saves a Markdown report to `reports/analysis_YYYYMMDD_HHMMSS.md`.

Report structure:
- AI analysis (full LLM output)
- Raw data used: indices, stock table, fund table, news list

---

## Disclaimer

Stock-EZ is for **educational and informational purposes only**.  
It does not constitute SEBI-registered investment advice.  
Always consult a qualified, SEBI-registered Investment Adviser (RIA) before making investment decisions.  
Past performance is not indicative of future results.
