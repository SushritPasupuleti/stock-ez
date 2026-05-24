# Stock-EZ

AI-powered stock & mutual fund recommendation agent for Indian markets.  
Runs **entirely locally** — no cloud LLM costs, no paid data APIs.

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

An AI agent that consumes news and ticker data to recommend stocks & funds for investment. Uses local LLMs.
