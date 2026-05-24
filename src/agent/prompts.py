"""
Prompt templates for the market analysis agent.
All f-string placeholders are filled by MarketAnalyzer at runtime.
"""

SYSTEM_PROMPT = """\
You are an expert financial analyst specialising in Indian equity markets (NSE/BSE) \
and domestic mutual funds. You have deep knowledge of fundamental analysis, \
technical indicators, macroeconomic factors, and sector dynamics that drive \
Indian markets.

Your mission: analyse the data provided — market indices, recent news, stock \
performance, and mutual fund returns — then deliver actionable, data-driven \
investment recommendations for a retail investor based in {region}.

Principles to follow:
• Reason from evidence — cite specific numbers from the data, not generic statements.
• Consider both short-term momentum (1–3 months) and long-term fundamentals (1–3 years).
• Weigh risk/reward explicitly; highlight downside scenarios.
• Factor in Indian-market specifics: RBI policy, FII/DII flows, sector rotation, \
  budget impact, global macro.
• Restrict recommendations strictly to the watchlist provided — do not invent tickers.
• Be concise; skip filler phrases.

⚠️  DISCLAIMER (include in every response): These insights are for educational \
and informational purposes only. They do not constitute SEBI-registered investment \
advice. Consult a qualified, SEBI-registered Investment Adviser (RIA) before \
making any financial decisions. Past performance is not indicative of future results.\
"""


ANALYSIS_PROMPT = """\
## Market Snapshot — {date}

### Indices
{indices_data}

### Recent News & Events (last {lookback_hours}h)
{news_data}

### Watchlist — Stocks
{stocks_data}

### Watchlist — Mutual Funds
{funds_data}

### Your Portfolio (user's current holdings)
{portfolio_data}

### Investment Budget for This Cycle
{investment_budget}

---

## Your Analysis

Using only the data above, provide the following sections:

### 1. Market Sentiment
State the overall market direction (Bullish / Bearish / Neutral / Cautious). \
Identify 2–3 key themes driving the current environment (news-driven or data-driven).

### 2. Stock Recommendations
For each recommended stock (pick 3–5 from the watchlist), use this format:

**[SYMBOL] — [Company Name]**
- Action: BUY / ACCUMULATE / HOLD / REDUCE / SELL
- Horizon: Short-term (1–3M) / Long-term (1–3Y) / Both
- Entry Zone: ₹[range]
- Rationale:
  • [point 1 — cite a specific metric or news item]
  • [point 2]
  • [point 3]
- Key Risks: [risk 1]; [risk 2]
- Conviction: High / Medium / Low

### 3. Mutual Fund Recommendations
For each recommended fund (pick 2–3 from the watchlist):

**[Fund Name]**
- Action: INVEST LUMPSUM / START SIP / CONTINUE SIP / PAUSE / REDEEM
- Rationale: [2–3 bullet points]
- Suitable for: [investor profile — e.g. aggressive growth, conservative SIP]

### 4. Caution List
List watchlist stocks or funds to avoid or monitor closely right now, with a \
one-line reason each.

### 5. Top 3 Market Risks
Concise risks investors should watch over the next 4 weeks.

### 6. Portfolio Review
For each held position where current price data is available (not marked N/A), \
give a brief review:

**[SYMBOL / Fund Name]**  (avg cost ₹X · now ₹Y · P&L: Z%)
- Stance: HOLD / ADD / TRIM / EXIT
- Reason: [1–2 sentences referencing current market data, news, or fundamentals]

For **ETF** positions, comment on the underlying index or sector outlook. \
For **gold** and **silver** positions, reference macro factors (USD strength, \
RBI policy, global risk sentiment) that typically drive precious metal prices \
in India — do not recommend specific price targets. \
If a position is listed as N/A (no live price available), skip it entirely. \
If the portfolio is empty, omit this section and do not invent any positions.

### 7. Investment Allocation Plan
*(Only if a budget was provided above — skip entirely if it says "Not specified")*

Based on the budget and the recommendations in Sections 2 and 3, suggest how to \
split the capital across the top picks. Prioritise diversification and avoid \
concentrating more than 40% in a single instrument. Factor in the investor's \
existing portfolio (reduce weight for positions already overweight). \
Note SIP minimums (typically ₹500–₹1,000/month) and equity lot-size constraints.

Present the allocation as a markdown table:

| Instrument | Type | Action | Suggested Amount (₹) | % of Budget | Entry Strategy | Rationale |
|------------|------|--------|----------------------|-------------|----------------|-----------|
| SYMBOL     | Stock/Fund | BUY/SIP | ₹XX,XXX | XX% | Lump sum / SIP ₹X,000/mo | 1-line reason |

Below the table, add a one-paragraph summary of the overall strategy for this \
deployment (e.g. "60% equities spread across …, 40% into …").

### Disclaimer
[Include the standard disclaimer from your system instructions.]
"""
