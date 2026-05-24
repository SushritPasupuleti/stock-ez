"""
Stock-EZ — Streamlit UI
Run with:  streamlit run app.py  (or: make ui)
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.llm import OllamaClient
from src.agent.prompts import ANALYSIS_PROMPT, SYSTEM_PROMPT
from src.config.settings import FundEntry, Settings, StockEntry
from src.data_sources.funds import FundData, FundFetcher
from src.data_sources.news import NewsArticle, NewsFetcher
from src.data_sources.news_cache import NewsCache
from src.data_sources.portfolio import EnrichedPosition, MetalSignal, Position, PortfolioStore
from src.data_sources.stocks import StockData, StockFetcher
from src.utils.pdf_export import build_pdf

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock-EZ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS: list[str] = [
    "qwen3:14b",
    "mistral-small3.2:24b",
    "deepseek-r1:14b",
    "phi4:14b",
    "gemma3:27b",
    "gemma3:12b",
    "qwen3.5:9b",
    "qwen3:8b",
]

# ── Lookback presets ──────────────────────────────────────────────────────────
# Maps display label → lookback hours (0 = custom)
_LOOKBACK_PRESETS: dict[str, int] = {
    "Last 24h": 24,
    "Last 48h": 48,
    "Last 3 Days": 72,
    "Last Week": 168,
    "Last 2 Weeks": 336,
    "Last Month": 720,
    "Custom": 0,
}

# Suggested max_articles per preset — balanced for signal richness vs. LLM context size.
# Rule of thumb: enough articles to cover the window without flooding the prompt.
_LOOKBACK_ARTICLE_DEFAULTS: dict[str, int] = {
    "Last 24h": 20,
    "Last 48h": 30,
    "Last 3 Days": 45,
    "Last Week": 60,
    "Last 2 Weeks": 80,
    "Last Month": 100,
}
# For custom durations: ~1.5 articles per hour, clamped to [15, 150]
_CUSTOM_ARTICLES_RATE: float = 1.5
_ARTICLES_WARN_THRESHOLD: int = 100  # above this, warn about LLM context pressure


def _suggested_max(preset_label: str, custom_hours: int = 24) -> int:
    """Return the default max_articles for the given lookback preset/hours."""
    if preset_label == "Custom":
        return max(15, min(150, int(_CUSTOM_ARTICLES_RATE * custom_hours)))
    return _LOOKBACK_ARTICLE_DEFAULTS.get(preset_label, 20)


# Callbacks fired by the selectbox / number-input on_change events.
# They update max_articles_input *before* the widget re-renders so the
# number input automatically resets to the new suggested default.
def _on_lookback_preset_change() -> None:
    label = st.session_state.get("lookback_preset", "Last 24h")
    custom_h = int(st.session_state.get("custom_hours_input", 24))
    st.session_state["max_articles_input"] = _suggested_max(label, custom_h)


def _on_custom_hours_change() -> None:
    custom_h = int(st.session_state.get("custom_hours_input", 24))
    st.session_state["max_articles_input"] = _suggested_max("Custom", custom_h)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_sections(text: str) -> dict[str, str]:
    """Split LLM markdown output into named sections by ### headers."""
    sections: dict[str, str] = {}
    # Prefix a newline so every header is preceded by \n
    parts = re.split(r"\n(?=### )", "\n" + text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        header = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections[header] = body
    return sections


def _style_pct(val: object) -> str:
    if not isinstance(val, (int, float)) or pd.isna(val):
        return ""
    return (
        "color: #16a34a; font-weight: 600"
        if float(val) >= 0
        else "color: #dc2626; font-weight: 600"
    )


def stocks_to_df(stocks: list[StockData]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": s.symbol.replace(".NS", "").replace(".BO", ""),
                "Name": s.name,
                "Price (₹)": s.current_price,
                "1D %": s.change_1d_pct,
                "5D %": s.change_5d_pct,
                "1M %": s.change_1m_pct,
                "P/E": s.pe_ratio,
                "MCap (Cr)": s.market_cap_cr,
                "52W High": s.week_52_high,
                "52W Low": s.week_52_low,
                "Sector": s.sector or "–",
            }
            for s in stocks
        ]
    )


def funds_to_df(funds: list[FundData]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Fund": f.name,
                "Code": f.scheme_code,
                "NAV (₹)": f.nav,
                "NAV Date": f.nav_date,
                "1M %": f.returns_1m,
                "3M %": f.returns_3m,
                "6M %": f.returns_6m,
                "1Y %": f.returns_1y,
                "Category": f.fund_category or f.fund_type or "–",
            }
            for f in funds
        ]
    )


def _fmt_pct(x: object) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "–"
    return f"{float(x):+.2f}%"


def _fmt_inr(x: object) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "–"
    return f"₹{float(x):,.2f}"


def _fmt_nav(x: object) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "–"
    return f"₹{float(x):.4f}"


def _fmt_mcap(x: object) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "–"
    return f"₹{float(x):,.0f}"


def _fmt_pe(x: object) -> str:
    if not isinstance(x, (int, float)) or pd.isna(x):
        return "–"
    return f"{float(x):.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, object] = {
    "analysis": None,
    "sections": {},
    "stocks": [],
    "funds": [],
    "news": [],
    "indices": [],
    "last_run": None,
    "is_running": False,
    "run_requested": False,
    # connection check state — persists so the pull button stays visible
    "conn_status": None,   # None | "ok" | "no_model" | "no_conn"
    "conn_model": None,    # which model the last check was for
    "conn_url": None,      # which URL the last check was for
    # config persistence
    "save_config_requested": False,
    # pdf report from last run
    "report_pdf_bytes": None,
    "report_pdf_name": None,
    # portfolio enrichment from last run
    "portfolio_enriched": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Load base settings once (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_settings() -> Settings:
    return Settings.load("config.yaml")


@st.cache_data(ttl=3600)
def _load_metal_signals() -> tuple[MetalSignal, MetalSignal, float | None]:
    """Fetch gold/silver market signals; cached for 1 hour."""
    return PortfolioStore.fetch_metal_signals()


_base = _load_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock-EZ")
    st.caption("AI-powered investment analysis · local LLM")
    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.subheader("🤖 Model")
    _model_idx = MODELS.index(_base.llm.model) if _base.llm.model in MODELS else 0
    selected_model = st.selectbox("Ollama model", MODELS, index=_model_idx)
    ollama_url = st.text_input("Ollama URL", value=_base.llm.base_url)
    temperature = st.slider("Temperature", 0.0, 1.0, float(_base.llm.temperature), 0.05)
    context_window = st.select_slider(
        "Context window",
        options=[4096, 8192, 16384, 32768],
        value=_base.llm.num_ctx,
    )

    # Reset stored status when user picks a different model or URL
    if (
        st.session_state.conn_model != selected_model
        or st.session_state.conn_url != ollama_url
    ):
        st.session_state.conn_status = None

    if st.button("🔌 Check connection", use_container_width=True):
        _chk = OllamaClient(model=selected_model, base_url=ollama_url)
        _conn = _chk.check_connection()
        _mdl = _chk.check_model() if _conn else False
        st.session_state.conn_model = selected_model
        st.session_state.conn_url = ollama_url
        if _conn and _mdl:
            st.session_state.conn_status = "ok"
        elif _conn:
            st.session_state.conn_status = "no_model"
        else:
            st.session_state.conn_status = "no_conn"

    # ── Persistent connection status + pull button ─────────────────────────
    _cs = st.session_state.conn_status
    if _cs == "ok":
        st.success(f"✓ Ollama running · **{selected_model}** available")
    elif _cs == "no_conn":
        st.error(f"Cannot reach Ollama at `{ollama_url}`")
    elif _cs == "no_model":
        st.warning(f"**{selected_model}** not found locally.")
        if st.button(
            f"⬇️ Pull {selected_model}",
            use_container_width=True,
            key="pull_model_btn",
        ):
            _puller = OllamaClient(model=selected_model, base_url=ollama_url)
            _pull_status = st.empty()
            _pull_bar = st.progress(0.0)
            try:
                for _msg, _pct in _puller.pull_model_stream():
                    _pull_status.caption(f"⏬ {_msg}")
                    if _pct > 0:
                        _pull_bar.progress(_pct)
                _pull_bar.progress(1.0)
                _pull_status.empty()
                _pull_bar.empty()
                st.success(f"✓ **{selected_model}** downloaded!")
                st.session_state.conn_status = "ok"
            except Exception as _pe:
                _pull_status.empty()
                _pull_bar.empty()
                st.error(f"Pull failed: {_pe}")

    st.divider()

    # ── Region ────────────────────────────────────────────────────────────────
    st.subheader("🌍 Region")
    region_name = st.text_input("Region name", value=_base.region.name)
    market = st.selectbox(
        "Market",
        ["NSE", "BSE"],
        index=0 if _base.region.market == "NSE" else 1,
    )
    st.divider()

    # ── Stocks watchlist ──────────────────────────────────────────────────────
    st.subheader("📋 Stocks Watchlist")
    _stocks_df = pd.DataFrame(
        [{"Symbol": s.symbol, "Name": s.name} for s in _base.watchlist.stocks]
    )
    edited_stocks = st.data_editor(
        _stocks_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="stocks_editor",
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", help="e.g. RELIANCE.NS"),
            "Name": st.column_config.TextColumn("Company name"),
        },
    )

    # ── Funds watchlist ───────────────────────────────────────────────────────
    st.subheader("💼 Funds Watchlist")
    _funds_df = pd.DataFrame(
        [{"Code": f.scheme_code, "Name": f.name} for f in _base.watchlist.funds]
    )
    edited_funds = st.data_editor(
        _funds_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="funds_editor",
        column_config={
            "Code": st.column_config.TextColumn(
                "Scheme Code", help="Find via the search box below"
            ),
            "Name": st.column_config.TextColumn("Fund name"),
        },
    )

    with st.expander("🔍 Find fund scheme code"):
        _q = st.text_input("Search", placeholder="Axis Bluechip", key="fund_search")
        if _q:
            with st.spinner("Searching mfapi.in…"):
                _results = FundFetcher.search(_q, limit=10)
            if _results:
                for _r in _results:
                    st.code(
                        f"{_r.get('schemeCode', '')}  {_r.get('schemeName', '')}",
                        language=None,
                    )
            else:
                st.caption("No results found.")

    if st.button("💾 Save Config to config.yaml", use_container_width=True, type="secondary"):
        st.session_state.save_config_requested = True

    st.divider()

    # ── News lookback ─────────────────────────────────────────────────────────
    st.subheader("⏱️ News Lookback")
    _preset_label = st.selectbox(
        "Duration preset",
        list(_LOOKBACK_PRESETS.keys()),
        index=0,
        key="lookback_preset",
        on_change=_on_lookback_preset_change,
        help="Pick a time window; max articles will auto-adjust to a sensible default.",
    )
    _preset_hours = _LOOKBACK_PRESETS[_preset_label]

    if _preset_hours == 0:  # ── Custom ───────────────────────────────────────
        lookback = int(
            st.number_input(
                "Custom hours",
                min_value=1,
                max_value=8760,
                value=max(1, int(_base.news.lookback_hours)),
                step=1,
                key="custom_hours_input",
                on_change=_on_custom_hours_change,
                help="Any value from 1 h to 8 760 h (1 year).",
            )
        )
        _prev_d = lookback / 24
        st.caption(
            f"Window: {lookback}h"
            + (f" (~{_prev_d:.1f} days)" if lookback >= 24 else "")
        )
    else:  # ── Preset ─────────────────────────────────────────────────────────
        lookback = _preset_hours
        _days, _rem_h = divmod(_preset_hours, 24)
        _parts: list[str] = ([f"{_days}d"] if _days else []) + ([f"{_rem_h}h"] if _rem_h else [])
        st.caption(f"Window: {' '.join(_parts)} of news history")

    # ── Max articles ──────────────────────────────────────────────────────────
    # Compute the suggested value for the current preset + custom hours.
    _custom_h_now = int(st.session_state.get("custom_hours_input", int(_base.news.lookback_hours)))
    _cur_suggested = _suggested_max(_preset_label, _custom_h_now)

    # Seed on first load (before any widget has been interacted with).
    if "max_articles_input" not in st.session_state:
        st.session_state["max_articles_input"] = _cur_suggested

    st.markdown("**📑 Max articles**")
    # The reset button is rendered first (column c2) so its click handler fires
    # before the number_input (column c1) is rendered, allowing session_state
    # to be updated in the same rerun.
    _ma_c2, _ma_c1 = st.columns([1, 4])
    with _ma_c2:
        if st.button(
            "↺",
            key="reset_ma_btn",
            help=f"Reset to suggested default ({_cur_suggested})",
            use_container_width=True,
        ):
            st.session_state["max_articles_input"] = _cur_suggested
    with _ma_c1:
        max_articles = st.number_input(
            "Max articles",
            min_value=5,
            max_value=500,
            step=5,
            key="max_articles_input",
            label_visibility="collapsed",
            help=(
                f"Suggested for **{_preset_label}**: {_cur_suggested}.  \n"
                "Increase for broader coverage; decrease for faster runs and "
                "lower LLM token usage."
            ),
        )

    # Contextual feedback below the inputs
    _ma_val = int(max_articles)
    _ma_ratio = _ma_val / max(_cur_suggested, 1)
    if _ma_val > _ARTICLES_WARN_THRESHOLD:
        st.warning(
            f"{_ma_val} articles may exceed the default LLM context window "
            f"({_base.llm.num_ctx:,} tokens). Consider raising Context Window "
            f"in Model settings or reducing to ≤ {_ARTICLES_WARN_THRESHOLD}."
        )
    elif _ma_ratio > 1.5:
        st.caption(
            f"↑ {_ma_val} articles (suggested {_cur_suggested}) "
            "— token usage will be higher than usual."
        )
    elif _ma_ratio < 0.5:
        st.caption(
            f"↓ {_ma_val} articles (suggested {_cur_suggested}) "
            "— news coverage may be sparse."
        )
    else:
        st.caption(f"✓ {_ma_val} articles · suggested {_cur_suggested}")

    # ── News cache stats ──────────────────────────────────────────────────────
    try:
        _sidebar_cache = NewsCache()
        _sc_stats = _sidebar_cache.stats()
        _sc_total = _sc_stats["total_articles"]
        if _sc_total > 0:
            _sc_age = _sc_stats.get("oldest_age_hours")
            _age_str = f" · oldest {_sc_age:.0f}h ago" if _sc_age else ""
            st.caption(f"📦 News cache: {_sc_total} articles{_age_str}")
            if st.button("🗑️ Clear news cache", use_container_width=True, key="clear_cache_btn"):
                _sidebar_cache.clear()
                st.success("Cache cleared")
        else:
            st.caption("📦 News cache: empty")
    except Exception:
        pass

    st.divider()

    # ── Run ───────────────────────────────────────────────────────────────────
    _run_btn = st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )
    if _run_btn:
        st.session_state.run_requested = True

    if st.session_state.last_run:
        st.caption(
            f"Last run: {st.session_state.last_run.strftime('%d %b %Y · %H:%M')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build runtime settings from sidebar values
# ─────────────────────────────────────────────────────────────────────────────
def _build_settings() -> Settings:
    s = Settings.load("config.yaml")
    s.llm.model = selected_model
    s.llm.base_url = ollama_url
    s.llm.temperature = temperature
    s.llm.num_ctx = context_window
    s.region.name = region_name
    s.region.market = market
    s.news.lookback_hours = int(lookback)
    s.news.max_articles = int(max_articles)
    s.watchlist.stocks = [
        StockEntry(symbol=str(row["Symbol"]).strip(), name=str(row.get("Name", "")).strip())
        for _, row in edited_stocks.iterrows()
        if pd.notna(row.get("Symbol")) and str(row["Symbol"]).strip()
    ]
    s.watchlist.funds = [
        FundEntry(scheme_code=str(row["Code"]).strip(), name=str(row.get("Name", "")).strip())
        for _, row in edited_funds.iterrows()
        if pd.notna(row.get("Code")) and str(row["Code"]).strip()
    ]
    return s


# Handle deferred save-config request (button was clicked in sidebar)
if st.session_state.get("save_config_requested"):
    st.session_state.save_config_requested = False
    try:
        _s_to_save = _build_settings()
        _s_to_save.save("config.yaml")
        _load_settings.clear()
        st.sidebar.success("✓ Saved to config.yaml")
    except Exception as _save_err:
        st.sidebar.error(f"Save failed: {_save_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Stock-EZ")
st.caption(
    "AI-powered stock & mutual fund analysis for Indian markets · "
    "powered by local LLMs via Ollama"
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis execution
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_requested:
    st.session_state.run_requested = False
    st.session_state.is_running = True

    settings = _build_settings()

    # ── Preflight ──────────────────────────────────────────────────────────
    _llm = OllamaClient(
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        num_ctx=settings.llm.num_ctx,
    )
    if not _llm.check_connection():
        st.error(f"❌ Cannot reach Ollama at **{settings.llm.base_url}**. Is it running?")
        st.session_state.is_running = False
        st.stop()
    if not _llm.check_model():
        st.warning(
            f"⚠️ Model **{settings.llm.model}** not found locally.  \n"
            f"Run `ollama pull {settings.llm.model}` then try again."
        )
        st.session_state.is_running = False
        st.stop()

    # ── Data fetching ──────────────────────────────────────────────────────
    with st.status("Gathering market data…", expanded=True) as _status:
        st.write("📰 Fetching news…")
        _news_cache = NewsCache()
        _news_fetcher = NewsFetcher(
            sources=settings.news.sources,
            max_articles=settings.news.max_articles,
            lookback_hours=settings.news.lookback_hours,
            cache=_news_cache,
            cache_ttl_minutes=30,
        )
        _news = _news_fetcher.fetch_all()
        _cache_total = _news_cache.stats()["total_articles"]
        st.write(f"✅ {len(_news)} articles  ·  {_cache_total} cached")

        st.write("📊 Fetching market indices…")
        _stock_fetcher = StockFetcher(region_config=settings.region)
        _indices = _stock_fetcher.fetch_indices(settings.market_indices)
        st.write(f"✅ {len(_indices)} indices")

        st.write(f"📈 Fetching {len(settings.watchlist.stocks)} stocks…")
        _stocks = _stock_fetcher.fetch_all(settings.watchlist.stocks)
        st.write(f"✅ {len(_stocks)} stocks")

        st.write(f"💼 Fetching {len(settings.watchlist.funds)} funds…")
        _fund_fetcher = FundFetcher()
        _funds = _fund_fetcher.fetch_all(settings.watchlist.funds)
        st.write(f"✅ {len(_funds)} funds")

        _status.update(label="Market data ready ✓", state="complete", expanded=False)

    # Store raw data in session state
    st.session_state.stocks = _stocks
    st.session_state.funds = _funds
    st.session_state.news = _news
    st.session_state.indices = _indices

    # ── Portfolio enrichment ───────────────────────────────────────────────
    _port_store = PortfolioStore()
    _portfolio_enriched = _port_store.enrich(_stocks, _funds)
    _portfolio_text = _port_store.format_for_prompt(_portfolio_enriched)
    st.session_state.portfolio_enriched = _portfolio_enriched

    # ── LLM streaming ──────────────────────────────────────────────────────
    st.subheader("🤖 AI Analysis")
    st.caption(f"Model: `{settings.llm.model}` · streaming…")

    _system_prompt = SYSTEM_PROMPT.format(region=settings.region.name)
    _user_prompt = ANALYSIS_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        lookback_hours=settings.news.lookback_hours,
        indices_data=_stock_fetcher.format_indices(_indices) or "No index data.",
        news_data=_news_fetcher.format_for_prompt(_news),
        stocks_data=_stock_fetcher.format_for_prompt(_stocks),
        funds_data=_fund_fetcher.format_for_prompt(_funds),
        portfolio_data=_portfolio_text,
    )

    with st.container(border=True):
        _full_response: str = st.write_stream(
            _llm.token_stream(_system_prompt, _user_prompt)
        )

    # ── Save report ────────────────────────────────────────────────────────
    _report_dir = Path(settings.output.report_dir)
    _report_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _report_path = _report_dir / f"analysis_{_ts}.md"
    _report_path.write_text(
        "# Stock-EZ Analysis\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  \n"
        f"**Model:** {settings.llm.model}  \n"
        f"**Region:** {settings.region.name}\n\n---\n\n"
        f"## AI Analysis\n\n{_full_response}\n\n---\n\n"
        f"## Raw Data\n\n### Indices\n\n{_stock_fetcher.format_indices(_indices)}\n\n"
        f"### Stocks\n\n{_stock_fetcher.format_for_prompt(_stocks)}\n\n"
        f"### Funds\n\n{_fund_fetcher.format_for_prompt(_funds)}\n\n"
        f"### News\n\n{_news_fetcher.format_for_prompt(_news)}\n",
        encoding="utf-8",
    )

    # ── Save PDF report ────────────────────────────────────────────────────
    _pdf_path = _report_dir / f"analysis_{_ts}.pdf"
    try:
        _pdf_bytes = build_pdf(
            analysis=_full_response,
            stocks_data=[vars(s) for s in _stocks],
            funds_data=[vars(f) for f in _funds],
            indices_data=[vars(i) for i in _indices],
            news_articles=_news,
            metadata={
                "date": datetime.now().strftime("%d %b %Y %H:%M IST"),
                "model": settings.llm.model,
                "region": settings.region.name,
            },
        )
        _pdf_path.write_bytes(_pdf_bytes)
        st.session_state.report_pdf_bytes = _pdf_bytes
        st.session_state.report_pdf_name = _pdf_path.name
    except Exception as _pdf_err:
        st.session_state.report_pdf_bytes = None
        st.session_state.report_pdf_name = None
        st.warning(f"PDF generation skipped: {_pdf_err}")

    # Persist analysis in session state
    st.session_state.analysis = _full_response
    st.session_state.sections = parse_sections(_full_response)
    st.session_state.last_run = datetime.now()
    st.session_state.is_running = False

    st.success(f"✓ Analysis complete · report saved → `{_report_path}`")
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_stocks, tab_funds, tab_portfolio, tab_news, tab_raw, tab_reports, tab_help = st.tabs(
    ["📊 Overview", "📈 Stocks", "💼 Funds", "💰 Portfolio", "📰 News", "🗂️ Raw Data", "🗃️ Reports", "❓ Help"]
)

_has_data: bool = st.session_state.analysis is not None

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    if not _has_data:
        st.info(
            "Run an analysis from the sidebar to see results here.\n\n"
            "**Quick start:**\n"
            "1. Check Ollama is running (`ollama serve`)\n"
            "2. Confirm model is pulled (`ollama pull qwen3:14b`)\n"
            "3. Click **▶ Run Analysis** in the sidebar\n\n"
            "➡️ New here? Open the **❓ Help** tab for a full feature guide and glossary."
        )
    else:
        _indices_list: list[StockData] = st.session_state.indices
        _sections: dict[str, str] = st.session_state.sections

        # Index metric cards
        if _indices_list:
            _idx_cols = st.columns(len(_indices_list))
            for _col, _idx in zip(_idx_cols, _indices_list):
                _price = f"{_idx.current_price:,.2f}" if _idx.current_price else "–"
                _delta = _idx.change_1d_pct
                _col.metric(
                    _idx.name,
                    _price,
                    f"{_delta:+.2f}%" if _delta is not None else None,
                    delta_color="normal",
                )
            st.divider()

        # Market sentiment
        _sent_key = next(
            (k for k in _sections if "Sentiment" in k or "sentiment" in k), None
        )
        if _sent_key:
            st.subheader("🌡️ Market Sentiment")
            _sent_text = _sections[_sent_key]
            _first = _sent_text.split("\n")[0].lower()
            _badge = (
                "🟢 **Bullish**"
                if "bullish" in _first
                else "🔴 **Bearish**"
                if "bearish" in _first
                else "🟡 **Cautious**"
                if "cautious" in _first
                else "⚪ **Neutral**"
            )
            st.markdown(_badge)
            st.markdown(_sent_text)
            st.divider()

        # Risks
        _risk_key = next(
            (k for k in _sections if "Risk" in k or "risk" in k), None
        )
        if _risk_key:
            st.subheader("⚠️ Top Market Risks")
            st.markdown(_sections[_risk_key])
            st.divider()

        # Caution list
        _caut_key = next(
            (k for k in _sections if "Caution" in k or "caution" in k or "Avoid" in k),
            None,
        )
        if _caut_key:
            st.subheader("🚫 Caution List")
            st.markdown(_sections[_caut_key])
            st.divider()

        # Disclaimer
        _disc_key = next((k for k in _sections if "Disclaimer" in k), None)
        if _disc_key:
            with st.expander("📋 Disclaimer"):
                st.markdown(_sections[_disc_key])

        if st.session_state.last_run:
            st.caption(
                f"Analysis generated: {st.session_state.last_run.strftime('%d %b %Y · %H:%M IST')}"
            )

# ── Stocks ─────────────────────────────────────────────────────────────────────
with tab_stocks:
    if not _has_data:
        st.info("Run an analysis from the sidebar to see stock data and recommendations.")
    else:
        _stocks_list: list[StockData] = st.session_state.stocks
        _sections = st.session_state.sections

        if _stocks_list:
            st.subheader("Watchlist Performance")

            # Top movers row
            _sorted_by_day = sorted(
                [s for s in _stocks_list if s.change_1d_pct is not None],
                key=lambda s: s.change_1d_pct,  # type: ignore[arg-type]
            )
            if len(_sorted_by_day) >= 2:
                _loser = _sorted_by_day[0]
                _gainer = _sorted_by_day[-1]
                _m1, _m2, _m3 = st.columns(3)
                _m1.metric(
                    "📈 Top Gainer (1D)",
                    _gainer.symbol.replace(".NS", "").replace(".BO", ""),
                    f"{_gainer.change_1d_pct:+.2f}%",
                )
                _m2.metric(
                    "📉 Top Loser (1D)",
                    _loser.symbol.replace(".NS", "").replace(".BO", ""),
                    f"{_loser.change_1d_pct:+.2f}%",
                )
                _m3.metric("Total Stocks", len(_stocks_list))
                st.divider()

            _sdf = stocks_to_df(_stocks_list)
            _pct_cols = ["1D %", "5D %", "1M %"]
            _styled_stocks = (
                _sdf.style.map(_style_pct, subset=_pct_cols)
                .format(
                    {
                        "Price (₹)": _fmt_inr,
                        "1D %": _fmt_pct,
                        "5D %": _fmt_pct,
                        "1M %": _fmt_pct,
                        "P/E": _fmt_pe,
                        "MCap (Cr)": _fmt_mcap,
                        "52W High": _fmt_inr,
                        "52W Low": _fmt_inr,
                    }
                )
            )
            st.dataframe(_styled_stocks, use_container_width=True, hide_index=True)
            st.divider()

        # LLM recommendations
        _srec_key = next(
            (k for k in _sections if "Stock Rec" in k or "Stocks" in k), None
        )
        if _srec_key:
            st.subheader("🤖 AI Stock Recommendations")
            st.markdown(_sections[_srec_key])
        elif st.session_state.analysis:
            st.subheader("🤖 Full AI Analysis")
            st.markdown(st.session_state.analysis)

# ── Funds ──────────────────────────────────────────────────────────────────────
with tab_funds:
    if not _has_data:
        st.info("Run an analysis from the sidebar to see fund data and recommendations.")
    else:
        _funds_list: list[FundData] = st.session_state.funds
        _sections = st.session_state.sections

        if _funds_list:
            st.subheader("Fund Performance")

            # Summary metrics
            _valid_1y = [f for f in _funds_list if f.returns_1y is not None]
            if _valid_1y:
                _best = max(_valid_1y, key=lambda f: f.returns_1y)  # type: ignore[arg-type]
                _worst = min(_valid_1y, key=lambda f: f.returns_1y)  # type: ignore[arg-type]
                _fm1, _fm2, _fm3 = st.columns(3)
                _fm1.metric(
                    "🏆 Best 1Y Return",
                    _best.name[:25] + "…" if len(_best.name) > 25 else _best.name,
                    f"{_best.returns_1y:+.2f}%",
                )
                _fm2.metric(
                    "📉 Worst 1Y Return",
                    _worst.name[:25] + "…" if len(_worst.name) > 25 else _worst.name,
                    f"{_worst.returns_1y:+.2f}%",
                )
                _fm3.metric("Total Funds", len(_funds_list))
                st.divider()

            _fdf = funds_to_df(_funds_list)
            _fpct_cols = ["1M %", "3M %", "6M %", "1Y %"]
            _styled_funds = (
                _fdf.style.map(_style_pct, subset=_fpct_cols)
                .format(
                    {
                        "NAV (₹)": _fmt_nav,
                        "1M %": _fmt_pct,
                        "3M %": _fmt_pct,
                        "6M %": _fmt_pct,
                        "1Y %": _fmt_pct,
                    }
                )
            )
            st.dataframe(_styled_funds, use_container_width=True, hide_index=True)
            st.divider()

        # LLM recommendations
        _frec_key = next(
            (k for k in _sections if "Fund Rec" in k or "Mutual Fund" in k), None
        )
        if _frec_key:
            st.subheader("🤖 AI Fund Recommendations")
            st.markdown(_sections[_frec_key])

# ── Portfolio ──────────────────────────────────────────────────────────────────
with tab_portfolio:
    _port_store = PortfolioStore()

    st.subheader("💰 My Portfolio")
    st.caption(
        "Track your holdings · edits persist to `data/portfolio.db` · "
        "positions are enriched with live prices on every analysis run"
    )

    # ── Precious metals market pulse (always visible) ─────────────────────
    _SIGNAL_ICON  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "N/A": "⚪"}
    _SIGNAL_COLOR = {"BUY": st.success, "SELL": st.error, "HOLD": st.warning, "N/A": st.info}

    with st.expander("📊 Gold & Silver Market Pulse", expanded=True):
        with st.spinner("Fetching COMEX spot data…"):
            _gold_sig, _silver_sig, _gs_ratio = _load_metal_signals()

        _pm_col1, _pm_col2 = st.columns(2)
        for _ms, _col in zip([_gold_sig, _silver_sig], [_pm_col1, _pm_col2]):
            with _col:
                _icon  = _SIGNAL_ICON.get(_ms.signal, "⚪")
                _cfn   = _SIGNAL_COLOR.get(_ms.signal, st.info)
                st.markdown(f"#### {_icon} {_ms.metal}")
                if _ms.price_inr is not None:
                    _delta_lbl = (
                        f"{_ms.change_52w_pct:+.1f}% (52w)"
                        if _ms.change_52w_pct is not None else None
                    )
                    st.metric("Spot price", f"₹{_ms.price_inr:,.2f} /gram", _delta_lbl)
                _cfn(f"**{_ms.signal}**")
                for _reason in _ms.reasons:
                    st.caption(f"• {_reason}")

        if _gs_ratio is not None:
            st.divider()
            if _gs_ratio > 85:
                st.info(
                    f"**Gold/Silver Ratio: {_gs_ratio}** — Historically elevated. "
                    "Silver is relatively cheap vs gold; may favour adding silver."
                )
            elif _gs_ratio < 65:
                st.info(
                    f"**Gold/Silver Ratio: {_gs_ratio}** — Historically low. "
                    "Gold is relatively cheap vs silver; may favour adding gold."
                )
            else:
                st.caption(
                    f"Gold/Silver ratio: {_gs_ratio} "
                    f"(historical range ~65–85; currently neutral)"
                )

    # ── Enriched P&L view (visible after a run) ───────────────────────────
    _ep_list: list[EnrichedPosition] = st.session_state.get("portfolio_enriched", [])
    if _has_data and _ep_list:
        _total_cost = sum(ep.cost_basis for ep in _ep_list)
        _valued_eps = [ep for ep in _ep_list if ep.current_value is not None]
        _total_current = sum(ep.current_value for ep in _valued_eps)  # type: ignore[misc]

        _pmc1, _pmc2, _pmc3 = st.columns(3)
        _pmc1.metric("Total Invested", f"₹{_total_cost:,.0f}")
        if _valued_eps and _total_cost > 0:
            _total_pnl = _total_current - _total_cost
            _total_pnl_pct = _total_pnl / _total_cost * 100
            _pmc2.metric("Current Value", f"₹{_total_current:,.0f}")
            _pmc3.metric(
                "Unrealised P&L",
                f"₹{_total_pnl:+,.0f}",
                f"{_total_pnl_pct:+.2f}%",
            )
        else:
            _pmc2.metric("Current Value", "N/A")
            _pmc3.metric("Unrealised P&L", "N/A")

        st.divider()

        # Enriched positions table
        _ep_rows = []
        for _ep in _ep_list:
            _p = _ep.position
            _ep_rows.append(
                {
                    "Symbol/Code": _p.symbol,
                    "Name": _p.name,
                    "Type": _p.asset_type,
                    "Qty": _p.quantity,
                    "Buy Price (₹)": _p.buy_price,
                    "Cost Basis (₹)": _ep.cost_basis,
                    "Current (₹)": _ep.current_price,
                    "Current Value (₹)": _ep.current_value,
                    "P&L (₹)": _ep.pnl_abs,
                    "P&L %": _ep.pnl_pct,
                    "Buy Date": _p.buy_date or "–",
                    "Notes": _p.notes or "–",
                }
            )
        _ep_df = pd.DataFrame(_ep_rows)
        _ep_num_cols = {
            "Buy Price (₹)": _fmt_inr,
            "Cost Basis (₹)": _fmt_inr,
            "Current (₹)": _fmt_inr,
            "Current Value (₹)": _fmt_inr,
            "P&L (₹)": lambda x: f"₹{float(x):+,.0f}",
            "P&L %": lambda x: f"{float(x):+.2f}%",
            "Qty": "{:.4f}",
        }
        st.dataframe(
            _ep_df.style.map(_style_pct, subset=["P&L %"]).format(
                _ep_num_cols, na_rep="N/A"
            ),
            use_container_width=True,
            hide_index=True,
        )

        # AI portfolio review section (generated by LLM)
        _port_rec_key = next(
            (k for k in st.session_state.sections if "Portfolio" in k), None
        )
        if _port_rec_key:
            st.divider()
            st.subheader("🤖 AI Portfolio Review")
            st.markdown(st.session_state.sections[_port_rec_key])

        st.divider()

    # ── Editor (always visible) ───────────────────────────────────────────
    st.subheader("✏️ Edit Positions")
    st.caption(
        "**stock** / **etf**: use the exact NSE symbol (e.g. `HDFCBANK.NS`, `GOLDBEES.NS`)  ·  "
        "**fund**: AMFI scheme code (e.g. `119551`)  ·  "
        "**gold** / **silver**: enter quantity in **grams**, buy price in **₹/gram** — "
        "live price is fetched from COMEX spot × USD/INR (no ETF proxy needed)"
    )

    _positions_all = _port_store.list_all()
    _port_edit_data = [
        {
            "id": p.id,
            "Symbol/Code": p.symbol,
            "Name": p.name,
            "Type": p.asset_type,
            "Quantity": p.quantity,
            "Buy Price (₹)": p.buy_price,
            "Buy Date": p.buy_date or "",
            "Notes": p.notes or "",
        }
        for p in _positions_all
    ]
    _port_edit_df = pd.DataFrame(
        _port_edit_data,
        columns=["id", "Symbol/Code", "Name", "Type", "Quantity", "Buy Price (₹)", "Buy Date", "Notes"],
    )
    _edited_port = st.data_editor(
        _port_edit_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="portfolio_editor",
        column_config={
            "id": None,  # hidden — preserves row identity through edits
            "Symbol/Code": st.column_config.TextColumn(
                "Symbol / Scheme Code",
                help="Stock: `HDFCBANK.NS`   Fund: `119551`",
            ),
            "Name": st.column_config.TextColumn("Name"),
            "Type": st.column_config.SelectboxColumn(
                "Type", options=["stock", "etf", "fund", "gold", "silver"], required=True
            ),
            "Quantity": st.column_config.NumberColumn(
                "Quantity", min_value=0.0001, format="%.4f", step=1.0,
                help="Shares for stocks; units for mutual funds",
            ),
            "Buy Price (₹)": st.column_config.NumberColumn(
                "Avg Buy Price (₹)", min_value=0.0, format="%.2f", step=0.01,
                help="Average price per share / unit at time of purchase",
            ),
            "Buy Date": st.column_config.TextColumn(
                "Buy Date", help="Optional — format YYYY-MM-DD"
            ),
            "Notes": st.column_config.TextColumn("Notes"),
        },
    )

    if st.button(
        "💾 Save Portfolio", type="secondary", use_container_width=True
    ):
        _new_positions: list[Position] = []
        for _, _row in _edited_port.iterrows():
            _sym = str(_row.get("Symbol/Code", "")).strip()
            if not _sym:
                continue
            # Recover the original id (hidden column, NaN for new rows)
            _pid_raw = _row.get("id")
            _pid: int | None = (
                int(_pid_raw)
                if _pid_raw is not None
                and not (isinstance(_pid_raw, float) and pd.isna(_pid_raw))
                else None
            )
            _qty = float(_row.get("Quantity") or 0)
            _bp = float(_row.get("Buy Price (₹)") or 0)
            if _qty <= 0 or _bp < 0:
                continue
            # buy_date — accepts strings, date objects, or NaN/None
            _bd_raw = _row.get("Buy Date")
            _bd: str | None = None
            if _bd_raw is not None and not (
                isinstance(_bd_raw, float) and pd.isna(_bd_raw)
            ):
                _bd_str = str(_bd_raw).strip()
                if _bd_str and _bd_str not in ("NaT", "None", "nan"):
                    _bd = _bd_str
            _notes_raw = _row.get("Notes")
            _notes: str | None = (
                str(_notes_raw).strip()
                if _notes_raw is not None
                and not (isinstance(_notes_raw, float) and pd.isna(_notes_raw))
                and str(_notes_raw).strip()
                else None
            )
            _new_positions.append(
                Position(
                    id=_pid,
                    symbol=_sym,
                    name=str(_row.get("Name", "")).strip() or _sym,
                    asset_type=str(_row.get("Type", "stock")).strip(),
                    quantity=_qty,
                    buy_price=_bp,
                    buy_date=_bd,
                    notes=_notes,
                )
            )
        _port_store.upsert_bulk(_new_positions)
        st.success(f"✓ Saved {len(_new_positions)} position(s)")
        st.rerun()

    if not _has_data and _positions_all:
        st.info(
            "▶ Run an analysis from the sidebar to enrich these positions "
            "with live prices and see your unrealised P&L."
        )

# ── News ───────────────────────────────────────────────────────────────────────
with tab_news:
    _news_list: list[NewsArticle] = st.session_state.news
    if not _news_list:
        st.info("Run an analysis from the sidebar to fetch news articles.")
    else:
        _sources_all = sorted({a.source for a in _news_list})
        _col_filter, _col_count = st.columns([3, 1])
        with _col_filter:
            _sel_sources = st.multiselect(
                "Filter by source", _sources_all, default=_sources_all
            )
        with _col_count:
            st.metric("Articles", len(_news_list))

        _filtered_news = [a for a in _news_list if a.source in _sel_sources]

        for _article in _filtered_news:
            with st.expander(f"**{_article.title}**  —  *{_article.source}*"):
                if _article.published:
                    st.caption(_article.published.strftime("%d %b %Y  %H:%M UTC"))
                if _article.summary:
                    st.write(_article.summary)
                if _article.url:
                    st.markdown(f"[Read full article ↗]({_article.url})")

# ── Raw Data ───────────────────────────────────────────────────────────────────
with tab_raw:
    if not _has_data:
        st.info("Run an analysis to see raw market data.")
    else:
        _raw_stocks: list[StockData] = st.session_state.stocks
        _raw_funds: list[FundData] = st.session_state.funds
        _raw_indices: list[StockData] = st.session_state.indices

        st.subheader("Market Indices")
        if _raw_indices:
            _idf = pd.DataFrame(
                [
                    {
                        "Index": i.name,
                        "Symbol": i.symbol,
                        "Price": i.current_price,
                        "1D %": i.change_1d_pct,
                        "5D %": i.change_5d_pct,
                        "1M %": i.change_1m_pct,
                    }
                    for i in _raw_indices
                ]
            )
            st.dataframe(
                _idf.style.map(_style_pct, subset=["1D %", "5D %", "1M %"]).format(
                    {"Price": _fmt_inr, "1D %": _fmt_pct, "5D %": _fmt_pct, "1M %": _fmt_pct}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No index data available.")

        st.subheader("Stocks")
        if _raw_stocks:
            st.dataframe(
                stocks_to_df(_raw_stocks).style.map(
                    _style_pct, subset=["1D %", "5D %", "1M %"]
                ).format(
                    {
                        "Price (₹)": _fmt_inr,
                        "1D %": _fmt_pct,
                        "5D %": _fmt_pct,
                        "1M %": _fmt_pct,
                        "P/E": _fmt_pe,
                        "MCap (Cr)": _fmt_mcap,
                        "52W High": _fmt_inr,
                        "52W Low": _fmt_inr,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No stock data available.")

        st.subheader("Mutual Funds")
        if _raw_funds:
            st.dataframe(
                funds_to_df(_raw_funds).style.map(
                    _style_pct, subset=["1M %", "3M %", "6M %", "1Y %"]
                ).format(
                    {
                        "NAV (₹)": _fmt_nav,
                        "1M %": _fmt_pct,
                        "3M %": _fmt_pct,
                        "6M %": _fmt_pct,
                        "1Y %": _fmt_pct,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No fund data available.")

        with st.expander("Full AI Analysis (raw markdown)"):
            st.code(st.session_state.analysis or "", language="markdown")

# ── Reports ─────────────────────────────────────────────────────────────────────
with tab_reports:
    _report_dir = Path("reports")
    _report_files = sorted(_report_dir.glob("*.md"), reverse=True) if _report_dir.exists() else []

    if not _report_files:
        st.info("No saved reports yet. Run an analysis to generate one.")
    else:
        _rc1, _rc2 = st.columns([4, 1])
        with _rc2:
            st.metric("Saved Reports", len(_report_files))
        with _rc1:
            _sel_report = st.selectbox(
                "Select report",
                _report_files,
                format_func=lambda p: (
                    p.stem.replace("analysis_", "").replace("_", " ")
                ),
            )

        if _sel_report:
            _rb1, _rb2, _rb3, _rb4 = st.columns([1, 1, 1, 3])
            with _rb1:
                if st.button("🗑️ Delete", type="secondary", key="del_report"):
                    _sel_report.unlink()
                    _pdf_sibling = _sel_report.with_suffix(".pdf")
                    if _pdf_sibling.exists():
                        _pdf_sibling.unlink()
                    st.rerun()
            with _rb2:
                st.download_button(
                    "⬇️ Markdown",
                    data=_sel_report.read_text(encoding="utf-8"),
                    file_name=_sel_report.name,
                    mime="text/markdown",
                    key="dl_report",
                )
            with _rb3:
                # Look for a PDF saved alongside this report
                _pdf_sibling = _sel_report.with_suffix(".pdf")
                if _pdf_sibling.exists():
                    st.download_button(
                        "📄 PDF",
                        data=_pdf_sibling.read_bytes(),
                        file_name=_pdf_sibling.name,
                        mime="application/pdf",
                        key="dl_report_pdf",
                    )
                elif st.session_state.get("report_pdf_bytes"):
                    st.download_button(
                        "📄 PDF",
                        data=st.session_state.report_pdf_bytes,
                        file_name=st.session_state.report_pdf_name or "analysis.pdf",
                        mime="application/pdf",
                        key="dl_report_pdf_session",
                    )

            st.divider()
            st.markdown(_sel_report.read_text(encoding="utf-8"))

# ── Help & Glossary ────────────────────────────────────────────────────────────
with tab_help:
    st.title("❓ Help & Getting Started")
    st.caption(
        "Everything you need to know about Stock-EZ — always available, "
        "no analysis run required."
    )

    # ── Quick-start ──────────────────────────────────────────────────────────
    with st.expander("🚀 Quick Start (5 steps)", expanded=False):
        st.markdown("""
**1 — Install Ollama and pull a model**
```bash
# macOS / Linux
brew install ollama          # or download from https://ollama.com
ollama pull qwen3:14b        # recommended default (~9 GB)
```

**2 — Install Stock-EZ dependencies**
```bash
git clone https://github.com/SushritPasupuleti/stock-ez
cd stock-ez
make install      # creates a .venv via uv and installs all packages
```

**3 — Launch the UI**
```bash
make ui           # opens http://localhost:8501
```

**4 — Configure your watchlist (sidebar)**
- Expand **📋 Stocks Watchlist** to add/remove NSE stocks (e.g. `RELIANCE.NS`)
- Expand **💼 Funds Watchlist** to add mutual fund scheme codes (e.g. `119551`)
- Use the **🔍 Find fund scheme code** search box to look up AMFI codes
- Adjust the **Ollama URL** if your model server is not on localhost
- Click **💾 Save Config** to persist your watchlist to `config.yaml`

**5 — Run an analysis**
- Click **▶ Run Analysis** in the sidebar
- The app fetches live prices, NAVs, and news, then streams an AI summary
- Results appear across the **Overview**, **Stocks**, **Funds**, and **Portfolio** tabs
""")

    # ── Feature tour ─────────────────────────────────────────────────────────
    with st.expander("🗺️ Feature Tour — what each tab does", expanded=True):
        st.markdown("""
| Tab | What you will find |
|---|---|
| **📊 Overview** | Market index metrics (NIFTY 50, SENSEX), AI market sentiment, top risks, and caution list |
| **📈 Stocks** | Watchlist price table (1D / 5D / 1M returns, P/E, market cap, 52-week range) + AI stock picks |
| **💼 Funds** | Mutual fund NAV table (1M / 3M / 6M / 1Y returns) + AI fund recommendations |
| **💰 Portfolio** | Unrealised P&L dashboard, Gold & Silver Market Pulse (always visible), position editor |
| **📰 News** | Financial news from 5+ sources, filterable by outlet |
| **🗂️ Raw Data** | Unformatted price tables and full LLM markdown output |
| **🗃️ Reports** | Saved Markdown and PDF analysis reports with download buttons |
| **❓ Help** | This guide + glossary |

---

**Portfolio tab — always-on features**

The **Gold & Silver Market Pulse** section is always shown (no analysis run needed). It displays:
- Live COMEX spot prices converted to ₹/gram
- RSI(14) momentum signal
- Position vs 50-day and 200-day moving averages
- 52-week return context
- A composite **BUY / HOLD / SELL** signal refreshed every hour
- Gold/Silver ratio with historical context

---

**Sidebar quick reference**

| Control | Purpose |
|---|---|
| **Ollama model** | Which local LLM to use |
| **Ollama URL** | Ollama server address (default `http://localhost:11434`) |
| **Temperature** | AI creativity — 0 = focused, 1 = varied |
| **Context window** | Max tokens fed to the LLM |
| **🔌 Check connection** | Verify Ollama is running and the model is downloaded |
| **⬇️ Pull model** | Download a model that is not yet available locally |
| **Region / Market** | NSE vs BSE preference for symbol resolution |
| **Stocks Watchlist** | Add / remove NSE/BSE symbols |
| **Funds Watchlist** | Add / remove AMFI scheme codes |
| **🔍 Find fund code** | Search AMFI by fund name |
| **💾 Save Config** | Write current watchlist + settings to `config.yaml` |
| **⏱️ News Lookback** | How far back to fetch news |
| **📑 Max articles** | Cap on articles passed to the LLM |
| **▶ Run Analysis** | Trigger a full data fetch + AI analysis |
""")

    # ── Asset types & symbols ────────────────────────────────────────────────
    with st.expander("🏷️ Asset Types & Symbol Formats", expanded=True):
        st.markdown("""
| Asset type | Format | Example | Notes |
|---|---|---|---|
| **stock** | `TICKER.NS` or `TICKER.BO` | `HDFCBANK.NS` | App auto-retries `.BO` if `.NS` has no data |
| **etf** | `TICKER.NS` | `NIFTYBEES.NS` | Same format as stocks |
| **fund** | AMFI scheme code (number) | `119551` | Use the 🔍 search box to find codes |
| **gold** | Any label | `GOLD_COINS` | Quantity in **grams**, price in **₹/gram** — live rate from COMEX |
| **silver** | Any label | `SILVER_BARS` | Quantity in **grams**, price in **₹/gram** — live rate from COMEX |

> **Tip:** To find an AMFI scheme code, open the **🔍 Find fund scheme code** expander  
> in the sidebar and type part of the fund name.
""")

    # ── Docker ───────────────────────────────────────────────────────────────
    with st.expander("🐳 Running via Docker"):
        st.markdown("""
Stock-EZ ships with a multi-stage `Dockerfile` that produces a lean Python 3.11-slim image.

```bash
# Build for your current machine
make docker-build

# Build for Apple Silicon (M-series)
make docker-build-arm64

# Build for Intel / AMD x86-64
make docker-build-amd64

# Run — mounts your local data/ reports/ config.yaml as volumes
make docker-run
```

**Connecting to Ollama from Docker**

Your Ollama process runs on the host, not inside the container. Update `config.yaml`:

```yaml
llm:
  base_url: "http://host.docker.internal:11434"
```

On Linux the container automatically adds `--add-host=host.docker.internal:host-gateway`.  
On macOS with Docker Desktop, `host.docker.internal` resolves automatically.

**Volumes mounted by `make docker-run`**

| Container path | Host path | Purpose |
|---|---|---|
| `/app/data` | `./data` | SQLite portfolio + news cache (persistent) |
| `/app/reports` | `./reports` | Saved Markdown / PDF reports (persistent) |
| `/app/config.yaml` | `./config.yaml` | Watchlist + settings (read-only) |
""")

    # ── Glossary ─────────────────────────────────────────────────────────────
    with st.expander("📖 Glossary", expanded=True):
        st.markdown("""
### Technical Analysis

**RSI — Relative Strength Index**
A momentum oscillator scaled 0–100 that measures how fast prices are changing.
- **< 30** — Oversold: price may have fallen too sharply, potential buying opportunity
- **30–45** — Weak / recovering momentum
- **45–55** — Neutral territory
- **55–70** — Strong upward momentum
- **> 70** — Overbought: price may have risen too sharply, potential selling pressure

*Stock-EZ uses Wilder's 14-period exponential smoothing — the original RSI definition.*

---

**Moving Average (MA50, MA200)**
The average closing price over the last N trading days. Smooths noise and reveals trends.
- **Price > MA200** → long-term uptrend
- **Price < MA200** → long-term downtrend
- **MA50 crossing above MA200** → "Golden Cross" (bullish)
- **MA50 crossing below MA200** → "Death Cross" (bearish)

---

**52-Week High / Low**
The highest and lowest traded prices over the trailing 52 weeks. Shows where the current price sits in its annual range.

---

**Gold/Silver Ratio**
Gold spot price ÷ silver spot price. Measures relative value between the two metals.
- **> 85** — Silver is historically cheap vs gold; may favour adding silver
- **65–85** — Normal historical range; neutral signal
- **< 65** — Gold is historically cheap vs silver; may favour adding gold

---

**P/E Ratio — Price-to-Earnings**
Stock price ÷ earnings per share. How much investors pay for each rupee of profit.
- High P/E → expensive, or strong growth expected
- Low P/E → cheap, or slow growth / concerns
- Compare within the same sector for a meaningful signal

---

**Market Cap — Market Capitalisation**
Total market value of a company (Price × Outstanding Shares). Shown in **₹ Crore** (1 Crore = 10 million).
- **Large-cap** > ₹20,000 Cr
- **Mid-cap** ₹5,000–20,000 Cr
- **Small-cap** < ₹5,000 Cr

---

**1D / 5D / 1M %**
Price returns over 1 trading day, 5 trading days, and 1 calendar month. Green = positive, red = negative.

---

**Unrealised P&L**
Paper profit/loss on positions you still hold:
`(Current Price − Buy Price) × Quantity`
Also shown as a % of your cost basis. "Unrealised" means you have not yet sold.

---

**Cost Basis**
Total amount originally invested: `Buy Price × Quantity`

---

### Asset Classes

**ETF — Exchange Traded Fund**
A fund that tracks an index, sector, or commodity and trades on a stock exchange like a regular share. Bought and sold in real time at market prices (unlike mutual funds which settle at end-of-day NAV).

**NAV — Net Asset Value**
Per-unit value of a mutual fund, published once daily after market close:
`NAV = (Total Assets − Liabilities) ÷ Outstanding Units`
Mutual fund orders execute at the next published NAV, not a live price.

**COMEX**
The commodity exchange division of CME Group (Chicago), where gold (`GC=F`) and silver (`SI=F`) futures are the global pricing benchmark. Stock-EZ pulls these via Yahoo Finance, converts USD → INR via the live exchange rate (`USDINR=X`), then converts troy ounces to grams.

**Troy Ounce**
Standard weight unit for precious metals: **1 troy oz = 31.1035 grams**
Stock-EZ stores and displays gold/silver quantities in grams for everyday convenience.

---

### Exchanges & Indices

**NSE — National Stock Exchange of India**
India's largest equity exchange by volume. Symbols end in `.NS` (e.g. `RELIANCE.NS`).

**BSE — Bombay Stock Exchange**
India's oldest exchange. Symbols end in `.BO` (e.g. `RELIANCE.BO`). Stock-EZ auto-falls back to `.BO` if a `.NS` symbol returns no data.

**NIFTY 50**
NSE's benchmark index tracking the 50 largest and most liquid Indian stocks.

**SENSEX**
BSE's benchmark index tracking the 30 largest Indian companies.

**AMFI — Association of Mutual Funds in India**
The industry body that publishes daily NAVs for all registered mutual funds. Stock-EZ fetches NAV data from the free **mfapi.in** public API using AMFI scheme codes.

---

### LLM / AI Settings

**Ollama**
Free open-source tool to run LLMs locally. No data ever leaves your machine. [ollama.com](https://ollama.com)

**Temperature**
Controls output randomness.
- **0.0** — Highly predictable (recommended for analysis)
- **0.5** — Balanced
- **1.0** — Creative but prone to hallucination

**Context window (num_ctx)**
Maximum tokens the LLM processes at once. One token ≈ 0.75 English words.
- **4,096** — Fast, minimal; small watchlists only
- **8,192** — Default — handles most run configurations
- **16,384** — Larger watchlists or week-long news windows
- **32,768** — Maximum — slow on CPU, best with a GPU

**Tokens**
The unit LLMs count text in. Roughly 1 token ≈ 4 characters or 0.75 words. Your prompt + AI response must together fit inside the context window.

---

### News Settings

**Lookback window**
How far back in time to collect news (hours). Longer windows give richer context but increase token usage and analysis time.

**Max articles**
Hard cap on articles included in the AI prompt. Stock-EZ auto-suggests a sensible value based on your lookback window; use the ↺ reset button to restore the suggestion.

**News cache**
Articles are stored in `data/news_cache.db` (SQLite) to avoid re-downloading duplicates. The sidebar shows cache stats and provides a clear button.
""")

    # ── FAQ ───────────────────────────────────────────────────────────────────
    with st.expander("🙋 FAQ", expanded=True):
        st.markdown("""
**Q: Do I need an internet connection?**
Yes — for live prices (Yahoo Finance), NAVs (mfapi.in), and news (RSS feeds).
The LLM analysis itself runs fully offline via Ollama.

---

**Q: My stock symbol shows no data. What should I try?**
1. Confirm the symbol ends in `.NS` (NSE) or `.BO` (BSE)
2. Verify it exists on Yahoo Finance: search `TICKER.NS` on finance.yahoo.com
3. The app automatically retries `.BO` when `.NS` is empty, but you can also add the `.BO` variant directly to the watchlist

---

**Q: My mutual fund is not found. How do I find the right code?**
Use the **🔍 Find fund scheme code** expander in the sidebar — it queries AMFI via mfapi.in.
You can also visit [mfapi.in](https://mfapi.in) and browse or search there.

---

**Q: The AI analysis looks cut off or incomplete. Why?**
The response hit the context window limit. Try:
- Reducing **Max articles** in the sidebar
- Raising the **Context window** slider (16,384 or 32,768)
- Using a model with a larger native context (e.g. `mistral-small3.2:24b`)

---

**Q: How often are gold/silver signals refreshed?**
Market Pulse signals are cached for **1 hour**. Restart the app or wait for the TTL to expire to force a fresh fetch.

---

**Q: Where is my portfolio data stored?**
In `data/portfolio.db` (SQLite). In Docker, this is preserved via the `./data:/app/data` volume mount. Back up this file to keep your portfolio history.

---

**Q: Can I add my own news sources?**
Yes — edit the `news.sources` list in `config.yaml`. Any RSS feed URL is supported.

---

**Q: Is this financial advice?**
No. Stock-EZ is an educational data aggregation and AI analysis tool.
Always consult a SEBI-registered Investment Adviser (RIA) before making investment decisions.
Past performance is not indicative of future results.
""")

    st.caption(
        "Stock-EZ · for educational purposes only · "
        "not SEBI-registered investment advice"
    )
