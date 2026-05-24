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
    _LOOKBACK_PRESETS = {
        "Last 24h": 24,
        "Last 48h": 48,
        "Last 3 Days": 72,
        "Last Week": 168,
        "Last 2 Weeks": 336,
        "Last Month": 720,
        "Custom": 0,
    }
    _preset_label = st.selectbox(
        "Quick select",
        list(_LOOKBACK_PRESETS.keys()),
        index=0,
        key="lookback_preset",
    )
    _preset_hours = _LOOKBACK_PRESETS[_preset_label]
    if _preset_hours == 0:
        lookback = int(
            st.number_input(
                "Custom hours",
                min_value=1,
                max_value=8760,
                value=max(1, int(_base.news.lookback_hours)),
                step=1,
            )
        )
    else:
        lookback = _preset_hours
        st.caption(f"{_preset_hours}h window · approx {_preset_hours // 24}d" if _preset_hours >= 24 else f"{_preset_hours}h window")

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
tab_overview, tab_stocks, tab_funds, tab_news, tab_raw, tab_reports = st.tabs(
    ["📊 Overview", "📈 Stocks", "💼 Funds", "📰 News", "🗂️ Raw Data", "🗃️ Reports"]
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
            "3. Click **▶ Run Analysis** in the sidebar"
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
