from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class LLMConfig:
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    num_ctx: int = 8192


@dataclass
class RegionConfig:
    country: str = "IN"
    market: str = "NSE"
    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    name: str = "India"


@dataclass
class StockEntry:
    symbol: str
    name: str


@dataclass
class FundEntry:
    scheme_code: str
    name: str


@dataclass
class MarketIndex:
    symbol: str
    name: str


@dataclass
class WatchlistConfig:
    stocks: List[StockEntry] = field(default_factory=list)
    funds: List[FundEntry] = field(default_factory=list)


@dataclass
class NewsSource:
    name: str
    url: str
    type: str = "rss"


@dataclass
class NewsConfig:
    max_articles: int = 20
    lookback_hours: int = 24
    sources: List[NewsSource] = field(default_factory=list)


@dataclass
class OutputConfig:
    format: str = "rich"
    save_report: bool = True
    report_dir: str = "reports"


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    watchlist: WatchlistConfig = field(default_factory=WatchlistConfig)
    market_indices: List[MarketIndex] = field(default_factory=list)
    news: NewsConfig = field(default_factory=NewsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Settings":
        config_file = Path(config_path)

        # Resolve relative to CWD first, then relative to project root
        if not config_file.is_absolute():
            candidates = [
                Path.cwd() / config_path,
                Path(__file__).parent.parent.parent / config_path,
            ]
            for c in candidates:
                if c.exists():
                    config_file = c
                    break

        if not config_file.exists():
            return cls()

        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Settings":
        llm_d = data.get("llm", {})
        llm = LLMConfig(
            model=llm_d.get("model", "qwen2.5:14b"),
            base_url=llm_d.get("base_url", "http://localhost:11434"),
            temperature=float(llm_d.get("temperature", 0.3)),
            num_ctx=int(llm_d.get("num_ctx", 8192)),
        )

        reg_d = data.get("region", {})
        region = RegionConfig(
            country=reg_d.get("country", "IN"),
            market=reg_d.get("market", "NSE"),
            currency=reg_d.get("currency", "INR"),
            timezone=reg_d.get("timezone", "Asia/Kolkata"),
            name=reg_d.get("name", "India"),
        )

        wl_d = data.get("watchlist", {})
        stocks = [
            StockEntry(symbol=s["symbol"], name=s.get("name", s["symbol"]))
            for s in wl_d.get("stocks", [])
        ]
        funds = [
            FundEntry(scheme_code=str(f["scheme_code"]), name=f.get("name", str(f["scheme_code"])))
            for f in wl_d.get("funds", [])
        ]
        watchlist = WatchlistConfig(stocks=stocks, funds=funds)

        indices = [
            MarketIndex(symbol=i["symbol"], name=i.get("name", i["symbol"]))
            for i in data.get("market_indices", [])
        ]

        news_d = data.get("news", {})
        news_sources = [
            NewsSource(name=s["name"], url=s["url"], type=s.get("type", "rss"))
            for s in news_d.get("sources", [])
        ]
        news = NewsConfig(
            max_articles=int(news_d.get("max_articles", 20)),
            lookback_hours=int(news_d.get("lookback_hours", 24)),
            sources=news_sources,
        )

        out_d = data.get("output", {})
        output = OutputConfig(
            format=out_d.get("format", "rich"),
            save_report=bool(out_d.get("save_report", True)),
            report_dir=out_d.get("report_dir", "reports"),
        )

        return cls(
            llm=llm,
            region=region,
            watchlist=watchlist,
            market_indices=indices,
            news=news,
            output=output,
        )
