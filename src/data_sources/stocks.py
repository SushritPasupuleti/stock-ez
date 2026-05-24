from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    symbol: str
    name: str
    current_price: Optional[float] = None
    prev_close: Optional[float] = None
    change_1d_pct: Optional[float] = None
    change_5d_pct: Optional[float] = None
    change_1m_pct: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap_cr: Optional[float] = None  # Indian crores (1 Cr = 10M)
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    sector: Optional[str] = None
    currency: str = "INR"


class StockFetcher:
    """
    Fetches stock price and fundamental data via yfinance (Yahoo Finance).
    Free, no API key required. Supports NSE (.NS) and BSE (.BO) suffixes.
    """

    def __init__(self, region_config=None, delay_between_calls: float = 0.5) -> None:
        self.region = region_config
        self.delay = delay_between_calls

    # ------------------------------------------------------------------
    # Single ticker
    # ------------------------------------------------------------------

    def fetch_stock(self, symbol: str, name: str) -> Optional[StockData]:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo", auto_adjust=True)

            if hist.empty:
                logger.warning("No historical data returned for %s", symbol)
                return None

            current_price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None

            change_1d = (
                (current_price - prev_close) / prev_close * 100
                if prev_close
                else None
            )
            change_5d = (
                (current_price - float(hist["Close"].iloc[-6]))
                / float(hist["Close"].iloc[-6])
                * 100
                if len(hist) >= 6
                else None
            )
            change_1m = (
                (current_price - float(hist["Close"].iloc[0]))
                / float(hist["Close"].iloc[0])
                * 100
                if len(hist) >= 21
                else None
            )

            # Fundamental data — optional; yfinance can be flaky for some tickers
            pe = market_cap_cr = w52h = w52l = sector = avg_vol = None
            currency = "INR"
            display_name = name

            try:
                info = ticker.info
                pe = info.get("trailingPE")
                mc = info.get("marketCap")
                market_cap_cr = mc / 1e7 if mc else None  # convert to Indian Crores
                w52h = info.get("fiftyTwoWeekHigh")
                w52l = info.get("fiftyTwoWeekLow")
                sector = info.get("sector") or info.get("quoteType")
                avg_vol = info.get("averageVolume")
                currency = info.get("currency", "INR")
                long_name = info.get("longName") or info.get("shortName")
                if long_name:
                    display_name = long_name
            except Exception as exc:
                logger.debug("Fundamental data unavailable for %s: %s", symbol, exc)

            vol = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None

            return StockData(
                symbol=symbol,
                name=display_name,
                current_price=round(current_price, 2),
                prev_close=round(prev_close, 2) if prev_close else None,
                change_1d_pct=round(change_1d, 2) if change_1d is not None else None,
                change_5d_pct=round(change_5d, 2) if change_5d is not None else None,
                change_1m_pct=round(change_1m, 2) if change_1m is not None else None,
                week_52_high=w52h,
                week_52_low=w52l,
                pe_ratio=round(pe, 1) if pe else None,
                market_cap_cr=round(market_cap_cr, 0) if market_cap_cr else None,
                volume=vol,
                avg_volume=avg_vol,
                sector=sector,
                currency=currency,
            )
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def fetch_all(self, watchlist) -> List[StockData]:
        results: List[StockData] = []
        for stock in watchlist:
            data = self.fetch_stock(stock.symbol, stock.name)
            if data:
                results.append(data)
            time.sleep(self.delay)
        return results

    def fetch_indices(self, indices) -> List[StockData]:
        results: List[StockData] = []
        for idx in indices:
            data = self.fetch_stock(idx.symbol, idx.name)
            if data:
                results.append(data)
            time.sleep(self.delay)
        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, stocks: List[StockData]) -> str:
        if not stocks:
            return "No stock data available."

        lines: List[str] = []
        for s in stocks:
            sym = "₹" if s.currency in ("INR", "INp") else s.currency + " "
            price = f"{sym}{s.current_price:,.2f}" if s.current_price else "N/A"
            d1 = f"{s.change_1d_pct:+.2f}%" if s.change_1d_pct is not None else "N/A"
            d5 = f"{s.change_5d_pct:+.2f}%" if s.change_5d_pct is not None else "N/A"
            d1m = f"{s.change_1m_pct:+.2f}%" if s.change_1m_pct is not None else "N/A"
            pe = f"{s.pe_ratio}" if s.pe_ratio else "N/A"
            mcap = f"₹{s.market_cap_cr:,.0f}Cr" if s.market_cap_cr else "N/A"

            lines.append(f"- {s.symbol}  ({s.name})")
            lines.append(f"  Price: {price}  |  1D: {d1}  |  5D: {d5}  |  1M: {d1m}")
            parts = [f"P/E: {pe}", f"MCap: {mcap}"]
            if s.sector:
                parts.append(f"Sector: {s.sector}")
            lines.append("  " + "  |  ".join(parts))
            if s.week_52_high and s.week_52_low and s.current_price:
                from_high = (s.current_price - s.week_52_high) / s.week_52_high * 100
                lines.append(
                    f"  52W Range: ₹{s.week_52_low:,.2f} – ₹{s.week_52_high:,.2f}"
                    f"  ({from_high:+.1f}% from 52W high)"
                )

        return "\n".join(lines)

    def format_indices(self, indices: List[StockData]) -> str:
        if not indices:
            return "No market index data available."

        lines: List[str] = []
        for idx in indices:
            price = f"{idx.current_price:,.2f}" if idx.current_price else "N/A"
            d1 = f"{idx.change_1d_pct:+.2f}%" if idx.change_1d_pct is not None else "N/A"
            d5 = f"{idx.change_5d_pct:+.2f}%" if idx.change_5d_pct is not None else "N/A"
            d1m = f"{idx.change_1m_pct:+.2f}%" if idx.change_1m_pct is not None else "N/A"
            lines.append(
                f"- {idx.name}: {price}  |  1D: {d1}  |  5D: {d5}  |  1M: {d1m}"
            )
        return "\n".join(lines)
