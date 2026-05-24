from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_MFAPI_BASE = "https://api.mfapi.in/mf"
_SEARCH_URL = f"{_MFAPI_BASE}/search"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockEZ/1.0)"
}


@dataclass
class FundData:
    scheme_code: str
    name: str
    nav: Optional[float] = None
    nav_date: Optional[str] = None
    returns_1m: Optional[float] = None
    returns_3m: Optional[float] = None
    returns_6m: Optional[float] = None
    returns_1y: Optional[float] = None
    fund_type: Optional[str] = None
    fund_category: Optional[str] = None


class FundFetcher:
    """
    Fetches Indian mutual fund NAV and returns from mfapi.in — completely free,
    no API key required. Data sourced from AMFI India.

    To find a fund's scheme code:
        GET https://api.mfapi.in/mf/search?q=FUND_NAME
    """

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> Optional[dict]:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None

    def _calc_return(
        self, current_nav: float, historical_nav: Optional[float]
    ) -> Optional[float]:
        if historical_nav and historical_nav > 0:
            return round((current_nav - historical_nav) / historical_nav * 100, 2)
        return None

    def _nav_n_days_ago(self, nav_series: list, days: int) -> Optional[float]:
        """Return the NAV value closest to `days` ago (but not after)."""
        target = datetime.now() - timedelta(days=days)
        for entry in nav_series:
            try:
                entry_date = datetime.strptime(entry["date"], "%d-%m-%Y")
                if entry_date <= target:
                    return float(entry["nav"])
            except (ValueError, KeyError):
                continue
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_fund(self, scheme_code: str, fallback_name: str = "") -> Optional[FundData]:
        url = f"{_MFAPI_BASE}/{scheme_code}"
        data = self._get_json(url)
        if not data or data.get("status") != "SUCCESS":
            logger.warning("No data for scheme %s", scheme_code)
            return None

        meta = data.get("meta", {})
        nav_series: list = data.get("data", [])
        if not nav_series:
            return None

        scheme_name = meta.get("scheme_name") or fallback_name or scheme_code
        scheme_type = meta.get("scheme_type", "")
        scheme_category = meta.get("scheme_category", "")

        latest = nav_series[0]
        current_nav = float(latest["nav"])
        nav_date = latest.get("date", "")

        return FundData(
            scheme_code=scheme_code,
            name=scheme_name,
            nav=round(current_nav, 4),
            nav_date=nav_date,
            returns_1m=self._calc_return(current_nav, self._nav_n_days_ago(nav_series, 30)),
            returns_3m=self._calc_return(current_nav, self._nav_n_days_ago(nav_series, 90)),
            returns_6m=self._calc_return(current_nav, self._nav_n_days_ago(nav_series, 180)),
            returns_1y=self._calc_return(current_nav, self._nav_n_days_ago(nav_series, 365)),
            fund_type=scheme_type,
            fund_category=scheme_category,
        )

    def fetch_all(self, watchlist) -> List[FundData]:
        results: List[FundData] = []
        for fund in watchlist:
            data = self.fetch_fund(fund.scheme_code, fund.name)
            if data:
                results.append(data)
            else:
                logger.warning("Skipping fund '%s' (code: %s)", fund.name, fund.scheme_code)
        return results

    @staticmethod
    def search(query: str, limit: int = 10) -> List[dict]:
        """Search for funds by name — useful for finding scheme codes."""
        try:
            resp = requests.get(
                _SEARCH_URL,
                params={"q": query},
                headers=_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()[:limit]
        except Exception as exc:
            logger.error("Fund search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, funds: List[FundData]) -> str:
        if not funds:
            return "No mutual fund data available."

        lines: List[str] = []
        for f in funds:
            nav_str = f"₹{f.nav:.4f}" if f.nav else "N/A"
            r1m = f"{f.returns_1m:+.2f}%" if f.returns_1m is not None else "N/A"
            r3m = f"{f.returns_3m:+.2f}%" if f.returns_3m is not None else "N/A"
            r6m = f"{f.returns_6m:+.2f}%" if f.returns_6m is not None else "N/A"
            r1y = f"{f.returns_1y:+.2f}%" if f.returns_1y is not None else "N/A"

            lines.append(f"- {f.name}  [Code: {f.scheme_code}]")
            lines.append(
                f"  NAV: {nav_str} (as of {f.nav_date or 'N/A'})  |  "
                f"1M: {r1m}  |  3M: {r3m}  |  6M: {r6m}  |  1Y: {r1y}"
            )
            if f.fund_category:
                lines.append(f"  Category: {f.fund_category}")

        return "\n".join(lines)
