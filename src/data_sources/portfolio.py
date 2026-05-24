"""
Portfolio tracker — SQLite-backed store for individual holdings.

Each ``Position`` records a stock or mutual-fund holding with its cost basis.
After a market-data fetch, ``PortfolioStore.enrich()`` cross-references live
prices and computes unrealised P&L.  The results can be fed to the LLM via
``format_for_prompt()`` so the model can tailor recommendations to what the
user actually holds.

DB file: ``data/portfolio.db`` (auto-created alongside the news cache).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single portfolio holding — stock or mutual fund."""

    symbol: str           # "HDFCBANK.NS" for stocks; scheme code e.g. "119551" for funds
    name: str
    asset_type: str       # "stock" | "fund"
    quantity: float       # shares / units
    buy_price: float      # price per share/unit (INR) at time of purchase
    buy_date: Optional[str] = None   # ISO date "YYYY-MM-DD", or None
    notes: Optional[str] = None
    id: Optional[int] = None         # DB rowid; None for unsaved records


@dataclass
class EnrichedPosition:
    """Position enriched with live market data after a fetch run."""

    position: Position
    current_price: Optional[float]   # latest price from market data; None if unavailable
    current_value: Optional[float]   # quantity × current_price
    cost_basis: float                # quantity × buy_price
    pnl_abs: Optional[float]         # current_value − cost_basis
    pnl_pct: Optional[float]         # pnl_abs / cost_basis × 100


@dataclass
class MetalSignal:
    """
    Technical buy/sell timing signal for a precious metal.

    Derived from 1-year COMEX futures history.
    ``signal`` is one of ``"BUY"`` | ``"HOLD"`` | ``"SELL"`` | ``"N/A"``.
    """

    metal: str                        # "Gold" | "Silver"
    price_inr: Optional[float]        # ₹ per gram (COMEX spot × USD/INR / 31.1035)
    change_52w_pct: Optional[float]   # % price change over last 52 weeks
    rsi_14: Optional[float]           # RSI(14) computed on daily COMEX closes
    vs_ma50_pct: Optional[float]      # % deviation from 50-day moving average
    vs_ma200_pct: Optional[float]     # % deviation from 200-day moving average
    signal: str                       # "BUY" | "HOLD" | "SELL" | "N/A"
    reasons: list = field(default_factory=list)   # human-readable factors


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioStore:
    """
    SQLite-backed store for portfolio positions.

    The ``data/`` directory is shared with the news cache so a single
    ``.gitignore`` entry covers both databases.
    """

    DEFAULT_PATH = Path("data/portfolio.db")

    # Troy ounce → gram conversion (COMEX quotes in troy oz)
    _TROY_OZ_TO_GRAMS: float = 31.1035

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    name        TEXT    NOT NULL DEFAULT '',
                    asset_type  TEXT    NOT NULL DEFAULT 'stock',
                    quantity    REAL    NOT NULL CHECK(quantity > 0),
                    buy_price   REAL    NOT NULL CHECK(buy_price >= 0),
                    buy_date    TEXT,
                    notes       TEXT,
                    updated_at  TEXT    DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol)"
            )
            # Upgrade existing DBs that have the old restrictive CHECK constraint
            self._migrate_asset_type_constraint(conn)

    def _migrate_asset_type_constraint(self, conn: sqlite3.Connection) -> None:
        """
        Probe whether the existing table accepts the new asset_type values
        (etf / gold / silver).  If the old CHECK constraint blocks the insert,
        recreate the table without that constraint, preserving all existing data.
        """
        try:
            conn.execute(
                "INSERT INTO positions (symbol, name, asset_type, quantity, buy_price)"
                " VALUES ('__probe__', 'probe', 'etf', 1.0, 1.0)"
            )
            conn.execute("DELETE FROM positions WHERE symbol='__probe__'")
        except sqlite3.IntegrityError:
            logger.info(
                "Upgrading portfolio.db: expanding asset_type to include etf/gold/silver"
            )
            conn.execute(
                """
                CREATE TABLE positions_v2 (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    name        TEXT    NOT NULL DEFAULT '',
                    asset_type  TEXT    NOT NULL DEFAULT 'stock',
                    quantity    REAL    NOT NULL CHECK(quantity > 0),
                    buy_price   REAL    NOT NULL CHECK(buy_price >= 0),
                    buy_date    TEXT,
                    notes       TEXT,
                    updated_at  TEXT    DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute("INSERT INTO positions_v2 SELECT * FROM positions")
            conn.execute("DROP TABLE positions")
            conn.execute("ALTER TABLE positions_v2 RENAME TO positions")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol)"
            )

    # ── CRUD ──────────────────────────────────────────────────────────────

    def list_all(self) -> List[Position]:
        """Return all positions ordered by type then symbol."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions ORDER BY asset_type, symbol"
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def upsert_bulk(self, positions: List[Position]) -> None:
        """
        Reconcile the DB with *positions*:

        - Rows **with** an ``id`` are updated in place.
        - Rows **without** an ``id`` are inserted as new records.
        - Rows in the DB whose ``id`` is absent from *positions* are deleted
          (the user removed them in the editor).
        """
        keep_ids: List[int] = []
        with self._connect() as conn:
            for pos in positions:
                if pos.id is not None:
                    conn.execute(
                        """
                        UPDATE positions
                        SET symbol=?, name=?, asset_type=?, quantity=?,
                            buy_price=?, buy_date=?, notes=?,
                            updated_at=datetime('now')
                        WHERE id=?
                        """,
                        (
                            pos.symbol, pos.name, pos.asset_type,
                            pos.quantity, pos.buy_price,
                            pos.buy_date, pos.notes,
                            pos.id,
                        ),
                    )
                    keep_ids.append(pos.id)
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO positions
                            (symbol, name, asset_type, quantity,
                             buy_price, buy_date, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pos.symbol, pos.name, pos.asset_type,
                            pos.quantity, pos.buy_price,
                            pos.buy_date, pos.notes,
                        ),
                    )
                    keep_ids.append(cur.lastrowid)  # type: ignore[arg-type]

            # Purge rows that were removed in the editor
            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                conn.execute(
                    f"DELETE FROM positions WHERE id NOT IN ({placeholders})",
                    keep_ids,
                )
            else:
                conn.execute("DELETE FROM positions")

    # ── Enrichment ────────────────────────────────────────────────────────

    def enrich(
        self,
        stocks_data: list,   # list[StockData]
        funds_data: list,    # list[FundData]
    ) -> List[EnrichedPosition]:
        """
        Match each position against live market data and compute P&L.

        Asset-type routing:
        - ``stock`` / ``etf`` — looked up in the yfinance price map by .NS symbol.
        - ``fund``            — looked up in the mfapi NAV map by scheme code.
        - ``gold``            — real-time COMEX gold spot (GC=F) converted to
                                INR per gram via live USD/INR rate.
        - ``silver``          — same using COMEX silver spot (SI=F).
          Quantity should be in **grams**; buy_price in **₹/gram**.

        Positions with no matching price receive ``None`` for value/P&L fields
        and are silently skipped by the LLM.
        """
        stock_price_map: dict[str, float] = {
            s.symbol: s.current_price
            for s in stocks_data
            if s.current_price is not None
        }
        fund_nav_map: dict[str, float] = {
            str(f.scheme_code): f.nav
            for f in funds_data
            if f.nav is not None
        }

        # Fetch real commodity spot prices only when the portfolio has metal positions
        positions = self.list_all()
        _has_metals = any(p.asset_type in ("gold", "silver") for p in positions)
        gold_per_gram: Optional[float] = None
        silver_per_gram: Optional[float] = None
        if _has_metals:
            gold_per_gram, silver_per_gram = self._fetch_metal_prices()

        enriched: List[EnrichedPosition] = []
        for pos in positions:
            cost_basis = round(pos.quantity * pos.buy_price, 2)

            if pos.asset_type in ("stock", "etf"):
                current_price = stock_price_map.get(pos.symbol)
            elif pos.asset_type == "fund":
                current_price = fund_nav_map.get(pos.symbol)
            elif pos.asset_type == "gold":
                current_price = gold_per_gram
            elif pos.asset_type == "silver":
                current_price = silver_per_gram
            else:
                current_price = stock_price_map.get(pos.symbol)

            if current_price is not None:
                current_value = round(pos.quantity * current_price, 2)
                pnl_abs = round(current_value - cost_basis, 2)
                pnl_pct = (
                    round(pnl_abs / cost_basis * 100, 2) if cost_basis else None
                )
            else:
                current_value = pnl_abs = pnl_pct = None

            enriched.append(
                EnrichedPosition(
                    position=pos,
                    current_price=current_price,
                    current_value=current_value,
                    cost_basis=cost_basis,
                    pnl_abs=pnl_abs,
                    pnl_pct=pnl_pct,
                )
            )

        return enriched

    # ── Commodity spot prices ────────────────────────────────────

    @classmethod
    def _fetch_metal_prices(cls) -> tuple[Optional[float], Optional[float]]:
        """
        Fetch live gold and silver spot prices in INR per gram.

        Methodology:
          1. Fetch COMEX front-month futures: GC=F (gold, USD/troy oz)
             and SI=F (silver, USD/troy oz).
          2. Fetch USD/INR spot rate: USDINR=X.
          3. Convert:  INR/gram = (USD/troy oz) × (INR/USD) / 31.1035

        Returns ``(gold_inr_per_gram, silver_inr_per_gram)``;
        either value is ``None`` when the fetch fails.
        """
        try:
            import yfinance as yf

            def _last_close(sym: str) -> Optional[float]:
                hist = yf.Ticker(sym).history(period="2d", auto_adjust=True)
                if hist.empty:
                    return None
                return float(hist["Close"].iloc[-1])

            gold_usd   = _last_close("GC=F")
            silver_usd = _last_close("SI=F")
            usd_inr    = _last_close("USDINR=X")

            if not usd_inr:
                logger.warning("Could not fetch USD/INR rate; metal prices unavailable")
                return None, None

            gold_per_gram = (
                round(gold_usd * usd_inr / cls._TROY_OZ_TO_GRAMS, 2)
                if gold_usd else None
            )
            silver_per_gram = (
                round(silver_usd * usd_inr / cls._TROY_OZ_TO_GRAMS, 2)
                if silver_usd else None
            )
            return gold_per_gram, silver_per_gram

        except Exception as exc:
            logger.warning("Could not fetch metal spot prices: %s", exc)
            return None, None

    @classmethod
    def fetch_metal_signals(
        cls,
    ) -> tuple[MetalSignal, MetalSignal, Optional[float]]:
        """
        Analyse 1-year COMEX futures history to produce general market-timing
        signals for physical gold and silver, independent of portfolio holdings.

        Scoring model (additive, range −4 to +4):
          RSI < 30              → +2  (oversold)
          RSI 30–45             → +1  (leaning cheap)
          RSI 55–70             → −1  (elevated momentum)
          RSI > 70              → −2  (overbought)
          Price > 5% below MA50 → +1  (short-term pullback)
          Price > 10% above MA50→ −1  (short-term stretched)
          Price >10% below MA200→ +1  (long-term dip)
          Price >25% above MA200→ −1  (long-term extended)
          52w change < −10%     → +1  (significant drawdown)

        Thresholds: score ≥ 2 → BUY · score ≤ −2 → SELL · else HOLD

        Also returns the gold/silver ratio (troy-oz terms, not INR).  A ratio
        above 85 historically suggests silver is cheap relative to gold; below
        65 suggests the opposite.

        Returns ``(gold_signal, silver_signal, gold_silver_ratio)``.
        """
        def _fetch_closes(sym: str) -> Optional[object]:
            try:
                import yfinance as yf
                hist = yf.Ticker(sym).history(period="1y", auto_adjust=True)
                if hist.empty:
                    return None
                return hist["Close"].dropna()
            except Exception:
                return None

        def _build_signal(metal: str, closes_usd: object, usd_inr: float) -> MetalSignal:
            _na = MetalSignal(
                metal=metal, price_inr=None, change_52w_pct=None,
                rsi_14=None, vs_ma50_pct=None, vs_ma200_pct=None,
                signal="N/A", reasons=["Price data unavailable"],
            )
            if closes_usd is None or len(closes_usd) < 30:  # type: ignore[arg-type]
                return _na
            try:
                price_usd = float(closes_usd.iloc[-1])  # type: ignore[union-attr]
                price_inr = round(price_usd * usd_inr / cls._TROY_OZ_TO_GRAMS, 2)

                change_52w = round(
                    (price_usd - float(closes_usd.iloc[0])) / float(closes_usd.iloc[0]) * 100, 1  # type: ignore[union-attr]
                )

                n = len(closes_usd)  # type: ignore[arg-type]
                ma50 = (
                    float(closes_usd.rolling(50).mean().iloc[-1])   # type: ignore[union-attr]
                    if n >= 50 else None
                )
                ma200 = (
                    float(closes_usd.rolling(200).mean().iloc[-1])  # type: ignore[union-attr]
                    if n >= 200 else None
                )
                vs_ma50  = round((price_usd - ma50)  / ma50  * 100, 1) if ma50  else None
                vs_ma200 = round((price_usd - ma200) / ma200 * 100, 1) if ma200 else None

                # RSI(14) — Wilder's exponential smoothing
                delta = closes_usd.diff()  # type: ignore[union-attr]
                gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                rs    = gain / loss
                rsi   = round(float((100 - 100 / (1 + rs)).iloc[-1]), 1)

                score: int = 0
                reasons: list[str] = []

                if rsi < 30:
                    score += 2
                    reasons.append(f"RSI {rsi} — oversold (historically favourable)")
                elif rsi < 45:
                    score += 1
                    reasons.append(f"RSI {rsi} — approaching oversold zone")
                elif rsi > 70:
                    score -= 2
                    reasons.append(f"RSI {rsi} — overbought, elevated pullback risk")
                elif rsi > 55:
                    score -= 1
                    reasons.append(f"RSI {rsi} — elevated momentum, caution on new entries")
                else:
                    reasons.append(f"RSI {rsi} — neutral momentum")

                if vs_ma50 is not None:
                    if vs_ma50 < -5:
                        score += 1
                        reasons.append(f"{abs(vs_ma50):.1f}% below 50-day MA — short-term pullback")
                    elif vs_ma50 > 10:
                        score -= 1
                        reasons.append(f"{vs_ma50:.1f}% above 50-day MA — stretched short-term")
                    else:
                        reasons.append(f"{vs_ma50:+.1f}% vs 50-day MA")

                if vs_ma200 is not None:
                    if vs_ma200 < -10:
                        score += 1
                        reasons.append(f"{abs(vs_ma200):.1f}% below 200-day MA — long-term dip")
                    elif vs_ma200 > 25:
                        score -= 1
                        reasons.append(f"{vs_ma200:.1f}% above 200-day MA — long-term extended")
                    else:
                        reasons.append(f"{vs_ma200:+.1f}% vs 200-day MA")

                if change_52w < -10:
                    score += 1
                    reasons.append(f"52w: {change_52w:+.1f}% — significant drawdown, potential value")
                elif change_52w > 20:
                    reasons.append(f"52w: +{change_52w:.1f}% — strong multi-month run")
                else:
                    reasons.append(f"52w change: {change_52w:+.1f}%")

                signal = "BUY" if score >= 2 else ("SELL" if score <= -2 else "HOLD")

                return MetalSignal(
                    metal=metal, price_inr=price_inr, change_52w_pct=change_52w,
                    rsi_14=rsi, vs_ma50_pct=vs_ma50, vs_ma200_pct=vs_ma200,
                    signal=signal, reasons=reasons,
                )
            except Exception as exc:
                logger.warning("Could not compute %s signal: %s", metal, exc)
                return MetalSignal(
                    metal=metal, price_inr=None, change_52w_pct=None,
                    rsi_14=None, vs_ma50_pct=None, vs_ma200_pct=None,
                    signal="N/A", reasons=[f"Error: {exc}"],
                )

        try:
            usd_inr_closes = _fetch_closes("USDINR=X")
            if usd_inr_closes is None:
                _na = lambda m: MetalSignal(  # noqa: E731
                    metal=m, price_inr=None, change_52w_pct=None,
                    rsi_14=None, vs_ma50_pct=None, vs_ma200_pct=None,
                    signal="N/A", reasons=["USD/INR rate unavailable"],
                )
                return _na("Gold"), _na("Silver"), None

            usd_inr     = float(usd_inr_closes.iloc[-1])  # type: ignore[union-attr]
            gold_closes = _fetch_closes("GC=F")
            silver_closes = _fetch_closes("SI=F")

            gold_sig   = _build_signal("Gold",   gold_closes,   usd_inr)
            silver_sig = _build_signal("Silver", silver_closes, usd_inr)

            gs_ratio: Optional[float] = None
            if gold_closes is not None and silver_closes is not None:
                _g = float(gold_closes.iloc[-1])   # type: ignore[union-attr]
                _s = float(silver_closes.iloc[-1])  # type: ignore[union-attr]
                if _s > 0:
                    gs_ratio = round(_g / _s, 1)

            return gold_sig, silver_sig, gs_ratio

        except Exception as exc:
            logger.warning("fetch_metal_signals failed: %s", exc)
            _err = lambda m: MetalSignal(  # noqa: E731
                metal=m, price_inr=None, change_52w_pct=None,
                rsi_14=None, vs_ma50_pct=None, vs_ma200_pct=None,
                signal="N/A", reasons=[str(exc)],
            )
            return _err("Gold"), _err("Silver"), None

    # ── Prompt text ───────────────────────────────────────────────────────

    def format_for_prompt(self, enriched: List[EnrichedPosition]) -> str:
        """Plain-text portfolio table for inclusion in the LLM prompt."""
        if not enriched:
            return "No portfolio positions recorded — skip the Portfolio Review section."

        col = "{:<20} {:<28} {:<6} {:>8} {:>12} {:>12} {:>8} {:>13}"
        header = col.format(
            "Symbol/Code", "Name", "Type", "Qty",
            "Buy@", "Now@", "P&L%", "P&L(Rs.)",
        )
        sep = "-" * 113
        lines = [header, sep]

        total_cost = 0.0
        total_current = 0.0
        priced_count = 0

        for ep in enriched:
            p = ep.position
            now_str = (
                f"Rs.{ep.current_price:,.2f}" if ep.current_price is not None else "N/A"
            )
            pnl_pct_str = (
                f"{ep.pnl_pct:+.2f}%" if ep.pnl_pct is not None else "N/A"
            )
            pnl_abs_str = (
                f"Rs.{ep.pnl_abs:+,.0f}" if ep.pnl_abs is not None else "N/A"
            )
            lines.append(
                col.format(
                    p.symbol,
                    p.name[:28],
                    p.asset_type,
                    f"{p.quantity:.2f}",
                    f"Rs.{p.buy_price:,.2f}",
                    now_str,
                    pnl_pct_str,
                    pnl_abs_str,
                )
            )
            total_cost += ep.cost_basis
            if ep.current_value is not None:
                total_current += ep.current_value
                priced_count += 1

        lines.append(sep)
        if priced_count > 0:
            total_pnl = total_current - total_cost
            total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0.0
            lines.append(
                f"Total invested: Rs.{total_cost:,.0f}  |  "
                f"Current value: Rs.{total_current:,.0f}  |  "
                f"Unrealised P&L: Rs.{total_pnl:+,.0f} ({total_pnl_pct:+.2f}%)"
            )
        else:
            lines.append(
                f"Total invested: Rs.{total_cost:,.0f}  |  "
                "Current value: N/A (no matching live prices)"
            )

        lines.append(
            "\nPositions marked N/A have no matching live price data — "
            "exclude them from any valuation calculations."
        )
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> Position:
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            name=row["name"],
            asset_type=row["asset_type"],
            quantity=row["quantity"],
            buy_price=row["buy_price"],
            buy_date=row["buy_date"],
            notes=row["notes"],
        )
