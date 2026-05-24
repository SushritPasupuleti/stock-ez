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
from dataclasses import dataclass
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
                    asset_type  TEXT    NOT NULL DEFAULT 'stock'
                                        CHECK(asset_type IN ('stock', 'fund')),
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

        Positions whose symbol/scheme-code is not present in the fetched data
        (e.g. the symbol is not on the watchlist, or the fetch failed) receive
        ``None`` for all price/value/P&L fields and are skipped by the LLM
        rather than silently producing wrong numbers.
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

        enriched: List[EnrichedPosition] = []
        for pos in self.list_all():
            cost_basis = round(pos.quantity * pos.buy_price, 2)
            current_price: Optional[float] = (
                stock_price_map.get(pos.symbol)
                if pos.asset_type == "stock"
                else fund_nav_map.get(pos.symbol)
            )
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
