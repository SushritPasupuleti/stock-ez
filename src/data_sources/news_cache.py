"""
SQLite-backed news article cache.

Avoids re-fetching overlapping time windows across runs.
Cache file: data/news_cache.db (auto-created, gitignored).
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

from src.data_sources.news import NewsArticle

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    url       TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    summary   TEXT,
    source    TEXT,
    published TEXT,
    fetched   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_published ON articles(published);

CREATE TABLE IF NOT EXISTS source_meta (
    source_name   TEXT PRIMARY KEY,
    last_fetched  TEXT NOT NULL
);
"""


class NewsCache:
    """
    Persist news articles in a local SQLite database.

    Usage pattern
    -------------
    cache = NewsCache()
    for source in sources:
        if cache.should_fetch(source.name, ttl_minutes=30):
            articles = fetcher.fetch_source(source)
            cache.store(articles, source.name)
    news = cache.get(lookback_hours=48)
    """

    DEFAULT_PATH = Path("data/news_cache.db")

    def __init__(self, db_path: str | Path = DEFAULT_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, articles: list[NewsArticle], source_name: str) -> int:
        """Upsert articles and mark the source as just fetched. Returns insert count."""
        now = self._now()
        inserted = 0
        with self._conn() as conn:
            for a in articles:
                # Use URL as primary key; fall back to title for feeds without links
                key = (a.url or "").strip() or a.title
                pub = a.published.isoformat() if a.published else None
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO articles
                            (url, title, summary, source, published, fetched)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (key, a.title, a.summary, a.source, pub, now),
                    )
                    inserted += 1
                except sqlite3.Error as exc:
                    logger.debug("Cache insert error for '%s': %s", a.title, exc)

            conn.execute(
                """
                INSERT OR REPLACE INTO source_meta (source_name, last_fetched)
                VALUES (?, ?)
                """,
                (source_name, now),
            )
        logger.debug("NewsCache: stored %d articles for '%s'", inserted, source_name)
        return inserted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        lookback_hours: int,
        max_articles: int = 100,
    ) -> list[NewsArticle]:
        """Return cached articles published within the lookback window, newest first."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT title, summary, source, published, url
                FROM articles
                WHERE published >= ?
                ORDER BY published DESC
                LIMIT ?
                """,
                (cutoff, max_articles),
            ).fetchall()

        articles: list[NewsArticle] = []
        for title, summary, source, published, url in rows:
            pub: Optional[datetime] = None
            if published:
                try:
                    pub = datetime.fromisoformat(published)
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            articles.append(
                NewsArticle(
                    title=title,
                    summary=summary or "",
                    source=source or "",
                    published=pub,
                    url=url or "",
                )
            )
        return articles

    def should_fetch(self, source_name: str, ttl_minutes: int = 30) -> bool:
        """Return True if the source hasn't been fetched within the TTL window."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_fetched FROM source_meta WHERE source_name = ?",
                (source_name,),
            ).fetchone()
        if not row:
            return True
        last = datetime.fromisoformat(row[0])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return age_min >= ttl_minutes

    # ------------------------------------------------------------------
    # Stats / maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return basic stats for display in the UI."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(published) FROM articles"
            ).fetchone()[0]
            sources = conn.execute(
                "SELECT source_name, last_fetched FROM source_meta"
            ).fetchall()
        age_hours: Optional[float] = None
        if oldest:
            try:
                dt = datetime.fromisoformat(oldest)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            except ValueError:
                pass
        return {
            "total_articles": total,
            "oldest_age_hours": age_hours,
            "sources": {name: last for name, last in sources},
        }

    def clear(self) -> None:
        """Delete all cached articles and source metadata."""
        with self._conn() as conn:
            conn.execute("DELETE FROM articles")
            conn.execute("DELETE FROM source_meta")
        logger.info("NewsCache cleared")
