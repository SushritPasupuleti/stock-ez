from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.data_sources.news_cache import NewsCache

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockEZ/1.0; "
        "+https://github.com/stock-ez)"
    )
}


@dataclass
class NewsArticle:
    title: str
    summary: str
    source: str
    published: Optional[datetime]
    url: str


class NewsFetcher:
    """Fetches articles from free RSS feeds — no API key required."""

    def __init__(
        self,
        sources: list,
        max_articles: int = 20,
        lookback_hours: int = 24,
        request_delay: float = 0.4,
        cache: "Optional[NewsCache]" = None,
        cache_ttl_minutes: int = 30,
    ) -> None:
        self.sources = sources
        self.max_articles = max_articles
        self.lookback_hours = lookback_hours
        self.request_delay = request_delay
        self._cache = cache
        self._cache_ttl = cache_ttl_minutes
        self._cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_html(self, raw: str, max_len: int = 400) -> str:
        try:
            return BeautifulSoup(raw, "lxml").get_text(separator=" ").strip()[:max_len]
        except Exception:
            return raw[:max_len]

    def _parse_entry_date(self, entry) -> Optional[datetime]:
        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    return datetime(*parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _fetch_rss(self, source) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        try:
            # feedparser handles HTTP internally; pass custom agent via request_headers
            feed = feedparser.parse(
                source.url,
                request_headers=_HEADERS,
            )
            if feed.bozo and not feed.entries:
                logger.debug("RSS parse issue for %s: %s", source.name, feed.bozo_exception)
                return articles

            for entry in feed.entries:
                pub = self._parse_entry_date(entry)

                # Skip articles older than the lookback window
                if pub and pub < self._cutoff:
                    continue

                raw_summary = (
                    getattr(entry, "summary", None)
                    or getattr(entry, "description", None)
                    or ""
                )
                articles.append(
                    NewsArticle(
                        title=(getattr(entry, "title", "") or "").strip(),
                        summary=self._strip_html(raw_summary),
                        source=source.name,
                        published=pub,
                        url=getattr(entry, "link", "") or "",
                    )
                )
        except Exception as exc:
            logger.warning("Failed to fetch RSS '%s': %s", source.name, exc)
        return articles

    def fetch_all(self) -> List[NewsArticle]:
        all_articles: List[NewsArticle] = []
        for source in self.sources:
            # Check cache: skip HTTP fetch if recently cached
            if self._cache and not self._cache.should_fetch(source.name, self._cache_ttl):
                logger.debug("Cache hit: skipping fetch for '%s'", source.name)
                time.sleep(0)  # preserve loop rhythm
                continue

            if source.type == "rss":
                articles = self._fetch_rss(source)
                all_articles.extend(articles)
                if self._cache and articles:
                    self._cache.store(articles, source.name)
                elif self._cache:
                    # Mark as fetched even if no new articles, to update TTL
                    self._cache.store([], source.name)
            time.sleep(self.request_delay)

        # If cache is in use, return results from DB (includes older cached articles)
        if self._cache:
            return self._cache.get(self.lookback_hours, self.max_articles)

        # No cache — deduplicate, sort, and return from memory
        seen: set[str] = set()
        unique: List[NewsArticle] = []
        for a in all_articles:
            key = a.title.lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        # Sort newest first
        unique.sort(
            key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return unique[: self.max_articles]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, articles: List[NewsArticle]) -> str:
        if not articles:
            return "No recent news articles found within the lookback window."

        lines: List[str] = []
        for i, a in enumerate(articles, 1):
            date_str = (
                a.published.strftime("%d %b %Y %H:%M UTC")
                if a.published
                else "date unknown"
            )
            lines.append(f"{i}. [{a.source}] {a.title} ({date_str})")
            if a.summary:
                lines.append(f"   {a.summary[:250]}")
        return "\n".join(lines)
