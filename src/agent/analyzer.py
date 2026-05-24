from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ..config.settings import Settings
from ..data_sources.funds import FundFetcher
from ..data_sources.news import NewsFetcher
from ..data_sources.stocks import StockFetcher
from .llm import OllamaClient
from .prompts import ANALYSIS_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)
console = Console()


class MarketAnalyzer:
    """
    Orchestrates data fetching → prompt assembly → LLM analysis → report output.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = OllamaClient(
            model=settings.llm.model,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            num_ctx=settings.llm.num_ctx,
        )
        self.news_fetcher = NewsFetcher(
            sources=settings.news.sources,
            max_articles=settings.news.max_articles,
            lookback_hours=settings.news.lookback_hours,
        )
        self.stock_fetcher = StockFetcher(region_config=settings.region)
        self.fund_fetcher = FundFetcher()

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def preflight_check(self) -> bool:
        console.print()
        console.print(Rule("[bold blue]Preflight Checks[/bold blue]"))

        ok = True

        if not self.llm.check_connection():
            console.print(
                f"[bold red]✗[/bold red] Ollama not reachable at [cyan]{self.settings.llm.base_url}[/cyan]"
            )
            console.print(
                "  [yellow]→ Start Ollama with:[/yellow]  [dim]ollama serve[/dim]"
            )
            ok = False
        else:
            console.print(f"[green]✓[/green] Ollama is running at [dim]{self.settings.llm.base_url}[/dim]")

        if ok and not self.llm.check_model():
            console.print(
                f"[bold red]✗[/bold red] Model [cyan]{self.settings.llm.model}[/cyan] not found locally"
            )
            console.print(
                f"  [yellow]→ Pull it with:[/yellow]  [dim]ollama pull {self.settings.llm.model}[/dim]"
            )
            available = self.llm.list_models()
            if available:
                console.print(
                    f"  [dim]Available models: {', '.join(available[:8])}[/dim]"
                )
            ok = False
        elif ok:
            console.print(f"[green]✓[/green] Model [cyan]{self.settings.llm.model}[/cyan] is available")

        return ok

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, stream: bool = True, skip_preflight: bool = False) -> Optional[str]:
        if not skip_preflight and not self.preflight_check():
            return None

        console.print()
        console.print(Rule("[bold blue]Gathering Market Data[/bold blue]"))

        # --- News ---
        console.print("[dim]Fetching market news...[/dim]")
        articles = self.news_fetcher.fetch_all()
        console.print(f"[green]✓[/green] {len(articles)} news articles ({self.settings.news.lookback_hours}h window)")

        # --- Indices ---
        console.print("[dim]Fetching market indices...[/dim]")
        indices = self.stock_fetcher.fetch_indices(self.settings.market_indices)
        console.print(f"[green]✓[/green] {len(indices)} indices")

        # --- Stocks ---
        console.print(
            f"[dim]Fetching {len(self.settings.watchlist.stocks)} watchlist stocks...[/dim]"
        )
        stocks = self.stock_fetcher.fetch_all(self.settings.watchlist.stocks)
        console.print(f"[green]✓[/green] {len(stocks)}/{len(self.settings.watchlist.stocks)} stocks fetched")

        # --- Funds ---
        console.print(
            f"[dim]Fetching {len(self.settings.watchlist.funds)} mutual funds...[/dim]"
        )
        funds = self.fund_fetcher.fetch_all(self.settings.watchlist.funds)
        console.print(f"[green]✓[/green] {len(funds)}/{len(self.settings.watchlist.funds)} funds fetched")

        # --- Build prompt context ---
        news_text = self.news_fetcher.format_for_prompt(articles)
        indices_text = self.stock_fetcher.format_indices(indices)
        stocks_text = self.stock_fetcher.format_for_prompt(stocks)
        funds_text = self.fund_fetcher.format_for_prompt(funds)

        system_prompt = SYSTEM_PROMPT.format(region=self.settings.region.name)
        user_prompt = ANALYSIS_PROMPT.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            lookback_hours=self.settings.news.lookback_hours,
            indices_data=indices_text or "No index data available.",
            news_data=news_text,
            stocks_data=stocks_text,
            funds_data=funds_text,
        )

        # --- LLM analysis ---
        console.print()
        console.print(
            Panel(
                f"[bold cyan]Analysing with {self.settings.llm.model}[/bold cyan]\n"
                f"[dim]Region: {self.settings.region.name}  •  "
                f"Market: {self.settings.region.market}  •  "
                f"Stream: {'on' if stream else 'off'}[/dim]",
                expand=False,
            )
        )
        console.print(Rule())
        console.print()

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=stream,
        )

        # --- Optional report save ---
        if self.settings.output.save_report and response:
            self._save_report(
                analysis=response,
                indices_text=indices_text,
                stocks_text=stocks_text,
                funds_text=funds_text,
                news_text=news_text,
            )

        return response

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def _save_report(
        self,
        analysis: str,
        indices_text: str,
        stocks_text: str,
        funds_text: str,
        news_text: str,
    ) -> None:
        report_dir = Path(self.settings.output.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"analysis_{ts}.md"

        content = (
            f"# Stock-EZ Market Analysis\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  \n"
            f"**Model:** {self.settings.llm.model}  \n"
            f"**Region:** {self.settings.region.name}  \n\n"
            f"---\n\n"
            f"## AI Analysis\n\n{analysis}\n\n"
            f"---\n\n"
            f"## Raw Data Used\n\n"
            f"### Market Indices\n\n{indices_text}\n\n"
            f"### Stocks\n\n{stocks_text}\n\n"
            f"### Mutual Funds\n\n{funds_text}\n\n"
            f"### News\n\n{news_text}\n"
        )

        report_path.write_text(content, encoding="utf-8")
        console.print(f"\n[dim]📄 Report saved → {report_path}[/dim]")
