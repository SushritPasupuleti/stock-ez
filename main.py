#!/usr/bin/env python3
"""
Stock-EZ — AI-powered stock & fund recommendation agent.
Uses local LLMs via Ollama; all data from free public sources.

Usage:
    python main.py                            # Run with config.yaml
    python main.py --model deepseek-r1:14b   # Override model
    python main.py --region "Mumbai, India"  # Override region
    python main.py --no-stream               # Non-streaming output
    python main.py --search-fund "Axis Blue" # Find fund scheme codes
    python main.py --list-models             # Show recommended models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import Settings
from src.agent.analyzer import MarketAnalyzer
from src.data_sources.funds import FundFetcher

console = Console()

BANNER = r"""
 ____  _             _        _____  _____
/ ___|| |_ ___   ___| | __   | ____|/ __  |
\___ \| __/ _ \ / __| |/ /   |  _|  `' / /
 ___) | || (_) | (__|   <    | |___   / /
|____/ \__\___/ \___|_|\_\   |_____|  \_\
"""

RECOMMENDED_MODELS = [
    ("qwen3:14b",             "~9 GB",  "Qwen3 14B — hybrid thinking mode, best overall (default)"),
    ("mistral-small3.2:24b", "~14 GB", "Mistral Small 3.2 — latest Mistral, fits in 16 GB"),
    ("deepseek-r1:14b",      "~9 GB",  "DeepSeek R1 — strong chain-of-thought reasoning"),
    ("phi4:14b",              "~9 GB",  "Microsoft Phi-4 — compact but capable"),
    ("gemma3:27b",            "~17 GB", "Google Gemma 3 27B — largest Gemma, near 16 GB limit"),
    ("gemma3:12b",            "~8 GB",  "Google Gemma 3 12B — balanced quality"),
    ("qwen3.5:9b",            "~6 GB",  "Qwen3.5 9B — next-gen Qwen, efficient mid-size"),
    ("qwen3:8b",              "~5 GB",  "Qwen3 8B — fast, still supports thinking mode"),
]


def show_banner() -> None:
    console.print(BANNER, style="bold cyan", highlight=False)
    console.print(
        "[dim]AI-powered stock & fund recommendations · Free data · Local LLMs[/dim]\n"
    )


def show_recommended_models() -> None:
    table = Table(
        title="Recommended Ollama Models for 16 GB VRAM",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model", style="green bold")
    table.add_column("VRAM", justify="right")
    table.add_column("Notes")

    for name, vram, notes in RECOMMENDED_MODELS:
        marker = " ◄ default" if "qwen3:14b" == name else ""
        table.add_row(name, vram, notes + marker)

    console.print(table)
    console.print(
        "\n[yellow]Pull a model:[/yellow]  [dim]ollama pull qwen3:14b[/dim]\n"
    )


def search_fund(query: str) -> None:
    console.print(f"\nSearching funds matching [cyan]{query!r}[/cyan]…\n")
    results = FundFetcher.search(query, limit=15)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Scheme Code", style="green", justify="right")
    table.add_column("Fund Name")

    for r in results:
        table.add_row(str(r.get("schemeCode", "")), r.get("schemeName", ""))

    console.print(table)
    console.print(
        "\n[dim]Add the scheme code to the [bold]watchlist.funds[/bold] section in config.yaml.[/dim]\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-ez",
        description="Stock-EZ: AI-powered investment recommendations (local LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument("--model", help="Override Ollama model, e.g. llama3.1:8b")
    parser.add_argument(
        "--region",
        help='Override region name shown in analysis, e.g. "Mumbai, India"',
    )
    parser.add_argument(
        "--market", choices=["NSE", "BSE"],
        help="Override stock market context",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable token streaming; print full response at once",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip Ollama connection and model checks",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Show recommended models for 16 GB VRAM and exit",
    )
    parser.add_argument(
        "--search-fund", metavar="QUERY",
        help="Search for a mutual fund scheme code by name and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return parser


def setup_logging(verbose: bool) -> None:
    import logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s  %(name)s — %(message)s",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    show_banner()

    # ── Utility commands ────────────────────────────────────────────────
    if args.list_models:
        show_recommended_models()
        return

    if args.search_fund:
        search_fund(args.search_fund)
        return

    # ── Load settings ───────────────────────────────────────────────────
    settings = Settings.load(args.config)

    if args.model:
        settings.llm.model = args.model
        console.print(f"[yellow]Model overridden →[/yellow] {args.model}")

    if args.region:
        settings.region.name = args.region
        console.print(f"[yellow]Region overridden →[/yellow] {args.region}")

    if args.market:
        settings.region.market = args.market

    # ── Run analysis ────────────────────────────────────────────────────
    analyzer = MarketAnalyzer(settings)
    result = analyzer.run(
        stream=not args.no_stream,
        skip_preflight=args.skip_preflight,
    )

    if not result:
        console.print("\n[bold red]Analysis did not complete. Check the output above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
