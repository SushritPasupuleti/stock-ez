"""
PDF report generation using fpdf2.

Produces a multi-page PDF with:
  • Cover page
  • Market indices table
  • Full AI analysis (rendered as plain text)
  • Stock performance table
  • Mutual fund performance table
  • News summary list

Note: PDF output uses Helvetica (Latin-1). The ₹ symbol is rendered as "Rs."
      and any other non-Latin-1 characters are dropped safely.
      For full Unicode output use the Markdown export.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

# fpdf2 is an optional dep – import errors are surfaced only at call time
try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_MD_PATTERNS = [
    (re.compile(r"#{1,6}\s+"), ""),          # ATX headers → remove marker
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),   # bold
    (re.compile(r"\*(.+?)\*"), r"\1"),       # italic
    (re.compile(r"`(.+?)`"), r"\1"),         # inline code
    (re.compile(r"```.*?```", re.S), ""),    # fenced code blocks
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"), # markdown links → keep text
]


def _safe(text: str) -> str:
    """Replace/drop characters that can't encode to Latin-1 (Helvetica core font)."""
    text = (
        text.replace("₹", "Rs.")
            .replace("\u2013", "-")   # en-dash
            .replace("\u2014", "--")  # em-dash
            .replace("\u2022", "*")   # bullet
            .replace("\u2019", "'")   # right single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u00a0", " ")   # non-breaking space
    )
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_md(text: str) -> str:
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _clean(text: str) -> str:
    return _safe(_strip_md(text))


def _fmt(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

_BLUE = (41, 98, 255)
_LIGHT_BLUE = (219, 234, 254)
_GREY = (245, 245, 245)
_WHITE = (255, 255, 255)
_BLACK = (30, 30, 30)
_DARK_GREY = (80, 80, 80)


class _ReportPDF(FPDF):
    _subtitle: str = ""

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 8, "Stock-EZ Market Analysis Report", align="L")
        self.set_x(-50)
        self.cell(40, 8, f"Page {self.page_no()}", align="R")
        self.set_text_color(*_BLACK)
        self.ln(0)
        self.set_draw_color(*_LIGHT_BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 6, "AI-generated analysis. Not financial advice. DYOR.", align="C")
        self.set_text_color(*_BLACK)

    # ------------------------------------------------------------------ helpers

    def section_header(self, title: str, icon: str = "") -> None:
        self.ln(3)
        self.set_fill_color(*_BLUE)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 11)
        label = f"  {icon}  {title}" if icon else f"  {title}"
        self.cell(0, 9, _safe(label), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_BLACK)
        self.ln(2)

    def body_text(self, text: str, font_size: int = 9) -> None:
        self.set_font("Helvetica", "", font_size)
        self.set_text_color(*_BLACK)
        for paragraph in text.split("\n\n"):
            para = _clean(paragraph).strip()
            if not para:
                continue
            # Treat lines starting with * or - as bullet points
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(("*", "-", "•")):
                    self.set_x(self.l_margin + 4)
                    self.multi_cell(0, 5.5, _safe("  " + line[1:].strip()))
                else:
                    self.multi_cell(0, 5.5, _safe(line))
            self.ln(2)

    def draw_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float],
        row_h: float = 6.5,
    ) -> None:
        page_w = self.w - self.l_margin - self.r_margin
        # Scale col_widths to fit the page exactly
        total = sum(col_widths)
        col_widths = [w / total * page_w for w in col_widths]

        # Header row
        self.set_fill_color(*_BLUE)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h, _safe(str(h)), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_BLACK)
        for r_idx, row in enumerate(rows):
            fill_color = _GREY if r_idx % 2 == 0 else _WHITE
            self.set_fill_color(*fill_color)
            # Page-break guard: start a new page before the row if needed
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
                # Redraw header on continuation page
                self.set_fill_color(*_BLUE)
                self.set_text_color(*_WHITE)
                self.set_font("Helvetica", "B", 8)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], row_h, _safe(str(h)), border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8)
                self.set_text_color(*_BLACK)
                self.set_fill_color(*fill_color)

            for i, cell_val in enumerate(row):
                text = _safe(str(cell_val))
                # Truncate long cells to avoid overflow
                max_chars = max(6, int(col_widths[i] / 2))
                if len(text) > max_chars:
                    text = text[: max_chars - 1] + "…"
                align = "R" if i > 0 else "L"
                self.cell(col_widths[i], row_h, text, border=1, fill=True, align=align)
            self.ln()
        self.ln(2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_pdf(
    analysis: str,
    stocks_data: Optional[list[dict]] = None,
    funds_data: Optional[list[dict]] = None,
    indices_data: Optional[list[dict]] = None,
    news_articles: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> bytes:
    """
    Build a multi-page PDF report and return the raw bytes.

    Parameters
    ----------
    analysis       : Full AI analysis text (markdown).
    stocks_data    : List of stock dicts from StockData (optional).
    funds_data     : List of fund dicts from FundData (optional).
    indices_data   : List of index dicts (optional).
    news_articles  : List of NewsArticle objects (optional).
    metadata       : Dict with keys: date, model, region (optional).
    """
    if not _FPDF_AVAILABLE:
        raise ImportError(
            "fpdf2 is required for PDF export. Run: uv add fpdf2"
        )

    meta = metadata or {}
    date_str = meta.get("date", datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"))
    model_str = meta.get("model", "Unknown")
    region_str = meta.get("region", "India")

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 18, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title("Stock-EZ Market Analysis")

    # ── Page 1: Cover ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*_BLUE)
    pdf.rect(0, 0, 210, 80, "F")

    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 12, "Stock-EZ", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 9, "AI Market Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, date_str, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(90)
    pdf.set_text_color(*_BLACK)

    # Meta info box
    pdf.set_fill_color(*_LIGHT_BLUE)
    pdf.set_font("Helvetica", "", 10)
    info_lines = [
        f"Model:    {model_str}",
        f"Region:   {region_str}",
        f"Generated: {date_str}",
    ]
    for line in info_lines:
        pdf.set_x(40)
        pdf.cell(130, 8, _safe(line), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "DISCLAIMER: This report is generated by an AI language model and is provided "
        "for informational purposes only. It does not constitute financial advice. "
        "Past performance is not indicative of future results. Please consult a SEBI-registered "
        "investment advisor before making investment decisions.",
        align="C",
    )
    pdf.set_text_color(*_BLACK)

    # ── Page 2: Market Indices (if available) ────────────────────────────────
    if indices_data:
        pdf.add_page()
        pdf.section_header("Market Indices Overview", "📊")
        headers = ["Index", "Price (Rs.)", "1D %", "5D %", "52W High", "52W Low"]
        col_w = [55, 25, 22, 22, 28, 28]
        rows = []
        for idx in indices_data:
            rows.append([
                str(idx.get("name", idx.get("symbol", ""))),
                _fmt(idx.get("current_price"), 2),
                _fmt(idx.get("change_1d_pct"), 2, "%"),
                _fmt(idx.get("change_5d_pct"), 2, "%"),
                _fmt(idx.get("week_52_high"), 2),
                _fmt(idx.get("week_52_low"), 2),
            ])
        pdf.draw_table(headers, rows, col_w)

    # ── AI Analysis sections ─────────────────────────────────────────────────
    if analysis:
        pdf.add_page()
        # Split the analysis into named sections by "###" markers
        raw_sections = re.split(r"(?m)^###\s+", analysis)
        for raw in raw_sections:
            raw = raw.strip()
            if not raw:
                continue
            first_newline = raw.find("\n")
            if first_newline == -1:
                section_title = raw
                section_body = ""
            else:
                section_title = raw[:first_newline].strip()
                section_body = raw[first_newline:].strip()

            # Number prefix in title (e.g. "1. Market Sentiment") → strip number
            section_title = re.sub(r"^\d+\.\s*", "", section_title)

            # Choose an icon for known section names
            icon_map = {
                "market sentiment": "📈",
                "stock recommendations": "⭐",
                "mutual fund recommendations": "💼",
                "caution list": "⚠️",
                "top": "🔥",
                "disclaimer": "ℹ️",
            }
            icon = next(
                (v for k, v in icon_map.items() if k in section_title.lower()), ""
            )
            pdf.section_header(section_title, icon)
            if section_body:
                pdf.body_text(section_body)

    # ── Stock Performance Table ───────────────────────────────────────────────
    if stocks_data:
        pdf.add_page()
        pdf.section_header("Stock Performance Data", "📈")
        s_headers = ["Stock", "Symbol", "Price (Rs.)", "1D%", "5D%", "1M%", "P/E", "MCap (Cr)"]
        s_col_w = [40, 25, 22, 14, 14, 14, 14, 27]
        s_rows = []
        for s in stocks_data:
            mcap = s.get("market_cap_cr")  # already in crores
            mcap_str = _fmt(mcap, 0) if mcap else "N/A"
            s_rows.append([
                _safe(str(s.get("name", "")))[:22],
                str(s.get("symbol", "")).replace(".NS", "").replace(".BO", ""),
                _fmt(s.get("current_price"), 2),
                _fmt(s.get("change_1d_pct"), 2, "%"),
                _fmt(s.get("change_5d_pct"), 2, "%"),
                _fmt(s.get("change_1m_pct"), 2, "%"),
                _fmt(s.get("pe_ratio"), 1),
                mcap_str,
            ])
        pdf.draw_table(s_headers, s_rows, s_col_w)

    # ── Mutual Fund Performance Table ─────────────────────────────────────────
    if funds_data:
        pdf.add_page()
        pdf.section_header("Mutual Fund Performance Data", "💼")
        f_headers = ["Fund", "NAV (Rs.)", "1M%", "3M%", "6M%", "1Y%", "Type"]
        f_col_w = [65, 22, 16, 16, 16, 16, 29]
        f_rows = []
        for f in funds_data:
            f_rows.append([
                _safe(str(f.get("name", "")))[:40],
                _fmt(f.get("nav"), 4),
                _fmt(f.get("returns_1m"), 2, "%"),
                _fmt(f.get("returns_3m"), 2, "%"),
                _fmt(f.get("returns_6m"), 2, "%"),
                _fmt(f.get("returns_1y"), 2, "%"),
                _safe(str(f.get("fund_category") or f.get("fund_type") or ""))[:15],
            ])
        pdf.draw_table(f_headers, f_rows, f_col_w)

    # ── News Summary ──────────────────────────────────────────────────────────
    if news_articles:
        pdf.add_page()
        pdf.section_header("News Summary", "📰")
        pdf.set_font("Helvetica", "", 9)
        for i, article in enumerate(news_articles[:40], 1):
            title = getattr(article, "title", str(article))
            source = getattr(article, "source", "")
            pub = getattr(article, "published", None)
            date_label = pub.strftime("%d %b %H:%M") if pub else ""

            # Page-break guard
            if pdf.get_y() + 9 > pdf.page_break_trigger:
                pdf.add_page()
                pdf.set_font("Helvetica", "", 9)

            pdf.set_text_color(*_DARK_GREY)
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(0, 4.5, _safe(f"[{source}]  {date_label}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_BLACK)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _safe(f"{i}. {title}"))
            pdf.ln(1)

    return bytes(pdf.output())
