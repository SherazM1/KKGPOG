"""
HTML/Playwright-based PDF renderer for Sam's Club price strips.

This is a parallel implementation to render_price_strips.py (ReportLab-based).
Uses Playwright/Chromium to render HTML/CSS to PDF with native Gibson OTF fonts.

To use this renderer, install Playwright:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import base64
import html
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib.units import inch

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow, SamsPriceStripSegment

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

_PAGE_WIDTH = 11.0 * inch
_PAGE_HEIGHT = 2.45 * inch
_DEFAULT_FOOTER_HEIGHT = 0.14 * inch
_STRIP_COMPOSITION_WIDTH_RATIO = 0.94
_STRIP_MARGIN_MIN = 0.07 * inch
_STRIP_MARGIN_MAX = 0.19 * inch
_MIN_TICKET_WIDTH = 0.70 * inch
_SAMS_BRAND_SIZE = 7.5
_SAMS_DESC_SIZE = 6.0
_SAMS_PRICE_SIZE = 47.0
_SAMS_ITEM_SIZE = 5.0
_SAMS_FOOTER_SIZE = 5.0
_SAMS_STACK_BRAND_GAP = 0.92
_SAMS_STACK_DESC_GAP = 0.72
_SAMS_STACK_TO_PRICE_OFFSET = 1.05
_SAMS_PRICE_SIGN_RISE_RATIO = 0.400
_SAMS_PRICE_CENTS_RISE_RATIO = 0.440
_SAMS_PRICE_SIGN_GAP_RATIO = 0.030
_SAMS_PRICE_CENTS_GAP_RATIO = 0.010
_RETAIL_MARGIN_PAD = 1.2
_DEFAULT_INNER_PAD_X = 0.055 * inch
_DEFAULT_INNER_PAD_TOP = 0.045 * inch
_DEFAULT_INNER_PAD_BOTTOM = 0.05 * inch
_MIN_TICKET_GAP = 0.0
_DEFAULT_TICKET_GAP = 0.02 * inch


def parse_strip_length(length_text: str) -> tuple[float, float] | None:
    raw = (length_text or "").strip()
    if raw == "":
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\"?\s*[xX×]\s*([0-9]+(?:\.[0-9]+)?)\s*\"?", raw)
    if not match:
        return None
    try:
        width_in = float(match.group(1))
        height_in = float(match.group(2))
    except ValueError:
        return None
    if width_in <= 0 or height_in <= 0:
        return None
    return width_in * inch, height_in * inch


def _resolve_group_length(row_data: SamsPriceStripRow, warnings: list[str]) -> tuple[float, float]:
    non_blank_lengths = [(seg.column, seg.length.strip()) for seg in row_data.segments if seg.length.strip()]
    if not non_blank_lengths:
        warnings.append(
            f"Missing Length for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; using default page size."
        )
        return _PAGE_WIDTH, _PAGE_HEIGHT

    first_length = non_blank_lengths[0][1]
    unique_lengths = sorted({value for _, value in non_blank_lengths})
    if len(unique_lengths) > 1:
        warnings.append(
            f"Multiple Length values for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; using first by column."
        )

    parsed = parse_strip_length(first_length)
    if parsed is None:
        warnings.append(
            f"Unparseable Length='{first_length}' for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; using default page size."
        )
        return _PAGE_WIDTH, _PAGE_HEIGHT
    return parsed


def _normalize_price_parts(raw_retail: str) -> tuple[str, str]:
    text = (raw_retail or "").strip()
    if text == "":
        return "-", "00"
    normalized = text.replace("$", "").replace(",", "").strip()
    try:
        amount = Decimal(normalized)
        quantized = amount.quantize(Decimal("0.01"))
        dollars = int(quantized)
        cents = int((quantized - Decimal(dollars)) * 100)
        return str(dollars), f"{abs(cents):02d}"
    except (InvalidOperation, ValueError):
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if len(digits) >= 3:
            return digits[:-2], digits[-2:]
        if len(digits) > 0:
            return digits, "00"
        return normalized or "-", "00"


def compute_strip_canvas(row_data: SamsPriceStripRow, warnings: list[str]) -> tuple[float, float, float]:
    strip_w, strip_h = _resolve_group_length(row_data, warnings)
    footer_h = min(_DEFAULT_FOOTER_HEIGHT, max(6.0, strip_h * 0.085))
    return strip_w, strip_h, footer_h


def _compute_strip_margins(strip_w: float, ticket_count: int) -> tuple[float, float]:
    base_margin = max(_STRIP_MARGIN_MIN, min(_STRIP_MARGIN_MAX, strip_w * 0.014))
    if ticket_count <= 2:
        base_margin = min(_STRIP_MARGIN_MAX, base_margin * 1.10)
    elif ticket_count >= 8:
        base_margin = max(_STRIP_MARGIN_MIN, base_margin * 0.90)
    return base_margin, base_margin


def _build_gap_sequence(ticket_count: int, base_gap: float) -> list[float]:
    if ticket_count <= 1:
        return []
    if ticket_count <= 3:
        return [base_gap] * (ticket_count - 1)
    modifiers = [0.06, -0.03, 0.02, -0.02]
    gaps: list[float] = []
    for idx in range(ticket_count - 1):
        m = modifiers[idx % len(modifiers)]
        gaps.append(max(_MIN_TICKET_GAP, base_gap * (1.0 + m)))
    return gaps


def compute_ticket_positions_across_strip(strip_w: float, ticket_count: int) -> list[tuple[float, float]]:
    if ticket_count <= 0:
        return []

    left_margin, right_margin = _compute_strip_margins(strip_w, ticket_count)
    gap = max(0.012 * inch, min(0.03 * inch, strip_w * 0.0019))
    gaps = _build_gap_sequence(ticket_count, gap)
    usable_w = strip_w - left_margin - right_margin
    total_gap_w = sum(gaps)
    composition_w = (usable_w - total_gap_w) / ticket_count

    if composition_w < _MIN_TICKET_WIDTH:
        gap = _MIN_TICKET_GAP
        gaps = [gap] * (ticket_count - 1)
        total_gap_w = sum(gaps)
        composition_w = (usable_w - total_gap_w) / ticket_count

    if composition_w < _MIN_TICKET_WIDTH:
        left_margin = right_margin = max(0.0, (strip_w - (ticket_count * _MIN_TICKET_WIDTH) - sum(gaps)) / 2.0)
        usable_w = strip_w - left_margin - right_margin
        composition_w = (usable_w - sum(gaps)) / ticket_count

    draw_w = max(0.60 * inch, composition_w * _STRIP_COMPOSITION_WIDTH_RATIO)
    inner_offset = max(0.0, (composition_w - draw_w) / 2.0)
    positions: list[tuple[float, float]] = []
    x = left_margin
    for idx in range(ticket_count):
        positions.append((x + inner_offset, draw_w))
        if idx < len(gaps):
            x += composition_w + gaps[idx]
        else:
            x += composition_w
    return positions


def _resolve_strip_footer_text(row_data: SamsPriceStripRow) -> str:
    raw = row_data.footer_text.strip()
    if raw and raw.lower() not in {"nan", "none", "null"}:
        return raw
    return f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"


def _ensure_playwright_chromium_installed() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            text=True,
        )
        return "Playwright Chromium install/check completed successfully."
    except Exception as exc:
        return f"Playwright Chromium install failed: {exc}"


def render_sams_price_strips_pdf(
    strip_rows: list[SamsPriceStripRow],
    generated_by: str = "Kendal King",
) -> SamsPriceStripPdfResult:
    """
    Render Sam's Club price strips to PDF using Playwright/Chromium with Gibson OTF fonts.

    Args:
        strip_rows: List of SamsPriceStripRow objects to render.
        generated_by: Attribution text (stored for metadata, not rendered yet).

    Returns:
        SamsPriceStripPdfResult with pdf_bytes, rendered_pages, rendered_segments, and warnings.
    """
    _ = generated_by
    warnings: list[str] = []
    warnings.append(
        "ACTIVE_SAMS_STRIP_RENDERER=app.sams_club.render_price_strips_html.render_sams_price_strips_pdf "
        "HTML/Playwright renderer with Gibson OTF fonts"
    )

    if not PLAYWRIGHT_AVAILABLE:
        warnings.append(
            "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        )
        return SamsPriceStripPdfResult(
            pdf_bytes=b"",
            rendered_pages=0,
            rendered_segments=0,
            warnings=warnings,
        )

    chromium_status = _ensure_playwright_chromium_installed()
    if chromium_status and "failed" in chromium_status:
        warnings.append(chromium_status)
        return SamsPriceStripPdfResult(
            pdf_bytes=b"",
            rendered_pages=0,
            rendered_segments=0,
            warnings=warnings,
        )
    else:
        if chromium_status:
            warnings.append(chromium_status)

    if not strip_rows:
        return SamsPriceStripPdfResult(
            pdf_bytes=b"",
            rendered_pages=0,
            rendered_segments=0,
            warnings=warnings,
        )

    try:
        htmls = _build_full_html(strip_rows, warnings)
        pdf_bytes, rendered_segments = asyncio.run(
            _render_strips_async(htmls, strip_rows, warnings)
        )
    except Exception as exc:
        warnings.append(f"HTML/Playwright renderer failed: {exc}")
        pdf_bytes = b""
        rendered_segments = 0

    return SamsPriceStripPdfResult(
        pdf_bytes=pdf_bytes,
        rendered_pages=len(strip_rows),
        rendered_segments=rendered_segments,
        warnings=warnings,
    )


def _build_full_html(strip_rows: list[SamsPriceStripRow], warnings: list[str]) -> list[str]:
    """Build all HTML strings synchronously before Playwright processing."""
    htmls = []
    for row_data in strip_rows:
        strip_w, strip_h, footer_h = compute_strip_canvas(row_data, warnings)
        html_content = _generate_strip_html(row_data, strip_w, strip_h, footer_h, warnings)
        htmls.append(html_content)
    return htmls


async def _render_strips_async(
    htmls: list[str],
    strip_rows: list[SamsPriceStripRow],
    warnings: list[str],
) -> tuple[bytes, int]:
    """Render strips asynchronously using Playwright."""
    from playwright.async_api import async_playwright

    rendered_segments = 0
    all_pdf_bytes = b""

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            for idx, row_data in enumerate(strip_rows):
                html_content = htmls[idx]
                strip_w, strip_h, footer_h = compute_strip_canvas(row_data, warnings)
                page_pdf = await _render_page_to_pdf(browser, html_content, strip_w, strip_h)

                if idx == 0:
                    all_pdf_bytes = page_pdf
                else:
                    # Merge PDFs by concatenating bytes (simple approach for now).
                    # In production, use PyPDF2 or similar for proper PDF merging.
                    all_pdf_bytes += page_pdf

                rendered_segments += len(row_data.segments)
        finally:
            await browser.close()

    return all_pdf_bytes, rendered_segments


async def _render_page_to_pdf(
    browser: Browser,
    html_content: str,
    width: float,
    height: float,
) -> bytes:
    """
    Render HTML content to PDF using Playwright.

    Args:
        browser: Playwright browser instance.
        html_content: HTML string to render.
        width: Page width in points.
        height: Page height in points.

    Returns:
        PDF bytes.
    """
    page = await browser.new_page()
    try:
        await page.set_content(html_content)
        # Make sure document fonts are loaded before exporting.
        await page.evaluate("() => document.fonts.ready")
        pdf_bytes = await page.pdf(width=f"{width / inch}in", height=f"{height / inch}in")
        return pdf_bytes
    finally:
        await page.close()


def _generate_strip_html(row_data: SamsPriceStripRow, strip_w: float, strip_h: float, footer_h: float, warnings: list[str]) -> str:
    """
    Generate HTML for a single price strip row.

    Layout:
    - One PDF page per row
    - Repeated ticket blocks across the page horizontally
    - Each ticket has: brand, description (2 lines), price, item number
    - Footer text at the bottom

    Args:
        row_data: SamsPriceStripRow data.
        strip_w: Strip width in points.
        strip_h: Strip height in points.
        footer_h: Footer height in points.
        warnings: List to append warnings.

    Returns:
        HTML string with embedded Gibson font faces.
    """
    root_path = Path(__file__).resolve().parents[2]
    gibson_regular_path = root_path / "assets" / "Gibson-Regular.otf"
    gibson_semibold_path = root_path / "assets" / "Gibson-SemiBold.otf"

    # Encode font files as data URIs for embedding in HTML.
    try:
        regular_data_uri = _font_file_to_data_uri(gibson_regular_path)
        semibold_data_uri = _font_file_to_data_uri(gibson_semibold_path)
        fonts_available = True
    except Exception:
        regular_data_uri = ""
        semibold_data_uri = ""
        fonts_available = False

    # Build CSS with @font-face for Gibson fonts.
    font_face_css = ""
    if fonts_available:
        font_face_css = f"""
@font-face {{
    font-family: "Gibson";
    src: url('{regular_data_uri}') format('opentype');
    font-weight: 400;
}}

@font-face {{
    font-family: "Gibson";
    src: url('{semibold_data_uri}') format('opentype');
    font-weight: 600;
}}
"""

    positions = compute_ticket_positions_across_strip(strip_w, len(row_data.segments))
    ticket_y = footer_h
    ticket_h = strip_h - footer_h

    ticket_htmls = []
    for idx, segment in enumerate(row_data.segments):
        x, w = positions[idx]
        ticket_html = _generate_ticket_html(segment, x, ticket_y, w, ticket_h)
        ticket_htmls.append(ticket_html)

    footer_text = _resolve_strip_footer_text(row_data)
    footer_html = f'<div class="footer">{html.escape(footer_text)}</div>'

    # HTML structure with absolute positioning.
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Sam's Club Price Strip</title>",
        "<style>",
        font_face_css,
        f"""
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: Gibson, Arial, sans-serif;
    background: white;
    margin: 0;
    padding: 0;
    position: relative;
    width: {strip_w}pt;
    height: {strip_h}pt;
}}

.ticket {{
    position: absolute;
}}

.brand {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 600;
    font-size: {_SAMS_BRAND_SIZE}pt;
    color: black;
}}

.desc {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 400;
    font-size: {_SAMS_DESC_SIZE}pt;
    color: black;
}}

.price {{
    position: absolute;
}}

.dollar-sign {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 600;
    font-size: 15.75pt;
    color: black;
}}

.dollars {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 600;
    font-size: 37.05pt;
    color: black;
}}

.cents {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 600;
    font-size: 16.8pt;
    color: black;
}}

.item-number {{
    position: absolute;
    font-family: "Gibson";
    font-weight: 400;
    font-size: {_SAMS_ITEM_SIZE}pt;
    color: black;
}}

.footer {{
    position: absolute;
    left: 0.03in;
    bottom: 0.01in;
    font-family: "Gibson";
    font-weight: 400;
    font-size: {_SAMS_FOOTER_SIZE}pt;
    color: #303030;
}}
        """,
        "</style>",
        "</head>",
        "<body>",
    ]

    html_parts.extend(ticket_htmls)
    html_parts.append(footer_html)
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


def _generate_ticket_html(segment: SamsPriceStripSegment, x: float, y: float, w: float, h: float) -> str:
    """
    Generate HTML for a single ticket block within the strip.

    Args:
        segment: SamsPriceStripSegment data.
        x: Left position in points.
        y: Top position in points.
        w: Width in points.
        h: Height in points.

    Returns:
        HTML string for the ticket.
    """
    dollars, cents = _normalize_price_parts(segment.retail)

    pad_x = _DEFAULT_INNER_PAD_X
    pad_top = _DEFAULT_INNER_PAD_TOP

    brand_x = x + pad_x
    brand_y = y + pad_top

    desc1_x = x + pad_x
    desc1_y = brand_y + _SAMS_BRAND_SIZE + _SAMS_STACK_BRAND_GAP

    desc2_x = x + pad_x
    desc2_y = desc1_y + _SAMS_DESC_SIZE + _SAMS_STACK_DESC_GAP

    price_y = desc2_y + _SAMS_DESC_SIZE + _SAMS_STACK_TO_PRICE_OFFSET

    price_left = x + max(_RETAIL_MARGIN_PAD, pad_x * 0.35)

    dollar_sign_x = price_left
    dollar_sign_y = price_y + (_SAMS_PRICE_SIZE * _SAMS_PRICE_SIGN_RISE_RATIO)

    dollars_x = dollar_sign_x + (_SAMS_PRICE_SIZE * 0.33) + (_SAMS_PRICE_SIZE * _SAMS_PRICE_SIGN_GAP_RATIO)
    dollars_y = price_y

    cents_x = dollars_x + _SAMS_PRICE_SIZE + (_SAMS_PRICE_SIZE * _SAMS_PRICE_CENTS_GAP_RATIO)
    cents_y = price_y + (_SAMS_PRICE_SIZE * _SAMS_PRICE_CENTS_RISE_RATIO)

    item_x = cents_x + (_SAMS_PRICE_SIZE * 0.40) + 0.1 * inch
    item_y = price_y + (_SAMS_PRICE_SIZE * 0.16)

    ticket_html = f'''
<div class="ticket" style="left: {x}pt; top: {y}pt; width: {w}pt; height: {h}pt;">
<div class="brand" style="left: {pad_x}pt; top: {pad_top}pt;">{html.escape(segment.brand or "")}</div>
<div class="desc" style="left: {pad_x}pt; top: {desc1_y - y}pt;">{html.escape(segment.desc_1 or "")}</div>
<div class="desc" style="left: {pad_x}pt; top: {desc2_y - y}pt;">{html.escape(segment.desc_2 or "")}</div>
<div class="dollar-sign" style="left: {dollar_sign_x - x}pt; top: {dollar_sign_y - y}pt;">$</div>
<div class="dollars" style="left: {dollars_x - x}pt; top: {dollars_y - y}pt;">{dollars}</div>
<div class="cents" style="left: {cents_x - x}pt; top: {cents_y - y}pt;">{cents}</div>
<div class="item-number" style="left: {item_x - x}pt; top: {item_y - y}pt;">{html.escape(segment.item_number or "")}</div>
</div>
'''

    return ticket_html.strip()


def _font_file_to_data_uri(font_path: Path) -> str:
    """
    Convert a font file to a data URI for embedding in HTML.

    Args:
        font_path: Path to the font file.

    Returns:
        Data URI string.

    Raises:
        FileNotFoundError: If the font file does not exist.
    """
    if not font_path.is_file():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    with open(font_path, "rb") as f:
        font_bytes = f.read()

    b64_str = base64.b64encode(font_bytes).decode("ascii")
    return f"data:font/otf;base64,{b64_str}"
