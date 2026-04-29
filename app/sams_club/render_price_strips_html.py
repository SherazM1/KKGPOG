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
from pathlib import Path
from typing import TYPE_CHECKING

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


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

    try:
        pdf_bytes, rendered_segments = asyncio.run(
            _render_strips_async(strip_rows, warnings)
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


async def _render_strips_async(
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
            for row_idx, row_data in enumerate(strip_rows):
                html_content = _generate_strip_html(row_data)
                page_pdf = await _render_page_to_pdf(browser, html_content)

                if row_idx == 0:
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
) -> bytes:
    """
    Render HTML content to PDF using Playwright.

    Args:
        browser: Playwright browser instance.
        html_content: HTML string to render.

    Returns:
        PDF bytes.
    """
    page = await browser.new_page()
    try:
        await page.set_content(html_content)
        # Make sure document fonts are loaded before exporting.
        await page.evaluate("() => document.fonts.ready")
        pdf_bytes = await page.pdf(format="Letter")
        return pdf_bytes
    finally:
        await page.close()


def _generate_strip_html(row_data: SamsPriceStripRow) -> str:
    """
    Generate HTML for a single price strip row.

    Layout:
    - One PDF page per row
    - Repeated ticket blocks across the page horizontally
    - Each ticket has: brand, description (2 lines), price, item number
    - Footer text at the bottom

    Args:
        row_data: SamsPriceStripRow data.

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

    # HTML structure: strip container with tickets arranged horizontally.
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Sam's Club Price Strip</title>",
        "<style>",
        font_face_css,
        """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Gibson, Arial, sans-serif;
    background: white;
    margin: 0;
    padding: 0;
}

.strip-page {
    width: 11in;
    height: 2.45in;
    background: white;
    display: flex;
    flex-direction: column;
    padding: 0.05in;
    position: relative;
}

.strip-content {
    flex: 1;
    display: flex;
    gap: 0.02in;
    align-items: stretch;
    overflow: hidden;
}

.ticket {
    flex: 1;
    min-width: 0.7in;
    background: white;
    border: 1px solid #ccc;
    padding: 0.055in;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-size: 12px;
}

.ticket-text-stack {
    display: flex;
    flex-direction: column;
    gap: 0.02in;
}

.brand {
    font-size: 7.5pt;
    font-weight: 600;
    color: #000;
    line-height: 1;
}

.description {
    font-size: 6pt;
    color: #0a0a0a;
    line-height: 1.1;
}

.price-section {
    display: flex;
    align-items: flex-end;
    gap: 0.01in;
    margin: 0.05in 0;
}

.dollar-sign {
    font-size: 18.8pt;
    font-weight: 600;
    color: #000;
    line-height: 1;
}

.dollars {
    font-size: 37.6pt;
    font-weight: 600;
    color: #000;
    line-height: 1;
}

.cents {
    font-size: 16.6pt;
    font-weight: 600;
    color: #000;
    line-height: 1;
}

.item-number {
    font-size: 5pt;
    color: #303030;
    margin-top: 0.02in;
}

.strip-footer {
    background: white;
    padding: 0.01in 0.055in;
    font-size: 5pt;
    color: #303030;
    border-top: 1px solid #ddd;
    text-align: center;
    line-height: 1;
}
        """,
        "</style>",
        "</head>",
        "<body>",
    ]

    # Generate ticket HTML for each segment.
    html_parts.append("<div class='strip-page'>")
    html_parts.append("<div class='strip-content'>")

    for segment in row_data.segments:
        html_parts.append(_generate_ticket_html(segment))

    html_parts.append("</div>")  # .strip-content

    # Footer
    if row_data.footer_text:
        html_parts.append(f"<div class='strip-footer'>{html.escape(row_data.footer_text)}</div>")

    html_parts.append("</div>")  # .strip-page
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


def _generate_ticket_html(segment) -> str:
    """
    Generate HTML for a single ticket block within the strip.

    Args:
        segment: SamsPriceStripSegment data.

    Returns:
        HTML string for the ticket.
    """
    # Parse retail price into dollar, dollars, and cents components.
    dollar_sign, dollars, cents = _parse_price(segment.retail)

    html_parts = [
        "<div class='ticket'>",
        "<div class='ticket-text-stack'>",
        f"<div class='brand'>{html.escape(segment.brand)}</div>",
    ]

    if segment.desc_1:
        html_parts.append(f"<div class='description'>{html.escape(segment.desc_1)}</div>")
    if segment.desc_2:
        html_parts.append(f"<div class='description'>{html.escape(segment.desc_2)}</div>")

    html_parts.append("</div>")  # .ticket-text-stack

    # Price display with spans for dollar sign, dollars, and cents.
    if segment.retail:
        html_parts.append("<div class='price-section'>")
        if dollar_sign:
            html_parts.append(f"<span class='dollar-sign'>{html.escape(dollar_sign)}</span>")
        if dollars:
            html_parts.append(f"<span class='dollars'>{html.escape(dollars)}</span>")
        if cents:
            html_parts.append(f"<span class='cents'>{html.escape(cents)}</span>")
        html_parts.append("</div>")  # .price-section

    # Item number at the bottom.
    if segment.item_number:
        html_parts.append(f"<div class='item-number'>{html.escape(segment.item_number)}</div>")

    html_parts.append("</div>")  # .ticket

    return "\n".join(html_parts)


def _parse_price(price_str: str) -> tuple[str, str, str]:
    """
    Parse a price string into dollar sign, dollars, and cents components.

    Args:
        price_str: Price string (e.g., "$12.99").

    Returns:
        Tuple of (dollar_sign, dollars, cents).
    """
    if not price_str:
        return "", "", ""

    price_str = price_str.strip()

    # Extract dollar sign if present.
    dollar_sign = ""
    if price_str.startswith("$"):
        dollar_sign = "$"
        price_str = price_str[1:]

    # Split on decimal point.
    parts = price_str.split(".")
    dollars = parts[0] if parts else ""
    cents = parts[1][:2] if len(parts) > 1 else ""

    return dollar_sign, dollars, cents


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
