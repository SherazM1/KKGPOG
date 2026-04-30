"""
HTML/Playwright-based PDF renderer for Sam's Club price strips.

This is a parallel implementation to render_price_strips.py (ReportLab-based).
Uses Playwright/Chromium to render HTML/SVG to PDF with native Gibson OTF fonts.

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
from typing import TYPE_CHECKING, NamedTuple

import fitz
from reportlab.lib.units import inch

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow, SamsPriceStripSegment

if TYPE_CHECKING:
    from playwright.async_api import Browser

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
_SAMS_PRICE_SIZE = 44.0
_SAMS_ITEM_SIZE = 5.0
_SAMS_FOOTER_SIZE = 5.0
_SAMS_STACK_BRAND_GAP = 0.92
_SAMS_STACK_DESC_GAP = 0.72
_SAMS_STACK_TO_PRICE_OFFSET = 2.0
_SAMS_PRICE_SIGN_RISE_RATIO = 0.365
_SAMS_PRICE_CENTS_RISE_RATIO = 0.405
_SAMS_PRICE_SIGN_GAP_RATIO = 0.022
_SAMS_PRICE_CENTS_GAP_RATIO = 0.006
_RETAIL_MARGIN_PAD = 1.2
_DEFAULT_INNER_PAD_X = 0.055 * inch
_DEFAULT_INNER_PAD_TOP = 0.045 * inch
_DEFAULT_INNER_PAD_BOTTOM = 0.05 * inch
_MIN_TICKET_GAP = 0.0
_DEFAULT_TICKET_GAP = 0.02 * inch
_PRICE_OBJECT_BAND_ANCHOR_RATIO = 0.30
_TICKET_VERTICAL_LIFT_RATIO = 0.08
_TICKET_VERTICAL_LIFT_MAX = 0.16 * inch


class _PriceObjectLayout(NamedTuple):
    dollar_sign_x: float
    dollar_sign_baseline_y: float
    dollar_sign_size: float
    dollars_x: float
    dollars_baseline_y: float
    dollars_size: float
    cents_x: float
    cents_baseline_y: float
    cents_size: float
    object_left_x: float
    object_bottom_y: float
    object_right_x: float
    object_top_y: float


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


def _estimate_text_width(text: str, font_size: float, weight: str = "regular") -> float:
    """
    Approximate Gibson text width in points for fitting/truncation only.

    Browser/SVG performs the actual render, but we need a deterministic estimate
    while calculating price object placement and truncation server-side.
    """
    value = text or ""
    if not value:
        return 0.0
    factor = 0.54 if weight == "semibold" else 0.50
    return len(value) * font_size * factor


def _truncate_svg_text(text: str, font_size: float, max_width: float, weight: str = "regular") -> str:
    value = (text or "").strip()
    if not value:
        return ""

    if _estimate_text_width(value, font_size, weight) <= max_width:
        return value

    suffix = "..."
    suffix_w = _estimate_text_width(suffix, font_size, weight)
    available = max(0.0, max_width - suffix_w)

    out = value
    while out and _estimate_text_width(out, font_size, weight) > available:
        out = out[:-1]

    return f"{out}{suffix}" if out else suffix


def _svg_y(page_h: float, reportlab_baseline_y: float) -> float:
    """
    Convert ReportLab-style bottom-origin baseline y into SVG top-origin baseline y.
    SVG <text y="..."> is baseline-based, so this preserves text sitting behavior.
    """
    return page_h - reportlab_baseline_y


def _compute_content_box_metrics(y: float, h: float) -> tuple[float, float, float]:
    inner_top = _DEFAULT_INNER_PAD_TOP * 0.55
    inner_bottom = _DEFAULT_INNER_PAD_BOTTOM * 0.55
    available_h = max(12.0, h - inner_bottom - inner_top)
    content_h = max(12.0, available_h * 0.92)
    min_content_y = y + inner_bottom
    max_content_y = max(min_content_y, y + h - inner_top - content_h)
    return min_content_y, max_content_y, content_h


def _compute_row_content_y(ticket_y: float, strip_content_h: float) -> tuple[float, float]:
    """Mirror the old centered content box behavior for the HTML/SVG renderer."""
    min_content_y, max_content_y, content_h = _compute_content_box_metrics(ticket_y, strip_content_h)
    lift = min(_TICKET_VERTICAL_LIFT_MAX, max(0.0, strip_content_h * _TICKET_VERTICAL_LIFT_RATIO))
    centered_content_y = ticket_y + ((strip_content_h - content_h) / 2.0)
    content_y = min(max(centered_content_y + lift, min_content_y), max_content_y)
    return content_y, content_h


def _layout_price_object_svg(retail: str, x: float, y: float, w: float, h: float) -> _PriceObjectLayout:
    dollars, cents = _normalize_price_parts(retail)

    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    price_left = x + max(_RETAIL_MARGIN_PAD, pad_x * 0.35)
    price_right = x + w - pad_x
    max_object_w = max(12.0, price_right - price_left)
    max_object_h = max(15.0, h * 0.60)

    base_dollars_size = _SAMS_PRICE_SIZE
    base_sign_size = _SAMS_PRICE_SIZE * 0.33
    base_cents_size = _SAMS_PRICE_SIZE * 0.40
    base_sign_gap = max(0.9, base_dollars_size * _SAMS_PRICE_SIGN_GAP_RATIO)
    base_cents_gap = max(0.7, base_dollars_size * _SAMS_PRICE_CENTS_GAP_RATIO)
    base_sign_rise = base_dollars_size * _SAMS_PRICE_SIGN_RISE_RATIO
    base_cents_rise = base_dollars_size * _SAMS_PRICE_CENTS_RISE_RATIO

    sign_w = _estimate_text_width("$", base_sign_size, "semibold")
    dollars_w = _estimate_text_width(dollars, base_dollars_size, "semibold")
    cents_w = _estimate_text_width(cents, base_cents_size, "semibold")

    base_object_w = sign_w + base_sign_gap + dollars_w + base_cents_gap + cents_w
    base_object_ascent = max(
        base_dollars_size * 0.86,
        base_sign_rise + (base_sign_size * 0.82),
        base_cents_rise + (base_cents_size * 0.82),
    )
    base_object_descent = max(base_dollars_size * 0.16, base_sign_size * 0.12, base_cents_size * 0.12)
    base_object_h = base_object_ascent + base_object_descent

    scale = min(1.0, max_object_w / max(base_object_w, 1.0), max_object_h / max(base_object_h, 1.0))
    scale = max(0.32, scale)

    dollars_size = base_dollars_size * scale
    sign_size = base_sign_size * scale
    cents_size = base_cents_size * scale
    sign_gap = base_sign_gap * scale
    cents_gap = base_cents_gap * scale
    sign_rise = base_sign_rise * scale
    cents_rise = base_cents_rise * scale

    sign_gap = min(max(sign_gap, dollars_size * 0.018), dollars_size * 0.036)
    cents_gap = min(max(cents_gap, dollars_size * 0.005), dollars_size * 0.014)
    sign_rise = min(max(sign_rise, dollars_size * 0.355), dollars_size * 0.425)
    cents_rise = min(max(cents_rise, dollars_size * 0.415), dollars_size * 0.465)

    sign_w = _estimate_text_width("$", sign_size, "semibold")
    dollars_w = _estimate_text_width(dollars, dollars_size, "semibold")
    cents_w = _estimate_text_width(cents, cents_size, "semibold")

    object_w = sign_w + sign_gap + dollars_w + cents_gap + cents_w
    object_ascent = max(
        dollars_size * 0.86,
        sign_rise + (sign_size * 0.82),
        cents_rise + (cents_size * 0.82),
    )
    object_descent = max(dollars_size * 0.16, sign_size * 0.12, cents_size * 0.12)

    object_bottom_y = y + max(1.25, h * _PRICE_OBJECT_BAND_ANCHOR_RATIO)
    dollars_baseline = object_bottom_y + object_descent

    object_left = price_left
    dollar_sign_x = object_left
    dollars_x = object_left + sign_w + sign_gap
    cents_x = dollars_x + dollars_w + cents_gap

    sign_baseline = dollars_baseline + sign_rise
    cents_baseline = dollars_baseline + cents_rise

    object_right_x = object_left + object_w
    object_top_y = dollars_baseline + object_ascent

    return _PriceObjectLayout(
        dollar_sign_x=dollar_sign_x,
        dollar_sign_baseline_y=sign_baseline,
        dollar_sign_size=sign_size,
        dollars_x=dollars_x,
        dollars_baseline_y=dollars_baseline,
        dollars_size=dollars_size,
        cents_x=cents_x,
        cents_baseline_y=cents_baseline,
        cents_size=cents_size,
        object_left_x=object_left,
        object_bottom_y=object_bottom_y,
        object_right_x=object_right_x,
        object_top_y=object_top_y,
    )


def _ensure_playwright_chromium_installed() -> str | None:
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            text=True,
        )
        return "Playwright Chromium install/check completed successfully."
    except Exception as exc:
        return f"Playwright Chromium install failed: {exc}"


def _merge_pdf_bytes(pdf_pages: list[bytes]) -> bytes:
    """
    Merge multiple PDF bytes into a single valid PDF document.

    Args:
        pdf_pages: List of PDF byte strings, one per page.

    Returns:
        Merged PDF bytes.

    Raises:
        ValueError: If pdf_pages is empty or if merge fails.
    """
    if not pdf_pages:
        raise ValueError("Cannot merge empty PDF list")

    merged = fitz.open()
    try:
        for pdf_bytes in pdf_pages:
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                merged.insert_pdf(src)
            finally:
                src.close()

        return merged.tobytes()
    finally:
        merged.close()


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

    warnings.append(f"HTML renderer received {len(strip_rows)} strip rows.")

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
        pdf_bytes, rendered_pages, rendered_segments = asyncio.run(
            _render_strips_async(htmls, strip_rows, warnings)
        )
    except Exception as exc:
        warnings.append(f"HTML/Playwright renderer failed: {exc}")
        pdf_bytes = b""
        rendered_pages = 0
        rendered_segments = 0

    return SamsPriceStripPdfResult(
        pdf_bytes=pdf_bytes,
        rendered_pages=rendered_pages,
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
    warnings.append(f"HTML renderer built {len(htmls)} row HTML documents.")
    return htmls


async def _render_strips_async(
    htmls: list[str],
    strip_rows: list[SamsPriceStripRow],
    warnings: list[str],
) -> tuple[bytes, int, int]:
    """Render strips asynchronously using Playwright and merge into one PDF."""
    rendered_segments = 0
    page_pdfs: list[bytes] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            for idx, row_data in enumerate(strip_rows):
                html_content = htmls[idx]
                strip_w, strip_h, _footer_h = compute_strip_canvas(row_data, warnings)
                page_pdf = await _render_page_to_pdf(browser, html_content, strip_w, strip_h)
                page_pdfs.append(page_pdf)
                rendered_segments += len(row_data.segments)
        finally:
            await browser.close()

    warnings.append(f"HTML renderer rendered {len(page_pdfs)} row PDF pages.")

    try:
        merged_pdf_bytes = _merge_pdf_bytes(page_pdfs)
        merged_doc = fitz.open(stream=merged_pdf_bytes, filetype="pdf")
        try:
            page_count = merged_doc.page_count
            warnings.append(f"HTML renderer final merged PDF page count: {page_count}")

            if page_count != len(strip_rows):
                warnings.append(
                    f"ERROR: expected {len(strip_rows)} pages but merged PDF has {page_count} pages."
                )

            rendered_pages = page_count
        finally:
            merged_doc.close()

    except Exception as exc:
        warnings.append(f"PDF merge failed: {exc}")
        return b"", 0, rendered_segments

    return merged_pdf_bytes, rendered_pages, rendered_segments


async def _render_page_to_pdf(
    browser: Browser,
    html_content: str,
    width: float,
    height: float,
) -> bytes:
    """
    Render HTML/SVG content to PDF using Playwright.

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
        await page.evaluate("() => document.fonts.ready")
        pdf_bytes = await page.pdf(
            width=f"{width / inch}in",
            height=f"{height / inch}in",
            print_background=True,
        )
        return pdf_bytes
    finally:
        await page.close()


def _generate_strip_html(row_data: SamsPriceStripRow, strip_w: float, strip_h: float, footer_h: float, warnings: list[str]) -> str:
    """
    Generate HTML containing div-based price strip with Gibson fonts.
    """
    root_path = Path(__file__).resolve().parents[2]
    gibson_regular_path = root_path / "assets" / "Gibson-Regular.otf"
    gibson_semibold_path = root_path / "assets" / "Gibson-SemiBold.otf"

    try:
        regular_data_uri = _font_file_to_data_uri(gibson_regular_path)
        semibold_data_uri = _font_file_to_data_uri(gibson_semibold_path)
        fonts_available = True
    except Exception as exc:
        regular_data_uri = ""
        semibold_data_uri = ""
        fonts_available = False
        warnings.append(f"Gibson font data URI load failed in HTML renderer: {exc}")

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
        if idx >= len(positions):
            break
        x, ticket_w = positions[idx]
        ticket_htmls.append(_generate_ticket_html(segment, x, ticket_y, ticket_w, ticket_h, strip_h))

    footer_text = _resolve_strip_footer_text(row_data)

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Sam's Club Price Strip</title>",
        "<style>",
        font_face_css,
        f"""
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body {{
    width: {strip_w}pt;
    height: {strip_h}pt;
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: white;
    font-family: "Gibson", Arial, sans-serif;
}}

.ticket {{
    position: absolute;
    overflow: hidden;
    font-family: "Gibson", Arial, sans-serif;
    color: black;
}}

.ticket-text-stack {{
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    overflow: hidden;
}}

.brand {{
    display: block;
    width: 100%;
    font-family: "Gibson", Arial, sans-serif;
    font-weight: 600;
    font-size: {_SAMS_BRAND_SIZE}pt;
    line-height: 1.0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: clip;
    letter-spacing: 0;
    color: black;
}}

.desc {{
    display: block;
    width: 100%;
    font-family: "Gibson", Arial, sans-serif;
    font-weight: 400;
    font-size: {_SAMS_DESC_SIZE}pt;
    line-height: 1.0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: clip;
    letter-spacing: 0;
    color: black;
}}

.price {{
    position: absolute;
    display: flex;
    align-items: flex-start;
    white-space: nowrap;
    line-height: 1;
    font-family: "Gibson", Arial, sans-serif;
    font-weight: 600;
    color: black;
}}

.dollar-sign {{
    display: inline-block;
    font-weight: 600;
    font-size: 15.5pt;
    line-height: 1;
    margin-right: 1.0pt;
    transform: translateY(13.5pt);
    color: black;
}}

.dollars {{
    display: inline-block;
    font-weight: 600;
    font-size: 44pt;
    line-height: 0.82;
    letter-spacing: -0.9pt;
    color: black;
}}

.cents {{
    display: inline-block;
    font-weight: 600;
    font-size: 18.5pt;
    line-height: 1;
    margin-left: 0.3pt;
    transform: translateY(2.8pt);
    letter-spacing: -0.4pt;
    color: black;
}}

.item-number {{
    position: absolute;
    font-weight: 400;
    font-size: {_SAMS_ITEM_SIZE}pt;
    line-height: 1;
    white-space: nowrap;
    text-align: right;
    color: black;
}}

.footer {{
    position: absolute;
    left: 0.08in;
    bottom: 0.055in;
    font-family: "Gibson", Arial, sans-serif;
    font-weight: 400;
    font-size: {_SAMS_FOOTER_SIZE}pt;
    line-height: 1;
    color: #303030;
    white-space: nowrap;
    letter-spacing: 0;
}}
        """,
        "</style>",
        "</head>",
        "<body>",
        "\n".join(ticket_htmls),
        f'<div class="footer">{html.escape(footer_text)}</div>',
        "</body>",
        "</html>",
    ]

    return "\n".join(html_parts)


def _generate_ticket_html(
    segment: SamsPriceStripSegment,
    x: float,
    y: float,
    w: float,
    h: float,
    page_h: float,
) -> str:
    """
    Generate HTML divs for one ticket block with fixed layout positions.
    """
    dollars, cents = _normalize_price_parts(segment.retail)

    # Fixed layout constants for consistent positioning
    TEXT_TOP_PT = 4.2
    DESC_1_MARGIN_TOP_PT = 0.6
    DESC_2_MARGIN_TOP_PT = 0.4

    PRICE_TOP_PT = 18.0
    PRICE_LEFT_PT = max(_RETAIL_MARGIN_PAD, _DEFAULT_INNER_PAD_X * 0.35)

    ITEM_TOP_PT = 51.0
    ITEM_RIGHT_PAD_PT = _DEFAULT_INNER_PAD_X

    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))

    text_x = pad_x
    text_y = TEXT_TOP_PT
    text_w = max(8.0, w - (2 * pad_x))

    price_x = PRICE_LEFT_PT
    price_y = PRICE_TOP_PT
    price_box_w = max(20.0, w - price_x - pad_x)
    price_box_h = 44.0

    item_w = min(max(34.0, price_box_w * 0.58), w - pad_x)
    item_x = w - pad_x - item_w
    item_y = ITEM_TOP_PT

    # Truncate texts
    brand = _truncate_svg_text(segment.brand or "-", _SAMS_BRAND_SIZE, text_w, "semibold")
    desc_1 = _truncate_svg_text(segment.desc_1 or "-", _SAMS_DESC_SIZE, text_w, "regular")
    desc_2 = _truncate_svg_text(segment.desc_2 or "-", _SAMS_DESC_SIZE, text_w, "regular")
    item_number = _truncate_svg_text(segment.item_number or "-", _SAMS_ITEM_SIZE, item_w, "regular")

    ticket_html = f"""
<div class="ticket" style="left: {x}pt; top: {y}pt; width: {w}pt; height: {h}pt;">
    <div class="ticket-text-stack" style="left: {text_x}pt; top: {text_y}pt; width: {text_w}pt;">
        <div class="brand">{html.escape(brand)}</div>
        <div class="desc" style="margin-top: {DESC_1_MARGIN_TOP_PT}pt;">{html.escape(desc_1)}</div>
        <div class="desc" style="margin-top: {DESC_2_MARGIN_TOP_PT}pt;">{html.escape(desc_2)}</div>
    </div>

    <div class="price" style="left: {price_x}pt; top: {price_y}pt; width: {price_box_w}pt; height: {price_box_h}pt;">
        <span class="dollar-sign">$</span>
        <span class="dollars">{html.escape(dollars)}</span>
        <span class="cents">{html.escape(cents)}</span>
    </div>

    <div class="item-number" style="left: {item_x}pt; top: {item_y}pt; width: {item_w}pt;">
        {html.escape(item_number)}
    </div>
</div>
"""

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
