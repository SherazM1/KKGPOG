from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow
from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT, TITLE_FONT

_PAGE_WIDTH = 11.0 * inch
_PAGE_HEIGHT = 3.10 * inch
_MARGIN_X = 18.0
_MARGIN_TOP = 14.0
_MARGIN_BOTTOM = 16.0
_FOOTER_HEIGHT = 17.0
_SEGMENT_GAP = 0.0
_SEGMENT_BORDER = (0.20, 0.20, 0.20)
_PRICE_BG = (0.965, 0.965, 0.965)
_TEXT_DARK = (0.10, 0.10, 0.10)
_TEXT_MUTED = (0.34, 0.34, 0.34)


def _fit_text(text: str, font_name: str, max_width: float, max_size: float, min_size: float = 6.0) -> float:
    if max_width <= 0:
        return min_size
    size = max_size
    while size > min_size and pdfmetrics.stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    return max(size, min_size)


def _truncate(text: str, font_name: str, font_size: float, max_width: float) -> str:
    value = (text or "").strip()
    if value == "":
        return ""
    if pdfmetrics.stringWidth(value, font_name, font_size) <= max_width:
        return value
    suffix = "..."
    width_limit = max_width - pdfmetrics.stringWidth(suffix, font_name, font_size)
    if width_limit <= 0:
        return suffix
    out = value
    while out and pdfmetrics.stringWidth(out, font_name, font_size) > width_limit:
        out = out[:-1]
    return f"{out}{suffix}"


def _strip_price_digits(raw_retail: str) -> str:
    text = (raw_retail or "").strip()
    if text == "":
        return "-"
    normalized = text.replace("$", "").replace(",", "").strip()
    try:
        amount = float(normalized)
        cents = int(round(amount * 100))
        return str(cents)
    except ValueError:
        digits = "".join(ch for ch in normalized if ch.isdigit())
        return digits or normalized


def _draw_strip_header(c: canvas.Canvas, page_w: float, page_h: float, row_data: SamsPriceStripRow) -> None:
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(TITLE_FONT, 8.5)
    c.drawString(_MARGIN_X, page_h - 10, f"POG {row_data.pog}  |  SIDE {row_data.side}  |  ROW {row_data.row}")


def _draw_segment(c: canvas.Canvas, x: float, y: float, w: float, h: float, row_data: SamsPriceStripRow, idx: int) -> None:
    segment = row_data.segments[idx]
    c.setStrokeColorRGB(*_SEGMENT_BORDER)
    c.setLineWidth(0.75)
    c.rect(x, y, w, h, stroke=1, fill=0)

    price_h = h * 0.40
    body_y = y
    body_h = h - price_h
    price_y = y + body_h

    c.setFillColorRGB(*_PRICE_BG)
    c.rect(x, price_y, w, price_h, stroke=0, fill=1)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(BODY_BOLD_FONT, 14)
    c.drawString(x + 5, price_y + price_h - 17, "$")

    price_digits = _strip_price_digits(segment.retail)
    digit_size = _fit_text(price_digits, BODY_BOLD_FONT, w - 20, 24, 11)
    c.setFont(BODY_BOLD_FONT, digit_size)
    c.drawString(x + 16, price_y + (price_h / 2) - (digit_size / 2) + 2, price_digits)

    pad_x = 4.0
    line_y = body_y + body_h - 9.0
    max_w = w - (2 * pad_x)

    brand = _truncate(segment.brand or "-", BODY_BOLD_FONT, 6.8, max_w)
    item = _truncate(segment.item_number or "-", BODY_FONT, 6.4, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", BODY_FONT, 6.2, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", BODY_FONT, 6.2, max_w)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(BODY_BOLD_FONT, 6.8)
    c.drawString(x + pad_x, line_y, brand)
    c.setFont(BODY_FONT, 6.4)
    c.drawString(x + pad_x, line_y - 8.0, item)
    c.setFont(BODY_FONT, 6.2)
    c.drawString(x + pad_x, line_y - 15.5, desc_1)
    c.drawString(x + pad_x, line_y - 23.0, desc_2)


def _draw_footer(c: canvas.Canvas, page_w: float, row_data: SamsPriceStripRow, generated_by: str, generated_at: str) -> None:
    footer_y = _MARGIN_BOTTOM - 2
    c.setStrokeColorRGB(0.85, 0.85, 0.85)
    c.setLineWidth(0.7)
    c.line(_MARGIN_X, footer_y + 10, page_w - _MARGIN_X, footer_y + 10)

    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 8.0)
    footer_text = row_data.footer_text.strip() or f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"
    c.drawString(_MARGIN_X, footer_y, footer_text)
    c.drawRightString(page_w - _MARGIN_X, footer_y, f"{generated_by} | {generated_at}")


def _render_strip_page(
    c: canvas.Canvas,
    row_data: SamsPriceStripRow,
    generated_by: str,
    generated_at: str,
    warnings: list[str],
) -> int:
    page_w = _PAGE_WIDTH
    page_h = _PAGE_HEIGHT
    c.setPageSize((page_w, page_h))
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

    _draw_strip_header(c, page_w, page_h, row_data)

    segment_count = len(row_data.segments)
    if segment_count <= 0:
        warnings.append(f"Skipping empty strip group: POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}.")
        _draw_footer(c, page_w, row_data, generated_by, generated_at)
        return 0

    strip_top = page_h - _MARGIN_TOP - 8
    strip_bottom = _MARGIN_BOTTOM + _FOOTER_HEIGHT
    strip_h = max(10.0, strip_top - strip_bottom)
    strip_y = strip_bottom
    strip_w = page_w - (2 * _MARGIN_X)
    segment_w = (strip_w - ((segment_count - 1) * _SEGMENT_GAP)) / segment_count
    if segment_w <= 8.0:
        warnings.append(
            f"Narrow strip segments for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; count={segment_count}."
        )

    for idx in range(segment_count):
        x = _MARGIN_X + idx * (segment_w + _SEGMENT_GAP)
        _draw_segment(c, x, strip_y, segment_w, strip_h, row_data, idx)

    _draw_footer(c, page_w, row_data, generated_by, generated_at)
    return segment_count


def render_sams_price_strips_pdf(
    strip_rows: list[SamsPriceStripRow],
    generated_by: str = "Kendal King",
) -> SamsPriceStripPdfResult:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    warnings: list[str] = []
    rendered_pages = 0
    rendered_segments = 0
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for row_data in strip_rows:
        rendered_segments += _render_strip_page(c, row_data, generated_by, generated_at, warnings)
        rendered_pages += 1
        c.showPage()

    c.save()
    return SamsPriceStripPdfResult(
        pdf_bytes=buffer.getvalue(),
        rendered_pages=rendered_pages,
        rendered_segments=rendered_segments,
        warnings=warnings,
    )
