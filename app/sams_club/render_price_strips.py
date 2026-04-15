from __future__ import annotations

import io
from decimal import Decimal, InvalidOperation

from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow
from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT

_PAGE_WIDTH = 11.0 * inch
_PAGE_HEIGHT = 2.45 * inch
_MARGIN_X = 12.0
_MARGIN_TOP = 8.0
_MARGIN_BOTTOM = 8.0
_FOOTER_HEIGHT = 10.0
_TICKET_MIN_GAP = 3.0
_PRICE_BG = (0.955, 0.955, 0.955)
_TEXT_DARK = (0.10, 0.10, 0.10)
_TEXT_MUTED = (0.34, 0.34, 0.34)
_GUIDE_COLOR = (0.45, 0.45, 0.45)
_GUIDE_LEN = 8.0
_GUIDE_INSET = 1.75


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


def draw_strip_guides(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.setStrokeColorRGB(*_GUIDE_COLOR)
    c.setLineWidth(0.6)
    gi = _GUIDE_INSET
    gl = _GUIDE_LEN
    c.line(x + gi, y + h - gi, x + gi + gl, y + h - gi)
    c.line(x + gi, y + h - gi, x + gi, y + h - gi - gl)
    c.line(x + w - gi - gl, y + h - gi, x + w - gi, y + h - gi)
    c.line(x + w - gi, y + h - gi, x + w - gi, y + h - gi - gl)
    c.line(x + gi, y + gi, x + gi + gl, y + gi)
    c.line(x + gi, y + gi, x + gi, y + gi + gl)
    c.line(x + w - gi - gl, y + gi, x + w - gi, y + gi)
    c.line(x + w - gi, y + gi, x + w - gi, y + gi + gl)


def draw_ticket_item_number(c: canvas.Canvas, item_number: str, x: float, y: float, w: float, price_h: float) -> None:
    text = _truncate(item_number or "-", BODY_FONT, 5.5, w * 0.46)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 5.5)
    c.drawRightString(x + w - 2.5, y + price_h - 6.6, text)


def _draw_ticket_text_stack(c: canvas.Canvas, row_data: SamsPriceStripRow, idx: int, x: float, y: float, w: float, h: float) -> None:
    segment = row_data.segments[idx]
    pad_x = max(2.5, min(4.2, w * 0.09))
    stack_top = y + h - 3.6
    max_w = max(8.0, w - (2 * pad_x))

    if w >= 58:
        brand_fs = 7.0
        desc1_fs = 6.4
        desc2_fs = 6.3
    elif w >= 46:
        brand_fs = 6.45
        desc1_fs = 6.0
        desc2_fs = 5.85
    else:
        brand_fs = 5.9
        desc1_fs = 5.55
        desc2_fs = 5.35

    brand = _truncate(segment.brand or "-", BODY_BOLD_FONT, brand_fs, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", BODY_FONT, desc1_fs, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", BODY_FONT, desc2_fs, max_w)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(BODY_BOLD_FONT, brand_fs)
    c.drawString(x + pad_x, stack_top - brand_fs, brand)
    c.setFont(BODY_FONT, desc1_fs)
    c.drawString(x + pad_x, stack_top - brand_fs - 1.3 - desc1_fs, desc_1)
    c.setFont(BODY_FONT, desc2_fs)
    c.drawString(x + pad_x, stack_top - brand_fs - 1.3 - desc1_fs - 1.15 - desc2_fs, desc_2)


def draw_ticket_price(c: canvas.Canvas, retail: str, x: float, y: float, w: float, h: float) -> float:
    c.setFillColorRGB(*_PRICE_BG)
    c.roundRect(x, y, w, h, 1.25, stroke=0, fill=1)
    dollars, cents = _normalize_price_parts(retail)
    max_dollar_width = max(14.0, w - 18.0)

    c.setFillColorRGB(*_TEXT_DARK)
    dollar_sign_fs = 11.5 if w >= 44 else 10.6
    c.setFont(BODY_BOLD_FONT, dollar_sign_fs)
    c.drawString(x + 2.6, y + h - (dollar_sign_fs + 2.2), "$")

    dollars_size = _fit_text(dollars, BODY_BOLD_FONT, max_dollar_width, 34.0 if w >= 58 else 30.0, 9.5)
    base_y = y + max(1.2, (h * 0.50) - (dollars_size * 0.37))
    c.setFont(BODY_BOLD_FONT, dollars_size)
    c.drawString(x + 11.6, base_y, dollars)

    cents_fs = max(7.0, dollars_size * 0.46)
    cents_x = x + 11.6 + pdfmetrics.stringWidth(dollars, BODY_BOLD_FONT, dollars_size) + 1.2
    cents_y = base_y + (dollars_size * 0.42)
    c.setFont(BODY_BOLD_FONT, cents_fs)
    c.drawString(cents_x, cents_y, cents)
    return h


def draw_ticket_block(c: canvas.Canvas, x: float, y: float, w: float, h: float, row_data: SamsPriceStripRow, idx: int) -> None:
    segment = row_data.segments[idx]
    text_h = h * 0.33
    price_h = h - text_h
    _draw_ticket_text_stack(c, row_data, idx, x, y + price_h, w, text_h)
    price_block_h = draw_ticket_price(c, segment.retail, x, y, w, price_h)
    draw_ticket_item_number(c, segment.item_number, x, y, w, price_block_h)


def draw_strip_footer(c: canvas.Canvas, row_data: SamsPriceStripRow, strip_x: float, strip_y: float, max_width: float) -> None:
    footer_text = row_data.footer_text.strip()
    if footer_text == "":
        footer_text = f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 6.0)
    c.drawString(strip_x + 1.9, strip_y - 6.1, _truncate(footer_text, BODY_FONT, 6.0, max_width))


def _render_strip_page(
    c: canvas.Canvas,
    row_data: SamsPriceStripRow,
    warnings: list[str],
) -> int:
    page_w = _PAGE_WIDTH
    page_h = _PAGE_HEIGHT
    c.setPageSize((page_w, page_h))
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

    segment_count = len(row_data.segments)
    if segment_count <= 0:
        warnings.append(f"Skipping empty strip group: POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}.")
        return 0

    strip_top = page_h - _MARGIN_TOP
    strip_bottom = _MARGIN_BOTTOM + _FOOTER_HEIGHT
    strip_h = max(10.0, strip_top - strip_bottom)
    strip_y = strip_bottom
    strip_w = page_w - (2 * _MARGIN_X)
    gap = _TICKET_MIN_GAP
    ticket_w = (strip_w - ((segment_count - 1) * gap)) / segment_count
    if ticket_w < 7.0:
        gap = 0.0
        ticket_w = strip_w / segment_count
    if ticket_w <= 7.0:
        warnings.append(
            f"Tight segment width for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; count={segment_count}."
        )

    draw_strip_guides(c, _MARGIN_X, strip_y, strip_w, strip_h)

    for idx in range(segment_count):
        x = _MARGIN_X + idx * (ticket_w + gap)
        draw_ticket_block(c, x, strip_y, ticket_w, strip_h, row_data, idx)

    draw_strip_footer(c, row_data, _MARGIN_X, strip_y, strip_w * 0.78)
    return segment_count


def render_sams_price_strips_pdf(
    strip_rows: list[SamsPriceStripRow],
    generated_by: str = "Kendal King",
) -> SamsPriceStripPdfResult:
    _ = generated_by
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    warnings: list[str] = []
    rendered_pages = 0
    rendered_segments = 0
    for row_data in strip_rows:
        rendered_segments += _render_strip_page(c, row_data, warnings)
        rendered_pages += 1
        c.showPage()

    c.save()
    return SamsPriceStripPdfResult(
        pdf_bytes=buffer.getvalue(),
        rendered_pages=rendered_pages,
        rendered_segments=rendered_segments,
        warnings=warnings,
    )
