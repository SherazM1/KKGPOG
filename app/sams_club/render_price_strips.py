from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow
from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT, TITLE_FONT

_PAGE_WIDTH = 11.0 * inch
_PAGE_HEIGHT = 2.45 * inch
_MARGIN_X = 12.0
_MARGIN_TOP = 8.0
_MARGIN_BOTTOM = 8.0
_FOOTER_HEIGHT = 10.0
_SEGMENT_GAP = 0.35
_SEGMENT_BORDER = (0.20, 0.20, 0.20)
_PRICE_BG = (0.965, 0.965, 0.965)
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
    c.setFont(TITLE_FONT, 7.2)
    c.drawString(_MARGIN_X, page_h - 7.2, f"POG {row_data.pog}  |  SIDE {row_data.side}  |  ROW {row_data.row}")


def _draw_corner_guides(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
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


def _draw_segment_item_number(c: canvas.Canvas, item_number: str, x: float, y: float, w: float, price_h: float) -> None:
    text = _truncate(item_number or "-", BODY_FONT, 5.9, w * 0.42)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 5.9)
    c.drawRightString(x + w - 3.4, y + price_h - 8.6, text)


def _draw_segment_text_stack(c: canvas.Canvas, row_data: SamsPriceStripRow, idx: int, x: float, y: float, w: float, h: float) -> None:
    segment = row_data.segments[idx]
    pad_x = max(2.6, min(4.0, w * 0.08))
    stack_top = y + h - 8.0
    max_w = max(8.0, w - (2 * pad_x))

    if w >= 52:
        brand_fs = 6.8
        desc1_fs = 6.35
        desc2_fs = 6.2
    elif w >= 42:
        brand_fs = 6.2
        desc1_fs = 5.85
        desc2_fs = 5.7
    else:
        brand_fs = 5.7
        desc1_fs = 5.45
        desc2_fs = 5.25

    brand = _truncate(segment.brand or "-", BODY_BOLD_FONT, brand_fs, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", BODY_FONT, desc1_fs, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", BODY_FONT, desc2_fs, max_w)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(BODY_BOLD_FONT, brand_fs)
    c.drawString(x + pad_x, stack_top - brand_fs, brand)
    c.setFont(BODY_FONT, desc1_fs)
    c.drawString(x + pad_x, stack_top - brand_fs - 1.5 - desc1_fs, desc_1)
    c.setFont(BODY_FONT, desc2_fs)
    c.drawString(x + pad_x, stack_top - brand_fs - 1.5 - desc1_fs - 1.35 - desc2_fs, desc_2)


def _draw_segment(c: canvas.Canvas, x: float, y: float, w: float, h: float, row_data: SamsPriceStripRow, idx: int) -> None:
    segment = row_data.segments[idx]
    c.setStrokeColorRGB(*_SEGMENT_BORDER)
    c.setLineWidth(0.65)
    c.rect(x, y, w, h, stroke=1, fill=0)

    price_h = h * 0.57
    text_h = h - price_h
    price_y = y

    c.setFillColorRGB(*_PRICE_BG)
    c.rect(x, price_y, w, price_h, stroke=0, fill=1)

    c.setFillColorRGB(*_TEXT_DARK)
    dollar_fs = 11 if w < 45 else 12
    c.setFont(BODY_BOLD_FONT, dollar_fs)
    c.drawString(x + 3.2, price_y + price_h - (dollar_fs + 2.4), "$")

    price_digits = _strip_price_digits(segment.retail)
    digit_size = _fit_text(price_digits, BODY_BOLD_FONT, w - 13, 22, 9)
    c.setFont(BODY_BOLD_FONT, digit_size)
    c.drawString(x + 11.0, price_y + (price_h / 2) - (digit_size / 2) - 1.1, price_digits)
    _draw_segment_item_number(c, segment.item_number, x, price_y, w, price_h)
    _draw_segment_text_stack(c, row_data, idx, x, y + price_h, w, text_h)


def _draw_strip_footer(c: canvas.Canvas, row_data: SamsPriceStripRow, strip_x: float, strip_y: float) -> None:
    footer_text = row_data.footer_text.strip()
    if footer_text == "":
        footer_text = f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 6.4)
    c.drawString(strip_x + 2.2, strip_y - 6.8, _truncate(footer_text, BODY_FONT, 6.4, _PAGE_WIDTH * 0.66))


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
        return 0

    strip_top = page_h - _MARGIN_TOP - 5.2
    strip_bottom = _MARGIN_BOTTOM + _FOOTER_HEIGHT
    strip_h = max(10.0, strip_top - strip_bottom)
    strip_y = strip_bottom
    strip_w = page_w - (2 * _MARGIN_X)
    gap = _SEGMENT_GAP if segment_count < 18 else 0.0
    segment_w = (strip_w - ((segment_count - 1) * gap)) / segment_count
    if segment_w <= 7.0:
        warnings.append(
            f"Tight segment width for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; count={segment_count}."
        )

    _draw_corner_guides(c, _MARGIN_X, strip_y, strip_w, strip_h)

    for idx in range(segment_count):
        x = _MARGIN_X + idx * (segment_w + gap)
        _draw_segment(c, x, strip_y, segment_w, strip_h, row_data, idx)

    _draw_strip_footer(c, row_data, _MARGIN_X, strip_y)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(BODY_FONT, 5.5)
    c.drawRightString(page_w - _MARGIN_X, 2.0, f"{generated_by} | {generated_at}")
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
