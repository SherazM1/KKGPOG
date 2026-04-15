from __future__ import annotations

import io
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.sams_club.price_strip_models import SamsPriceStripPdfResult, SamsPriceStripRow, SamsPriceStripSegment
from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT

_PAGE_WIDTH = 11.0 * inch
_PAGE_HEIGHT = 2.45 * inch
_TEXT_DARK = (0.10, 0.10, 0.10)
_TEXT_MUTED = (0.34, 0.34, 0.34)
_STRIP_BG = (1.0, 1.0, 1.0)
_HAIRLINE = (0.74, 0.74, 0.74)
_RETAIL_MARGIN_PAD = 1.2
_DEFAULT_INNER_PAD_X = 0.055 * inch
_DEFAULT_INNER_PAD_TOP = 0.045 * inch
_DEFAULT_INNER_PAD_BOTTOM = 0.05 * inch
_DEFAULT_FOOTER_HEIGHT = 0.14 * inch
_DEFAULT_TICKET_GAP = 0.035 * inch
_MIN_TICKET_GAP = 0.0
_MIN_TICKET_WIDTH = 0.70 * inch
_SAMS_FONT_REGULAR = "Sams-Gibson-Regular"
_SAMS_FONT_SEMIBOLD = "Sams-Gibson-SemiBold"
_SAMS_BRAND_SIZE = 7.5
_SAMS_DESC_SIZE = 6.0
_SAMS_PRICE_SIZE = 47.0
_SAMS_ITEM_SIZE = 5.0
_SAMS_FOOTER_SIZE = 5.0
_FONTS_READY = False
_SAMS_GIBSON_AVAILABLE = False


def register_sams_strip_fonts() -> None:
    global _FONTS_READY, _SAMS_GIBSON_AVAILABLE
    if _FONTS_READY:
        return
    root = Path(__file__).resolve().parents[2]
    regular_path = root / "assets" / "Gibson-Regular.otf"
    semibold_path = root / "assets" / "Gibson-SemiBold.otf"
    try:
        if regular_path.is_file():
            pdfmetrics.registerFont(TTFont(_SAMS_FONT_REGULAR, str(regular_path)))
    except Exception:
        pass
    try:
        if semibold_path.is_file():
            pdfmetrics.registerFont(TTFont(_SAMS_FONT_SEMIBOLD, str(semibold_path)))
    except Exception:
        pass
    try:
        pdfmetrics.getFont(_SAMS_FONT_REGULAR)
        pdfmetrics.getFont(_SAMS_FONT_SEMIBOLD)
        _SAMS_GIBSON_AVAILABLE = True
    except Exception:
        _SAMS_GIBSON_AVAILABLE = False
    _FONTS_READY = True


def get_sams_strip_font(weight: str) -> str:
    register_sams_strip_fonts()
    preferred = _SAMS_FONT_SEMIBOLD if weight == "semibold" else _SAMS_FONT_REGULAR
    fallback = BODY_BOLD_FONT if weight == "semibold" else BODY_FONT
    try:
        pdfmetrics.getFont(preferred)
        return preferred
    except Exception:
        return fallback


def sams_gibson_available() -> bool:
    register_sams_strip_fonts()
    return _SAMS_GIBSON_AVAILABLE


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


def compute_ticket_positions(strip_w: float, ticket_count: int) -> list[tuple[float, float]]:
    if ticket_count <= 0:
        return []

    side_pad = max(0.05 * inch, min(0.16 * inch, strip_w * 0.012))
    gap = _DEFAULT_TICKET_GAP
    usable_w = strip_w - (2 * side_pad) - ((ticket_count - 1) * gap)
    ticket_w = usable_w / ticket_count

    if ticket_w < _MIN_TICKET_WIDTH:
        gap = _MIN_TICKET_GAP
        usable_w = strip_w - (2 * side_pad)
        ticket_w = usable_w / ticket_count

    if ticket_w < _MIN_TICKET_WIDTH:
        side_pad = max(0.0, (strip_w - (ticket_count * _MIN_TICKET_WIDTH)) / 2.0)
        ticket_w = (strip_w - (2 * side_pad)) / ticket_count

    positions: list[tuple[float, float]] = []
    x = side_pad
    for _ in range(ticket_count):
        positions.append((x, ticket_w))
        x += ticket_w + gap
    return positions


def draw_ticket_item_number(c: canvas.Canvas, item_number: str, x: float, y: float, w: float) -> None:
    font = get_sams_strip_font("regular")
    text = _truncate(item_number or "-", font, _SAMS_ITEM_SIZE, max(20.0, w * 0.62))
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(font, _SAMS_ITEM_SIZE)
    c.drawRightString(x + w - _RETAIL_MARGIN_PAD, y + 1.1, text)


def _draw_ticket_text_stack(c: canvas.Canvas, segment: SamsPriceStripSegment, x: float, y: float, w: float, h: float) -> None:
    brand_font = get_sams_strip_font("semibold")
    desc_font = get_sams_strip_font("regular")
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.055), max(_DEFAULT_INNER_PAD_X, w * 0.10))
    stack_top = y + h - _DEFAULT_INNER_PAD_TOP
    max_w = max(8.0, w - (2 * pad_x))

    brand = _truncate(segment.brand or "-", brand_font, _SAMS_BRAND_SIZE, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", desc_font, _SAMS_DESC_SIZE, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", desc_font, _SAMS_DESC_SIZE, max_w)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(brand_font, _SAMS_BRAND_SIZE)
    c.drawString(x + pad_x, stack_top - _SAMS_BRAND_SIZE, brand)
    c.setFont(desc_font, _SAMS_DESC_SIZE)
    c.drawString(x + pad_x, stack_top - _SAMS_BRAND_SIZE - 1.35 - _SAMS_DESC_SIZE, desc_1)
    c.setFont(desc_font, _SAMS_DESC_SIZE)
    c.drawString(x + pad_x, stack_top - _SAMS_BRAND_SIZE - 1.35 - _SAMS_DESC_SIZE - 1.15 - _SAMS_DESC_SIZE, desc_2)


def draw_ticket_price(c: canvas.Canvas, retail: str, x: float, y: float, w: float, h: float) -> None:
    price_font = get_sams_strip_font("semibold")
    dollars, cents = _normalize_price_parts(retail)
    max_dollar_width = max(12.0, w - 17.0)

    c.setFillColorRGB(*_TEXT_DARK)
    dollars_size = _fit_text(dollars, price_font, max_dollar_width, _SAMS_PRICE_SIZE, 12.0)
    dollar_sign_fs = max(8.0, min(18.0, _SAMS_PRICE_SIZE * 0.33))
    c.setFont(price_font, dollar_sign_fs)
    c.drawString(x + _RETAIL_MARGIN_PAD, y + h - (dollar_sign_fs + 1.6), "$")

    base_y = y + max(1.0, (h * 0.16) - (dollars_size * 0.05))
    c.setFont(price_font, dollars_size)
    c.drawString(x + max(6.0, dollar_sign_fs * 0.82), base_y, dollars)

    cents_fs = max(10.0, min(21.0, _SAMS_PRICE_SIZE * 0.40))
    dollars_w = pdfmetrics.stringWidth(dollars, price_font, dollars_size)
    cents_x = x + max(6.0, dollar_sign_fs * 0.82) + dollars_w + 0.8
    cents_y = base_y + (dollars_size * 0.50)
    c.setFont(price_font, cents_fs)
    c.drawString(cents_x, cents_y, cents)


def draw_ticket_block(c: canvas.Canvas, segment: SamsPriceStripSegment, x: float, y: float, w: float, h: float) -> None:
    content_bottom = y + _DEFAULT_INNER_PAD_BOTTOM
    content_h = max(12.0, h - _DEFAULT_INNER_PAD_BOTTOM - _DEFAULT_INNER_PAD_TOP)
    text_h = content_h * 0.34
    price_h = max(20.0, content_h - text_h)
    _draw_ticket_text_stack(c, segment, x, content_bottom + price_h, w, text_h)
    draw_ticket_price(c, segment.retail, x, content_bottom, w, price_h)
    draw_ticket_item_number(c, segment.item_number, x, content_bottom, w)


def draw_strip_footer(c: canvas.Canvas, row_data: SamsPriceStripRow, y: float, max_width: float) -> None:
    font = get_sams_strip_font("regular")
    footer_text = row_data.footer_text.strip()
    if footer_text == "":
        footer_text = f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(font, _SAMS_FOOTER_SIZE)
    c.drawString(0.04 * inch, y + 1.2, _truncate(footer_text, font, _SAMS_FOOTER_SIZE, max_width))


def _render_strip_page(
    c: canvas.Canvas,
    row_data: SamsPriceStripRow,
    warnings: list[str],
) -> int:
    page_w, page_h, footer_h = compute_strip_canvas(row_data, warnings)
    c.setPageSize((page_w, page_h))
    c.setFillColorRGB(*_STRIP_BG)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

    segment_count = len(row_data.segments)
    if segment_count <= 0:
        warnings.append(f"Skipping empty strip group: POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}.")
        return 0

    strip_h = page_h - footer_h
    if strip_h <= 12.0:
        warnings.append(
            f"Very short strip height for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; Length may be invalid."
        )
        strip_h = page_h

    positions = compute_ticket_positions(page_w, segment_count)
    if positions and positions[0][1] < _MIN_TICKET_WIDTH:
        warnings.append(
            f"Tight segment width for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; count={segment_count}."
        )

    ticket_y = footer_h
    for idx, (x, ticket_w) in enumerate(positions):
        draw_ticket_block(c, row_data.segments[idx], x, ticket_y, ticket_w, strip_h)
        if idx > 0:
            line_x = x - (_DEFAULT_TICKET_GAP / 2.0)
            c.setStrokeColorRGB(*_HAIRLINE)
            c.setLineWidth(0.25)
            c.line(line_x, ticket_y + 2.0, line_x, page_h - 2.0)

    draw_strip_footer(c, row_data, 0.0, page_w * 0.72)
    return segment_count


def render_sams_price_strips_pdf(
    strip_rows: list[SamsPriceStripRow],
    generated_by: str = "Kendal King",
) -> SamsPriceStripPdfResult:
    _ = generated_by
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    warnings: list[str] = []
    if not sams_gibson_available():
        warnings.append(
            "Gibson OTF fonts were detected but ReportLab cannot load PostScript-outline OTF in this environment; using fallback fonts."
        )
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
