from __future__ import annotations

import io
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple

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
_RETAIL_MARGIN_PAD = 1.2
_DEFAULT_INNER_PAD_X = 0.055 * inch
_DEFAULT_INNER_PAD_TOP = 0.045 * inch
_DEFAULT_INNER_PAD_BOTTOM = 0.05 * inch
_DEFAULT_FOOTER_HEIGHT = 0.14 * inch
_DEFAULT_TICKET_GAP = 0.02 * inch
_MIN_TICKET_GAP = 0.0
_MIN_TICKET_WIDTH = 0.70 * inch
_STRIP_MARGIN_MIN = 0.07 * inch
_STRIP_MARGIN_MAX = 0.19 * inch
_TICKET_COMPOSITION_RATIO = 0.90
_SAMS_FONT_REGULAR = "Sams-Gibson-Regular"
_SAMS_FONT_SEMIBOLD = "Sams-Gibson-SemiBold"
_SAMS_BRAND_SIZE = 7.5
_SAMS_DESC_SIZE = 6.0
_SAMS_PRICE_SIZE = 47.0
_SAMS_ITEM_SIZE = 5.0
_SAMS_FOOTER_SIZE = 5.0
_FONTS_READY = False
_SAMS_GIBSON_AVAILABLE = False
_SAMS_FONT_WARNING: str | None = None


class _PriceAnchor(NamedTuple):
    text_stack_anchor_top: float
    item_baseline_y: float
    item_right_x: float
    item_max_w: float


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
    object_right_x: float
    object_top_y: float


class _StripContinuityLayout(NamedTuple):
    positions: list[tuple[float, float]]
    left_margin: float
    right_margin: float


def register_sams_strip_fonts() -> None:
    global _FONTS_READY, _SAMS_GIBSON_AVAILABLE, _SAMS_FONT_WARNING
    if _FONTS_READY:
        return
    root = Path(__file__).resolve().parents[2]
    regular_path = root / "assets" / "24354.ttf"
    semibold_path = root / "assets" / "24355.ttf"
    issues: list[str] = []

    if not regular_path.is_file():
        issues.append(f"missing {regular_path.as_posix()}")
    else:
        try:
            pdfmetrics.registerFont(TTFont(_SAMS_FONT_REGULAR, str(regular_path)))
        except Exception as exc:
            issues.append(f"failed to register {regular_path.as_posix()} ({exc})")

    if not semibold_path.is_file():
        issues.append(f"missing {semibold_path.as_posix()}")
    else:
        try:
            pdfmetrics.registerFont(TTFont(_SAMS_FONT_SEMIBOLD, str(semibold_path)))
        except Exception as exc:
            issues.append(f"failed to register {semibold_path.as_posix()} ({exc})")

    try:
        pdfmetrics.getFont(_SAMS_FONT_REGULAR)
        pdfmetrics.getFont(_SAMS_FONT_SEMIBOLD)
        _SAMS_GIBSON_AVAILABLE = True
        _SAMS_FONT_WARNING = None
    except Exception:
        _SAMS_GIBSON_AVAILABLE = False
        detail = "; ".join(issues) if issues else "font names unavailable after registration attempt"
        _SAMS_FONT_WARNING = (
            "Sam's strip Gibson TTF font load failed. "
            "Expected assets/24354.ttf and assets/24355.ttf. "
            f"Using fallback fonts. Details: {detail}"
        )
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


def sams_gibson_warning() -> str | None:
    register_sams_strip_fonts()
    return _SAMS_FONT_WARNING


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


def _compute_strip_margins(strip_w: float, ticket_count: int) -> tuple[float, float]:
    base_margin = max(_STRIP_MARGIN_MIN, min(_STRIP_MARGIN_MAX, strip_w * 0.015))
    if ticket_count <= 2:
        base_margin = min(_STRIP_MARGIN_MAX, base_margin * 1.12)
    elif ticket_count >= 9:
        base_margin = max(_STRIP_MARGIN_MIN, base_margin * 0.92)
    return base_margin, base_margin


def _build_gap_sequence(ticket_count: int, base_gap: float) -> list[float]:
    if ticket_count <= 1:
        return []
    if ticket_count <= 3:
        return [base_gap] * (ticket_count - 1)
    modifiers = [0.05, -0.04, 0.03, -0.03]
    gaps: list[float] = []
    for idx in range(ticket_count - 1):
        m = modifiers[idx % len(modifiers)]
        gaps.append(max(_MIN_TICKET_GAP, base_gap * (1.0 + m)))
    return gaps


def compute_ticket_positions_across_strip(strip_w: float, ticket_count: int) -> _StripContinuityLayout:
    if ticket_count <= 0:
        return _StripContinuityLayout(positions=[], left_margin=0.0, right_margin=0.0)

    left_margin, right_margin = _compute_strip_margins(strip_w, ticket_count)
    gap = _DEFAULT_TICKET_GAP
    gaps = _build_gap_sequence(ticket_count, gap)
    usable_w = strip_w - left_margin - right_margin - sum(gaps)
    slot_w = usable_w / ticket_count
    if slot_w < _MIN_TICKET_WIDTH:
        gap = _MIN_TICKET_GAP
        gaps = [gap] * (ticket_count - 1)
        usable_w = strip_w - left_margin - right_margin - sum(gaps)
        slot_w = usable_w / ticket_count

    if slot_w < _MIN_TICKET_WIDTH:
        left_margin = right_margin = max(0.0, (strip_w - (ticket_count * _MIN_TICKET_WIDTH) - sum(gaps)) / 2.0)
        usable_w = strip_w - left_margin - right_margin - sum(gaps)
        slot_w = usable_w / ticket_count

    positions: list[tuple[float, float]] = []
    comp_w = max(0.60 * inch, slot_w * _TICKET_COMPOSITION_RATIO)
    x = left_margin
    for idx in range(ticket_count):
        comp_x = x + max(0.0, (slot_w - comp_w) / 2.0)
        positions.append((comp_x, comp_w))
        if idx < len(gaps):
            x += slot_w + gaps[idx]
        else:
            x += slot_w
    return _StripContinuityLayout(positions=positions, left_margin=left_margin, right_margin=right_margin)


def draw_ticket_item_number(c: canvas.Canvas, item_number: str, right_x: float, baseline_y: float, max_w: float) -> None:
    font = get_sams_strip_font("regular")
    text = _truncate(item_number or "-", font, _SAMS_ITEM_SIZE, max(20.0, max_w))
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(font, _SAMS_ITEM_SIZE)
    c.drawRightString(right_x, baseline_y, text)


def draw_ticket_text_stack(
    c: canvas.Canvas,
    segment: SamsPriceStripSegment,
    x: float,
    y: float,
    w: float,
    h: float,
    stack_anchor_top: float,
) -> None:
    brand_font = get_sams_strip_font("semibold")
    desc_font = get_sams_strip_font("regular")
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    stack_top_limit = y + h - _DEFAULT_INNER_PAD_TOP
    stack_top = min(stack_top_limit, stack_anchor_top)
    max_w = max(8.0, w - (2 * pad_x))

    brand = _truncate(segment.brand or "-", brand_font, _SAMS_BRAND_SIZE, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", desc_font, _SAMS_DESC_SIZE, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", desc_font, _SAMS_DESC_SIZE, max_w)

    desc_gap = 0.8
    brand_gap = 1.0
    brand_y = stack_top - _SAMS_BRAND_SIZE
    desc_1_y = brand_y - brand_gap - _SAMS_DESC_SIZE
    desc_2_y = desc_1_y - desc_gap - _SAMS_DESC_SIZE

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(brand_font, _SAMS_BRAND_SIZE)
    c.drawString(x + pad_x, brand_y, brand)
    c.setFont(desc_font, _SAMS_DESC_SIZE)
    c.drawString(x + pad_x, desc_1_y, desc_1)
    c.setFont(desc_font, _SAMS_DESC_SIZE)
    c.drawString(x + pad_x, desc_2_y, desc_2)


def _layout_price_object(retail: str, x: float, y: float, w: float, h: float, font_name: str) -> _PriceObjectLayout:
    dollars, cents = _normalize_price_parts(retail)
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    price_left = x + max(_RETAIL_MARGIN_PAD, pad_x * 0.35)
    price_right = x + w - pad_x
    max_object_w = max(12.0, price_right - price_left)
    max_object_ascent = max(15.0, h * 0.53)

    base_dollars_size = _SAMS_PRICE_SIZE
    base_sign_size = _SAMS_PRICE_SIZE * 0.33
    base_cents_size = _SAMS_PRICE_SIZE * 0.40
    base_sign_gap = max(0.9, base_dollars_size * 0.055)
    base_cents_gap = max(0.7, base_dollars_size * 0.020)
    base_sign_rise = base_dollars_size * 0.44
    base_cents_rise = base_dollars_size * 0.50

    sign_w = pdfmetrics.stringWidth("$", font_name, base_sign_size)
    dollars_w = pdfmetrics.stringWidth(dollars, font_name, base_dollars_size)
    cents_w = pdfmetrics.stringWidth(cents, font_name, base_cents_size)
    base_object_w = sign_w + base_sign_gap + dollars_w + base_cents_gap + cents_w
    base_object_ascent = max(
        base_dollars_size * 0.86,
        base_sign_rise + (base_sign_size * 0.82),
        base_cents_rise + (base_cents_size * 0.82),
    )
    scale = min(1.0, max_object_w / max(base_object_w, 1.0), max_object_ascent / max(base_object_ascent, 1.0))
    scale = max(0.30, scale)

    dollars_size = base_dollars_size * scale
    sign_size = base_sign_size * scale
    cents_size = base_cents_size * scale
    sign_gap = base_sign_gap * scale
    cents_gap = base_cents_gap * scale
    sign_rise = base_sign_rise * scale
    cents_rise = base_cents_rise * scale

    sign_w = pdfmetrics.stringWidth("$", font_name, sign_size)
    dollars_w = pdfmetrics.stringWidth(dollars, font_name, dollars_size)
    cents_w = pdfmetrics.stringWidth(cents, font_name, cents_size)
    object_w = sign_w + sign_gap + dollars_w + cents_gap + cents_w
    object_ascent = max(
        dollars_size * 0.86,
        sign_rise + (sign_size * 0.82),
        cents_rise + (cents_size * 0.82),
    )
    final_fit_scale = min(1.0, max_object_w / max(object_w, 1.0), max_object_ascent / max(object_ascent, 1.0))
    if final_fit_scale < 1.0:
        dollars_size *= final_fit_scale
        sign_size *= final_fit_scale
        cents_size *= final_fit_scale
        sign_gap *= final_fit_scale
        cents_gap *= final_fit_scale
        sign_rise *= final_fit_scale
        cents_rise *= final_fit_scale
        sign_w = pdfmetrics.stringWidth("$", font_name, sign_size)
        dollars_w = pdfmetrics.stringWidth(dollars, font_name, dollars_size)
        cents_w = pdfmetrics.stringWidth(cents, font_name, cents_size)
        object_w = sign_w + sign_gap + dollars_w + cents_gap + cents_w
        object_ascent = max(
            dollars_size * 0.86,
            sign_rise + (sign_size * 0.82),
            cents_rise + (cents_size * 0.82),
        )

    dollars_baseline = y + max(1.8, h * 0.13)
    object_left = price_left
    dollars_x = object_left + sign_w + sign_gap
    dollar_sign_x = object_left
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
        object_right_x=object_right_x,
        object_top_y=object_top_y,
    )


def draw_price_object(c: canvas.Canvas, retail: str, x: float, y: float, w: float, h: float) -> _PriceAnchor:
    price_font = get_sams_strip_font("semibold")
    dollars, cents = _normalize_price_parts(retail)
    layout = _layout_price_object(retail, x, y, w, h, price_font)

    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(price_font, layout.dollar_sign_size)
    c.drawString(layout.dollar_sign_x, layout.dollar_sign_baseline_y, "$")
    c.setFont(price_font, layout.dollars_size)
    c.drawString(layout.dollars_x, layout.dollars_baseline_y, dollars)
    c.setFont(price_font, layout.cents_size)
    c.drawString(layout.cents_x, layout.cents_baseline_y, cents)

    price_top = layout.object_top_y
    text_stack_anchor_top = price_top + _SAMS_BRAND_SIZE + (_SAMS_DESC_SIZE * 2) + 2.6
    item_baseline_y = max(y + 1.4, layout.dollars_baseline_y - (_SAMS_ITEM_SIZE + 1.0))
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    return _PriceAnchor(
        text_stack_anchor_top=text_stack_anchor_top,
        item_baseline_y=item_baseline_y,
        item_right_x=min(x + w - pad_x, layout.object_right_x),
        item_max_w=max(20.0, min(w * 0.68, (layout.object_right_x - x) + 1.5)),
    )


def draw_ticket_composition(c: canvas.Canvas, segment: SamsPriceStripSegment, x: float, y: float, w: float, h: float) -> None:
    content_bottom = y + _DEFAULT_INNER_PAD_BOTTOM
    content_h = max(12.0, h - _DEFAULT_INNER_PAD_BOTTOM - _DEFAULT_INNER_PAD_TOP)
    anchor = draw_price_object(c, segment.retail, x, content_bottom, w, content_h)
    draw_ticket_text_stack(c, segment, x, content_bottom, w, content_h, anchor.text_stack_anchor_top)
    draw_ticket_item_number(c, segment.item_number, anchor.item_right_x, anchor.item_baseline_y, anchor.item_max_w)


def draw_strip_footer(
    c: canvas.Canvas,
    row_data: SamsPriceStripRow,
    y: float,
    max_width: float,
    left_margin: float,
) -> None:
    font = get_sams_strip_font("regular")
    footer_text = row_data.footer_text.strip()
    if footer_text == "":
        footer_text = f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"
    footer_x = max(0.03 * inch, left_margin * 0.55)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(font, _SAMS_FOOTER_SIZE)
    c.drawString(footer_x, y + 1.0, _truncate(footer_text, font, _SAMS_FOOTER_SIZE, max_width))


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

    layout = compute_ticket_positions_across_strip(page_w, segment_count)
    positions = layout.positions
    if positions and positions[0][1] < _MIN_TICKET_WIDTH:
        warnings.append(
            f"Tight segment width for POG={row_data.pog}, Side={row_data.side}, Row={row_data.row}; count={segment_count}."
        )

    ticket_y = footer_h
    for idx, (x, ticket_w) in enumerate(positions):
        draw_ticket_composition(c, row_data.segments[idx], x, ticket_y, ticket_w, strip_h)

    footer_w = min(page_w * 0.64, max(46.0, page_w - layout.left_margin - layout.right_margin - (0.05 * inch)))
    draw_strip_footer(c, row_data, 0.0, footer_w, layout.left_margin)
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
        warning = sams_gibson_warning()
        if warning:
            warnings.append(warning)
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
