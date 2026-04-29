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
_TEXT_DARK = (0.0, 0.0, 0.0)
_TEXT_STACK_BRAND = (0.0, 0.0, 0.0)
_TEXT_STACK_DESC = (0.03, 0.03, 0.03)
_TEXT_ITEM = (0.04, 0.04, 0.04)
_TEXT_MUTED = (0.12, 0.12, 0.12)
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
_STRIP_COMPOSITION_WIDTH_RATIO = 0.94
_SAMS_FONT_REGULAR = "Sams-Gibson-Regular"
_SAMS_FONT_SEMIBOLD = "Sams-Gibson-SemiBold"
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
_TICKET_VERTICAL_LIFT_RATIO = 0.08
_TICKET_VERTICAL_LIFT_MAX = 0.16 * inch
_PRICE_OBJECT_BAND_ANCHOR_RATIO = 0.30
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
    object_left_x: float
    object_bottom_y: float
    object_right_x: float
    object_top_y: float


class _StripContinuityLayout(NamedTuple):
    positions: list[tuple[float, float]]
    left_margin: float
    right_margin: float


class _TicketCompositionLayout(NamedTuple):
    origin_x: float
    origin_y: float
    content_x: float
    content_y: float
    content_w: float
    content_h: float


def _compute_content_box_metrics(y: float, h: float) -> tuple[float, float, float]:
    inner_top = _DEFAULT_INNER_PAD_TOP * 0.55
    inner_bottom = _DEFAULT_INNER_PAD_BOTTOM * 0.55
    available_h = max(12.0, h - inner_bottom - inner_top)
    content_h = max(12.0, available_h * 0.92)
    min_content_y = y + inner_bottom
    max_content_y = max(min_content_y, y + h - inner_top - content_h)
    return min_content_y, max_content_y, content_h


def _estimate_ticket_block_offsets(segment: SamsPriceStripSegment, w: float, h: float) -> tuple[float, float]:
    """Return (top, bottom) visual offsets from content_y for one ticket block."""
    price_font = get_sams_strip_font("semibold")
    layout = _layout_price_object(segment.retail, 0.0, 0.0, w, h, price_font)
    price_top = layout.object_top_y
    stack_top_limit = h - _DEFAULT_INNER_PAD_TOP
    stack_anchor_top = price_top + _SAMS_BRAND_SIZE + (_SAMS_DESC_SIZE * 2) + _SAMS_STACK_TO_PRICE_OFFSET
    stack_top = min(stack_top_limit, stack_anchor_top)

    brand_y = stack_top - _SAMS_BRAND_SIZE
    desc_1_y = brand_y - _SAMS_STACK_BRAND_GAP - _SAMS_DESC_SIZE
    desc_2_y = desc_1_y - _SAMS_STACK_DESC_GAP - _SAMS_DESC_SIZE
    text_top = max(
        brand_y + (_SAMS_BRAND_SIZE * 0.82),
        desc_1_y + (_SAMS_DESC_SIZE * 0.82),
        desc_2_y + (_SAMS_DESC_SIZE * 0.82),
    )

    item_baseline_y = max(1.1, layout.object_bottom_y + (_SAMS_ITEM_SIZE * 0.16))
    item_bottom = item_baseline_y - (_SAMS_ITEM_SIZE * 0.20)
    block_top = max(price_top, text_top)
    block_bottom = min(layout.object_bottom_y, item_bottom)
    return block_top, block_bottom


def _compute_row_centered_content_y(
    row_data: SamsPriceStripRow,
    positions: list[tuple[float, float]],
    ticket_y: float,
    strip_h: float,
) -> tuple[float, float]:
    """
    Center the full ticket composition block (brand/desc/price/item) in strip usable area.
    Returns (content_y, content_h) applied consistently to every ticket in the row.
    """
    min_content_y, max_content_y, content_h = _compute_content_box_metrics(ticket_y, strip_h)
    if not positions:
        return min_content_y, content_h

    # Use the narrowest width in-row to keep the anchor valid for all repeated tickets.
    composition_w = min(ticket_w for _, ticket_w in positions)
    block_top = float("-inf")
    block_bottom = float("inf")
    for segment in row_data.segments[: len(positions)]:
        seg_top, seg_bottom = _estimate_ticket_block_offsets(segment, composition_w, content_h)
        block_top = max(block_top, seg_top)
        block_bottom = min(block_bottom, seg_bottom)

    if block_top == float("-inf") or block_bottom == float("inf") or block_top <= block_bottom:
        return min_content_y, content_h

    block_center_offset = (block_top + block_bottom) / 2.0
    usable_center_y = ticket_y + (strip_h / 2.0)
    centered_content_y = usable_center_y - block_center_offset
    centered_content_y = min(max(centered_content_y, min_content_y), max_content_y)
    return centered_content_y, content_h


def register_sams_strip_fonts() -> None:
    global _FONTS_READY, _SAMS_GIBSON_AVAILABLE, _SAMS_FONT_WARNING
    if _FONTS_READY:
        return
    root = Path(__file__).resolve().parents[2]
    regular_path = root / "assets" / "Gibson-Regular.otf"
    semibold_path = root / "assets" / "Gibson-SemiBold.otf"
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
            "Expected assets/Gibson-Regular.otf and assets/Gibson-SemiBold.otf. "
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


def compute_ticket_positions_across_strip(strip_w: float, ticket_count: int) -> _StripContinuityLayout:
    if ticket_count <= 0:
        return _StripContinuityLayout(positions=[], left_margin=0.0, right_margin=0.0)

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
    return _StripContinuityLayout(positions=positions, left_margin=left_margin, right_margin=right_margin)


def draw_ticket_item_number(c: canvas.Canvas, item_number: str, right_x: float, baseline_y: float, max_w: float) -> None:
    font = get_sams_strip_font("semibold")
    text = _truncate(item_number or "-", font, _SAMS_ITEM_SIZE, max(20.0, max_w))
    c.setFillColorRGB(*_TEXT_ITEM)
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
    desc_font = get_sams_strip_font("semibold")
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    stack_top_limit = y + h - _DEFAULT_INNER_PAD_TOP
    stack_top = min(stack_top_limit, stack_anchor_top)
    max_w = max(8.0, w - (2 * pad_x))

    brand = _truncate(segment.brand or "-", brand_font, _SAMS_BRAND_SIZE, max_w)
    desc_1 = _truncate(segment.desc_1 or "-", desc_font, _SAMS_DESC_SIZE, max_w)
    desc_2 = _truncate(segment.desc_2 or "-", desc_font, _SAMS_DESC_SIZE, max_w)

    desc_gap = _SAMS_STACK_DESC_GAP
    brand_gap = _SAMS_STACK_BRAND_GAP
    brand_y = stack_top - _SAMS_BRAND_SIZE
    desc_1_y = brand_y - brand_gap - _SAMS_DESC_SIZE
    desc_2_y = desc_1_y - desc_gap - _SAMS_DESC_SIZE

    c.setFillColorRGB(*_TEXT_STACK_BRAND)
    c.setFont(brand_font, _SAMS_BRAND_SIZE)
    c.drawString(x + pad_x, brand_y, brand)
    c.setFillColorRGB(*_TEXT_STACK_DESC)
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
    max_object_h = max(15.0, h * 0.60)

    base_dollars_size = _SAMS_PRICE_SIZE
    base_sign_size = _SAMS_PRICE_SIZE * 0.33
    base_cents_size = _SAMS_PRICE_SIZE * 0.40
    base_sign_gap = max(0.9, base_dollars_size * _SAMS_PRICE_SIGN_GAP_RATIO)
    base_cents_gap = max(0.7, base_dollars_size * _SAMS_PRICE_CENTS_GAP_RATIO)
    base_sign_rise = base_dollars_size * _SAMS_PRICE_SIGN_RISE_RATIO
    base_cents_rise = base_dollars_size * _SAMS_PRICE_CENTS_RISE_RATIO

    sign_w = pdfmetrics.stringWidth("$", font_name, base_sign_size)
    dollars_w = pdfmetrics.stringWidth(dollars, font_name, base_dollars_size)
    cents_w = pdfmetrics.stringWidth(cents, font_name, base_cents_size)
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
    # Keep the three-part retail sign visually locked as one object across sizes.
    sign_gap = min(max(sign_gap, dollars_size * 0.018), dollars_size * 0.036)
    cents_gap = min(max(cents_gap, dollars_size * 0.005), dollars_size * 0.014)
    sign_rise = min(max(sign_rise, dollars_size * 0.355), dollars_size * 0.425)
    cents_rise = min(max(cents_rise, dollars_size * 0.415), dollars_size * 0.465)

    sign_w = pdfmetrics.stringWidth("$", font_name, sign_size)
    dollars_w = pdfmetrics.stringWidth(dollars, font_name, dollars_size)
    cents_w = pdfmetrics.stringWidth(cents, font_name, cents_size)
    object_w = sign_w + sign_gap + dollars_w + cents_gap + cents_w
    object_ascent = max(
        dollars_size * 0.86,
        sign_rise + (sign_size * 0.82),
        cents_rise + (cents_size * 0.82),
    )
    object_descent = max(dollars_size * 0.16, sign_size * 0.12, cents_size * 0.12)
    object_h = object_ascent + object_descent
    final_fit_scale = min(1.0, max_object_w / max(object_w, 1.0), max_object_h / max(object_h, 1.0))
    if final_fit_scale < 1.0:
        dollars_size *= final_fit_scale
        sign_size *= final_fit_scale
        cents_size *= final_fit_scale
        sign_gap *= final_fit_scale
        cents_gap *= final_fit_scale
        sign_rise *= final_fit_scale
        cents_rise *= final_fit_scale
        sign_gap = min(max(sign_gap, dollars_size * 0.018), dollars_size * 0.036)
        cents_gap = min(max(cents_gap, dollars_size * 0.005), dollars_size * 0.014)
        sign_rise = min(max(sign_rise, dollars_size * 0.355), dollars_size * 0.425)
        cents_rise = min(max(cents_rise, dollars_size * 0.415), dollars_size * 0.465)
        sign_w = pdfmetrics.stringWidth("$", font_name, sign_size)
        dollars_w = pdfmetrics.stringWidth(dollars, font_name, dollars_size)
        cents_w = pdfmetrics.stringWidth(cents, font_name, cents_size)
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
        object_left_x=object_left,
        object_bottom_y=object_bottom_y,
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
    text_stack_anchor_top = price_top + _SAMS_BRAND_SIZE + (_SAMS_DESC_SIZE * 2) + _SAMS_STACK_TO_PRICE_OFFSET
    item_baseline_y = max(y + 1.1, layout.object_bottom_y + (_SAMS_ITEM_SIZE * 0.16))
    pad_x = min(max(_DEFAULT_INNER_PAD_X, w * 0.052), max(_DEFAULT_INNER_PAD_X, w * 0.095))
    return _PriceAnchor(
        text_stack_anchor_top=text_stack_anchor_top,
        item_baseline_y=item_baseline_y,
        item_right_x=min(x + w - pad_x, layout.object_right_x + 0.1),
        item_max_w=max(20.0, min(w * 0.68, (layout.object_right_x - layout.object_left_x) + 2.0)),
    )


def layout_ticket_composition(
    x: float,
    y: float,
    w: float,
    h: float,
    row_content_y: float | None = None,
    row_content_h: float | None = None,
) -> _TicketCompositionLayout:
    # Single anchor/origin for every ticket composition; all internals hang off this.
    origin_x = x
    origin_y = y
    min_content_y, max_content_y, default_content_h = _compute_content_box_metrics(origin_y, h)
    content_h = default_content_h if row_content_h is None else max(12.0, row_content_h)
    if row_content_y is not None:
        content_y = min(max(row_content_y, min_content_y), max_content_y)
    else:
        lift = min(_TICKET_VERTICAL_LIFT_MAX, max(0.0, h * _TICKET_VERTICAL_LIFT_RATIO))
        centered_content_y = origin_y + ((h - content_h) / 2.0)
        content_y = min(max(centered_content_y + lift, min_content_y), max_content_y)
    content_x = origin_x
    content_w = w
    return _TicketCompositionLayout(
        origin_x=origin_x,
        origin_y=origin_y,
        content_x=content_x,
        content_y=content_y,
        content_w=content_w,
        content_h=content_h,
    )


def draw_ticket_composition(
    c: canvas.Canvas,
    segment: SamsPriceStripSegment,
    x: float,
    y: float,
    w: float,
    h: float,
    row_content_y: float | None = None,
    row_content_h: float | None = None,
) -> None:
    comp = layout_ticket_composition(x, y, w, h, row_content_y=row_content_y, row_content_h=row_content_h)
    anchor = draw_price_object(c, segment.retail, comp.content_x, comp.content_y, comp.content_w, comp.content_h)
    draw_ticket_text_stack(
        c,
        segment,
        comp.content_x,
        comp.content_y,
        comp.content_w,
        comp.content_h,
        anchor.text_stack_anchor_top,
    )
    draw_ticket_item_number(c, segment.item_number, anchor.item_right_x, anchor.item_baseline_y, anchor.item_max_w)


def _resolve_strip_footer_text(row_data: SamsPriceStripRow) -> str:
    raw = row_data.footer_text.strip()
    if raw and raw.lower() not in {"nan", "none", "null"}:
        return raw
    return f"Side: {row_data.side}, Row: {row_data.row} - POG: {row_data.pog}"


def draw_strip_footer(
    c: canvas.Canvas,
    row_data: SamsPriceStripRow,
    y: float,
    footer_band_h: float,
    max_width: float,
    left_margin: float,
    first_ticket_x: float,
) -> None:
    font = get_sams_strip_font("regular")
    footer_text = _resolve_strip_footer_text(row_data)
    _ = left_margin
    footer_x = max(0.03 * inch, first_ticket_x + (0.01 * inch))
    footer_baseline_y = y + max(0.75, min(1.5, footer_band_h * 0.20))
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(font, _SAMS_FOOTER_SIZE)
    c.drawString(footer_x, footer_baseline_y, _truncate(footer_text, font, _SAMS_FOOTER_SIZE, max_width))


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
    row_content_y, row_content_h = _compute_row_centered_content_y(row_data, positions, ticket_y, strip_h)
    for idx, (x, ticket_w) in enumerate(positions):
        draw_ticket_composition(
            c,
            row_data.segments[idx],
            x,
            ticket_y,
            ticket_w,
            strip_h,
            row_content_y=row_content_y,
            row_content_h=row_content_h,
        )

    footer_w = min(page_w * 0.58, max(40.0, page_w - layout.left_margin - layout.right_margin - (0.08 * inch)))
    first_ticket_x = positions[0][0] if positions else layout.left_margin
    draw_strip_footer(c, row_data, 0.0, footer_h, footer_w, layout.left_margin, first_ticket_x)
    return segment_count


def render_sams_price_strips_pdf(
    strip_rows: list[SamsPriceStripRow],
    generated_by: str = "Kendal King",
) -> SamsPriceStripPdfResult:
    _ = generated_by
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    warnings: list[str] = []
    warnings.append(
        "ACTIVE_SAMS_STRIP_RENDERER=app.sams_club.render_price_strips.render_sams_price_strips_pdf "
        "vertical_anchor_mode=row_composition_centered"
    )
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
