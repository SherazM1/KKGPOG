from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.sams_club.models import SamsPlanogram, SamsSidePage, SamsSlot

_PAGE_SIZE = landscape(letter)
_HEADER_BLUE = (0.09, 0.31, 0.66)
_HEADER_TEXT = (1.0, 1.0, 1.0)
_CARD_BORDER = (0.72, 0.72, 0.72)
_CARD_META_BG = (0.95, 0.96, 0.98)
_TEXT_DARK = (0.12, 0.12, 0.12)
_TEXT_MUTED = (0.42, 0.42, 0.42)
_FONT_HEADER_FALLBACK = "Helvetica-Bold"
_FONT_FOOTER_FALLBACK = "Helvetica"
_FONT_BODY_BOLD = "Helvetica-Bold"
_FONT_BODY_REGULAR = "Helvetica"
_FONT_ITALIC = "Helvetica-Oblique"
_RALEWAY_FONT_NAME = "Raleway-Regular"
_RALEWAY_REGISTERED = False
_BASE_MARGIN = 20.0
_BASE_HEADER_H = 42.0
_BASE_FOOTER_H = 24.0
_BASE_BODY_BOTTOM_GAP = 8.0


@dataclass(frozen=True)
class _SideLayoutMetrics:
    page_w: float
    page_h: float
    margin: float
    header_h: float
    footer_h: float
    body_bottom_gap: float
    row_gap: float
    col_gap: float


@dataclass
class SamsPdfRenderResult:
    pdf_bytes: bytes
    warnings: list[str] = field(default_factory=list)
    rendered_slots: int = 0
    missing_image_slots: int = 0


def _resolve_assets_root() -> Path:
    # render_planogram.py -> sams_club -> app -> project_root
    return Path(__file__).resolve().parents[2] / "assets"


def _register_raleway_if_available() -> bool:
    global _RALEWAY_REGISTERED
    if _RALEWAY_REGISTERED:
        return True

    font_path = _resolve_assets_root() / "Raleway-Regular.ttf"
    if not font_path.exists():
        return False

    try:
        if _RALEWAY_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(_RALEWAY_FONT_NAME, str(font_path)))
        _RALEWAY_REGISTERED = True
        return True
    except Exception:
        return False


def _header_font_name() -> str:
    return _RALEWAY_FONT_NAME if _register_raleway_if_available() else _FONT_HEADER_FALLBACK


def _footer_font_name() -> str:
    return _RALEWAY_FONT_NAME if _register_raleway_if_available() else _FONT_FOOTER_FALLBACK


def _fit_text(text: str, font_name: str, max_width: float, max_size: float, min_size: float = 6.0) -> float:
    size = max_size
    while size > min_size and pdfmetrics.stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    return max(size, min_size)


def _truncate_text(text: str, font_name: str, font_size: float, max_width: float) -> str:
    if pdfmetrics.stringWidth(text, font_name, font_size) <= max_width:
        return text
    suffix = "..."
    limit = max_width - pdfmetrics.stringWidth(suffix, font_name, font_size)
    if limit <= 0:
        return suffix
    out = text
    while out and pdfmetrics.stringWidth(out, font_name, font_size) > limit:
        out = out[:-1]
    return f"{out}{suffix}"


def _fit_protected_single_line_font(
    text: str,
    font_name: str,
    max_width: float,
    max_size: float,
    min_size: float,
) -> float:
    if max_width <= 0:
        return min_size
    size = max_size
    while size > min_size and pdfmetrics.stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    return max(size, min_size)


def _wrap_text_lines(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    raw_text = " ".join((text or "").split())
    if raw_text == "":
        return [""]
    if max_width <= 0:
        return [raw_text]

    words = raw_text.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue
        if not current and pdfmetrics.stringWidth(word, font_name, font_size) <= max_width:
            current = word
            continue

        if current:
            lines.append(current)
            current = ""

        if pdfmetrics.stringWidth(word, font_name, font_size) <= max_width:
            current = word
            continue

        token = ""
        for char in word:
            test = f"{token}{char}"
            if token and pdfmetrics.stringWidth(test, font_name, font_size) > max_width:
                lines.append(token)
                token = char
            else:
                token = test
        current = token

    if current:
        lines.append(current)

    return lines


def _wrapped_block_height(line_count: int, font_size: float, line_gap: float = 1.4) -> float:
    if line_count <= 0:
        return 0.0
    return line_count * (font_size + line_gap)


def _draw_wrapped_lines(
    c: canvas.Canvas,
    lines: list[str],
    x: float,
    top_y: float,
    font_name: str,
    font_size: float,
    line_gap: float = 1.4,
) -> float:
    if not lines:
        return top_y
    y = top_y
    step = font_size + line_gap
    c.setFont(font_name, font_size)
    for line in lines:
        c.drawString(x, y - font_size, line)
        y -= step
    return y


def _compute_image_height(available_height: float, preferred_height: float, min_height: float) -> float:
    if available_height <= 0:
        return 0.0
    if available_height >= preferred_height:
        return preferred_height
    return max(min_height, available_height)


def _compute_image_box(card_h: float, text_and_image_h: float, text_h: float) -> tuple[float, float]:
    preferred_h = max(36.0, max(card_h * 0.52, text_and_image_h * 0.5))
    min_h = max(18.0, min(card_h * 0.38, text_and_image_h * 0.45))
    available = text_and_image_h - text_h
    image_h = _compute_image_height(available, preferred_h, min_h)
    return max(0.0, image_h), min_h


def _effective_column_count(side_page: SamsSidePage) -> int:
    populated_counts = []
    for row in side_page.rows:
        populated = row.populated_column_count
        if populated <= 0:
            populated = len({slot.column for slot in row.slots})
        populated_counts.append(populated)
    max_populated = max(populated_counts) if populated_counts else 0
    if max_populated <= 0:
        return 1
    return max_populated


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _compute_card_dimensions(
    *,
    page_w: float,
    page_h: float,
    row_count: int,
    col_count: int,
    margin: float,
    header_h: float,
    footer_h: float,
    body_bottom_gap: float,
    row_gap: float,
    col_gap: float,
) -> tuple[float, float]:
    content_w = max(1.0, page_w - (2 * margin))
    content_h = max(1.0, page_h - header_h - (2 * margin) - footer_h - body_bottom_gap)
    card_w = (content_w - ((col_count - 1) * col_gap)) / max(1, col_count)
    card_h = (content_h - ((row_count - 1) * row_gap)) / max(1, row_count)
    return max(1.0, card_w), max(1.0, card_h)


def _compute_page_dimensions_for_side(side_page: SamsSidePage) -> _SideLayoutMetrics:
    base_w, base_h = _PAGE_SIZE
    margin = _BASE_MARGIN
    header_h = _BASE_HEADER_H
    footer_h = _BASE_FOOTER_H
    body_bottom_gap = _BASE_BODY_BOTTOM_GAP

    sorted_rows = sorted(side_page.rows, key=lambda row: row.row_number)
    row_count = max(1, len(sorted_rows))
    col_count = max(1, _effective_column_count(side_page))

    row_gap = _clamp(8.0 - (row_count - 3) * 0.4, 5.0, 8.0)
    col_gap = _clamp(6.0 - (col_count - 4) * 0.3, 4.0, 6.0)

    # Expand page dimensions when density is high so cards stay readable and visually full.
    min_card_w = 124.0
    min_card_h = 164.0
    needed_content_w = (col_count * min_card_w) + ((col_count - 1) * col_gap)
    needed_content_h = (row_count * min_card_h) + ((row_count - 1) * row_gap)

    page_w = max(base_w, needed_content_w + (2 * margin))
    page_h = max(base_h, needed_content_h + header_h + (2 * margin) + footer_h + body_bottom_gap)

    return _SideLayoutMetrics(
        page_w=page_w,
        page_h=page_h,
        margin=margin,
        header_h=header_h,
        footer_h=footer_h,
        body_bottom_gap=body_bottom_gap,
        row_gap=row_gap,
        col_gap=col_gap,
    )


def _format_retail_price(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if text == "":
        return "-"
    normalized = text.replace("$", "").replace(",", "").strip()
    try:
        amount = float(normalized)
        return f"${amount:.2f}"
    except ValueError:
        return text if text.startswith("$") else f"${text}"


def _format_cpp_value(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if text == "":
        return "CPP -"

    normalized = text.replace(",", "").strip()
    try:
        value = float(normalized)
        if value.is_integer():
            formatted = str(int(value))
        else:
            formatted = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"CPP {formatted}"
    except ValueError:
        return f"CPP {text}"


def _description_for_slot(slot: SamsSlot) -> str:
    primary = (slot.description or "").strip()
    if primary:
        return primary
    fallback_parts = [(slot.desc_1 or "").strip(), (slot.desc_2 or "").strip(), (slot.brand or "").strip()]
    fallback = " ".join([part for part in fallback_parts if part])
    return fallback or "No description"


def _load_slot_image(path_text: str, cache: dict[str, Image.Image | None]) -> tuple[Image.Image | None, str | None]:
    file_path = (path_text or "").strip()
    if not file_path:
        return None, "missing file_path"

    if file_path in cache:
        cached = cache[file_path]
        if cached is None:
            return None, "cached load failure"
        return cached, None

    try:
        path = Path(file_path)
        if not path.exists():
            cache[file_path] = None
            return None, "file not found"
        with Image.open(path) as img:
            loaded = img.convert("RGB")
            cache[file_path] = loaded
            return loaded, None
    except Exception as exc:  # noqa: BLE001 - non-fatal image load path
        cache[file_path] = None
        return None, f"load error: {exc}"


def _draw_header(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    pog: str,
    side_number: int,
    header_h: float,
    title_override: str | None = None,
) -> None:
    y = page_h - header_h
    c.setFillColorRGB(*_HEADER_BLUE)
    c.setStrokeColorRGB(*_HEADER_BLUE)
    c.rect(0, y, page_w, header_h, stroke=0, fill=1)

    header_font = _header_font_name()
    c.setFillColorRGB(*_HEADER_TEXT)
    c.setFont(header_font, 12)
    c.drawString(20, y + (header_h / 2) - 4.5, "POG")

    title = (title_override or "").strip() or pog.strip() or "Sam's Club Planogram"
    title_fs = _fit_text(title, header_font, page_w * 0.52, 16, 10)
    title = _truncate_text(title, header_font, title_fs, page_w * 0.52)
    title_w = pdfmetrics.stringWidth(title, header_font, title_fs)
    c.setFont(header_font, title_fs)
    c.drawString((page_w - title_w) / 2, y + (header_h / 2) - (title_fs / 2) + 0.5, title)

    side_label = f"SIDE {side_number}"
    c.setFont(header_font, 12)
    side_w = pdfmetrics.stringWidth(side_label, header_font, 12)
    c.drawString(page_w - side_w - 20, y + (header_h / 2) - 4.5, side_label)


def _draw_footer(c: canvas.Canvas, page_w: float, margin: float, generated_by: str, generated_at: str) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.line(margin, margin + 12, page_w - margin, margin + 12)
    c.setFillColorRGB(0.32, 0.32, 0.32)
    c.setFont(_footer_font_name(), 8)
    c.drawString(margin, margin, f"Generated: {generated_at}")
    c.drawRightString(page_w - margin, margin, f"Generated by: {generated_by}")


def _draw_image_area(c: canvas.Canvas, x: float, y: float, w: float, h: float, image: Image.Image | None) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.rect(x, y, w, h, stroke=1, fill=0)

    if image is None:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont(_FONT_ITALIC, 7)
        c.drawCentredString(x + (w / 2), y + (h / 2) - 3, "Image unavailable")
        return

    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont(_FONT_ITALIC, 7)
        c.drawCentredString(x + (w / 2), y + (h / 2) - 3, "Invalid image")
        return

    scale = min(w / source_w, h / source_h)
    draw_w = source_w * scale
    draw_h = source_h * scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2

    c.drawImage(ImageReader(image), draw_x, draw_y, draw_w, draw_h, preserveAspectRatio=True, mask="auto")


def _draw_slot_card(
    c: canvas.Canvas,
    slot: SamsSlot,
    x: float,
    y: float,
    w: float,
    h: float,
    image: Image.Image | None,
) -> None:
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.setStrokeColorRGB(*_CARD_BORDER)
    c.setLineWidth(0.7)
    c.rect(x, y, w, h, stroke=1, fill=1)

    size_scale = _clamp(min(w / 150.0, h / 185.0), 0.9, 1.45)
    pad = _clamp(w * 0.06, 3.8, 8.0)
    metadata_h = max(11.5, h * 0.102)
    inner_x = x + pad
    inner_w = w - (2 * pad)
    inner_h = h - (2 * pad)
    text_and_image_h = max(8.0, inner_h - metadata_h)

    meta_y = y + h - metadata_h
    c.setFillColorRGB(*_CARD_META_BG)
    c.setStrokeColorRGB(*_CARD_META_BG)
    c.rect(x, meta_y, w, metadata_h, stroke=0, fill=1)

    c.setFillColorRGB(*_TEXT_DARK)
    meta_max_fs = 7.5 * size_scale
    meta_min_fs = max(5.75, 5.0 * size_scale)
    meta_fs = _fit_text("RETAIL", _FONT_BODY_BOLD, (w / 2) - (pad * 2), meta_max_fs, meta_min_fs)
    c.setFont(_FONT_BODY_BOLD, meta_fs)
    retail = _truncate_text(_format_retail_price(slot.retail), _FONT_BODY_BOLD, meta_fs, (w / 2) - (pad * 2))
    cpp = _truncate_text(_format_cpp_value(slot.cpp), _FONT_BODY_BOLD, meta_fs, (w / 2) - (pad * 2))
    meta_baseline = meta_y + (metadata_h - meta_fs) / 2
    c.drawString(x + pad, meta_baseline, retail)
    c.drawRightString(x + w - pad, meta_baseline, cpp)

    upc_text = f"UPC {(slot.upc or '').strip() or '-'}"
    item_text = f"ITEM {(slot.item_number or '').strip() or '-'}"
    description = _description_for_slot(slot)

    upc_max_fs = 7.5 * size_scale
    item_max_fs = 7.0 * size_scale
    desc_default_fs = 7.25 * size_scale
    upc_fs = _fit_protected_single_line_font(upc_text, _FONT_BODY_BOLD, inner_w, upc_max_fs, 4.2)
    item_fs = _fit_protected_single_line_font(item_text, _FONT_BODY_REGULAR, inner_w, item_max_fs, 4.0)
    item_lines = _wrap_text_lines(item_text, _FONT_BODY_REGULAR, item_fs, inner_w)
    desc_fs = max(6.0, desc_default_fs)
    min_desc_fs = 4.0
    block_gap = 1.6

    desc_lines = _wrap_text_lines(description, _FONT_BODY_REGULAR, desc_fs, inner_w)
    while desc_fs > min_desc_fs:
        item_lines = _wrap_text_lines(item_text, _FONT_BODY_REGULAR, item_fs, inner_w)
        desc_lines = _wrap_text_lines(description, _FONT_BODY_REGULAR, desc_fs, inner_w)
        text_h = (
            (upc_fs + 1.2)
            + block_gap
            + _wrapped_block_height(len(item_lines), item_fs, 1.05)
            + block_gap
            + _wrapped_block_height(len(desc_lines), desc_fs, 1.0)
        )
        _, image_min_h = _compute_image_box(h, text_and_image_h, text_h)
        if text_h + image_min_h <= text_and_image_h:
            break
        if item_fs > 4.0:
            item_fs = max(4.0, item_fs - 0.25)
        elif upc_fs > 4.0:
            upc_fs = max(4.0, upc_fs - 0.25)
        else:
            desc_fs = max(min_desc_fs, desc_fs - 0.25)

    item_lines = _wrap_text_lines(item_text, _FONT_BODY_REGULAR, item_fs, inner_w)
    desc_lines = _wrap_text_lines(description, _FONT_BODY_REGULAR, desc_fs, inner_w)
    text_h = (
        (upc_fs + 1.2)
        + block_gap
        + _wrapped_block_height(len(item_lines), item_fs, 1.05)
        + block_gap
        + _wrapped_block_height(len(desc_lines), desc_fs, 1.0)
    )
    image_h, _ = _compute_image_box(h, text_and_image_h, text_h)

    content_top = y + pad + inner_h - metadata_h
    text_top = content_top - 0.8
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(_FONT_BODY_BOLD, upc_fs)
    c.drawString(inner_x, text_top - upc_fs, upc_text)

    current_y = text_top - (upc_fs + 1.2) - block_gap
    c.setFillColorRGB(*_TEXT_DARK)
    current_y = _draw_wrapped_lines(c, item_lines, inner_x, current_y, _FONT_BODY_REGULAR, item_fs, 1.05) - block_gap

    c.setFillColorRGB(*_TEXT_DARK)
    current_y = _draw_wrapped_lines(c, desc_lines, inner_x, current_y, _FONT_BODY_REGULAR, desc_fs, 1.0)

    image_y = y + pad
    image_h = max(0.0, min(image_h, max(0.0, current_y - image_y - 1.0)))
    _draw_image_area(c, inner_x, image_y, inner_w, image_h, image)


def _render_side_page(
    c: canvas.Canvas,
    side_page: SamsSidePage,
    generated_by: str,
    generated_at: str,
    image_cache: dict[str, Image.Image | None],
    warnings: list[str],
    title_override: str | None = None,
) -> tuple[int, int]:
    metrics = _compute_page_dimensions_for_side(side_page)
    page_w = metrics.page_w
    page_h = metrics.page_h
    margin = metrics.margin
    header_h = metrics.header_h
    footer_h = metrics.footer_h
    row_gap = metrics.row_gap
    col_gap = metrics.col_gap

    c.setPageSize((page_w, page_h))

    _draw_header(c, page_w, page_h, side_page.pog, side_page.side, header_h, title_override=title_override)
    _draw_footer(c, page_w, margin, generated_by, generated_at)

    content_left = margin
    content_right = page_w - margin
    content_top = page_h - header_h - margin
    content_bottom = margin + footer_h + metrics.body_bottom_gap

    sorted_rows = sorted(side_page.rows, key=lambda row: row.row_number)
    row_count = len(sorted_rows)
    if row_count <= 0:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont(_FONT_ITALIC, 11)
        c.drawCentredString(page_w / 2, (content_top + content_bottom) / 2, "No populated rows for this side.")
        return 0, 0

    effective_columns = _effective_column_count(side_page)
    card_w, card_h = _compute_card_dimensions(
        page_w=page_w,
        page_h=page_h,
        row_count=row_count,
        col_count=effective_columns,
        margin=margin,
        header_h=header_h,
        footer_h=footer_h,
        body_bottom_gap=metrics.body_bottom_gap,
        row_gap=row_gap,
        col_gap=col_gap,
    )

    rendered_slots = 0
    missing_image_slots = 0

    for row_index, row in enumerate(sorted_rows):
        row_top = content_top - row_index * (card_h + row_gap)
        card_y = row_top - card_h
        sorted_slots = sorted(row.slots, key=lambda slot: slot.column)
        for slot_index, slot in enumerate(sorted_slots):
            card_x = content_left + slot_index * (card_w + col_gap)
            image_path = (slot.resolved_image_path or slot.file_path or "").strip()
            image, image_warning = _load_slot_image(image_path, image_cache)
            if image is None:
                missing_image_slots += 1
                detail = f"side={slot.side} row={slot.row} column={slot.column}"
                if image_warning:
                    warnings.append(f"Image unavailable ({detail}): {image_warning}.")
                else:
                    warnings.append(f"Image unavailable ({detail}).")
            _draw_slot_card(c, slot, card_x, card_y, card_w, card_h, image)
            rendered_slots += 1

    return rendered_slots, missing_image_slots


def render_sams_planogram_pdf(
    planogram: SamsPlanogram,
    generated_by: str = "Kendal King",
    title_override: str | None = None,
) -> SamsPdfRenderResult:
    """
    Render a first-pass Sam's Club completed planogram PDF.

    The output contains one landscape page per populated side, preserving side/row/column order
    and rendering only populated slots from the provided structure.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=_PAGE_SIZE)

    image_cache: dict[str, Image.Image | None] = {}
    warnings: list[str] = []
    rendered_slots = 0
    missing_image_slots = 0
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for side_page in sorted(planogram.side_pages, key=lambda page: page.side):
        side_rendered, side_missing_images = _render_side_page(
            c,
            side_page,
            generated_by=generated_by,
            generated_at=generated_at,
            image_cache=image_cache,
            warnings=warnings,
            title_override=title_override,
        )
        rendered_slots += side_rendered
        missing_image_slots += side_missing_images
        c.showPage()

    c.save()
    return SamsPdfRenderResult(
        pdf_bytes=buffer.getvalue(),
        warnings=warnings,
        rendered_slots=rendered_slots,
        missing_image_slots=missing_image_slots,
    )
