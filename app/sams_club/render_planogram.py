from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.sams_club.models import SamsPlanogram, SamsSidePage, SamsSlot

_PAGE_SIZE = landscape(letter)
_HEADER_BLUE = (0.09, 0.31, 0.66)
_HEADER_TEXT = (1.0, 1.0, 1.0)
_CARD_BORDER = (0.72, 0.72, 0.72)
_CARD_META_BG = (0.95, 0.96, 0.98)
_TEXT_DARK = (0.12, 0.12, 0.12)
_TEXT_MUTED = (0.42, 0.42, 0.42)


@dataclass
class SamsPdfRenderResult:
    pdf_bytes: bytes
    warnings: list[str] = field(default_factory=list)
    rendered_slots: int = 0
    missing_image_slots: int = 0


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


def _draw_header(c: canvas.Canvas, page_w: float, page_h: float, pog: str, side_number: int, header_h: float) -> None:
    y = page_h - header_h
    c.setFillColorRGB(*_HEADER_BLUE)
    c.setStrokeColorRGB(*_HEADER_BLUE)
    c.rect(0, y, page_w, header_h, stroke=0, fill=1)

    c.setFillColorRGB(*_HEADER_TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20, y + (header_h / 2) - 4, "POG")

    title = pog.strip() or "Sam's Club Planogram"
    title_fs = _fit_text(title, "Helvetica-Bold", page_w * 0.52, 16, 10)
    title = _truncate_text(title, "Helvetica-Bold", title_fs, page_w * 0.52)
    title_w = pdfmetrics.stringWidth(title, "Helvetica-Bold", title_fs)
    c.setFont("Helvetica-Bold", title_fs)
    c.drawString((page_w - title_w) / 2, y + (header_h / 2) - (title_fs / 2) + 1, title)

    side_label = f"SIDE {side_number}"
    c.setFont("Helvetica-Bold", 12)
    side_w = pdfmetrics.stringWidth(side_label, "Helvetica-Bold", 12)
    c.drawString(page_w - side_w - 20, y + (header_h / 2) - 4, side_label)


def _draw_footer(c: canvas.Canvas, page_w: float, margin: float, generated_by: str, generated_at: str) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.line(margin, margin + 12, page_w - margin, margin + 12)
    c.setFillColorRGB(0.32, 0.32, 0.32)
    c.setFont("Helvetica", 8)
    c.drawString(margin, margin, f"Generated: {generated_at}")
    c.drawRightString(page_w - margin, margin, f"Generated by: {generated_by}")


def _draw_image_area(c: canvas.Canvas, x: float, y: float, w: float, h: float, image: Image.Image | None) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.rect(x, y, w, h, stroke=1, fill=0)

    if image is None:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(x + (w / 2), y + (h / 2) - 3, "Image unavailable")
        return

    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont("Helvetica-Oblique", 7)
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

    pad = max(4.0, min(7.0, w * 0.08))
    metadata_h = max(12.0, h * 0.11)
    inner_x = x + pad
    inner_w = w - (2 * pad)
    inner_h = h - (2 * pad)
    text_and_image_h = max(8.0, inner_h - metadata_h)

    meta_y = y + h - metadata_h
    c.setFillColorRGB(*_CARD_META_BG)
    c.setStrokeColorRGB(*_CARD_META_BG)
    c.rect(x, meta_y, w, metadata_h, stroke=0, fill=1)

    c.setFillColorRGB(*_TEXT_DARK)
    meta_fs = _fit_text("RETAIL", "Helvetica-Bold", (w / 2) - (pad * 2), 8, 6)
    c.setFont("Helvetica-Bold", meta_fs)
    retail = _truncate_text((slot.retail or "").strip() or "-", "Helvetica-Bold", meta_fs, (w / 2) - (pad * 2))
    cpp = _truncate_text((slot.cpp or "").strip() or "-", "Helvetica-Bold", meta_fs, (w / 2) - (pad * 2))
    c.drawString(x + pad, meta_y + (metadata_h - meta_fs) / 2, retail)
    c.drawRightString(x + w - pad, meta_y + (metadata_h - meta_fs) / 2, cpp)

    upc_text = f"UPC {(slot.upc or '').strip() or '-'}"
    item_text = f"ITEM {(slot.item_number or '').strip() or '-'}"
    description = _description_for_slot(slot)

    upc_fs = _fit_protected_single_line_font(upc_text, "Helvetica-Bold", inner_w, 8.0, 4.0)
    item_fs = _fit_protected_single_line_font(item_text, "Helvetica", inner_w, 7.5, 4.0)
    item_lines = _wrap_text_lines(item_text, "Helvetica", item_fs, inner_w)
    desc_fs = 8.0
    min_desc_fs = 2.5
    image_preferred_h = max(24.0, h * 0.34)
    image_min_h = 10.0
    block_gap = 2.0

    desc_lines = _wrap_text_lines(description, "Helvetica", desc_fs, inner_w)
    while desc_fs > min_desc_fs:
        item_lines = _wrap_text_lines(item_text, "Helvetica", item_fs, inner_w)
        desc_lines = _wrap_text_lines(description, "Helvetica", desc_fs, inner_w)
        text_h = (
            (upc_fs + 1.5)
            + block_gap
            + _wrapped_block_height(len(item_lines), item_fs, 1.2)
            + block_gap
            + _wrapped_block_height(len(desc_lines), desc_fs, 1.2)
        )
        if text_h + image_min_h <= text_and_image_h:
            break
        if item_fs > 4.0:
            item_fs = max(4.0, item_fs - 0.25)
        elif upc_fs > 4.0:
            upc_fs = max(4.0, upc_fs - 0.25)
        else:
            desc_fs = max(min_desc_fs, desc_fs - 0.25)

    item_lines = _wrap_text_lines(item_text, "Helvetica", item_fs, inner_w)
    desc_lines = _wrap_text_lines(description, "Helvetica", desc_fs, inner_w)
    text_h = (
        (upc_fs + 1.5)
        + block_gap
        + _wrapped_block_height(len(item_lines), item_fs, 1.2)
        + block_gap
        + _wrapped_block_height(len(desc_lines), desc_fs, 1.2)
    )
    image_h = _compute_image_height(text_and_image_h - text_h, image_preferred_h, image_min_h)

    content_top = y + pad + inner_h - metadata_h
    text_top = content_top - 1.0
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont("Helvetica-Bold", upc_fs)
    c.drawString(inner_x, text_top - upc_fs, upc_text)

    current_y = text_top - (upc_fs + 1.5) - block_gap
    c.setFillColorRGB(*_TEXT_DARK)
    current_y = _draw_wrapped_lines(c, item_lines, inner_x, current_y, "Helvetica", item_fs, 1.2) - block_gap

    c.setFillColorRGB(*_TEXT_DARK)
    current_y = _draw_wrapped_lines(c, desc_lines, inner_x, current_y, "Helvetica", desc_fs, 1.2)

    image_y = y + pad
    image_h = max(0.0, min(image_h, max(0.0, current_y - image_y - 1.5)))
    _draw_image_area(c, inner_x, image_y, inner_w, image_h, image)


def _render_side_page(
    c: canvas.Canvas,
    side_page: SamsSidePage,
    generated_by: str,
    generated_at: str,
    image_cache: dict[str, Image.Image | None],
    warnings: list[str],
) -> tuple[int, int]:
    page_w, page_h = _PAGE_SIZE
    margin = 20.0
    header_h = 42.0
    footer_h = 24.0
    row_gap = 8.0
    col_gap = 6.0

    _draw_header(c, page_w, page_h, side_page.pog, side_page.side, header_h)
    _draw_footer(c, page_w, margin, generated_by, generated_at)

    content_left = margin
    content_right = page_w - margin
    content_top = page_h - header_h - margin
    content_bottom = margin + footer_h + 8
    content_w = content_right - content_left
    content_h = content_top - content_bottom

    sorted_rows = sorted(side_page.rows, key=lambda row: row.row_number)
    row_count = len(sorted_rows)
    if row_count <= 0:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont("Helvetica-Oblique", 11)
        c.drawCentredString(page_w / 2, (content_top + content_bottom) / 2, "No populated rows for this side.")
        return 0, 0

    capacity = side_page.column_limit if side_page.column_limit > 0 else (10 if side_page.side in (2, 4) else 8)
    card_w = (content_w - ((capacity - 1) * col_gap)) / capacity
    card_h = (content_h - ((row_count - 1) * row_gap)) / row_count

    rendered_slots = 0
    missing_image_slots = 0

    for row_index, row in enumerate(sorted_rows):
        row_top = content_top - row_index * (card_h + row_gap)
        card_y = row_top - card_h
        sorted_slots = sorted(row.slots, key=lambda slot: slot.column)
        for slot_index, slot in enumerate(sorted_slots):
            card_x = content_left + slot_index * (card_w + col_gap)
            image, image_warning = _load_slot_image(slot.file_path, image_cache)
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


def render_sams_planogram_pdf(planogram: SamsPlanogram, generated_by: str = "Kendal King") -> SamsPdfRenderResult:
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
