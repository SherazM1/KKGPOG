from __future__ import annotations

import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.holiday_planograms.models import HolidayPlacement, HolidayQpRenderResult


_PAGE_SIZE = portrait(letter)
_HEADER_BLUE = (0.09, 0.31, 0.66)
_HEADER_TEXT = (1.0, 1.0, 1.0)
_WHITE = (1.0, 1.0, 1.0)
_CARD_BORDER = (0.72, 0.72, 0.72)
_CARD_META_BG = (0.95, 0.96, 0.98)
_IMAGE_BORDER = (0.86, 0.86, 0.86)
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
class _QpLayoutMetrics:
    page_w: float
    page_h: float
    margin: float
    header_h: float
    footer_h: float
    body_bottom_gap: float
    row_gap: float
    col_gap: float


@dataclass(frozen=True)
class _QpGridLayout:
    card_w: float
    card_h: float
    grid_left: float
    grid_w: float
    col_gap: float


def _resolve_assets_root() -> Path:
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


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
    return f"{out.rstrip()}{suffix}"


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


def _wrapped_block_height(line_count: int, font_size: float, line_gap: float = 1.2) -> float:
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
    line_gap: float = 1.2,
) -> float:
    y = top_y
    step = font_size + line_gap
    c.setFont(font_name, font_size)
    for line in lines:
        c.drawString(x, y - font_size, line)
        y -= step
    return y


def _format_cpp_value(raw_value: object) -> str:
    text = "" if raw_value is None else str(raw_value).strip()
    if text == "":
        return "CPP -"
    normalized = text.replace(",", "")
    try:
        value = float(normalized)
        if value.is_integer():
            formatted = str(int(value))
        else:
            formatted = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"CPP {formatted}"
    except ValueError:
        return f"CPP {text}"


def _format_price_value(raw_value: object) -> str:
    text = "" if raw_value is None else str(raw_value).strip()
    if text == "":
        return "$-"
    compact = text.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    if compact.startswith("$"):
        return compact
    try:
        value = float(compact.replace(",", ""))
        return f"${value:.2f}"
    except ValueError:
        return compact


def _contain_image_rect(
    source_w: float,
    source_h: float,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
) -> tuple[float, float, float, float]:
    if source_w <= 0 or source_h <= 0 or box_w <= 0 or box_h <= 0:
        return box_x, box_y, 0.0, 0.0
    scale = min(box_w / source_w, box_h / source_h)
    draw_w = source_w * scale
    draw_h = source_h * scale
    draw_x = box_x + (box_w - draw_w) / 2
    draw_y = box_y + (box_h - draw_h) / 2
    return draw_x, draw_y, draw_w, draw_h


def _compute_image_height(card_h: float, text_and_image_h: float, text_h: float) -> float:
    available = max(0.0, text_and_image_h - text_h)
    preferred = max(42.0, max(card_h * 0.48, text_and_image_h * 0.5))
    minimum = max(28.0, min(card_h * 0.34, text_and_image_h * 0.42))
    if available >= preferred:
        return preferred
    return min(available, max(minimum, available))


def _panel_layout(placements: list[HolidayPlacement]) -> dict[int, list[HolidayPlacement]]:
    panels: dict[int, list[HolidayPlacement]] = defaultdict(list)
    for placement in placements:
        panels[placement.panel_index].append(placement)
    return dict(sorted(panels.items()))


def _compute_page_dimensions_for_panel(placements: list[HolidayPlacement]) -> _QpLayoutMetrics:
    base_w, base_h = _PAGE_SIZE
    rows = max((placement.row for placement in placements), default=1)
    cols = max((placement.column for placement in placements), default=1)
    row_gap = _clamp(7.0 - (rows - 4) * 0.4, 5.0, 7.0)
    col_gap = _clamp(base_w * 0.03, 10.0, 18.0)

    needed_content_w = (cols * 136.0) + ((cols - 1) * col_gap)
    needed_content_h = (rows * 130.0) + ((rows - 1) * row_gap)
    page_w = max(base_w, needed_content_w + (2 * _BASE_MARGIN))
    page_h = max(
        base_h,
        needed_content_h + _BASE_HEADER_H + (2 * _BASE_MARGIN) + _BASE_FOOTER_H + _BASE_BODY_BOTTOM_GAP,
    )
    return _QpLayoutMetrics(
        page_w=page_w,
        page_h=page_h,
        margin=_BASE_MARGIN,
        header_h=_BASE_HEADER_H,
        footer_h=_BASE_FOOTER_H,
        body_bottom_gap=_BASE_BODY_BOTTOM_GAP,
        row_gap=row_gap,
        col_gap=col_gap,
    )


def _compute_card_dimensions(metrics: _QpLayoutMetrics, row_count: int, col_count: int) -> tuple[float, float]:
    content_w = max(1.0, metrics.page_w - (2 * metrics.margin))
    content_h = max(
        1.0,
        metrics.page_h - metrics.header_h - (2 * metrics.margin) - metrics.footer_h - metrics.body_bottom_gap,
    )
    card_w = (content_w - ((col_count - 1) * metrics.col_gap)) / max(1, col_count)
    card_h = (content_h - ((row_count - 1) * metrics.row_gap)) / max(1, row_count)
    return max(1.0, card_w), max(1.0, card_h)


def _preferred_max_card_width(page_w: float, col_count: int) -> float:
    if col_count <= 1:
        return page_w * 0.42
    return _clamp(page_w * 0.22, 118.0, 138.0)


def _compute_grid_layout(metrics: _QpLayoutMetrics, row_count: int, col_count: int) -> _QpGridLayout:
    natural_card_w, card_h = _compute_card_dimensions(metrics, row_count, col_count)
    preferred_max_w = _preferred_max_card_width(metrics.page_w, col_count)
    card_w = min(natural_card_w, preferred_max_w)
    grid_w = (col_count * card_w) + ((col_count - 1) * metrics.col_gap)
    grid_left = (metrics.page_w - grid_w) / 2
    return _QpGridLayout(
        card_w=max(1.0, card_w),
        card_h=card_h,
        grid_left=grid_left,
        grid_w=grid_w,
        col_gap=metrics.col_gap,
    )


def _open_slot_image(path_text: str, cache: dict[str, Image.Image | None]) -> tuple[Image.Image | None, str]:
    file_path = (path_text or "").strip()
    if not file_path:
        return None, ""
    if file_path in cache:
        cached = cache[file_path]
        if cached is None:
            return None, "cached load failure"
        return cached, ""
    path = Path(file_path)
    if not path.is_file():
        cache[file_path] = None
        return None, f"Image path is not available: {file_path}"
    try:
        with Image.open(path) as image:
            image.load()
            loaded = ImageOps.exif_transpose(image).convert("RGBA")
            cache[file_path] = loaded
            return loaded, ""
    except Exception as exc:
        cache[file_path] = None
        return None, f"Image could not be opened: {file_path} ({exc})"


def _draw_header(c: canvas.Canvas, page_w: float, page_h: float, header_h: float, side_label: str) -> None:
    y = page_h - header_h
    c.setFillColorRGB(*_HEADER_BLUE)
    c.setStrokeColorRGB(*_HEADER_BLUE)
    c.rect(0, y, page_w, header_h, stroke=0, fill=1)

    header_font = _header_font_name()
    title = "XMAS 2026 - HOLIDAY QUARTER PALLET"
    c.setFillColorRGB(*_HEADER_TEXT)
    c.setFont(header_font, 12)
    c.drawString(20, y + (header_h / 2) - 4.5, "POG")

    title_fs = _fit_text(title, header_font, page_w * 0.58, 15.5, 9.5)
    title = _truncate_text(title, header_font, title_fs, page_w * 0.58)
    title_w = pdfmetrics.stringWidth(title, header_font, title_fs)
    c.setFont(header_font, title_fs)
    c.drawString((page_w - title_w) / 2, y + (header_h / 2) - (title_fs / 2) + 0.5, title)

    side_fs = _fit_text(side_label, header_font, page_w * 0.25, 11.5, 7.0)
    side_label = _truncate_text(side_label, header_font, side_fs, page_w * 0.25)
    c.setFont(header_font, side_fs)
    side_w = pdfmetrics.stringWidth(side_label, header_font, side_fs)
    c.drawString(page_w - side_w - 20, y + (header_h / 2) - (side_fs / 2) + 0.5, side_label)


def _draw_footer(c: canvas.Canvas, page_w: float, margin: float, generated_at: str) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.line(margin, margin + 12, page_w - margin, margin + 12)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(_footer_font_name(), 8)
    c.drawString(margin, margin, f"Generated: {generated_at}")
    c.drawRightString(page_w - margin, margin, "Generated by Kendal King")


def _draw_image_area(c: canvas.Canvas, x: float, y: float, w: float, h: float, image: Image.Image | None) -> None:
    c.setStrokeColorRGB(*_IMAGE_BORDER)
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, stroke=1, fill=0)

    if image is None:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont(_FONT_ITALIC, 7.5)
        c.drawCentredString(x + (w / 2), y + (h / 2) - 3, "IMAGE MISSING")
        return

    source_w, source_h = image.size
    draw_x, draw_y, draw_w, draw_h = _contain_image_rect(source_w, source_h, x, y, w, h)
    if draw_w <= 0 or draw_h <= 0:
        c.setFillColorRGB(*_TEXT_MUTED)
        c.setFont(_FONT_ITALIC, 7.5)
        c.drawCentredString(x + (w / 2), y + (h / 2) - 3, "IMAGE MISSING")
        return
    c.drawImage(ImageReader(image), draw_x, draw_y, draw_w, draw_h, preserveAspectRatio=True, mask="auto")


def _draw_qp_slot(
    c: canvas.Canvas,
    placement: HolidayPlacement,
    x: float,
    y: float,
    w: float,
    h: float,
    image: Image.Image | None,
) -> None:
    c.setFillColorRGB(*_WHITE)
    c.setStrokeColorRGB(*_CARD_BORDER)
    c.setLineWidth(0.7)
    c.rect(x, y, w, h, stroke=1, fill=1)

    size_scale = _clamp(min(w / 176.0, h / 132.0), 0.86, 1.35)
    pad = _clamp(w * 0.045, 4.0, 7.0)
    metadata_h = _clamp(h * 0.125, 14.0, 20.0)
    inner_x = x + pad
    inner_w = w - (2 * pad)
    inner_h = h - (2 * pad)
    text_and_image_h = max(8.0, inner_h - metadata_h)

    meta_y = y + h - metadata_h
    c.setFillColorRGB(*_CARD_META_BG)
    c.setStrokeColorRGB(*_CARD_META_BG)
    c.rect(x, meta_y, w, metadata_h, stroke=0, fill=1)

    price_text = _format_price_value(placement.denomination)
    cpp_text = _format_cpp_value(placement.cpp)
    meta_fs = _fit_protected_single_line_font(price_text, _FONT_BODY_BOLD, inner_w * 0.52, 9.6 * size_scale, 6.0)
    cpp_fs = _fit_protected_single_line_font(cpp_text, _FONT_BODY_BOLD, inner_w * 0.43, 9.6 * size_scale, 6.0)
    baseline = meta_y + (metadata_h - max(meta_fs, cpp_fs)) / 2 + 1.0
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(_FONT_BODY_BOLD, meta_fs)
    c.drawString(inner_x, baseline, _truncate_text(price_text, _FONT_BODY_BOLD, meta_fs, inner_w * 0.52))
    c.setFont(_FONT_BODY_BOLD, cpp_fs)
    c.drawRightString(x + w - pad, baseline, _truncate_text(cpp_text, _FONT_BODY_BOLD, cpp_fs, inner_w * 0.43))

    upc_text = f"UPC {(placement.product_upc or '').strip() or '-'}"
    item_text = f"ITEM {(placement.item_number or '').strip() or '-'}"
    upc_fs = _fit_protected_single_line_font(upc_text, _FONT_BODY_BOLD, inner_w, 6.8 * size_scale, 4.4)
    item_fs = _fit_protected_single_line_font(item_text, _FONT_BODY_REGULAR, inner_w, 6.4 * size_scale, 4.2)

    name = (placement.product_name or "").strip() or "No product name"
    name_fs = max(5.9, 6.9 * size_scale)
    min_name_fs = 5.0
    name_lines = _wrap_text_lines(name, _FONT_BODY_REGULAR, name_fs, inner_w)
    max_name_lines = 3
    while name_fs > min_name_fs:
        name_lines = _wrap_text_lines(name, _FONT_BODY_REGULAR, name_fs, inner_w)
        if len(name_lines) > max_name_lines:
            name_lines = name_lines[:max_name_lines]
            name_lines[-1] = _truncate_text(name_lines[-1], _FONT_BODY_REGULAR, name_fs, inner_w)
        text_h = (
            (upc_fs + 1.0)
            + (item_fs + 1.0)
            + _wrapped_block_height(len(name_lines), name_fs, 1.0)
            + 6.0
        )
        image_h = _compute_image_height(h, text_and_image_h, text_h)
        if text_h + image_h <= text_and_image_h:
            break
        name_fs = max(min_name_fs, name_fs - 0.25)

    name_lines = _wrap_text_lines(name, _FONT_BODY_REGULAR, name_fs, inner_w)
    if len(name_lines) > max_name_lines:
        name_lines = name_lines[:max_name_lines]
        name_lines[-1] = _truncate_text(name_lines[-1], _FONT_BODY_REGULAR, name_fs, inner_w)
    text_h = (
        (upc_fs + 1.0)
        + (item_fs + 1.0)
        + _wrapped_block_height(len(name_lines), name_fs, 1.0)
        + 6.0
    )
    image_h = _compute_image_height(h, text_and_image_h, text_h)

    content_top = y + pad + inner_h - metadata_h
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(_FONT_BODY_BOLD, upc_fs)
    c.drawString(inner_x, content_top - upc_fs - 1.0, upc_text)
    current_y = content_top - upc_fs - 3.0
    c.setFont(_FONT_BODY_REGULAR, item_fs)
    c.drawString(inner_x, current_y - item_fs, item_text)
    current_y -= item_fs + 3.0
    after_name_y = _draw_wrapped_lines(c, name_lines, inner_x, current_y, _FONT_BODY_REGULAR, name_fs, 1.0)

    image_y = y + pad
    image_h = max(0.0, min(image_h, max(0.0, after_name_y - image_y - 2.0)))
    _draw_image_area(c, inner_x, image_y, inner_w, image_h, image)


def _missing_image_rows(
    placements: Iterable[HolidayPlacement],
    missing_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    all_placements = list(placements)
    total_by_product = Counter((placement.item_number, placement.product_upc) for placement in all_placements)
    by_product: dict[tuple[str, str], dict[str, object]] = {}
    for placement in all_placements:
        key = (placement.item_number, placement.product_upc)
        effectively_missing = placement.image_status != "resolved" or (missing_keys is not None and key in missing_keys)
        if not effectively_missing:
            continue
        row = by_product.setdefault(
            key,
            {
                "Product Name": placement.product_name,
                "Denom / Load Range": placement.denomination,
                "Product 12 Digit UPC": placement.product_upc,
                "WM Item Number": placement.item_number,
                "QP Facings": total_by_product.get(key, 0),
                "Missing Placement Count": 0,
                "Image Status": placement.image_status,
            },
        )
        row["Missing Placement Count"] = int(row["Missing Placement Count"]) + 1
        if placement.image_status == "resolved":
            row["Image Status"] = "placeholder_bad_image"
    return sorted(by_product.values(), key=lambda row: (str(row["Product Name"]), str(row["WM Item Number"])))


def _render_preview_from_pdf(pdf_bytes: bytes) -> bytes:
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA"))
        if not pages:
            return b""
        preview_w = max(page.width for page in pages)
        gap = 18
        preview_h = sum(page.height for page in pages) + max(0, len(pages) - 1) * gap
        stitched = Image.new("RGBA", (preview_w, preview_h), (248, 249, 252, 255))
        y = 0
        for page in pages:
            stitched.alpha_composite(page, ((preview_w - page.width) // 2, y))
            y += page.height + gap
        output = io.BytesIO()
        stitched.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        return b""


def render_holiday_qp(placements: list[HolidayPlacement]) -> HolidayQpRenderResult:
    if not placements:
        raise ValueError("No resolved Holiday QP placements are available to render.")

    panels = _panel_layout(placements)
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=_PAGE_SIZE)
    image_cache: dict[str, Image.Image | None] = {}
    image_slots = 0
    placeholder_slots = 0
    skipped_slots = 0
    warnings: list[str] = []
    render_rows: list[dict[str, object]] = []
    missing_keys: set[tuple[str, str]] = set()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for panel_index, panel_placements in panels.items():
        metrics = _compute_page_dimensions_for_panel(panel_placements)
        c.setPageSize((metrics.page_w, metrics.page_h))
        rows = max((placement.row for placement in panel_placements), default=1)
        cols = max((placement.column for placement in panel_placements), default=1)
        panel_name = panel_placements[0].panel_name if panel_placements else f"panel_{panel_index}"
        side_label = f"SIDE {panel_index} - {panel_name}"

        _draw_header(c, metrics.page_w, metrics.page_h, metrics.header_h, side_label)
        _draw_footer(c, metrics.page_w, metrics.margin, generated_at)

        grid = _compute_grid_layout(metrics, rows, cols)
        content_top = metrics.page_h - metrics.header_h - metrics.margin

        ordered = sorted(panel_placements, key=lambda placement: (placement.row, placement.column))
        for placement in ordered:
            x = grid.grid_left + (placement.column - 1) * (grid.card_w + grid.col_gap)
            row_top = content_top - (placement.row - 1) * (grid.card_h + metrics.row_gap)
            y = row_top - grid.card_h
            image, image_warning = _open_slot_image(placement.image_path, image_cache) if placement.image_status == "resolved" else (None, "")
            if image is not None:
                image_slots += 1
                content_type = "image"
                image_status = placement.image_status
            else:
                placeholder_slots += 1
                content_type = "placeholder"
                image_status = placement.image_status
                if image_warning:
                    warnings.append(image_warning)
                    missing_keys.add((placement.item_number, placement.product_upc))
                    image_status = "placeholder_bad_image"
            _draw_qp_slot(c, placement, x, y, grid.card_w, grid.card_h, image)
            render_rows.append(
                {
                    "Panel": panel_name,
                    "Row": placement.row,
                    "Column": placement.column,
                    "Product Name": placement.product_name,
                    "UPC": placement.product_upc,
                    "ITEM": placement.item_number,
                    "Price": price_text if (price_text := _format_price_value(placement.denomination)) else "$-",
                    "CPP": placement.cpp,
                    "Metadata Fields": "Price; CPP; UPC; ITEM; Product Name",
                    "Content Type": content_type,
                    "Image Status": image_status,
                    "Image Path": placement.image_path,
                    "Warning": image_warning,
                }
            )

        c.showPage()

    c.save()
    pdf_bytes = pdf_buffer.getvalue()
    preview_png_bytes = _render_preview_from_pdf(pdf_bytes)

    return HolidayQpRenderResult(
        pdf_bytes=pdf_bytes,
        preview_png_bytes=preview_png_bytes,
        panels_rendered=len(panels),
        slots_rendered=len(render_rows),
        image_slots=image_slots,
        placeholder_slots=placeholder_slots,
        skipped_slots=skipped_slots,
        missing_image_rows=_missing_image_rows(placements, missing_keys),
        render_rows=render_rows,
        warnings=warnings,
    )
