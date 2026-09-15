from __future__ import annotations

import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.holiday_planograms.geometry_nre import SLOT_AMBIGUOUS, SLOT_FILLER, SLOT_MERCHANDISE
from app.holiday_planograms.models import HolidayPlacement, HolidayQpRenderResult
from app.holiday_planograms.render_qp import (
    _CARD_BORDER,
    _CARD_META_BG,
    _FONT_BODY_BOLD,
    _FONT_BODY_REGULAR,
    _FONT_ITALIC,
    _HEADER_BLUE,
    _HEADER_TEXT,
    _TEXT_DARK,
    _TEXT_MUTED,
    _WHITE,
    _draw_footer,
    _draw_image_area,
    _draw_wrapped_lines,
    _contain_image_rect,
    _fit_text,
    _fit_protected_single_line_font,
    _format_cpp_value,
    _format_price_value,
    _header_font_name,
    _open_slot_image,
    _render_preview_from_pdf,
    _truncate_text,
    _wrap_text_lines,
    _wrapped_block_height,
)


_BASE_PAGE_W = 612.0
_BASE_MARGIN = 24.0
_HEADER_H = 42.0
_FOOTER_H = 24.0
_BODY_TOP_GAP = 34.0
_BODY_BOTTOM_GAP = 12.0
_MIN_CARD_W = 96.0
_MIN_CARD_H = 112.0
_MAX_PAGE_H = 1750.0
_FILLER_BG = (0.91, 0.96, 0.91)
_FILLER_BORDER = (0.62, 0.76, 0.62)
_FILLER_TEXT = (0.35, 0.49, 0.35)
_GROUP_ROWS = (
    ("ZIFT", "NINTENDO"),
    ("PLAYSTATION", "XBOX"),
    ("ROBLOX", "FORTNITE"),
    ("META", "RAZER GOLD"),
    ("POKEMON GO", "GOOGLE PLAY"),
)
_GROUP_CARD_COUNT = 4
_TEMPLATE_CARD_W = 82.0
_TEMPLATE_CARD_H = 104.0
_TEMPLATE_CARD_GAP = 8.0
_TEMPLATE_GROUP_GAP = 32.0
_TEMPLATE_ROW_GAP = 36.0
_TEMPLATE_HEADING_H = 12.0
_TEMPLATE_FOOTPRINT_GAP = 42.0
_TEMPLATE_TOP_GAP = 32.0


@dataclass(frozen=True)
class _TemplateSlot:
    slot_type: str
    group_name: str
    group_index: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class _TemplateHeading:
    group_name: str
    x: float
    y: float
    w: float


@dataclass(frozen=True)
class _NreTemplate:
    configuration: str
    page_w: float
    page_h: float
    content_left: float
    content_bottom: float
    headings: tuple[_TemplateHeading, ...]
    slots: tuple[_TemplateSlot, ...]


@dataclass(frozen=True)
class _NreLayout:
    page_w: float
    page_h: float
    margin: float
    header_h: float
    footer_h: float
    scale: float
    src_left: float
    src_top: float
    body_left: float
    body_top: float


def _slot_type(placement: HolidayPlacement) -> str:
    if placement.placement_status == "non_product":
        return SLOT_FILLER
    if placement.placement_status == "ambiguous_slot":
        return SLOT_AMBIGUOUS
    return SLOT_MERCHANDISE


def _placement_center(placement: HolidayPlacement) -> tuple[float, float]:
    x1, y1, x2, y2 = placement.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _compute_layout(placements: list[HolidayPlacement]) -> _NreLayout:
    left = min(placement.bbox[0] for placement in placements)
    top = min(placement.bbox[1] for placement in placements)
    right = max(placement.bbox[2] for placement in placements)
    bottom = max(placement.bbox[3] for placement in placements)
    src_w = max(1.0, right - left)
    src_h = max(1.0, bottom - top)

    merchandise = [placement for placement in placements if _slot_type(placement) == SLOT_MERCHANDISE]
    avg_card_w = sum(max(1.0, placement.bbox[2] - placement.bbox[0]) for placement in merchandise) / max(1, len(merchandise))
    avg_card_h = sum(max(1.0, placement.bbox[3] - placement.bbox[1]) for placement in merchandise) / max(1, len(merchandise))
    scale = max(_MIN_CARD_W / avg_card_w, _MIN_CARD_H / avg_card_h, 1.0)

    page_w = max(_BASE_PAGE_W, src_w * scale + (2 * _BASE_MARGIN))
    page_h = src_h * scale + _HEADER_H + _FOOTER_H + (2 * _BASE_MARGIN) + _BODY_TOP_GAP + _BODY_BOTTOM_GAP
    if page_h > _MAX_PAGE_H:
        available_h = _MAX_PAGE_H - _HEADER_H - _FOOTER_H - (2 * _BASE_MARGIN) - _BODY_TOP_GAP - _BODY_BOTTOM_GAP
        scale = min(scale, available_h / src_h)
        page_w = max(_BASE_PAGE_W, src_w * scale + (2 * _BASE_MARGIN))
        page_h = _MAX_PAGE_H

    body_left = (page_w - src_w * scale) / 2
    body_top = page_h - _HEADER_H - _BASE_MARGIN - _BODY_TOP_GAP
    return _NreLayout(
        page_w=page_w,
        page_h=page_h,
        margin=_BASE_MARGIN,
        header_h=_HEADER_H,
        footer_h=_FOOTER_H,
        scale=scale,
        src_left=left,
        src_top=top,
        body_left=body_left,
        body_top=body_top,
    )


def _placement_rect(placement: HolidayPlacement, layout: _NreLayout) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = placement.bbox
    x = layout.body_left + (x1 - layout.src_left) * layout.scale
    y = layout.body_top - (y2 - layout.src_top) * layout.scale
    w = max(1.0, (x2 - x1) * layout.scale)
    h = max(1.0, (y2 - y1) * layout.scale)
    return x, y, w, h


def _compute_render_rects(placements: list[HolidayPlacement], layout: _NreLayout) -> list[tuple[float, float, float, float]]:
    rects = [_placement_rect(placement, layout) for placement in placements]
    rows: dict[int, list[int]] = defaultdict(list)
    for index, placement in enumerate(placements):
        rows[placement.row].append(index)

    for row_indexes in rows.values():
        ordered = sorted(row_indexes, key=lambda index: _placement_center(placements[index])[0])
        for position, index in enumerate(ordered):
            placement = placements[index]
            center_x, _center_y = _placement_center(placement)
            x, y, w, h = rects[index]
            available_source_w = placement.bbox[2] - placement.bbox[0]
            if position > 0:
                prev_center_x, _ = _placement_center(placements[ordered[position - 1]])
                available_source_w = min(available_source_w, center_x - prev_center_x)
            if position < len(ordered) - 1:
                next_center_x, _ = _placement_center(placements[ordered[position + 1]])
                available_source_w = min(available_source_w, next_center_x - center_x)
            if available_source_w > 0:
                adjusted_w = min(w, available_source_w * layout.scale * 0.94)
                rects[index] = (x + (w - adjusted_w) / 2, y, adjusted_w, h)
    return rects


def _draw_header(c: canvas.Canvas, layout: _NreLayout, configuration: str) -> None:
    y = layout.page_h - layout.header_h
    c.setFillColorRGB(*_HEADER_BLUE)
    c.setStrokeColorRGB(*_HEADER_BLUE)
    c.rect(0, y, layout.page_w, layout.header_h, stroke=0, fill=1)

    header_font = _header_font_name()
    title = "XMAS 2026 - HOLIDAY NRE"
    c.setFillColorRGB(*_HEADER_TEXT)
    c.setFont(header_font, 12)
    c.drawString(20, y + (layout.header_h / 2) - 4.5, "POG")

    title_fs = _fit_text(title, header_font, layout.page_w * 0.58, 15.5, 9.5)
    title = _truncate_text(title, header_font, title_fs, layout.page_w * 0.58)
    title_w = pdfmetrics.stringWidth(title, header_font, title_fs)
    c.setFont(header_font, title_fs)
    c.drawString((layout.page_w - title_w) / 2, y + (layout.header_h / 2) - (title_fs / 2) + 0.5, title)

    config_label = configuration.upper()
    side_fs = _fit_text(config_label, header_font, layout.page_w * 0.25, 11.5, 7.0)
    c.setFont(header_font, side_fs)
    side_w = pdfmetrics.stringWidth(config_label, header_font, side_fs)
    c.drawString(layout.page_w - side_w - 20, y + (layout.header_h / 2) - (side_fs / 2) + 0.5, config_label)


def _heading_for_reference(text: str) -> str:
    normalized = " ".join((text or "").upper().split())
    if "NINTENDO" in normalized:
        return "NINTENDO"
    if "PLAYSTATION" in normalized or "SONY" in normalized:
        return "PLAYSTATION"
    if "ROBLOX" in normalized:
        return "ROBLOX"
    if "FORTNITE" in normalized:
        return "FORTNITE"
    if "META" in normalized:
        return "META"
    if "RAZER" in normalized:
        return "RAZER GOLD"
    if "POKEMON" in normalized:
        return "POKEMON GO"
    if "GOOGLE" in normalized:
        return "GOOGLE PLAY"
    if "XBOX" in normalized:
        return "XBOX"
    if "ZIFT" in normalized or "ONLINE EXCHANGE" in normalized:
        return "ZIFT"
    return ""


def _left_filler_rows(configuration: str) -> int:
    if configuration in {"4x78", "4x60"}:
        return 5
    return 0


def _top_filler_slots(configuration: str) -> int:
    if configuration == "3x60":
        return 4
    return 0


def _bottom_filler_slots(configuration: str) -> int:
    if configuration == "4x78":
        return 10
    if configuration == "3x78":
        return 8
    return 0


def _template_for_configuration(configuration: str) -> _NreTemplate:
    if configuration not in {"4x78", "3x78", "3x60", "4x60"}:
        raise ValueError(f"Unsupported Holiday NRE configuration: {configuration}.")

    group_w = (_GROUP_CARD_COUNT * _TEMPLATE_CARD_W) + ((_GROUP_CARD_COUNT - 1) * _TEMPLATE_CARD_GAP)
    group_row_h = _TEMPLATE_HEADING_H + _TEMPLATE_CARD_H
    left_filler_w = (2 * _TEMPLATE_CARD_W) + _TEMPLATE_CARD_GAP if _left_filler_rows(configuration) else 0.0
    left_gap = _TEMPLATE_FOOTPRINT_GAP if left_filler_w else 0.0
    group_area_w = (2 * group_w) + _TEMPLATE_GROUP_GAP
    bottom_filler_count = _bottom_filler_slots(configuration)
    bottom_filler_w = (
        (bottom_filler_count * _TEMPLATE_CARD_W) + ((bottom_filler_count - 1) * _TEMPLATE_CARD_GAP)
        if bottom_filler_count
        else 0.0
    )
    template_area_w = max(group_area_w, bottom_filler_w)
    body_w = left_filler_w + left_gap + template_area_w

    top_filler_count = _top_filler_slots(configuration)
    has_top_filler = top_filler_count > 0
    has_bottom_filler = bottom_filler_count > 0
    top_filler_h = _TEMPLATE_CARD_H + _TEMPLATE_ROW_GAP if has_top_filler else 0.0
    bottom_filler_h = _TEMPLATE_ROW_GAP + _TEMPLATE_CARD_H if has_bottom_filler else 0.0
    group_rows_h = (len(_GROUP_ROWS) * group_row_h) + ((len(_GROUP_ROWS) - 1) * _TEMPLATE_ROW_GAP)
    body_h = _TEMPLATE_TOP_GAP + top_filler_h + group_rows_h + bottom_filler_h

    page_w = body_w + (2 * _BASE_MARGIN)
    page_h = body_h + _HEADER_H + _FOOTER_H + (2 * _BASE_MARGIN) + _BODY_BOTTOM_GAP
    content_left = _BASE_MARGIN
    content_top = page_h - _HEADER_H - _BASE_MARGIN - _TEMPLATE_TOP_GAP

    slots: list[_TemplateSlot] = []
    headings: list[_TemplateHeading] = []
    left_filler_x = content_left
    groups_x = content_left + left_filler_w + left_gap
    group_area_x = groups_x + (template_area_w - group_area_w) / 2
    y_cursor = content_top

    if has_top_filler:
        total_top_w = (top_filler_count * _TEMPLATE_CARD_W) + ((top_filler_count - 1) * _TEMPLATE_CARD_GAP)
        top_x = groups_x + (template_area_w - total_top_w) / 2
        y_cursor -= _TEMPLATE_CARD_H
        for index in range(top_filler_count):
            slots.append(
                _TemplateSlot(
                    slot_type=SLOT_FILLER,
                    group_name="FILLER",
                    group_index=index,
                    x=top_x + index * (_TEMPLATE_CARD_W + _TEMPLATE_CARD_GAP),
                    y=y_cursor,
                    w=_TEMPLATE_CARD_W,
                    h=_TEMPLATE_CARD_H,
                )
            )
        y_cursor -= _TEMPLATE_ROW_GAP

    for row_index, (left_group, right_group) in enumerate(_GROUP_ROWS):
        heading_y = y_cursor - _TEMPLATE_HEADING_H + 2.0
        card_y = y_cursor - group_row_h
        if row_index < _left_filler_rows(configuration):
            for filler_col in range(2):
                slots.append(
                    _TemplateSlot(
                        slot_type=SLOT_FILLER,
                        group_name="FILLER",
                        group_index=(row_index * 2) + filler_col,
                        x=left_filler_x + filler_col * (_TEMPLATE_CARD_W + _TEMPLATE_CARD_GAP),
                        y=card_y,
                        w=_TEMPLATE_CARD_W,
                        h=_TEMPLATE_CARD_H,
                    )
                )
        for side_index, group_name in enumerate((left_group, right_group)):
            group_x = group_area_x + side_index * (group_w + _TEMPLATE_GROUP_GAP)
            headings.append(_TemplateHeading(group_name=group_name, x=group_x, y=heading_y, w=group_w))
            for card_index in range(_GROUP_CARD_COUNT):
                slots.append(
                    _TemplateSlot(
                        slot_type=SLOT_MERCHANDISE,
                        group_name=group_name,
                        group_index=card_index,
                        x=group_x + card_index * (_TEMPLATE_CARD_W + _TEMPLATE_CARD_GAP),
                        y=card_y,
                        w=_TEMPLATE_CARD_W,
                        h=_TEMPLATE_CARD_H,
                    )
                )
        y_cursor = card_y - _TEMPLATE_ROW_GAP

    if has_bottom_filler:
        y_cursor -= _TEMPLATE_CARD_H
        total_bottom_w = (bottom_filler_count * _TEMPLATE_CARD_W) + ((bottom_filler_count - 1) * _TEMPLATE_CARD_GAP)
        bottom_x = groups_x + (template_area_w - total_bottom_w) / 2
        for index in range(bottom_filler_count):
            slots.append(
                _TemplateSlot(
                    slot_type=SLOT_FILLER,
                    group_name="FILLER",
                    group_index=index,
                    x=bottom_x + index * (_TEMPLATE_CARD_W + _TEMPLATE_CARD_GAP),
                    y=y_cursor,
                    w=_TEMPLATE_CARD_W,
                    h=_TEMPLATE_CARD_H,
                )
            )

    return _NreTemplate(
        configuration=configuration,
        page_w=page_w,
        page_h=page_h,
        content_left=content_left,
        content_bottom=_BASE_MARGIN + _FOOTER_H + _BODY_BOTTOM_GAP,
        headings=tuple(headings),
        slots=tuple(slots),
    )


def _grouped_merchandise(placements: list[HolidayPlacement]) -> dict[str, list[HolidayPlacement]]:
    grouped: dict[str, list[HolidayPlacement]] = {group: [] for row in _GROUP_ROWS for group in row}
    for placement in sorted(placements, key=lambda item: (_placement_center(item)[1], _placement_center(item)[0])):
        if _slot_type(placement) != SLOT_MERCHANDISE:
            continue
        group_name = _heading_for_reference(placement.reference_text)
        if group_name in grouped:
            grouped[group_name].append(placement)
    return grouped


def _template_assignments(
    placements: list[HolidayPlacement],
    template: _NreTemplate,
) -> list[tuple[HolidayPlacement | None, _TemplateSlot]]:
    grouped = _grouped_merchandise(placements)
    filler_placements = [placement for placement in placements if _slot_type(placement) == SLOT_FILLER]
    filler_index = 0
    assignments: list[tuple[HolidayPlacement | None, _TemplateSlot]] = []

    for slot in template.slots:
        if slot.slot_type == SLOT_FILLER:
            placement = filler_placements[filler_index] if filler_index < len(filler_placements) else None
            filler_index += 1
            assignments.append((placement, slot))
            continue
        group_items = grouped.get(slot.group_name, [])
        placement = group_items[slot.group_index] if slot.group_index < len(group_items) else None
        assignments.append((placement, slot))
    return assignments


def _draw_template_headings(c: canvas.Canvas, template: _NreTemplate) -> None:
    for heading in template.headings:
        fs = _fit_text(heading.group_name, _FONT_BODY_BOLD, heading.w, 9.2, 6.0)
        c.setFillColorRGB(*_TEXT_DARK)
        c.setFont(_FONT_BODY_BOLD, fs)
        c.drawCentredString(heading.x + heading.w / 2, heading.y, heading.group_name)


def _draw_group_headings(c: canvas.Canvas, placements: list[HolidayPlacement], layout: _NreLayout) -> None:
    merchandise = [placement for placement in placements if _slot_type(placement) == SLOT_MERCHANDISE]
    rows: dict[int, list[HolidayPlacement]] = defaultdict(list)
    for placement in merchandise:
        rows[placement.row].append(placement)

    for row_placements in rows.values():
        ordered = sorted(row_placements, key=lambda placement: _placement_center(placement)[0])
        group_start = 0
        current_heading = _heading_for_reference(ordered[0].reference_text)
        for index in range(1, len(ordered) + 1):
            next_heading = _heading_for_reference(ordered[index].reference_text) if index < len(ordered) else None
            if next_heading == current_heading:
                continue
            group = ordered[group_start:index]
            if current_heading:
                rects = [_placement_rect(placement, layout) for placement in group]
                left = min(rect[0] for rect in rects)
                right = max(rect[0] + rect[2] for rect in rects)
                top = max(rect[1] + rect[3] for rect in rects)
                heading_w = right - left
                fs = _fit_text(current_heading, _FONT_BODY_BOLD, heading_w - 4.0, 8.6, 5.5)
                label = _truncate_text(current_heading, _FONT_BODY_BOLD, fs, heading_w - 4.0)
                c.setFillColorRGB(*_TEXT_DARK)
                c.setFont(_FONT_BODY_BOLD, fs)
                c.drawCentredString(left + heading_w / 2, top + 5.0, label)
            group_start = index
            current_heading = next_heading or ""


def _draw_filler_block(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str = "FILLER") -> None:
    c.setFillColorRGB(*_FILLER_BG)
    c.setStrokeColorRGB(*_FILLER_BORDER)
    c.setLineWidth(0.6)
    c.rect(x, y, w, h, stroke=1, fill=1)
    c.setFillColorRGB(*_FILLER_TEXT)
    c.setFont(_FONT_BODY_BOLD, min(7.0, max(4.8, h * 0.12)))
    c.drawCentredString(x + w / 2, y + h / 2 - 2.5, label)


def _draw_ambiguous_block(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.setFillColorRGB(1.0, 0.95, 0.88)
    c.setStrokeColorRGB(0.86, 0.58, 0.28)
    c.setLineWidth(0.6)
    c.rect(x, y, w, h, stroke=1, fill=1)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(_FONT_ITALIC, min(7.0, max(4.8, h * 0.12)))
    c.drawCentredString(x + w / 2, y + h / 2 - 2.5, "RESERVED")


def _draw_missing_merchandise_card(c: canvas.Canvas, placement: HolidayPlacement, x: float, y: float, w: float, h: float) -> None:
    c.setFillColorRGB(*_WHITE)
    c.setStrokeColorRGB(*_CARD_BORDER)
    c.setLineWidth(0.7)
    c.rect(x, y, w, h, stroke=1, fill=1)
    c.setFillColorRGB(*_CARD_META_BG)
    c.rect(x, y + h - min(18.0, h * 0.18), w, min(18.0, h * 0.18), stroke=0, fill=1)
    c.setFillColorRGB(*_TEXT_MUTED)
    c.setFont(_FONT_ITALIC, 7.0)
    c.drawCentredString(x + w / 2, y + h / 2 - 3.0, "UNRESOLVED")


def _draw_nre_merchandise_card(
    c: canvas.Canvas,
    placement: HolidayPlacement,
    x: float,
    y: float,
    w: float,
    h: float,
    image: Image.Image | None,
) -> tuple[str, str]:
    c.setFillColorRGB(*_WHITE)
    c.setStrokeColorRGB(*_CARD_BORDER)
    c.setLineWidth(0.7)
    c.rect(x, y, w, h, stroke=1, fill=1)

    pad = max(4.5, min(7.0, w * 0.045))
    metadata_h = max(14.0, min(18.0, h * 0.14))
    inner_x = x + pad
    inner_w = w - 2 * pad
    meta_y = y + h - metadata_h

    c.setFillColorRGB(*_CARD_META_BG)
    c.setStrokeColorRGB(*_CARD_META_BG)
    c.rect(x, meta_y, w, metadata_h, stroke=0, fill=1)

    price_text = _format_price_value(placement.denomination)
    cpp_text = _format_cpp_value(placement.cpp)
    price_fs = _fit_protected_single_line_font(price_text, _FONT_BODY_BOLD, inner_w * 0.58, 7.6, 4.4)
    cpp_fs = _fit_protected_single_line_font(cpp_text, _FONT_BODY_BOLD, inner_w * 0.36, 7.6, 4.4)
    baseline = meta_y + (metadata_h - max(price_fs, cpp_fs)) / 2 + 1.0
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(_FONT_BODY_BOLD, price_fs)
    c.drawString(inner_x, baseline, _truncate_text(price_text, _FONT_BODY_BOLD, price_fs, inner_w * 0.58))
    c.setFont(_FONT_BODY_BOLD, cpp_fs)
    c.drawRightString(x + w - pad, baseline, _truncate_text(cpp_text, _FONT_BODY_BOLD, cpp_fs, inner_w * 0.36))

    upc_text = f"UPC {(placement.product_upc or '').strip() or '-'}"
    item_text = f"ITEM {(placement.item_number or '').strip() or '-'}"
    upc_fs = _fit_protected_single_line_font(upc_text, _FONT_BODY_BOLD, inner_w, 5.8, 4.2)
    item_fs = _fit_protected_single_line_font(item_text, _FONT_BODY_REGULAR, inner_w, 5.5, 4.0)

    name = (placement.product_name or "").strip() or "No product name"
    name_fs = 5.4
    max_name_lines = 3
    while name_fs > 4.1:
        name_lines = _wrap_text_lines(name, _FONT_BODY_REGULAR, name_fs, inner_w)
        if len(name_lines) > max_name_lines:
            name_lines = name_lines[:max_name_lines]
            name_lines[-1] = _truncate_text(name_lines[-1], _FONT_BODY_REGULAR, name_fs, inner_w)
        text_h = (upc_fs + 1.0) + (item_fs + 1.0) + _wrapped_block_height(len(name_lines), name_fs, 0.8) + 5.0
        if text_h <= max(1.0, h - metadata_h - 24.0):
            break
        name_fs -= 0.2

    name_lines = _wrap_text_lines(name, _FONT_BODY_REGULAR, name_fs, inner_w)
    if len(name_lines) > max_name_lines:
        name_lines = name_lines[:max_name_lines]
        name_lines[-1] = _truncate_text(name_lines[-1], _FONT_BODY_REGULAR, name_fs, inner_w)

    content_top = y + h - metadata_h - pad
    c.setFillColorRGB(*_TEXT_DARK)
    c.setFont(_FONT_BODY_BOLD, upc_fs)
    c.drawString(inner_x, content_top - upc_fs, upc_text)
    current_y = content_top - upc_fs - 2.0
    c.setFont(_FONT_BODY_REGULAR, item_fs)
    c.drawString(inner_x, current_y - item_fs, item_text)
    current_y -= item_fs + 2.5
    after_name_y = _draw_wrapped_lines(c, name_lines, inner_x, current_y, _FONT_BODY_REGULAR, name_fs, 0.8)

    image_y = y + pad
    image_h = max(14.0, after_name_y - image_y - 2.5)
    _draw_image_area(c, inner_x, image_y, inner_w, image_h, image)

    image_box_rect = f"{inner_x:.2f},{image_y:.2f},{inner_w:.2f},{image_h:.2f}"
    image_draw_rect = ""
    if image is not None:
        draw_x, draw_y, draw_w, draw_h = _contain_image_rect(image.width, image.height, inner_x, image_y, inner_w, image_h)
        image_draw_rect = f"{draw_x:.2f},{draw_y:.2f},{draw_w:.2f},{draw_h:.2f}"
    return image_box_rect, image_draw_rect


def _missing_image_rows(
    placements: Iterable[HolidayPlacement],
    missing_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    product_placements = [placement for placement in placements if _slot_type(placement) == SLOT_MERCHANDISE]
    total_by_product = Counter((placement.item_number, placement.product_upc) for placement in product_placements)
    by_product: dict[tuple[str, str], dict[str, object]] = {}
    for placement in product_placements:
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
                "NRE Facings": total_by_product.get(key, 0),
                "Missing Placement Count": 0,
                "Image Status": placement.image_status,
            },
        )
        row["Missing Placement Count"] = int(row["Missing Placement Count"]) + 1
        if placement.image_status == "resolved":
            row["Image Status"] = "placeholder_bad_image"
    return sorted(by_product.values(), key=lambda row: (str(row["Product Name"]), str(row["WM Item Number"])))


def render_holiday_nre(placements: list[HolidayPlacement], configuration: str) -> HolidayQpRenderResult:
    if not placements:
        raise ValueError("No Holiday NRE placements are available to render.")

    template = _template_for_configuration(configuration)
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=(template.page_w, template.page_h))
    image_cache: dict[str, Image.Image | None] = {}
    image_slots = 0
    placeholder_slots = 0
    filler_slots = 0
    warnings: list[str] = []
    render_rows: list[dict[str, object]] = []
    missing_keys: set[tuple[str, str]] = set()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    header_layout = _NreLayout(
        page_w=template.page_w,
        page_h=template.page_h,
        margin=_BASE_MARGIN,
        header_h=_HEADER_H,
        footer_h=_FOOTER_H,
        scale=1.0,
        src_left=0.0,
        src_top=0.0,
        body_left=template.content_left,
        body_top=template.page_h - _HEADER_H - _BASE_MARGIN,
    )
    _draw_header(c, header_layout, configuration)
    _draw_footer(c, template.page_w, _BASE_MARGIN, generated_at)
    _draw_template_headings(c, template)

    for placement, slot in _template_assignments(placements, template):
        x, y, w, h = slot.x, slot.y, slot.w, slot.h
        slot_type = slot.slot_type
        content_type = "placeholder"
        image_status = placement.image_status if placement is not None else "missing"
        image_warning = ""
        image_draw_rect = ""
        image_box_rect = ""

        if slot_type == SLOT_FILLER:
            filler_slots += 1
            content_type = "filler"
            image_status = "not_applicable"
            _draw_filler_block(c, x, y, w, h)
        elif slot_type == SLOT_AMBIGUOUS:
            content_type = "reserved"
            image_status = "not_applicable"
            _draw_ambiguous_block(c, x, y, w, h)
        elif placement is None or not placement.product_name:
            placeholder_slots += 1
            if placement is not None:
                _draw_missing_merchandise_card(c, placement, x, y, w, h)
            else:
                _draw_filler_block(c, x, y, w, h, "UNRESOLVED")
        else:
            image, image_warning = _open_slot_image(placement.image_path, image_cache) if placement.image_status == "resolved" else (None, "")
            if image is not None:
                image_slots += 1
                content_type = "image"
            else:
                placeholder_slots += 1
                if image_warning:
                    warnings.append(image_warning)
                    missing_keys.add((placement.item_number, placement.product_upc))
                    image_status = "placeholder_bad_image"
            image_box_rect, image_draw_rect = _draw_nre_merchandise_card(c, placement, x, y, w, h, image)

        render_rows.append(
            {
                "Configuration": configuration,
                "Row": "",
                "Column": "",
                "Slot Type": slot_type,
                "Group": slot.group_name,
                "Group Index": slot.group_index + 1,
                "Reference Text": placement.reference_text if placement is not None else "",
                "Product Name": placement.product_name if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "UPC": placement.product_upc if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "ITEM": placement.item_number if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "Price": _format_price_value(placement.denomination) if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "CPP": _format_cpp_value(placement.cpp) if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "Metadata Fields": "Price; CPP; UPC; ITEM; Product Name" if slot_type == SLOT_MERCHANDISE else "",
                "Content Type": content_type,
                "Image Status": image_status,
                "Image Path": placement.image_path if placement is not None and slot_type == SLOT_MERCHANDISE else "",
                "Card Rect": f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}",
                "Image Box Rect": image_box_rect,
                "Image Draw Rect": image_draw_rect,
                "Warning": image_warning,
            }
        )

    c.showPage()
    c.save()
    pdf_bytes = pdf_buffer.getvalue()

    return HolidayQpRenderResult(
        pdf_bytes=pdf_bytes,
        preview_png_bytes=_render_preview_from_pdf(pdf_bytes),
        panels_rendered=1,
        slots_rendered=len(render_rows),
        image_slots=image_slots,
        placeholder_slots=placeholder_slots,
        skipped_slots=filler_slots,
        missing_image_rows=_missing_image_rows(placements, missing_keys),
        render_rows=render_rows,
        warnings=warnings,
    )
