from __future__ import annotations

from dataclasses import dataclass
import io
from typing import List, Optional, Sequence, Tuple

import fitz
from PIL import Image


BBox = Tuple[float, float, float, float]


@dataclass
class ImageTransferResult:
    pdf_bytes: bytes
    audit_rows: List[dict]


def add_images_to_existing_label_pog(
    label_pdf_bytes: bytes,
    image_pdf_bytes: bytes,
    include_middle: bool = True,
    include_bonus: bool = True,
) -> ImageTransferResult:
    """Overlay middle/bonus artwork from an official image PDF onto an existing label POG.

    The generated label POG is the target of record.  Images are transferred by
    side/section/row/column slot, not by a compacted list of detected image
    objects, so blank target slots do not shift later cards.
    """
    target_doc = fitz.open(stream=label_pdf_bytes, filetype="pdf")
    source_doc = fitz.open(stream=image_pdf_bytes, filetype="pdf")
    audit_rows: List[dict] = []

    page_count = min(target_doc.page_count, source_doc.page_count)
    for page_index in range(page_count):
        target_page = target_doc.load_page(page_index)
        source_page = source_doc.load_page(page_index)
        side = "ABCD"[page_index] if page_index < 4 else str(page_index + 1)

        target_sections = _target_sections(target_page)
        source_profile = _source_profile(source_page, page_index)

        if include_middle:
            _transfer_section(
                target_page,
                source_page,
                side,
                "middle",
                target_sections.middle_rows,
                source_profile,
                audit_rows,
            )

        if include_bonus:
            _transfer_section(
                target_page,
                source_page,
                side,
                "bonus",
                target_sections.bonus_rows,
                source_profile,
                audit_rows,
            )

    out = target_doc.tobytes(garbage=4, deflate=True)
    target_doc.close()
    source_doc.close()
    return ImageTransferResult(pdf_bytes=out, audit_rows=audit_rows)


@dataclass
class _TargetSections:
    middle_rows: List[List[fitz.Rect]]
    bonus_rows: List[List[fitz.Rect]]


@dataclass
class _SourceProfile:
    kind: str
    middle_rows: List[List[fitz.Rect]]
    bonus_rows: List[List[fitz.Rect]]
    strip_union: Optional[fitz.Rect]
    page_index: int


def _transfer_section(
    target_page: fitz.Page,
    source_page: fitz.Page,
    side: str,
    section: str,
    target_rows: List[List[fitz.Rect]],
    source_profile: _SourceProfile,
    audit_rows: List[dict],
) -> None:
    if not target_rows:
        audit_rows.append(_audit(side, section, None, None, "SKIPPED", "no_target_slots"))
        return

    row_shapes = [len(row) for row in target_rows]
    source_rows = _source_rows_for_shape(source_profile, section, row_shapes)
    if not source_rows:
        for row_index, row in enumerate(target_rows):
            for col_index, _target_rect in enumerate(row):
                audit_rows.append(_audit(side, section, row_index, col_index, "SKIPPED", "no_source_grid"))
        return

    for row_index, target_row in enumerate(target_rows):
        source_row = source_rows[row_index] if row_index < len(source_rows) else []
        for col_index, target_rect in enumerate(target_row):
            has_label = _rect_has_text(target_page, target_rect)
            if not has_label:
                audit_rows.append(_audit(side, section, row_index, col_index, "SKIPPED", "blank_target_slot"))
                continue
            if col_index >= len(source_row):
                audit_rows.append(_audit(side, section, row_index, col_index, "SKIPPED", "missing_source_slot"))
                continue
            source_rect = source_row[col_index]
            try:
                pix = source_page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0), clip=source_rect, alpha=False)
                if pix.width < 8 or pix.height < 8:
                    audit_rows.append(_audit(side, section, row_index, col_index, "SKIPPED", "tiny_source_crop"))
                    continue
                image_stream = _trim_raster_pixmap(pix) if source_profile.kind == "raster_strip" else None
                if image_stream:
                    target_page.insert_image(_target_image_area(target_rect), stream=image_stream, keep_proportion=True)
                else:
                    target_page.insert_image(_target_image_area(target_rect), pixmap=pix, keep_proportion=True)
                audit_rows.append(
                    _audit(
                        side,
                        section,
                        row_index,
                        col_index,
                        "INSERTED",
                        f"{source_profile.kind}_slot_grid",
                        target_rect,
                        source_rect,
                    )
                )
            except Exception as exc:
                audit_rows.append(_audit(side, section, row_index, col_index, "SKIPPED", str(exc), target_rect, source_rect))


def _trim_raster_pixmap(pix: fitz.Pixmap) -> Optional[bytes]:
    try:
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None
    if image.width < 12 or image.height < 12:
        return None

    rgb = image.convert("RGB")
    px = rgb.load()
    xs: List[int] = []
    ys: List[int] = []
    central_xs: List[int] = []
    central_ys: List[int] = []
    central_left = int(rgb.width * 0.18)
    central_right = int(rgb.width * 0.82)
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = px[x, y]
            maxc = max(r, g, b)
            minc = min(r, g, b)
            saturation = maxc - minc
            if maxc > 248 and minc > 240:
                continue
            if saturation < 8 and 170 <= minc <= 238:
                continue
            if saturation > 18 or (maxc < 180 and saturation > 8):
                xs.append(x)
                ys.append(y)
                if central_left <= x <= central_right:
                    central_xs.append(x)
                    central_ys.append(y)
    if not xs or not ys:
        return None

    if len(central_xs) >= max(80, int(len(xs) * 0.35)):
        xs = central_xs
        ys = central_ys

    left = max(0, min(xs) - 4)
    top = max(0, min(ys) - 4)
    right = min(rgb.width, max(xs) + 5)
    bottom = min(rgb.height, max(ys) + 5)
    if right - left < 8 or bottom - top < 8:
        return None

    cropped = rgb.crop((left, top, right, bottom))
    if cropped.width / max(1, cropped.height) < 0.28:
        return None
    out = io.BytesIO()
    cropped.save(out, format="PNG")
    return out.getvalue()


def _audit(
    side: str,
    section: str,
    row: Optional[int],
    col: Optional[int],
    status: str,
    reason: str,
    target_rect: Optional[fitz.Rect] = None,
    source_rect: Optional[fitz.Rect] = None,
) -> dict:
    return {
        "Side": side,
        "Section": section,
        "Row": "" if row is None else row,
        "Col": "" if col is None else col,
        "Status": status,
        "Reason": reason,
        "Target BBox": _bbox_text(target_rect),
        "Source BBox": _bbox_text(source_rect),
    }


def _bbox_text(rect: Optional[fitz.Rect]) -> str:
    if rect is None:
        return ""
    return ",".join(f"{v:.2f}" for v in (rect.x0, rect.y0, rect.x1, rect.y1))


def _target_sections(page: fitz.Page) -> _TargetSections:
    boxes = _target_card_boxes(page)
    rows = _group_rows(boxes, tolerance=24.0)
    bonus_y = _find_marker_y(page, "BONUS")
    if bonus_y is None:
        return _TargetSections(middle_rows=[], bonus_rows=[])

    before_bonus = [row for row in rows if row and row[0].y0 < bonus_y]
    middle_rows = before_bonus[-3:] if len(before_bonus) >= 3 else []
    bonus_rows = [row for row in rows if row and row[0].y0 > bonus_y]
    return _TargetSections(middle_rows=middle_rows, bonus_rows=bonus_rows)


def _target_card_boxes(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if not rect:
            continue
        r = fitz.Rect(rect)
        if 45.0 <= r.width <= 130.0 and 45.0 <= r.height <= 135.0:
            rects.append(r)
    return _unique_rects(rects)


def _find_marker_y(page: fitz.Page, token: str) -> Optional[float]:
    token = token.upper()
    ys: List[float] = []
    for word in page.get_text("words"):
        text = str(word[4] or "").upper()
        if text == token:
            ys.append((float(word[1]) + float(word[3])) / 2.0)
    if not ys:
        return None
    if token == "BONUS":
        section_header_ys = [y for y in ys if y > 850.0]
        return min(section_header_ys) if section_header_ys else min(ys)
    return max(ys)


def _rect_has_text(page: fitz.Page, rect: fitz.Rect) -> bool:
    words = page.get_text("words", clip=rect)
    meaningful = [
        str(word[4]).strip()
        for word in words
        if str(word[4]).strip() and str(word[4]).strip() not in {"|", "_"}
    ]
    return bool(meaningful)


def _target_image_area(rect: fitz.Rect) -> fitz.Rect:
    mx = max(3.0, rect.width * 0.08)
    top = rect.y0 + max(4.0, rect.height * 0.05)
    bottom = rect.y0 + rect.height * 0.56
    return fitz.Rect(rect.x0 + mx, top, rect.x1 - mx, bottom)


def _source_profile(page: fitz.Page, page_index: int) -> _SourceProfile:
    image_blocks = _image_blocks(page)
    small = [r for r in image_blocks if 14.0 <= r.width <= 48.0 and 20.0 <= r.height <= 52.0]
    large = [r for r in image_blocks if r.width > 220.0 and r.height > 70.0]

    if len(small) >= 40:
        middle = _group_rows([r for r in small if 235.0 <= r.y0 <= 365.0], tolerance=12.0)
        bonus = _group_rows([r for r in small if 365.0 <= r.y0 <= 615.0], tolerance=12.0)
        return _SourceProfile(kind="individual_art", middle_rows=middle, bonus_rows=bonus, strip_union=None, page_index=page_index)

    union = _union_rect(large)
    return _SourceProfile(kind="raster_strip", middle_rows=[], bonus_rows=[], strip_union=union, page_index=page_index)


def _source_rows_for_shape(profile: _SourceProfile, section: str, row_shapes: Sequence[int]) -> List[List[fitz.Rect]]:
    if profile.kind == "individual_art":
        rows = profile.middle_rows if section == "middle" else profile.bonus_rows
        if section == "middle":
            return _fit_rows_to_shape(rows, row_shapes, centered=False)
        return _fit_rows_to_shape(rows, row_shapes, centered=True)

    if profile.strip_union is None:
        return []
    return _strip_rows_for_shape(profile.strip_union, section, row_shapes, profile.page_index)


def _fit_rows_to_shape(rows: List[List[fitz.Rect]], row_shapes: Sequence[int], centered: bool) -> List[List[fitz.Rect]]:
    if not row_shapes:
        return []
    fitted: List[List[fitz.Rect]] = []
    rows = sorted(rows, key=lambda row: _row_center(row)[1])

    full_x0, full_x1 = _dominant_full_width(rows)
    for row_index, target_count in enumerate(row_shapes):
        source_row = rows[row_index] if row_index < len(rows) else []
        if len(source_row) == target_count:
            fitted.append(source_row)
            continue

        row_y0, row_y1 = _row_band(source_row, rows, row_index)
        if source_row and len(source_row) > target_count:
            fitted.append(source_row[:target_count])
            continue

        if target_count <= 0:
            fitted.append([])
            continue
        if centered and full_x1 > full_x0 and target_count < 8:
            max_cols = 10
            pitch = (full_x1 - full_x0) / max_cols
            x0 = full_x0 + (max_cols - target_count) * pitch / 2.0
            x1 = x0 + target_count * pitch
        elif source_row:
            x0 = min(r.x0 for r in source_row)
            x1 = max(r.x1 for r in source_row)
            if len(source_row) < target_count and full_x1 > full_x0:
                x0, x1 = full_x0, full_x1
        else:
            x0, x1 = full_x0, full_x1
        fitted.append(_split_rect_grid(fitz.Rect(x0, row_y0, x1, row_y1), 1, target_count)[0])
    return fitted


def _dominant_full_width(rows: List[List[fitz.Rect]]) -> Tuple[float, float]:
    wide_rows = [row for row in rows if len(row) >= 8]
    selected = wide_rows or rows
    if not selected:
        return 0.0, 0.0
    return min(r.x0 for row in selected for r in row), max(r.x1 for row in selected for r in row)


def _row_band(source_row: List[fitz.Rect], rows: List[List[fitz.Rect]], row_index: int) -> Tuple[float, float]:
    if source_row:
        return min(r.y0 for r in source_row), max(r.y1 for r in source_row)
    heights = [r.height for row in rows for r in row]
    height = sum(heights) / len(heights) if heights else 34.0
    centers = [_row_center(row)[1] for row in rows if row]
    if centers:
        if row_index < len(centers):
            cy = centers[row_index]
        else:
            step = (centers[-1] - centers[0]) / max(1, len(centers) - 1) if len(centers) > 1 else height * 1.1
            cy = centers[-1] + step * (row_index - len(centers) + 1)
    else:
        cy = height * (row_index + 0.5)
    return cy - height / 2.0, cy + height / 2.0


def _strip_rows_for_shape(union: fitz.Rect, section: str, row_shapes: Sequence[int], page_index: int) -> List[List[fitz.Rect]]:
    template_rows = _bd_raster_template_rows(section, row_shapes, page_index)
    if template_rows:
        return template_rows

    w = union.width
    h = union.height
    if section == "middle":
        region = fitz.Rect(
            union.x0 + 0.18 * w,
            union.y0 + 0.22 * h,
            union.x1 - 0.02 * w,
            union.y0 + 0.52 * h,
        )
    else:
        region = fitz.Rect(
            union.x0 + 0.05 * w,
            union.y0 + 0.55 * h,
            union.x1 - 0.03 * w,
            union.y0 + 0.96 * h,
        )

    rows: List[List[fitz.Rect]] = []
    row_count = len(row_shapes)
    if row_count <= 0:
        return rows
    for row_index, col_count in enumerate(row_shapes):
        y0 = region.y0 + region.height * row_index / row_count
        y1 = region.y0 + region.height * (row_index + 1) / row_count
        row_region = fitz.Rect(region.x0, y0, region.x1, y1)
        rows.append(_split_rect_grid(row_region, 1, max(1, col_count))[0])
    return rows


def _bd_raster_template_rows(section: str, row_shapes: Sequence[int], page_index: Optional[int] = None) -> List[List[fitz.Rect]]:
    """Known B/D raster layout tracks.

    The official image PDFs expose B/D as large horizontal raster strips, but
    the artwork inside those strips still lands on the same page-coordinate
    tracks as the individually exposed A/C artwork.  These coordinates crop the
    visible card art slots directly instead of slicing signage/rulers/neighbors.
    """
    if not row_shapes:
        return []

    if section == "middle":
        x_ranges = [
            (158.6, 181.2),
            (185.2, 207.7),
            (254.2, 278.3),
            (280.7, 304.9),
            (307.3, 331.4),
            (333.8, 358.1),
            (404.2, 426.7),
            (430.7, 453.2),
        ]
        y_ranges = [
            (249.5, 283.1),
            (286.1, 319.6),
            (322.6, 356.0),
        ]
        rows: List[List[fitz.Rect]] = []
        for row_index, col_count in enumerate(row_shapes):
            if row_index >= len(y_ranges):
                break
            y0, y1 = y_ranges[row_index]
            rows.append([fitz.Rect(x0, y0, x1, y1) for x0, x1 in x_ranges[:col_count]])
        return rows

    if section != "bonus":
        return []

    is_side_d = page_index == 3
    main_x_ranges = [
        (178.2, 202.8),
        (204.0, 228.5),
        (231.2, 255.8),
        (258.0, 281.0),
        (284.0, 308.8),
        (311.0, 335.2),
        (337.8, 361.0),
        (364.2, 387.8),
        (390.0, 414.8),
        (417.2, 440.8),
    ]
    top_x_ranges = [
        (231.2, 255.8),
        (258.0, 281.8),
        (284.0, 308.8),
        (311.0, 335.2),
        (337.8, 361.5),
        (364.2, 387.8),
    ]
    if is_side_d:
        main_x_shift = -10.4
        top_x_shift = -10.4
        main_x_ranges = [(x0 + main_x_shift, x1 + main_x_shift) for x0, x1 in main_x_ranges]
        top_x_ranges = [(x0 + top_x_shift, x1 + top_x_shift) for x0, x1 in top_x_ranges]
    top_y = (381.0, 416.4)
    main_y_ranges = [
        (418.0, 452.0),
        (454.5, 488.8),
        (491.0, 524.8),
        (527.2, 561.0),
        (564.0, 598.0),
    ]

    rows = []
    main_row_index = 0
    for row_index, col_count in enumerate(row_shapes):
        if row_index == 0 and col_count <= len(top_x_ranges):
            y0, y1 = top_y
            rows.append([fitz.Rect(x0, y0, x1, y1) for x0, x1 in top_x_ranges[:col_count]])
            continue
        if main_row_index >= len(main_y_ranges):
            break
        y0, y1 = main_y_ranges[main_row_index]
        main_row_index += 1
        rows.append([fitz.Rect(x0, y0, x1, y1) for x0, x1 in main_x_ranges[:col_count]])
    return rows


def _image_blocks(page: fitz.Page) -> List[fitz.Rect]:
    blocks: List[fitz.Rect] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        blocks.append(fitz.Rect(*bbox))
    return blocks


def _group_rows(rects: Sequence[fitz.Rect], tolerance: float = 12.0) -> List[List[fitz.Rect]]:
    rows: List[Tuple[float, List[fitz.Rect]]] = []
    for rect in sorted(rects, key=lambda r: (r.y0, r.x0)):
        cy = (rect.y0 + rect.y1) / 2.0
        for idx, (row_cy, row) in enumerate(rows):
            if abs(row_cy - cy) <= tolerance:
                row.append(rect)
                rows[idx] = ((row_cy * (len(row) - 1) + cy) / len(row), row)
                break
        else:
            rows.append((cy, [rect]))
    return [sorted(row, key=lambda r: r.x0) for _cy, row in sorted(rows, key=lambda item: item[0])]


def _row_center(row: Sequence[fitz.Rect]) -> Tuple[float, float]:
    if not row:
        return 0.0, 0.0
    return (
        sum((r.x0 + r.x1) / 2.0 for r in row) / len(row),
        sum((r.y0 + r.y1) / 2.0 for r in row) / len(row),
    )


def _split_rect_grid(rect: fitz.Rect, rows: int, cols: int) -> List[List[fitz.Rect]]:
    grid: List[List[fitz.Rect]] = []
    if rows <= 0 or cols <= 0:
        return grid
    for row_index in range(rows):
        row: List[fitz.Rect] = []
        y0 = rect.y0 + rect.height * row_index / rows
        y1 = rect.y0 + rect.height * (row_index + 1) / rows
        for col_index in range(cols):
            x0 = rect.x0 + rect.width * col_index / cols
            x1 = rect.x0 + rect.width * (col_index + 1) / cols
            row.append(fitz.Rect(x0, y0, x1, y1))
        grid.append(row)
    return grid


def _union_rect(rects: Sequence[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _unique_rects(rects: Sequence[fitz.Rect], tolerance: float = 1.0) -> List[fitz.Rect]:
    unique: List[fitz.Rect] = []
    for rect in sorted(rects, key=lambda r: (round(r.y0, 2), round(r.x0, 2), r.width * r.height)):
        if any(
            abs(rect.x0 - other.x0) <= tolerance
            and abs(rect.y0 - other.y0) <= tolerance
            and abs(rect.x1 - other.x1) <= tolerance
            and abs(rect.y1 - other.y1) <= tolerance
            for other in unique
        ):
            continue
        unique.append(rect)
    return unique
