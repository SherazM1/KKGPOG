from __future__ import annotations

import io
from statistics import median

from PIL import Image, ImageDraw, ImageOps

from app.holiday_planograms.geometry import _cluster_numbers
from app.holiday_planograms.models import GeometryResult, LayoutPanel, LayoutSlot, ReferenceImage


Box = tuple[int, int, int, int]
SLOT_MERCHANDISE = "merchandise"
SLOT_FILLER = "filler"
SLOT_AMBIGUOUS = "ambiguous"


def _iou(left: Box, right: Box) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _filled_green_slot_boxes(image: Image.Image) -> list[Box]:
    pix = image.load()
    width, height = image.size
    seen: set[tuple[int, int]] = set()
    boxes: list[Box] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            red, green, blue = pix[x, y]
            if not (green > 145 and red < 150 and blue < 160 and green > red + 25):
                continue
            queue = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            for px, py in queue:
                xs.append(px)
                ys.append(py)
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in seen:
                        continue
                    red, green, blue = pix[nx, ny]
                    if green > 145 and red < 150 and blue < 160 and green > red + 25:
                        seen.add((nx, ny))
                        queue.append((nx, ny))
            if len(xs) <= 500:
                continue
            box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            if 20 <= box_w <= 80 and 35 <= box_h <= 100:
                boxes.append(box)
    return boxes


def _bordered_slot_boxes(image: Image.Image, filled_boxes: list[Box]) -> list[Box]:
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda pixel: 255 if pixel < 170 else 0)
    width, height = mask.size
    pix = mask.load()

    vertical_runs: list[tuple[int, int, int]] = []
    for x in range(width):
        y = 0
        while y < height:
            if pix[x, y] == 0:
                y += 1
                continue
            y0 = y
            while y < height and pix[x, y] != 0:
                y += 1
            run_h = y - y0
            if 35 <= run_h <= 95:
                vertical_runs.append((x, y0, y))

    def horizontal_count(y: int, x_start: int, x_end: int) -> int:
        return sum(1 for x in range(x_start, x_end + 1) if pix[x, y] != 0)

    candidates: list[Box] = []
    for index, (x0, y0, y1) in enumerate(vertical_runs):
        for x1, y2, y3 in vertical_runs[index + 1 :]:
            box_w = x1 - x0
            if not (24 <= box_w <= 62):
                continue
            if abs(y0 - y2) > 2 or abs(y1 - y3) > 2:
                continue
            top = horizontal_count(y0, x0, x1)
            bottom = horizontal_count(y1 - 1, x0, x1)
            if top >= box_w * 0.82 and bottom >= box_w * 0.82:
                box = (x0, y0, x1 + 1, y1)
                if not any(_iou(box, filled) > 0.15 for filled in filled_boxes):
                    candidates.append(box)

    kept: list[Box] = []
    ranked = sorted(candidates, key=lambda box: (-((box[2] - box[0]) * (box[3] - box[1])), box[1], box[0]))
    for box in ranked:
        if any(_iou(box, existing) > 0.35 for existing in kept):
            continue
        kept.append(box)
    return kept


def _has_relaxed_border_evidence(image: Image.Image, box: Box) -> bool:
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda pixel: 255 if pixel < 185 else 0)
    pix = mask.load()
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return False
    def best_horizontal(target_y: int) -> float:
        candidates = []
        for y in range(max(0, target_y - 2), min(image.height, target_y + 3)):
            candidates.append(sum(1 for x in range(x0, x1) if pix[x, y] != 0) / max(1, x1 - x0))
        return max(candidates, default=0.0)

    def best_vertical(target_x: int) -> float:
        candidates = []
        for x in range(max(0, target_x - 2), min(image.width, target_x + 3)):
            candidates.append(sum(1 for y in range(y0, y1) if pix[x, y] != 0) / max(1, y1 - y0))
        return max(candidates, default=0.0)

    top = best_horizontal(y0)
    bottom = best_horizontal(y1 - 1)
    left = best_vertical(x0)
    right = best_vertical(x1 - 1)
    return top >= 0.70 and bottom >= 0.70 and left >= 0.70 and right >= 0.70


def _group_boxes_by_panel(boxes: list[Box], widths: list[int]) -> list[list[Box]]:
    centers_x = [(box[0] + box[2]) / 2.0 for box in boxes]
    panel_centers = _cluster_numbers(centers_x, tolerance=max(72.0, median(widths) * 2.0))
    grouped_by_center: dict[float, list[Box]] = {panel_center: [] for panel_center in panel_centers}
    for box in boxes:
        center_x = (box[0] + box[2]) / 2.0
        nearest = min(panel_centers, key=lambda panel_center: abs(panel_center - center_x))
        grouped_by_center[nearest].append(box)
    panel_groups = [
        sorted(group, key=lambda box: (box[1], box[0]))
        for _panel_center, group in sorted(grouped_by_center.items(), key=lambda item: min(box[0] for box in item[1]) if item[1] else 0)
        if group
    ]
    panel_groups.sort(key=lambda group: min(box[0] for box in group))
    merged: list[list[Box]] = []
    for group in panel_groups:
        if merged and len(group) <= 6:
            merged[-1] = sorted(merged[-1] + group, key=lambda box: (box[1], box[0]))
        else:
            merged.append(group)
    panel_groups = merged
    return panel_groups


def _recover_missing_merchandise_cells(image: Image.Image, boxes: list[Box]) -> list[Box]:
    if not boxes:
        return boxes
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    recovered = list(boxes)
    for group in _group_boxes_by_panel(boxes, widths):
        row_centers = _cluster_numbers([(box[1] + box[3]) / 2.0 for box in group], tolerance=median(heights) * 0.75)
        col_centers = _cluster_numbers([(box[0] + box[2]) / 2.0 for box in group], tolerance=median(widths) * 0.75)
        if len(row_centers) * len(col_centers) <= len(group):
            continue
        occupied = {
            (
                min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - ((box[1] + box[3]) / 2.0))),
                min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - ((box[0] + box[2]) / 2.0))),
            )
            for box in group
        }
        group_w = int(round(median([box[2] - box[0] for box in group])))
        group_h = int(round(median([box[3] - box[1] for box in group])))
        for row_index, center_y in enumerate(row_centers):
            for col_index, center_x in enumerate(col_centers):
                if (row_index, col_index) in occupied:
                    continue
                candidate = (
                    int(round(center_x - group_w / 2)),
                    int(round(center_y - group_h / 2)),
                    int(round(center_x + group_w / 2)),
                    int(round(center_y + group_h / 2)),
                )
                candidate = (
                    max(0, candidate[0]),
                    max(0, candidate[1]),
                    min(image.width, candidate[2]),
                    min(image.height, candidate[3]),
                )
                slot_type, _confidence, _diagnostics = classify_nre_slot(image.crop(candidate))
                if slot_type == SLOT_MERCHANDISE and _has_relaxed_border_evidence(image, candidate):
                    recovered.append(candidate)
    return sorted(recovered, key=lambda box: (box[1], box[0]))


def classify_nre_slot(image_crop: Image.Image) -> tuple[str, float, dict[str, float]]:
    crop = image_crop.convert("RGB")
    width, height = crop.size
    if width < 10 or height < 10:
        return SLOT_AMBIGUOUS, 0.0, {"green_ratio": 0.0, "sampled_pixels": 0.0}

    left = max(0, int(width * 0.12))
    right = min(width, int(width * 0.88))
    top = max(0, int(height * 0.12))
    bottom = min(height, int(height * 0.88))
    if right <= left or bottom <= top:
        return SLOT_AMBIGUOUS, 0.0, {"green_ratio": 0.0, "sampled_pixels": 0.0}

    pix = crop.load()
    sampled = 0
    green_like = 0
    dark_or_red_like = 0
    for y in range(top, bottom):
        for x in range(left, right):
            red, green, blue = pix[x, y]
            sampled += 1
            if green > 145 and red < 160 and blue < 170 and green > red + 25:
                green_like += 1
            if red > 150 and green < 100 and blue < 100:
                dark_or_red_like += 1
            elif red < 90 and green < 90 and blue < 90:
                dark_or_red_like += 1

    if sampled == 0:
        return SLOT_AMBIGUOUS, 0.0, {"green_ratio": 0.0, "sampled_pixels": 0.0}

    green_ratio = green_like / sampled
    mark_ratio = dark_or_red_like / sampled
    diagnostics = {"green_ratio": green_ratio, "mark_ratio": mark_ratio, "sampled_pixels": float(sampled)}
    if green_ratio >= 0.30:
        return SLOT_FILLER, min(1.0, green_ratio / 0.60), diagnostics
    if 0.16 <= green_ratio < 0.30:
        return SLOT_AMBIGUOUS, 1.0 - abs(green_ratio - 0.23) / 0.07, diagnostics
    return SLOT_MERCHANDISE, min(1.0, max(0.45, (0.16 - green_ratio) / 0.16 + mark_ratio)), diagnostics


def detect_nre_geometry(reference_image: ReferenceImage) -> GeometryResult:
    warnings: list[str] = []
    image = Image.open(io.BytesIO(reference_image.bytes_data)).convert("RGB")
    filled_boxes = _filled_green_slot_boxes(image)
    bordered_boxes = _bordered_slot_boxes(image, filled_boxes)
    boxes = _recover_missing_merchandise_cells(image, sorted(filled_boxes + bordered_boxes, key=lambda box: (box[1], box[0])))
    if not boxes:
        return GeometryResult(panels=[], slots=[], warnings=["No NRE card slot rectangles were detected."])

    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    panel_groups = _group_boxes_by_panel(boxes, widths)

    slots: list[LayoutSlot] = []
    panels: list[LayoutPanel] = []
    for panel_index, group in enumerate(panel_groups, start=1):
        row_centers = _cluster_numbers([(box[1] + box[3]) / 2.0 for box in group], tolerance=median(heights) * 0.75)
        col_centers = _cluster_numbers([(box[0] + box[2]) / 2.0 for box in group], tolerance=median(widths) * 0.75)
        group_slots: list[LayoutSlot] = []
        for box in group:
            crop = image.crop(box)
            slot_type, confidence, _diagnostics = classify_nre_slot(crop)
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            row = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - center_y)) + 1
            column = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - center_x)) + 1
            slot = LayoutSlot(
                panel_index=panel_index,
                panel_name=f"panel_{panel_index}",
                row=row,
                column=column,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                center_x=center_x,
                center_y=center_y,
                width=float(box[2] - box[0]),
                height=float(box[3] - box[1]),
                slot_type=slot_type,
                classification_confidence=confidence,
            )
            group_slots.append(slot)
            slots.append(slot)
        x0 = min(box[0] for box in group)
        y0 = min(box[1] for box in group)
        x1 = max(box[2] for box in group)
        y1 = max(box[3] for box in group)
        panels.append(
            LayoutPanel(
                panel_index=panel_index,
                panel_name=f"panel_{panel_index}",
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                rows=len(row_centers),
                columns=len(col_centers),
                active_slots=len(group_slots),
                slots=sorted(group_slots, key=lambda slot: (slot.row, slot.column, slot.center_x)),
            )
        )
    slots.sort(key=lambda slot: (slot.panel_index, slot.row, slot.column, slot.center_x))

    expected_grid_slots = sum(panel.rows * panel.columns for panel in panels)
    if expected_grid_slots != len(slots):
        warnings.append(
            f"NRE geometry contains missing/irregular cells: detected {len(slots)} active slots across {expected_grid_slots} grid positions."
        )

    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    for panel in panels:
        draw.rectangle(panel.bbox, outline=(0, 96, 255), width=3)
        draw.text((panel.bbox[0], max(0, panel.bbox[1] - 16)), panel.panel_name, fill=(0, 96, 255))
    for index, slot in enumerate(slots, start=1):
        if slot.slot_type == SLOT_FILLER:
            color = (0, 150, 0)
        elif slot.slot_type == SLOT_AMBIGUOUS:
            color = (220, 140, 0)
        else:
            color = (255, 0, 0)
        draw.rectangle(slot.bbox, outline=color, width=2)
        label = f"{index} r{slot.row}c{slot.column} {slot.slot_type[:1].upper()}"
        draw.text((slot.bbox[0] + 2, slot.bbox[1] + 2), label, fill=color)
    out = io.BytesIO()
    debug.save(out, format="PNG")
    return GeometryResult(panels=panels, slots=slots, warnings=warnings, debug_image_bytes=out.getvalue())
