from __future__ import annotations

import io
from statistics import median

from PIL import Image, ImageDraw, ImageOps

from app.holiday_planograms.models import GeometryResult, LayoutPanel, LayoutSlot, ReferenceImage


def _cluster_numbers(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - median(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _detect_rectangles(image: Image.Image) -> list[tuple[int, int, int, int]]:
    gray = ImageOps.grayscale(image)
    threshold = 190
    mask = gray.point(lambda pixel: 255 if pixel < threshold else 0)
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
            if 35 <= run_h <= 120:
                vertical_runs.append((x, y0, y))

    boxes: list[tuple[int, int, int, int]] = []

    def max_horizontal_run(y: int, x_start: int, x_end: int) -> int:
        best = 0
        current = 0
        for x in range(x_start, x_end + 1):
            if pix[x, y] != 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    for x0, y0, y1 in vertical_runs:
        for x1, y2, y3 in vertical_runs:
            box_w = x1 - x0
            if box_w < 25 or box_w > 90:
                continue
            if abs(y0 - y2) > 4 or abs(y1 - y3) > 4:
                continue
            bottom_y = max(y0, y1 - 1)
            top = max_horizontal_run(y0, x0, x1)
            bottom = max_horizontal_run(bottom_y, x0, x1)
            if top >= box_w * 0.75 and bottom >= box_w * 0.75:
                boxes.append((x0, y0, x1 + 1, y1))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0], item[2], item[3])):
        if any(
            abs(box[0] - existing[0]) < 3
            and abs(box[1] - existing[1]) < 3
            and abs(box[2] - existing[2]) < 3
            and abs(box[3] - existing[3]) < 3
            for existing in deduped
        ):
            continue
        deduped.append(box)
    return deduped


def _recover_single_grid_holes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    centers = [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in boxes]
    widths = [x1 - x0 for x0, _y0, x1, _y1 in boxes]
    heights = [y1 - y0 for _x0, y0, _x1, y1 in boxes]
    panel_centers = _cluster_numbers([center[0] for center in centers], tolerance=max(100.0, median(widths) * 3.5))
    recovered = list(boxes)
    for panel_center in panel_centers:
        panel_boxes = [
            box
            for box, center in zip(boxes, centers)
            if abs(center[0] - panel_center) <= max(160.0, median(widths) * 4.5)
        ]
        if len(panel_boxes) < 4:
            continue
        row_centers = _cluster_numbers([(box[1] + box[3]) / 2.0 for box in panel_boxes], tolerance=median(heights) * 0.75)
        col_centers = _cluster_numbers([(box[0] + box[2]) / 2.0 for box in panel_boxes], tolerance=median(widths) * 0.75)
        if len(row_centers) * len(col_centers) - len(panel_boxes) != 1:
            continue
        occupied = {
            (
                min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - ((box[1] + box[3]) / 2.0))),
                min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - ((box[0] + box[2]) / 2.0))),
            )
            for box in panel_boxes
        }
        for row_index, cy in enumerate(row_centers):
            for col_index, cx in enumerate(col_centers):
                if (row_index, col_index) in occupied:
                    continue
                w = median(widths)
                h = median(heights)
                recovered.append((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))
    return recovered


def detect_geometry(reference_image: ReferenceImage) -> GeometryResult:
    warnings: list[str] = []
    image = Image.open(io.BytesIO(reference_image.bytes_data)).convert("RGB")
    boxes = _detect_rectangles(image)
    if not boxes:
        return GeometryResult(panels=[], slots=[], warnings=["No card slot rectangles were detected."])

    centers_x = [(box[0] + box[2]) / 2.0 for box in boxes]
    widths = [box[2] - box[0] for box in boxes]
    panel_center_values = _cluster_numbers(centers_x, tolerance=max(100.0, median(widths) * 3.5))
    panel_groups: list[list[tuple[int, int, int, int]]] = []
    for panel_center in panel_center_values:
        group = [box for box in boxes if abs(((box[0] + box[2]) / 2.0) - panel_center) <= max(160.0, median(widths) * 4.5)]
        if group:
            panel_groups.append(sorted(group, key=lambda box: (box[1], box[0])))

    panel_groups.sort(key=lambda group: min(box[0] for box in group))
    slots: list[LayoutSlot] = []
    panels: list[LayoutPanel] = []
    known_names = ["LEFT SIDE", "FRONT", "RIGHT SIDE", "BACK"]
    for panel_index, group in enumerate(panel_groups, start=1):
        row_centers = _cluster_numbers([(box[1] + box[3]) / 2.0 for box in group], tolerance=median([b[3] - b[1] for b in group]) * 0.75)
        col_centers = _cluster_numbers([(box[0] + box[2]) / 2.0 for box in group], tolerance=median([b[2] - b[0] for b in group]) * 0.75)
        panel_name = known_names[panel_index - 1] if panel_index <= len(known_names) else f"panel_{panel_index}"
        group_slots: list[LayoutSlot] = []
        for box in group:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            row = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - cy)) + 1
            col = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - cx)) + 1
            slot = LayoutSlot(
                panel_index=panel_index,
                panel_name=panel_name,
                row=row,
                column=col,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                center_x=cx,
                center_y=cy,
                width=float(box[2] - box[0]),
                height=float(box[3] - box[1]),
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
                panel_name=panel_name,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                rows=len(row_centers),
                columns=len(col_centers),
                active_slots=len(group_slots),
                slots=sorted(group_slots, key=lambda slot: (slot.row, slot.column)),
            )
        )
    slots.sort(key=lambda slot: (slot.panel_index, slot.row, slot.column, slot.center_x))
    if sum(panel.active_slots for panel in panels) != len(slots):
        warnings.append("Panel slot accounting did not match total slot count.")

    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    for panel in panels:
        draw.rectangle(panel.bbox, outline=(0, 96, 255), width=3)
        draw.text((panel.bbox[0], max(0, panel.bbox[1] - 18)), panel.panel_name, fill=(0, 96, 255))
    for index, slot in enumerate(slots, start=1):
        draw.rectangle(slot.bbox, outline=(255, 0, 0), width=2)
        draw.text((slot.bbox[0] + 2, slot.bbox[1] + 2), str(index), fill=(255, 0, 0))
    out = io.BytesIO()
    debug.save(out, format="PNG")
    return GeometryResult(panels=panels, slots=slots, warnings=warnings, debug_image_bytes=out.getvalue())
