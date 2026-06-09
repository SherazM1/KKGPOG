from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pdfplumber

from app.shared.clustering import boundaries_from_centers, cluster_positions, kmeans_1d
from app.shared.constants import LAST5_RE
from app.shared.models import CellData, PageData


def _without_wm_gift_card_placeholder(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            cleaned.extend(buffer)
            buffer = []

    for line in lines:
        token = re.sub(r"[^A-Z0-9]", "", line.upper())
        if token in {"WM", "GIFT", "GIFTCARD", "GIFTCA", "CARD", "RD"}:
            buffer.append(line)
            joined = re.sub(r"[^A-Z0-9]", "", "".join(buffer).upper())
            if joined == "WMGIFTCARD":
                buffer = []
            elif not "WMGIFTCARD".startswith(joined):
                flush_buffer()
            continue
        flush_buffer()
        cleaned.append(line)

    flush_buffer()
    return cleaned


def parse_label_cell_text(text: str) -> Tuple[str, str, Optional[int]]:
    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    lines = _without_wm_gift_card_placeholder(raw_lines)
    joined = " ".join(lines)

    m = LAST5_RE.search(joined)
    last5 = m.group(1) if m else ""

    nums = re.findall(r"\b(\d{1,3})\b", joined)
    qty = int(nums[-1]) if nums else None

    name = " ".join(
        ln
        for ln in lines
        if not (last5 and last5 in ln)
        and not (qty is not None and re.fullmatch(str(qty), ln))
    ).strip()

    return name, last5, qty


def _token_centers(words: List[dict]) -> Tuple[List[float], List[float]]:
    xs = [(w["x0"] + w["x1"]) / 2 for w in words]
    ys = [(w["top"] + w["bottom"]) / 2 for w in words]
    return xs, ys


def _cluster_axis(values: List[float], page_span: float, *, tol: Optional[float] = None) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    if len(values) == 1:
        return np.array([float(values[0])], dtype=float)

    sorted_values = sorted(float(v) for v in values)
    if tol is None:
        tol = max(8.0, page_span * 0.02)
    return cluster_positions(sorted_values, tol=tol)


def _infer_grid_centers(
    five_words: List[dict],
    page_width: float,
    page_height: float,
    n_cols: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = _token_centers(five_words)
    if n_cols and n_cols > 0:
        x_centers = kmeans_1d(xs, n_cols)
        y_centers = kmeans_1d(ys, max(1, round(len(five_words) / max(1, n_cols))))
        return x_centers, y_centers

    x_centers = _cluster_axis(xs, page_width)
    y_centers = _cluster_axis(ys, page_height, tol=max(14.0, page_height * 0.025))

    if len(x_centers) * len(y_centers) < len(five_words):
        # Fallback for mildly staggered text placement: choose a rectangular grid
        # closest to the detected aspect instead of dropping occupied cells.
        count = len(five_words)
        aspect = max(0.25, min(4.0, page_width / max(1.0, page_height)))
        candidates = []
        for cols in range(1, count + 1):
            rows = int(np.ceil(count / cols))
            capacity = rows * cols
            shape_score = abs((cols / max(1, rows)) - aspect)
            candidates.append((capacity - count, shape_score, cols, rows))
        _, _, cols, rows = min(candidates)
        x_centers = kmeans_1d(xs, cols)
        y_centers = kmeans_1d(ys, rows)

    return x_centers, y_centers


def _extract_fixed_grid_cells(
    page: pdfplumber.page.Page,
    five: List[dict],
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[CellData]]:
    x_bounds = boundaries_from_centers(x_centers)
    y_bounds = boundaries_from_centers(y_centers)
    xs, ys = _token_centers(five)

    cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
    for w, xc, yc in zip(five, xs, ys):
        col = int(np.argmin(np.abs(x_centers - xc)))
        row = int(np.argmin(np.abs(y_centers - yc)))
        dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
        key = (row, col)
        if key not in cell_map or dist < cell_map[key][0]:
            cell_map[key] = (dist, w)

    cells: List[CellData] = []
    for (row, col), (_, _w) in sorted(cell_map.items()):
        bbox = (
            float(x_bounds[col]),
            float(y_bounds[row]),
            float(x_bounds[col + 1]),
            float(y_bounds[row + 1]),
        )
        txt = (page.crop(bbox).extract_text() or "").strip()
        name, last5, qty = parse_label_cell_text(txt)
        cells.append(
            CellData(
                row=row,
                col=col,
                bbox=bbox,
                name=name,
                last5=last5,
                qty=qty,
                upc12=None,
            )
        )
    return x_bounds, y_bounds, cells


def _extract_row_shaped_cells(
    page: pdfplumber.page.Page,
    five: List[dict],
    y_centers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[CellData]]:
    y_bounds = boundaries_from_centers(y_centers)
    rows: Dict[int, List[dict]] = {idx: [] for idx in range(len(y_centers))}
    for w in five:
        yc = (w["top"] + w["bottom"]) / 2
        row = int(np.argmin(np.abs(y_centers - yc)))
        rows.setdefault(row, []).append(w)

    cells: List[CellData] = []
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    for row, row_words in sorted(rows.items()):
        if not row_words:
            continue
        row_xs = [(w["x0"] + w["x1"]) / 2 for w in row_words]
        x_centers = _cluster_axis(row_xs, float(page.width), tol=max(8.0, float(page.width) * 0.018))
        x_bounds = boundaries_from_centers(x_centers)
        x_min = float(x_bounds[0]) if x_min is None else min(x_min, float(x_bounds[0]))
        x_max = float(x_bounds[-1]) if x_max is None else max(x_max, float(x_bounds[-1]))

        row_cell_map: Dict[int, Tuple[float, dict]] = {}
        for w in row_words:
            xc = (w["x0"] + w["x1"]) / 2
            col = int(np.argmin(np.abs(x_centers - xc)))
            dist = abs(x_centers[col] - xc)
            if col not in row_cell_map or dist < row_cell_map[col][0]:
                row_cell_map[col] = (dist, w)

        for col, (_, _w) in sorted(row_cell_map.items()):
            bbox = (
                float(x_bounds[col]),
                float(y_bounds[row]),
                float(x_bounds[col + 1]),
                float(y_bounds[row + 1]),
            )
            txt = (page.crop(bbox).extract_text() or "").strip()
            name, last5, qty = parse_label_cell_text(txt)
            cells.append(
                CellData(
                    row=row,
                    col=col,
                    bbox=bbox,
                    name=name,
                    last5=last5,
                    qty=qty,
                    upc12=None,
                )
            )

    if x_min is None or x_max is None:
        x_bounds = np.array([0.0, float(page.width)], dtype=float)
    else:
        x_bounds = np.array([x_min, x_max], dtype=float)
    return x_bounds, y_bounds, cells


def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: Optional[int] = None) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [
                w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))
            ]
            if not five:
                continue

            x_centers, y_centers = _infer_grid_centers(
                five,
                float(page.width),
                float(page.height),
                n_cols,
            )

            if n_cols and n_cols > 0:
                x_bounds, y_bounds, cells = _extract_fixed_grid_cells(page, five, x_centers, y_centers)
            else:
                x_bounds, y_bounds, cells = _extract_row_shaped_cells(page, five, y_centers)

            pages.append(
                PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells)
            )
    return pages
