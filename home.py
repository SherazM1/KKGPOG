# home.py
"""
Streamlit Planogram Generator

Supports:
1. Standard Flat Display
2. Full Pallet / Multi-Zone Display
"""

from __future__ import annotations

import io
import os
import re
import math
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


LAST5_RE = re.compile(r"\b(\d{5})\b")
DIGITS_RE = re.compile(r"\D+")

N_COLS = 3
NAVY_RGB = (0.10, 0.16, 0.33)

DISPLAY_STANDARD = "Standard Flat Display"
DISPLAY_FULL_PALLET = "Full Pallet / Multi-Zone Display"


@dataclass(frozen=True)
class MatrixRow:
    upc12: str
    norm_name: str


@dataclass(frozen=True)
class CellData:
    row: int
    col: int
    bbox: Tuple[float, float, float, float]
    name: str
    last5: str
    qty: Optional[int]
    upc12: Optional[str]


@dataclass(frozen=True)
class PageData:
    page_index: int
    x_bounds: np.ndarray
    y_bounds: np.ndarray
    cells: List[CellData]


@dataclass(frozen=True)
class FullPalletCell:
    row: int
    col: int
    bbox: Tuple[float, float, float, float]
    name: str
    last5: str
    qty: Optional[int]
    upc12: Optional[str]
    zone: str


@dataclass(frozen=True)
class AnnotationBox:
    kind: str
    label: str
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class FullPalletPageData:
    page_index: int
    side_letter: str
    page_width: float
    page_height: float
    cells: List[FullPalletCell]
    annotations: List[AnnotationBox]
    bonus_y: Optional[float]


def resource_path(rel_path: str) -> str:
    return str(Path(__file__).resolve().parent / rel_path)


def _safe_register_font(name: str, rel_path: str) -> None:
    try:
        full_path = resource_path(os.path.join("assets", rel_path))
        if os.path.isfile(full_path):
            pdfmetrics.registerFont(TTFont(name, full_path))
        else:
            print(f"[fonts] Skipping {name}: not found at {full_path}")
    except Exception as e:
        print(f"[fonts] Skipping {name}: {e}")


_safe_register_font("Raleway", "Raleway-Regular.ttf")
_safe_register_font("Raleway-Bold", "Raleway-Bold.ttf")


def _font_name(preferred: str, fallback: str) -> str:
    try:
        pdfmetrics.getFont(preferred)
        return preferred
    except Exception:
        return fallback


TITLE_FONT = _font_name("Raleway-Bold", "Helvetica-Bold")
BODY_FONT = _font_name("Raleway", "Helvetica")
BODY_BOLD_FONT = _font_name("Raleway-Bold", "Helvetica-Bold")


def _norm_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _coerce_upc12(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    s = DIGITS_RE.sub("", s)
    if not s:
        return None
    return s.zfill(12)


def _find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(len(df), 50)):
        row = df.iloc[i].astype(str).str.upper().tolist()
        if any("UPC" in c for c in row) and any("NAME" in c for c in row):
            return i
    return 0


@st.cache_data(show_spinner=False)
def load_matrix_index(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    df_raw = pd.read_excel(io.BytesIO(matrix_bytes), header=None)
    header_row = _find_header_row(df_raw)
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = ["UPC", "Name", "Extra"][: df.shape[1]] + [
        f"col_{i}" for i in range(max(0, df.shape[1] - 3))
    ]

    df["upc12"] = df["UPC"].map(_coerce_upc12)
    df = df[df["upc12"].notna()].copy()
    df["last5"] = df["upc12"].str[-5:]
    df["norm_name"] = df["Name"].astype(str).map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["last5"])
        idx.setdefault(last5, []).append(
            MatrixRow(upc12=str(r["upc12"]), norm_name=str(r["norm_name"]))
        )

    return idx


def resolve_full_upc(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[str]:
    rows = idx.get(last5, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0].upc12

    target = _norm_name(label_name)
    best_upc = rows[0].upc12
    best_score = -1.0
    for r in rows:
        score = difflib.SequenceMatcher(None, target, r.norm_name).ratio()
        if score > best_score:
            best_score = score
            best_upc = r.upc12
    return best_upc


def kmeans_1d(values: List[float], k: int, iters: int = 40) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.array([], dtype=float)
    k = max(1, min(k, len(v)))

    qs = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    centers = np.quantile(v, qs)

    for _ in range(iters):
        d = np.abs(v[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            pts = v[labels == i]
            if len(pts) > 0:
                new_centers[i] = float(pts.mean())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return np.sort(centers)


def cluster_positions(values: List[float], tolerance: float) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)

    vals = sorted(float(v) for v in values)
    groups: List[List[float]] = [[vals[0]]]

    for v in vals[1:]:
        if abs(v - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])

    return np.array([float(np.mean(g)) for g in groups], dtype=float)


def boundaries_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(sorted(centers), dtype=float)
    if len(centers) == 0:
        return np.array([], dtype=float)
    if len(centers) == 1:
        step = 100.0
        return np.array([centers[0] - step / 2, centers[0] + step / 2], dtype=float)

    mids = (centers[:-1] + centers[1:]) / 2.0
    left = centers[0] - (mids[0] - centers[0])
    right = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[left], mids, [right]])


def parse_label_cell_text(text: str) -> Tuple[str, str, Optional[int]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    joined = " ".join(lines)

    m = LAST5_RE.search(joined)
    last5 = m.group(1) if m else ""

    nums = re.findall(r"\b(\d{1,3})\b", joined)
    qty = int(nums[-1]) if nums else None

    name_lines: List[str] = []
    for ln in lines:
        if last5 and last5 in ln:
            continue
        if qty is not None and re.fullmatch(str(qty), ln):
            continue
        name_lines.append(ln)
    name = " ".join(name_lines).strip()

    return name, last5, qty


def _word_text(word: dict) -> str:
    text = str(word.get("text", "")).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


def _word_center(word: dict) -> Tuple[float, float]:
    return ((float(word["x0"]) + float(word["x1"])) / 2.0, (float(word["top"]) + float(word["bottom"])) / 2.0)


def _union_bbox_from_words(words: List[dict], pad_x: float = 0.0, pad_y: float = 0.0) -> Tuple[float, float, float, float]:
    x0 = min(float(w["x0"]) for w in words) - pad_x
    top = min(float(w["top"]) for w in words) - pad_y
    x1 = max(float(w["x1"]) for w in words) + pad_x
    bottom = max(float(w["bottom"]) for w in words) + pad_y
    return (x0, top, x1, bottom)


def _group_words_by_proximity(words: List[dict], x_tol: float, y_tol: float) -> List[List[dict]]:
    groups: List[List[dict]] = []

    for word in sorted(words, key=lambda w: (_word_center(w)[1], _word_center(w)[0])):
        cx, cy = _word_center(word)
        placed = False

        for grp in groups:
            gx0, gt, gx1, gb = _union_bbox_from_words(grp)
            gcx = (gx0 + gx1) / 2.0
            gcy = (gt + gb) / 2.0

            within_x = abs(cx - gcx) <= max(x_tol, (gx1 - gx0) / 2.0 + x_tol * 0.3)
            within_y = abs(cy - gcy) <= max(y_tol, (gb - gt) / 2.0 + y_tol * 0.3)
            if within_x and within_y:
                grp.append(word)
                placed = True
                break

        if not placed:
            groups.append([word])

    merged = True
    while merged:
        merged = False
        new_groups: List[List[dict]] = []
        while groups:
            base = groups.pop(0)
            bx0, bt, bx1, bb = _union_bbox_from_words(base)

            i = 0
            while i < len(groups):
                gx0, gt, gx1, gb = _union_bbox_from_words(groups[i])
                cx_close = not (bx1 + x_tol < gx0 or gx1 + x_tol < bx0)
                cy_close = not (bb + y_tol < gt or gb + y_tol < bt)
                if cx_close and cy_close:
                    base.extend(groups.pop(i))
                    bx0, bt, bx1, bb = _union_bbox_from_words(base)
                    merged = True
                else:
                    i += 1
            new_groups.append(base)
        groups = new_groups

    return groups


def wrap_text(text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
    parts = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []
    for w in parts:
        trial = " ".join(cur + [w]).strip()
        if not cur or pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) != 6:
        return NAVY_RGB
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _draw_horizontal_gradient(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    left_rgb: Tuple[float, float, float],
    right_rgb: Tuple[float, float, float],
    steps: int = 80,
) -> None:
    steps = max(8, int(steps))
    for i in range(steps):
        t = i / (steps - 1)
        r_ = left_rgb[0] * (1 - t) + right_rgb[0] * t
        g_ = left_rgb[1] * (1 - t) + right_rgb[1] * t
        b_ = left_rgb[2] * (1 - t) + right_rgb[2] * t
        c.setFillColorRGB(r_, g_, b_)
        c.setStrokeColorRGB(r_, g_, b_)
        xi = x + (w * i / steps)
        wi = w / steps + 0.5
        c.rect(xi, y, wi, h, stroke=0, fill=1)


def _try_load_logo() -> Optional[Image.Image]:
    candidates = [
        Path.cwd() / "assets" / "KKG-Logo-02.png",
        Path(__file__).resolve().parent / "assets" / "KKG-Logo-02.png",
        Path.cwd() / "KKG-Logo-02.png",
        Path(__file__).resolve().parent / "KKG-Logo-02.png",
    ]
    for p in candidates:
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                continue
    return None


def _fit_single_line_font(
    text: str,
    font_name: str,
    max_width: float,
    max_height: float,
    min_size: float,
    max_size: float,
    step: float = 0.5,
) -> float:
    t = (text or "").strip()
    if not t:
        return min_size
    size = max_size
    while size >= min_size:
        if size <= max_height and pdfmetrics.stringWidth(t, font_name, size) <= max_width:
            return size
        size -= step
    return min_size


def _draw_cell_text_block(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    name: str,
    upc12: Optional[str],
    last5: str,
    qty: Optional[int],
) -> None:
    pad_x = 5
    pad_y = 4
    max_w = max(12.0, w - pad_x * 2)
    avail_h = max(10.0, h - pad_y * 2)

    upc_str = upc12 if upc12 else (f"???????{last5}" if last5 else "")

    name_size = 13.0
    while name_size >= 5.0:
        meta_size = max(5.5, name_size * 0.92)
        line_h_name = name_size * 1.16
        line_h_meta = meta_size * 1.14
        gap = 2.0

        name_lines = wrap_text(name or "", max_w, BODY_BOLD_FONT, name_size)
        upc_line = f"UPC: {upc_str}" if upc_str else ""
        qty_line = f"Qty: {qty}" if qty is not None else ""

        needed = (
            len(name_lines) * line_h_name
            + (gap if name_lines and (upc_line or qty_line) else 0)
            + (line_h_meta if upc_line else 0)
            + (line_h_meta if qty_line else 0)
        )

        if needed <= avail_h:
            ty = y + h - pad_y - name_size
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(BODY_BOLD_FONT, name_size)
            for ln in name_lines:
                c.drawString(x + pad_x, max(y + pad_y, ty), ln)
                ty -= line_h_name

            if name_lines and (upc_line or qty_line):
                ty -= gap

            c.setFont(BODY_BOLD_FONT, meta_size)
            if upc_line:
                c.drawString(x + pad_x, max(y + pad_y, ty), upc_line)
                ty -= line_h_meta
            if qty_line:
                c.drawString(x + pad_x, max(y + pad_y, ty), qty_line)
            return

        name_size -= 0.5

    fs = 5.0
    ty = y + h - pad_y - fs
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, fs)
    for ln in wrap_text(name or "", max_w, BODY_BOLD_FONT, fs):
        c.drawString(x + pad_x, max(y + pad_y, ty), ln)
        ty -= fs * 1.16
    if upc_str:
        c.drawString(x + pad_x, max(y + pad_y, ty), f"UPC: {upc_str}")
        ty -= fs * 1.14
    if qty is not None:
        c.drawString(x + pad_x, max(y + pad_y, ty), f"Qty: {qty}")


def _draw_centered_label(
    c: canvas.Canvas,
    label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    font_name: str,
    font_size: float,
    fill_rgb: Tuple[float, float, float],
) -> None:
    lines = wrap_text(label, max(20.0, w - 10), font_name, font_size)
    line_h = font_size * 1.12
    total_h = len(lines) * line_h
    start_y = y + (h + total_h) / 2.0 - font_size

    c.setFillColorRGB(*fill_rgb)
    c.setFont(font_name, font_size)
    for i, line in enumerate(lines):
        line_w = pdfmetrics.stringWidth(line, font_name, font_size)
        lx = x + (w - line_w) / 2.0
        ly = start_y - i * line_h
        c.drawString(lx, ly, line)


@st.cache_data(show_spinner=False)
def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: int) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]

            if not five:
                continue

            xs = [(w["x0"] + w["x1"]) / 2.0 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2.0 for w in five]

            x_centers = kmeans_1d(xs, n_cols)
            est_rows = max(1, int(round(len(five) / max(1, n_cols))))
            y_centers = kmeans_1d(ys, est_rows)

            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col = int(np.argmin(np.abs(x_centers - xc)))
                row = int(np.argmin(np.abs(y_centers - yc)))
                dist = float(abs(x_centers[col] - xc) + abs(y_centers[row] - yc))
                key = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_dist, _w) in sorted(cell_map.items()):
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

            pages.append(PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells))

    return pages


def _detect_phrase_boxes(words: List[dict], keywords: set[str], x_tol: float, y_tol: float) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    matched = [w for w in words if _word_text(w) in keywords]
    if not matched:
        return []

    groups = _group_words_by_proximity(matched, x_tol=x_tol, y_tol=y_tol)
    results: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for grp in groups:
        label = " ".join(str(w["text"]).strip() for w in sorted(grp, key=lambda ww: (_word_center(ww)[1], _word_center(ww)[0])))
        bbox = _union_bbox_from_words(grp, pad_x=6, pad_y=4)
        results.append((label, bbox))
    return results


def _detect_top_octagon_boxes(words: List[dict]) -> List[AnnotationBox]:
    results: List[AnnotationBox] = []
    top_words = [w for w in words if _word_text(w) == "TOP"]
    if not top_words:
        return results

    for top_word in top_words:
        top_cx, top_cy = _word_center(top_word)
        nearby = [
            w
            for w in words
            if abs(_word_center(w)[0] - top_cx) <= 70
            and abs(_word_center(w)[1] - top_cy) <= 60
        ]
        tokens = {_word_text(w) for w in nearby}
        if "OCTAGON" not in tokens:
            continue

        bbox = _union_bbox_from_words(nearby, pad_x=8, pad_y=6)
        ordered = sorted(nearby, key=lambda ww: (_word_center(ww)[1], _word_center(ww)[0]))
        raw_label = " ".join(str(w["text"]).strip() for w in ordered)
        raw_label = re.sub(r"\s+", " ", raw_label).strip()
        if not raw_label:
            continue

        duplicate = any(
            abs(bbox[0] - existing.bbox[0]) < 5
            and abs(bbox[1] - existing.bbox[1]) < 5
            and abs(bbox[2] - existing.bbox[2]) < 5
            and abs(bbox[3] - existing.bbox[3]) < 5
            for existing in results
        )
        if not duplicate:
            results.append(AnnotationBox(kind="top_octagon", label=raw_label, bbox=bbox))

    return results


def _detect_wm_placeholder_boxes(words: List[dict], page_height: float) -> List[AnnotationBox]:
    wanted = {"WM", "GIFTCARD", "GIFTCAR", "IN", "NEW", "PKG"}
    matched = [
        w for w in words
        if _word_text(w) in wanted and float(w["top"]) <= page_height * 0.38
    ]
    if not matched:
        return []

    groups = _group_words_by_proximity(matched, x_tol=14, y_tol=32)
    boxes: List[AnnotationBox] = []
    for grp in groups:
        tokens = {_word_text(w) for w in grp}
        if "WM" not in tokens or "NEW" not in tokens:
            continue
        bbox = _union_bbox_from_words(grp, pad_x=6, pad_y=4)
        boxes.append(
            AnnotationBox(
                kind="wm_new_pkg",
                label="WM GIFTCARD IN NEW PKG",
                bbox=bbox,
            )
        )
    return boxes


def _detect_bonus_box(words: List[dict], cells: List[FullPalletCell]) -> Tuple[Optional[AnnotationBox], Optional[float]]:
    bonus_words = [w for w in words if _word_text(w) == "BONUS"]
    if not bonus_words:
        return None, None

    bw = bonus_words[0]
    _, bonus_y = _word_center(bw)

    if not cells:
        bbox = _union_bbox_from_words([bw], pad_x=12, pad_y=8)
        return AnnotationBox(kind="bonus_strip", label="BONUS", bbox=bbox), bonus_y

    x0 = min(cell.bbox[0] for cell in cells) + 10
    x1 = max(cell.bbox[2] for cell in cells) - 10
    bbox = (x0, bonus_y - 10, x1, bonus_y + 10)
    return AnnotationBox(kind="bonus_strip", label="BONUS", bbox=bbox), bonus_y


def _classify_full_pallet_zone(cell_bbox: Tuple[float, float, float, float], bonus_y: Optional[float]) -> str:
    _, top, _, bottom = cell_bbox
    cy = (top + bottom) / 2.0
    if bonus_y is not None and cy < bonus_y - 8:
        return "upper_feature_grid"
    return "main_body_grid"


@st.cache_data(show_spinner=False)
def extract_full_pallet_pages(labels_pdf_bytes: bytes) -> List[FullPalletPageData]:
    pages: List[FullPalletPageData] = []

    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]

            if not five:
                continue

            page_width = float(page.width)
            page_height = float(page.height)

            xs = [(float(w["x0"]) + float(w["x1"])) / 2.0 for w in five]
            ys = [(float(w["top"]) + float(w["bottom"])) / 2.0 for w in five]

            x_tol = max(8.0, page_width * 0.015)
            y_tol = max(7.0, page_height * 0.011)

            x_centers = cluster_positions(xs, tolerance=x_tol)
            y_centers = cluster_positions(ys, tolerance=y_tol)

            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col = int(np.argmin(np.abs(x_centers - xc)))
                row = int(np.argmin(np.abs(y_centers - yc)))
                dist = float(abs(x_centers[col] - xc) + abs(y_centers[row] - yc))
                key = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            raw_cells: List[Tuple[int, int, Tuple[float, float, float, float], str, str, Optional[int]]] = []
            for (row, col), (_dist, _w) in sorted(cell_map.items()):
                bbox = (
                    float(x_bounds[col]),
                    float(y_bounds[row]),
                    float(x_bounds[col + 1]),
                    float(y_bounds[row + 1]),
                )
                txt = (page.crop(bbox).extract_text() or "").strip()
                name, last5, qty = parse_label_cell_text(txt)
                raw_cells.append((row, col, bbox, name, last5, qty))

            provisional_cells = [
                FullPalletCell(
                    row=row,
                    col=col,
                    bbox=bbox,
                    name=name,
                    last5=last5,
                    qty=qty,
                    upc12=None,
                    zone="main_body_grid",
                )
                for row, col, bbox, name, last5, qty in raw_cells
            ]

            annotations: List[AnnotationBox] = []
            bonus_annotation, bonus_y = _detect_bonus_box(words, provisional_cells)
            if bonus_annotation is not None:
                annotations.append(bonus_annotation)

            for label, bbox in _detect_phrase_boxes(words, {"MARKETING", "MESSAGE", "PANEL"}, x_tol=30, y_tol=26):
                annotations.append(AnnotationBox(kind="marketing_signage", label="MARKETING MESSAGE PANEL", bbox=bbox))

            for label, bbox in _detect_phrase_boxes(words, {"FRAUD", "SIGNAGE"}, x_tol=30, y_tol=20):
                annotations.append(AnnotationBox(kind="fraud_signage", label="FRAUD SIGNAGE", bbox=bbox))

            annotations.extend(_detect_top_octagon_boxes(words))
            annotations.extend(_detect_wm_placeholder_boxes(words, page_height=page_height))

            cells: List[FullPalletCell] = []
            for row, col, bbox, name, last5, qty in raw_cells:
                cells.append(
                    FullPalletCell(
                        row=row,
                        col=col,
                        bbox=bbox,
                        name=name,
                        last5=last5,
                        qty=qty,
                        upc12=None,
                        zone=_classify_full_pallet_zone(bbox, bonus_y),
                    )
                )

            side_letter = chr(ord("A") + pidx)
            pages.append(
                FullPalletPageData(
                    page_index=pidx,
                    side_letter=side_letter,
                    page_width=page_width,
                    page_height=page_height,
                    cells=cells,
                    annotations=annotations,
                    bonus_y=bonus_y,
                )
            )

    return pages


def crop_image_cell(
    images_doc: fitz.Document,
    page_index: int,
    bbox: Tuple[float, float, float, float],
    zoom: float = 3.0,
    inset: float = 0.08,
) -> Image.Image:
    page = images_doc.load_page(page_index)
    x0, top, x1, bottom = bbox
    w = x1 - x0
    h = bottom - top

    x0i = x0 + w * inset
    x1i = x1 - w * inset
    topi = top + h * inset
    bottomi = bottom - h * inset

    rect = fitz.Rect(x0i, topi, x1i, bottomi)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_standard_pog_pdf(
    pages: List[PageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    n_cols: int,
    title_prefix: str = "POG",
) -> bytes:
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    scale_factor = 1.5
    outer_margin = 44
    side_gap = 28
    top_bar_h = 90
    footer_h = 44
    side_label_h = 56

    cell_inset = 5
    border_w = 0.75
    img_frac = 0.58

    grad_left = _hex_to_rgb01("#5B63A9")
    grad_right = _hex_to_rgb01("#3E4577")

    logo_img = _try_load_logo()

    n_sides = len(pages)
    per_side_w = int(310 * scale_factor)

    side_scales: List[float] = []
    side_scaled_heights: List[float] = []

    for p in pages:
        x_min = float(p.x_bounds[0])
        x_max = float(p.x_bounds[-1])
        y_min = float(p.y_bounds[0])
        y_max = float(p.y_bounds[-1])
        sc = per_side_w / max(1e-6, x_max - x_min)
        side_scales.append(sc)
        side_scaled_heights.append(sc * max(1e-6, y_max - y_min))

    content_h = max(side_scaled_heights) if side_scaled_heights else 600.0
    page_w = outer_margin * 2 + n_sides * per_side_w + (n_sides - 1) * side_gap
    page_h = outer_margin + top_bar_h + side_label_h + content_h + footer_h + outer_margin

    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    try:
        header_y = page_h - top_bar_h
        _draw_horizontal_gradient(c, 0, header_y, page_w, top_bar_h, grad_left, grad_right, steps=120)

        left_pad = 16
        title_x = left_pad

        if logo_img is not None:
            lw, lh = logo_img.size
            target_h = top_bar_h * 0.64
            r = target_h / max(1, lh)
            dw, dh = lw * r, lh * r
            c.drawImage(ImageReader(logo_img), left_pad, header_y + (top_bar_h - dh) / 2, dw, dh, mask="auto")
            title_x = left_pad + dw + 16

        title_text = title_prefix.strip() or "POG"
        title_max_w = max(120.0, page_w - title_x - left_pad)
        title_size = _fit_single_line_font(
            title_text,
            TITLE_FONT,
            max_width=title_max_w,
            max_height=top_bar_h * 0.82,
            min_size=24.0,
            max_size=46.0,
            step=0.5,
        )
        c.setFillColorRGB(1, 1, 1)
        c.setFont(TITLE_FONT, title_size)
        c.drawString(title_x, header_y + (top_bar_h - title_size) / 2 + 2, title_text)

        c.setLineWidth(1.0)
        c.setStrokeColorRGB(0.88, 0.88, 0.92)
        c.line(0, header_y, page_w, header_y)

        cells_top = header_y - side_label_h
        content_bottom = outer_margin + footer_h

        c.setLineWidth(0.8)
        c.setStrokeColorRGB(0.82, 0.82, 0.82)
        c.line(outer_margin, content_bottom, page_w - outer_margin, content_bottom)

        footer_y = outer_margin + 13
        c.setFillColorRGB(*NAVY_RGB)
        c.setFont(BODY_BOLD_FONT, 12)
        footer_left = f"Date: {date.today().isoformat()}"
        footer_mid = "Generated by Kendal King"
        c.drawString(outer_margin, footer_y, footer_left)
        mid_w = pdfmetrics.stringWidth(footer_mid, BODY_BOLD_FONT, 12)
        c.drawString((page_w - mid_w) / 2, footer_y, footer_mid)

        for out_i, p in enumerate(pages):
            side_letter = chr(ord("A") + out_i)
            side_origin_x = outer_margin + out_i * (per_side_w + side_gap)

            badge_h = 34
            badge_w = 148
            badge_y = cells_top + (side_label_h - badge_h) / 2
            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(0.85, 0.85, 0.90)
            c.setLineWidth(0.85)
            c.roundRect(side_origin_x, badge_y, badge_w, badge_h, 8, stroke=1, fill=1)

            side_text = f"Side {side_letter}"
            side_font_size = _fit_single_line_font(
                side_text,
                TITLE_FONT,
                max_width=badge_w - 16,
                max_height=badge_h - 8,
                min_size=14.0,
                max_size=22.0,
                step=0.25,
            )
            side_text_w = pdfmetrics.stringWidth(side_text, TITLE_FONT, side_font_size)
            side_x = side_origin_x + (badge_w - side_text_w) / 2
            side_y = badge_y + (badge_h - side_font_size) / 2 + 1
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(TITLE_FONT, side_font_size)
            c.drawString(side_x, side_y, side_text)

            if out_i > 0:
                sep_x = side_origin_x - side_gap / 2
                c.setLineWidth(0.6)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(sep_x, content_bottom + 2, sep_x, header_y)

            x_min = float(p.x_bounds[0])
            y_min = float(p.y_bounds[0])
            sc = side_scales[out_i]

            for cell in p.cells:
                upc12 = resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                x0, top, x1, bottom = cell.bbox

                ox0 = side_origin_x + (x0 - x_min) * sc
                ox1 = side_origin_x + (x1 - x_min) * sc
                oy_top = cells_top - (top - y_min) * sc
                oy_bottom = cells_top - (bottom - y_min) * sc

                ox0 += cell_inset
                ox1 -= cell_inset
                oy_top -= cell_inset
                oy_bottom += cell_inset

                ow = ox1 - ox0
                oh = oy_top - oy_bottom
                if ow <= 2 or oh <= 2:
                    continue

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.72, 0.72, 0.72)
                c.setLineWidth(border_w)
                c.rect(ox0, oy_bottom, ow, oh, stroke=1, fill=1)

                img_area_h = oh * img_frac
                text_area_h = oh - img_area_h

                img = crop_image_cell(images_doc, p.page_index, cell.bbox, zoom=3.2, inset=0.08)
                iw, ih = img.size
                img_box_w = ow * 0.86
                img_box_h = img_area_h * 0.84
                r = min(img_box_w / max(1, iw), img_box_h / max(1, ih))
                dw, dh = iw * r, ih * r
                ix = ox0 + (ow - dw) / 2
                iy = oy_bottom + text_area_h + (img_area_h - dh) / 2
                c.drawImage(ImageReader(img), ix, iy, dw, dh, preserveAspectRatio=True, mask="auto")

                sep_y = oy_bottom + text_area_h
                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(ox0 + 3, sep_y, ox1 - 3, sep_y)

                _draw_cell_text_block(
                    c,
                    x=ox0,
                    y=oy_bottom,
                    w=ow,
                    h=text_area_h,
                    name=cell.name,
                    upc12=upc12,
                    last5=cell.last5,
                    qty=cell.qty,
                )

        c.showPage()
        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


def _transform_source_bbox_to_panel(
    bbox: Tuple[float, float, float, float],
    src_w: float,
    src_h: float,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> Tuple[float, float, float, float]:
    x0, top, x1, bottom = bbox
    ox0 = panel_x + (x0 / src_w) * panel_w
    ox1 = panel_x + (x1 / src_w) * panel_w
    oy_top = panel_y + panel_h - (top / src_h) * panel_h
    oy_bottom = panel_y + panel_h - (bottom / src_h) * panel_h
    return (ox0, oy_top, ox1, oy_bottom)


def _annotation_style(kind: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    if kind == "marketing_signage":
        return ((0.26, 0.92, 0.94), (0.35, 0.60, 0.62), (0.05, 0.10, 0.10))
    if kind == "fraud_signage":
        return ((0.98, 0.96, 0.50), (0.70, 0.68, 0.28), (0.12, 0.12, 0.12))
    if kind == "bonus_strip":
        return ((0.32, 0.95, 0.95), (0.25, 0.62, 0.62), NAVY_RGB)
    if kind == "top_octagon":
        return ((0.96, 0.96, 0.96), (0.72, 0.72, 0.72), (0.10, 0.10, 0.10))
    if kind == "wm_new_pkg":
        return ((_hex_to_rgb01("#4C75DD"))[0:3], (0.25, 0.35, 0.75), (1.0, 1.0, 1.0))
    return ((0.94, 0.94, 0.94), (0.75, 0.75, 0.75), NAVY_RGB)


def render_full_pallet_pdf(
    pages: List[FullPalletPageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
) -> bytes:
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    outer_margin = 42
    panel_gap = 24
    panel_pad = 10
    top_bar_h = 90
    footer_h = 44

    grad_left = _hex_to_rgb01("#5B63A9")
    grad_right = _hex_to_rgb01("#3E4577")
    logo_img = _try_load_logo()

    if pages:
        src_ratio = pages[0].page_width / max(1e-6, pages[0].page_height)
    else:
        src_ratio = 0.78

    panel_h = 640.0
    panel_w = panel_h * src_ratio

    page_w = outer_margin * 2 + len(pages) * panel_w + max(0, len(pages) - 1) * panel_gap
    page_h = outer_margin + top_bar_h + panel_h + footer_h + outer_margin

    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    try:
        header_y = page_h - top_bar_h
        _draw_horizontal_gradient(c, 0, header_y, page_w, top_bar_h, grad_left, grad_right, steps=120)

        left_pad = 16
        title_x = left_pad

        if logo_img is not None:
            lw, lh = logo_img.size
            target_h = top_bar_h * 0.64
            r = target_h / max(1, lh)
            dw, dh = lw * r, lh * r
            c.drawImage(ImageReader(logo_img), left_pad, header_y + (top_bar_h - dh) / 2, dw, dh, mask="auto")
            title_x = left_pad + dw + 16

        title_text = title_prefix.strip() or "POG"
        subtitle = "Full Pallet / Multi-Zone Display"
        c.setFillColorRGB(1, 1, 1)

        title_max_w = max(120.0, page_w - title_x - left_pad - 10)
        title_size = _fit_single_line_font(
            title_text,
            TITLE_FONT,
            max_width=title_max_w,
            max_height=36,
            min_size=24,
            max_size=40,
            step=0.5,
        )
        c.setFont(TITLE_FONT, title_size)
        c.drawString(title_x, header_y + top_bar_h * 0.56, title_text)
        c.setFont(BODY_FONT, 13)
        c.drawString(title_x, header_y + top_bar_h * 0.28, subtitle)

        c.setLineWidth(1.0)
        c.setStrokeColorRGB(0.88, 0.88, 0.92)
        c.line(0, header_y, page_w, header_y)

        footer_top = outer_margin + footer_h
        c.setLineWidth(0.8)
        c.setStrokeColorRGB(0.82, 0.82, 0.82)
        c.line(outer_margin, footer_top, page_w - outer_margin, footer_top)

        footer_y = outer_margin + 13
        c.setFillColorRGB(*NAVY_RGB)
        c.setFont(BODY_BOLD_FONT, 12)
        footer_left = f"Date: {date.today().isoformat()}"
        footer_mid = "Generated by Kendal King"
        c.drawString(outer_margin, footer_y, footer_left)
        mid_w = pdfmetrics.stringWidth(footer_mid, BODY_BOLD_FONT, 12)
        c.drawString((page_w - mid_w) / 2, footer_y, footer_mid)

        panel_y = footer_top + 12

        for idx, page in enumerate(pages):
            panel_x = outer_margin + idx * (panel_w + panel_gap)

            c.setFillColorRGB(0.985, 0.985, 0.985)
            c.setStrokeColorRGB(0.80, 0.80, 0.80)
            c.setLineWidth(0.9)
            c.rect(panel_x, panel_y, panel_w, panel_h, stroke=1, fill=1)

            side_title = f"Side {page.side_letter}"
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(TITLE_FONT, 15)
            title_w = pdfmetrics.stringWidth(side_title, TITLE_FONT, 15)
            c.drawString(panel_x + (panel_w - title_w) / 2, panel_y + panel_h - 20, side_title)

            if idx > 0:
                sep_x = panel_x - panel_gap / 2
                c.setLineWidth(0.6)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(sep_x, panel_y, sep_x, page_h - top_bar_h)

            inner_x = panel_x + panel_pad
            inner_y = panel_y + panel_pad
            inner_w = panel_w - panel_pad * 2
            inner_h = panel_h - panel_pad * 2

            page_letter_box_w = 56
            page_letter_box_h = 38
            page_letter_box_x = inner_x + (inner_w - page_letter_box_w) / 2
            page_letter_box_y = inner_y + 8

            for ann in page.annotations:
                ax0, ay_top, ax1, ay_bottom = _transform_source_bbox_to_panel(
                    ann.bbox,
                    src_w=page.page_width,
                    src_h=page.page_height,
                    panel_x=inner_x,
                    panel_y=inner_y,
                    panel_w=inner_w,
                    panel_h=inner_h,
                )
                aw = ax1 - ax0
                ah = ay_top - ay_bottom
                if aw <= 2 or ah <= 2:
                    continue

                fill_rgb, stroke_rgb, text_rgb = _annotation_style(ann.kind)
                c.setFillColorRGB(*fill_rgb)
                c.setStrokeColorRGB(*stroke_rgb)
                c.setLineWidth(0.7)
                c.rect(ax0, ay_bottom, aw, ah, stroke=1, fill=1)

                font_name = TITLE_FONT if ann.kind in {"bonus_strip", "top_octagon"} else BODY_BOLD_FONT
                font_size = _fit_single_line_font(
                    ann.label,
                    font_name,
                    max_width=max(20.0, aw - 8),
                    max_height=max(8.0, ah - 6),
                    min_size=6.0,
                    max_size=min(16.0, ah - 4),
                    step=0.5,
                )
                _draw_centered_label(
                    c,
                    ann.label,
                    x=ax0,
                    y=ay_bottom,
                    w=aw,
                    h=ah,
                    font_name=font_name,
                    font_size=font_size,
                    fill_rgb=text_rgb,
                )

            for cell in page.cells:
                upc12 = resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None

                ox0, oy_top, ox1, oy_bottom = _transform_source_bbox_to_panel(
                    cell.bbox,
                    src_w=page.page_width,
                    src_h=page.page_height,
                    panel_x=inner_x,
                    panel_y=inner_y,
                    panel_w=inner_w,
                    panel_h=inner_h,
                )

                cell_inset = 1.8 if cell.zone == "upper_feature_grid" else 1.2
                ox0 += cell_inset
                ox1 -= cell_inset
                oy_top -= cell_inset
                oy_bottom += cell_inset

                ow = ox1 - ox0
                oh = oy_top - oy_bottom
                if ow <= 2 or oh <= 2:
                    continue

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.65, 0.65, 0.65)
                c.setLineWidth(0.6 if cell.zone == "upper_feature_grid" else 0.45)
                c.rect(ox0, oy_bottom, ow, oh, stroke=1, fill=1)

                img_frac = 0.68 if cell.zone == "upper_feature_grid" else 0.58
                img_area_h = oh * img_frac
                text_area_h = oh - img_area_h

                img = crop_image_cell(images_doc, page.page_index, cell.bbox, zoom=3.0, inset=0.08)
                iw, ih = img.size
                img_box_w = ow * 0.84
                img_box_h = img_area_h * 0.82
                r = min(img_box_w / max(1, iw), img_box_h / max(1, ih))
                dw, dh = iw * r, ih * r

                ix = ox0 + (ow - dw) / 2
                iy = oy_bottom + text_area_h + (img_area_h - dh) / 2
                c.drawImage(ImageReader(img), ix, iy, dw, dh, preserveAspectRatio=True, mask="auto")

                sep_y = oy_bottom + text_area_h
                c.setLineWidth(0.4)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(ox0 + 2, sep_y, ox1 - 2, sep_y)

                _draw_cell_text_block(
                    c,
                    x=ox0,
                    y=oy_bottom,
                    w=ow,
                    h=text_area_h,
                    name=cell.name,
                    upc12=upc12,
                    last5=cell.last5,
                    qty=cell.qty,
                )

            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(0.68, 0.68, 0.68)
            c.setLineWidth(0.8)
            c.rect(page_letter_box_x, page_letter_box_y, page_letter_box_w, page_letter_box_h, stroke=1, fill=1)
            c.setFillColorRGB(0.08, 0.08, 0.08)
            side_font = 18
            c.setFont(TITLE_FONT, side_font)
            side_w = pdfmetrics.stringWidth(page.side_letter, TITLE_FONT, side_font)
            c.drawString(
                page_letter_box_x + (page_letter_box_w - side_w) / 2,
                page_letter_box_y + (page_letter_box_h - side_font) / 2 + 2,
                page.side_letter,
            )

        c.showPage()
        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


def main() -> None:
    st.set_page_config(page_title="Planogram Generator", layout="wide")
    st.title("Planogram Generator")

    with st.sidebar:
        st.header("Inputs")
        display_type = st.selectbox(
            "Display type",
            options=[DISPLAY_STANDARD, DISPLAY_FULL_PALLET],
            index=0,
        )

        st.caption("Uploads shown below are used by the selected display type.")
        matrix_file = st.file_uploader("Matrix/UPC Excel (.xlsx)", type=["xlsx"])
        labels_pdf = st.file_uploader("Labels PDF", type=["pdf"])
        images_pdf = st.file_uploader("Images PDF", type=["pdf"])

        st.divider()
        title_prefix = st.text_input("PDF title prefix", value="POG")
        out_name = st.text_input("Output filename", value="pog_export.pdf")

        generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Excels and PDFs to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

    if display_type == DISPLAY_STANDARD:
        pages = extract_pages_from_labels(labels_bytes, N_COLS)
        if not pages:
            st.error("Could not detect any grid cells from Labels PDF (no 5-digit UPCs found).")
            return

        st.subheader("Detected pages / sides")
        st.write(f"Pages detected: **{len(pages)}** (will export each page as Side A, B, C...)")

        preview_rows: List[dict] = []
        for i, p in enumerate(pages):
            side = chr(ord("A") + i)
            for cell in p.cells:
                preview_rows.append(
                    {
                        "Display Type": display_type,
                        "Side": side,
                        "Row": cell.row,
                        "Col": cell.col,
                        "Name": cell.name,
                        "Last5": cell.last5,
                        "Qty": cell.qty,
                        "UPC12 (resolved)": resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None,
                    }
                )

        st.dataframe(
            pd.DataFrame(preview_rows).sort_values(["Side", "Row", "Col"]),
            use_container_width=True,
            height=420,
        )

        if generate:
            with st.spinner("Generating PDF..."):
                pdf_bytes = render_standard_pog_pdf(
                    pages=pages,
                    images_pdf_bytes=images_bytes,
                    matrix_idx=matrix_idx,
                    n_cols=N_COLS,
                    title_prefix=title_prefix.strip() or "POG",
                )

            st.success("Done.")
            st.download_button(
                label="Download Planogram PDF",
                data=pdf_bytes,
                file_name=out_name if out_name.lower().endswith(".pdf") else f"{out_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        return

    full_pallet_pages = extract_full_pallet_pages(labels_bytes)
    if not full_pallet_pages:
        st.error("Could not detect any Full Pallet product cells from Labels PDF.")
        return

    st.subheader("Detected pages / sides")
    st.write(f"Pages detected: **{len(full_pallet_pages)}** (will export as one wide page with Side A, B, C...)")

    preview_rows_fp: List[dict] = []
    for page in full_pallet_pages:
        for cell in page.cells:
            preview_rows_fp.append(
                {
                    "Display Type": display_type,
                    "Side": page.side_letter,
                    "Zone": cell.zone,
                    "Row": cell.row,
                    "Col": cell.col,
                    "Name": cell.name,
                    "Last5": cell.last5,
                    "Qty": cell.qty,
                    "UPC12 (resolved)": resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None,
                }
            )

    st.dataframe(
        pd.DataFrame(preview_rows_fp).sort_values(["Side", "Zone", "Row", "Col"]),
        use_container_width=True,
        height=420,
    )

    if generate:
        with st.spinner("Generating PDF..."):
            pdf_bytes = render_full_pallet_pdf(
                pages=full_pallet_pages,
                images_pdf_bytes=images_bytes,
                matrix_idx=matrix_idx,
                title_prefix=title_prefix.strip() or "POG",
            )

        st.success("Done.")
        st.download_button(
            label="Download Planogram PDF",
            data=pdf_bytes,
            file_name=out_name if out_name.lower().endswith(".pdf") else f"{out_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()