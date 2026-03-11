# home.py
"""
Streamlit Planogram Generator
  • Standard Flat Display  — all sides on one wide page (original behaviour)
  • Full Pallet Display    — one portrait page per side, raster-background approach
"""

from __future__ import annotations

import io
import os
import re
import math
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
import fitz           # PyMuPDF
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LAST5_RE  = re.compile(r"\b(\d{5})\b")
DIGITS_RE = re.compile(r"\D+")

N_COLS    = 3          # standard flat display only
NAVY_RGB  = (0.10, 0.16, 0.33)

DISPLAY_STANDARD     = "Standard Flat Display"
DISPLAY_FULL_PALLET  = "Full Pallet / Multi-Zone Display"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixRow:
    upc12:    str
    norm_name: str
    cpp_qty:  Optional[int]      # from CPP column in Excel — authoritative qty


@dataclass(frozen=True)
class CellData:
    row:   int
    col:   int
    bbox:  Tuple[float, float, float, float]   # (x0, top, x1, bottom) pdfplumber coords
    name:  str
    last5: str
    qty:   Optional[int]   # from label text (standard display only; full-pallet uses cpp_qty)
    upc12: Optional[str]


@dataclass(frozen=True)
class PageData:                  # standard display
    page_index: int
    x_bounds:   np.ndarray
    y_bounds:   np.ndarray
    cells:      List[CellData]


@dataclass(frozen=True)
class AnnotationBox:
    kind:  str    # bonus_strip | gift_card_holders | marketing_signage | fraud_signage | wm_new_pkg
    label: str
    bbox:  Tuple[float, float, float, float]   # pdfplumber coords


@dataclass(frozen=True)
class FullPalletPage:
    page_index:  int
    side_letter: str
    cells:       List[CellData]
    annotations: List[AnnotationBox]


# ──────────────────────────────────────────────────────────────────────────────
# Fonts
# ──────────────────────────────────────────────────────────────────────────────

def _safe_register_font(name: str, rel_path: str) -> None:
    try:
        full = str(Path(__file__).resolve().parent / "assets" / rel_path)
        if os.path.isfile(full):
            pdfmetrics.registerFont(TTFont(name, full))
    except Exception:
        pass


_safe_register_font("Raleway",      "Raleway-Regular.ttf")
_safe_register_font("Raleway-Bold", "Raleway-Bold.ttf")


def _font(preferred: str, fallback: str) -> str:
    try:
        pdfmetrics.getFont(preferred)
        return preferred
    except Exception:
        return fallback


TITLE_FONT     = _font("Raleway-Bold", "Helvetica-Bold")
BODY_FONT      = _font("Raleway",      "Helvetica")
BODY_BOLD_FONT = _font("Raleway-Bold", "Helvetica-Bold")


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _norm_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _coerce_upc12(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"\.0$", "", str(v).strip())
    s = DIGITS_RE.sub("", s)
    return s.zfill(12) if s else None


def _coerce_int(v: object) -> Optional[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"[^\d\-]", "", re.sub(r"\.0$", "", str(v).strip()))
    try:
        return int(s) if s else None
    except ValueError:
        return None


def _norm_header(v: object) -> str:
    s = re.sub(r"[^A-Z0-9]+", "_", str(v).strip().upper()).strip("_")
    return s or "COL"


def _find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(len(df), 50)):
        row = df.iloc[i].astype(str).str.upper().tolist()
        if any("UPC" in c for c in row) and any(
            tok in " ".join(row) for tok in ("NAME", "DESCRIPTION", "CPP")
        ):
            return i
    return 0


def _pick_col(cols: List[str], tokens: List[str], fallback: int) -> str:
    uc = [c.upper() for c in cols]
    for t in tokens:
        for i, c in enumerate(uc):
            if t in c:
                return cols[i]
    return cols[min(fallback, len(cols) - 1)]


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = (h or "").lstrip("#").strip()
    if len(h) != 6:
        return NAVY_RGB
    return int(h[:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:], 16) / 255


def _try_load_logo() -> Optional[Image.Image]:
    for p in [
        Path.cwd() / "assets" / "KKG-Logo-02.png",
        Path(__file__).resolve().parent / "assets" / "KKG-Logo-02.png",
    ]:
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Matrix loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_matrix_index(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    df_raw   = pd.read_excel(io.BytesIO(matrix_bytes), header=None)
    hrow     = _find_header_row(df_raw)
    headers  = []
    seen: Dict[str, int] = {}
    for v in df_raw.iloc[hrow].tolist():
        base = _norm_header(v)
        n    = seen.get(base, 0)
        seen[base] = n + 1
        headers.append(base if n == 0 else f"{base}_{n+1}")

    df = df_raw.iloc[hrow + 1:].copy()
    df.columns = headers

    upc_col  = _pick_col(headers, ["UPC"],                     0)
    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"],     1 if len(headers) > 1 else 0)
    cpp_col  = _pick_col(headers, ["CPP"],                     -1)

    df["__upc12"]    = df[upc_col].map(_coerce_upc12)
    df["__name"]     = df[name_col].astype(str).fillna("")
    df["__cpp"]      = df[cpp_col].map(_coerce_int) if cpp_col in df.columns else None
    df = df[df["__upc12"].notna()].copy()
    df["__last5"]    = df["__upc12"].str[-5:]
    df["__norm"]     = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["__last5"])
        cpp   = None if df["__cpp"] is None else _coerce_int(r["__cpp"])
        idx.setdefault(last5, []).append(
            MatrixRow(upc12=str(r["__upc12"]), norm_name=str(r["__norm"]), cpp_qty=cpp)
        )
    return idx


def _resolve(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[MatrixRow]:
    rows = idx.get(last5, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    target = _norm_name(label_name)
    return max(rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio())


# ──────────────────────────────────────────────────────────────────────────────
# Clustering helpers
# ──────────────────────────────────────────────────────────────────────────────

def kmeans_1d(values: List[float], k: int, iters: int = 40) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.array([], dtype=float)
    k = max(1, min(k, len(v)))
    qs      = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    centers = np.quantile(v, qs)
    for _ in range(iters):
        d       = np.abs(v[:, None] - centers[None, :])
        labels  = d.argmin(axis=1)
        nc      = centers.copy()
        for i in range(k):
            pts = v[labels == i]
            if len(pts):
                nc[i] = float(pts.mean())
        if np.allclose(nc, centers):
            break
        centers = nc
    return np.sort(centers)


def cluster_positions(values: List[float], tol: float) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    vals   = sorted(float(v) for v in values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - float(np.mean(groups[-1]))) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([float(np.mean(g)) for g in groups])


def boundaries_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.sort(np.asarray(centers, dtype=float))
    if len(c) == 0:
        return c
    if len(c) == 1:
        return np.array([c[0] - 50, c[0] + 50])
    mids  = (c[:-1] + c[1:]) / 2
    left  = c[0]  - (mids[0]  - c[0])
    right = c[-1] + (c[-1]  - mids[-1])
    return np.concatenate([[left], mids, [right]])


# ──────────────────────────────────────────────────────────────────────────────
# Label parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_label_cell_text(text: str) -> Tuple[str, str, Optional[int]]:
    lines  = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    joined = " ".join(lines)
    m      = LAST5_RE.search(joined)
    last5  = m.group(1) if m else ""
    nums   = re.findall(r"\b(\d{1,3})\b", joined)
    qty    = int(nums[-1]) if nums else None
    name   = " ".join(
        ln for ln in lines
        if not (last5 and last5 in ln)
        and not (qty is not None and re.fullmatch(str(qty), ln))
    ).strip()
    return name, last5, qty


# ──────────────────────────────────────────────────────────────────────────────
# Standard display extraction  (UNCHANGED from original)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: int) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five  = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            if not five:
                continue

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            x_centers = kmeans_1d(xs, n_cols)
            y_centers = kmeans_1d(ys, max(1, round(len(five) / max(1, n_cols))))
            x_bounds  = boundaries_from_centers(x_centers)
            y_bounds  = boundaries_from_centers(y_centers)

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col  = int(np.argmin(np.abs(x_centers - xc)))
                row  = int(np.argmin(np.abs(y_centers - yc)))
                dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                key  = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_, _w) in sorted(cell_map.items()):
                bbox = (float(x_bounds[col]), float(y_bounds[row]),
                        float(x_bounds[col + 1]), float(y_bounds[row + 1]))
                txt  = (page.crop(bbox).extract_text() or "").strip()
                name, last5, qty = parse_label_cell_text(txt)
                cells.append(CellData(row=row, col=col, bbox=bbox,
                                      name=name, last5=last5, qty=qty, upc12=None))

            pages.append(PageData(page_index=pidx, x_bounds=x_bounds,
                                  y_bounds=y_bounds, cells=cells))
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Full-pallet page extraction
# ──────────────────────────────────────────────────────────────────────────────

def _wc(w: dict) -> Tuple[float, float]:
    return (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2


def _union(words: List[dict], px: float = 0, py: float = 0
           ) -> Tuple[float, float, float, float]:
    return (min(w["x0"] for w in words) - px,
            min(w["top"] for w in words) - py,
            max(w["x1"] for w in words) + px,
            max(w["bottom"] for w in words) + py)


def _group_nearby(words: List[dict], x_tol: float, y_tol: float
                  ) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for w in sorted(words, key=lambda ww: (_wc(ww)[1], _wc(ww)[0])):
        cx, cy = _wc(w)
        placed = False
        for g in groups:
            bx0, bt, bx1, bb = _union(g)
            gcx, gcy = (bx0 + bx1) / 2, (bt + bb) / 2
            if abs(cx - gcx) <= max(x_tol, (bx1 - bx0) / 2 + x_tol * 0.4) and \
               abs(cy - gcy) <= max(y_tol, (bb - bt) / 2 + y_tol * 0.4):
                g.append(w)
                placed = True
                break
        if not placed:
            groups.append([w])
    # merge pass
    changed = True
    while changed:
        changed = False
        merged: List[List[dict]] = []
        while groups:
            base = groups.pop(0)
            bx0, bt, bx1, bb = _union(base)
            i = 0
            while i < len(groups):
                gx0, gt, gx1, gb = _union(groups[i])
                if not (bx1 + x_tol < gx0 or gx1 + x_tol < bx0 or
                        bb  + y_tol < gt  or gb  + y_tol < bt):
                    base.extend(groups.pop(i))
                    bx0, bt, bx1, bb = _union(base)
                    changed = True
                else:
                    i += 1
            merged.append(base)
        groups = merged
    return groups


@st.cache_data(show_spinner=False)
def extract_full_pallet_pages(labels_pdf_bytes: bytes) -> List[FullPalletPage]:
    pages: List[FullPalletPage] = []

    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five  = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            if not five:
                continue

            pw, ph = float(page.width), float(page.height)

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            # Dense grid clustering for the full-pallet layout
            x_centers = cluster_positions(xs, tol=max(8, pw * 0.015))
            y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))
            x_bounds  = boundaries_from_centers(x_centers)
            y_bounds  = boundaries_from_centers(y_centers)

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col  = int(np.argmin(np.abs(x_centers - xc)))
                row  = int(np.argmin(np.abs(y_centers - yc)))
                dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                key  = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_, _w) in sorted(cell_map.items()):
                bbox = (float(x_bounds[col]), float(y_bounds[row]),
                        float(x_bounds[col + 1]), float(y_bounds[row + 1]))
                txt  = (page.crop(bbox).extract_text() or "").strip()
                name, last5, qty = parse_label_cell_text(txt)
                cells.append(CellData(row=row, col=col, bbox=bbox,
                                      name=name, last5=last5, qty=qty, upc12=None))

            # ── Annotation detection ─────────────────────────────────────────

            annotations: List[AnnotationBox] = []
            wt = lambda w: str(w.get("text", "")).strip().upper()

            # Content x-span (for full-width banners)
            if xs:
                cx0_content = min(xs) - 15
                cx1_content = max(xs) + 15
            else:
                cx0_content, cx1_content = 150.0, 470.0

            # 1. GIFT CARD HOLDERS banner — derived from WM / GIFTCAR / NEW / PKG cluster
            wm_grp = [w for w in words
                      if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "IN", "NEW", "PKG", "D"}
                      and float(w["top"]) < ph * 0.30]
            if wm_grp:
                bx0 = min(w["x0"] for w in wm_grp) - 6
                bx1 = max(w["x1"] for w in wm_grp) + 6
                by0 = min(w["top"] for w in wm_grp) - 8
                by1 = max(w["bottom"] for w in wm_grp) + 8
                # Full-width banner just above the WM boxes
                annotations.append(AnnotationBox(
                    kind="gift_card_holders", label="GIFT CARD HOLDERS",
                    bbox=(cx0_content, by0 - 22, cx1_content, by0 - 1)
                ))
                # Individual WM NEW PKG boxes
                sub_groups = _group_nearby(
                    [w for w in wm_grp if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "D", "IN", "NEW", "PKG"}],
                    x_tol=14, y_tol=22
                )
                for sg in sub_groups:
                    if len(sg) >= 3:   # must have at least a few words to be a real box
                        annotations.append(AnnotationBox(
                            kind="wm_new_pkg", label="WM GIFTCARD\nIN NEW PKG",
                            bbox=_union(sg, px=4, py=3)
                        ))

            # 2. BONUS full-width banner
            bonus_words = [w for w in words if wt(w) == "BONUS"]
            if bonus_words:
                bw   = min(bonus_words, key=lambda w: float(w["top"]))
                bcy  = (float(bw["top"]) + float(bw["bottom"])) / 2
                annotations.append(AnnotationBox(
                    kind="bonus_strip", label="BONUS",
                    bbox=(cx0_content, bcy - 12, cx1_content, bcy + 12)
                ))

            # 3. MARKETING MESSAGE PANEL (may appear on left/right edge, or both)
            mkt_words = [w for w in words if wt(w) in {"MARKETING", "MESSAGE", "PANEL"}]
            for grp in _group_nearby(mkt_words, x_tol=40, y_tol=20):
                if {wt(w) for w in grp} & {"MARKETING"}:
                    annotations.append(AnnotationBox(
                        kind="marketing_signage", label="MARKETING\nMESSAGE PANEL",
                        bbox=_union(grp, px=6, py=4)
                    ))

            # 4. FRAUD SIGNAGE (may appear on left/right edge, or both)
            fraud_words = [w for w in words if wt(w) in {"FRAUD", "SIGNAGE"}]
            for grp in _group_nearby(fraud_words, x_tol=30, y_tol=16):
                if {wt(w) for w in grp} & {"FRAUD"}:
                    annotations.append(AnnotationBox(
                        kind="fraud_signage", label="FRAUD\nSIGNAGE",
                        bbox=_union(grp, px=6, py=4)
                    ))

            pages.append(FullPalletPage(
                page_index=pidx,
                side_letter=chr(ord("A") + pidx),
                cells=cells,
                annotations=annotations,
            ))

    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Card-image cropping helper
# ──────────────────────────────────────────────────────────────────────────────

def crop_image_cell(
    images_doc: fitz.Document,
    page_index: int,
    bbox: Tuple[float, float, float, float],
    zoom: float = 3.0,
    inset: float = 0.08,
) -> Image.Image:
    page = images_doc.load_page(page_index)
    x0, top, x1, bottom = bbox
    w, h = x1 - x0, bottom - top
    rect = fitz.Rect(x0 + w * inset, top + h * inset,
                     x1 - w * inset, bottom - h * inset)
    pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_gradient(c: canvas.Canvas, x, y, w, h, left, right, steps=80) -> None:
    for i in range(max(8, steps)):
        t = i / (steps - 1)
        c.setFillColorRGB(
            left[0] * (1 - t) + right[0] * t,
            left[1] * (1 - t) + right[1] * t,
            left[2] * (1 - t) + right[2] * t,
        )
        xi = x + w * i / steps
        c.rect(xi, y, w / steps + 0.5, h, stroke=0, fill=1)


def wrap_text(text: str, max_w: float, font: str, size: float) -> List[str]:
    parts  = (text or "").split()
    lines, cur = [], []
    for p in parts:
        trial = " ".join(cur + [p])
        if not cur or pdfmetrics.stringWidth(trial, font, size) <= max_w:
            cur.append(p)
        else:
            lines.append(" ".join(cur))
            cur = [p]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _fit_font(text: str, font: str, max_w: float, max_h: float,
              lo: float, hi: float, step: float = 0.5) -> float:
    t = (text or "").strip()
    if not t:
        return lo
    s = hi
    while s >= lo:
        if s <= max_h and pdfmetrics.stringWidth(t, font, s) <= max_w:
            return s
        s -= step
    return lo


def _draw_centered(c: canvas.Canvas, text: str,
                   x, y, w, h, font, size, rgb) -> None:
    lines  = wrap_text(text, max(10.0, w - 8), font, size)
    lh     = size * 1.12
    total  = len(lines) * lh
    ty     = y + (h + total) / 2 - size
    c.setFillColorRGB(*rgb)
    c.setFont(font, size)
    for i, line in enumerate(lines):
        lw = pdfmetrics.stringWidth(line, font, size)
        c.drawString(x + (w - lw) / 2, ty - i * lh, line)


def _draw_header(c: canvas.Canvas, page_w, page_h, top_bar_h,
                 title_text, right_label, logo_img,
                 grad_l, grad_r) -> None:
    hy = page_h - top_bar_h
    _draw_gradient(c, 0, hy, page_w, top_bar_h, grad_l, grad_r, steps=120)

    tx = 14.0
    if logo_img:
        lw, lh = logo_img.size
        th     = top_bar_h * 0.62
        r      = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(ImageReader(logo_img), 14, hy + (top_bar_h - dh) / 2, dw, dh, mask="auto")
        tx = 14 + dw + 10

    c.setFillColorRGB(1, 1, 1)
    fs = _fit_font(title_text, TITLE_FONT, page_w - tx - 130, top_bar_h * 0.72, 18, 36)
    c.setFont(TITLE_FONT, fs)
    c.drawString(tx, hy + (top_bar_h - fs) / 2 + 2, title_text)

    if right_label:
        rfs  = _fit_font(right_label, TITLE_FONT, 120, top_bar_h * 0.68, 14, 28)
        rw   = pdfmetrics.stringWidth(right_label, TITLE_FONT, rfs)
        c.setFont(TITLE_FONT, rfs)
        c.drawString(page_w - rw - 14, hy + (top_bar_h - rfs) / 2 + 2, right_label)

    c.setLineWidth(0.8)
    c.setStrokeColorRGB(0.88, 0.88, 0.92)
    c.line(0, hy, page_w, hy)


def _draw_footer(c: canvas.Canvas, page_w, outer_margin, footer_h) -> None:
    fy = outer_margin + footer_h
    c.setLineWidth(0.7)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.line(outer_margin, fy, page_w - outer_margin, fy)
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, 11)
    left = f"Date: {date.today().isoformat()}"
    mid  = "Generated by Kendal King"
    c.drawString(outer_margin, outer_margin + 11, left)
    mw = pdfmetrics.stringWidth(mid, BODY_BOLD_FONT, 11)
    c.drawString((page_w - mw) / 2, outer_margin + 11, mid)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation styling
# ──────────────────────────────────────────────────────────────────────────────

_ANN_STYLES: Dict[str, Tuple[Tuple, Tuple, Tuple]] = {
    "gift_card_holders": (_hex_to_rgb("#1B5EC3"), _hex_to_rgb("#0E3A78"), (1.0, 1.0, 1.0)),
    "wm_new_pkg":        (_hex_to_rgb("#2563BE"), _hex_to_rgb("#1E4D9A"), (1.0, 1.0, 1.0)),
    "bonus_strip":       (_hex_to_rgb("#00BFCF"), _hex_to_rgb("#008A96"), (1.0, 1.0, 1.0)),
    "marketing_signage": (_hex_to_rgb("#5BC8D8"), _hex_to_rgb("#3A9BAA"), (0.05, 0.15, 0.28)),
    "fraud_signage":     (_hex_to_rgb("#F5E642"), _hex_to_rgb("#B8AC22"), (0.18, 0.14, 0.02)),
}

def _ann_style(kind: str):
    return _ANN_STYLES.get(kind, (_hex_to_rgb("#E8ECF5"), _hex_to_rgb("#A0AEC0"), NAVY_RGB))

def _ann_label(kind: str, raw: str) -> str:
    return {
        "gift_card_holders": "GIFT CARD HOLDERS",
        "bonus_strip":       "BONUS",
        "marketing_signage": "MARKETING\nMESSAGE PANEL",
        "fraud_signage":     "FRAUD\nSIGNAGE",
        "wm_new_pkg":        "WM GIFTCARD\nIN NEW PKG",
    }.get(kind, raw.upper())

def _ann_font(kind: str) -> str:
    return TITLE_FONT if kind in {"gift_card_holders", "bonus_strip"} else BODY_BOLD_FONT


# ──────────────────────────────────────────────────────────────────────────────
# Cell text block (standard display)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_cell_text_block(c: canvas.Canvas, x, y, w, h,
                          name, upc12, last5, qty) -> None:
    px, py  = 5, 4
    mw      = max(12.0, w - px * 2)
    avail   = max(10.0, h - py * 2)
    upc_str = upc12 if upc12 else (f"???????{last5}" if last5 else "")

    ns = 13.0
    while ns >= 5.0:
        ms    = max(5.0, ns * 0.86)
        lhn   = ns * 1.18
        lhm   = ms * 1.18
        nls   = wrap_text(name or "", mw, BODY_BOLD_FONT, ns)
        ul    = f"UPC: {upc_str}" if upc_str else ""
        ql    = f"Qty: {qty}"     if qty is not None else ""
        need  = len(nls) * lhn + (2.5 if nls and (ul or ql) else 0) + \
                (lhm if ul else 0) + (lhm if ql else 0)
        if need <= avail:
            ty = y + h - py - ns
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(BODY_BOLD_FONT, ns)
            for ln in nls:
                c.drawString(x + px, max(y + py, ty), ln)
                ty -= lhn
            if nls and (ul or ql):
                ty -= 2.5
            c.setFont(BODY_FONT, ms)
            if ul:
                c.drawString(x + px, max(y + py, ty), ul); ty -= lhm
            if ql:
                c.drawString(x + px, max(y + py, ty), ql)
            return
        ns -= 0.5

    fs = 5.0
    ty = y + h - py - fs
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, fs)
    for ln in wrap_text(name or "", mw, BODY_BOLD_FONT, fs):
        c.drawString(x + px, max(y + py, ty), ln); ty -= fs * 1.18
    c.setFont(BODY_FONT, fs)
    if upc_str:
        c.drawString(x + px, max(y + py, ty), f"UPC: {upc_str}"); ty -= fs * 1.18
    if qty is not None:
        c.drawString(x + px, max(y + py, ty), f"Qty: {qty}")


# ──────────────────────────────────────────────────────────────────────────────
# Standard Flat Display renderer  (UNCHANGED logic)
# ──────────────────────────────────────────────────────────────────────────────

def render_standard_pog_pdf(
    pages: List[PageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
) -> bytes:
    buf        = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    scale_factor = 1.5
    outer_margin = 44
    side_gap     = 28
    top_bar_h    = 90
    footer_h     = 44
    side_label_h = 56
    cell_inset   = 5
    border_w     = 0.75
    img_frac     = 0.58

    grad_l = _hex_to_rgb("#5B63A9")
    grad_r = _hex_to_rgb("#3E4577")
    logo   = _try_load_logo()

    n       = len(pages)
    psw     = int(310 * scale_factor)
    scales  = []
    heights = []

    for p in pages:
        xmin, xmax = float(p.x_bounds[0]), float(p.x_bounds[-1])
        ymin, ymax = float(p.y_bounds[0]), float(p.y_bounds[-1])
        sc = psw / max(1e-6, xmax - xmin)
        scales.append(sc)
        heights.append(sc * max(1e-6, ymax - ymin))

    ch    = max(heights) if heights else 600.0
    pw    = outer_margin * 2 + n * psw + max(0, n - 1) * side_gap
    ph    = outer_margin + top_bar_h + side_label_h + ch + footer_h + outer_margin
    c     = canvas.Canvas(buf, pagesize=(pw, ph))

    try:
        _draw_header(c, pw, ph, top_bar_h, title_prefix or "POG", "", logo, grad_l, grad_r)

        cells_top = ph - top_bar_h - side_label_h
        cb        = outer_margin + footer_h

        _draw_footer(c, pw, outer_margin, footer_h)

        for oi, p in enumerate(pages):
            sl  = chr(ord("A") + oi)
            sox = outer_margin + oi * (psw + side_gap)

            bh, bw = 34, 148
            by = cells_top + (side_label_h - bh) / 2
            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(0.85, 0.85, 0.90)
            c.setLineWidth(0.85)
            c.roundRect(sox, by, bw, bh, 8, stroke=1, fill=1)
            st_  = f"Side {sl}"
            sfs  = _fit_font(st_, TITLE_FONT, bw - 16, bh - 8, 14, 22)
            sw_  = pdfmetrics.stringWidth(st_, TITLE_FONT, sfs)
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(TITLE_FONT, sfs)
            c.drawString(sox + (bw - sw_) / 2, by + (bh - sfs) / 2 + 1, st_)

            if oi > 0:
                sx = sox - side_gap / 2
                c.setLineWidth(0.6)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(sx, cb + 2, sx, ph - top_bar_h)

            xmin = float(p.x_bounds[0])
            ymin = float(p.y_bounds[0])
            sc   = scales[oi]

            for cell in p.cells:
                match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                upc12 = match.upc12 if match else None
                qty   = cell.qty if cell.qty is not None else (match.cpp_qty if match else None)

                x0, top, x1, bot = cell.bbox
                ox0 = sox + (x0  - xmin) * sc + cell_inset
                ox1 = sox + (x1  - xmin) * sc - cell_inset
                ot  = cells_top - (top - ymin) * sc - cell_inset
                ob  = cells_top - (bot - ymin) * sc + cell_inset
                ow, oh = ox1 - ox0, ot - ob
                if ow <= 2 or oh <= 2:
                    continue

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.72, 0.72, 0.72)
                c.setLineWidth(border_w)
                c.rect(ox0, ob, ow, oh, stroke=1, fill=1)

                iah  = oh * img_frac
                tah  = oh - iah
                img  = crop_image_cell(images_doc, p.page_index, cell.bbox, zoom=3.2, inset=0.08)
                iw_, ih_ = img.size
                r    = min(ow * 0.86 / max(1, iw_), iah * 0.84 / max(1, ih_))
                dw, dh = iw_ * r, ih_ * r
                c.drawImage(ImageReader(img),
                            ox0 + (ow - dw) / 2, ob + tah + (iah - dh) / 2,
                            dw, dh, preserveAspectRatio=True, mask="auto")

                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(ox0 + 3, ob + tah, ox1 - 3, ob + tah)

                _draw_cell_text_block(c, ox0, ob, ow, tah,
                                      cell.name, upc12, cell.last5, qty)

        c.showPage()
        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# Full-Pallet Display renderer  — RASTER-BACKGROUND APPROACH
#
#   One portrait page per side.
#   • Render the images PDF page as a high-res raster → scale to fill content area.
#   • Map every annotation bbox (from labels PDF coords) onto the raster space
#     and draw solid coloured overlays.
#   • Map every cell bbox and draw a small text badge showing UPC (from Excel)
#     and CPP (from Excel — authoritative, never from label text).
# ──────────────────────────────────────────────────────────────────────────────

def render_full_pallet_pdf(
    pages: List[FullPalletPage],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
) -> bytes:
    buf        = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    # Output page — large portrait for readability
    PW, PH   = 936.0, 1296.0   # 13 × 18 in @ 72 dpi
    TOP_H    = 68.0
    FOOT_H   = 38.0
    MARGIN   = 18.0
    INNER    = 8.0

    # Content rectangle (where the raster lives)
    cx0 = MARGIN + INNER
    cy0 = MARGIN + FOOT_H + INNER
    cx1 = PW - MARGIN - INNER
    cy1 = PH - TOP_H - MARGIN - INNER
    CW  = cx1 - cx0
    CH  = cy1 - cy0

    grad_l = _hex_to_rgb("#5B63A9")
    grad_r = _hex_to_rgb("#3E4577")
    logo   = _try_load_logo()

    # Raster render zoom — higher = sharper images in output
    RASTER_ZOOM = 2.8

    c = canvas.Canvas(buf, pagesize=(PW, PH))

    try:
        for pdata in pages:
            # ── Header & footer ───────────────────────────────────────────────
            _draw_header(c, PW, PH, TOP_H,
                         title_prefix or "POG",
                         f"SIDE {pdata.side_letter}",
                         logo, grad_l, grad_r)
            _draw_footer(c, PW, MARGIN, FOOT_H)

            # Light border around content area
            c.setStrokeColorRGB(0.82, 0.84, 0.90)
            c.setLineWidth(0.6)
            c.rect(cx0, cy0, CW, CH, stroke=1, fill=0)

            # ── Render images PDF page as raster ─────────────────────────────
            img_page = images_doc.load_page(pdata.page_index)
            src_w    = float(img_page.rect.width)    # pdfplumber / source coords
            src_h    = float(img_page.rect.height)

            pix    = img_page.get_pixmap(
                        matrix=fitz.Matrix(RASTER_ZOOM, RASTER_ZOOM), alpha=False)
            raster = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Fit raster into content area, maintaining aspect ratio, centred
            r      = min(CW / src_w, CH / src_h)
            dw     = src_w * r
            dh     = src_h * r
            dx     = cx0 + (CW - dw) / 2
            dy     = cy0 + (CH - dh) / 2

            c.drawImage(ImageReader(raster), dx, dy, dw, dh)

            # ── Coordinate transform helper ───────────────────────────────────
            # src bbox  : (x0, top, x1, bottom)  — pdfplumber, y=0 at top
            # output RL : (out_x, out_y_bottom, out_w, out_h)  — y=0 at bottom
            def s2o(bbox: Tuple[float, float, float, float]
                    ) -> Tuple[float, float, float, float]:
                x0, t, x1, b = bbox
                ox0   = dx + x0 * r
                ox1   = dx + x1 * r
                oy_t  = dy + dh - t * r   # source top  → high RL y
                oy_b  = dy + dh - b * r   # source bot  → low  RL y
                return ox0, oy_b, ox1 - ox0, oy_t - oy_b   # x, y_bottom, w, h

            # ── Annotation overlays ───────────────────────────────────────────
            for ann in pdata.annotations:
                ax, ay, aw, ah = s2o(ann.bbox)
                if aw < 6 or ah < 2:
                    continue

                # Clamp to content area
                ax  = max(cx0, ax);  ax1 = min(cx1, ax + aw);  aw = ax1 - ax
                if aw < 4:
                    continue

                fill, stroke, txt_col = _ann_style(ann.kind)
                label = _ann_label(ann.kind, ann.label)
                font  = _ann_font(ann.kind)

                c.setFillColorRGB(*fill)
                c.setStrokeColorRGB(*stroke)
                c.setLineWidth(0.9)
                radius = max(2.0, min(8.0, ah * 0.30))
                c.roundRect(ax, ay, aw, ah, radius, stroke=1, fill=1)

                # For multi-line labels handle newlines
                lines = label.split("\n")
                lh    = ah / max(1, len(lines))
                for li, line in enumerate(lines):
                    fs  = _fit_font(line, font, max(8, aw - 8), max(4, lh - 3), 4.5, min(18, lh))
                    lw_ = pdfmetrics.stringWidth(line, font, fs)
                    c.setFillColorRGB(*txt_col)
                    c.setFont(font, fs)
                    c.drawString(ax + (aw - lw_) / 2,
                                 ay + ah - (li + 1) * lh + (lh - fs) / 2, line)

            # ── Cell text badges ──────────────────────────────────────────────
            # Each badge: small white pill at the bottom of the cell area,
            # showing the full UPC-12 and CPP qty (both from Excel matrix).
            for cell in pdata.cells:
                match  = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                upc12  = match.upc12    if match else None
                # RULE: CPP always from Excel matrix, never from label text
                qty    = match.cpp_qty  if match else None

                bx, by, bw, bh = s2o(cell.bbox)
                if bw < 8 or bh < 8:
                    continue

                # Badge occupies the bottom ~30% of the cell (max 22pt, min 10pt)
                badge_h = max(10.0, min(22.0, bh * 0.30))
                badge_y = by                         # bottom of cell
                badge_x = bx + 0.8
                badge_w = bw - 1.6

                # White background with light blue border
                c.setFillColorRGB(0.97, 0.98, 1.0)
                c.setStrokeColorRGB(0.65, 0.74, 0.90)
                c.setLineWidth(0.3)
                c.rect(badge_x, badge_y, badge_w, badge_h, stroke=1, fill=1)

                # Text: UPC + CPP, auto-sized
                short_upc  = (upc12[-5:] if upc12 else cell.last5) or "?"
                full_upc   = upc12 if upc12 else f"?????{short_upc}"
                cpp_text   = f"CPP {qty}" if qty is not None else ""

                fs = max(3.5, min(6.5, badge_h * 0.40))
                c.setFillColorRGB(*NAVY_RGB)
                c.setFont(BODY_BOLD_FONT, fs)
                lh_ = fs * 1.08

                two_lines = bool(cpp_text) and badge_h >= fs * 2.6
                if two_lines:
                    # line1: UPC; line2: CPP
                    ty = badge_y + badge_h - (badge_h - 2 * lh_) / 2 - 1.5 - fs
                    for txt_line in [full_upc, cpp_text]:
                        tw = pdfmetrics.stringWidth(txt_line, BODY_BOLD_FONT, fs)
                        # shrink to fit width
                        fs2 = fs
                        while tw > badge_w - 2 and fs2 > 3.0:
                            fs2 -= 0.3
                            tw  = pdfmetrics.stringWidth(txt_line, BODY_BOLD_FONT, fs2)
                        c.setFont(BODY_BOLD_FONT, fs2)
                        c.drawString(badge_x + max(0, (badge_w - tw) / 2), ty, txt_line)
                        ty -= lh_
                else:
                    # One line: last5 UPC + CPP
                    combined = short_upc + (f"  C{qty}" if qty is not None else "")
                    tw = pdfmetrics.stringWidth(combined, BODY_BOLD_FONT, fs)
                    while tw > badge_w - 2 and fs > 3.0:
                        fs -= 0.3
                        tw  = pdfmetrics.stringWidth(combined, BODY_BOLD_FONT, fs)
                    c.setFont(BODY_BOLD_FONT, fs)
                    ty = badge_y + (badge_h - fs) / 2
                    c.drawString(badge_x + max(0, (badge_w - tw) / 2), ty, combined)

            c.showPage()    # ← one page per side

        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Planogram Generator", layout="wide")
    st.title("Planogram Generator")

    with st.sidebar:
        st.header("Configuration")
        display_type = st.selectbox(
            "Display type",
            [DISPLAY_STANDARD, DISPLAY_FULL_PALLET],
            index=0,
        )
        st.divider()
        matrix_file = st.file_uploader("Matrix Excel (.xlsx)", type=["xlsx"])
        labels_pdf  = st.file_uploader("Labels PDF",           type=["pdf"])
        images_pdf  = st.file_uploader("Images PDF",           type=["pdf"])
        st.divider()
        title_prefix = st.text_input("PDF title prefix", "POG")
        out_name     = st.text_input("Output filename",   "pog_export.pdf")
        generate     = st.button("Generate POG PDF", type="primary",
                                 use_container_width=True)

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

    # ── Standard Flat Display ─────────────────────────────────────────────────
    if display_type == DISPLAY_STANDARD:
        pages = extract_pages_from_labels(labels_bytes, N_COLS)
        if not pages:
            st.error("No 5-digit UPC tokens found in Labels PDF.")
            return

        st.subheader(f"Detected {len(pages)} side(s)")
        rows = []
        for i, p in enumerate(pages):
            sl = chr(ord("A") + i)
            for cell in p.cells:
                match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                rows.append({
                    "Side": sl, "Row": cell.row, "Col": cell.col,
                    "Name": cell.name, "Last5": cell.last5,
                    "Qty": cell.qty if cell.qty is not None else
                           (match.cpp_qty if match else None),
                    "UPC12": match.upc12 if match else None,
                })
        st.dataframe(pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
                     use_container_width=True, height=420)

        if generate:
            with st.spinner("Rendering PDF…"):
                pdf = render_standard_pog_pdf(pages, images_bytes, matrix_idx,
                                              title_prefix.strip() or "POG")
            st.success("Done.")
            st.download_button("⬇ Download Planogram PDF", pdf,
                               file_name=out_name if out_name.endswith(".pdf")
                               else f"{out_name}.pdf",
                               mime="application/pdf",
                               use_container_width=True)

    # ── Full Pallet Display ───────────────────────────────────────────────────
    else:
        fp_pages = extract_full_pallet_pages(labels_bytes)
        if not fp_pages:
            st.error("No product cells detected in Labels PDF.")
            return

        st.subheader(f"Detected {len(fp_pages)} side(s) — one output page per side")

        rows = []
        for pg in fp_pages:
            for cell in pg.cells:
                match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                rows.append({
                    "Side": pg.side_letter, "Row": cell.row, "Col": cell.col,
                    "Name": cell.name, "Last5": cell.last5,
                    # CPP always from Excel
                    "CPP (Excel)": match.cpp_qty if match else None,
                    "UPC12":       match.upc12   if match else None,
                })
        st.dataframe(pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
                     use_container_width=True, height=420)

        if generate:
            with st.spinner("Rendering PDF…"):
                pdf = render_full_pallet_pdf(fp_pages, images_bytes, matrix_idx,
                                             title_prefix.strip() or "POG")
            st.success("Done.")
            st.download_button("⬇ Download Planogram PDF", pdf,
                               file_name=out_name if out_name.endswith(".pdf")
                               else f"{out_name}.pdf",
                               mime="application/pdf",
                               use_container_width=True)


if __name__ == "__main__":
    main()