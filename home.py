# home.py
"""
Streamlit Planogram Generator
  • Standard Flat Display  — all sides on one wide page (original behaviour)
  • Full Pallet Display    — template-style rebuild (NO raster background), one page per side
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


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LAST5_RE = re.compile(r"\b(\d{5})\b")
DIGITS_RE = re.compile(r"\D+")

N_COLS = 3  # standard flat display only
NAVY_RGB = (0.10, 0.16, 0.33)

DISPLAY_STANDARD = "Standard Flat Display"
DISPLAY_FULL_PALLET = "Full Pallet / Multi-Zone Display"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixRow:
    upc12: str
    norm_name: str
    display_name: str
    cpp_qty: Optional[int]  # from CPP column in Excel — authoritative qty


@dataclass(frozen=True)
class CellData:
    row: int
    col: int
    bbox: Tuple[float, float, float, float]  # (x0, top, x1, bottom) pdfplumber coords
    name: str
    last5: str
    qty: Optional[int]  # label text qty (standard display only; full-pallet uses cpp_qty)
    upc12: Optional[str]


@dataclass(frozen=True)
class PageData:  # standard display
    page_index: int
    x_bounds: np.ndarray
    y_bounds: np.ndarray
    cells: List[CellData]


@dataclass(frozen=True)
class AnnotationBox:
    kind: str  # bonus_strip | gift_card_holders | marketing_signage | fraud_signage | wm_new_pkg
    label: str
    bbox: Tuple[float, float, float, float]  # pdfplumber coords


@dataclass(frozen=True)
class FullPalletPage:
    page_index: int
    side_letter: str
    cells: List[CellData]
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


_safe_register_font("Raleway", "Raleway-Regular.ttf")
_safe_register_font("Raleway-Bold", "Raleway-Bold.ttf")


def _font(preferred: str, fallback: str) -> str:
    try:
        pdfmetrics.getFont(preferred)
        return preferred
    except Exception:
        return fallback


TITLE_FONT = _font("Raleway-Bold", "Helvetica-Bold")
BODY_FONT = _font("Raleway", "Helvetica")
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
    df_raw = pd.read_excel(io.BytesIO(matrix_bytes), header=None)
    hrow = _find_header_row(df_raw)
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for v in df_raw.iloc[hrow].tolist():
        base = _norm_header(v)
        n = seen.get(base, 0)
        seen[base] = n + 1
        headers.append(base if n == 0 else f"{base}_{n+1}")

    df = df_raw.iloc[hrow + 1:].copy()
    df.columns = headers

    upc_col = _pick_col(headers, ["UPC"], 0)
    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"], 1 if len(headers) > 1 else 0)
    cpp_col = _pick_col(headers, ["CPP"], -1)

    df["__upc12"] = df[upc_col].map(_coerce_upc12)
    df["__name"] = df[name_col].astype(str).fillna("")
    df["__cpp"] = df[cpp_col].map(_coerce_int) if cpp_col in df.columns else None
    df = df[df["__upc12"].notna()].copy()
    df["__last5"] = df["__upc12"].str[-5:]
    df["__norm"] = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["__last5"])
        cpp = None if df["__cpp"] is None else _coerce_int(r["__cpp"])
        display_name = str(r["__name"]).strip()
        idx.setdefault(last5, []).append(
            MatrixRow(
                upc12=str(r["__upc12"]),
                norm_name=str(r["__norm"]),
                display_name=display_name,
                cpp_qty=cpp,
            )
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
    qs = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    centers = np.quantile(v, qs)
    for _ in range(iters):
        d = np.abs(v[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        nc = centers.copy()
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
    vals = sorted(float(v) for v in values)
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
    mids = (c[:-1] + c[1:]) / 2
    left = c[0] - (mids[0] - c[0])
    right = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[left], mids, [right]])


# ──────────────────────────────────────────────────────────────────────────────
# Label parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_label_cell_text(text: str) -> Tuple[str, str, Optional[int]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    joined = " ".join(lines)
    m = LAST5_RE.search(joined)
    last5 = m.group(1) if m else ""
    nums = re.findall(r"\b(\d{1,3})\b", joined)
    qty = int(nums[-1]) if nums else None
    name = " ".join(
        ln for ln in lines
        if not (last5 and last5 in ln)
        and not (qty is not None and re.fullmatch(str(qty), ln))
    ).strip()
    return name, last5, qty


# ──────────────────────────────────────────────────────────────────────────────
# Standard display extraction  (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: int) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            if not five:
                continue

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            x_centers = kmeans_1d(xs, n_cols)
            y_centers = kmeans_1d(ys, max(1, round(len(five) / max(1, n_cols))))
            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)

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
                    CellData(row=row, col=col, bbox=bbox, name=name, last5=last5, qty=qty, upc12=None)
                )

            pages.append(PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells))
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Full-pallet page extraction (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

def _wc(w: dict) -> Tuple[float, float]:
    return (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2


def _union(words: List[dict], px: float = 0, py: float = 0) -> Tuple[float, float, float, float]:
    return (
        min(w["x0"] for w in words) - px,
        min(w["top"] for w in words) - py,
        max(w["x1"] for w in words) + px,
        max(w["bottom"] for w in words) + py,
    )


def _group_nearby(words: List[dict], x_tol: float, y_tol: float) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for w in sorted(words, key=lambda ww: (_wc(ww)[1], _wc(ww)[0])):
        cx, cy = _wc(w)
        placed = False
        for g in groups:
            bx0, bt, bx1, bb = _union(g)
            gcx, gcy = (bx0 + bx1) / 2, (bt + bb) / 2
            if abs(cx - gcx) <= max(x_tol, (bx1 - bx0) / 2 + x_tol * 0.4) and abs(cy - gcy) <= max(
                y_tol, (bb - bt) / 2 + y_tol * 0.4
            ):
                g.append(w)
                placed = True
                break
        if not placed:
            groups.append([w])
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
                if not (bx1 + x_tol < gx0 or gx1 + x_tol < bx0 or bb + y_tol < gt or gb + y_tol < bt):
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
            five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            if not five:
                continue

            pw, ph = float(page.width), float(page.height)

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            x_centers = cluster_positions(xs, tol=max(8, pw * 0.015))
            y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))
            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)

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
                    CellData(row=row, col=col, bbox=bbox, name=name, last5=last5, qty=qty, upc12=None)
                )

            annotations: List[AnnotationBox] = []
            wt = lambda w: str(w.get("text", "")).strip().upper()

            if xs:
                cx0_content = min(xs) - 15
                cx1_content = max(xs) + 15
            else:
                cx0_content, cx1_content = 150.0, 470.0

            wm_grp = [
                w
                for w in words
                if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "IN", "NEW", "PKG", "D"} and float(w["top"]) < ph * 0.30
            ]
            if wm_grp:
                by0 = min(w["top"] for w in wm_grp) - 8
                annotations.append(
                    AnnotationBox(
                        kind="gift_card_holders",
                        label="GIFT CARD HOLDERS",
                        bbox=(cx0_content, by0 - 22, cx1_content, by0 - 1),
                    )
                )
                sub_groups = _group_nearby(
                    [w for w in wm_grp if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "D", "IN", "NEW", "PKG"}],
                    x_tol=14,
                    y_tol=22,
                )
                for sg in sub_groups:
                    if len(sg) >= 3:
                        annotations.append(
                            AnnotationBox(
                                kind="wm_new_pkg",
                                label="WM GIFTCARD\nIN NEW PKG",
                                bbox=_union(sg, px=4, py=3),
                            )
                        )

            bonus_words = [w for w in words if wt(w) == "BONUS"]
            if bonus_words:
                bw = min(bonus_words, key=lambda w: float(w["top"]))
                bcy = (float(bw["top"]) + float(bw["bottom"])) / 2
                annotations.append(
                    AnnotationBox(
                        kind="bonus_strip",
                        label="BONUS",
                        bbox=(cx0_content, bcy - 12, cx1_content, bcy + 12),
                    )
                )

            mkt_words = [w for w in words if wt(w) in {"MARKETING", "MESSAGE", "PANEL"}]
            for grp in _group_nearby(mkt_words, x_tol=40, y_tol=20):
                if {wt(w) for w in grp} & {"MARKETING"}:
                    annotations.append(
                        AnnotationBox(
                            kind="marketing_signage",
                            label="MARKETING\nMESSAGE PANEL",
                            bbox=_union(grp, px=6, py=4),
                        )
                    )

            fraud_words = [w for w in words if wt(w) in {"FRAUD", "SIGNAGE"}]
            for grp in _group_nearby(fraud_words, x_tol=30, y_tol=16):
                if {wt(w) for w in grp} & {"FRAUD"}:
                    annotations.append(
                        AnnotationBox(
                            kind="fraud_signage",
                            label="FRAUD\nSIGNAGE",
                            bbox=_union(grp, px=6, py=4),
                        )
                    )

            pages.append(
                FullPalletPage(
                    page_index=pidx,
                    side_letter=chr(ord("A") + pidx),
                    cells=cells,
                    annotations=annotations,
                )
            )

    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Card-image cropping helper
# ──────────────────────────────────────────────────────────────────────────────

def crop_image_cell(
    images_doc: fitz.Document,
    page_index: int,
    bbox: Tuple[float, float, float, float],
    zoom: float = 2.6,
    inset: float = 0.045,
) -> Image.Image:
    page = images_doc.load_page(page_index)
    x0, top, x1, bottom = bbox
    w, h = x1 - x0, bottom - top
    rect = fitz.Rect(x0 + w * inset, top + h * inset, x1 - w * inset, bottom - h * inset)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers (existing + new for Full Pallet)
# ──────────────────────────────────────────────────────────────────────────────

def wrap_text(text: str, max_w: float, font: str, size: float) -> List[str]:
    parts = (text or "").split()
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


def _fit_font(
    text: str,
    font: str,
    max_w: float,
    max_h: float,
    lo: float,
    hi: float,
    step: float = 0.5,
) -> float:
    t = (text or "").strip()
    if not t:
        return lo
    s = hi
    while s >= lo:
        if s <= max_h and pdfmetrics.stringWidth(t, font, s) <= max_w:
            return s
        s -= step
    return lo


def _ellipsis(text: str, font: str, size: float, max_w: float) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if pdfmetrics.stringWidth(t, font, size) <= max_w:
        return t
    ell = "…"
    lo, hi = 0, len(t)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = t[:mid].rstrip() + ell
        if pdfmetrics.stringWidth(cand, font, size) <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


_STOPWORDS = {
    "GIFT",
    "CARD",
    "CARDS",
    "VALUE",
    "THE",
    "AND",
    "FOR",
    "WITH",
    "IN",
    "OF",
    "A",
    "AN",
    "WALMART",
    "WM",
    "VGC",
    "GC",
}


def _is_important_token(tok: str) -> bool:
    t = tok.strip().upper()
    if not t:
        return False
    if "$" in t and re.search(r"\d", t):
        return True
    if re.fullmatch(r"\d{1,3}", t):
        return True
    if re.search(r"\d", t) and any(u in t for u in ("PK", "CT", "OZ", "LB", "MG", "ML")):
        return True
    if re.fullmatch(r"[A-Z]\d{1,3}", t):  # e.g., D82
        return True
    return False


def _compact_one_line_name(name: str) -> str:
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    tokens = raw.split()
    imp = [t for t in tokens if _is_important_token(t)]
    core = [t for t in tokens if t.upper() not in _STOPWORDS]
    use = core if core else tokens
    for t in imp:
        if t not in use:
            use.append(t)
    return " ".join(use)


def _draw_full_pallet_header(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    header_h: float,
    title_text: str,
    right_label: str,
    logo_img: Optional[Image.Image],
) -> None:
    y0 = page_h - header_h
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, y0, page_w, header_h, stroke=0, fill=1)

    tx = 18.0
    if logo_img:
        lw, lh = logo_img.size
        th = header_h * 0.62
        r = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(ImageReader(logo_img), 18, y0 + (header_h - dh) / 2, dw, dh, mask="auto")
        tx = 18 + dw + 12

    c.setFillColorRGB(*NAVY_RGB)
    fs = _fit_font(title_text, BODY_BOLD_FONT, page_w - tx - 160, header_h * 0.70, 14, 26)
    c.setFont(BODY_BOLD_FONT, fs)
    c.drawString(tx, y0 + (header_h - fs) / 2 + 1, title_text)

    if right_label:
        rfs = _fit_font(right_label, BODY_BOLD_FONT, 150, header_h * 0.68, 12, 24)
        rw = pdfmetrics.stringWidth(right_label, BODY_BOLD_FONT, rfs)
        c.setFont(BODY_BOLD_FONT, rfs)
        c.drawString(page_w - rw - 18, y0 + (header_h - rfs) / 2 + 1, right_label)

    c.setLineWidth(0.8)
    c.setStrokeColorRGB(0.86, 0.86, 0.88)
    c.line(0, y0, page_w, y0)


def _draw_footer(c: canvas.Canvas, page_w: float, outer_margin: float, footer_h: float) -> None:
    fy = outer_margin + footer_h
    c.setLineWidth(0.7)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.line(outer_margin, fy, page_w - outer_margin, fy)
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, 11)
    left = f"Date: {date.today().isoformat()}"
    mid = "Generated by Kendal King"
    c.drawString(outer_margin, outer_margin + 11, left)
    mw = pdfmetrics.stringWidth(mid, BODY_BOLD_FONT, 11)
    c.drawString((page_w - mw) / 2, outer_margin + 11, mid)


# ──────────────────────────────────────────────────────────────────────────────
# Full-Pallet Display renderer — TEMPLATE-STYLE REBUILD (NO RASTER BG)
# ──────────────────────────────────────────────────────────────────────────────

def render_full_pallet_pdf(
    pages: List[FullPalletPage],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
) -> bytes:
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    PORTRAIT = (936.0, 1296.0)
    LANDSCAPE = (1296.0, 936.0)

    MARGIN = 28.0
    HEADER_H = 56.0
    FOOTER_H = 38.0

    SECTION_BAR_H = 26.0
    SECTION_BAR_GAP = 10.0
    BETWEEN_SECTIONS_GAP = 16.0

    BASE_GUTTER_X = 14.0

    BAR_FILL = _hex_to_rgb("#77B5F0")
    BAR_TEXT = NAVY_RGB

    EMPTY_STROKE = (0.88, 0.88, 0.88)
    FILLED_STROKE = (0.78, 0.78, 0.80)

    logo = _try_load_logo()

    def _marker_y(p: FullPalletPage, kind: str) -> Optional[float]:
        for ann in p.annotations:
            if ann.kind == kind:
                x0, t, x1, b = ann.bbox
                return (t + b) / 2
        return None

    def _cell_center(cell: CellData) -> Tuple[float, float]:
        x0, t, x1, b = cell.bbox
        return (x0 + x1) / 2, (t + b) / 2

    def _row_order(cells: List[CellData]) -> List[int]:
        row_to_y: Dict[int, List[float]] = {}
        for cl in cells:
            _, y = _cell_center(cl)
            row_to_y.setdefault(cl.row, []).append(y)
        return sorted(row_to_y.keys(), key=lambda r: float(np.mean(row_to_y[r])))

    def _col_order_and_gaps(cells: List[CellData]) -> Tuple[List[int], List[int]]:
        col_to_x: Dict[int, List[float]] = {}
        for cl in cells:
            x, _ = _cell_center(cl)
            col_to_x.setdefault(cl.col, []).append(x)
        cols = sorted(col_to_x.keys(), key=lambda c_: float(np.mean(col_to_x[c_])))
        if len(cols) <= 1:
            return cols, []
        xs = [float(np.mean(col_to_x[c_])) for c_ in cols]
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        med = float(np.median(diffs)) if diffs else 1.0
        if med <= 1e-6:
            return cols, [1] * (len(cols) - 1)

        units: List[int] = []
        for d in diffs:
            u = int(round(d / med))
            units.append(max(1, min(4, u)))
        return cols, units

    def _split_sections(p: FullPalletPage) -> List[Tuple[Optional[str], List[CellData]]]:
        bonus_y = _marker_y(p, "bonus_strip")
        if bonus_y is None:
            rows = _row_order(p.cells)
            if len(rows) < 2:
                return [("GIFT CARD HOLDERS", p.cells)]
            row_to_y: Dict[int, float] = {}
            for r in rows:
                ys = [(_cell_center(c)[1]) for c in p.cells if c.row == r]
                row_to_y[r] = float(np.mean(ys)) if ys else float("inf")
            yc = [row_to_y[r] for r in rows]
            gaps = [yc[i + 1] - yc[i] for i in range(len(yc) - 1)]
            if not gaps:
                return [("GIFT CARD HOLDERS", p.cells)]
            i_max = int(np.argmax(gaps))
            if gaps[i_max] < float(np.median(gaps)) * 1.6:
                return [("GIFT CARD HOLDERS", p.cells)]
            top_rows = set(rows[: i_max + 1])
            bot_rows = set(rows[i_max + 1 :])
        else:
            rows = _row_order(p.cells)
            row_to_y: Dict[int, float] = {}
            for r in rows:
                ys = [(_cell_center(c)[1]) for c in p.cells if c.row == r]
                row_to_y[r] = float(np.mean(ys)) if ys else float("inf")
            yc = [row_to_y[r] for r in rows]
            boundary = 0
            for i in range(len(yc) - 1):
                if yc[i] < bonus_y < yc[i + 1]:
                    boundary = i
                    break
            top_rows = set(rows[: boundary + 1])
            bot_rows = set(rows[boundary + 1 :])

        gift_cells = [c for c in p.cells if c.row in top_rows]
        bonus_cells = [c for c in p.cells if c.row in bot_rows]

        out: List[Tuple[Optional[str], List[CellData]]] = []
        if gift_cells:
            out.append(("GIFT CARD HOLDERS", gift_cells))
        if bonus_cells:
            out.append(("BONUS", bonus_cells))
        return out or [("GIFT CARD HOLDERS", p.cells)]

    def _choose_geometry(sections_meta: List[Tuple[int, int, List[int]]]) -> Tuple[float, float, float, float, float, float]:
        def eval_page(pw: float, ph: float) -> Optional[Tuple[float, float, float, float, float, float]]:
            cx0 = MARGIN
            cx1 = pw - MARGIN
            cy0 = MARGIN + FOOTER_H
            cy1 = ph - MARGIN - HEADER_H
            avail_w = cx1 - cx0
            avail_h = cy1 - cy0
            if avail_w <= 100 or avail_h <= 100:
                return None

            card_w_candidates: List[float] = []
            for _, n_cols, gap_units in sections_meta:
                gap_sum = sum(gap_units) * BASE_GUTTER_X
                card_w = (avail_w - gap_sum) / max(1, n_cols)
                card_w_candidates.append(card_w)
            card_w = min(card_w_candidates) if card_w_candidates else 80.0

            gutter_y = max(6.0, min(12.0, card_w * 0.10))

            total_rows = sum(r for r, _, _ in sections_meta)
            if total_rows <= 0:
                return None

            bars_total = SECTION_BAR_H * len(sections_meta)
            gaps_total = BETWEEN_SECTIONS_GAP * max(0, len(sections_meta) - 1)
            gutters_total = sum(max(0, r - 1) * gutter_y for r, _, _ in sections_meta)

            card_h = (avail_h - bars_total - gaps_total - gutters_total - (SECTION_BAR_GAP * len(sections_meta))) / total_rows
            card_h = min(card_h, card_w * 1.35)

            if card_w < 62.0 or card_h < 54.0:
                return None

            return pw, ph, card_w, card_h, gutter_y, BASE_GUTTER_X

        cand = eval_page(*LANDSCAPE)
        if cand is not None:
            return cand
        cand = eval_page(*PORTRAIT)
        if cand is not None:
            return cand
        pw, ph = LANDSCAPE
        return pw, ph, 62.0, 54.0, 6.0, 10.0

    def _draw_section_bar(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
        c.setFillColorRGB(*BAR_FILL)
        c.setStrokeColorRGB(*BAR_FILL)
        c.rect(x, y, w, h, stroke=0, fill=1)
        fs = _fit_font(label, BODY_BOLD_FONT, w - 20, h - 6, 10, 18, step=0.5)
        c.setFillColorRGB(*BAR_TEXT)
        c.setFont(BODY_BOLD_FONT, fs)
        tw = pdfmetrics.stringWidth(label, BODY_BOLD_FONT, fs)
        c.drawString(x + (w - tw) / 2, y + (h - fs) / 2 + 1, label)

    def _draw_card(
        c: canvas.Canvas,
        x: float,
        y: float,
        w: float,
        h: float,
        img: Optional[Image.Image],
        upc12: str,
        name: str,
        cpp: Optional[int],
    ) -> None:
        pad = max(3.0, min(6.0, w * 0.06))
        ix = x + pad
        iy = y + pad
        iw = w - 2 * pad
        ih = h - 2 * pad

        text_h = max(28.0, min(46.0, ih * 0.32))
        img_h = max(10.0, ih - text_h - 2.0)

        img_y = iy + text_h + 2.0

        if img is not None and iw > 6 and img_h > 6:
            sw, sh = img.size
            r = min(iw / max(1, sw), img_h / max(1, sh))
            dw, dh = sw * r, sh * r
            c.drawImage(
                ImageReader(img),
                ix + (iw - dw) / 2,
                img_y + (img_h - dh) / 2,
                dw,
                dh,
                preserveAspectRatio=True,
                mask="auto",
            )

        cpp_str = f"CPP: {cpp}" if cpp is not None else "CPP:"
        upc = (upc12 or "").strip()

        cpp_h = max(10.0, text_h * 0.30)
        name_h = max(9.0, text_h * 0.26)
        upc_h = max(10.0, text_h - cpp_h - name_h)

        upc_fs = _fit_font(upc, BODY_BOLD_FONT, iw, upc_h, 6.0, 14.0, step=0.25)
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.setFont(BODY_BOLD_FONT, upc_fs)
        tw = pdfmetrics.stringWidth(upc, BODY_BOLD_FONT, upc_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + name_h + (upc_h - upc_fs) / 2, upc)

        nm = _compact_one_line_name(name).upper()
        name_fs = 6.5
        if pdfmetrics.stringWidth(nm, BODY_FONT, name_fs) > iw:
            nm = _ellipsis(nm, BODY_FONT, name_fs, iw)
        c.setFillColorRGB(0.10, 0.10, 0.10)
        c.setFont(BODY_FONT, name_fs)
        tw = pdfmetrics.stringWidth(nm, BODY_FONT, name_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + (name_h - name_fs) / 2, nm)

        cpp_fs = _fit_font(cpp_str, BODY_BOLD_FONT, iw, cpp_h, 6.0, 12.0, step=0.25)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.setFont(BODY_BOLD_FONT, cpp_fs)
        tw = pdfmetrics.stringWidth(cpp_str, BODY_BOLD_FONT, cpp_fs)
        c.drawString(ix + (iw - tw) / 2, iy + (cpp_h - cpp_fs) / 2, cpp_str)

    try:
        if not pages:
            c = canvas.Canvas(buf, pagesize=LANDSCAPE)
            c.save()
            return buf.getvalue()

        first_sections = _split_sections(pages[0])
        meta_first: List[Tuple[int, int, List[int]]] = []
        for _, cells in first_sections:
            rows = _row_order(cells)
            cols, gap_units = _col_order_and_gaps(cells)
            meta_first.append((len(rows), len(cols), gap_units))
        PW, PH, CARD_W, CARD_H, GUTTER_Y, GUTTER_X = _choose_geometry(meta_first)

        c = canvas.Canvas(buf, pagesize=(PW, PH))

        for pdata in pages:
            _draw_full_pallet_header(
                c,
                PW,
                PH,
                HEADER_H,
                title_prefix.strip() or "POG",
                f"SIDE {pdata.side_letter}",
                logo,
            )
            _draw_footer(c, PW, MARGIN, FOOTER_H)

            cx0 = MARGIN
            cx1 = PW - MARGIN
            cy0 = MARGIN + FOOTER_H
            cy1 = PH - MARGIN - HEADER_H
            content_w = cx1 - cx0
            y_cursor = cy1

            sections = _split_sections(pdata)

            sec_render: List[Tuple[str, List[int], List[int], List[int], Dict[Tuple[int, int], CellData]]] = []
            for label, cells in sections:
                rows = _row_order(cells)
                cols, gap_units = _col_order_and_gaps(cells)
                rmap = {r: i for i, r in enumerate(rows)}
                cmap = {col: i for i, col in enumerate(cols)}
                occ: Dict[Tuple[int, int], CellData] = {}
                for cell in cells:
                    if cell.row in rmap and cell.col in cmap:
                        occ[(rmap[cell.row], cmap[cell.col])] = cell
                sec_render.append((label or "", rows, cols, gap_units, occ))

            total_rows = sum(len(rows) for _, rows, _, _, _ in sec_render)
            bars_total = SECTION_BAR_H * len(sec_render)
            gutters_total = sum(max(0, len(rows) - 1) * GUTTER_Y for _, rows, _, _, _ in sec_render)
            gaps_total = BETWEEN_SECTIONS_GAP * max(0, len(sec_render) - 1)
            bar_gaps_total = SECTION_BAR_GAP * len(sec_render)
            needed_h = total_rows * CARD_H + gutters_total + bars_total + gaps_total + bar_gaps_total

            extra = max(0.0, (cy1 - cy0) - needed_h)
            extra_per_block = extra / max(1, len(sec_render) * 2)

            for _, (label, rows, cols, gap_units, occ) in enumerate(sec_render):
                y_cursor -= extra_per_block

                y_cursor -= SECTION_BAR_H
                _draw_section_bar(c, cx0, y_cursor, content_w, SECTION_BAR_H, label)
                y_cursor -= SECTION_BAR_GAP

                n_cols = len(cols)
                if n_cols <= 0:
                    continue

                sx = cx0

                x_positions: List[float] = []
                x = sx
                for ci in range(n_cols):
                    x_positions.append(x)
                    x += CARD_W
                    if ci < n_cols - 1:
                        x += GUTTER_X * (gap_units[ci] if ci < len(gap_units) else 1)

                n_rows = len(rows)
                grid_h = n_rows * CARD_H + max(0, n_rows - 1) * GUTTER_Y
                grid_top = y_cursor
                for ri in range(n_rows):
                    y = grid_top - (ri + 1) * CARD_H - ri * GUTTER_Y
                    for ci in range(n_cols):
                        x = x_positions[ci]

                        cell = occ.get((ri, ci))
                        if cell is None:
                            c.setFillColorRGB(1, 1, 1)
                            c.setStrokeColorRGB(*EMPTY_STROKE)
                            c.setLineWidth(0.45)
                            c.rect(x, y, CARD_W, CARD_H, stroke=1, fill=0)
                            continue

                        match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                        upc12 = match.upc12 if match else None
                        cpp = match.cpp_qty if match else None
                        disp_name = (match.display_name if match and match.display_name else cell.name).strip()

                        upc_str = upc12 if upc12 else (f"???????{cell.last5}" if cell.last5 else "????????????")

                        c.setFillColorRGB(1, 1, 1)
                        c.setStrokeColorRGB(*FILLED_STROKE)
                        c.setLineWidth(0.75)
                        c.rect(x, y, CARD_W, CARD_H, stroke=1, fill=1)

                        img = crop_image_cell(images_doc, pdata.page_index, cell.bbox, zoom=2.6, inset=0.045)
                        _draw_card(c, x, y, CARD_W, CARD_H, img, upc_str, disp_name, cpp)

                y_cursor = grid_top - grid_h - BETWEEN_SECTIONS_GAP
                y_cursor -= extra_per_block

            c.showPage()

        c.save()
        return buf.getvalue()
    finally:
        images_doc.close()
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


def _draw_header(
    c: canvas.Canvas,
    page_w,
    page_h,
    top_bar_h,
    title_text,
    right_label,
    logo_img,
    grad_l,
    grad_r,
) -> None:
    hy = page_h - top_bar_h
    _draw_gradient(c, 0, hy, page_w, top_bar_h, grad_l, grad_r, steps=120)

    tx = 14.0
    if logo_img:
        lw, lh = logo_img.size
        th = top_bar_h * 0.62
        r = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(ImageReader(logo_img), 14, hy + (top_bar_h - dh) / 2, dw, dh, mask="auto")
        tx = 14 + dw + 10

    c.setFillColorRGB(1, 1, 1)
    fs = _fit_font(title_text, TITLE_FONT, page_w - tx - 130, top_bar_h * 0.72, 18, 36)
    c.setFont(TITLE_FONT, fs)
    c.drawString(tx, hy + (top_bar_h - fs) / 2 + 2, title_text)

    if right_label:
        rfs = _fit_font(right_label, TITLE_FONT, 120, top_bar_h * 0.68, 14, 28)
        rw = pdfmetrics.stringWidth(right_label, TITLE_FONT, rfs)
        c.setFont(TITLE_FONT, rfs)
        c.drawString(page_w - rw - 14, hy + (top_bar_h - rfs) / 2 + 2, right_label)

    c.setLineWidth(0.8)
    c.setStrokeColorRGB(0.88, 0.88, 0.92)
    c.line(0, hy, page_w, hy)


def _draw_cell_text_block(c: canvas.Canvas, x, y, w, h, name, upc12, last5, qty) -> None:
    px, py = 5, 4
    mw = max(12.0, w - px * 2)
    avail = max(10.0, h - py * 2)
    upc_str = upc12 if upc12 else (f"???????{last5}" if last5 else "")

    ns = 13.0
    while ns >= 5.0:
        ms = max(5.0, ns * 0.86)
        lhn = ns * 1.18
        lhm = ms * 1.18
        nls = wrap_text(name or "", mw, BODY_BOLD_FONT, ns)
        ul = f"UPC: {upc_str}" if upc_str else ""
        ql = f"Qty: {qty}" if qty is not None else ""
        need = (
            len(nls) * lhn
            + (2.5 if nls and (ul or ql) else 0)
            + (lhm if ul else 0)
            + (lhm if ql else 0)
        )
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
                c.drawString(x + px, max(y + py, ty), ul)
                ty -= lhm
            if ql:
                c.drawString(x + px, max(y + py, ty), ql)
            return
        ns -= 0.5

    fs = 5.0
    ty = y + h - py - fs
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, fs)
    for ln in wrap_text(name or "", mw, BODY_BOLD_FONT, fs):
        c.drawString(x + px, max(y + py, ty), ln)
        ty -= fs * 1.18
    c.setFont(BODY_FONT, fs)
    if upc_str:
        c.drawString(x + px, max(y + py, ty), f"UPC: {upc_str}")
        ty -= fs * 1.18
    if qty is not None:
        c.drawString(x + px, max(y + py, ty), f"Qty: {qty}")

def render_standard_pog_pdf(
    pages: List[PageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
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

    grad_left = _hex_to_rgb("#5B63A9")
    grad_right = _hex_to_rgb("#3E4577")
    logo_img = _try_load_logo()

    side_count = len(pages)
    per_side_w = int(310 * scale_factor)
    side_scales: List[float] = []
    side_heights: List[float] = []

    for page in pages:
        x_min = float(page.x_bounds[0])
        x_max = float(page.x_bounds[-1])
        y_min = float(page.y_bounds[0])
        y_max = float(page.y_bounds[-1])
        scale = per_side_w / max(1e-6, x_max - x_min)
        side_scales.append(scale)
        side_heights.append(scale * max(1e-6, y_max - y_min))

    content_h = max(side_heights) if side_heights else 600.0
    page_w = outer_margin * 2 + side_count * per_side_w + max(0, side_count - 1) * side_gap
    page_h = outer_margin + top_bar_h + side_label_h + content_h + footer_h + outer_margin
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    try:
        _draw_header(
            c,
            page_w,
            page_h,
            top_bar_h,
            title_prefix or "POG",
            "",
            logo_img,
            grad_left,
            grad_right,
        )

        cells_top = page_h - top_bar_h - side_label_h
        content_bottom = outer_margin + footer_h

        _draw_footer(c, page_w, outer_margin, footer_h)

        for side_idx, page in enumerate(pages):
            side_letter = chr(ord("A") + side_idx)
            side_origin_x = outer_margin + side_idx * (per_side_w + side_gap)

            badge_h = 34
            badge_w = 148
            badge_y = cells_top + (side_label_h - badge_h) / 2
            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(0.85, 0.85, 0.90)
            c.setLineWidth(0.85)
            c.roundRect(side_origin_x, badge_y, badge_w, badge_h, 8, stroke=1, fill=1)

            side_text = f"Side {side_letter}"
            side_font_size = _fit_font(side_text, TITLE_FONT, badge_w - 16, badge_h - 8, 14, 22)
            side_text_w = pdfmetrics.stringWidth(side_text, TITLE_FONT, side_font_size)
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(TITLE_FONT, side_font_size)
            c.drawString(
                side_origin_x + (badge_w - side_text_w) / 2,
                badge_y + (badge_h - side_font_size) / 2 + 1,
                side_text,
            )

            if side_idx > 0:
                sep_x = side_origin_x - side_gap / 2
                c.setLineWidth(0.6)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(sep_x, content_bottom + 2, sep_x, page_h - top_bar_h)

            x_min = float(page.x_bounds[0])
            y_min = float(page.y_bounds[0])
            scale = side_scales[side_idx]

            for cell in page.cells:
                match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                upc12 = match.upc12 if match else None
                qty = cell.qty if cell.qty is not None else (match.cpp_qty if match else None)

                x0, top, x1, bottom = cell.bbox
                out_x0 = side_origin_x + (x0 - x_min) * scale + cell_inset
                out_x1 = side_origin_x + (x1 - x_min) * scale - cell_inset
                out_top = cells_top - (top - y_min) * scale - cell_inset
                out_bottom = cells_top - (bottom - y_min) * scale + cell_inset
                out_w = out_x1 - out_x0
                out_h = out_top - out_bottom
                if out_w <= 2 or out_h <= 2:
                    continue

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.72, 0.72, 0.72)
                c.setLineWidth(border_w)
                c.rect(out_x0, out_bottom, out_w, out_h, stroke=1, fill=1)

                img_area_h = out_h * img_frac
                text_area_h = out_h - img_area_h
                img = crop_image_cell(images_doc, page.page_index, cell.bbox, zoom=3.2, inset=0.08)
                img_w, img_h = img.size
                img_scale = min(out_w * 0.86 / max(1, img_w), img_area_h * 0.84 / max(1, img_h))
                draw_w, draw_h = img_w * img_scale, img_h * img_scale
                c.drawImage(
                    ImageReader(img),
                    out_x0 + (out_w - draw_w) / 2,
                    out_bottom + text_area_h + (img_area_h - draw_h) / 2,
                    draw_w,
                    draw_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )

                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(out_x0 + 3, out_bottom + text_area_h, out_x1 - 3, out_bottom + text_area_h)

                _draw_cell_text_block(
                    c,
                    out_x0,
                    out_bottom,
                    out_w,
                    text_area_h,
                    cell.name,
                    upc12,
                    cell.last5,
                    qty,
                )

        c.showPage()
        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI (UNCHANGED except it will use the updated render_full_pallet_pdf)
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
        labels_pdf = st.file_uploader("Labels PDF", type=["pdf"])
        images_pdf = st.file_uploader("Images PDF", type=["pdf"])
        st.divider()
        title_prefix = st.text_input("PDF title prefix", "POG")
        out_name = st.text_input("Output filename", "pog_export.pdf")
        generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

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
                rows.append(
                    {
                        "Side": sl,
                        "Row": cell.row,
                        "Col": cell.col,
                        "Name": cell.name,
                        "Last5": cell.last5,
                        "Qty": cell.qty if cell.qty is not None else (match.cpp_qty if match else None),
                        "UPC12": match.upc12 if match else None,
                    }
                )
        st.dataframe(pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]), use_container_width=True, height=420)

        if generate:
            with st.spinner("Rendering PDF…"):
                pdf = render_standard_pog_pdf(  # type: ignore[name-defined]
                    pages,
                    images_bytes,
                    matrix_idx,
                    title_prefix.strip() or "POG",
                )
            st.success("Done.")
            st.download_button(
                "⬇ Download Planogram PDF",
                pdf,
                file_name=out_name if out_name.endswith(".pdf") else f"{out_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

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
                rows.append(
                    {
                        "Side": pg.side_letter,
                        "Row": cell.row,
                        "Col": cell.col,
                        "Name": cell.name,
                        "Last5": cell.last5,
                        "CPP (Excel)": match.cpp_qty if match else None,
                        "UPC12": match.upc12 if match else None,
                    }
                )
        st.dataframe(pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]), use_container_width=True, height=420)

        if generate:
            with st.spinner("Rendering PDF…"):
                pdf = render_full_pallet_pdf(
                    fp_pages,
                    images_bytes,
                    matrix_idx,
                    title_prefix.strip() or "POG",
                )
            st.success("Done.")
            st.download_button(
                "⬇ Download Planogram PDF",
                pdf,
                file_name=out_name if out_name.endswith(".pdf") else f"{out_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()