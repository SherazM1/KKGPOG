# home.py
"""
Streamlit Planogram Generator

- Standard Flat Display — all sides on one wide page (original behaviour)
- Full Pallet / Multi-Zone Display — template-style rebuild (one page per side)

IMPORTANT:
- Standard Flat Display section (UI branch + extraction + rendering) is preserved as-is.
- All new logic is gated behind the Full Pallet display type.

Full Pallet additions:
- Top cards from PPTX: 8 across + 6 right-block, show ID# (not UPC), CPP from one global input.
- Gift Card Holders from '2025 D82 POG.xlsx': show 6-digit Item # (not UPC) + description + Qty as CPP.
- Keeps existing UPC12 + CPP matching for the main grid + BONUS (your current logic).
"""

from __future__ import annotations

import difflib
import io
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import openpyxl
import pdfplumber
import streamlit as st
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

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
class PageData:
    # standard display
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


@dataclass(frozen=True)
class PptCard:
    side: str  # A-D
    card_id: str  # e.g., "19"
    title: str  # first line title


@dataclass(frozen=True)
class PptSideCards:
    side: str
    top8: List[PptCard]   # 8 cards across the top
    side6: List[PptCard]  # 6 cards in the right block


@dataclass(frozen=True)
class HolderItem:
    item_no: str  # 6-digit item number
    description: str
    qty: Optional[int]  # Qty in the workbook (treat as CPP)


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
    except Exception:
        return None


def _hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        return (0.5, 0.5, 0.5)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def _fit_font(
    text: str,
    font_name: str,
    max_w: float,
    max_h: float,
    min_size: float,
    max_size: float,
    step: float = 0.5,
) -> float:
    t = (text or "").strip()
    if not t:
        return min_size
    fs = max_size
    while fs >= min_size:
        w = pdfmetrics.stringWidth(t, font_name, fs)
        if w <= max_w and fs <= max_h:
            return fs
        fs -= step
    return min_size


def _fit_name_preserve_qualifiers(name: str, font: str, fs: float, max_w: float) -> str:
    nm = (name or "").strip()
    if not nm:
        return ""
    if pdfmetrics.stringWidth(nm, font, fs) <= max_w:
        return nm

    words = nm.split()
    while words and pdfmetrics.stringWidth(" ".join(words), font, fs) > max_w:
        words.pop()
    base = " ".join(words).strip()
    if not base:
        base = nm[: max(6, int(max_w / (fs * 0.55)))]
    return base


def _try_load_logo() -> Optional[Image.Image]:
    for rel in ["logo.png", "assets/logo.png"]:
        p = Path(__file__).resolve().parent / rel
        if p.is_file():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Matrix loading (shared)
# ──────────────────────────────────────────────────────────────────────────────


def _find_header_row(df_raw: pd.DataFrame) -> int:
    for i in range(min(20, len(df_raw))):
        row = df_raw.iloc[i].tolist()
        s = " ".join([str(v).upper() for v in row if v is not None])
        if "UPC" in s and ("NAME" in s or "DESCRIPTION" in s):
            return i
    return 0


def _norm_header(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s.upper()


def _pick_col(headers: List[str], candidates: List[str], fallback_idx: int) -> str:
    for cand in candidates:
        for h in headers:
            if cand.upper() == h.upper():
                return h
    return headers[fallback_idx]


def _pick_col_optional(headers: List[str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        for h in headers:
            if cand.upper() == h.upper():
                return h
    return None


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

    df = df_raw.iloc[hrow + 1 :].copy()
    df.columns = headers

    upc_col = _pick_col(headers, ["UPC"], 0)
    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"], 1 if len(headers) > 1 else 0)
    cpp_col = _pick_col_optional(headers, ["CPP"])

    df["__upc12"] = df[upc_col].map(_coerce_upc12)
    df["__name"] = df[name_col].astype(str).fillna("")

    if cpp_col and cpp_col in df.columns:
        df["__cpp"] = df[cpp_col].map(_coerce_int)
    else:
        df["__cpp"] = None

    df = df[df["__upc12"].notna()].copy()
    df["__last5"] = df["__upc12"].str[-5:]
    df["__norm"] = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["__last5"])
        cpp_val = r.get("__cpp")
        cpp = None if pd.isna(cpp_val) else int(cpp_val) if cpp_val is not None else None
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
# Full-pallet additional inputs (PowerPoint + Holder Excel)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def parse_ppt_side_cards(pptx_bytes: bytes) -> Dict[str, PptSideCards]:
    """
    Parse PowerPoint top-area blueprint.

    Expected:
    - One slide per side labeled "SIDE A" / "SIDE B" / ...
    - Each card entry has a line containing "ID #<n>".
    - Order logic:
      * top8: the 8 cards with the smallest 'top' coordinate, ordered left->right.
      * side6: next 6 cards, ordered top->bottom then left->right.
    """
    # Lazy import so flat display never depends on python-pptx at import time.
    from pptx import Presentation  # type: ignore

    prs = Presentation(io.BytesIO(pptx_bytes))

    by_side: Dict[str, List[Tuple[int, int, str]]] = {}

    for slide in prs.slides:
        side_letter: Optional[str] = None
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            t = (shape.text or "").strip()
            m_side = re.search(r"\bSIDE\s+([A-D])\b", t.upper())
            if m_side:
                side_letter = m_side.group(1)
                break

        if not side_letter:
            continue

        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            txt = (shape.text or "").strip()
            if not txt or "ID #" not in txt:
                continue
            m_id = re.search(r"ID\s*#\s*(\d+)", txt, re.IGNORECASE)
            if not m_id:
                continue
            title = txt.splitlines()[0].strip()
            card_id = m_id.group(1).strip()
            by_side.setdefault(side_letter, []).append((int(shape.top), int(shape.left), f"{card_id}|{title}"))

    out: Dict[str, PptSideCards] = {}
    for side, entries in by_side.items():
        uniq: Dict[str, Tuple[int, int, str]] = {}
        for top, left, payload in entries:
            uniq[payload] = (top, left, payload)
        items = list(uniq.values())

        items_sorted_by_top = sorted(items, key=lambda t: (t[0], t[1]))
        top8_raw = items_sorted_by_top[:8]
        side6_raw = items_sorted_by_top[8:14]

        top8 = []
        for top, left, payload in sorted(top8_raw, key=lambda t: t[1]):
            card_id, title = payload.split("|", 1)
            top8.append(PptCard(side=side, card_id=card_id, title=title))

        side6 = []
        for top, left, payload in sorted(side6_raw, key=lambda t: (t[0], t[1])):
            card_id, title = payload.split("|", 1)
            side6.append(PptCard(side=side, card_id=card_id, title=title))

        out[side] = PptSideCards(side=side, top8=top8, side6=side6)

    return out


@st.cache_data(show_spinner=False)
def load_holder_items(holder_xlsx_bytes: bytes) -> List[HolderItem]:
    """
    Load gift card holder items from the '2025 D82 POG.xlsx' workbook.

    We treat:
    - Item # (6-digit) as the identifier to display (NOT UPC).
    - Qty as CPP.
    - Description comes from the workbook's item description table.
    """
    wb = openpyxl.load_workbook(io.BytesIO(holder_xlsx_bytes), data_only=True)

    if "FULL PALLET" not in wb.sheetnames:
        raise ValueError("Holder workbook is missing a 'FULL PALLET' sheet.")

    ws = wb["FULL PALLET"]

    header_row = None
    col_item = None
    col_desc = None
    col_qty = None

    for r in range(1, min(60, ws.max_row) + 1):
        row_vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        row_norm = [str(v).strip().upper() if isinstance(v, str) else v for v in row_vals]
        if any(isinstance(v, str) and v.strip().upper() == "ITEM #" for v in row_norm) and any(
            isinstance(v, str) and "ITEM DESCRIPTION" in v.strip().upper() for v in row_norm
        ):
            header_row = r
            for c, v in enumerate(row_norm, start=1):
                if isinstance(v, str) and v.strip().upper() == "ITEM #":
                    col_item = c
                if isinstance(v, str) and "ITEM DESCRIPTION" in v.strip().upper():
                    col_desc = c
                if isinstance(v, str) and "QTY PER HOOK" in v.strip().upper():
                    col_qty = c
            break

    if not header_row or not col_item or not col_desc:
        raise ValueError("Could not find the holder item description table in the FULL PALLET sheet.")

    catalog: Dict[str, Tuple[str, Optional[int]]] = {}
    for r in range(header_row + 1, ws.max_row + 1):
        item = ws.cell(r, col_item).value
        desc = ws.cell(r, col_desc).value
        qty_val = ws.cell(r, col_qty).value if col_qty else None
        if item is None:
            continue
        item_no = str(int(item)) if isinstance(item, (int, float)) and not math.isnan(float(item)) else str(item).strip()
        item_digits = re.sub(r"\D", "", item_no)
        if not re.fullmatch(r"\d{6}", item_digits):
            if catalog:
                break
            continue
        item_no = item_digits.zfill(6)
        description = str(desc or "").strip()
        qty = _coerce_int(qty_val)
        catalog[item_no] = (description, qty)

    layout_item_row = None
    layout_qty_row = None
    for r in range(1, min(25, ws.max_row) + 1):
        v = ws.cell(r, 1).value
        if isinstance(v, str) and v.strip().upper() == "ITEM #":
            layout_item_row = r
        if isinstance(v, str) and v.strip().upper() == "QTY":
            layout_qty_row = r

    used_order: List[str] = []
    if layout_item_row:
        for c in range(2, ws.max_column + 1):
            item = ws.cell(layout_item_row, c).value
            if item is None:
                continue
            item_no = str(int(item)) if isinstance(item, (int, float)) and not math.isnan(float(item)) else str(item).strip()
            item_no = re.sub(r"\D", "", item_no)
            if not item_no:
                continue
            item_no = item_no.zfill(6)
            if item_no not in used_order:
                used_order.append(item_no)

    items: List[HolderItem] = []
    for item_no in used_order:
        desc, qty = catalog.get(item_no, ("", None))
        if qty is None and layout_qty_row and layout_item_row:
            for c in range(2, ws.max_column + 1):
                v_item = ws.cell(layout_item_row, c).value
                if v_item is None:
                    continue
                v_item_no = re.sub(r"\D", "", str(v_item)).zfill(6)
                if v_item_no != item_no:
                    continue
                qty = _coerce_int(ws.cell(layout_qty_row, c).value)
                if qty is not None:
                    break
        items.append(HolderItem(item_no=item_no, description=desc or item_no, qty=qty))

    return items


# ──────────────────────────────────────────────────────────────────────────────
# Clustering helpers (shared)
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
    if len(centers) == 0:
        return np.array([], dtype=float)
    if len(centers) == 1:
        c = float(centers[0])
        return np.array([c - 1.0, c + 1.0], dtype=float)

    mids = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def parse_label_cell_text(txt: str) -> Tuple[str, str, Optional[int]]:
    lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
    name_parts: List[str] = []
    last5 = ""
    qty: Optional[int] = None

    for ln in lines:
        m = LAST5_RE.search(ln)
        if m:
            last5 = m.group(1)
            ln = LAST5_RE.sub("", ln).strip()

        q = re.search(r"\bQTY[:\s]*([0-9]+)\b", ln, re.IGNORECASE)
        if q:
            qty = _coerce_int(q.group(1))
            ln = re.sub(r"\bQTY[:\s]*[0-9]+\b", "", ln, flags=re.IGNORECASE).strip()

        if ln:
            name_parts.append(ln)

    name = " ".join(name_parts).strip()
    return name, last5, qty


def _union(words: List[dict], px: float = 0.0, py: float = 0.0) -> Tuple[float, float, float, float]:
    x0 = float(min(w["x0"] for w in words)) - px
    x1 = float(max(w["x1"] for w in words)) + px
    t = float(min(w["top"] for w in words)) - py
    b = float(max(w["bottom"] for w in words)) + py
    return x0, t, x1, b


def _group_nearby(words: List[dict], x_tol: float, y_tol: float) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for w in sorted(words, key=lambda d: (float(d["top"]), float(d["x0"]))):
        placed = False
        for g in groups:
            gx0, gt, gx1, _ = _union(g)
            if (abs(float(w["top"]) - gt) <= y_tol) and (abs(float(w["x0"]) - gx1) <= x_tol):
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
            _, bt, bx1, _ = _union(base)
            i = 0
            while i < len(groups):
                gx0, gt, _, _ = _union(groups[i])
                if abs(gt - bt) <= y_tol and (gx0 - bx1) <= x_tol:
                    base.extend(groups.pop(i))
                    _, bt, bx1, _ = _union(base)
                    changed = True
                else:
                    i += 1
            merged.append(base)
        groups = merged

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Full-pallet extraction (labels) + annotations
# ──────────────────────────────────────────────────────────────────────────────


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

            x_centers = cluster_positions(xs, tol=max(10, pw * 0.025))
            y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))
            if len(x_centers) == 0 or len(y_centers) == 0:
                continue

            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)
            if len(x_bounds) < 2 or len(y_bounds) < 2:
                continue

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col = int(np.argmin(np.abs(x_centers - xc)))
                row = int(np.argmin(np.abs(y_centers - yc)))
                dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                key = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_, token_w) in sorted(cell_map.items()):
                bbox = (
                    float(x_bounds[col]),
                    float(y_bounds[row]),
                    float(x_bounds[col + 1]),
                    float(y_bounds[row + 1]),
                )
                txt = (page.crop(bbox).extract_text() or "").strip()
                parsed_name, parsed_last5, qty = parse_label_cell_text(txt)
                token_last5 = str(token_w.get("text", "")).strip()
                last5 = token_last5 if re.fullmatch(r"\d{5}", token_last5) else parsed_last5

                cells.append(
                    CellData(
                        row=row,
                        col=col,
                        bbox=bbox,
                        name=parsed_name,
                        last5=last5 or "",
                        qty=qty,
                        upc12=None,
                    )
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
                if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "IN", "NEW", "PKG", "D"}
                and float(w["top"]) < ph * 0.30
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
# Image cropping helper (used by Full Pallet)
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

    w = float(page.rect.width)
    h = float(page.rect.height)

    pad_x = (x1 - x0) * inset
    pad_y = (bottom - top) * inset

    rx0 = max(0.0, x0 + pad_x)
    rx1 = min(w, x1 - pad_x)
    rt = max(0.0, top + pad_y)
    rb = min(h, bottom - pad_y)

    rect = fitz.Rect(rx0, rt, rx1, rb)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Header / footer (shared)
# ──────────────────────────────────────────────────────────────────────────────


def _draw_footer(c: canvas.Canvas, page_w: float, outer_margin: float, footer_h: float) -> None:
    c.setFillColorRGB(0.98, 0.98, 0.99)
    c.setStrokeColorRGB(0.90, 0.90, 0.92)
    c.setLineWidth(0.8)
    c.rect(outer_margin, outer_margin, page_w - outer_margin * 2, footer_h, stroke=1, fill=1)

    stamp = f"Generated: {date.today().isoformat()}"
    c.setFillColorRGB(0.25, 0.25, 0.30)
    c.setFont(BODY_FONT, 9)
    c.drawString(outer_margin + 12, outer_margin + (footer_h - 9) / 2, stamp)


def _draw_full_pallet_header(
    c: canvas.Canvas,
    pw: float,
    ph: float,
    header_h: float,
    title_prefix: str,
    side_label: str,
    logo: Optional[Image.Image],
) -> None:
    c.setFillColorRGB(0.98, 0.98, 1.0)
    c.setStrokeColorRGB(0.88, 0.88, 0.92)
    c.setLineWidth(0.8)
    c.rect(0, ph - header_h, pw, header_h, stroke=1, fill=1)

    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(TITLE_FONT, 18)
    c.drawString(24, ph - header_h + 18, f"{title_prefix}: {side_label}")

    if logo is not None:
        try:
            sw, sh = logo.size
            max_h = header_h - 18
            r = min(1.0, max_h / max(1, sh))
            dw, dh = sw * r, sh * r
            c.drawImage(ImageReader(logo), pw - dw - 24, ph - header_h + (header_h - dh) / 2, dw, dh, mask="auto")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Full Pallet renderer (new logic)
# ──────────────────────────────────────────────────────────────────────────────


def render_full_pallet_pdf(
    pages: List[FullPalletPage],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    ppt_cards_by_side: Dict[str, PptSideCards],
    holder_items: List[HolderItem],
    ppt_cpp_global: Optional[int],
    title_prefix: str = "POG",
) -> bytes:
    """
    Full Pallet render:
    - Page size fixed to 11x17 portrait (792x1224) to match completed planogram.
    - Renders:
      * Top cards (PPT): ID# + title + CPP global.
      * Gift Card Holders: Item # + description + Qty as CPP.
      * Main grid + BONUS: uses existing UPC12/CPP logic (last5 → matrix).
    """
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    PW, PH = (792.0, 1224.0)
    MARGIN = 28.0
    HEADER_H = 56.0
    FOOTER_H = 38.0

    SECTION_BAR_H = 26.0
    SECTION_BAR_GAP = 10.0
    BETWEEN_SECTIONS_GAP = 16.0

    TOP_GUTTER_X = 10.0
    TOP_GUTTER_Y = 10.0

    BAR_FILL = _hex_to_rgb("#77B5F0")
    BAR_TEXT = NAVY_RGB

    EMPTY_STROKE = (0.88, 0.88, 0.88)
    FILLED_STROKE = (0.78, 0.78, 0.80)

    logo = _try_load_logo()

    def _cell_center(cell: CellData) -> Tuple[float, float]:
        x0, t, x1, b = cell.bbox
        return (x0 + x1) / 2, (t + b) / 2

    def _global_row_order(p: FullPalletPage) -> List[int]:
        row_to_y: Dict[int, List[float]] = {}
        for cl in p.cells:
            _, y = _cell_center(cl)
            row_to_y.setdefault(cl.row, []).append(y)
        return sorted(row_to_y.keys(), key=lambda r: float(np.mean(row_to_y[r])))

    def _global_col_order_and_gaps(p: FullPalletPage) -> Tuple[List[int], List[int]]:
        col_to_x: Dict[int, List[float]] = {}
        for cl in p.cells:
            x, _ = _cell_center(cl)
            col_to_x.setdefault(cl.col, []).append(x)

        cols = sorted(col_to_x.keys(), key=lambda c_: float(np.mean(col_to_x[c_]))) if col_to_x else []
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

    def _split_rows_by_bonus(p: FullPalletPage, global_rows: List[int]) -> Tuple[List[int], List[int]]:
        bonus_y = None
        for ann in p.annotations:
            if ann.kind == "bonus_strip":
                _, t, _, b = ann.bbox
                bonus_y = (t + b) / 2
                break

        row_to_y: Dict[int, float] = {}
        for r in global_rows:
            ys = [(_cell_center(c)[1]) for c in p.cells if c.row == r]
            row_to_y[r] = float(np.mean(ys)) if ys else float("inf")

        yc = [row_to_y[r] for r in global_rows]

        if bonus_y is not None and len(yc) >= 2:
            boundary = 0
            for i in range(len(yc) - 1):
                if yc[i] < bonus_y < yc[i + 1]:
                    boundary = i
                    break
            return global_rows[: boundary + 1], global_rows[boundary + 1 :]

        if len(yc) < 2:
            return global_rows, []

        gaps = [yc[i + 1] - yc[i] for i in range(len(yc) - 1)]
        i_max = int(np.argmax(gaps))
        if gaps[i_max] < float(np.median(gaps)) * 1.6:
            return global_rows, []
        return global_rows[: i_max + 1], global_rows[i_max + 1 :]

    def _draw_section_bar(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
        c.setFillColorRGB(*BAR_FILL)
        c.setStrokeColorRGB(*BAR_FILL)
        c.rect(x, y, w, h, stroke=0, fill=1)

        fs = _fit_font(label, BODY_BOLD_FONT, w - 20, h - 6, 10, 18, step=0.5)
        c.setFillColorRGB(*BAR_TEXT)
        c.setFont(BODY_BOLD_FONT, fs)
        tw = pdfmetrics.stringWidth(label, BODY_BOLD_FONT, fs)
        c.drawString(x + (w - tw) / 2, y + (h - fs) / 2 + 1, label)

    def _draw_card_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, filled: bool) -> None:
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(*(FILLED_STROKE if filled else EMPTY_STROKE))
        c.setLineWidth(0.75 if filled else 0.45)
        c.rect(x, y, w, h, stroke=1, fill=1 if filled else 0)

    def _draw_card(
        c: canvas.Canvas,
        x: float,
        y: float,
        w: float,
        h: float,
        img: Optional[Image.Image],
        top_id: str,   # UPC12 for grid OR "ID #xx" OR "109113"
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
        top_line = (top_id or "").strip()
        nm = (name or "").strip()

        cpp_h = max(10.0, text_h * 0.30)
        name_h = max(9.0, text_h * 0.26)
        upc_h = max(10.0, text_h - cpp_h - name_h)

        upc_fs = _fit_font(top_line, BODY_BOLD_FONT, iw, upc_h, 6.0, 14.0, step=0.25)
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.setFont(BODY_BOLD_FONT, upc_fs)
        tw = pdfmetrics.stringWidth(top_line, BODY_BOLD_FONT, upc_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + name_h + (upc_h - upc_fs) / 2, top_line)

        name_fs = 6.5
        nm = _fit_name_preserve_qualifiers(nm.upper(), BODY_FONT, name_fs, iw)
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
            c = canvas.Canvas(buf, pagesize=(PW, PH))
            c.save()
            return buf.getvalue()

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

            # Zone 1: PPT cards (ID# + CPP global)
            ppt_side = ppt_cards_by_side.get(pdata.side_letter)
            if ppt_side:
                top_cols = 8
                side_cols = 3
                side_rows = 2

                top_w = (content_w - (top_cols - 1) * TOP_GUTTER_X) / top_cols
                top_h = min(top_w * 1.25, 96.0)

                y_top = y_cursor - top_h
                for i, card in enumerate(ppt_side.top8[:top_cols]):
                    x = cx0 + i * (top_w + TOP_GUTTER_X)
                    _draw_card_box(c, x, y_top, top_w, top_h, filled=True)
                    _draw_card(c, x, y_top, top_w, top_h, None, f"ID #{card.card_id}", card.title, ppt_cpp_global)

                x_side0 = cx0 + (top_cols - side_cols) * (top_w + TOP_GUTTER_X)
                y_side_top = y_top - TOP_GUTTER_Y
                for j in range(side_rows):
                    for i in range(side_cols):
                        idx = j * side_cols + i
                        x = x_side0 + i * (top_w + TOP_GUTTER_X)
                        y = y_side_top - (j + 1) * top_h - j * TOP_GUTTER_Y
                        if idx >= len(ppt_side.side6):
                            _draw_card_box(c, x, y, top_w, top_h, filled=False)
                            continue
                        card = ppt_side.side6[idx]
                        _draw_card_box(c, x, y, top_w, top_h, filled=True)
                        _draw_card(c, x, y, top_w, top_h, None, f"ID #{card.card_id}", card.title, ppt_cpp_global)

                zone_h = top_h + TOP_GUTTER_Y + side_rows * top_h + (side_rows - 1) * TOP_GUTTER_Y
                y_cursor = y_cursor - zone_h - BETWEEN_SECTIONS_GAP

            # Zone 2: Gift Card Holders (Item # + Qty as CPP)
            if holder_items:
                y_cursor -= SECTION_BAR_H
                _draw_section_bar(c, cx0, y_cursor, content_w, SECTION_BAR_H, "GIFT CARD HOLDERS")
                y_cursor -= SECTION_BAR_GAP

                cols = min(5, max(1, len(holder_items)))
                rows = int(math.ceil(len(holder_items) / cols)) if cols else 0
                holder_w = (content_w - (cols - 1) * TOP_GUTTER_X) / cols if cols else content_w
                holder_h = min(holder_w * 1.20, 82.0)
                grid_h = rows * holder_h + max(0, rows - 1) * TOP_GUTTER_Y

                grid_top = y_cursor
                for r in range(rows):
                    y = grid_top - (r + 1) * holder_h - r * TOP_GUTTER_Y
                    for col in range(cols):
                        idx = r * cols + col
                        x = cx0 + col * (holder_w + TOP_GUTTER_X)
                        if idx >= len(holder_items):
                            _draw_card_box(c, x, y, holder_w, holder_h, filled=False)
                            continue
                        it = holder_items[idx]
                        _draw_card_box(c, x, y, holder_w, holder_h, filled=True)
                        _draw_card(c, x, y, holder_w, holder_h, None, it.item_no, it.description, it.qty)

                y_cursor = grid_top - grid_h - BETWEEN_SECTIONS_GAP

            # Zones 3/4: existing matching logic for main grid + BONUS
            global_cols, gap_units = _global_col_order_and_gaps(pdata)
            global_rows = _global_row_order(pdata)
            pre_rows, bonus_rows = _split_rows_by_bonus(pdata, global_rows)

            n_cols = len(global_cols)
            if n_cols <= 0:
                c.showPage()
                continue

            gutter_x = 10.0
            gap_sum = sum(gap_units) * gutter_x
            card_w = (content_w - gap_sum) / n_cols

            x_positions: List[float] = []
            x = cx0
            for ci in range(n_cols):
                x_positions.append(x)
                x += card_w
                if ci < n_cols - 1:
                    x += gutter_x * (gap_units[ci] if ci < len(gap_units) else 1)

            remaining_h = max(0.0, y_cursor - cy0)
            total_rows = len(pre_rows) + len(bonus_rows)
            gutter_y = max(6.0, min(12.0, card_w * 0.10))

            reserve = 0.0
            if bonus_rows:
                reserve += SECTION_BAR_H + SECTION_BAR_GAP + BETWEEN_SECTIONS_GAP
            if total_rows <= 0 or remaining_h <= 50:
                c.showPage()
                continue

            available_for_cards = max(10.0, remaining_h - reserve - max(0, total_rows - 1) * gutter_y)
            card_h = min(card_w * 1.35, available_for_cards / total_rows)

            def _render_grid(rows_: List[int]) -> None:
                nonlocal y_cursor
                if not rows_:
                    return

                rmap = {r: i for i, r in enumerate(rows_)}
                cmap = {c_: i for i, c_ in enumerate(global_cols)}
                row_set = set(rows_)

                occ: Dict[Tuple[int, int], CellData] = {}
                for cell in pdata.cells:
                    if cell.row in row_set and cell.col in cmap:
                        occ[(rmap[cell.row], cmap[cell.col])] = cell

                n_rows = len(rows_)
                grid_h = n_rows * card_h + max(0, n_rows - 1) * gutter_y
                grid_top = y_cursor

                for ri in range(n_rows):
                    y = grid_top - (ri + 1) * card_h - ri * gutter_y
                    for ci in range(n_cols):
                        x = x_positions[ci]
                        cell = occ.get((ri, ci))
                        if cell is None:
                            _draw_card_box(c, x, y, card_w, card_h, filled=False)
                            continue

                        match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                        upc12 = match.upc12 if match else None
                        cpp = match.cpp_qty if match else None
                        disp_name = (match.display_name if match and match.display_name else cell.name).strip()
                        upc_str = upc12 if upc12 else f"???????{cell.last5}"

                        _draw_card_box(c, x, y, card_w, card_h, filled=True)
                        img = crop_image_cell(images_doc, pdata.page_index, cell.bbox, zoom=2.6, inset=0.045)
                        _draw_card(c, x, y, card_w, card_h, img, upc_str, disp_name, cpp)

                y_cursor = grid_top - grid_h - BETWEEN_SECTIONS_GAP

            _render_grid(pre_rows)

            if bonus_rows:
                y_cursor -= SECTION_BAR_H
                _draw_section_bar(c, cx0, y_cursor, content_w, SECTION_BAR_H, "BONUS")
                y_cursor -= SECTION_BAR_GAP
                _render_grid(bonus_rows)

            c.showPage()

        c.save()
        return buf.getvalue()
    finally:
        images_doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# Standard Flat Display renderer (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────


def render_standard_pog_pdf(
    pages: List[PageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
) -> bytes:
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    scale_factor = 1.5
    outer_margin = 44.0
    side_gap = 28.0
    top_bar_h = 90.0
    footer_h = 44.0
    side_label_h = 56.0
    cell_inset = 5.0
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

            badge_h = 34.0
            badge_w = 148.0
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
                c.setStrokeColorRGB(0.80, 0.80, 0.83)
                c.setLineWidth(border_w)
                c.rect(out_x0, out_bottom, out_w, out_h, stroke=1, fill=1)

                if upc12:
                    c.setFillColorRGB(0.08, 0.08, 0.10)
                    c.setFont(BODY_BOLD_FONT, 9)
                    c.drawString(out_x0 + 4, out_bottom + 4, upc12)

    finally:
        images_doc.close()

    c.save()
    return buf.getvalue()


def _draw_header(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    top_bar_h: float,
    title_prefix: str,
    subtitle: str,
    logo: Optional[Image.Image],
    grad_left: Tuple[float, float, float],
    grad_right: Tuple[float, float, float],
) -> None:
    c.setFillColorRGB(*grad_left)
    c.rect(0, page_h - top_bar_h, page_w, top_bar_h, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(TITLE_FONT, 22)
    c.drawString(44, page_h - top_bar_h + 30, title_prefix)
    if subtitle:
        c.setFont(BODY_FONT, 12)
        c.drawString(44, page_h - top_bar_h + 12, subtitle)
    if logo is not None:
        try:
            sw, sh = logo.size
            max_h = top_bar_h - 22
            r = min(1.0, max_h / max(1, sh))
            dw, dh = sw * r, sh * r
            c.drawImage(ImageReader(logo), page_w - dw - 44, page_h - top_bar_h + (top_bar_h - dh) / 2, dw, dh, mask="auto")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Standard Flat Display extraction (UNCHANGED)
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
            x_centers = kmeans_1d(xs, k=n_cols)
            y_centers = cluster_positions(ys, tol=max(8, float(page.height) * 0.015))

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
            for (row, col), (_, token_w) in sorted(cell_map.items()):
                bbox = (
                    float(x_bounds[col]),
                    float(y_bounds[row]),
                    float(x_bounds[col + 1]),
                    float(y_bounds[row + 1]),
                )
                txt = (page.crop(bbox).extract_text() or "").strip()
                parsed_name, parsed_last5, qty = parse_label_cell_text(txt)
                token_last5 = str(token_w.get("text", "")).strip()
                last5 = token_last5 if re.fullmatch(r"\d{5}", token_last5) else parsed_last5

                cells.append(
                    CellData(
                        row=row,
                        col=col,
                        bbox=bbox,
                        name=parsed_name,
                        last5=last5 or "",
                        qty=qty,
                        upc12=None,
                    )
                )

            pages.append(PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells))

    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Main UI (Standard Flat Display branch preserved)
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

        # New inputs (Full Pallet only)
        pptx_file = (
            st.file_uploader("Top Cards PowerPoint (.pptx)", type=["pptx"])
            if display_type == DISPLAY_FULL_PALLET
            else None
        )
        holder_pog_file = (
            st.file_uploader("Gift Card Holders POG (.xlsx)", type=["xlsx"], key="holder_pog")
            if display_type == DISPLAY_FULL_PALLET
            else None
        )
        ppt_cpp_global = (
            st.number_input(
                "PPT CPP (applies to all PPT cards)",
                min_value=0,
                max_value=999,
                value=30,
                step=1,
            )
            if display_type == DISPLAY_FULL_PALLET
            else None
        )

        st.divider()
        title_prefix = st.text_input("PDF title prefix", "POG")
        out_name = st.text_input("Output filename", "pog_export.pdf")
        generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    # Requirements per mode (Standard unchanged)
    if display_type == DISPLAY_FULL_PALLET:
        if not (matrix_file and labels_pdf and images_pdf and pptx_file and holder_pog_file):
            st.info(
                "Upload Matrix XLSX + Labels PDF + Images PDF + Top Cards PPTX + Gift Card Holders POG XLSX to begin."
            )
            return
    else:
        if not (matrix_file and labels_pdf and images_pdf):
            st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
            return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

    # Standard Flat Display (unchanged path)
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
                pdf = render_standard_pog_pdf(
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
        return

    # Full Pallet / Multi-Zone Display (new path)
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
            ppt_cards = parse_ppt_side_cards(pptx_file.getvalue())
            holder_items = load_holder_items(holder_pog_file.getvalue())

            pdf = render_full_pallet_pdf(
                fp_pages,
                images_bytes,
                matrix_idx,
                ppt_cards,
                holder_items,
                int(ppt_cpp_global) if ppt_cpp_global is not None else None,
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