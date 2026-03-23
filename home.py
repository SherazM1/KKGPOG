"""
Streamlit Planogram Generator

- Standard Flat Display — all sides on one wide page (original behaviour)
- Full Pallet Display — template-style rebuild (NO raster background), one page per side
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
    card_id: str
    title: str
    image_bytes: Optional[bytes] = None
    image_ext: Optional[str] = None


@dataclass(frozen=True)
class PptSideCards:
    side: str
    top8: List[PptCard]
    side6: List[PptCard]


@dataclass(frozen=True)
class GiftHolder:
    side: str
    item_no: str
    name: str
    qty: Optional[int]
    image_bytes: Optional[bytes] = None
    image_ext: Optional[str] = None


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


def _to_last5(v: object) -> str:
    s = DIGITS_RE.sub("", str(v or "").strip())
    if not s:
        return ""
    if len(s) >= 5:
        return s[-5:]
    return s.zfill(5)


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


def _pick_col_optional(cols: List[str], tokens: List[str]) -> Optional[str]:
    uc = [c.upper() for c in cols]
    for t in tokens:
        for i, c in enumerate(uc):
            if t in c:
                return cols[i]
    return None


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


@st.cache_data(show_spinner=False)
def load_full_pallet_matrix_index(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    """Full-pallet matrix index.

    Index key  = last 5 digits of the 11-digit UPC column (column header "UPC").
    Stored upc12 = 11-digit UPC zero-padded to 12 chars.

    The labels PDF encodes each product as the last 5 digits of its 11-digit UPC,
    so indexing on that column is required for correct resolution.  The zero-padded
    11-digit value matches what the reference planogram displays as the product UPC.
    """
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

    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"], 1 if len(headers) > 1 else 0)
    cpp_col = _pick_col_optional(headers, ["CPP"])

    # ── FIX A: use ONLY the 11-digit UPC column as the index key ─────────────
    # The labels PDF encodes items as the last 5 digits of the 11-digit UPC.
    # Using the 12-digit UPC column would give a different last-5 (check digit
    # appended) and break all lookups.  Prefer the exact header "UPC"; fall back
    # to the first column whose normalised name contains "UPC".
    upc11_col: Optional[str] = None
    for h in headers:
        if h.upper() == "UPC":
            upc11_col = h
            break
    if upc11_col is None:
        upc11_col = _pick_col(headers, ["UPC"], 0)

    df["__name"] = df[name_col].astype(str).fillna("")
    if cpp_col and cpp_col in df.columns:
        df["__cpp"] = df[cpp_col].map(_coerce_int)
    else:
        df["__cpp"] = None
    df["__norm"] = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    seen_pairs: set = set()
    for _, r in df.iterrows():
        display_name = str(r["__name"]).strip()
        cpp_val = r.get("__cpp")
        cpp = None if pd.isna(cpp_val) else int(cpp_val) if cpp_val is not None else None

        raw_upc11 = r.get(upc11_col)
        if raw_upc11 is None:
            continue
        upc11_str = re.sub(r"\.0$", "", str(raw_upc11).strip())
        digits11 = re.sub(r"[^0-9]", "", upc11_str)
        if len(digits11) < 5:
            continue
        last5 = digits11[-5:]
        # Store as 11-digit UPC zero-padded to 12 chars — this is the format
        # shown in the reference planogram output (e.g. "019674209969").
        upc12_display = digits11.zfill(12)

        pair = (last5, upc12_display)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        idx.setdefault(last5, []).append(
            MatrixRow(
                upc12=upc12_display,
                norm_name=str(r["__norm"]),
                display_name=display_name,
                cpp_qty=cpp,
            )
        )
    return idx


def resolve_full_pallet(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[MatrixRow]:
    key = _to_last5(last5)
    rows = idx.get(key, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    target = _norm_name(label_name)
    return max(rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio())


def _resolve(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[MatrixRow]:
    rows = idx.get(last5, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    target = _norm_name(label_name)
    return max(
        rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio()
    )


def _coerce_item_no(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"\.0$", "", str(v).strip())
    s = DIGITS_RE.sub("", s)
    return s.zfill(6) if s else None


@st.cache_data(show_spinner=False)
def load_ppt_cards(pptx_bytes: bytes) -> Dict[str, PptSideCards]:
    """
    Full Pallet / Multi-Zone: Robust PPT top/side card extraction.

    Extracts per SIDE slide:
      - card_id from "ID #nn"
      - title as lines above the ID line in the same textbox
      - image as the matched image primitive directly (Option A)

    Robustness:
      - Supports nested groups
      - Supports picture shapes and picture-fill shapes (blipFill)
      - Does not assume grouping consistency
      - Option A: matches ID-label textboxes directly to image primitives (NO global clustering into tiles)
      - Orders tiles in visual reading order (row clustering then left->right)

    Output:
      Dict[side_letter] = PptSideCards(side, top8, side6)

    Notes:
      - Assumes (per your constraint) images and text never overlap: prim.bottom <= label.top (+tol)
      - If matching fails, raises ValueError with an actionable message.
    """
    import io
    import re
    import math
    import hashlib
    import statistics
    from io import BytesIO
    from typing import Any, Dict, List, Optional, Tuple

    from PIL import Image
    from pptx import Presentation  # type: ignore
    from pptx.enum.shapes import MSO_SHAPE_TYPE  # type: ignore
    from pptx.oxml.ns import qn  # type: ignore

    # --- regex ---
    id_re = re.compile(r"\bID\s*#?\s*[:\-]?\s*(\d{1,8})\b", re.IGNORECASE)
    side_re = re.compile(r"\bSIDE\s*([A-D])\b", re.IGNORECASE)

    # --- PPT units ---
    # 1 inch = 914400 EMU, 1 pt = 12700 EMU
    EMU_PER_PT = 12700.0

    prs = Presentation(io.BytesIO(pptx_bytes))
    slide_w = float(prs.slide_width)
    slide_h = float(prs.slide_height)
    slide_area = slide_w * slide_h

    # Namespaces for relationship lookup (NOT for BaseOxmlElement.xpath kwargs)
    A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
    R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    NS = {"a": A_NS, "r": R_NS}

    # ----------------------------
    # helpers: geometry
    # ----------------------------
    def _bbox(sh) -> Tuple[float, float, float, float]:
        x = float(getattr(sh, "left", 0) or 0)
        y = float(getattr(sh, "top", 0) or 0)
        w = float(getattr(sh, "width", 0) or 0)
        h = float(getattr(sh, "height", 0) or 0)
        return x, y, w, h

    def _union_bbox(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        l = min(x1, x2)
        t = min(y1, y2)
        r = max(x1 + w1, x2 + w2)
        b = max(y1 + h1, y2 + h2)
        return (l, t, r - l, b - t)

    def _x_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1.0, min(a1 - a0, b1 - b0))
        return inter / denom

    def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        if inter <= 0:
            return 0.0
        a1 = w1 * h1
        a2 = w2 * h2
        return inter / max(1.0, (a1 + a2 - inter))

    def _cluster_rows(items: List[dict], tol: float) -> List[List[dict]]:
        rows: List[dict] = []
        for it in sorted(items, key=lambda d: d["cy"]):
            placed = False
            for row in rows:
                if abs(it["cy"] - row["cy"]) <= tol:
                    row["items"].append(it)
                    row["cy"] = statistics.mean([x["cy"] for x in row["items"]])
                    placed = True
                    break
            if not placed:
                rows.append({"cy": it["cy"], "items": [it]})
        rows.sort(key=lambda r: r["cy"])
        for row in rows:
            row["items"].sort(key=lambda d: d["cx"])
        return [r["items"] for r in rows]

    # ----------------------------
    # helpers: shape traversal
    # ----------------------------
    def _iter_shapes_recursive(shapes) -> List[Any]:
        out: List[Any] = []
        stack = list(shapes)
        while stack:
            sh = stack.pop(0)
            out.append(sh)
            if getattr(sh, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
                stack = list(sh.shapes) + stack
        return out

    # ----------------------------
    # helpers: parse label
    # ----------------------------
    def _parse_label(txt: str) -> Optional[Tuple[str, str]]:
        lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
        if not lines:
            return None
        id_idx = None
        card_id = None
        for i, ln in enumerate(lines):
            m = id_re.search(ln)
            if m:
                id_idx = i
                card_id = m.group(1).strip()
                break
        if id_idx is None or not card_id:
            return None
        title = " ".join(lines[:id_idx]).strip()
        return card_id, title

    # ----------------------------
    # helpers: image extraction from shapes
    # ----------------------------
    def _extract_picture_blob(sh) -> Optional[Tuple[bytes, str]]:
        # (A) Normal picture shape
        if getattr(sh, "shape_type", None) == MSO_SHAPE_TYPE.PICTURE:
            img = getattr(sh, "image", None)
            blob = getattr(img, "blob", None)
            if blob:
                ext = str(getattr(img, "ext", "png") or "png")
                return bytes(blob), ext

        # (B) Picture fill (blipFill) on non-picture shapes
        try:
            # Prefer namespace-aware lookup; fall back to local-name XPath for decks
            # where namespace prefixes are not resolved by the element wrapper.
            blips = sh._element.xpath(".//a:blip")
            if not blips:
                blips = sh._element.xpath(".//*[local-name()='blip']")
            if blips:
                blip = blips[0]
                rid = (
                    blip.get(f"{{{R_NS}}}embed")
                    or blip.get("r:embed")
                    or blip.get("embed")
                )
                if not rid:
                    for k, v in blip.attrib.items():
                        if str(k).endswith("}embed") or str(k).endswith(":embed") or str(k) == "embed":
                            rid = v
                            break
                if rid:
                    part = sh.part.related_parts.get(rid)
                    if part is not None and hasattr(part, "blob"):
                        return bytes(part.blob), "png"
        except Exception:
            pass

        return None

    # ----------------------------
    # helpers: cluster images into "tiles"  (kept for later, unused in Option A)
    # ----------------------------
    class DSU:
        def __init__(self, n: int):
            self.p = list(range(n))
            self.r = [0] * n

        def find(self, a: int) -> int:
            while self.p[a] != a:
                self.p[a] = self.p[self.p[a]]
                a = self.p[a]
            return a

        def union(self, a: int, b: int):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.r[ra] < self.r[rb]:
                ra, rb = rb, ra
            self.p[rb] = ra
            if self.r[ra] == self.r[rb]:
                self.r[ra] += 1

    def _build_image_tiles(primitives: List[dict]) -> List[dict]:
        """
        Unused in Option A (kept for reference).
        """
        if not primitives:
            return []

        heights = [p["h"] for p in primitives if p["h"] > 0]
        med_h = statistics.median(heights) if heights else (slide_h * 0.1)

        dsu = DSU(len(primitives))

        for i in range(len(primitives)):
            bi = primitives[i]["bbox"]
            xi, yi, wi, hi = bi
            for j in range(i + 1, len(primitives)):
                bj = primitives[j]["bbox"]
                xj, yj, wj, hj = bj

                iou = _iou(bi, bj)
                if iou >= 0.05:
                    dsu.union(i, j)
                    continue

                x_ov = _x_overlap_ratio(xi, xi + wi, xj, xj + wj)
                if x_ov >= 0.60:
                    gap = min(abs((yi + hi) - yj), abs((yj + hj) - yi))
                    if gap <= 0.25 * med_h:
                        dsu.union(i, j)

        groups: Dict[int, List[int]] = {}
        for idx in range(len(primitives)):
            root = dsu.find(idx)
            groups.setdefault(root, []).append(idx)

        tiles: List[dict] = []
        for _, idxs in groups.items():
            bb = primitives[idxs[0]]["bbox"]
            for k in idxs[1:]:
                bb = _union_bbox(bb, primitives[k]["bbox"])

            layers = sorted([primitives[k] for k in idxs], key=lambda p: p["z"])
            x, y, w, h = bb
            tiles.append(
                {
                    "bbox": bb,
                    "x0": x,
                    "x1": x + w,
                    "top": y,
                    "bottom": y + h,
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "h": h,
                    "layers": layers,
                    "area": w * h,
                }
            )

        tiles = [t for t in tiles if 0 < t["area"] <= slide_area * 0.40]
        tiles = [t for t in tiles if t["area"] >= slide_area * 0.001]

        return tiles

    def _composite_tile_png(tile: dict, scale_px_per_pt: float = 2.2) -> bytes:
        x, y, w, h = tile["bbox"]
        w_px = max(1, int(round((w / EMU_PER_PT) * scale_px_per_pt)))
        h_px = max(1, int(round((h / EMU_PER_PT) * scale_px_per_pt)))
        canvas = Image.new("RGBA", (w_px, h_px), (255, 255, 255, 0))

        cache: Dict[str, Image.Image] = {}
        for layer in tile["layers"]:
            blob = layer["blob"]
            lx, ly, lw, lh = layer["bbox"]
            if lw <= 1 or lh <= 1:
                continue

            key = hashlib.sha1(blob).hexdigest()
            if key in cache:
                src = cache[key]
            else:
                src = Image.open(BytesIO(blob)).convert("RGBA")
                cache[key] = src

            rx = ((lx - x) / EMU_PER_PT) * scale_px_per_pt
            ry = ((ly - y) / EMU_PER_PT) * scale_px_per_pt
            rw = (lw / EMU_PER_PT) * scale_px_per_pt
            rh = (lh / EMU_PER_PT) * scale_px_per_pt

            if rw <= 1 or rh <= 1:
                continue

            resized = src.resize((int(round(rw)), int(round(rh))), Image.LANCZOS)
            canvas.alpha_composite(resized, dest=(int(round(rx)), int(round(ry))))

        out = BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()

    # ----------------------------
    # main loop: per slide
    # ----------------------------
    best_by_side: Dict[str, Tuple[int, List[PptCard], List[PptCard]]] = {}

    for sidx, slide in enumerate(prs.slides):
        # --- side detection ---
        side_letter: Optional[str] = None
        for sh in _iter_shapes_recursive(slide.shapes):
            if getattr(sh, "has_text_frame", False):
                txt = (getattr(sh, "text", "") or "").strip()
                m = side_re.search(txt)
                if m:
                    side_letter = m.group(1).upper()
                    break
        if side_letter is None:
            side_letter = chr(ord("A") + min(sidx, 3))

        # --- labels: ID-anchored text boxes ---
        labels: List[dict] = []
        for sh in _iter_shapes_recursive(slide.shapes):
            if not getattr(sh, "has_text_frame", False):
                continue
            txt = (getattr(sh, "text", "") or "").strip()
            parsed = _parse_label(txt)
            if not parsed:
                continue
            card_id, title = parsed
            x, y, w, h = _bbox(sh)
            labels.append(
                {
                    "card_id": str(card_id),
                    "title": str(title),
                    "top": y,
                    "bottom": y + h,
                    "x0": x,
                    "x1": x + w,
                    "cx": x + w / 2.0,
                }
            )

        if not labels:
            # not a SIDE slide with cards
            continue

        # --- image primitives (pictures and picture fills) ---
        primitives: List[dict] = []
        z = 0
        recursive_shapes = _iter_shapes_recursive(slide.shapes)
        for sh in recursive_shapes:
            # Group containers are not directly renderable image tiles and can
            # duplicate nested picture references.
            if getattr(sh, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
                continue
            res = _extract_picture_blob(sh)
            if not res:
                continue
            blob, ext = res
            x, y, w, h = _bbox(sh)
            area = w * h
            if area <= 0:
                continue
            # drop huge backgrounds + tiny icons early
            if area > slide_area * 0.55 or area < slide_area * 0.0004:
                continue

            primitives.append(
                {
                    "blob": blob,
                    "ext": ext,
                    "bbox": (x, y, w, h),
                    "x0": x,
                    "x1": x + w,
                    "top": y,
                    "bottom": y + h,
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "h": h,
                    "z": z,
                }
            )
            z += 1

        if not primitives:
            raise ValueError(
                f"PPT side {side_letter}: found {len(labels)} ID labels but 0 image primitives "
                f"(scanned {len(recursive_shapes)} shapes recursively)"
            )

        # --- Option A: direct label -> primitive matching (NO clustering into tiles) ---
        # Deterministic label order (top -> bottom, then left -> right)
        labels.sort(key=lambda d: (d["top"], d["cx"]))

        def _build_candidates(cfg: Dict[str, float]) -> List[List[Tuple[float, int]]]:
            out: List[List[Tuple[float, int]]] = []
            for lab in labels:
                lab_w = max(1.0, float(lab["x1"] - lab["x0"]))
                lab_h = max(1.0, float(lab["bottom"] - lab["top"]))
                row_radius = max(lab_h * 1.8, slide_h * float(cfg["row_frac"]))
                max_dx = max(lab_w * float(cfg["cx_mult"]), slide_w * float(cfg["dx_frac"]))
                cands: List[Tuple[float, int]] = []

                for pidx, p in enumerate(primitives):
                    x_ov = _x_overlap_ratio(lab["x0"], lab["x1"], p["x0"], p["x1"])
                    dx = abs(lab["cx"] - p["cx"])
                    if x_ov < float(cfg["xmin"]) and dx > max_dx:
                        continue
                    if p["top"] > lab["top"] + row_radius:
                        continue
                    if bool(cfg["require_above"]) and p["bottom"] > lab["top"] + float(cfg["tol_y"]):
                        continue

                    dy_up = max(0.0, lab["top"] - p["bottom"])
                    dy_down = max(0.0, p["bottom"] - (lab["top"] + float(cfg["tol_y"])))
                    score = (
                        dy_up
                        + float(cfg["down_w"]) * dy_down
                        + float(cfg["dx_w"]) * dx
                        + float(cfg["size_w"]) * abs(p["h"] - lab_h)
                        - float(cfg["xov_bonus"]) * x_ov * slide_w
                    )
                    cands.append((score, pidx))

                cands.sort(key=lambda t: (t[0], t[1]))
                out.append(cands)
            return out

        def _try_bipartite(cands: List[List[Tuple[float, int]]]) -> Optional[List[int]]:
            if not cands:
                return []
            if any(len(c) == 0 for c in cands):
                return None

            match_prim_to_lab = [-1] * len(primitives)

            def _dfs(lab_idx: int, seen: set[int]) -> bool:
                for _, prim_idx in cands[lab_idx]:
                    if prim_idx in seen:
                        continue
                    seen.add(prim_idx)
                    prev_lab = match_prim_to_lab[prim_idx]
                    if prev_lab == -1 or _dfs(prev_lab, seen):
                        match_prim_to_lab[prim_idx] = lab_idx
                        return True
                return False

            order = sorted(range(len(labels)), key=lambda i: (len(cands[i]), labels[i]["top"], labels[i]["cx"]))
            for lab_idx in order:
                if not _dfs(lab_idx, set()):
                    return None

            lab_to_prim = [-1] * len(labels)
            for prim_idx, lab_idx in enumerate(match_prim_to_lab):
                if lab_idx != -1:
                    lab_to_prim[lab_idx] = prim_idx
            if any(pidx < 0 for pidx in lab_to_prim):
                return None
            return lab_to_prim

        matching_phases: List[Dict[str, float]] = [
            {
                "xmin": 0.30,
                "tol_y": slide_h * 0.08,
                "dx_w": 0.35,
                "down_w": 6.0,
                "xov_bonus": 0.06,
                "size_w": 0.02,
                "cx_mult": 1.1,
                "dx_frac": 0.05,
                "row_frac": 0.10,
                "require_above": 1.0,
            },
            {
                "xmin": 0.16,
                "tol_y": slide_h * 0.14,
                "dx_w": 0.30,
                "down_w": 5.0,
                "xov_bonus": 0.05,
                "size_w": 0.015,
                "cx_mult": 1.5,
                "dx_frac": 0.08,
                "row_frac": 0.14,
                "require_above": 1.0,
            },
            {
                "xmin": 0.05,
                "tol_y": slide_h * 0.22,
                "dx_w": 0.25,
                "down_w": 3.0,
                "xov_bonus": 0.03,
                "size_w": 0.01,
                "cx_mult": 2.0,
                "dx_frac": 0.12,
                "row_frac": 0.20,
                "require_above": 0.0,
            },
        ]

        assignments: Optional[List[int]] = None
        loosest_cands: Optional[List[List[Tuple[float, int]]]] = None
        for phase in matching_phases:
            cands = _build_candidates(phase)
            loosest_cands = cands
            assignments = _try_bipartite(cands)
            if assignments is not None:
                break

        if assignments is None:
            # Identify the first problematic label for actionable diagnostics.
            bad_idx = 0
            if loosest_cands:
                for i, c in enumerate(loosest_cands):
                    if not c:
                        bad_idx = i
                        break
            bad = labels[bad_idx]
            cand_count = len(loosest_cands[bad_idx]) if loosest_cands else 0
            raise ValueError(
                f"PPT side {side_letter}: could not match label ID {bad['card_id']} to any image primitive "
                f"(candidates in loosest phase={cand_count}, labels={len(labels)}, primitives={len(primitives)})"
            )

        matched: List[dict] = []
        for lab_idx, lab in enumerate(labels):
            prim = primitives[assignments[lab_idx]]

            # Use the matched primitive's image directly (in this deck it is the full card image).
            img_bytes = prim["blob"]
            img_ext = prim.get("ext") or "png"

            # Normalize to PNG for consistent downstream rendering.
            if str(img_ext).lower() != "png":
                try:
                    im = Image.open(BytesIO(img_bytes)).convert("RGBA")
                    out = BytesIO()
                    im.save(out, format="PNG")
                    img_bytes = out.getvalue()
                    img_ext = "png"
                except Exception:
                    pass

            card = PptCard(
                card_id=lab["card_id"],
                title=lab["title"],
                image_bytes=img_bytes,
                image_ext=img_ext,
            )
            matched.append({"card": card, "cx": prim["cx"], "cy": prim["cy"], "h": prim["h"]})

        # --- order in reading order ---
        med_h = statistics.median([m["h"] for m in matched]) if matched else (slide_h * 0.1)
        rows = _cluster_rows(matched, tol=0.35 * med_h)
        ordered_cards = [it["card"] for row in rows for it in row]

        # split by slide storage order: top row then remaining block
        top8 = ordered_cards[:8]
        side6 = ordered_cards[8:14]  # variable count is OK; caller can render len(side6)

        total = len(ordered_cards)
        prev = best_by_side.get(side_letter)
        if prev is None or total > prev[0]:
            best_by_side[side_letter] = (total, top8, side6)

    parsed: Dict[str, PptSideCards] = {}
    for side in "ABCD":
        entry = best_by_side.get(side)
        if entry:
            _, top8, side6 = entry
        else:
            top8, side6 = [], []
        parsed[side] = PptSideCards(side=side, top8=top8, side6=side6)

    return parsed    


def validate_ppt_side_cards(ppt_cards: Dict[str, PptSideCards]) -> List[str]:
    issues: List[str] = []
    for side in "ABCD":
        side_cards = ppt_cards.get(side, PptSideCards(side=side, top8=[], side6=[]))
        if len(side_cards.top8) != 8 or len(side_cards.side6) != 6:
            issues.append(
                f"SIDE {side}: found top8={len(side_cards.top8)} and side6={len(side_cards.side6)} (expected 8 and 6)."
            )
    return issues


@st.cache_data(show_spinner=False)
def load_gift_card_holders(gift_bytes: bytes) -> Dict[str, List[GiftHolder]]:
    """Parse FULL PALLET holder band and return ordered per-side holders (A/B/C/D).

    Full Pallet / Multi-Zone only.
    Sides are assigned by block order left→right, where blocks are separated by
    "MARKETING MESSAGE PANEL" divider columns.
    """

    try:
        from openpyxl import load_workbook  # type: ignore
        from openpyxl.utils.cell import coordinate_from_string, column_index_from_string  # type: ignore
    except Exception as e:
        raise ImportError("openpyxl is required for holder extraction") from e

    wb = load_workbook(io.BytesIO(gift_bytes), data_only=True)
    sheet_name = next(
        (n for n in wb.sheetnames if "FULL" in n.upper() and "PALLET" in n.upper()),
        None,
    )
    if sheet_name is None:
        raise ValueError("FULL PALLET sheet/table missing in workbook.")

    ws = wb[sheet_name]

    def cell_text(v: object) -> str:
        return str(v or "").strip()

    def is_mmp(v: object) -> bool:
        t = cell_text(v).upper()
        return "MARKETING" in t and "MESSAGE" in t and "PANEL" in t

    def is_item_no(v: object) -> Optional[str]:
        return _coerce_item_no(v)

    max_scan_rows = min(ws.max_row, 220)
    max_cols = ws.max_column

    # Find the item#, qty, description rows (robust; no hardcoded row numbers).
    best_item_row = -1
    best_item_hits = 0
    for r in range(1, max_scan_rows + 1):
        hits = 0
        for c in range(1, max_cols + 1):
            if is_item_no(ws.cell(r, c).value):
                hits += 1
        if hits > best_item_hits and hits >= 4:
            best_item_hits = hits
            best_item_row = r

    if best_item_row < 0:
        raise ValueError("Unable to locate ITEM # row in FULL PALLET sheet.")

    item_row = best_item_row

    holder_cols = [c for c in range(1, max_cols + 1) if is_item_no(ws.cell(item_row, c).value)]
    if not holder_cols:
        return {s: [] for s in "ABCD"}

    # Divider columns (MARKETING MESSAGE PANEL) – scan header band above item row.
    divider_cols: List[int] = []
    for r in range(1, min(item_row, 90) + 1):
        for c in range(1, max_cols + 1):
            if is_mmp(ws.cell(r, c).value):
                divider_cols.append(c)
    divider_cols = sorted(set(divider_cols))

    # Qty row – prefer explicit 'QTY' label, else choose row with many small ints aligned to holder cols.
    qty_row = -1
    for r in range(max(1, item_row - 12), item_row):
        row_vals = [cell_text(ws.cell(r, c).value).upper() for c in range(1, max_cols + 1)]
        if any(v == "QTY" or v.startswith("QTY") for v in row_vals):
            qty_row = r
            break
    if qty_row < 0:
        best_r, best_hits = -1, 0
        for r in range(max(1, item_row - 12), item_row):
            hits = 0
            for c in holder_cols:
                v = _coerce_int(ws.cell(r, c).value)
                if v is not None and 0 <= v <= 200:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_r = r
        qty_row = best_r

    # Description row – prefer explicit label, else choose first row below item row with many text cells.
    desc_row = -1
    for r in range(item_row, min(ws.max_row, item_row + 18) + 1):
        row_vals = [cell_text(ws.cell(r, c).value).upper() for c in range(1, max_cols + 1)]
        if any("DESCRIPTION" in v for v in row_vals):
            desc_row = r
            break
    if desc_row < 0:
        for r in range(item_row + 1, min(ws.max_row, item_row + 18) + 1):
            hits = 0
            for c in holder_cols:
                t = cell_text(ws.cell(r, c).value)
                if t and any(ch.isalpha() for ch in t):
                    hits += 1
            if hits >= 3:
                desc_row = r
                break

    # Map embedded images by anchored column.
    col_to_img: Dict[int, Tuple[bytes, str]] = {}
    for img in getattr(ws, "_images", []) or []:
        try:
            col: Optional[int] = None
            anch = getattr(img, "anchor", None)

            if hasattr(anch, "_from") and hasattr(anch._from, "col"):
                col = int(anch._from.col) + 1
            elif isinstance(anch, str):
                col_letters, _ = coordinate_from_string(anch)
                col = column_index_from_string(col_letters)

            if col is None:
                continue

            data = img._data()
            if not data:
                continue
            ext = "png"
            try:
                ext = str(getattr(getattr(img, "_data", None), "ext", "png"))
            except Exception:
                pass

            prev = col_to_img.get(col)
            if prev is None or len(data) > len(prev[0]):
                col_to_img[col] = (bytes(data), ext)
        except Exception:
            continue

    # Assign holder columns into blocks split by divider columns.
    col_to_block: Dict[int, int] = {}
    block = 0
    div_i = 0
    holder_cols_sorted = sorted(holder_cols)
    prev_col = holder_cols_sorted[0]
    col_to_block[prev_col] = block
    for col in holder_cols_sorted[1:]:
        while div_i < len(divider_cols) and divider_cols[div_i] <= prev_col:
            div_i += 1
        while div_i < len(divider_cols) and divider_cols[div_i] < col:
            block += 1
            div_i += 1
        col_to_block[col] = block
        prev_col = col

    blocks: Dict[int, List[int]] = {}
    for col in holder_cols_sorted:
        blocks.setdefault(col_to_block[col], []).append(col)

    side_letters = "ABCD"
    holders: Dict[str, List[GiftHolder]] = {s: [] for s in side_letters}

    overrides: Dict[str, str] = {"109107": "A"}

    for bidx in sorted(blocks.keys()):
        if bidx >= len(side_letters):
            continue
        side = side_letters[bidx]
        for col in blocks[bidx]:
            item_no = is_item_no(ws.cell(item_row, col).value)
            if not item_no:
                continue

            side_eff = overrides.get(item_no, side)

            qty = _coerce_int(ws.cell(qty_row, col).value) if qty_row > 0 else None
            name = cell_text(ws.cell(desc_row, col).value) if desc_row > 0 else ""
            if not name:
                name = "(missing description)"

            img_bytes = None
            img_ext = None
            if col in col_to_img:
                img_bytes, img_ext = col_to_img[col]

            holders[side_eff].append(
                GiftHolder(
                    side=side_eff,
                    item_no=item_no,
                    name=name,
                    qty=qty,
                    image_bytes=img_bytes,
                    image_ext=img_ext,
                )
            )

    # If an override item lives in a later block (e.g., 109107), pull it into its override side.
    # (Keeps the behaviour deterministic even when the sheet layout is slightly off.)
    for col in holder_cols_sorted:
        item_no = is_item_no(ws.cell(item_row, col).value)
        if not item_no or item_no not in overrides:
            continue
        target_side = overrides[item_no]
        already = {h.item_no for h in holders.get(target_side, [])}
        if item_no in already:
            continue
        qty = _coerce_int(ws.cell(qty_row, col).value) if qty_row > 0 else None
        name = cell_text(ws.cell(desc_row, col).value) if desc_row > 0 else ""
        img_bytes = col_to_img.get(col, (None, None))[0]
        img_ext = col_to_img.get(col, (None, None))[1]
        holders[target_side].append(
            GiftHolder(
                side=target_side,
                item_no=item_no,
                name=name or "(missing description)",
                qty=qty,
                image_bytes=img_bytes,
                image_ext=img_ext,
            )
        )

    return holders


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
        ln
        for ln in lines
        if not (last5 and last5 in ln)
        and not (qty is not None and re.fullmatch(str(qty), ln))
    ).strip()

    return name, last5, qty


# ──────────────────────────────────────────────────────────────────────────────
# Standard display extraction (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: int) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [
                w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))
            ]
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

            pages.append(
                PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells)
            )
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Full-pallet page extraction
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
            if abs(cx - gcx) <= max(x_tol, (bx1 - bx0) / 2 + x_tol * 0.4) and abs(
                cy - gcy
            ) <= max(y_tol, (bb - bt) / 2 + y_tol * 0.4):
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
                if not (
                    bx1 + x_tol < gx0
                    or gx1 + x_tol < bx0
                    or bb + y_tol < gt
                    or gb + y_tol < bt
                ):
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
    def _detect_side_letter(words: List[dict], pw: float, ph: float, fallback: str) -> str:
        bottom = [
            w for w in (words or [])
            if str(w.get("text", "") or "").strip().upper() in {"A", "B", "C", "D"}
            and float(w.get("top", 0) or 0) > ph * 0.72
            and pw * 0.25 < float(w.get("x0", 0) or 0) < pw * 0.75
        ]
        if bottom:
            best = max(
                bottom,
                key=lambda w: float(w.get("bottom", 0) or 0) - float(w.get("top", 0) or 0),
            )
            return str(best.get("text", "") or "").strip().upper()

        joined = " ".join(str(w.get("text", "") or "") for w in (words or []))
        m = re.search(r"\bSIDE\s*([A-D])\b", joined, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # IMPORTANT: do NOT use "top-right single letter" fallback; it can match the header "… PKG D"
        return fallback

    side_map: Dict[str, FullPalletPage] = {}

    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            pw, ph = float(page.width), float(page.height)
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []

            fallback = chr(ord("A") + min(pidx, 3))
            side_letter = _detect_side_letter(words, pw, ph, fallback)

            five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            holder_zone_bottom = ph * 0.31
            five = [w for w in five if float(w.get("top", 0)) >= holder_zone_bottom]

            cells: List[CellData] = []
            xs: List[float] = []
            ys: List[float] = []

            if five:
                xs = [(w["x0"] + w["x1"]) / 2 for w in five]
                ys = [(w["top"] + w["bottom"]) / 2 for w in five]

                x_centers = cluster_positions(xs, tol=max(10, pw * 0.025))
                y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))

                if len(x_centers) and len(y_centers):
                    x_bounds = boundaries_from_centers(x_centers)
                    y_bounds = boundaries_from_centers(y_centers)

                    if len(x_bounds) >= 2 and len(y_bounds) >= 2:
                        cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
                        for w, xc, yc in zip(five, xs, ys):
                            col = int(np.argmin(np.abs(x_centers - xc)))
                            row = int(np.argmin(np.abs(y_centers - yc)))
                            dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                            key = (row, col)
                            if key not in cell_map or dist < cell_map[key][0]:
                                cell_map[key] = (dist, w)

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
                            raw_last5 = token_last5 if re.fullmatch(r"\d{5}", token_last5) else parsed_last5
                            last5 = _to_last5(raw_last5)

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

            fp = FullPalletPage(
                page_index=pidx,
                side_letter=side_letter,
                cells=cells,
                annotations=annotations,
            )

            prev = side_map.get(side_letter)
            if prev is None or len(fp.cells) > len(prev.cells):
                side_map[side_letter] = fp

    return [side_map[s] for s in "ABCD" if s in side_map]

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


def image_from_bytes(img_bytes: Optional[bytes]) -> Optional[Image.Image]:
    if not img_bytes:
        return None
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────


def wrap_text(text: str, max_w: float, font: str, size: float) -> List[str]:
    parts = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []
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


def _draw_gradient(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
    steps: int = 80,
) -> None:
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
    page_w: float,
    page_h: float,
    top_bar_h: float,
    title_text: str,
    right_label: str,
    logo_img: Optional[Image.Image],
    grad_l: Tuple[float, float, float],
    grad_r: Tuple[float, float, float],
) -> None:
    hy = page_h - top_bar_h
    _draw_gradient(c, 0, hy, page_w, top_bar_h, grad_l, grad_r, steps=120)

    tx = 14.0
    if logo_img:
        lw, lh = logo_img.size
        th = top_bar_h * 0.62
        r = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(
            ImageReader(logo_img),
            14,
            hy + (top_bar_h - dh) / 2,
            dw,
            dh,
            mask="auto",
        )
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


_STOPWORDS = {
    "GIFT", "CARD", "CARDS", "VALUE", "THE", "AND", "FOR", "WITH", "IN",
    "OF", "A", "AN", "WALMART", "WM", "VGC", "GC",
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
    if re.fullmatch(r"[A-Z]\d{1,3}", t):
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


def _fit_name_preserve_qualifiers(name: str, font: str, size: float, max_w: float) -> str:
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    tokens = raw.split()

    def is_important(t: str) -> bool:
        tt = t.upper()
        if "$" in tt and re.search(r"\d", tt):
            return True
        if re.search(r"\d", tt) and any(x in tt for x in ("PK", "CT", "OZ", "LB", "MG", "ML")):
            return True
        if re.fullmatch(r"[A-Z]\d{1,3}", tt):
            return True
        return False

    keep = tokens[:]
    while keep and pdfmetrics.stringWidth(" ".join(keep), font, size) > max_w:
        removable = [i for i, t in enumerate(keep) if 0 < i < len(keep) - 1 and not is_important(t)]
        if not removable:
            break
        mid = removable[len(removable) // 2]
        keep.pop(mid)

    candidate = " ".join(keep)
    if pdfmetrics.stringWidth(candidate, font, size) <= max_w:
        return candidate
    return _ellipsis(candidate, font, size, max_w)


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
        c.drawImage(
            ImageReader(logo_img),
            18,
            y0 + (header_h - dh) / 2,
            dw,
            dh,
            mask="auto",
        )
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
# Full-Pallet Display renderer
# ──────────────────────────────────────────────────────────────────────────────


def render_full_pallet_pdf(
    pages: List[FullPalletPage],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str = "POG",
    ppt_cards: Optional[Dict[str, PptSideCards]] = None,
    gift_holders: Optional[Dict[str, List[GiftHolder]]] = None,
    ppt_cpp_global: Optional[int] = None,
    debug: bool = False,
    debug_overlay: bool = False,
) -> bytes:
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    PAGE_W, PAGE_H = 792.0, 1224.0  # 11x17 portrait template
    MARGIN = 24.0
    HEADER_H = 56.0
    FOOTER_H = 36.0
    SECTION_BAR_H = 24.0
    SECTION_BAR_GAP = 10.0
    BUCKET_GAP = 12.0
    BAR_FILL = _hex_to_rgb("#77B5F0")
    BAR_TEXT = NAVY_RGB

    EMPTY_STROKE = (0.88, 0.88, 0.88)
    FILLED_STROKE = (0.78, 0.78, 0.80)

    logo = _try_load_logo()

    def _marker_y(p: FullPalletPage, kind: str) -> Optional[float]:
        for ann in p.annotations:
            if ann.kind == kind:
                _, t, _, b = ann.bbox
                return (t + b) / 2
        return None

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

    def _split_rows_by_bonus(p: FullPalletPage, global_rows: List[int]) -> Tuple[List[int], List[int]]:
        bonus_y = _marker_y(p, "bonus_strip")

        row_to_y: Dict[int, float] = {}
        for r in global_rows:
            ys = [(_cell_center(c)[1]) for c in p.cells if c.row == r]
            row_to_y[r] = float(np.mean(ys)) if ys else float("inf")

        yc = [row_to_y[r] for r in global_rows]

        if bonus_y is not None and len(yc) >= 2:
            above = [r for r in global_rows if row_to_y[r] < bonus_y]
            below = [r for r in global_rows if row_to_y[r] >= bonus_y]
            if above and below:
                return above, below

        if len(yc) < 2:
            return global_rows, []

        gaps = [yc[i + 1] - yc[i] for i in range(len(yc) - 1)]
        i_max = int(np.argmax(gaps))
        if gaps[i_max] < float(np.median(gaps)) * 1.6:
            return global_rows, []
        return global_rows[: i_max + 1], global_rows[i_max + 1 :]

    def _fit_x_layout(
        x0: float,
        x1: float,
        n_cols: int,
        gap_units: List[int],
        desired_card_w: float,
        desired_gap: float,
    ) -> Tuple[List[float], float, float, float, bool]:
        if n_cols <= 0:
            return [], desired_card_w, desired_gap, x0, False

        units = [max(1, int(gap_units[i])) if i < len(gap_units) else 1 for i in range(n_cols - 1)]
        avail_w = max(1.0, x1 - x0)
        gap = max(0.0, desired_gap)
        card_w = max(24.0, desired_card_w)

        def total_w(cw: float, gp: float) -> float:
            return n_cols * cw + sum(units) * gp

        t = total_w(card_w, gap)
        overflowed = t > avail_w

        if t > avail_w and sum(units) > 0:
            max_gap = (avail_w - n_cols * card_w) / sum(units)
            gap = max(2.0, min(gap, max_gap))
            t = total_w(card_w, gap)
        if t > avail_w:
            card_w = max(24.0, (avail_w - sum(units) * gap) / n_cols)
            t = total_w(card_w, gap)
        if t > avail_w:
            gap = 0.0
            card_w = max(24.0, avail_w / n_cols)
            t = total_w(card_w, gap)

        xs: List[float] = []
        cur = x0
        for i in range(n_cols):
            xs.append(cur)
            cur += card_w
            if i < n_cols - 1:
                cur += gap * units[i]

        right = xs[-1] + card_w
        if right > x1:
            shift = right - x1
            xs = [x - shift for x in xs]
            right = xs[-1] + card_w
            overflowed = True

        return xs, card_w, gap, right, overflowed

    def _section_cols_and_gaps(
        p: FullPalletPage,
        sec_rows: List[int],
        global_cols: List[int],
        global_gap_units: List[int],
    ) -> Tuple[List[int], List[int]]:
        if not sec_rows:
            return [], []
        row_set = set(sec_rows)
        global_rank = {c_: i for i, c_ in enumerate(global_cols)}
        sec_cols = sorted(
            {c.col for c in p.cells if c.row in row_set},
            key=lambda c_: global_rank.get(c_, 10**6 + c_),
        )
        if len(sec_cols) <= 1:
            return sec_cols, []
        gap_units: List[int] = []
        for i in range(len(sec_cols) - 1):
            lidx = global_rank.get(sec_cols[i], i)
            ridx = global_rank.get(sec_cols[i + 1], lidx + 1)
            if ridx <= lidx:
                gap_units.append(1)
                continue
            unit = sum(global_gap_units[lidx:ridx]) if global_gap_units else 1
            gap_units.append(max(1, int(unit)))
        return sec_cols, gap_units

    def _draw_section_bar(
        c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str
    ) -> None:
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

        text_h = max(8.0, min(34.0, ih * 0.42))
        img_h = max(8.0, ih - text_h - 2.0)
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

        name_fs = 6.5
        nm = _fit_name_preserve_qualifiers((name or "").upper(), BODY_FONT, name_fs, iw)
        c.setFillColorRGB(0.10, 0.10, 0.10)
        c.setFont(BODY_FONT, name_fs)
        tw = pdfmetrics.stringWidth(nm, BODY_FONT, name_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + (name_h - name_fs) / 2, nm)

        cpp_fs = _fit_font(cpp_str, BODY_BOLD_FONT, iw, cpp_h, 6.0, 12.0, step=0.25)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.setFont(BODY_BOLD_FONT, cpp_fs)
        tw = pdfmetrics.stringWidth(cpp_str, BODY_BOLD_FONT, cpp_fs)
        c.drawString(ix + (iw - tw) / 2, iy + (cpp_h - cpp_fs) / 2, cpp_str)

    def _draw_debug_box(
        c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str
    ) -> None:
        c.setStrokeColorRGB(0.92, 0.20, 0.20)
        c.setLineWidth(0.7)
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFillColorRGB(0.75, 0.10, 0.10)
        c.setFont("Helvetica", 7)
        c.drawString(x + 2, y + h - 9, label)

    try:
        if not pages:
            c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))
            c.save()
            return buf.getvalue()

        c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))

        for pdata in pages:
            _draw_full_pallet_header(
                c,
                PAGE_W,
                PAGE_H,
                HEADER_H,
                title_prefix.strip() or "POG",
                f"SIDE {pdata.side_letter}",
                logo,
            )
            _draw_footer(c, PAGE_W, MARGIN, FOOTER_H)

            cx0, cx1 = MARGIN, PAGE_W - MARGIN
            cy0, cy1 = MARGIN + FOOTER_H, PAGE_H - MARGIN - HEADER_H
            content_w = cx1 - cx0
            content_h = cy1 - cy0

            bucket_a_h = content_h * 0.27
            bucket_b_h = content_h * 0.18
            bucket_d_h = content_h * 0.21
            bucket_c_h = content_h - bucket_a_h - bucket_b_h - bucket_d_h - BUCKET_GAP * 3

            bucket_a_top = cy1
            bucket_a_bottom = bucket_a_top - bucket_a_h
            bucket_b_top = bucket_a_bottom - BUCKET_GAP
            bucket_b_bottom = bucket_b_top - bucket_b_h
            bucket_c_top = bucket_b_bottom - BUCKET_GAP
            bucket_c_bottom = bucket_c_top - bucket_c_h
            bucket_d_top = bucket_c_bottom - BUCKET_GAP
            bucket_d_bottom = cy0

            global_cols, global_gap_units = _global_col_order_and_gaps(pdata)
            global_rows = _global_row_order(pdata)
            above_bonus_rows, below_bonus_rows = _split_rows_by_bonus(pdata, global_rows)

            side_ppt = (
                ppt_cards.get(pdata.side_letter, PptSideCards(pdata.side_letter, [], []))
                if ppt_cards
                else PptSideCards(pdata.side_letter, [], [])
            )
            side_holders = gift_holders.get(pdata.side_letter, []) if gift_holders else []

            product_map: Dict[Tuple[int, int], Tuple[Optional[MatrixRow], CellData]] = {}
            for cell in pdata.cells:
                if cell.last5:
                    match = resolve_full_pallet(cell.last5, cell.name, matrix_idx)
                    product_map[(cell.row, cell.col)] = (match, cell)

            if debug_overlay:
                c.setStrokeColorRGB(0.20, 0.70, 0.25)
                c.setLineWidth(0.8)
                c.rect(cx0, cy0, cx1 - cx0, cy1 - cy0, stroke=1, fill=0)
                c.setStrokeColorRGB(0.95, 0.10, 0.10)
                c.line(cx1, cy0, cx1, cy1)

            rightmost_used = cx0
            adjusted_to_fit = False
            matched_cells = 0
            unmatched_cells = 0
            unresolved_main: List[str] = []
            unresolved_bonus: List[str] = []

            # Bucket A: PPT Top Cards
            a_gap_y = 8.0
            top_card_h = max(30.0, min(86.0, bucket_a_h * 0.34))
            top_xs, top_card_w, _, top_right, top_overflow = _fit_x_layout(
                cx0, cx1, 8, [1] * 7, 72.0, 6.0
            )
            adjusted_to_fit = adjusted_to_fit or top_overflow
            rightmost_used = max(rightmost_used, top_right)

            top_row_y = bucket_a_top - top_card_h
            for i in range(8):
                x = top_xs[i]
                card = side_ppt.top8[i] if i < len(side_ppt.top8) else None
                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE if card else EMPTY_STROKE)
                c.setLineWidth(0.75 if card else 0.45)
                c.rect(x, top_row_y, top_card_w, top_card_h, stroke=1, fill=1)
                if card:
                    _draw_card(
                        c, x, top_row_y, top_card_w, top_card_h,
                        image_from_bytes(card.image_bytes), f"ID# {card.card_id}", card.title, ppt_cpp_global,
                    )

            side_block_w = 2 * top_card_w + 6.0
            sx0 = max(cx0, cx1 - side_block_w)
            side_top = top_row_y - SECTION_BAR_GAP
            side_available_h = max(36.0, side_top - bucket_a_bottom)
            side_xs = [
    sx0,
    min(cx1 - top_card_w, sx0 + (top_card_w + 6.0)),
    min(cx1 - top_card_w, sx0 + 2 * (top_card_w + 6.0)),
]
            side_card_h = max(20.0, min(top_card_h, (side_available_h - 2 * a_gap_y) / 3))
            # 3 cols x 2 rows (row-major)
        for row in range(2):
            y = side_top - (row + 1) * side_card_h - row * a_gap_y
            for col in range(3):
                idx = row * 3 + col  # [0,1,2] then [3,4,5]
                x = side_xs[col]
                card = side_ppt.side6[idx] if idx < len(side_ppt.side6) else None
                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE if card else EMPTY_STROKE)
                c.setLineWidth(0.75 if card else 0.45)
                c.rect(x, y, top_card_w, side_card_h, stroke=1, fill=1)
                if card:
                    _draw_card(
                            c, x, y, top_card_w, side_card_h,
                            image_from_bytes(card.image_bytes), f"ID# {card.card_id}", card.title, ppt_cpp_global,
                        )
                    rightmost_used = max(rightmost_used, x + top_card_w)

            if debug_overlay:
                _draw_debug_box(c, cx0, bucket_a_bottom, content_w, bucket_a_top - bucket_a_bottom, "PPT")

            # Bucket B: Gift Card Holders
            holder_bar_y = bucket_b_top - SECTION_BAR_H
            _draw_section_bar(c, cx0, holder_bar_y, content_w, SECTION_BAR_H, "GIFT CARD HOLDERS")
            holder_grid_top = holder_bar_y - SECTION_BAR_GAP
            holder_grid_h = max(24.0, holder_grid_top - bucket_b_bottom)
            holder_cols = max(1, min(8, len(global_cols) if global_cols else 8))
            holder_rows = max(1, int(math.ceil(len(side_holders) / holder_cols)))
            holder_gutter_y = 6.0
            holder_card_h = max(16.0, (holder_grid_h - max(0, holder_rows - 1) * holder_gutter_y) / holder_rows)

            holder_xs, holder_card_w, _, holder_right, holder_overflow = _fit_x_layout(
                cx0, cx1, holder_cols, [1] * max(0, holder_cols - 1), 74.0, 6.0
            )
            adjusted_to_fit = adjusted_to_fit or holder_overflow
            rightmost_used = max(rightmost_used, holder_right)

            gift_top = holder_grid_top
            for ri in range(holder_rows):
                y = gift_top - (ri + 1) * holder_card_h - ri * holder_gutter_y
                for ci in range(holder_cols):
                    idx = ri * holder_cols + ci
                    x = holder_xs[ci]
                    holder = side_holders[idx] if idx < len(side_holders) else None
                    c.setFillColorRGB(1, 1, 1)
                    c.setStrokeColorRGB(*FILLED_STROKE if holder else EMPTY_STROKE)
                    c.setLineWidth(0.75 if holder else 0.45)
                    c.rect(x, y, holder_card_w, holder_card_h, stroke=1, fill=1)
                    if holder:
                        _draw_card(
                            c, x, y, holder_card_w, holder_card_h,
                            image_from_bytes(holder.image_bytes), holder.item_no, holder.name, holder.qty,
                        )

            if debug_overlay:
                _draw_debug_box(c, cx0, bucket_b_bottom, content_w, bucket_b_top - bucket_b_bottom, "HOLDERS")

            def draw_product_grid(
                sec_rows: List[int],
                sec_top: float,
                sec_bottom: float,
                label: Optional[str],
                unresolved_bucket: List[str],
            ) -> Tuple[int, int, bool]:
                nonlocal rightmost_used, adjusted_to_fit, matched_cells, unmatched_cells
                y_cursor = sec_top
                if label is not None:
                    bar_y = y_cursor - SECTION_BAR_H
                    _draw_section_bar(c, cx0, bar_y, content_w, SECTION_BAR_H, label)
                    y_cursor = bar_y - SECTION_BAR_GAP

                sec_rows_sorted = list(sec_rows)
                if not sec_rows_sorted:
                    return 0, 0, False

                sec_cols, sec_gap_units = _section_cols_and_gaps(
                    pdata, sec_rows_sorted, global_cols, global_gap_units
                )
                sec_col_rank = {c_: i for i, c_ in enumerate(sec_cols)}
                n_rows = len(sec_rows_sorted)
                row_gutter = 6.0
                available_h = max(24.0, y_cursor - sec_bottom)
                card_h = max(12.0, (available_h - max(0, n_rows - 1) * row_gutter) / n_rows)
                xs, sec_card_w, _, right, overflow = _fit_x_layout(
                    cx0, cx1, len(sec_cols), sec_gap_units, 74.0, 6.0
                )
                adjusted_to_fit = adjusted_to_fit or overflow
                rightmost_used = max(rightmost_used, right)

                rmap = {r: i for i, r in enumerate(sec_rows_sorted)}
                row_set = set(sec_rows_sorted)
                occ: Dict[Tuple[int, int], CellData] = {}
                for cell in pdata.cells:
                    if cell.row in row_set and cell.col in sec_col_rank:
                        occ[(rmap[cell.row], sec_col_rank[cell.col])] = cell

                grid_top = y_cursor
                for ri in range(n_rows):
                    y = grid_top - (ri + 1) * card_h - ri * row_gutter
                    for ci in range(len(sec_cols)):
                        x = xs[ci]
                        cell = occ.get((ri, ci))
                        if cell is None:
                            c.setFillColorRGB(1, 1, 1)
                            c.setStrokeColorRGB(*EMPTY_STROKE)
                            c.setLineWidth(0.45)
                            c.rect(x, y, sec_card_w, card_h, stroke=1, fill=0)
                            continue

                        key = (cell.row, cell.col)
                        match, _cell = product_map.get(key, (None, cell))
                        last5_key = _to_last5(cell.last5)
                        upc12 = match.upc12 if match else None
                        cpp = match.cpp_qty if match else None
                        disp_name = (
                            match.display_name if match and match.display_name else cell.name
                        ).strip()
                        if upc12:
                            upc_str = upc12
                        else:
                            upc_str = f"LAST5 {last5_key}"
                            disp_name = f"UNRESOLVED {last5_key}"
                            unresolved_bucket.append(last5_key)
                        try:
                            img = crop_image_cell(
                                images_doc, pdata.page_index, cell.bbox, zoom=2.6, inset=0.045
                            )
                        except Exception:
                            img = None

                        if match:
                            matched_cells += 1
                        else:
                            unmatched_cells += 1

                        c.setFillColorRGB(1, 1, 1)
                        c.setStrokeColorRGB(*FILLED_STROKE)
                        c.setLineWidth(0.75)
                        c.rect(x, y, sec_card_w, card_h, stroke=1, fill=1)
                        _draw_card(c, x, y, sec_card_w, card_h, img, upc_str, disp_name, cpp)

                if debug_overlay:
                    _draw_debug_box(
                        c, cx0, sec_bottom, content_w, sec_top - sec_bottom, label or "MAIN",
                    )
                return len(sec_cols), n_rows, overflow

            # Bucket C: main product grid (rows above BONUS bar in labels PDF)
            main_cols, main_rows_count, main_over = draw_product_grid(
                above_bonus_rows, bucket_c_top, bucket_c_bottom, None, unresolved_main
            )
            # Bucket D: bonus product grid (rows below BONUS bar in labels PDF)
            bonus_cols, bonus_rows_count, bonus_over = draw_product_grid(
                below_bonus_rows, bucket_d_top, bucket_d_bottom, "BONUS", unresolved_bonus
            )
            adjusted_to_fit = adjusted_to_fit or main_over or bonus_over

            right_limit = cx1
            exceeded = rightmost_used > right_limit + 0.001
            adjusted_to_fit = adjusted_to_fit or exceeded

            main_last5_codes = sorted({_to_last5(c.last5) for c in pdata.cells if c.row in set(above_bonus_rows)})
            bonus_last5_codes = sorted({_to_last5(c.last5) for c in pdata.cells if c.row in set(below_bonus_rows)})
            main_slots_total = sum(1 for c in pdata.cells if c.row in set(above_bonus_rows))
            bonus_slots_total = sum(1 for c in pdata.cells if c.row in set(below_bonus_rows))

            if debug:
                st.write(
                    {
                        "side": pdata.side_letter,
                        "page_size": f"{PAGE_W}x{PAGE_H}",
                        "margins": {"left": cx0, "right": cx1, "top": cy1, "bottom": cy0},
                        "bucket_regions": {
                            "ppt": [round(bucket_a_top, 1), round(bucket_a_bottom, 1)],
                            "holders": [round(bucket_b_top, 1), round(bucket_b_bottom, 1)],
                            "main": [round(bucket_c_top, 1), round(bucket_c_bottom, 1)],
                            "bonus": [round(bucket_d_top, 1), round(bucket_d_bottom, 1)],
                        },
                        "grid_detected": {
                            "main_cols": main_cols,
                            "main_rows": main_rows_count,
                            "bonus_cols": bonus_cols,
                            "bonus_rows": bonus_rows_count,
                        },
                        "layout_width": {
                            "rightmost_x": round(rightmost_used, 2),
                            "right_limit": round(right_limit, 2),
                            "exceeded": exceeded,
                            "adjusted_to_fit": adjusted_to_fit,
                        },
                        "ppt_counts": {
                            "top8": len(side_ppt.top8),
                            "side6": len(side_ppt.side6),
                            "ids_top8": [c_.card_id for c_ in side_ppt.top8],
                            "ids_side6": [c_.card_id for c_ in side_ppt.side6],
                        },
                        "holders": {
                            "count": len(side_holders),
                            "items": [h.item_no for h in side_holders],
                        },
                        "slot_codes": {
                            "main_total_slots": main_slots_total,
                            "bonus_total_slots": bonus_slots_total,
                            "main_unique_count": len(main_last5_codes),
                            "bonus_unique_count": len(bonus_last5_codes),
                            "main": main_last5_codes,
                            "bonus": bonus_last5_codes,
                        },
                        "unresolved_last5": {
                            "main": sorted({x for x in unresolved_main if x}),
                            "bonus": sorted({x for x in unresolved_bonus if x}),
                        },
                        "matrix_matches": {
                            "matched": matched_cells,
                            "unmatched": unmatched_cells,
                        },
                    }
                )

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
            c, page_w, page_h, top_bar_h,
            title_prefix or "POG", "", logo_img, grad_left, grad_right,
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
                    c, out_x0, out_bottom, out_w, text_area_h,
                    cell.name, upc12, cell.last5, qty,
                )

        c.showPage()
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

        st.dataframe(
            pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
            use_container_width=True,
            height=420,
        )

        if generate:
            with st.spinner("Rendering PDF…"):
                pdf = render_standard_pog_pdf(
                    pages, images_bytes, matrix_idx, title_prefix.strip() or "POG",
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
        pptx_file = st.file_uploader("Top Cards Blueprint (.pptx)", type=["pptx"])
        gift_file = st.file_uploader("2025 D82 POG Workbook (.xlsx)", type=["xlsx"])
        ppt_cpp_global = st.number_input("PPT Cards CPP (Global)", min_value=0, value=0, step=1)
        show_debug = st.checkbox("Show debug details")
        show_layout_overlay = st.checkbox("Show Full Pallet layout overlay")

        if not (pptx_file and gift_file):
            st.info("Upload PPTX + POG XLSX for Full Pallet mode.")
            return

        try:
            ppt_cards = load_ppt_cards(pptx_file.getvalue())
        except ImportError:
            st.error(
                "python-pptx is not installed. Full Pallet mode requires python-pptx to parse the Top Cards Blueprint."
            )
            return
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error("Unable to parse Top Cards Blueprint (.pptx). Please verify the file.")
            return

        ppt_issues = validate_ppt_side_cards(ppt_cards)
        if ppt_issues:
            st.error("Top Cards Blueprint validation failed:\n" + "\n".join(f"- {msg}" for msg in ppt_issues))
            return

        if ppt_cpp_global in (None, 0):
            st.warning("PPT CPP global is 0 or empty. PPT cards will render with CPP: 0.")

        try:
            gift_holders = load_gift_card_holders(gift_file.getvalue())
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error(f"Unable to parse POG workbook holder table: {e}")
            return

        try:
            fp_matrix_idx = load_full_pallet_matrix_index(matrix_bytes)
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error("Unable to parse D82 Item List for Full Pallet matching.")
            return

        fp_pages = extract_full_pallet_pages(labels_bytes)
        if not fp_pages:
            st.error("No product cells detected in Labels PDF.")
            return

        st.subheader(f"Detected {len(fp_pages)} side(s) — one output page per side")
        rows = []
        for pg in fp_pages:
            for cell in pg.cells:
                match = resolve_full_pallet(cell.last5, cell.name, fp_matrix_idx) if cell.last5 else None
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

        st.dataframe(
            pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
            use_container_width=True,
            height=420,
        )

        if generate:
            with st.spinner("Rendering PDF…"):
                try:
                    pdf = render_full_pallet_pdf(
                        fp_pages,
                        images_bytes,
                        fp_matrix_idx,
                        title_prefix.strip() or "POG",
                        ppt_cards=ppt_cards,
                        gift_holders=gift_holders,
                        ppt_cpp_global=ppt_cpp_global,
                        debug=show_debug,
                        debug_overlay=show_layout_overlay,
                    )
                except Exception as e:
                    if show_debug:
                        st.exception(e)
                    else:
                        st.error("Error generating Full Pallet PDF. Enable debug for details.")
                    return
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
