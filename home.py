# home.py
"""
Streamlit POG generator
"""

from __future__ import annotations

import io
import re
import math
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics


LAST5_RE = re.compile(r"\b(\d{5})\b")
DIGITS_RE = re.compile(r"\D+")

# Hardcoded — matches the 3-column structure of the source PDFs
N_COLS = 3


@dataclass(frozen=True)
class MatrixRow:
    upc12: str
    norm_name: str


@dataclass(frozen=True)
class CellData:
    row: int
    col: int
    bbox: Tuple[float, float, float, float]  # (x0, top, x1, bottom) in PDF top-left coords
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
    df.columns = ["UPC", "Name", "Extra"][: df.shape[1]] + [f"col_{i}" for i in range(max(0, df.shape[1] - 3))]

    df["upc12"] = df["UPC"].map(_coerce_upc12)
    df = df[df["upc12"].notna()].copy()
    df["last5"] = df["upc12"].str[-5:]
    df["norm_name"] = df["Name"].astype(str).map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["last5"])
        idx.setdefault(last5, []).append(MatrixRow(upc12=str(r["upc12"]), norm_name=str(r["norm_name"])))

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


def boundaries_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(sorted(centers), dtype=float)
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
                bbox = (float(x_bounds[col]), float(y_bounds[row]), float(x_bounds[col + 1]), float(y_bounds[row + 1]))
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


def render_pog_pdf(
    pages: List[PageData],
    images_pdf_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    n_cols: int,
    title_prefix: str = "POG",
) -> bytes:
    """
    Renders all sides onto a SINGLE wide page, laid out horizontally
    left-to-right: Side A | Side B | Side C | Side D ...
    Small gaps between cells for visual clarity.
    """
    buf = io.BytesIO()

    n_sides = len(pages)

    # Layout constants
    outer_margin = 32          # page outer margin (pt)
    side_gap = 18              # gap between adjacent sides
    cell_inset = 3             # inner gap shrink per cell edge (visual separation)
    top_header_h = 52          # space for title + side label at top
    bottom_margin = 28

    # Compute per-side width so everything fits in a reasonably wide page
    # Target: each side ~300pt wide; adjust if many/few sides
    per_side_w = 310
    total_page_w = outer_margin * 2 + n_sides * per_side_w + (n_sides - 1) * side_gap
    total_page_h = 8.5 * 72   # 612 pt — standard letter height

    pagesize = (total_page_w, total_page_h)
    c = canvas.Canvas(buf, pagesize=pagesize)

    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    try:
        content_top = total_page_h - outer_margin - top_header_h
        content_bottom = bottom_margin
        avail_h = content_top - content_bottom

        # ── Main title (top-left) ────────────────────────────────────────────
        c.setFont("Helvetica-Bold", 13)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.drawString(outer_margin, total_page_h - outer_margin - 13, title_prefix)

        # ── Thin horizontal rule under title ────────────────────────────────
        rule_y = total_page_h - outer_margin - 22
        c.setLineWidth(0.6)
        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.line(outer_margin, rule_y, total_page_w - outer_margin, rule_y)

        for out_i, p in enumerate(pages):
            side = chr(ord("A") + out_i)
            # X origin for this side's content block
            side_origin_x = outer_margin + out_i * (per_side_w + side_gap)

            # ── Side label ───────────────────────────────────────────────────
            c.setFont("Helvetica-Bold", 9)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.drawString(side_origin_x, total_page_h - outer_margin - 38, f"Side {side}")

            # Thin vertical separator between sides (skip before first)
            if out_i > 0:
                sep_x = side_origin_x - side_gap / 2
                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.82, 0.82, 0.82)
                c.line(sep_x, content_bottom, sep_x, total_page_h - outer_margin - 4)

            # ── Scale this side's grid to fit per_side_w × avail_h ──────────
            x_min = float(p.x_bounds[0])
            x_max = float(p.x_bounds[-1])
            y_min = float(p.y_bounds[0])
            y_max = float(p.y_bounds[-1])

            scale = min(
                per_side_w / max(1e-6, x_max - x_min),
                avail_h   / max(1e-6, y_max - y_min),
            )

            for cell in p.cells:
                upc12 = resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None

                x0, top, x1, bottom = cell.bbox

                # Map PDF coords → page coords
                ox0 = side_origin_x + (x0 - x_min) * scale
                ox1 = side_origin_x + (x1 - x_min) * scale
                oy_top    = content_top - (top    - y_min) * scale
                oy_bottom = content_top - (bottom - y_min) * scale

                # Apply cell inset for visual gap between adjacent cells
                ox0      += cell_inset
                ox1      -= cell_inset
                oy_top   -= cell_inset
                oy_bottom += cell_inset

                ow = ox1 - ox0
                oh = oy_top - oy_bottom

                if ow <= 0 or oh <= 0:
                    continue

                # ── Cell border ──────────────────────────────────────────────
                c.setLineWidth(0.45)
                c.setStrokeColorRGB(0.78, 0.78, 0.78)
                c.setFillColorRGB(1, 1, 1)
                c.rect(ox0, oy_bottom, ow, oh, stroke=1, fill=1)

                # ── Card image (top 62 % of cell) ────────────────────────────
                img = crop_image_cell(images_doc, p.page_index, cell.bbox, zoom=3.0, inset=0.08)
                img_box_h = oh * 0.62
                img_box_w = ow * 0.90
                iw, ih = img.size
                r = min(img_box_w / max(1, iw), img_box_h / max(1, ih))
                dw, dh = iw * r, ih * r
                ix = ox0 + (ow - dw) / 2
                iy = oy_bottom + oh - img_box_h + (img_box_h - dh) / 2
                c.drawImage(
                    ImageReader(img), ix, iy, dw, dh,
                    preserveAspectRatio=True, mask="auto"
                )

                # ── Light separator line between image and text area ─────────
                sep_line_y = oy_bottom + oh * 0.38
                c.setLineWidth(0.3)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(ox0 + 2, sep_line_y, ox1 - 2, sep_line_y)

                # ── Text block (bottom 38 % of cell) ─────────────────────────
                tx = ox0 + 4
                ty = oy_bottom + 4

                c.setFillColorRGB(0.12, 0.12, 0.12)
                c.setFont("Helvetica", 6)

                if cell.qty is not None:
                    c.drawString(tx, ty, f"Qty: {cell.qty}")
                    ty += 7.5

                if upc12:
                    c.drawString(tx, ty, f"UPC: {upc12}")
                elif cell.last5:
                    c.drawString(tx, ty, f"UPC: ???????{cell.last5}")
                ty += 7.5

                c.setFont("Helvetica-Bold", 6)
                for ln in wrap_text(cell.name, max(10.0, ow - 8), "Helvetica-Bold", 6)[:2]:
                    c.drawString(tx, ty, ln)
                    ty += 7.5

        c.showPage()
        c.save()
        return buf.getvalue()

    finally:
        images_doc.close()


def main() -> None:
    st.set_page_config(page_title="POG Generator", layout="wide")
    st.title("POG Generator (Labels PDF + Images PDF + Matrix XLSX → Export PDF)")

    with st.sidebar:
        st.header("Inputs")
        matrix_file = st.file_uploader("Matrix Excel (.xlsx)", type=["xlsx"])
        labels_pdf  = st.file_uploader("Labels PDF (text + last5 + qty)", type=["pdf"])
        images_pdf  = st.file_uploader("Images PDF (card pictures)", type=["pdf"])

        st.divider()
        title_prefix = st.text_input("PDF title prefix", value="POG")
        out_name     = st.text_input("Output filename", value="pog_export.pdf")

        generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

    pages = extract_pages_from_labels(labels_bytes, N_COLS)
    if not pages:
        st.error("Could not detect any grid cells from Labels PDF (no 5-digit UPCs found).")
        return

    st.subheader("Detected pages / sides")
    st.write(f"Pages detected: **{len(pages)}** (will export as Side A, B, C...)")

    preview_rows: List[dict] = []
    for i, p in enumerate(pages):
        side = chr(ord("A") + i)
        for cell in p.cells:
            preview_rows.append(
                {
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
            pdf_bytes = render_pog_pdf(
                pages=pages,
                images_pdf_bytes=images_bytes,
                matrix_idx=matrix_idx,
                n_cols=N_COLS,
                title_prefix=title_prefix.strip() or "POG",
            )

        st.success("Done.")
        st.download_button(
            label="Download POG PDF",
            data=pdf_bytes,
            file_name=out_name if out_name.lower().endswith(".pdf") else f"{out_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()