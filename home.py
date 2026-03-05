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
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter

    margin = 36
    title_h = 44

    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    try:
        for out_i, p in enumerate(pages):
            side = chr(ord("A") + out_i)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, page_h - margin - 14, f"{title_prefix} — Side {side}")

            content_top = page_h - margin - title_h
            content_bottom = margin

            x_min, x_max = float(p.x_bounds[0]), float(p.x_bounds[-1])
            y_min, y_max = float(p.y_bounds[0]), float(p.y_bounds[-1])

            avail_w = page_w - 2 * margin
            avail_h = content_top - content_bottom
            scale = min(avail_w / max(1e-6, (x_max - x_min)), avail_h / max(1e-6, (y_max - y_min)))

            for cell in p.cells:
                upc12 = resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None

                x0, top, x1, bottom = cell.bbox
                ox0 = margin + (x0 - x_min) * scale
                ox1 = margin + (x1 - x_min) * scale

                oy_top = content_top - (top - y_min) * scale
                oy_bottom = content_top - (bottom - y_min) * scale

                ow = ox1 - ox0
                oh = oy_top - oy_bottom

                c.setLineWidth(0.5)
                c.rect(ox0, oy_bottom, ow, oh, stroke=1, fill=0)

                img = crop_image_cell(images_doc, p.page_index, cell.bbox, zoom=3.0, inset=0.08)
                img_box_h = oh * 0.62
                img_box_w = ow * 0.90

                iw, ih = img.size
                r = min(img_box_w / max(1, iw), img_box_h / max(1, ih))
                dw, dh = iw * r, ih * r

                ix = ox0 + (ow - dw) / 2
                iy = oy_bottom + oh - img_box_h + (img_box_h - dh) / 2
                c.drawImage(ImageReader(img), ix, iy, dw, dh, preserveAspectRatio=True, mask="auto")

                tx = ox0 + 4
                ty = oy_bottom + 4

                c.setFont("Helvetica", 7)
                if cell.qty is not None:
                    c.drawString(tx, ty, f"Qty: {cell.qty}")
                ty += 8

                if upc12:
                    c.drawString(tx, ty, f"UPC: {upc12}")
                elif cell.last5:
                    c.drawString(tx, ty, f"UPC: ???????{cell.last5}")
                ty += 8

                c.setFont("Helvetica-Bold", 7)
                for ln in wrap_text(cell.name, max(10.0, ow - 8), "Helvetica-Bold", 7)[:2]:
                    c.drawString(tx, ty, ln)
                    ty += 8

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
        labels_pdf = st.file_uploader("Labels PDF (text + last5 + qty)", type=["pdf"])
        images_pdf = st.file_uploader("Images PDF (card pictures)", type=["pdf"])

        st.divider()
        n_cols = st.number_input("Columns in grid", min_value=2, max_value=6, value=3, step=1)
        title_prefix = st.text_input("PDF title prefix", value="POG")
        out_name = st.text_input("Output filename", value="pog_export.pdf")

        generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    matrix_idx = load_matrix_index(matrix_bytes)

    pages = extract_pages_from_labels(labels_bytes, int(n_cols))
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

    st.dataframe(pd.DataFrame(preview_rows).sort_values(["Side", "Row", "Col"]), use_container_width=True, height=420)

    if generate:
        with st.spinner("Generating PDF..."):
            pdf_bytes = render_pog_pdf(
                pages=pages,
                images_pdf_bytes=images_bytes,
                matrix_idx=matrix_idx,
                n_cols=int(n_cols),
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