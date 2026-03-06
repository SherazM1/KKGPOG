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


def _hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) != 6:
        return (0.2, 0.2, 0.2)
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
        r = left_rgb[0] * (1 - t) + right_rgb[0] * t
        g = left_rgb[1] * (1 - t) + right_rgb[1] * t
        b = left_rgb[2] * (1 - t) + right_rgb[2] * t
        c.setFillColorRGB(r, g, b)
        c.setStrokeColorRGB(r, g, b)
        xi = x + (w * i / steps)
        wi = w / steps + 0.5
        c.rect(xi, y, wi, h, stroke=0, fill=1)


def _try_load_logo() -> Optional[Image.Image]:
    # Looks in repo/app directory for the exact filename.
    candidates = [
        Path.cwd() / "KKG-Logo-02.png",
        Path(__file__).resolve().parent / "KKG-Logo-02.png",
    ]
    for p in candidates:
        if p.exists():
            try:
                img = Image.open(p).convert("RGBA")
                return img
            except Exception:
                continue
    return None


def _ellipsize_to_width(text: str, font_name: str, font_size: float, max_width: float) -> str:
    t = (text or "").strip()
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width:
        return t
    ell = "…"
    lo, hi = 0, len(t)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = (t[:mid].rstrip() + ell).strip()
        if pdfmetrics.stringWidth(cand, font_name, font_size) <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


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
    pad = 6
    max_w = max(10.0, w - pad * 2)
    max_h = max(10.0, h - pad * 2)

    meta_upc = upc12 or (f"???????{last5}" if last5 else "")
    meta_parts = []
    if meta_upc:
        meta_parts.append(f"UPC: {meta_upc}")
    if qty is not None:
        meta_parts.append(f"Qty: {qty}")
    meta_line = "   ".join(meta_parts).strip()

    name_font = "Helvetica-Bold"
    meta_font = "Helvetica"

    # Shrink-to-fit loop (prevents bleed)
    for font_size in [10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5]:
        name_lines = wrap_text(name, max_w, name_font, font_size)
        name_lines = name_lines[:2] if name_lines else [""]

        meta_size = max(6.5, font_size - 1.0)
        meta_line_fit = _ellipsize_to_width(meta_line, meta_font, meta_size, max_w) if meta_line else ""

        line_h_name = font_size * 1.15
        line_h_meta = meta_size * 1.20
        needed_h = len(name_lines) * line_h_name + (line_h_meta if meta_line_fit else 0)

        if needed_h <= max_h:
            ty = y + h - pad - font_size  # top-down
            c.setFillColorRGB(0.10, 0.10, 0.10)

            c.setFont(name_font, font_size)
            for ln in name_lines:
                ln_fit = _ellipsize_to_width(ln, name_font, font_size, max_w)
                c.drawString(x + pad, ty, ln_fit)
                ty -= line_h_name

            if meta_line_fit:
                c.setFont(meta_font, meta_size)
                c.setFillColorRGB(0.18, 0.18, 0.18)
                c.drawString(x + pad, max(y + pad, ty), meta_line_fit)

            return

    # Last resort (tiny)
    c.setFont("Helvetica", 6.5)
    c.setFillColorRGB(0.18, 0.18, 0.18)
    c.drawString(x + pad, y + pad, _ellipsize_to_width(meta_line, "Helvetica", 6.5, max_w))


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
    Visual upgrades:
      - 1.5× scale
      - gradient header + logo
      - larger type
      - contain images on white
      - no text bleed (wrap + shrink + ellipsis)
      - footer with date + generated by
    """
    buf = io.BytesIO()
    images_doc = fitz.open(stream=images_pdf_bytes, filetype="pdf")

    # ── Styling constants ──────────────────────────────────────────────────
    scale_factor = 1.5  # requested
    outer_margin = 44
    side_gap = 28
    top_bar_h = 84
    footer_h = 36

    cell_inset = 5
    border_w = 0.75

    img_frac = 0.72  # contain; rest is text
    img_inset = 0.06

    # Gradient (approx brand blue from logo)
    grad_left = _hex_to_rgb01("#5B63A9")
    grad_right = _hex_to_rgb01("#3E4577")

    logo_img = _try_load_logo()

    # ── Compute page size dynamically (one big sheet) ──────────────────────
    n_sides = len(pages)
    per_side_w = int(310 * scale_factor)

    side_ranges: List[Tuple[float, float]] = []
    side_scales: List[float] = []
    side_scaled_heights: List[float] = []

    for p in pages:
        x_min = float(p.x_bounds[0])
        x_max = float(p.x_bounds[-1])
        y_min = float(p.y_bounds[0])
        y_max = float(p.y_bounds[-1])
        w0 = max(1e-6, x_max - x_min)
        h0 = max(1e-6, y_max - y_min)

        sc = per_side_w / w0  # width-driven => height grows as needed
        side_ranges.append((w0, h0))
        side_scales.append(sc)
        side_scaled_heights.append(sc * h0)

    content_h = max(side_scaled_heights) if side_scaled_heights else 600.0
    page_w = outer_margin * 2 + n_sides * per_side_w + (n_sides - 1) * side_gap
    page_h = outer_margin + top_bar_h + content_h + footer_h + outer_margin

    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    try:
        # ── Header bar (gradient) ──────────────────────────────────────────
        header_y = page_h - top_bar_h
        _draw_horizontal_gradient(c, 0, header_y, page_w, top_bar_h, grad_left, grad_right, steps=120)

        # Logo (top-left)
        left_pad = 16
        if logo_img is not None:
            lw, lh = logo_img.size
            target_h = top_bar_h * 0.62
            r = target_h / max(1, lh)
            dw, dh = lw * r, lh * r
            c.drawImage(ImageReader(logo_img), left_pad, header_y + (top_bar_h - dh) / 2, dw, dh, mask="auto")
            title_x = left_pad + dw + 14
        else:
            title_x = left_pad

        # Title
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(title_x, header_y + top_bar_h * 0.55, title_prefix)

        # Subtle rule under header
        c.setLineWidth(1.0)
        c.setStrokeColorRGB(0.88, 0.88, 0.92)
        c.line(0, header_y, page_w, header_y)

        # Content frame
        content_top = header_y - 10
        content_bottom = outer_margin + footer_h
        # Footer rule
        c.setLineWidth(0.8)
        c.setStrokeColorRGB(0.82, 0.82, 0.82)
        c.line(outer_margin, content_bottom, page_w - outer_margin, content_bottom)

        # Footer text
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.setFont("Helvetica", 10)
        footer_text_left = f"Date: {date.today().isoformat()}"
        footer_text_mid = "Generated by Kendal King"
        c.drawString(outer_margin, outer_margin + 12, footer_text_left)
        mid_w = pdfmetrics.stringWidth(footer_text_mid, "Helvetica", 10)
        c.drawString((page_w - mid_w) / 2, outer_margin + 12, footer_text_mid)

        # ── Render each side block ─────────────────────────────────────────
        for out_i, p in enumerate(pages):
            side_letter = chr(ord("A") + out_i)
            side_origin_x = outer_margin + out_i * (per_side_w + side_gap)

            # Side badge above the side area (inside content region, under header)
            badge_y = content_top + 6
            badge_h = 26
            badge_w = 110
            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(0.85, 0.85, 0.90)
            c.setLineWidth(0.8)
            c.roundRect(side_origin_x, badge_y, badge_w, badge_h, 8, stroke=1, fill=1)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(side_origin_x + 12, badge_y + 7, f"Side {side_letter}")

            # Vertical separator between sides
            if out_i > 0:
                sep_x = side_origin_x - side_gap / 2
                c.setLineWidth(0.6)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(sep_x, content_bottom + 2, sep_x, content_top + 40)

            x_min = float(p.x_bounds[0])
            x_max = float(p.x_bounds[-1])
            y_min = float(p.y_bounds[0])
            y_max = float(p.y_bounds[-1])

            sc = side_scales[out_i]

            for cell in p.cells:
                upc12 = resolve_full_upc(cell.last5, cell.name, matrix_idx) if cell.last5 else None

                x0, top, x1, bottom = cell.bbox

                ox0 = side_origin_x + (x0 - x_min) * sc
                ox1 = side_origin_x + (x1 - x_min) * sc
                oy_top = content_top - (top - y_min) * sc
                oy_bottom = content_top - (bottom - y_min) * sc

                # Inset
                ox0 += cell_inset
                ox1 -= cell_inset
                oy_top -= cell_inset
                oy_bottom += cell_inset

                ow = ox1 - ox0
                oh = oy_top - oy_bottom
                if ow <= 2 or oh <= 2:
                    continue

                # Cell background + border (white)
                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.72, 0.72, 0.72)
                c.setLineWidth(border_w)
                c.rect(ox0, oy_bottom, ow, oh, stroke=1, fill=1)

                # Image area (contain)
                img_area_h = oh * img_frac
                text_area_h = oh - img_area_h

                img = crop_image_cell(
                    images_doc,
                    p.page_index,
                    cell.bbox,
                    zoom=3.2,
                    inset=0.08,
                )

                iw, ih = img.size
                img_box_w = ow * (1 - 2 * img_inset)
                img_box_h = img_area_h * (1 - 2 * img_inset)

                r = min(img_box_w / max(1, iw), img_box_h / max(1, ih))  # contain
                dw, dh = iw * r, ih * r

                ix = ox0 + (ow - dw) / 2
                iy = oy_bottom + text_area_h + (img_area_h - dh) / 2
                c.drawImage(ImageReader(img), ix, iy, dw, dh, preserveAspectRatio=True, mask="auto")

                # Separator between image and text
                sep_y = oy_bottom + text_area_h
                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.line(ox0 + 3, sep_y, ox1 - 3, sep_y)

                # Text block (no bleed)
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


def main() -> None:
    st.set_page_config(page_title="POG Generator", layout="wide")
    st.title("POG Generator")

    with st.sidebar:
        st.header("Inputs")
        matrix_file = st.file_uploader("Matrix Excel (.xlsx)", type=["xlsx"])
        labels_pdf = st.file_uploader("Labels PDF", type=["pdf"])
        images_pdf = st.file_uploader("Images PDF", type=["pdf"])

        st.divider()
        title_prefix = st.text_input("PDF title", value="POG")
        out_name = st.text_input("Output filename", value="pog_export.pdf")

        generate = st.button("Generate PDF", type="primary", use_container_width=True)

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