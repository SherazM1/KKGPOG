from __future__ import annotations

import io
from datetime import date
from typing import Dict, List

import fitz
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.shared.constants import NAVY_RGB
from app.shared.fonts import TITLE_FONT
from app.shared.image_utils import _hex_to_rgb, _try_load_logo, crop_image_cell
from app.shared.matching import _resolve
from app.shared.models import MatrixRow, PageData
from app.shared.text_utils import _draw_cell_text_block, _draw_footer, _draw_header, _fit_font


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
    base_side_w = int(310 * scale_factor)

    side_widths: List[float] = []
    for page in pages:
        row_counts = [
            sum(1 for cell in page.cells if cell.row == row)
            for row in sorted({cell.row for cell in page.cells})
        ]
        max_row_count = max(row_counts, default=1)
        if len(page.cells) > 40 or max_row_count >= 8:
            side_widths.append(base_side_w * max(1.0, max_row_count / 4.0))
        else:
            side_widths.append(float(base_side_w))

    side_scales: List[float] = []
    side_heights: List[float] = []
    side_y_spans: List[float] = []
    for page, side_w in zip(pages, side_widths):
        x_min = float(page.x_bounds[0])
        x_max = float(page.x_bounds[-1])
        y_min = float(page.y_bounds[0])
        y_max = float(page.y_bounds[-1])
        y_span = max(1e-6, y_max - y_min)

        scale = side_w / max(1e-6, x_max - x_min)
        side_scales.append(scale)
        side_heights.append(scale * y_span)
        side_y_spans.append(y_span)

    content_h = max(side_heights) if side_heights else 600.0
    side_y_scales: List[float] = []
    for page, scale, y_span, side_h in zip(pages, side_scales, side_y_spans, side_heights):
        row_counts = [
            sum(1 for cell in page.cells if cell.row == row)
            for row in sorted({cell.row for cell in page.cells})
        ]
        dense_side = len(page.cells) > 40 or max(row_counts, default=0) >= 8
        if dense_side and side_h < content_h:
            side_y_scales.append(content_h / y_span)
        else:
            side_y_scales.append(scale)

    side_origins: List[float] = []
    cursor_x = outer_margin
    for side_w in side_widths:
        side_origins.append(cursor_x)
        cursor_x += side_w + side_gap

    page_w = outer_margin * 2 + sum(side_widths) + max(0, side_count - 1) * side_gap
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
            side_origin_x = side_origins[side_idx]

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
            x_scale = side_scales[side_idx]
            y_scale = side_y_scales[side_idx]

            for cell in page.cells:
                match = _resolve(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                upc12 = match.upc12 if match else None
                qty = cell.qty if cell.qty is not None else (match.cpp_qty if match else None)

                x0, top, x1, bottom = cell.bbox
                out_x0 = side_origin_x + (x0 - x_min) * x_scale + cell_inset
                out_x1 = side_origin_x + (x1 - x_min) * x_scale - cell_inset
                out_top = cells_top - (top - y_min) * y_scale - cell_inset
                out_bottom = cells_top - (bottom - y_min) * y_scale + cell_inset

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
