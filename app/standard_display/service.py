from __future__ import annotations

from typing import Dict, List, Tuple

from app.shared.matching import _resolve, load_matrix_index
from app.shared.models import MatrixRow, PageData
from app.standard_display.extract import extract_pages_from_labels
from app.standard_display.render_pdf import render_standard_pog_pdf


def prepare_standard_display(
    matrix_bytes: bytes,
    labels_bytes: bytes,
    n_cols: int,
) -> Tuple[List[PageData], Dict[str, List[MatrixRow]], List[dict]]:
    matrix_idx = load_matrix_index(matrix_bytes)
    pages = extract_pages_from_labels(labels_bytes, n_cols)
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
    return pages, matrix_idx, rows


def render_standard_display_pdf(
    pages: List[PageData],
    images_bytes: bytes,
    matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str,
) -> bytes:
    return render_standard_pog_pdf(pages, images_bytes, matrix_idx, title_prefix.strip() or "POG")


def run_standard_display(
    matrix_bytes: bytes,
    labels_bytes: bytes,
    images_bytes: bytes,
    title_prefix: str,
    n_cols: int,
) -> Tuple[List[dict], bytes]:
    pages, matrix_idx, rows = prepare_standard_display(matrix_bytes, labels_bytes, n_cols)
    pdf = render_standard_display_pdf(pages, images_bytes, matrix_idx, title_prefix)
    return rows, pdf
