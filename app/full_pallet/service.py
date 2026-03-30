from __future__ import annotations

from typing import Dict, List, Tuple

from app.full_pallet.extract import extract_full_pallet_pages
from app.full_pallet.gift_holders import load_gift_card_holders
from app.full_pallet.ppt import load_ppt_cards, validate_ppt_side_cards
from app.full_pallet.render_pdf import render_full_pallet_pdf
from app.shared.matching import load_full_pallet_matrix_index, resolve_full_pallet
from app.shared.models import FullPalletPage, GiftHolder, MatrixRow, PptSideCards


def parse_ppt_cards(pptx_bytes: bytes) -> Dict[str, PptSideCards]:
    return load_ppt_cards(pptx_bytes)


def validate_ppt_cards(ppt_cards: Dict[str, PptSideCards]) -> List[str]:
    return validate_ppt_side_cards(ppt_cards)


def parse_gift_holders(gift_bytes: bytes) -> Dict[str, List[GiftHolder]]:
    return load_gift_card_holders(gift_bytes)


def load_full_pallet_matrix(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    return load_full_pallet_matrix_index(matrix_bytes)


def parse_full_pallet_pages(labels_bytes: bytes) -> List[FullPalletPage]:
    return extract_full_pallet_pages(labels_bytes)


def build_full_pallet_rows(
    fp_pages: List[FullPalletPage],
    fp_matrix_idx: Dict[str, List[MatrixRow]],
) -> List[dict]:
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
    return rows


def render_full_pallet_display_pdf(
    fp_pages: List[FullPalletPage],
    images_bytes: bytes,
    fp_matrix_idx: Dict[str, List[MatrixRow]],
    title_prefix: str,
    ppt_cards: Dict[str, PptSideCards],
    gift_holders: Dict[str, List[GiftHolder]],
    ppt_cpp_global: int,
    debug: bool = False,
    debug_overlay: bool = False,
) -> bytes:
    return render_full_pallet_pdf(
        fp_pages,
        images_bytes,
        fp_matrix_idx,
        title_prefix.strip() or "POG",
        ppt_cards=ppt_cards,
        gift_holders=gift_holders,
        ppt_cpp_global=ppt_cpp_global,
        debug=debug,
        debug_overlay=debug_overlay,
    )


def run_full_pallet(
    matrix_bytes: bytes,
    labels_bytes: bytes,
    images_bytes: bytes,
    pptx_bytes: bytes,
    gift_bytes: bytes,
    title_prefix: str,
    ppt_cpp_global: int,
    debug: bool = False,
    debug_overlay: bool = False,
) -> Tuple[List[dict], bytes, List[str]]:
    ppt_cards = parse_ppt_cards(pptx_bytes)
    issues = validate_ppt_cards(ppt_cards)
    if issues:
        return [], b"", issues

    gift_holders = parse_gift_holders(gift_bytes)
    fp_matrix_idx = load_full_pallet_matrix(matrix_bytes)
    fp_pages = parse_full_pallet_pages(labels_bytes)
    rows = build_full_pallet_rows(fp_pages, fp_matrix_idx)

    pdf = render_full_pallet_display_pdf(
        fp_pages,
        images_bytes,
        fp_matrix_idx,
        title_prefix,
        ppt_cards,
        gift_holders,
        ppt_cpp_global,
        debug,
        debug_overlay,
    )
    return rows, pdf, []
