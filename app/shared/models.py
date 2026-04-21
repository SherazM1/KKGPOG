from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MatrixRow:
    upc12: str
    norm_name: str
    display_name: str
    cpp_qty: Optional[int]


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
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class FullPalletMidBandSlot:
    slot_id: str
    side_letter: str
    row_index: int
    block_name: str  # left | center | right
    block_pos_index: int
    slot_order: int  # 0..23
    slot_in_row: int  # 0..7
    bbox: Tuple[float, float, float, float]
    raw_label_text: str
    parsed_name: str
    last5: str
    qty: Optional[int]
    extraction_bbox: Optional[Tuple[float, float, float, float]] = None
    accepted_words: Optional[List[str]] = None
    rejected_nearby_word_count: Optional[int] = None
    resolved_upc12: Optional[str] = None
    resolved_display_name: Optional[str] = None
    resolved_cpp_qty: Optional[int] = None


@dataclass(frozen=True)
class FullPalletMidBandRow:
    row_index: int
    slots: List[FullPalletMidBandSlot]


@dataclass(frozen=True)
class FullPalletMidBandSection:
    section_id: str  # mid_band_above_bonus
    rows: List[FullPalletMidBandRow]
    slot_count: int
    row_slot_counts: List[int]
    row_block_grouping: List[List[int]]
    shape_valid: bool
    anchor_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class FullPalletPage:
    page_index: int
    side_letter: str
    cells: List[CellData]
    annotations: List[AnnotationBox]
    mid_band_above_bonus: Optional[FullPalletMidBandSection] = None


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
    slot_label: Optional[str] = None
    slot_order: Optional[int] = None
    slot_start_col: Optional[int] = None
    slot_end_col: Optional[int] = None
