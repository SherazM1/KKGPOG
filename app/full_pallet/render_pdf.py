from __future__ import annotations

import difflib
import io
import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from app.shared.constants import NAVY_RGB
from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT, TITLE_FONT
from app.shared.image_utils import _hex_to_rgb, _try_load_logo, crop_image_cell, image_from_bytes
from app.shared.matching import resolve_full_pallet
from app.shared.models import (
    CellData,
    FullPalletMidBandSection,
    FullPalletMidBandSlot,
    FullPalletPage,
    GiftHolder,
    MatrixRow,
    PptSideCards,
)
from app.shared.text_utils import (
    _draw_footer,
    _draw_full_pallet_header,
    _ellipsis,
    _fit_font,
    _fit_name_preserve_qualifiers,
    _norm_name,
    _to_last5,
)


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

    PAGE_W, BASE_PAGE_H = 792.0, 1224.0  # 11x17 portrait template; pages may grow taller
    MARGIN = 24.0
    HEADER_H = 56.0
    FOOTER_H = 36.0
    SECTION_BAR_H = 24.0
    SECTION_BAR_GAP = 10.0
    BUCKET_GAP = 12.0

    BASE_CONTENT_H = BASE_PAGE_H - (2 * MARGIN) - HEADER_H - FOOTER_H
    PPT_SECTION_H = BASE_CONTENT_H * 0.27
    HOLDER_SECTION_H = BASE_CONTENT_H * 0.18
    BAR_FILL = _hex_to_rgb("#77B5F0")
    BAR_TEXT = NAVY_RGB

    EMPTY_STROKE = (0.88, 0.88, 0.88)
    FILLED_STROKE = (0.78, 0.78, 0.80)

    logo = _try_load_logo()
    all_matrix_rows = [r for rows in matrix_idx.values() for r in rows]

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

    def _build_non_ppt_slot_map_for_side(
        p: FullPalletPage,
        global_rows: List[int],
        global_cols: List[int],
        above_bonus_rows: List[int],
        below_bonus_rows: List[int],
        product_map: Dict[Tuple[int, int], Tuple[Optional[MatrixRow], CellData]],
    ) -> Dict[str, object]:
        global_row_rank = {r: i for i, r in enumerate(global_rows)}
        global_col_rank = {c_: i for i, c_ in enumerate(global_cols)}

        def _build_section(section_kind: str, sec_rows: List[int]) -> Dict[str, object]:
            if not sec_rows:
                return {
                    "section_kind": section_kind,
                    "row_ids": [],
                    "col_ids": [],
                    "rows": [],
                    "slots_flat": [],
                }

            row_ids = list(sec_rows)
            row_set = set(row_ids)

            col_ids = sorted(
                {cell.col for cell in p.cells if cell.row in row_set},
                key=lambda c_: global_col_rank.get(c_, 10**6 + c_),
            )

            row_index_map = {r: i for i, r in enumerate(row_ids)}
            col_index_map = {c_: i for i, c_ in enumerate(col_ids)}

            rows_out: List[dict] = []
            slots_flat: List[dict] = []

            for row_id in row_ids:
                row_cells = [cell for cell in p.cells if cell.row == row_id]
                row_cells.sort(key=lambda cell: global_col_rank.get(cell.col, 10**6 + cell.col))

                slot_items: List[dict] = []
                for slot_order, cell in enumerate(row_cells):
                    key = (cell.row, cell.col)
                    match, _cell = product_map.get(key, (None, cell))

                    slot = {
                        "section_kind": section_kind,
                        "row_id": cell.row,
                        "col_id": cell.col,
                        "row_index": row_index_map[cell.row],
                        "col_index": col_index_map[cell.col],
                        "slot_order": slot_order,
                        "bbox": cell.bbox,
                        "label_name": cell.name,
                        "last5": cell.last5,
                        "qty": cell.qty,
                        "resolved_match": match,
                        "upc12": match.upc12 if match else None,
                        "display_name": (match.display_name if match and match.display_name else cell.name).strip(),
                        "cpp_qty": match.cpp_qty if match else None,
                        "occupied": True,
                        "cell": cell,
                    }
                    slot_items.append(slot)
                    slots_flat.append(slot)

                rows_out.append(
                    {
                        "row_id": row_id,
                        "row_index": row_index_map[row_id],
                        "slots": slot_items,
                    }
                )

            return {
                "section_kind": section_kind,
                "row_ids": row_ids,
                "col_ids": col_ids,
                "rows": rows_out,
                "slots_flat": slots_flat,
            }

        return {
            "side_letter": p.side_letter,
            "main": _build_section("main", above_bonus_rows),
            "bonus": _build_section("bonus", below_bonus_rows),
        }

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

    def _section_shape_policy(section_kind: str) -> Dict[str, float]:
        if section_kind == "main":
            return {
                "desired_card_w": 64.0,
                "desired_gap": 6.0,
                "row_gutter": 8.0,
                "card_ratio": 1.04,   # h / w
                "min_card_h": 50.0,
                "max_card_h": 82.0,
                "crop_zoom": 2.40,
                "crop_inset": 0.018,
            }

        return {
            "desired_card_w": 64.0,
            "desired_gap": 6.0,
            "row_gutter": 8.0,
            "card_ratio": 1.04,   # h / w
            "min_card_h": 50.0,
            "max_card_h": 82.0,
            "crop_zoom": 2.40,
            "crop_inset": 0.018,
        }

    def _measure_section_shape(
        p: FullPalletPage,
        sec_rows: List[int],
        section_kind: str,
        content_w: float,
        global_cols: List[int],
        global_gap_units: List[int],
        include_bar: bool,
    ) -> Dict[str, object]:
        if not sec_rows:
            return {
                "rows": [],
                "n_rows": 0,
                "n_cols": 0,
                "sec_cols": [],
                "sec_gap_units": [],
                "xs": [],
                "card_w": 0.0,
                "card_h": 0.0,
                "row_gutter": 0.0,
                "right": 0.0,
                "overflow": False,
                "total_h": 0.0,
                "crop_zoom": 2.4,
                "crop_inset": 0.018,
            }

        policy = _section_shape_policy(section_kind)
        sec_rows_sorted = list(sec_rows)
        sec_cols, sec_gap_units = _section_cols_and_gaps(
            p, sec_rows_sorted, global_cols, global_gap_units
        )

        xs, card_w, _gap, right, overflow = _fit_x_layout(
            0.0,
            content_w,
            len(sec_cols),
            sec_gap_units,
            policy["desired_card_w"],
            policy["desired_gap"],
        )

        card_h = max(
            policy["min_card_h"],
            min(policy["max_card_h"], card_w * policy["card_ratio"]),
        )
        row_gutter = policy["row_gutter"]
        total_h = len(sec_rows_sorted) * card_h + max(0, len(sec_rows_sorted) - 1) * row_gutter
        if include_bar:
            total_h += SECTION_BAR_H + SECTION_BAR_GAP

        return {
            "rows": sec_rows_sorted,
            "n_rows": len(sec_rows_sorted),
            "n_cols": len(sec_cols),
            "sec_cols": sec_cols,
            "sec_gap_units": sec_gap_units,
            "xs": xs,
            "card_w": card_w,
            "card_h": card_h,
            "row_gutter": row_gutter,
            "right": right,
            "overflow": overflow,
            "total_h": total_h,
            "crop_zoom": policy["crop_zoom"],
            "crop_inset": policy["crop_inset"],
        }

    def _draw_shape_preserving_section(
        p: FullPalletPage,
        plan: Dict[str, object],
        sec_top: float,
        label: Optional[str],
        unresolved_bucket: List[str],
        content_x0: float,
        product_map: Dict[Tuple[int, int], Tuple[Optional[MatrixRow], CellData]],
        missing_image_slots: Optional[List[str]] = None,
    ) -> Tuple[int, int, bool, float, int, int]:
        nonlocal rightmost_used, matched_cells, unmatched_cells

        sec_rows_sorted = plan["rows"]
        if not sec_rows_sorted:
            return 0, 0, False, sec_top, 0, 0

        y_cursor = sec_top
        if label is not None:
            bar_y = y_cursor - SECTION_BAR_H
            _draw_section_bar(c, content_x0, bar_y, content_w, SECTION_BAR_H, label)
            y_cursor = bar_y - SECTION_BAR_GAP

        sec_cols = plan["sec_cols"]
        sec_col_rank = {c_: i for i, c_ in enumerate(sec_cols)}
        rmap = {r: i for i, r in enumerate(sec_rows_sorted)}
        row_set = set(sec_rows_sorted)

        occ: Dict[Tuple[int, int], CellData] = {}
        for cell in p.cells:
            if cell.row in row_set and cell.col in sec_col_rank:
                occ[(rmap[cell.row], sec_col_rank[cell.col])] = cell

        card_w = float(plan["card_w"])
        card_h = float(plan["card_h"])
        row_gutter = float(plan["row_gutter"])
        xs = [content_x0 + float(x) for x in plan["xs"]]
        rightmost_used = max(rightmost_used, content_x0 + float(plan["right"]))

        grid_top = y_cursor
        n_rows = int(plan["n_rows"])
        n_cols = int(plan["n_cols"])

        for ri in range(n_rows):
            y = grid_top - (ri + 1) * card_h - ri * row_gutter
            for ci in range(n_cols):
                x = xs[ci]
                cell = occ.get((ri, ci))

                if cell is None:
                    continue

                key = (cell.row, cell.col)
                match, _cell = product_map.get(key, (None, cell))
                last5_key = _to_last5(cell.last5)
                upc12 = match.upc12 if match else None
                cpp = match.cpp_qty if match else None
                disp_name = (match.display_name if match and match.display_name else cell.name).strip()

                if upc12:
                    upc_str = upc12
                else:
                    upc_str = f"LAST5 {last5_key}"
                    disp_name = f"UNRESOLVED {last5_key}"
                    unresolved_bucket.append(last5_key)

                try:
                    img = crop_image_cell(
                        images_doc,
                        p.page_index,
                        cell.bbox,
                        zoom=float(plan["crop_zoom"]),
                        inset=float(plan["crop_inset"]),
                    )
                except Exception:
                    img = None
                if img is None and missing_image_slots is not None:
                    missing_image_slots.append(f"r{cell.row}c{cell.col}")

                if match:
                    matched_cells += 1
                else:
                    unmatched_cells += 1

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE)
                c.setLineWidth(0.75)
                c.rect(x, y, card_w, card_h, stroke=1, fill=1)
                _draw_card(c, x, y, card_w, card_h, img, upc_str, disp_name, cpp)

        sec_bottom = grid_top - n_rows * card_h - max(0, n_rows - 1) * row_gutter

        return (
            n_cols,
            n_rows,
            bool(plan["overflow"]),
            sec_bottom,
            len([1 for k in occ.keys()]),
            len(sec_cols),
        )

    def _mid_band_shape_ok(section: Optional[FullPalletMidBandSection]) -> bool:
        if section is None:
            return False
        if not section.shape_valid:
            return False
        if section.slot_count != 24 or len(section.rows) != 3:
            return False
        if section.row_slot_counts != [8, 8, 8]:
            return False
        if section.row_block_grouping != [[2, 4, 2], [2, 4, 2], [2, 4, 2]]:
            return False
        for row in section.rows:
            if len(row.slots) != 8:
                return False
            slot_ordered = sorted(row.slots, key=lambda s: s.slot_in_row)
            for i, s in enumerate(slot_ordered):
                if i <= 1 and s.block_name != "left":
                    return False
                if 2 <= i <= 5 and s.block_name != "center":
                    return False
                if i >= 6 and s.block_name != "right":
                    return False
        return True

    def _measure_canonical_mid_band(content_w: float, include_bar: bool) -> Dict[str, float]:
        intra_gap = 6.0
        inter_gap = 14.0
        row_gutter = 8.0
        min_card_h = 50.0
        max_card_h = 82.0
        card_ratio = 1.04  # h / w
        crop_zoom = 2.40
        crop_inset = 0.018

        avail = max(1.0, content_w)
        card_w = (avail - (5.0 * intra_gap) - (2.0 * inter_gap)) / 8.0
        card_w = max(24.0, card_w)
        card_h = max(min_card_h, min(max_card_h, card_w * card_ratio))
        total_w = 8.0 * card_w + 5.0 * intra_gap + 2.0 * inter_gap
        overflow = total_w > (avail + 0.001)
        if overflow:
            card_w = max(24.0, (avail - (5.0 * intra_gap) - (2.0 * inter_gap)) / 8.0)
            total_w = 8.0 * card_w + 5.0 * intra_gap + 2.0 * inter_gap
            card_h = max(min_card_h, min(max_card_h, card_w * card_ratio))
            overflow = total_w > (avail + 0.001)

        total_h = 3.0 * card_h + 2.0 * row_gutter
        if include_bar:
            total_h += SECTION_BAR_H + SECTION_BAR_GAP
        return {
            "card_w": card_w,
            "card_h": card_h,
            "intra_gap": intra_gap,
            "inter_gap": inter_gap,
            "row_gutter": row_gutter,
            "total_w": total_w,
            "total_h": total_h,
            "overflow": overflow,
            "crop_zoom": crop_zoom,
            "crop_inset": crop_inset,
        }

    def _resolve_mid_band_slot(slot: FullPalletMidBandSlot) -> Optional[MatrixRow]:
        primary = resolve_full_pallet(slot.last5, slot.parsed_name, matrix_idx) if slot.last5 else None
        if primary is not None:
            return primary

        # Secondary recovery: name-only best candidate when last5 is missing/unusable.
        label = (slot.parsed_name or slot.raw_label_text or "").strip()
        if not label or not all_matrix_rows:
            return None

        target = _norm_name(label)
        best: Optional[Tuple[float, MatrixRow]] = None
        for row in all_matrix_rows:
            ratio = difflib.SequenceMatcher(None, target, row.norm_name).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, row)
        if best is None:
            return None
        return best[1] if best[0] >= 0.78 else None

    def _draw_canonical_mid_band_section(
        p: FullPalletPage,
        section: FullPalletMidBandSection,
        plan: Dict[str, float],
        sec_top: float,
        unresolved_bucket: List[str],
        missing_image_slots: List[str],
        content_x0: float,
    ) -> Tuple[int, int, bool, float, int, int]:
        nonlocal rightmost_used, matched_cells, unmatched_cells

        y_cursor = sec_top
        if len(section.rows) == 0:
            return 0, 0, False, sec_top, 0, 0

        card_w = float(plan["card_w"])
        card_h = float(plan["card_h"])
        row_gutter = float(plan["row_gutter"])
        intra_gap = float(plan["intra_gap"])
        inter_gap = float(plan["inter_gap"])
        total_w = float(plan["total_w"])

        start_x = content_x0 + max(0.0, (content_w - total_w) / 2.0)
        row_xs = [
            start_x,
            start_x + card_w + intra_gap,
            start_x + 2 * card_w + intra_gap + inter_gap,
            start_x + 3 * card_w + 2 * intra_gap + inter_gap,
            start_x + 4 * card_w + 3 * intra_gap + inter_gap,
            start_x + 5 * card_w + 4 * intra_gap + inter_gap,
            start_x + 6 * card_w + 5 * intra_gap + 2 * inter_gap,
            start_x + 7 * card_w + 6 * intra_gap + 2 * inter_gap,
        ]
        rightmost_used = max(rightmost_used, start_x + total_w)

        grid_top = y_cursor
        slots_drawn = 0
        overflow = bool(plan["overflow"])

        for row in sorted(section.rows, key=lambda r: r.row_index):
            y = grid_top - (row.row_index + 1) * card_h - row.row_index * row_gutter
            ordered_slots: List[FullPalletMidBandSlot] = sorted(row.slots, key=lambda s: s.slot_in_row)

            for slot in ordered_slots:
                x = row_xs[slot.slot_in_row]

                match = _resolve_mid_band_slot(slot)
                upc12 = match.upc12 if match else None
                cpp = match.cpp_qty if match else None
                disp_name = (
                    match.display_name
                    if match and match.display_name
                    else (slot.parsed_name or slot.raw_label_text or "").strip()
                )

                if upc12:
                    upc_str = upc12
                    matched_cells += 1
                else:
                    last5_key = _to_last5(slot.last5)
                    upc_str = f"LAST5 {last5_key}" if last5_key else "LAST5 MISSING"
                    disp_name = f"UNRESOLVED {last5_key}" if last5_key else "UNRESOLVED"
                    unresolved_bucket.append(last5_key or slot.slot_id)
                    unmatched_cells += 1

                try:
                    img = crop_image_cell(
                        images_doc,
                        p.page_index,
                        slot.bbox,
                        zoom=float(plan["crop_zoom"]),
                        inset=float(plan["crop_inset"]),
                    )
                except Exception:
                    img = None

                if img is None:
                    missing_image_slots.append(slot.slot_id)

                if x < content_x0 - 0.001 or (x + card_w) > (content_x0 + content_w + 0.001):
                    overflow = True

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE)
                c.setLineWidth(0.75)
                c.rect(x, y, card_w, card_h, stroke=1, fill=1)
                _draw_card(c, x, y, card_w, card_h, img, upc_str, disp_name, cpp)

                if debug_overlay:
                    c.setFillColorRGB(0.35, 0.35, 0.35)
                    c.setFont("Helvetica", 6)
                    c.drawString(x + 2, y + card_h - 8, slot.slot_id)

                slots_drawn += 1

        sec_bottom = grid_top - 3 * card_h - 2 * row_gutter
        return 8, 3, overflow, sec_bottom, slots_drawn, 8

    def _draw_mid_band_placeholder_section(
        plan: Dict[str, float],
        sec_top: float,
        unresolved_bucket: List[str],
        content_x0: float,
    ) -> Tuple[int, int, bool, float, int, int]:
        nonlocal rightmost_used, unmatched_cells

        card_w = float(plan["card_w"])
        card_h = float(plan["card_h"])
        row_gutter = float(plan["row_gutter"])
        intra_gap = float(plan["intra_gap"])
        inter_gap = float(plan["inter_gap"])
        total_w = float(plan["total_w"])

        start_x = content_x0 + max(0.0, (content_w - total_w) / 2.0)
        row_xs = [
            start_x,
            start_x + card_w + intra_gap,
            start_x + 2 * card_w + intra_gap + inter_gap,
            start_x + 3 * card_w + 2 * intra_gap + inter_gap,
            start_x + 4 * card_w + 3 * intra_gap + inter_gap,
            start_x + 5 * card_w + 4 * intra_gap + inter_gap,
            start_x + 6 * card_w + 5 * intra_gap + 2 * inter_gap,
            start_x + 7 * card_w + 6 * intra_gap + 2 * inter_gap,
        ]
        rightmost_used = max(rightmost_used, start_x + total_w)

        grid_top = sec_top
        overflow = bool(plan["overflow"])
        for ri in range(3):
            y = grid_top - (ri + 1) * card_h - ri * row_gutter
            for ci in range(8):
                x = row_xs[ci]
                if x < content_x0 - 0.001 or (x + card_w) > (content_x0 + content_w + 0.001):
                    overflow = True
                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*EMPTY_STROKE)
                c.setLineWidth(0.6)
                c.rect(x, y, card_w, card_h, stroke=1, fill=1)
                sid = f"MB-R{ri + 1}-S{ci + 1}"
                unresolved_bucket.append(sid)
                if debug_overlay:
                    c.setFillColorRGB(0.55, 0.25, 0.25)
                    c.setFont("Helvetica", 6)
                    c.drawString(x + 2, y + card_h - 8, sid)

        unmatched_cells += 24
        sec_bottom = grid_top - 3 * card_h - 2 * row_gutter
        return 8, 3, overflow, sec_bottom, 24, 8
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
            c = canvas.Canvas(buf, pagesize=(PAGE_W, BASE_PAGE_H))
            c.save()
            return buf.getvalue()

        c = canvas.Canvas(buf, pagesize=(PAGE_W, BASE_PAGE_H))

        def _wrap_text_lines(
            text: str,
            font: str,
            size: float,
            max_w: float,
            max_lines: int,
        ) -> List[str]:
            words = [w for w in (text or "").strip().split() if w]
            if not words:
                return []

            lines_out: List[str] = []
            cur = words[0]

            for word in words[1:]:
                trial = f"{cur} {word}"
                if pdfmetrics.stringWidth(trial, font, size) <= max_w:
                    cur = trial
                else:
                    lines_out.append(cur)
                    cur = word
                    if len(lines_out) == max_lines - 1:
                        break

            if len(lines_out) < max_lines:
                remaining_words = words[len(" ".join(lines_out).split()) :]
                remaining = " ".join(remaining_words).strip()
                if remaining:
                    remaining = _ellipsis(remaining, font, size, max_w)
                    lines_out.append(remaining)

            return lines_out[:max_lines]

        def _draw_non_ppt_card(
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
            pad = max(4.0, min(8.0, w * 0.05))
            ix = x + pad
            iy = y + pad
            iw = max(8.0, w - 2 * pad)
            ih = max(8.0, h - 2 * pad)

            footer_h = max(14.0, min(18.0, ih * 0.16))
            title_h = max(18.0, min(24.0, ih * 0.20))
            image_h = max(12.0, ih - footer_h - title_h - 4.0)

            image_y = iy + footer_h
            title_y = image_y + image_h + 4.0

            if img is not None and iw > 8 and image_h > 8:
                sw, sh = img.size
                r = min(iw / max(1, sw), image_h / max(1, sh))
                dw, dh = sw * r, sh * r
                c.drawImage(
                    ImageReader(img),
                    ix + (iw - dw) / 2,
                    image_y + (image_h - dh) / 2,
                    dw,
                    dh,
                    preserveAspectRatio=True,
                    mask="auto",
                )

            title_font = 7.5 if w < 96 else 8.0
            title_text = _fit_name_preserve_qualifiers((name or "").upper(), BODY_FONT, title_font, iw)
            title_lines = _wrap_text_lines(title_text, BODY_FONT, title_font, iw, 2)

            c.setFillColorRGB(0.10, 0.10, 0.10)
            c.setFont(BODY_FONT, title_font)
            if len(title_lines) == 1:
                tw = pdfmetrics.stringWidth(title_lines[0], BODY_FONT, title_font)
                c.drawString(ix + (iw - tw) / 2, title_y + (title_h - title_font) / 2 - 1, title_lines[0])
            else:
                line_gap = 1.5
                total_text_h = title_font * 2 + line_gap
                y1 = title_y + (title_h - total_text_h) / 2 + title_font + line_gap
                y2 = y1 - title_font - line_gap
                for yy, line in [(y1, title_lines[0]), (y2, title_lines[1])]:
                    tw = pdfmetrics.stringWidth(line, BODY_FONT, title_font)
                    c.drawString(ix + (iw - tw) / 2, yy, line)

            footer_y = iy
            upc_text = (upc12 or "").strip()
            cpp_text = f"CPP: {cpp}" if cpp is not None else "CPP:"

            upc_fs = 7.6 if w >= 96 else 7.1
            cpp_fs = 7.4 if w >= 96 else 7.0

            upc_text = _ellipsis(upc_text, BODY_BOLD_FONT, upc_fs, iw)
            cpp_text = _ellipsis(cpp_text, BODY_BOLD_FONT, cpp_fs, iw)

            c.setFillColorRGB(0.05, 0.05, 0.05)
            c.setFont(BODY_BOLD_FONT, upc_fs)
            upc_tw = pdfmetrics.stringWidth(upc_text, BODY_BOLD_FONT, upc_fs)
            c.drawString(ix + (iw - upc_tw) / 2, footer_y + footer_h * 0.56, upc_text)

            c.setFillColorRGB(0.15, 0.15, 0.15)
            c.setFont(BODY_BOLD_FONT, cpp_fs)
            cpp_tw = pdfmetrics.stringWidth(cpp_text, BODY_BOLD_FONT, cpp_fs)
            c.drawString(ix + (iw - cpp_tw) / 2, footer_y + 1.5, cpp_text)

            

        def _flow_policy(section_kind: str) -> Dict[str, float]:
            if section_kind == "main":
                return {
                    "min_cols": 5.0,
                    "max_cols": 8.0,
                    "min_card_w": 78.0,
                    "max_card_w": 112.0,
                    "card_ratio": 1.02,
                    "min_card_h": 88.0,
                    "max_card_h": 122.0,
                    "col_gap": 10.0,
                    "row_gap": 10.0,
                    "crop_zoom": 2.9,
                    "crop_inset": 0.020,
                }

            return {
                "min_cols": 3.0,
                "max_cols": 5.0,
                "min_card_w": 88.0,
                "max_card_w": 124.0,
                "card_ratio": 1.06,
                "min_card_h": 94.0,
                "max_card_h": 132.0,
                "col_gap": 12.0,
                "row_gap": 12.0,
                "crop_zoom": 2.6,
                "crop_inset": 0.018,
            }

        def _collect_section_items(
            p: FullPalletPage,
            sec_rows: List[int],
            section_kind: str,
            unresolved_bucket: List[str],
            global_rows: List[int],
            global_cols: List[int],
            product_map: Dict[Tuple[int, int], Tuple[Optional[MatrixRow], CellData]],
        ) -> Tuple[List[dict], int, int]:

            if not sec_rows:
                return []

            row_set = set(sec_rows)
            row_rank = {r: i for i, r in enumerate(global_rows)}
            col_rank = {c_: i for i, c_ in enumerate(global_cols)}
            policy = _flow_policy(section_kind)

            sec_cells = [cell for cell in p.cells if cell.row in row_set]
            sec_cells.sort(
                key=lambda cell: (
                    row_rank.get(cell.row, 10**6 + cell.row),
                    col_rank.get(cell.col, 10**6 + cell.col),
                    cell.row,
                    cell.col,
                )
            )

            items: List[dict] = []
            matched = 0
            unmatched = 0

            for cell in sec_cells:
                key = (cell.row, cell.col)
                match, _cell = product_map.get(key, (None, cell))
                last5_key = _to_last5(cell.last5)
                upc12 = match.upc12 if match else None
                cpp = match.cpp_qty if match else None
                disp_name = (match.display_name if match and match.display_name else cell.name).strip()

                if upc12:
                    upc_str = upc12
                else:
                    upc_str = f"LAST5 {last5_key}"
                    disp_name = f"UNRESOLVED {last5_key}"
                    unresolved_bucket.append(last5_key)

                try:
                    img = crop_image_cell(
                        images_doc,
                        p.page_index,
                        cell.bbox,
                        zoom=policy["crop_zoom"],
                        inset=policy["crop_inset"],
                    )
                except Exception:
                    img = None

                if match:
                    matched += 1
                else:
                    unmatched += 1

                items.append(
                    {
                        "cell": cell,
                        "img": img,
                        "upc": upc_str,
                        "name": disp_name,
                        "cpp": cpp,
                    }
                )

            return items, matched, unmatched

        def _section_layout_candidates(
            item_count: int,
            section_kind: str,
            avail_w: float,
            include_bar: bool,
        ) -> List[dict]:
            policy = _flow_policy(section_kind)

            if item_count <= 0:
                return [
                    {
                        "cols": 0,
                        "rows": 0,
                        "card_w": 0.0,
                        "card_h": 0.0,
                        "total_h": 0.0,
                        "block_w": 0.0,
                        "score": 0.0,
                    }
                ]

            min_cols = int(policy["min_cols"])
            max_cols = min(int(policy["max_cols"]), item_count)

            if item_count < min_cols:
                min_cols = item_count

            candidates: List[dict] = []
            for cols in range(min_cols, max_cols + 1):
                raw_card_w = (avail_w - (cols - 1) * policy["col_gap"]) / cols
                if raw_card_w < policy["min_card_w"]:
                    continue

                card_w = min(policy["max_card_w"], raw_card_w)
                card_h = min(
                    policy["max_card_h"],
                    max(policy["min_card_h"], card_w * policy["card_ratio"]),
                )
                rows = int(math.ceil(item_count / cols))
                block_w = cols * card_w + (cols - 1) * policy["col_gap"]
                total_h = rows * card_h + max(0, rows - 1) * policy["row_gap"]
                if include_bar:
                    total_h += SECTION_BAR_H + SECTION_BAR_GAP

                col_penalty = 2.0 * cols if section_kind == "main" else 4.0 * cols
                score = (card_w * card_h) - col_penalty

                candidates.append(
                    {
                        "cols": cols,
                        "rows": rows,
                        "card_w": card_w,
                        "card_h": card_h,
                        "total_h": total_h,
                        "block_w": block_w,
                        "score": score,
                    }
                )

            if not candidates:
                cols = max(1, min(item_count, max_cols))
                card_w = max(60.0, (avail_w - max(0, cols - 1) * policy["col_gap"]) / cols)
                rows = int(math.ceil(item_count / cols))
                card_h = max(72.0, card_w * 0.92)
                block_w = cols * card_w + max(0, cols - 1) * policy["col_gap"]
                total_h = rows * card_h + max(0, rows - 1) * policy["row_gap"]
                if include_bar:
                    total_h += SECTION_BAR_H + SECTION_BAR_GAP

                candidates.append(
                    {
                        "cols": cols,
                        "rows": rows,
                        "card_w": card_w,
                        "card_h": card_h,
                        "total_h": total_h,
                        "block_w": block_w,
                        "score": card_w * card_h,
                    }
                )

            return candidates

        def _choose_product_layout(
            item_count: int,
            section_kind: str,
            avail_w: float,
            include_bar: bool,
        ) -> dict:
            candidates = _section_layout_candidates(item_count, section_kind, avail_w, include_bar)
            return max(candidates, key=lambda d: (d["score"], -d["cols"], -d["rows"]))

        def _choose_product_layouts(
            main_count: int,
            bonus_count: int,
            avail_w: float,
            avail_h: float,
        ) -> Tuple[dict, dict]:
            main_candidates = _section_layout_candidates(main_count, "main", avail_w, include_bar=False)
            bonus_candidates = _section_layout_candidates(bonus_count, "bonus", avail_w, include_bar=bonus_count > 0)

            best_fit: Optional[Tuple[dict, dict, float]] = None
            best_any: Optional[Tuple[dict, dict, float, float]] = None

            for main_plan in main_candidates:
                for bonus_plan in bonus_candidates:
                    inter_gap = BUCKET_GAP if main_count > 0 and bonus_count > 0 else 0.0
                    total_h = main_plan["total_h"] + inter_gap + bonus_plan["total_h"]
                    overflow = max(0.0, total_h - avail_h)
                    score = main_plan["score"] + bonus_plan["score"]

                    if overflow <= 0.0:
                        if best_fit is None or score > best_fit[2]:
                            best_fit = (main_plan, bonus_plan, score)
                    else:
                        if best_any is None or overflow < best_any[2] or (
                            abs(overflow - best_any[2]) < 0.001 and score > best_any[3]
                        ):
                            best_any = (main_plan, bonus_plan, overflow, score)

            if best_fit is not None:
                return best_fit[0], best_fit[1]
            if best_any is not None:
                return best_any[0], best_any[1]

            return main_candidates[0], bonus_candidates[0]

        def _draw_flow_section(
            items: List[dict],
            section_kind: str,
            plan: dict,
            top_y: float,
            bottom_limit: float,
            label: Optional[str],
        ) -> Tuple[float, float, int, int, bool]:
            nonlocal rightmost_used, adjusted_to_fit

            if not items:
                return top_y, top_y, 0, 0, False

            policy = _flow_policy(section_kind)
            y_cursor = top_y

            if label:
                bar_y = y_cursor - SECTION_BAR_H
                _draw_section_bar(c, cx0, bar_y, content_w, SECTION_BAR_H, label)
                y_cursor = bar_y - SECTION_BAR_GAP

            cols = int(plan["cols"])
            rows = int(plan["rows"])
            card_w = float(plan["card_w"])
            card_h = float(plan["card_h"])
            col_gap = float(policy["col_gap"])
            row_gap = float(policy["row_gap"])
            block_w = float(plan["block_w"])
            start_x = cx0 + max(0.0, (content_w - block_w) / 2.0)

            bottom_y_used = y_cursor
            for idx, item in enumerate(items):
                ri = idx // cols
                ci = idx % cols
                x = start_x + ci * (card_w + col_gap)
                y = y_cursor - (ri + 1) * card_h - ri * row_gap

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE)
                c.setLineWidth(0.75)
                c.rect(x, y, card_w, card_h, stroke=1, fill=1)
                _draw_non_ppt_card(
                    c,
                    x,
                    y,
                    card_w,
                    card_h,
                    item["img"],
                    item["upc"],
                    item["name"],
                    item["cpp"],
                )

                rightmost_used = max(rightmost_used, x + card_w)
                bottom_y_used = min(bottom_y_used, y)

            overflow = bottom_y_used < bottom_limit - 0.001
            adjusted_to_fit = adjusted_to_fit or overflow

            return top_y, bottom_y_used, cols, rows, overflow

        for pdata in pages:
            cx0, cx1 = MARGIN, PAGE_W - MARGIN
            content_w = cx1 - cx0

            global_cols, global_gap_units = _global_col_order_and_gaps(pdata)
            global_rows = _global_row_order(pdata)
            above_bonus_rows, below_bonus_rows = _split_rows_by_bonus(pdata, global_rows)
            mid_band = pdata.mid_band_above_bonus
            canonical_mid_band_ok = _mid_band_shape_ok(mid_band)

            if debug and mid_band is not None:
                mid_slots_dbg = [s for r in mid_band.rows for s in r.slots]
                st.write(
                    {
                        "side": pdata.side_letter,
                        "mid_band_extract_debug": {
                            "extract_path": "template_slots",
                            "whitelist_only": True,
                            "present": True,
                            "shape_valid": mid_band.shape_valid,
                            "anchor_bbox": list(mid_band.anchor_bbox) if mid_band.anchor_bbox else None,
                            "slot_count": mid_band.slot_count,
                            "row_slot_counts": mid_band.row_slot_counts,
                            "row_block_grouping": mid_band.row_block_grouping,
                            "slot_bboxes": {s.slot_id: [round(v, 2) for v in s.bbox] for s in mid_slots_dbg},
                            "slot_extraction_bboxes": {
                                s.slot_id: ([round(v, 2) for v in s.extraction_bbox] if s.extraction_bbox else None)
                                for s in mid_slots_dbg
                            },
                            "slot_accepted_words": {s.slot_id: (s.accepted_words or []) for s in mid_slots_dbg},
                            "slot_rejected_nearby_count": {
                                s.slot_id: int(s.rejected_nearby_word_count or 0) for s in mid_slots_dbg
                            },
                        },
                    }
                )

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

            slot_map = _build_non_ppt_slot_map_for_side(
                pdata,
                global_rows,
                global_cols,
                above_bonus_rows,
                below_bonus_rows,
                product_map,
            )

            canonical_main_plan = _measure_canonical_mid_band(content_w, include_bar=False)
            bonus_plan = _measure_section_shape(
                pdata,
                slot_map["bonus"]["row_ids"],
                "bonus",
                content_w,
                global_cols,
                global_gap_units,
                include_bar=bool(slot_map["bonus"]["row_ids"]),
            )

            products_block_h = float(canonical_main_plan["total_h"])
            if below_bonus_rows:
                products_block_h += BUCKET_GAP
            products_block_h += float(bonus_plan["total_h"])

            required_content_h = (
                PPT_SECTION_H
                + BUCKET_GAP
                + HOLDER_SECTION_H
                + BUCKET_GAP
                + products_block_h
            )
            PAGE_H = max(BASE_PAGE_H, (2 * MARGIN) + HEADER_H + FOOTER_H + required_content_h)

            c.setPageSize((PAGE_W, PAGE_H))

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

            cy0, cy1 = MARGIN + FOOTER_H, PAGE_H - MARGIN - HEADER_H
            content_h = cy1 - cy0

            bucket_a_h = PPT_SECTION_H
            bucket_b_h = HOLDER_SECTION_H

            bucket_a_top = cy1
            bucket_a_bottom = bucket_a_top - bucket_a_h
            bucket_b_top = bucket_a_bottom - BUCKET_GAP
            bucket_b_bottom = bucket_b_top - bucket_b_h
            products_top = bucket_b_bottom - BUCKET_GAP
            products_bottom = cy0

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
            missing_main_images: List[str] = []
            missing_bonus_images: List[str] = []
            main_render_source = "mid_band_template" if canonical_mid_band_ok else "mid_band_template_placeholder"
            if debug and not canonical_mid_band_ok:
                st.write(
                    {
                        "side": pdata.side_letter,
                        "canonical_mid_band_failure": {
                            "present": bool(mid_band),
                            "shape_valid": bool(mid_band.shape_valid) if mid_band else False,
                            "anchor_bbox": list(mid_band.anchor_bbox) if (mid_band and mid_band.anchor_bbox) else None,
                            "slot_count": mid_band.slot_count if mid_band else None,
                            "row_slot_counts": mid_band.row_slot_counts if mid_band else None,
                            "row_block_grouping": mid_band.row_block_grouping if mid_band else None,
                            "fallback": "disabled_for_mid_band",
                            "render_mode": "mid_band_template_placeholder",
                        },
                    }
                )

            # Bucket A: PPT Top Cards
            a_gap_y = 8.0
            top_card_h = max(30.0, min(86.0, bucket_a_h * 0.34))
            # Desired card width spans full content width (8 equal slots with gaps).
            _desired_top_card_w = max(72.0, (content_w - 7 * 6.0) / 8)
            top_xs, top_card_w, _, top_right, top_overflow = _fit_x_layout(
                cx0, cx1, 8, [1] * 7, _desired_top_card_w, 6.0
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
                        image_from_bytes(card.image_bytes),
                        f"ID# {card.card_id}", card.title, ppt_cpp_global,
                    )

            # Side panel: 3 columns × 2 rows, right-aligned.
            # Correct formula: block must be wide enough for 3 cards + 2 gaps.
            _side_gap = 6.0
            _side_cols = 3
            side_block_w = _side_cols * top_card_w + (_side_cols - 1) * _side_gap
            sx0 = max(cx0, cx1 - side_block_w)
            side_xs = [sx0 + i * (top_card_w + _side_gap) for i in range(_side_cols)]
            side_top = top_row_y - SECTION_BAR_GAP
            side_available_h = max(36.0, side_top - bucket_a_bottom)
            side_card_h = max(20.0, min(top_card_h, (side_available_h - a_gap_y) / 2))
            # 3 cols × 2 rows, row-major: indices 0-5
            for row in range(2):
                y = side_top - (row + 1) * side_card_h - row * a_gap_y
                for col in range(_side_cols):
                    idx = row * _side_cols + col
                    x = side_xs[col]
                    card = side_ppt.side6[idx] if idx < len(side_ppt.side6) else None
                    c.setFillColorRGB(1, 1, 1)
                    c.setStrokeColorRGB(*FILLED_STROKE if card else EMPTY_STROKE)
                    c.setLineWidth(0.75 if card else 0.45)
                    c.rect(x, y, top_card_w, side_card_h, stroke=1, fill=1)
                    if card:
                        _draw_card(
                            c, x, y, top_card_w, side_card_h,
                            image_from_bytes(card.image_bytes),
                            f"ID# {card.card_id}", card.title, ppt_cpp_global,
                        )
                    rightmost_used = max(rightmost_used, x + top_card_w)

            if debug_overlay:
                _draw_debug_box(c, cx0, bucket_a_bottom, content_w, bucket_a_top - bucket_a_bottom, "PPT")

            # Bucket B: Gift Card Holders
            holder_bar_y = bucket_b_top - SECTION_BAR_H
            _draw_section_bar(c, cx0, holder_bar_y, content_w, SECTION_BAR_H, "GIFT CARD HOLDERS")
            holder_grid_top = holder_bar_y - SECTION_BAR_GAP
            holder_grid_h = max(24.0, holder_grid_top - bucket_b_bottom)
            fixed_holder_slots = 5
            holder_gap_x = 8.0
            holder_card_w_max = (content_w - holder_gap_x * (fixed_holder_slots - 1)) / fixed_holder_slots
            holder_card_w = max(54.0, min(holder_card_w_max, top_card_w * 1.10))
            holder_card_h = max(34.0, min(holder_grid_h, top_card_h * 1.08))
            holder_row_w = fixed_holder_slots * holder_card_w + (fixed_holder_slots - 1) * holder_gap_x
            holder_x0 = cx0 + max(0.0, (content_w - holder_row_w) / 2.0)
            holder_y = holder_grid_top - holder_card_h - max(0.0, (holder_grid_h - holder_card_h) / 2.0)

            for idx in range(fixed_holder_slots):
                x = holder_x0 + idx * (holder_card_w + holder_gap_x)
                holder = side_holders[idx] if idx < len(side_holders) else None

                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(*FILLED_STROKE)
                c.setLineWidth(0.75)
                c.rect(x, holder_y, holder_card_w, holder_card_h, stroke=1, fill=1)

                if holder is not None:
                    _draw_card(
                        c,
                        x,
                        holder_y,
                        holder_card_w,
                        holder_card_h,
                        image_from_bytes(holder.image_bytes),
                        holder.item_no,
                        holder.name,
                        holder.qty,
                    )

                rightmost_used = max(rightmost_used, x + holder_card_w)

            if debug_overlay:
                _draw_debug_box(c, cx0, bucket_b_bottom, content_w, bucket_b_top - bucket_b_bottom, "HOLDERS")

            main_cols = 0
            main_rows_count = 0
            main_over = False
            main_bottom = products_top

            if canonical_mid_band_ok and mid_band is not None:
                (
                    main_cols,
                    main_rows_count,
                    main_over,
                    main_bottom,
                    _main_occ_count,
                    _main_sec_cols,
                ) = _draw_canonical_mid_band_section(
                    pdata,
                    mid_band,
                    canonical_main_plan,
                    products_top,
                    unresolved_main,
                    missing_main_images,
                    cx0,
                )
            else:
                (
                    main_cols,
                    main_rows_count,
                    main_over,
                    main_bottom,
                    _main_occ_count,
                    _main_sec_cols,
                ) = _draw_mid_band_placeholder_section(
                    canonical_main_plan,
                    products_top,
                    unresolved_main,
                    cx0,
                )

            bonus_top = main_bottom - (
                BUCKET_GAP if (main_rows_count > 0 and slot_map["bonus"]["row_ids"]) else 0.0
            )
            bonus_cols = 0
            bonus_rows_count = 0
            bonus_over = False
            bonus_bottom = bonus_top

            if slot_map["bonus"]["row_ids"]:
                (
                    bonus_cols,
                    bonus_rows_count,
                    bonus_over,
                    bonus_bottom,
                    _bonus_occ_count,
                    _bonus_sec_cols,
                ) = _draw_shape_preserving_section(
                    pdata,
                    bonus_plan,
                    bonus_top,
                    "BONUS",
                    unresolved_bonus,
                    cx0,
                    product_map,
                    missing_bonus_images,
                )

            adjusted_to_fit = adjusted_to_fit or main_over or bonus_over

            if debug_overlay:
                _draw_debug_box(c, cx0, products_bottom, content_w, products_top - products_bottom, "PRODUCTS")
                if main_rows_count > 0:
                    _draw_debug_box(
                        c,
                        cx0,
                        main_bottom,
                        content_w,
                        products_top - main_bottom,
                        "MAIN TEMPLATE",
                    )
                if below_bonus_rows:
                    _draw_debug_box(
                        c,
                        cx0,
                        bonus_bottom,
                        content_w,
                        bonus_top - bonus_bottom,
                        "BONUS SHAPE",
                    )

            right_limit = cx1
            exceeded = rightmost_used > right_limit + 0.001
            lowest_bottom = bonus_bottom if slot_map["bonus"]["row_ids"] else main_bottom
            vertical_overflow = lowest_bottom < products_bottom - 0.001
            adjusted_to_fit = adjusted_to_fit or exceeded or vertical_overflow

            if canonical_mid_band_ok and mid_band is not None:
                main_slots = [s for r in mid_band.rows for s in r.slots]
                main_last5_codes = sorted({_to_last5(s.last5) for s in main_slots if _to_last5(s.last5)})
                main_slots_total = len(main_slots)
                canonical_row_counts = [len(r.slots) for r in mid_band.rows]
                canonical_block_counts = mid_band.row_block_grouping
                anchor_bbox = list(mid_band.anchor_bbox) if mid_band.anchor_bbox else None
                suspicious_patterns = ("WM", "PKG", "GCI", "MARKETING", "FRAUD", "BONUS")
                suspicious_slots = [
                    s.slot_id
                    for s in main_slots
                    if any(p in (s.raw_label_text or "").upper() for p in suspicious_patterns)
                ]
                suspicious_top_row_slots = [
                    s.slot_id
                    for s in main_slots
                    if s.row_index == 0 and any(p in (s.raw_label_text or "").upper() for p in suspicious_patterns)
                ]
            else:
                main_last5_codes = []
                main_slots_total = 24
                canonical_row_counts = [8, 8, 8]
                canonical_block_counts = [[2, 4, 2], [2, 4, 2], [2, 4, 2]]
                anchor_bbox = None
                suspicious_slots = []
                suspicious_top_row_slots = []
                main_slots = []

            bonus_last5_codes = sorted({_to_last5(c.last5) for c in pdata.cells if c.row in set(below_bonus_rows)})
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
                            "products_available": [round(products_top, 1), round(products_bottom, 1)],
                            "main_used": [round(products_top, 1), round(main_bottom, 1)] if main_rows_count > 0 else None,
                            "bonus_used": [round(bonus_top, 1), round(bonus_bottom, 1)] if below_bonus_rows else None,
                        },
                        "mid_band_above_bonus": {
                            "render_source": main_render_source,
                            "path": "deterministic_template",
                            "whitelist_only": True,
                            "present": bool(mid_band),
                            "shape_valid": bool(mid_band.shape_valid) if mid_band else False,
                            "shape_matches_3x8_242": canonical_mid_band_ok,
                            "anchor_bbox": anchor_bbox,
                            "row_slot_counts": canonical_row_counts,
                            "row_block_grouping": canonical_block_counts,
                            "suspicious_slot_text_ids": sorted(set(suspicious_slots)),
                            "suspicious_top_row_ids": sorted(set(suspicious_top_row_slots)),
                            "slot_local_text": {s.slot_id: s.raw_label_text for s in main_slots},
                            "slot_accepted_words": {s.slot_id: (s.accepted_words or []) for s in main_slots},
                            "slot_rejected_nearby_count": {
                                s.slot_id: int(s.rejected_nearby_word_count or 0) for s in main_slots
                            },
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
                            "vertical_overflow": vertical_overflow,
                            "adjusted_to_fit": adjusted_to_fit,
                        },
                        "slot_map_summary": {
                            "main_rows": [len(r["slots"]) for r in slot_map["main"]["rows"]],
                            "bonus_rows": [len(r["slots"]) for r in slot_map["bonus"]["rows"]],
                            "main_total_slots": len(slot_map["main"]["slots_flat"]),
                            "bonus_total_slots": len(slot_map["bonus"]["slots_flat"]),
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
                            "slot_ranges": [
                                {
                                    "item_no": h.item_no,
                                    "slot_order": h.slot_order,
                                    "slot_label": h.slot_label,
                                    "start_col": h.slot_start_col,
                                    "end_col": h.slot_end_col,
                                }
                                for h in side_holders
                            ],
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
                        "missing_image_crops": {
                            "main": sorted({x for x in missing_main_images if x}),
                            "bonus": sorted({x for x in missing_bonus_images if x}),
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
