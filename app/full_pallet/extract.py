from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pdfplumber

from app.shared.clustering import boundaries_from_centers, cluster_positions
from app.shared.models import (
    AnnotationBox,
    CellData,
    FullPalletMidBandRow,
    FullPalletMidBandSection,
    FullPalletMidBandSlot,
    FullPalletPage,
)
from app.shared.pdf_utils import _group_nearby, _union
from app.shared.text_utils import _to_last5
from app.standard_display.extract import parse_label_cell_text


def _wt(w: dict) -> str:
    return str(w.get("text", "")).strip().upper()


def _word_center(w: dict) -> Tuple[float, float]:
    return (float(w["x0"]) + float(w["x1"])) / 2.0, (float(w["top"]) + float(w["bottom"])) / 2.0


def _bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_area(b: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _normalize_bbox(obj: dict) -> Optional[Tuple[float, float, float, float]]:
    x0 = float(obj.get("x0", 0.0))
    x1 = float(obj.get("x1", 0.0))
    top = float(obj.get("top", 0.0))
    bottom = float(obj.get("bottom", 0.0))
    if x1 <= x0 or bottom <= top:
        return None
    return (x0, top, x1, bottom)


def _contains_point(b: Tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = b
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _pick_row_window(groups: List[dict], bonus_y: Optional[float]) -> List[dict]:
    if len(groups) < 3:
        return []

    eligible: List[List[dict]] = []
    for i in range(len(groups) - 2):
        window = groups[i : i + 3]
        ys = [float(g["y"]) for g in window]
        counts = [len(g["rects"]) for g in window]

        if bonus_y is not None and any(y >= bonus_y for y in ys):
            continue
        if min(counts) < 8:
            continue

        eligible.append(window)

    if eligible:
        def score(window: List[dict]) -> Tuple[float, float, float]:
            ys = [float(g["y"]) for g in window]
            counts = [len(g["rects"]) for g in window]
            bottom_dist = (bonus_y - ys[-1]) if bonus_y is not None else -ys[-1]
            gaps = [ys[1] - ys[0], ys[2] - ys[1]]
            gap_std = float(np.std(gaps)) if len(gaps) == 2 else 999.0
            count_penalty = float(sum(abs(c - 8) for c in counts))
            return (bottom_dist, gap_std, count_penalty)

        return min(eligible, key=score)

    best: Optional[Tuple[float, List[dict]]] = None
    for i in range(len(groups) - 2):
        window = groups[i : i + 3]
        counts = [len(g["rects"]) for g in window]
        ys = [float(g["y"]) for g in window]

        score = float(sum(abs(c - 8) * 20.0 for c in counts))
        if any(c < 8 for c in counts):
            score += 1000.0

        if bonus_y is not None:
            for y in ys:
                if y >= bonus_y:
                    score += 1000.0
            score += max(0.0, bonus_y - ys[-1]) * 0.01

        gaps = [ys[1] - ys[0], ys[2] - ys[1]]
        if min(gaps) <= 0:
            score += 500.0
        else:
            score += float(np.std(gaps)) * 4.0

        if best is None or score < best[0]:
            best = (score, window)

    return best[1] if best else []


def _score_eight_rect_window(rects: List[Tuple[float, float, float, float]]) -> float:
    xs = [_bbox_center(r)[0] for r in rects]
    gaps = [xs[i + 1] - xs[i] for i in range(7)]
    if min(gaps) <= 0:
        return 1e9

    left_gaps = gaps[0:1]
    center_gaps = gaps[2:5]
    right_gaps = gaps[6:7]
    intra = left_gaps + center_gaps + right_gaps
    inter = [gaps[1], gaps[5]]

    intra_med = float(np.median(intra)) if intra else 1.0
    inter_med = float(np.median(inter)) if inter else intra_med

    if intra_med <= 0:
        return 1e9

    ratio_penalty = 0.0
    if inter_med < intra_med * 1.2:
        ratio_penalty += (intra_med * 1.2 - inter_med) * 20.0

    intra_std = float(np.std(intra)) if intra else 0.0
    inter_std = float(np.std(inter)) if inter else 0.0
    span = xs[-1] - xs[0]

    return span + intra_std * 10.0 + inter_std * 6.0 + ratio_penalty


def _pick_eight_in_row(rects: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    if len(rects) < 8:
        return []

    sorted_rects = sorted(rects, key=lambda r: _bbox_center(r)[0])
    if len(sorted_rects) == 8:
        return sorted_rects

    best: Optional[Tuple[float, List[Tuple[float, float, float, float]]]] = None
    for i in range(len(sorted_rects) - 7):
        window = sorted_rects[i : i + 8]
        score = _score_eight_rect_window(window)
        if best is None or score < best[0]:
            best = (score, window)

    return best[1] if best else []


def _extract_mid_band_above_bonus(
    page: pdfplumber.page.Page,
    words: List[dict],
    side_letter: str,
    bonus_y: Optional[float],
) -> Optional[FullPalletMidBandSection]:
    pw, ph = float(page.width), float(page.height)
    holder_zone_bottom = ph * 0.31

    five_words = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
    five_words = [w for w in five_words if float(w.get("top", 0.0)) >= holder_zone_bottom]
    if bonus_y is not None:
        five_words = [w for w in five_words if float(w.get("bottom", 0.0)) < (bonus_y - 2.0)]

    raw_rects = page.rects or []
    candidate_rects: List[Tuple[float, float, float, float]] = []
    for r in raw_rects:
        b = _normalize_bbox(r)
        if not b:
            continue
        x0, y0, x1, y1 = b
        w = x1 - x0
        h = y1 - y0
        if w < pw * 0.05 or w > pw * 0.24:
            continue
        if h < ph * 0.02 or h > ph * 0.17:
            continue
        if y0 < holder_zone_bottom - 16.0:
            continue
        if bonus_y is not None and ((y0 + y1) / 2.0) >= (bonus_y - 2.0):
            continue
        candidate_rects.append(b)

    rect_map: Dict[Tuple[float, float, float, float], dict] = {}
    for w in five_words:
        xc, yc = _word_center(w)
        containing = [r for r in candidate_rects if _contains_point(r, xc, yc)]
        if not containing:
            continue
        containing.sort(key=_bbox_area)
        chosen = containing[0]
        cur = rect_map.get(chosen)
        if cur is None:
            rect_map[chosen] = {"rect": chosen, "tokens": [w]}
        else:
            cur["tokens"].append(w)

    if len(rect_map) < 24:
        return None

    rect_items = list(rect_map.values())
    ys = [_bbox_center(it["rect"])[1] for it in rect_items]
    y_centers = cluster_positions(ys, tol=max(8.0, ph * 0.012))
    if len(y_centers) < 3:
        return None

    by_row: Dict[int, List[dict]] = {}
    for it in rect_items:
        _, yc = _bbox_center(it["rect"])
        ridx = int(np.argmin(np.abs(y_centers - yc)))
        by_row.setdefault(ridx, []).append(it)

    groups: List[dict] = []
    for ridx, items in by_row.items():
        y = float(np.mean([_bbox_center(it["rect"])[1] for it in items]))
        groups.append({"ridx": ridx, "y": y, "rects": [it["rect"] for it in items]})
    groups.sort(key=lambda g: g["y"])

    chosen_groups = _pick_row_window(groups, bonus_y)
    if len(chosen_groups) != 3:
        return None

    row_slots: List[FullPalletMidBandRow] = []
    slot_order = 0

    for row_index, grp in enumerate(chosen_groups):
        picked = _pick_eight_in_row(grp["rects"])
        if len(picked) != 8:
            return None

        slots: List[FullPalletMidBandSlot] = []
        for slot_in_row, rect in enumerate(picked):
            raw_text = (page.crop(rect).extract_text() or "").strip()
            parsed_name, parsed_last5, qty = parse_label_cell_text(raw_text)

            last5_from_name = _to_last5(parsed_last5)
            if not last5_from_name:
                token_hits = [w for w in five_words if _contains_point(rect, *_word_center(w))]
                if token_hits:
                    token_hits.sort(key=lambda w: (_word_center(w)[1], _word_center(w)[0]))
                    last5_from_name = _to_last5(str(token_hits[0].get("text", "")))

            if slot_in_row <= 1:
                block_name = "left"
                block_pos = slot_in_row
            elif slot_in_row <= 5:
                block_name = "center"
                block_pos = slot_in_row - 2
            else:
                block_name = "right"
                block_pos = slot_in_row - 6

            slot_id = f"{side_letter}-MB-R{row_index + 1}-S{slot_in_row + 1}"
            slots.append(
                FullPalletMidBandSlot(
                    slot_id=slot_id,
                    side_letter=side_letter,
                    row_index=row_index,
                    block_name=block_name,
                    block_pos_index=block_pos,
                    slot_order=slot_order,
                    slot_in_row=slot_in_row,
                    bbox=rect,
                    raw_label_text=raw_text,
                    parsed_name=parsed_name,
                    last5=last5_from_name,
                    qty=qty,
                )
            )
            slot_order += 1

        row_slots.append(FullPalletMidBandRow(row_index=row_index, slots=slots))

    row_slot_counts = [len(r.slots) for r in row_slots]
    row_block_grouping = [
        [
            len([s for s in r.slots if s.block_name == "left"]),
            len([s for s in r.slots if s.block_name == "center"]),
            len([s for s in r.slots if s.block_name == "right"]),
        ]
        for r in row_slots
    ]
    slot_count = sum(row_slot_counts)
    shape_valid = (
        len(row_slots) == 3
        and row_slot_counts == [8, 8, 8]
        and row_block_grouping == [[2, 4, 2], [2, 4, 2], [2, 4, 2]]
        and slot_count == 24
    )

    return FullPalletMidBandSection(
        section_id="mid_band_above_bonus",
        rows=row_slots,
        slot_count=slot_count,
        row_slot_counts=row_slot_counts,
        row_block_grouping=row_block_grouping,
        shape_valid=shape_valid,
    )

def _extract_legacy_cells(
    page: pdfplumber.page.Page,
    words: List[dict],
) -> List[CellData]:
    pw, ph = float(page.width), float(page.height)
    five = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
    if not five:
        return []

    holder_zone_bottom = ph * 0.31
    five = [w for w in five if float(w.get("top", 0)) >= holder_zone_bottom]
    if not five:
        return []

    xs = [(w["x0"] + w["x1"]) / 2 for w in five]
    ys = [(w["top"] + w["bottom"]) / 2 for w in five]

    x_centers = cluster_positions(xs, tol=max(10, pw * 0.025))
    y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))
    if len(x_centers) == 0 or len(y_centers) == 0:
        return []

    x_bounds = boundaries_from_centers(x_centers)
    y_bounds = boundaries_from_centers(y_centers)
    if len(x_bounds) < 2 or len(y_bounds) < 2:
        return []

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

    return cells


def extract_full_pallet_pages(labels_pdf_bytes: bytes) -> List[FullPalletPage]:
    pages: List[FullPalletPage] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []

            wt = _wt
            bonus_words = [w for w in words if wt(w) == "BONUS"]
            bonus_y: Optional[float] = None
            if bonus_words:
                bw = min(bonus_words, key=lambda w: float(w["top"]))
                bonus_y = (float(bw["top"]) + float(bw["bottom"])) / 2

            mid_band = _extract_mid_band_above_bonus(
                page,
                words,
                side_letter=chr(ord("A") + min(pidx, 3)),
                bonus_y=bonus_y,
            )

            cells = _extract_legacy_cells(page, words)
            if not cells and not mid_band:
                continue

            pw, ph = float(page.width), float(page.height)
            annotations: List[AnnotationBox] = []

            xs_for_ann = [(_word_center(w)[0]) for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
            if xs_for_ann:
                cx0_content = min(xs_for_ann) - 15
                cx1_content = max(xs_for_ann) + 15
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
                    side_letter=chr(ord("A") + min(pidx, 3)),
                    cells=cells,
                    annotations=annotations,
                    mid_band_above_bonus=mid_band,
                )
            )
    return pages
