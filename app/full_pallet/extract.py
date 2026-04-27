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


def _contains_point(b: Tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = b
    return x0 <= x <= x1 and y0 <= y <= y1


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _word_bbox(w: dict) -> Tuple[float, float, float, float]:
    return (
        float(w.get("x0", 0.0)),
        float(w.get("top", 0.0)),
        float(w.get("x1", 0.0)),
        float(w.get("bottom", 0.0)),
    )


def _inset_bbox(
    bbox: Tuple[float, float, float, float],
    inset_x: float,
    inset_y: float,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    nx0 = x0 + inset_x
    ny0 = y0 + inset_y
    nx1 = x1 - inset_x
    ny1 = y1 - inset_y
    if nx1 <= nx0 or ny1 <= ny0:
        return bbox
    return (nx0, ny0, nx1, ny1)


def _bbox_contains_word_center(bbox: Tuple[float, float, float, float], word: dict) -> bool:
    return _contains_point(bbox, *_word_center(word))


def _filter_words_to_bboxes(words: List[dict], allowed_bboxes: List[Tuple[float, float, float, float]]) -> List[dict]:
    out: List[dict] = []
    for w in words:
        if any(_bbox_contains_word_center(b, w) for b in allowed_bboxes):
            out.append(w)
    return out


def _filter_words_to_excluded_regions(
    words: List[dict], excluded_bboxes: List[Tuple[float, float, float, float]]
) -> List[dict]:
    if not excluded_bboxes:
        return words
    out: List[dict] = []
    for w in words:
        if any(_bbox_contains_word_center(b, w) for b in excluded_bboxes):
            continue
        out.append(w)
    return out


def _collect_words_in_zone(
    words: List[dict],
    allowed_bbox: Tuple[float, float, float, float],
    excluded_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[dict]:
    picked = [w for w in words if _bbox_contains_word_center(allowed_bbox, w)]
    if excluded_bboxes:
        picked = _filter_words_to_excluded_regions(picked, excluded_bboxes)
    return picked


def _words_to_text(words: List[dict]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))))
    lines: List[List[str]] = []
    current: List[str] = []
    current_y: Optional[float] = None
    y_tol = 3.5
    for w in ordered:
        y = float(w.get("top", 0.0))
        if current_y is None or abs(y - current_y) <= y_tol:
            current.append(str(w.get("text", "")).strip())
            if current_y is None:
                current_y = y
            else:
                current_y = (current_y + y) / 2.0
        else:
            if current:
                lines.append(current)
            current = [str(w.get("text", "")).strip()]
            current_y = y
    if current:
        lines.append(current)
    return "\n".join(" ".join(tok for tok in line if tok) for line in lines).strip()


def _compute_mid_band_anchor_bounds(
    words: List[dict],
    pw: float,
    ph: float,
    bonus_top: Optional[float],
) -> Optional[Tuple[float, float, float, float]]:
    holder_words = [
        w
        for w in words
        if _wt(w) in {"WM", "GIFTCARD", "GIFTCAR", "PKG"}
        and float(w.get("top", 0.0)) < ph * 0.35
    ]
    holder_bottom = max(float(w.get("bottom", 0.0)) for w in holder_words) + 8.0 if holder_words else ph * 0.31

    if bonus_top is None:
        bonus_top = ph * 0.84

    # Anchor middle band from BONUS upward (deterministic 3-row zone).
    row_h = max(44.0, min(76.0, ph * 0.052))
    row_gap = max(8.0, min(18.0, ph * 0.012))
    outer_pad_top = 8.0
    outer_pad_bottom = 6.0

    mid_bottom = bonus_top - outer_pad_bottom
    mid_top = mid_bottom - (3.0 * row_h + 2.0 * row_gap + outer_pad_top)

    # Never allow the middle-band template to climb into holder/GCI area.
    min_top = holder_bottom + 24.0
    if mid_top < min_top:
        mid_top = min_top
        mid_bottom = mid_top + (3.0 * row_h + 2.0 * row_gap + outer_pad_top)
    if mid_bottom >= bonus_top - 2.0:
        mid_bottom = bonus_top - 6.0

    # Constrain to the visual product band near the middle section, not global token span.
    x0 = pw * 0.16
    x1 = pw * 0.84
    x_window_top = max(0.0, mid_top - ph * 0.04)
    x_window_bottom = min(ph, mid_bottom + ph * 0.03)
    five_words = [w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))]
    five_words = [
        w
        for w in five_words
        if x_window_top <= float(w.get("top", 0.0)) <= x_window_bottom
    ]
    if five_words:
        x0 = max(x0, min(float(w["x0"]) for w in five_words) - 10.0)
        x1 = min(x1, max(float(w["x1"]) for w in five_words) + 10.0)

    # Clamp inward from known signage bands.
    mkt_words = [w for w in words if _wt(w) in {"MARKETING", "MESSAGE", "PANEL"}]
    fraud_words = [w for w in words if _wt(w) in {"FRAUD", "SIGNAGE"}]
    signage_groups = _group_nearby(mkt_words + fraud_words, x_tol=40, y_tol=20)
    left_clamp = x0
    right_clamp = x1
    for grp in signage_groups:
        gx0 = min(float(w["x0"]) for w in grp)
        gx1 = max(float(w["x1"]) for w in grp)
        gy0 = min(float(w["top"]) for w in grp)
        gy1 = max(float(w["bottom"]) for w in grp)
        # Only clamp using groups that vertically overlap the middle band zone.
        if gy1 < mid_top - 20.0 or gy0 > mid_bottom + 20.0:
            continue
        gcx = (gx0 + gx1) / 2.0
        if gcx < pw / 2.0:
            left_clamp = max(left_clamp, gx1 + 8.0)
        else:
            right_clamp = min(right_clamp, gx0 - 8.0)

    x0 = max(x0, left_clamp)
    x1 = min(x1, right_clamp)

    top = max(mid_top, holder_bottom + 20.0)
    bottom = min(mid_bottom, bonus_top - 4.0)

    if x1 - x0 < 140.0 or bottom - top < 120.0:
        return None

    return (x0, top, x1, bottom)


def _build_mid_band_template_slots(
    x0: float,
    top: float,
    x1: float,
    bottom: float,
) -> Optional[List[List[Tuple[float, float, float, float]]]]:
    section_w = x1 - x0
    section_h = bottom - top
    if section_w <= 0.0 or section_h <= 0.0:
        return None

    inter_gap = max(10.0, min(28.0, section_w * 0.045))
    intra_gap = max(4.0, min(12.0, section_w * 0.014))
    row_gap = max(6.0, min(14.0, section_h * 0.08))

    slot_w = (section_w - (2.0 * inter_gap) - (5.0 * intra_gap)) / 8.0
    row_h = max(40.0, min(62.0, section_h * 0.28))
    needed_h = 3.0 * row_h + 2.0 * row_gap

    if slot_w < 18.0:
        return None
    if section_h < 90.0:
        return None

    if needed_h > section_h:
        # Prefer keeping row gaps stable and slightly compress rows to fit.
        row_h = (section_h - 2.0 * row_gap) / 3.0
    if row_h < 24.0:
        return None

    slot_h = row_h

    row_xs = [
        x0,
        x0 + slot_w + intra_gap,
        x0 + 2 * slot_w + intra_gap + inter_gap,
        x0 + 3 * slot_w + 2 * intra_gap + inter_gap,
        x0 + 4 * slot_w + 3 * intra_gap + inter_gap,
        x0 + 5 * slot_w + 4 * intra_gap + inter_gap,
        x0 + 6 * slot_w + 5 * intra_gap + 2 * inter_gap,
        x0 + 7 * slot_w + 6 * intra_gap + 2 * inter_gap,
    ]

    rows: List[List[Tuple[float, float, float, float]]] = []
    for r in range(3):
        ry0 = top + r * (row_h + row_gap)
        ry1 = ry0 + slot_h
        if r == 2 and ry1 > bottom:
            # final guardrail against a slightly tight anchor
            overflow = ry1 - bottom
            ry0 -= overflow
            ry1 -= overflow
        rows.append([(sx, ry0, sx + slot_w, ry1) for sx in row_xs])

    return rows


def _collect_mid_band_product_tokens(
    page: pdfplumber.page.Page,
    words: List[dict],
    bonus_top: Optional[float],
    page_width: float,
    page_height: float,
) -> List[dict]:
    _ = page
    holder_words = [
        w
        for w in words
        if _wt(w) in {"WM", "GIFTCARD", "GIFTCAR", "PKG"}
        and float(w.get("top", 0.0)) < page_height * 0.35
    ]
    holder_bottom = (
        max(float(w.get("bottom", 0.0)) for w in holder_words) + 8.0
        if holder_words
        else page_height * 0.31
    )
    zone_top = max(0.0, holder_bottom + 6.0)
    zone_bottom = min(page_height, (bonus_top - 8.0) if bonus_top is not None else page_height * 0.84)
    if zone_bottom <= zone_top:
        return []

    excluded_terms = {
        "MARKETING",
        "MESSAGE",
        "PANEL",
        "FRAUD",
        "SIGNAGE",
        "WM",
        "GIFTCARD",
        "GIFTCAR",
        "PKG",
        "BONUS",
    }
    excluded_words = [w for w in words if _wt(w) in excluded_terms]
    excluded_bboxes: List[Tuple[float, float, float, float]] = []
    for grp in _group_nearby(excluded_words, x_tol=35, y_tol=18):
        gx0 = min(float(w.get("x0", 0.0)) for w in grp) - 6.0
        gy0 = min(float(w.get("top", 0.0)) for w in grp) - 5.0
        gx1 = max(float(w.get("x1", 0.0)) for w in grp) + 6.0
        gy1 = max(float(w.get("bottom", 0.0)) for w in grp) + 5.0
        excluded_bboxes.append((max(0.0, gx0), max(0.0, gy0), min(page_width, gx1), min(page_height, gy1)))

    if holder_words:
        h_top = min(float(w.get("top", 0.0)) for w in holder_words) - 3.0
        h_bottom = max(float(w.get("bottom", 0.0)) for w in holder_words) + 8.0
        excluded_bboxes.append((0.0, max(0.0, h_top), page_width, min(page_height, h_bottom)))

    picked: List[dict] = []
    for w in words:
        txt = str(w.get("text", "")).strip()
        if not re.fullmatch(r"\d{5}", txt):
            continue
        wx, wy = _word_center(w)
        wt = float(w.get("top", 0.0))
        wb = float(w.get("bottom", 0.0))
        if not (zone_top <= wy <= zone_bottom):
            continue
        if not (zone_top <= wt and wb <= zone_bottom + 4.0):
            continue
        if bonus_top is not None and wy >= bonus_top - 8.0:
            continue
        if wx < page_width * 0.03 or wx > page_width * 0.97:
            continue
        if any(_bbox_contains_word_center(b, w) for b in excluded_bboxes):
            continue
        picked.append(w)

    picked.sort(key=lambda w: (_word_center(w)[1], _word_center(w)[0]))
    return picked


def _collect_mid_band_five_digit_words(
    words: List[dict],
    anchor_bbox: Tuple[float, float, float, float],
    bonus_top: Optional[float],
    page_width: float,
    page_height: float,
) -> List[dict]:
    ax0, atop, ax1, abottom = anchor_bbox
    zone_top = max(0.0, atop - 20.0)
    zone_bottom = min(page_height, abottom + 20.0)
    zone_left = max(0.0, ax0 - 24.0)
    zone_right = min(page_width, ax1 + 24.0)

    picked: List[dict] = []
    for w in words:
        txt = str(w.get("text", "")).strip()
        if not re.fullmatch(r"\d{5}", txt):
            continue
        wx = (float(w.get("x0", 0.0)) + float(w.get("x1", 0.0))) / 2.0
        wy = (float(w.get("top", 0.0)) + float(w.get("bottom", 0.0))) / 2.0
        if not (zone_left <= wx <= zone_right):
            continue
        if not (zone_top <= float(w.get("top", 0.0)) and float(w.get("bottom", 0.0)) <= zone_bottom):
            continue
        if bonus_top is not None and wy >= bonus_top - 8.0:
            continue
        picked.append(w)

    picked.sort(key=lambda w: (_word_center(w)[1], _word_center(w)[0]))
    return picked


def _assign_mid_band_tokens_to_slots(
    five_words: List[dict],
    template_rows: List[List[Tuple[float, float, float, float]]],
) -> Dict[Tuple[int, int], dict]:
    slot_meta: List[dict] = []
    for row_index, row_boxes in enumerate(template_rows):
        for slot_in_row, bbox in enumerate(row_boxes):
            sx0, sy0, sx1, sy1 = bbox
            sw = max(1.0, sx1 - sx0)
            sh = max(1.0, sy1 - sy0)
            slot_meta.append(
                {
                    "key": (row_index, slot_in_row),
                    "cx": (sx0 + sx1) / 2.0,
                    "cy": (sy0 + sy1) / 2.0,
                    "x_tol": max(sw * 0.75, 18.0),
                    "y_tol": max(sh * 0.75, 18.0),
                }
            )

    candidates: List[Tuple[float, int, Tuple[int, int]]] = []
    for token_idx, w in enumerate(five_words):
        wx, wy = _word_center(w)
        for sm in slot_meta:
            dx = abs(wx - float(sm["cx"]))
            dy = abs(wy - float(sm["cy"]))
            x_tol = float(sm["x_tol"])
            y_tol = float(sm["y_tol"])
            if dx > x_tol or dy > y_tol:
                continue
            norm_dist = ((dx / x_tol) ** 2 + (dy / y_tol) ** 2) ** 0.5
            candidates.append((norm_dist, token_idx, sm["key"]))

    candidates.sort(key=lambda t: t[0])
    token_used: set = set()
    slot_used: set = set()
    assigned: Dict[Tuple[int, int], dict] = {}
    for _dist, token_idx, slot_key in candidates:
        if token_idx in token_used or slot_key in slot_used:
            continue
        assigned[slot_key] = five_words[token_idx]
        token_used.add(token_idx)
        slot_used.add(slot_key)
    return assigned


def _extract_mid_band_above_bonus(
    page: pdfplumber.page.Page,
    words: List[dict],
    side_letter: str,
    bonus_y: Optional[float],
) -> Optional[FullPalletMidBandSection]:
    pw, ph = float(page.width), float(page.height)
    bonus_words = [w for w in words if _wt(w) == "BONUS"]
    bonus_top = min((float(w.get("top", 0.0)) for w in bonus_words), default=(bonus_y - 10.0 if bonus_y else None))

    token_words = _collect_mid_band_product_tokens(
        page=page,
        words=words,
        bonus_top=bonus_top,
        page_width=pw,
        page_height=ph,
    )
    if len(token_words) >= 8:
        token_xs = [float((_word_center(w)[0])) for w in token_words]
        token_ys = [float((_word_center(w)[1])) for w in token_words]
        y_centers = cluster_positions(token_ys, tol=max(10.0, ph * 0.013))
        if len(y_centers) == 0:
            y_centers = np.array([float(np.mean(token_ys))], dtype=float)

        row_bins: Dict[int, List[dict]] = {}
        for w, yc in zip(token_words, token_ys):
            row_idx_raw = int(np.argmin(np.abs(y_centers - yc)))
            row_bins.setdefault(row_idx_raw, []).append(w)

        sorted_row_keys = sorted(row_bins.keys(), key=lambda rid: float(np.mean([_word_center(w)[1] for w in row_bins[rid]])))
        q1 = float(np.quantile(token_xs, 1.0 / 3.0)) if token_xs else pw * 0.33
        q2 = float(np.quantile(token_xs, 2.0 / 3.0)) if token_xs else pw * 0.66

        row_slots: List[FullPalletMidBandRow] = []
        slot_order = 0
        for row_index, row_key in enumerate(sorted_row_keys):
            row_tokens = sorted(row_bins[row_key], key=lambda w: _word_center(w)[0])
            slots: List[FullPalletMidBandSlot] = []
            block_counts = {"left": 0, "center": 0, "right": 0}

            for slot_in_row, token_w in enumerate(row_tokens):
                tx0, ty0, tx1, ty1 = _word_bbox(token_w)
                tw = max(1.0, tx1 - tx0)
                th = max(1.0, ty1 - ty0)
                tcx, tcy = _word_center(token_w)

                source_bbox = (
                    max(0.0, tcx - max(54.0, tw * 3.0)),
                    max(0.0, tcy - max(20.0, th * 2.2)),
                    min(pw, tcx + max(54.0, tw * 3.0)),
                    min(ph, tcy + max(22.0, th * 2.8)),
                )
                nearby_words = _collect_words_in_zone(words, source_bbox)
                raw_text = _words_to_text(nearby_words) or str(token_w.get("text", "")).strip()
                parsed_name, _parsed_last5, qty = parse_label_cell_text(raw_text)
                last5 = _to_last5(str(token_w.get("text", "")).strip())

                if tcx <= q1:
                    block_name = "left"
                elif tcx >= q2:
                    block_name = "right"
                else:
                    block_name = "center"
                block_pos_index = block_counts[block_name]
                block_counts[block_name] += 1

                slot_id = f"{side_letter}-MB-R{row_index + 1}-S{slot_in_row + 1}"
                accepted_tokens = [str(w.get("text", "")).strip() for w in nearby_words]
                if last5 and last5 not in accepted_tokens:
                    accepted_tokens.append(last5)
                slots.append(
                    FullPalletMidBandSlot(
                        slot_id=slot_id,
                        side_letter=side_letter,
                        row_index=row_index,
                        block_name=block_name,
                        block_pos_index=block_pos_index,
                        slot_order=slot_order,
                        slot_in_row=slot_in_row,
                        bbox=(tx0, ty0, tx1, ty1),
                        raw_label_text=raw_text,
                        parsed_name=parsed_name,
                        last5=last5,
                        qty=qty,
                        extraction_bbox=source_bbox,
                        accepted_words=accepted_tokens,
                        rejected_nearby_word_count=0,
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
        anchor_bbox = (
            max(0.0, min(float(w.get("x0", 0.0)) for w in token_words) - 8.0),
            max(0.0, min(float(w.get("top", 0.0)) for w in token_words) - 10.0),
            min(pw, max(float(w.get("x1", 0.0)) for w in token_words) + 8.0),
            min(ph, max(float(w.get("bottom", 0.0)) for w in token_words) + 10.0),
        )
        return FullPalletMidBandSection(
            section_id="mid_band_above_bonus_token_first",
            rows=row_slots,
            slot_count=slot_count,
            row_slot_counts=row_slot_counts,
            row_block_grouping=row_block_grouping,
            shape_valid=shape_valid,
            anchor_bbox=anchor_bbox,
        )

    anchors = _compute_mid_band_anchor_bounds(words, pw, ph, bonus_top)
    if anchors is None:
        return None

    ax0, atop, ax1, abottom = anchors
    template_rows = _build_mid_band_template_slots(ax0, atop, ax1, abottom)
    if template_rows is None:
        return None

    mid_five_words = _collect_mid_band_five_digit_words(
        words=words,
        anchor_bbox=anchors,
        bonus_top=bonus_top,
        page_width=pw,
        page_height=ph,
    )
    assigned_five_by_slot = _assign_mid_band_tokens_to_slots(mid_five_words, template_rows)

    slot_bboxes = [b for row in template_rows for b in row]
    # Whitelist model: only slot-contained words are candidates for middle-band extraction.
    slot_whitelist_words = _filter_words_to_bboxes(words, slot_bboxes)

    excluded_bboxes: List[Tuple[float, float, float, float]] = []
    if bonus_top is not None:
        excluded_bboxes.append((0.0, bonus_top - 14.0, pw, min(ph, bonus_top + 28.0)))
    excluded_terms = {"MARKETING", "MESSAGE", "PANEL", "FRAUD", "SIGNAGE", "WM", "GIFTCARD", "GIFTCAR", "PKG"}
    excluded_words = [w for w in words if _wt(w) in excluded_terms]
    for grp in _group_nearby(excluded_words, x_tol=30, y_tol=18):
        gx0 = min(float(w["x0"]) for w in grp) - 4.0
        gy0 = min(float(w["top"]) for w in grp) - 3.0
        gx1 = max(float(w["x1"]) for w in grp) + 4.0
        gy1 = max(float(w["bottom"]) for w in grp) + 3.0
        excluded_bboxes.append((max(0.0, gx0), max(0.0, gy0), min(pw, gx1), min(ph, gy1)))

    row_slots: List[FullPalletMidBandRow] = []
    slot_order = 0
    for row_index, row_boxes in enumerate(template_rows):
        slots: List[FullPalletMidBandSlot] = []

        for slot_in_row, bbox in enumerate(row_boxes):
            slot_w = bbox[2] - bbox[0]
            slot_h = bbox[3] - bbox[1]
            extract_bbox = _inset_bbox(bbox, inset_x=max(1.2, slot_w * 0.04), inset_y=max(1.0, slot_h * 0.05))

            slot_full_words = _collect_words_in_zone(slot_whitelist_words, bbox, excluded_bboxes=excluded_bboxes)
            in_slot_words = _collect_words_in_zone(slot_whitelist_words, extract_bbox, excluded_bboxes=excluded_bboxes)
            rejected_nearby = max(0, len(slot_full_words) - len(in_slot_words))

            raw_text = _words_to_text(in_slot_words)
            parsed_name, parsed_last5, qty = parse_label_cell_text(raw_text)
            accepted_word_tokens = [str(w.get("text", "")).strip() for w in in_slot_words]

            last5 = _to_last5(parsed_last5)
            if not last5:
                five_tokens = [
                    w for w in in_slot_words if re.fullmatch(r"\d{5}", str(w.get("text", "")).strip())
                ]
                if five_tokens:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    five_tokens.sort(
                        key=lambda w: (
                            abs(_word_center(w)[1] - cy),
                            abs(_word_center(w)[0] - cx),
                            float(w["top"]),
                            float(w["x0"]),
                        )
                    )
                    last5 = _to_last5(str(five_tokens[0].get("text", "")))

            if not last5:
                recovered = assigned_five_by_slot.get((row_index, slot_in_row))
                if recovered is not None:
                    recovered_last5 = _to_last5(str(recovered.get("text", "")).strip())
                    if recovered_last5:
                        last5 = recovered_last5
                        accepted_word_tokens.append(f"TOKEN_RECOVERY:{recovered_last5}")

            if slot_in_row <= 1:
                block_name = "left"
                block_pos_index = slot_in_row
            elif slot_in_row <= 5:
                block_name = "center"
                block_pos_index = slot_in_row - 2
            else:
                block_name = "right"
                block_pos_index = slot_in_row - 6

            slot_id = f"{side_letter}-MB-R{row_index + 1}-S{slot_in_row + 1}"
            slots.append(
                FullPalletMidBandSlot(
                    slot_id=slot_id,
                    side_letter=side_letter,
                    row_index=row_index,
                    block_name=block_name,
                    block_pos_index=block_pos_index,
                    slot_order=slot_order,
                    slot_in_row=slot_in_row,
                    bbox=bbox,
                    raw_label_text=raw_text,
                    parsed_name=parsed_name,
                    last5=last5,
                    qty=qty,
                    extraction_bbox=extract_bbox,
                    accepted_words=accepted_word_tokens,
                    rejected_nearby_word_count=rejected_nearby,
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
        anchor_bbox=(ax0, atop, ax1, abottom),
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

            side_letter = chr(ord("A") + min(pidx, 3))
            mid_band = _extract_mid_band_above_bonus(page, words, side_letter=side_letter, bonus_y=bonus_y)
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
                    side_letter=side_letter,
                    cells=cells,
                    annotations=annotations,
                    mid_band_above_bonus=mid_band,
                )
            )
    return pages
