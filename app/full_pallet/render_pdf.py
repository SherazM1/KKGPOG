from __future__ import annotations

import difflib
import io
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
import pandas as pd
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

FULL_PALLET_MID_BAND_PROFILES = {
    "A": {
        "profile_name": "profile_1_ac",
        "physical_groups": ["left", "center", "right"],
        "expected_max_per_group": {"left": 6, "center": 12, "right": 6},
        "expected_rows_per_group": {
            "left": [2, 2, 2],
            "center": [4, 4, 4],
            "right": [2, 2, 2],
        },
        "render_layout_hints": {
            "rows": 3,
            "group_order": ["left", "center", "right"],
            "row_grouping": [[2, 4, 2], [2, 4, 2], [2, 4, 2]],
        },
    },
    "B": {
        "profile_name": "profile_2_bd",
        "physical_groups": ["left", "center", "right"],
        "expected_max_per_group": {"left": 6, "center": 12, "right": 6},
        "expected_rows_per_group": {
            "left": [2, 2, 2],
            "center": [4, 4, 4],
            "right": [2, 2, 2],
        },
        "render_layout_hints": {
            "rows": 3,
            "group_order": ["left", "center", "right"],
            "row_grouping": [[2, 4, 2], [2, 4, 2], [2, 4, 2]],
            "strict_image_cell_guardrails": True,
        },
    },
    "C": {
        "profile_name": "profile_1_ac",
        "physical_groups": ["left", "center", "right"],
        "expected_max_per_group": {"left": 6, "center": 12, "right": 6},
        "expected_rows_per_group": {
            "left": [2, 2, 2],
            "center": [4, 4, 4],
            "right": [2, 2, 2],
        },
        "render_layout_hints": {
            "rows": 3,
            "group_order": ["left", "center", "right"],
            "row_grouping": [[2, 4, 2], [2, 4, 2], [2, 4, 2]],
        },
    },
    "D": {
        "profile_name": "profile_2_bd",
        "physical_groups": ["left", "center", "right"],
        "expected_max_per_group": {"left": 6, "center": 12, "right": 6},
        "expected_rows_per_group": {
            "left": [2, 2, 2],
            "center": [4, 4, 4],
            "right": [2, 2, 2],
        },
        "render_layout_hints": {
            "rows": 3,
            "group_order": ["left", "center", "right"],
            "row_grouping": [[2, 4, 2], [2, 4, 2], [2, 4, 2]],
            "strict_image_cell_guardrails": True,
        },
    },
}


def get_mid_band_physical_profile(side_letter: str) -> Dict[str, object]:
    side = str(side_letter or "").strip().upper()
    return FULL_PALLET_MID_BAND_PROFILES.get(side, FULL_PALLET_MID_BAND_PROFILES["A"])


@dataclass(frozen=True)
class SelectionResult:
    selected_cards: List[dict]
    omitted_cards: List[dict]
    selected_by_group: Dict[str, List[dict]]
    omitted_by_group: Dict[str, List[dict]]
    debug_summary: Dict[str, object]


def select_mid_band_cards_for_display(
    side_letter: str,
    candidates: List[dict],
    profile: Dict[str, object],
) -> SelectionResult:
    layout_hints = dict(profile.get("render_layout_hints", {}) or {})
    row_grouping = list(layout_hints.get("row_grouping") or [[2, 4, 2], [2, 4, 2], [2, 4, 2]])
    groups = [str(g) for g in layout_hints.get("group_order") or profile.get("physical_groups", ["left", "center", "right"])]
    row_capacities = [int(sum(row)) for row in row_grouping]
    expected_selected_count = int(sum(row_capacities))
    target_row_count = len(row_capacities)

    def _slot_id(candidate: dict) -> str:
        slot = candidate.get("slot")
        return str(candidate.get("slot_id") or getattr(slot, "slot_id", ""))

    def _slot_value(candidate: dict, key: str, default: int = 0) -> int:
        slot = candidate.get("slot")
        return int(candidate.get(key, getattr(slot, key, default)))

    def _center(candidate: dict) -> Tuple[float, float]:
        slot = candidate.get("slot")
        bbox = getattr(slot, "bbox", None)
        if bbox:
            x0, y0, x1, y1 = bbox
            return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)
        return (float(_slot_value(candidate, "slot_in_row")), float(_slot_value(candidate, "row_index")))

    def _group_for_col(row_pattern: List[int], col_index: int) -> str:
        running = 0
        for idx, count in enumerate(row_pattern):
            running += int(count)
            if col_index < running:
                return groups[idx] if idx < len(groups) else f"group_{idx + 1}"
        return groups[-1] if groups else "unknown"

    def _visual_key(candidate: dict) -> Tuple[float, float, int]:
        x, y = _center(candidate)
        return (y, x, _slot_value(candidate, "slot_order"))

    ordered_candidates = sorted(candidates, key=_visual_key)
    natural_rows: List[List[dict]] = []
    if ordered_candidates:
        y_values = [_center(c)[1] for c in ordered_candidates]
        y_diffs = [
            y_values[i + 1] - y_values[i]
            for i in range(len(y_values) - 1)
            if (y_values[i + 1] - y_values[i]) > 0.001
        ]
        y_tol = max(8.0, (float(np.median(y_diffs)) * 0.55) if y_diffs else 8.0)
        for candidate in ordered_candidates:
            _x, y = _center(candidate)
            if not natural_rows:
                natural_rows.append([candidate])
                continue
            last_row = natural_rows[-1]
            last_y = float(np.mean([_center(c)[1] for c in last_row]))
            if abs(y - last_y) <= y_tol:
                last_row.append(candidate)
            else:
                natural_rows.append([candidate])

    natural_rows = [sorted(row, key=lambda c: (_center(c)[0], _slot_value(c, "slot_order"))) for row in natural_rows]
    if len(natural_rows) > target_row_count:
        merged_rows: List[List[dict]] = [[] for _ in range(target_row_count)]
        for idx, row in enumerate(natural_rows):
            target_idx = min(target_row_count - 1, int(round(idx * (target_row_count - 1) / max(1, len(natural_rows) - 1))))
            merged_rows[target_idx].extend(row)
        natural_rows = [
            sorted(row, key=lambda c: (_center(c)[0], _slot_value(c, "slot_order")))
            for row in merged_rows
        ]
    while len(natural_rows) < target_row_count:
        natural_rows.append([])

    row_clusters_debug = []
    for row_idx, row in enumerate(natural_rows[:target_row_count]):
        row_clusters_debug.append(
            {
                "detected_row": row_idx,
                "candidate_count": len(row),
                "slot_ids": [_slot_id(c) for c in row],
                "source_row_indices": sorted({_slot_value(c, "row_index") for c in row}),
            }
        )

    selected_by_group: Dict[str, List[dict]] = {g: [] for g in groups}
    omitted_by_group: Dict[str, List[dict]] = {g: [] for g in groups}
    selected_cards: List[dict] = []
    omitted_cards: List[dict] = []
    overflow_pool: List[dict] = []
    selected_rows: List[List[dict]] = [[] for _ in range(target_row_count)]

    def _assign_selected(candidate: dict, display_row: int, display_col: int) -> dict:
        row_pattern = row_grouping[display_row] if display_row < len(row_grouping) else row_grouping[-1]
        display_group = _group_for_col([int(v) for v in row_pattern], display_col)
        return (
            {
                **candidate,
                "selected_row": display_row,
                "selected_col": display_col,
                "selected_group": display_group,
                "group": display_group,
            }
        )

    for row_idx, row in enumerate(natural_rows[:target_row_count]):
        row_capacity = row_capacities[row_idx]
        for source_col, candidate in enumerate(row):
            if len(selected_rows[row_idx]) < row_capacity:
                selected = _assign_selected(candidate, row_idx, len(selected_rows[row_idx]))
                selected_rows[row_idx].append(selected)
            else:
                row_pattern = row_grouping[row_idx] if row_idx < len(row_grouping) else row_grouping[-1]
                overflow_pool.append(
                    {
                        **candidate,
                        "source_detected_row": row_idx,
                        "source_detected_col": source_col,
                        "source_group": _group_for_col([int(v) for v in row_pattern], min(source_col, row_capacity - 1)),
                    }
                )

    remaining_overflow: List[dict] = []
    for candidate in overflow_pool:
        target_row = next((idx for idx, row in enumerate(selected_rows) if len(row) < row_capacities[idx]), None)
        if target_row is None:
            remaining_overflow.append(candidate)
            continue
        selected = _assign_selected(candidate, target_row, len(selected_rows[target_row]))
        selected["filled_shortage_from_overflow"] = True
        selected_rows[target_row].append(selected)

    for row_idx, row in enumerate(selected_rows):
        for selected in row:
            group = str(selected.get("selected_group", selected.get("group", "unknown")))
            selected_by_group.setdefault(group, []).append(selected)
            selected_cards.append(selected)

    for candidate in remaining_overflow:
        group = str(candidate.get("source_group") or candidate.get("group") or "unknown")
        omitted = {**candidate, "group": group, "omit_reason": "group_overflow"}
        omitted_by_group.setdefault(group, []).append(omitted)
        omitted_cards.append(omitted)

    for final_index, selected in enumerate(selected_cards):
        selected["final_index"] = final_index

    def _ids(rows: List[dict]) -> List[str]:
        return [_slot_id(r) for r in rows]

    def _upcs(rows: List[dict]) -> List[Optional[str]]:
        return [r.get("upc12") for r in rows]

    def _compact_row(row: dict, index_key: str, index_value: int) -> dict:
        return {
            index_key: index_value,
            "source_index": row.get("source_index"),
            "slot_id": _slot_id(row),
            "last5": row.get("last5"),
            "upc12": row.get("upc12"),
            "resolved_name": row.get("resolved_name") or row.get("display_name"),
        }

    candidate_order = [
        _compact_row(row, "source_index", int(row.get("source_index", idx)))
        for idx, row in enumerate(sorted(candidates, key=lambda r: int(r.get("source_index", 0))))
    ]
    selected_order = [
        {
            **_compact_row(row, "final_index", int(row.get("final_index", idx))),
            "selected_row": row.get("selected_row"),
            "selected_col": row.get("selected_col"),
            "selected_group": row.get("selected_group"),
        }
        for idx, row in enumerate(selected_cards)
    ]
    omitted_order = [
        {
            **_compact_row(row, "source_index", int(row.get("source_index", idx))),
            "omit_reason": row.get("omit_reason"),
        }
        for idx, row in enumerate(omitted_cards)
    ]

    selected_row_col_assignment = [
        {
            "final_index": idx,
            "slot_id": _slot_id(row),
            "selected_row": row.get("selected_row"),
            "selected_col": row.get("selected_col"),
            "selected_group": row.get("selected_group"),
            "source_row_index": _slot_value(row, "row_index"),
            "source_slot_in_row": _slot_value(row, "slot_in_row"),
        }
        for idx, row in enumerate(selected_cards)
    ]
    selected_group_assignment_per_row: Dict[int, Dict[str, List[str]]] = {}
    for row in selected_cards:
        display_row = int(row.get("selected_row", 0))
        display_group = str(row.get("selected_group", "unknown"))
        selected_group_assignment_per_row.setdefault(display_row, {}).setdefault(display_group, []).append(_slot_id(row))

    debug_summary = {
        "side": side_letter,
        "profile": profile.get("profile_name"),
        "candidate_count": len(candidates),
        "candidate_order": candidate_order,
        "selected_count": len(selected_cards),
        "selected_order": selected_order,
        "omitted_count": len(omitted_cards),
        "omitted_order": omitted_order,
        "expected_selected_count": expected_selected_count,
        "row_clusters_detected": row_clusters_debug,
        "selected_row_col_assignment": selected_row_col_assignment,
        "selected_group_assignment_per_row": selected_group_assignment_per_row,
        "selected_slot_ids_by_group": {g: _ids(rows) for g, rows in selected_by_group.items()},
        "selected_upcs_by_group": {g: _upcs(rows) for g, rows in selected_by_group.items()},
        "omitted_slot_ids_by_group": {g: _ids(rows) for g, rows in omitted_by_group.items()},
        "omitted_upcs_by_group": {g: _upcs(rows) for g, rows in omitted_by_group.items()},
        "omit_reasons_by_slot": {
            str(row.get("slot_id") or getattr(row.get("slot"), "slot_id", "")): row.get("omit_reason")
            for row in omitted_cards
        },
        "shortage_after_row_aware_selection": max(0, expected_selected_count - len(selected_cards)),
        "overage_after_row_aware_selection": len(omitted_cards),
        "shortage_or_overage": len(selected_cards) - expected_selected_count,
        "render_assignment": [],
        "final_render_order_matches_selected_order": None,
    }

    return SelectionResult(
        selected_cards=selected_cards,
        omitted_cards=omitted_cards,
        selected_by_group=selected_by_group,
        omitted_by_group=omitted_by_group,
        debug_summary=debug_summary,
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
    FOOTER_HEIGHT = 36.0
    SECTION_BAR_H = 24.0
    SECTION_BAR_GAP = 10.0
    BUCKET_GAP = 12.0

    BASE_CONTENT_H = BASE_PAGE_H - (2 * MARGIN) - HEADER_H - FOOTER_HEIGHT
    PPT_SECTION_H = BASE_CONTENT_H * 0.27
    HOLDER_SECTION_H = BASE_CONTENT_H * 0.18
    BAR_FILL = _hex_to_rgb("#77B5F0")
    BAR_TEXT = NAVY_RGB

    EMPTY_STROKE = (0.88, 0.88, 0.88)
    FILLED_STROKE = (0.78, 0.78, 0.80)

    logo = _try_load_logo()
    all_matrix_rows = [r for rows in matrix_idx.values() for r in rows]
    upc_to_rows: Dict[str, List[MatrixRow]] = {}
    for row in all_matrix_rows:
        upc_to_rows.setdefault((row.upc12 or "").strip(), []).append(row)

    def _best_row_for_label(candidates: List[MatrixRow], label_text: str) -> Optional[MatrixRow]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        target = _norm_name(label_text or "")
        if not target:
            return candidates[0]
        return max(
            candidates,
            key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio(),
        )

    def _extract_mid_slot_label_hints(slot: FullPalletMidBandSlot) -> Dict[str, List[str]]:
        raw_label = (slot.raw_label_text or "").strip()
        accepted = [str(t or "").strip() for t in (slot.accepted_words or [])]
        joined = " ".join([raw_label] + accepted)

        digit_tokens = re.findall(r"\d{4,14}", joined)
        upc_candidates: List[str] = []
        last5_candidates: List[str] = []
        seen_upc: set = set()
        seen_last5: set = set()

        for token in digit_tokens:
            digits = re.sub(r"[^0-9]", "", token)
            if len(digits) >= 11:
                candidate = digits[-12:].zfill(12)
                if candidate not in seen_upc:
                    upc_candidates.append(candidate)
                    seen_upc.add(candidate)

            last5 = _to_last5(digits[-5:] if len(digits) >= 5 else "")
            if last5 and last5 not in seen_last5:
                last5_candidates.append(last5)
                seen_last5.add(last5)

        return {
            "upc12_candidates": upc_candidates,
            "last5_candidates": last5_candidates,
        }

    def _resolve_mid_band_slot_no_position(slot: FullPalletMidBandSlot) -> Tuple[Optional[MatrixRow], Dict[str, object]]:
        label = (slot.parsed_name or slot.raw_label_text or "").strip()
        extracted_last5 = _to_last5(slot.last5)
        hints = _extract_mid_slot_label_hints(slot)

        if extracted_last5:
            primary = resolve_full_pallet(extracted_last5, label, matrix_idx)
            if primary is not None:
                return primary, {
                    "fallback_path": "labels_exact_last5",
                    "labels_last5": extracted_last5,
                    "label_hint_upc_candidates": hints["upc12_candidates"],
                    "label_hint_last5_candidates": hints["last5_candidates"],
                }

        for upc_hint in hints["upc12_candidates"]:
            upc_rows = upc_to_rows.get(upc_hint, [])
            match = _best_row_for_label(upc_rows, label)
            if match is not None:
                return match, {
                    "fallback_path": "labels_hint_upc",
                    "labels_last5": extracted_last5,
                    "label_hint_upc_candidates": hints["upc12_candidates"],
                    "label_hint_last5_candidates": hints["last5_candidates"],
                }

        for last5_hint in hints["last5_candidates"]:
            match = resolve_full_pallet(last5_hint, label, matrix_idx)
            if match is not None:
                return match, {
                    "fallback_path": "labels_hint_last5",
                    "labels_last5": extracted_last5,
                    "label_hint_upc_candidates": hints["upc12_candidates"],
                    "label_hint_last5_candidates": hints["last5_candidates"],
                }

        # Secondary recovery retained for compatibility: name-only best candidate.
        if label and all_matrix_rows:
            fallback = _best_row_for_label(all_matrix_rows, label)
            if fallback is not None:
                ratio = difflib.SequenceMatcher(None, _norm_name(label), fallback.norm_name).ratio()
                if ratio >= 0.78:
                    return fallback, {
                        "fallback_path": "labels_name_similarity",
                        "labels_last5": extracted_last5,
                        "label_hint_upc_candidates": hints["upc12_candidates"],
                        "label_hint_last5_candidates": hints["last5_candidates"],
                    }

        return None, {
            "fallback_path": "unresolved",
            "labels_last5": extracted_last5,
            "label_hint_upc_candidates": hints["upc12_candidates"],
            "label_hint_last5_candidates": hints["last5_candidates"],
        }

    def _build_mid_slot_position_lookup() -> Tuple[
        Dict[Tuple[str, int, int], List[MatrixRow]],
        Dict[Tuple[int, int], List[MatrixRow]],
    ]:
        per_side_candidates: Dict[Tuple[str, int, int], Dict[str, MatrixRow]] = {}
        global_candidates: Dict[Tuple[int, int], Dict[str, MatrixRow]] = {}

        for page in pages:
            section = page.mid_band_above_bonus
            if section is None:
                continue
            if len(section.rows) != 3 or section.row_slot_counts != [8, 8, 8]:
                continue
            for row in section.rows:
                if len(row.slots) != 8:
                    continue
                for slot in row.slots:
                    match, _ = _resolve_mid_band_slot_no_position(slot)
                    if match is None:
                        continue
                    per_side_key = (page.side_letter, int(slot.row_index), int(slot.slot_in_row))
                    global_key = (int(slot.row_index), int(slot.slot_in_row))
                    per_side_candidates.setdefault(per_side_key, {})[match.upc12] = match
                    global_candidates.setdefault(global_key, {})[match.upc12] = match

        per_side_lookup: Dict[Tuple[str, int, int], List[MatrixRow]] = {
            key: list(rows.values()) for key, rows in per_side_candidates.items()
        }
        global_lookup: Dict[Tuple[int, int], List[MatrixRow]] = {
            key: list(rows.values()) for key, rows in global_candidates.items()
        }

        return per_side_lookup, global_lookup

    mid_slot_lookup_by_side, mid_slot_lookup_global = _build_mid_slot_position_lookup()

    def _resolve_mid_slot_position_candidates(
        slot: FullPalletMidBandSlot,
        candidates: List[MatrixRow],
    ) -> Tuple[Optional[MatrixRow], Dict[str, object]]:
        candidate_upcs: List[str] = []
        similarity_scores: Dict[str, float] = {}
        if not candidates:
            return None, {
                "fallback_candidate_upcs": candidate_upcs,
                "fallback_similarity_scores": similarity_scores,
                "chosen_candidate": None,
            }

        parsed_target = _norm_name(slot.parsed_name or "")
        raw_target = _norm_name(slot.raw_label_text or "")

        ranked: List[Tuple[float, float, MatrixRow]] = []
        for row in candidates:
            upc = (row.upc12 or "").strip()
            candidate_upcs.append(upc)
            row_name = row.norm_name or ""
            parsed_ratio = (
                difflib.SequenceMatcher(None, parsed_target, row_name).ratio() if parsed_target else -1.0
            )
            raw_ratio = difflib.SequenceMatcher(None, raw_target, row_name).ratio() if raw_target else -1.0
            primary_ratio = parsed_ratio if parsed_target else raw_ratio
            secondary_ratio = raw_ratio if parsed_target else -1.0
            ranked.append((primary_ratio, secondary_ratio, row))
            similarity_scores[upc] = float(primary_ratio if primary_ratio >= 0.0 else 0.0)

        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_primary, _, best_row = ranked[0]
        threshold = 0.60
        if best_primary >= threshold:
            return best_row, {
                "fallback_candidate_upcs": candidate_upcs,
                "fallback_similarity_scores": similarity_scores,
                "chosen_candidate": best_row.upc12,
            }

        return None, {
            "fallback_candidate_upcs": candidate_upcs,
            "fallback_similarity_scores": similarity_scores,
            "chosen_candidate": None,
        }

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

    def _prepare_mid_band_image_draw(
        img: Optional[Image.Image],
        source_crop_bbox: Optional[Tuple[float, float, float, float]],
        card_bbox: Tuple[float, float, float, float],
        image_area_bbox: Tuple[float, float, float, float],
        *,
        side: str,
        final_index: int,
        row: int,
        col: int,
        upc12: str,
    ) -> Dict[str, object]:
        card_x0, card_y0, card_x1, card_y1 = card_bbox
        area_x0, area_y0, area_x1, area_y1 = image_area_bbox
        area_w = max(1.0, area_x1 - area_x0)
        area_h = max(1.0, area_y1 - area_y0)
        notes: List[str] = []
        proposed_trim_bbox: Optional[List[int]] = None
        original_size: Optional[List[int]] = None
        thin_crop_detected = False
        excessive_whitespace_detected = False
        image_draw_bbox: Optional[List[float]] = None

        if img is None:
            notes.append("missing_image")
            return {
                "side": side,
                "final_index": final_index,
                "row": row,
                "col": col,
                "upc12": upc12,
                "source_crop_bbox": list(source_crop_bbox) if source_crop_bbox else None,
                "original_crop_size": original_size,
                "proposed_trimmed_crop_bbox": proposed_trim_bbox,
                "trim_applied": False,
                "image_fit_mode": "contain",
                "card_bbox": [round(card_x0, 2), round(card_y0, 2), round(card_x1, 2), round(card_y1, 2)],
                "image_area_bbox": [round(area_x0, 2), round(area_y0, 2), round(area_x1, 2), round(area_y1, 2)],
                "proposed_image_draw_bbox": image_draw_bbox,
                "thin_crop_detected": thin_crop_detected,
                "excessive_whitespace_detected": excessive_whitespace_detected,
                "overflow_or_bleed_detected": False,
                "normalization_notes": notes,
            }

        try:
            src = img.convert("RGB")
            arr = np.asarray(src)
            h_px, w_px = arr.shape[:2]
            original_size = [int(w_px), int(h_px)]
            aspect = w_px / max(1, h_px)
            thin_crop_detected = bool(aspect < 0.42 or aspect > 2.85)
            if thin_crop_detected:
                notes.append("thin_or_strip_like_crop_detected")

            if w_px < 18 or h_px < 18:
                notes.append("trim_plan_skipped_tiny_image")
            else:
                edge = max(2, min(10, int(min(w_px, h_px) * 0.04)))
                samples = np.concatenate(
                    [
                        arr[:edge, :, :].reshape(-1, 3),
                        arr[-edge:, :, :].reshape(-1, 3),
                        arr[:, :edge, :].reshape(-1, 3),
                        arr[:, -edge:, :].reshape(-1, 3),
                    ],
                    axis=0,
                )
                bg = np.median(samples, axis=0)
                diff = np.max(np.abs(arr.astype(np.int16) - bg.astype(np.int16)), axis=2)
                not_bg = diff > 18
                not_near_white = np.min(arr, axis=2) < 246
                content = not_bg & not_near_white
                ys, xs = np.where(content)
                if len(xs) == 0 or len(ys) == 0:
                    notes.append("trim_plan_skipped_no_content_mask")
                else:
                    left = int(xs.min())
                    right = int(xs.max()) + 1
                    top = int(ys.min())
                    bottom = int(ys.max()) + 1
                    max_trim_x = int(w_px * 0.18)
                    max_trim_y = int(h_px * 0.18)
                    left = min(left, max_trim_x)
                    top = min(top, max_trim_y)
                    right = max(right, w_px - max_trim_x)
                    bottom = max(bottom, h_px - max_trim_y)
                    trim_x = left + max(0, w_px - right)
                    trim_y = top + max(0, h_px - bottom)
                    excessive_whitespace_detected = bool(trim_x > w_px * 0.16 or trim_y > h_px * 0.16)
                    if excessive_whitespace_detected:
                        notes.append("excessive_outer_whitespace_detected")
                    if right <= left or bottom <= top:
                        notes.append("trim_plan_skipped_invalid_bbox")
                    elif (right - left) < w_px * 0.45 or (bottom - top) < h_px * 0.45:
                        notes.append("trim_plan_skipped_aggressive_mask")
                    elif (left, top, right, bottom) != (0, 0, w_px, h_px):
                        proposed_trim_bbox = [left, top, right, bottom]
                        notes.append("conservative_trim_candidate_detected")

            fit = min(area_w / max(1, w_px), area_h / max(1, h_px))
            dw = max(1.0, w_px * fit)
            dh = max(1.0, h_px * fit)
            dx = area_x0 + (area_w - dw) / 2.0
            dy = area_y0 + (area_h - dh) / 2.0
            image_draw_bbox = [round(dx, 2), round(dy, 2), round(dx + dw, 2), round(dy + dh, 2)]
            overflow = (
                image_draw_bbox[0] < card_x0 - 0.001
                or image_draw_bbox[1] < card_y0 - 0.001
                or image_draw_bbox[2] > card_x1 + 0.001
                or image_draw_bbox[3] > card_y1 + 0.001
            )
        except Exception:
            notes.append("normalization_plan_error")
            overflow = False

        return {
            "side": side,
            "final_index": final_index,
            "row": row,
            "col": col,
            "upc12": upc12,
            "source_crop_bbox": list(source_crop_bbox) if source_crop_bbox else None,
            "original_crop_size": original_size,
            "proposed_trimmed_crop_bbox": proposed_trim_bbox,
            "trim_applied": False,
            "image_fit_mode": "contain",
            "card_bbox": [round(card_x0, 2), round(card_y0, 2), round(card_x1, 2), round(card_y1, 2)],
            "image_area_bbox": [round(area_x0, 2), round(area_y0, 2), round(area_x1, 2), round(area_y1, 2)],
            "proposed_image_draw_bbox": image_draw_bbox,
            "thin_crop_detected": thin_crop_detected,
            "excessive_whitespace_detected": excessive_whitespace_detected,
            "overflow_or_bleed_detected": bool(overflow),
            "normalization_notes": notes,
        }

    def _draw_mid_band_card(
        c: canvas.Canvas,
        x: float,
        y: float,
        w: float,
        h: float,
        img: Optional[Image.Image],
        upc12: str,
        name: str,
        cpp: Optional[int],
        *,
        section: str,
        side: str,
        final_index: int,
        row: int,
        col: int,
        source_crop_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Dict[str, object]:
        if section == "bonus":
            pad = max(3.5, min(5.0, w * 0.060))
        else:
            pad = max(4.0, min(6.0, w * 0.075))
        ix = x + pad
        iy = y + pad
        iw = max(1.0, w - 2 * pad)
        ih = max(1.0, h - 2 * pad)
        text_h = max(18.0, min(24.0, ih * 0.26)) if section == "bonus" else max(20.0, min(28.0, ih * 0.30))
        text_gap = 2.2 if section == "bonus" else 3.0
        img_h = max(8.0, ih - text_h - text_gap)
        img_x = ix
        img_y = iy + text_h + text_gap
        image_inner_inset = max(1.5, min(3.0, w * 0.035)) if section == "bonus" else 0.0
        image_draw_bbox: Optional[List[float]] = None

        card_bbox = (x, y, x + w, y + h)
        image_area_bbox = (
            img_x + image_inner_inset,
            img_y + image_inner_inset,
            img_x + iw - image_inner_inset,
            img_y + img_h - image_inner_inset,
        )
        draw_plan = _prepare_mid_band_image_draw(
            img,
            source_crop_bbox,
            card_bbox,
            image_area_bbox,
            side=side,
            final_index=final_index,
            row=row,
            col=col,
            upc12=upc12,
        )
        draw_img = img
        trim_bbox = draw_plan.get("proposed_trimmed_crop_bbox")
        initial_plan = dict(draw_plan)
        trim_applied = False
        if img is not None and trim_bbox is not None:
            try:
                l, t, r, b = [int(v) for v in trim_bbox]
                if r > l and b > t:
                    draw_img = img.convert("RGB").crop((l, t, r, b))
                    trim_applied = True
                    draw_plan = _prepare_mid_band_image_draw(
                        draw_img,
                        source_crop_bbox,
                        card_bbox,
                        image_area_bbox,
                        side=side,
                        final_index=final_index,
                        row=row,
                        col=col,
                        upc12=upc12,
                    )
                    draw_plan["proposed_trimmed_crop_bbox"] = trim_bbox
                    draw_plan["trimmed_crop_bbox"] = trim_bbox
                    draw_plan["trim_applied"] = True
                    draw_plan["original_crop_size"] = initial_plan.get("original_crop_size")
                    draw_plan["thin_crop_detected"] = bool(
                        initial_plan.get("thin_crop_detected") or draw_plan.get("thin_crop_detected")
                    )
                    draw_plan["excessive_whitespace_detected"] = bool(
                        initial_plan.get("excessive_whitespace_detected")
                        or draw_plan.get("excessive_whitespace_detected")
                    )
                    draw_plan["normalization_notes"] = list(initial_plan.get("normalization_notes", [])) + list(
                        draw_plan.get("normalization_notes", [])
                    ) + [
                        "conservative_trim_applied"
                    ]
            except Exception:
                draw_img = img
                draw_plan["normalization_notes"] = list(draw_plan.get("normalization_notes", [])) + [
                    "trim_apply_failed"
                ]

        image_draw_bbox = draw_plan.get("proposed_image_draw_bbox")
        overflow_before_clamp = False
        clamped_to_fit = False
        final_contained = True
        residual_bleed_warning = False
        if section in {"mid_band", "bonus"} and image_draw_bbox is not None:
            ax0, ay0, ax1, ay1 = image_area_bbox
            cx0, cy0, cx1, cy1 = card_bbox
            dx0, dy0, dx1, dy1 = [float(v) for v in image_draw_bbox]
            overflow_before_clamp = bool(
                dx0 < ax0 - 0.001
                or dy0 < ay0 - 0.001
                or dx1 > ax1 + 0.001
                or dy1 > ay1 + 0.001
                or dx0 < cx0 - 0.001
                or dy0 < cy0 - 0.001
                or dx1 > cx1 + 0.001
                or dy1 > cy1 + 0.001
            )

            src_w = max(1.0, dx1 - dx0)
            src_h = max(1.0, dy1 - dy0)
            max_w = max(1.0, ax1 - ax0)
            max_h = max(1.0, ay1 - ay0)
            containment_scale = 0.965 if section == "bonus" else 0.985
            scale = min(1.0, max_w / src_w, max_h / src_h) * containment_scale
            new_w = max(1.0, src_w * scale)
            new_h = max(1.0, src_h * scale)
            ndx0 = ax0 + (max_w - new_w) / 2.0
            ndy0 = ay0 + (max_h - new_h) / 2.0
            ndx1 = ndx0 + new_w
            ndy1 = ndy0 + new_h
            clamped_to_fit = bool(
                overflow_before_clamp
                or abs(ndx0 - dx0) > 0.01
                or abs(ndy0 - dy0) > 0.01
                or abs(ndx1 - dx1) > 0.01
                or abs(ndy1 - dy1) > 0.01
            )
            image_draw_bbox = [round(ndx0, 2), round(ndy0, 2), round(ndx1, 2), round(ndy1, 2)]
            draw_plan["proposed_image_draw_bbox"] = image_draw_bbox
            draw_plan["image_draw_bbox"] = image_draw_bbox
            draw_plan["normalization_notes"] = list(draw_plan.get("normalization_notes", [])) + (
                [f"{section}_containment_clamp"] if clamped_to_fit else []
            )
            final_contained = bool(
                image_draw_bbox[0] >= ax0 - 0.001
                and image_draw_bbox[1] >= ay0 - 0.001
                and image_draw_bbox[2] <= ax1 + 0.001
                and image_draw_bbox[3] <= ay1 + 0.001
                and image_draw_bbox[0] >= cx0 - 0.001
                and image_draw_bbox[1] >= cy0 - 0.001
                and image_draw_bbox[2] <= cx1 + 0.001
                and image_draw_bbox[3] <= cy1 + 0.001
            )
            residual_bleed_warning = not final_contained

        if draw_img is not None and image_draw_bbox is not None:
            dx0, dy0, dx1, dy1 = [float(v) for v in image_draw_bbox]
            if section == "bonus":
                c.saveState()
                clip = c.beginPath()
                ax0, ay0, ax1, ay1 = image_area_bbox
                clip.rect(ax0, ay0, max(1.0, ax1 - ax0), max(1.0, ay1 - ay0))
                c.clipPath(clip, stroke=0, fill=0)
            try:
                c.drawImage(
                    ImageReader(draw_img),
                    dx0,
                    dy0,
                    max(1.0, dx1 - dx0),
                    max(1.0, dy1 - dy0),
                    preserveAspectRatio=True,
                    mask="auto",
                )
            finally:
                if section == "bonus":
                    c.restoreState()

        cpp_str = f"CPP: {cpp}" if cpp is not None else "CPP:"
        upc = (upc12 or "").strip()
        cpp_h = max(8.0, text_h * 0.30)
        name_h = max(8.0, text_h * 0.28)
        upc_h = max(8.0, text_h - cpp_h - name_h)

        upc_fs = _fit_font(upc, BODY_BOLD_FONT, iw, upc_h, 5.5, 12.0, step=0.25)
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.setFont(BODY_BOLD_FONT, upc_fs)
        tw = pdfmetrics.stringWidth(upc, BODY_BOLD_FONT, upc_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + name_h + (upc_h - upc_fs) / 2, upc)

        name_fs = _fit_font((name or "").upper(), BODY_FONT, iw, name_h, 5.3, 6.6, step=0.1)
        nm = _fit_name_preserve_qualifiers((name or "").upper(), BODY_FONT, name_fs, iw)
        c.setFillColorRGB(0.10, 0.10, 0.10)
        c.setFont(BODY_FONT, name_fs)
        tw = pdfmetrics.stringWidth(nm, BODY_FONT, name_fs)
        c.drawString(ix + (iw - tw) / 2, iy + cpp_h + (name_h - name_fs) / 2, nm)

        cpp_fs = _fit_font(cpp_str, BODY_BOLD_FONT, iw, cpp_h, 5.5, 10.5, step=0.25)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.setFont(BODY_BOLD_FONT, cpp_fs)
        tw = pdfmetrics.stringWidth(cpp_str, BODY_BOLD_FONT, cpp_fs)
        c.drawString(ix + (iw - tw) / 2, iy + (cpp_h - cpp_fs) / 2, cpp_str)

        text_bbox = [round(ix, 2), round(iy, 2), round(ix + iw, 2), round(iy + text_h, 2)]
        draw_plan["section"] = section
        draw_plan["text_bbox"] = text_bbox
        draw_plan["image_draw_bbox"] = draw_plan.get("proposed_image_draw_bbox")
        draw_plan["slot_rect"] = draw_plan.get("card_bbox")
        draw_plan["inset_used"] = round(float(image_inner_inset), 2)
        draw_plan["fit_mode_used"] = "contain"
        draw_plan["hard_clip_applied"] = bool(section == "bonus")
        draw_plan["draw_rect_reduced_to_avoid_overflow"] = bool(clamped_to_fit)
        draw_plan["overflow_before_clamp"] = bool(overflow_before_clamp)
        draw_plan["clamped_to_fit"] = bool(clamped_to_fit)
        draw_plan["final_contained"] = bool(final_contained)
        draw_plan["residual_bleed_warning"] = bool(residual_bleed_warning)
        if section in {"mid_band", "bonus"}:
            draw_plan["overflow_or_bleed_detected"] = bool(residual_bleed_warning)
        draw_plan["normalization_applied"] = bool(
            trim_applied
            or clamped_to_fit
            or draw_plan.get("thin_crop_detected")
            or draw_plan.get("excessive_whitespace_detected")
        )
        draw_plan["trim_applied"] = bool(trim_applied)
        draw_plan.setdefault("trimmed_crop_bbox", trim_bbox if trim_applied else None)
        return draw_plan

    def _section_shape_policy(section_kind: str) -> Dict[str, float]:
        if section_kind == "main":
            return {
                "desired_card_w": 64.0,
                "desired_gap": 6.0,
                "row_gutter": 8.0,
                "card_ratio": 1.10,   # h / w
                "min_card_h": 56.0,
                "max_card_h": 92.0,
                "crop_zoom": 2.40,
                "crop_inset": 0.018,
            }

        return {
            "desired_card_w": 72.0,
            "desired_gap": 4.0,
            "row_gutter": 6.0,
            "card_ratio": 1.20,   # h / w
            "min_card_h": 62.0,
            "max_card_h": 102.0,
            "crop_zoom": 3.00,
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
        normalization_debug_rows: Optional[List[dict]] = None,
        bonus_crop_debug_rows: Optional[List[dict]] = None,
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
        is_bonus_section = str(label or "").upper() == "BONUS"
        pixmap_page_img: Optional[Image.Image] = None
        pixmap_page_zoom = 3.0
        if is_bonus_section:
            try:
                img_page = images_doc[p.page_index]
                pixmap_page_img, pixmap_page_zoom = _render_page_pixmap_image(img_page, zoom=float(plan["crop_zoom"]))
            except Exception:
                pixmap_page_img = None

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

                image_crop_bbox_used = cell.bbox
                img: Optional[Image.Image] = None
                crop_error: Optional[str] = None
                pixmap_debug: Dict[str, object] = {}
                render_path_used = "pdf_object_crop"
                fallback_used = False
                source_crop_id = f"r{cell.row}c{cell.col}"
                original_source_bbox = cell.bbox
                sanitized_source_bbox = cell.bbox
                source_inset_debug: Dict[str, object] = {}
                image_sanitize_debug: Dict[str, object] = {}
                if is_bonus_section and pixmap_page_img is not None:
                    sanitized_source_bbox, source_inset_debug = _inset_bonus_source_bbox(cell.bbox)
                    pix_img, pix_bbox, pixmap_debug = _crop_mid_band_pixmap_slot(
                        pixmap_page_img,
                        pixmap_page_zoom,
                        sanitized_source_bbox,
                    )
                    if pix_img is not None and pix_bbox is not None:
                        img, image_sanitize_debug = _sanitize_bonus_crop_image(pix_img)
                        image_crop_bbox_used = pix_bbox
                        render_path_used = "pixmap_bonus"
                    else:
                        fallback_used = True

                if img is None:
                    try:
                        img = crop_image_cell(
                            images_doc,
                            p.page_index,
                            sanitized_source_bbox if is_bonus_section else cell.bbox,
                            zoom=float(plan["crop_zoom"]),
                            inset=float(plan["crop_inset"]),
                        )
                        if is_bonus_section:
                            img, image_sanitize_debug = _sanitize_bonus_crop_image(img)
                            image_crop_bbox_used = sanitized_source_bbox
                        render_path_used = "fallback_previous_path" if fallback_used else render_path_used
                    except Exception as exc:
                        img = None
                        crop_error = str(exc)
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
                norm_debug = _draw_mid_band_card(
                    c,
                    x,
                    y,
                    card_w,
                    card_h,
                    img,
                    upc_str,
                    disp_name,
                    cpp,
                    section="bonus",
                    side=p.side_letter,
                    final_index=ri * n_cols + ci,
                    row=ri,
                    col=ci,
                    source_crop_bbox=image_crop_bbox_used,
                )
                if normalization_debug_rows is not None:
                    normalization_debug_rows.append(norm_debug)
                if is_bonus_section and bonus_crop_debug_rows is not None:
                    bw = float(image_crop_bbox_used[2] - image_crop_bbox_used[0]) if image_crop_bbox_used else 0.0
                    bh = float(image_crop_bbox_used[3] - image_crop_bbox_used[1]) if image_crop_bbox_used else 0.0
                    bonus_crop_debug_rows.append(
                        {
                            "side": p.side_letter,
                            "section": "bonus",
                            "slot_index": int(ri * n_cols + ci),
                            "row": int(ri),
                            "col": int(ci),
                            "source_page_index": p.page_index,
                            "slot_id": source_crop_id,
                            "upc12": upc12,
                            "render_path_used": render_path_used,
                            "source_crop_id": source_crop_id,
                            "original_source_bbox": list(original_source_bbox) if original_source_bbox else None,
                            "source_crop_bbox": list(image_crop_bbox_used) if image_crop_bbox_used else None,
                            "cleaned_crop_bbox": list(image_crop_bbox_used) if image_crop_bbox_used else None,
                            "sanitized_source_bbox": list(sanitized_source_bbox) if sanitized_source_bbox else None,
                            "source_inset_applied": bool(source_inset_debug.get("source_inset_applied")),
                            "source_inset_x": source_inset_debug.get("source_inset_x", 0.0),
                            "source_inset_y": source_inset_debug.get("source_inset_y", 0.0),
                            "trim_inset_applied": bool(
                                source_inset_debug.get("source_inset_applied")
                                or image_sanitize_debug.get("sanitized_crop_used")
                            ),
                            "neighboring_edge_contamination_detected": bool(
                                image_sanitize_debug.get("neighboring_edge_contamination_detected")
                            ),
                            "image_edge_trim_bbox": image_sanitize_debug.get("image_edge_trim_bbox"),
                            "image_edge_trim_px": image_sanitize_debug.get("image_edge_trim_px", [0, 0, 0, 0]),
                            "crop_clipped_to_hard_bounds": bool(norm_debug.get("hard_clip_applied")),
                            "sanitized_crop_used": bool(
                                source_inset_debug.get("source_inset_applied")
                                or image_sanitize_debug.get("sanitized_crop_used")
                            ),
                            "source_crop_width": bw,
                            "source_crop_height": bh,
                            "crop_pixel_width": int(getattr(img, "width")) if img is not None and hasattr(img, "width") else None,
                            "crop_pixel_height": int(getattr(img, "height")) if img is not None and hasattr(img, "height") else None,
                            "contain_fit_used": True,
                            "fit_mode_used": norm_debug.get("fit_mode_used", "contain"),
                            "inset_used": norm_debug.get("inset_used"),
                            "hard_clip_applied": norm_debug.get("hard_clip_applied"),
                            "draw_rect_reduced_to_avoid_overflow": norm_debug.get("draw_rect_reduced_to_avoid_overflow"),
                            "suspicious_crop": bool(pixmap_debug.get("suspicious_crop", False)),
                            "suspicious_reason": pixmap_debug.get("suspicious_reason", []),
                            "fallback_used": bool(fallback_used),
                            "blank_or_missing_crop": bool(img is None),
                            "slot_rect": norm_debug.get("slot_rect", norm_debug.get("card_bbox")),
                            "card_bbox": norm_debug.get("card_bbox"),
                            "image_draw_bbox": norm_debug.get("image_draw_bbox"),
                            "text_bbox": norm_debug.get("text_bbox"),
                            "overflow_or_bleed_detected": bool(norm_debug.get("residual_bleed_warning", False)),
                            "crop_error": crop_error,
                        }
                    )

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
        min_card_h = 56.0
        max_card_h = 90.0
        card_ratio = 1.10  # h / w
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

    def _measure_token_first_mid_band(
        content_w: float,
        slot_count: int,
        include_bar: bool = False,
    ) -> Dict[str, float]:
        intra_gap = 6.0
        inter_gap = 14.0
        col_gap = 6.0
        row_gutter = 8.0
        min_card_h = 56.0
        max_card_h = 90.0
        card_ratio = 1.10  # h / w
        crop_zoom = 2.40
        crop_inset = 0.018

        if slot_count <= 0:
            return {
                "cols": 0,
                "rows": 0,
                "card_w": 0.0,
                "card_h": 0.0,
                "col_gap": col_gap,
                "intra_gap": intra_gap,
                "inter_gap": inter_gap,
                "row_gutter": row_gutter,
                "total_w": 0.0,
                "total_h": 0.0,
                "overflow": False,
                "crop_zoom": crop_zoom,
                "crop_inset": crop_inset,
                "layout_mode": "empty",
            }

        avail = max(1.0, content_w)
        cols = min(8, max(1, slot_count))
        while cols > 1:
            if cols == 8:
                probe_w = (avail - (5.0 * intra_gap) - (2.0 * inter_gap)) / 8.0
            else:
                probe_w = (avail - (cols - 1) * col_gap) / cols
            if probe_w >= 24.0:
                break
            cols -= 1

        if cols == 8:
            card_w = max(24.0, (avail - (5.0 * intra_gap) - (2.0 * inter_gap)) / 8.0)
            total_w = 8.0 * card_w + 5.0 * intra_gap + 2.0 * inter_gap
            layout_mode = "canonical_8col_spacing"
        else:
            card_w = max(24.0, (avail - (cols - 1) * col_gap) / cols)
            total_w = cols * card_w + max(0, cols - 1) * col_gap
            layout_mode = "uniform_wrap_spacing"
        card_h = max(min_card_h, min(max_card_h, card_w * card_ratio))
        rows = int(math.ceil(slot_count / cols))
        total_h = rows * card_h + max(0, rows - 1) * row_gutter
        if include_bar:
            total_h += SECTION_BAR_H + SECTION_BAR_GAP

        return {
            "cols": cols,
            "rows": rows,
            "card_w": card_w,
            "card_h": card_h,
            "col_gap": col_gap,
            "intra_gap": intra_gap,
            "inter_gap": inter_gap,
            "row_gutter": row_gutter,
            "total_w": total_w,
            "total_h": total_h,
            "overflow": total_w > (avail + 0.001),
            "crop_zoom": crop_zoom,
            "crop_inset": crop_inset,
            "layout_mode": layout_mode,
        }

    def _mid_band_row_xs(
        start_x: float,
        card_w: float,
        cols: int,
        intra_gap: float,
        inter_gap: float,
        col_gap: float,
    ) -> List[float]:
        if cols == 8:
            return [
                start_x,
                start_x + card_w + intra_gap,
                start_x + 2 * card_w + intra_gap + inter_gap,
                start_x + 3 * card_w + 2 * intra_gap + inter_gap,
                start_x + 4 * card_w + 3 * intra_gap + inter_gap,
                start_x + 5 * card_w + 4 * intra_gap + inter_gap,
                start_x + 6 * card_w + 5 * intra_gap + 2 * inter_gap,
                start_x + 7 * card_w + 6 * intra_gap + 2 * inter_gap,
            ]
        return [start_x + i * (card_w + col_gap) for i in range(cols)]

    def _build_anchor_mid_band_slot_grid(
        anchor_bbox: Optional[Tuple[float, float, float, float]],
        page_width: float,
        page_height: float,
    ) -> Optional[List[List[Tuple[float, float, float, float]]]]:
        if anchor_bbox is None:
            return None
        x0, top, x1, bottom = anchor_bbox
        section_w = float(x1 - x0)
        section_h = float(bottom - top)
        if section_w <= 0.0 or section_h <= 0.0:
            return None

        inter_gap = max(10.0, min(28.0, section_w * 0.045))
        intra_gap = max(4.0, min(12.0, section_w * 0.014))
        row_gap = max(6.0, min(14.0, section_h * 0.08))
        slot_w = (section_w - (2.0 * inter_gap) - (5.0 * intra_gap)) / 8.0
        row_h = max(40.0, min(62.0, section_h * 0.28))
        needed_h = 3.0 * row_h + 2.0 * row_gap
        if slot_w < 18.0 or section_h < 90.0:
            return None
        if needed_h > section_h:
            row_h = (section_h - 2.0 * row_gap) / 3.0
        if row_h < 24.0:
            return None

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
            ry1 = ry0 + row_h
            if r == 2 and ry1 > bottom:
                overflow = ry1 - bottom
                ry0 -= overflow
                ry1 -= overflow
            rows.append(
                [
                    (
                        max(0.0, float(sx)),
                        max(0.0, float(ry0)),
                        min(page_width, float(sx + slot_w)),
                        min(page_height, float(ry1)),
                    )
                    for sx in row_xs
                ]
            )
        return rows

    def _derive_token_first_fallback_image_bbox(
        slot: FullPalletMidBandSlot,
        section: FullPalletMidBandSection,
        page_width: float,
        page_height: float,
    ) -> Tuple[float, float, float, float]:
        all_slots = [s for r in (section.rows or []) for s in (r.slots or [])]
        centers = [(s, (float(s.bbox[0] + s.bbox[2]) / 2.0, float(s.bbox[1] + s.bbox[3]) / 2.0)) for s in all_slots if s.bbox]

        row_x_gaps: List[float] = []
        for row in section.rows or []:
            xs = sorted([(float(s.bbox[0] + s.bbox[2]) / 2.0) for s in row.slots if s.bbox])
            if len(xs) >= 2:
                row_x_gaps.extend([xs[i + 1] - xs[i] for i in range(len(xs) - 1) if (xs[i + 1] - xs[i]) > 1.0])

        row_centers: List[float] = []
        for row in section.rows or []:
            ys = [(float(s.bbox[1] + s.bbox[3]) / 2.0) for s in row.slots if s.bbox]
            if ys:
                row_centers.append(float(np.mean(ys)))
        row_centers = sorted(set(row_centers))
        row_y_gaps = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1) if (row_centers[i + 1] - row_centers[i]) > 1.0]

        median_x_spacing = float(np.median(row_x_gaps)) if row_x_gaps else None
        median_y_spacing = float(np.median(row_y_gaps)) if row_y_gaps else None

        if median_x_spacing is None and len(centers) >= 2:
            xs_all = sorted([c[0] for _s, c in centers])
            all_x_gaps = [xs_all[i + 1] - xs_all[i] for i in range(len(xs_all) - 1) if (xs_all[i + 1] - xs_all[i]) > 1.0]
            median_x_spacing = float(np.median(all_x_gaps)) if all_x_gaps else None

        crop_w = max(36.0, min(72.0, (median_x_spacing * 0.90) if median_x_spacing is not None else 54.0))
        crop_h = max(36.0, min(58.0, (median_y_spacing * 0.90) if median_y_spacing is not None else 44.0))

        sx0, sy0, sx1, sy1 = slot.bbox
        cx = float(sx0 + sx1) / 2.0
        cy = float(sy0 + sy1) / 2.0
        center_y_for_crop = cy - crop_h * 0.25

        x0 = cx - crop_w / 2.0
        x1 = cx + crop_w / 2.0
        y0 = center_y_for_crop - crop_h / 2.0
        y1 = center_y_for_crop + crop_h / 2.0

        # Clamp to anchor range first (with allowance), then to page bounds.
        ax0_lim = 0.0
        ay0_lim = 0.0
        ax1_lim = page_width
        ay1_lim = page_height
        if section.anchor_bbox is not None:
            ax0, ay0, ax1, ay1 = section.anchor_bbox
            ax0_lim = max(0.0, float(ax0) - 30.0)
            ay0_lim = max(0.0, float(ay0) - 20.0)
            ax1_lim = min(page_width, float(ax1) + 30.0)
            ay1_lim = min(page_height, float(ay1) + 20.0)

        def _shift_within(v0: float, v1: float, lo: float, hi: float) -> Tuple[float, float]:
            if v0 < lo:
                dv = lo - v0
                v0 += dv
                v1 += dv
            if v1 > hi:
                dv = v1 - hi
                v0 -= dv
                v1 -= dv
            return v0, v1

        x0, x1 = _shift_within(x0, x1, ax0_lim, ax1_lim)
        y0, y1 = _shift_within(y0, y1, ay0_lim, ay1_lim)
        x0, x1 = _shift_within(x0, x1, 0.0, page_width)
        y0, y1 = _shift_within(y0, y1, 0.0, page_height)

        # Final safety: preserve a minimum crop footprint.
        if (x1 - x0) < 30.0:
            half = 15.0
            x0 = max(0.0, cx - half)
            x1 = min(page_width, cx + half)
        if (y1 - y0) < 30.0:
            half = 15.0
            y0 = max(0.0, center_y_for_crop - half)
            y1 = min(page_height, center_y_for_crop + half)

        return (float(x0), float(y0), float(x1), float(y1))

    def _derive_token_first_image_bbox(
        slot: FullPalletMidBandSlot,
        section: FullPalletMidBandSection,
        page_width: float,
        page_height: float,
    ) -> Tuple[Tuple[float, float, float, float], str, bool]:
        grid = _build_anchor_mid_band_slot_grid(section.anchor_bbox, page_width, page_height)
        if grid:
            sx0, sy0, sx1, sy1 = slot.bbox
            tcx = float(sx0 + sx1) / 2.0
            tcy = float(sy0 + sy1) / 2.0
            nearest_bbox: Optional[Tuple[float, float, float, float]] = None
            nearest_dist = float("inf")
            for row in grid:
                for b in row:
                    bx0, by0, bx1, by1 = b
                    bcx = (bx0 + bx1) / 2.0
                    bcy = (by0 + by1) / 2.0
                    d = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_bbox = b
            if nearest_bbox is not None:
                bx0, by0, bx1, by1 = nearest_bbox
                bw = max(1.0, bx1 - bx0)
                bh = max(1.0, by1 - by0)
                crop_w = max(36.0, min(72.0, bw * 0.95))
                crop_h = max(36.0, min(58.0, bh * 1.05))
                cx = (bx0 + bx1) / 2.0
                cy = (by0 + by1) / 2.0 - crop_h * 0.20
                x0 = cx - crop_w / 2.0
                x1 = cx + crop_w / 2.0
                y0 = cy - crop_h / 2.0
                y1 = cy + crop_h / 2.0
                # clamp to page bounds
                x0 = max(0.0, x0)
                y0 = max(0.0, y0)
                x1 = min(page_width, x1)
                y1 = min(page_height, y1)
                if (x1 - x0) >= 30.0 and (y1 - y0) >= 30.0:
                    return (float(x0), float(y0), float(x1), float(y1)), "mid_band_slot_grid_bbox", True

        fb = _derive_token_first_fallback_image_bbox(slot, section, page_width, page_height)
        return fb, "derived_token_cell_bbox_fallback", False

    def _compute_mid_band_image_region_bbox(
        p: FullPalletPage,
        section: FullPalletMidBandSection,
        page_width: float,
        page_height: float,
    ) -> Tuple[float, float, float, float]:
        slots = [s for r in section.rows for s in r.slots if s.bbox]
        if section.anchor_bbox is not None:
            ax0, ay0, ax1, ay1 = section.anchor_bbox
            x0 = max(0.0, float(ax0) - 24.0)
            y0 = max(0.0, float(ay0) - 12.0)
            x1 = min(page_width, float(ax1) + 24.0)
            y1 = min(page_height, float(ay1) + 14.0)
        elif slots:
            x0 = max(0.0, min(float(s.bbox[0]) for s in slots) - 18.0)
            y0 = max(0.0, min(float(s.bbox[1]) for s in slots) - 14.0)
            x1 = min(page_width, max(float(s.bbox[2]) for s in slots) + 18.0)
            y1 = min(page_height, max(float(s.bbox[3]) for s in slots) + 14.0)
        else:
            x0, y0, x1, y1 = 0.0, 0.0, page_width, page_height

        bonus_y = _marker_y(p, "bonus_strip")
        if bonus_y is not None:
            y1 = min(y1, float(bonus_y) - 8.0)
        if y1 <= y0 + 1.0:
            y0 = max(0.0, y0 - 20.0)
            y1 = min(page_height, y0 + 120.0)
        return (float(x0), float(y0), float(x1), float(y1))

    def _detect_mid_band_image_cells(
        image_page: fitz.Page,
        region_bbox: Tuple[float, float, float, float],
    ) -> List[dict]:
        rx0, ry0, rx1, ry1 = region_bbox

        def _intersect_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            return float((ix1 - ix0) * (iy1 - iy0))

        blocks = image_page.get_text("dict").get("blocks", []) or []
        candidates: List[dict] = []
        for blk in blocks:
            if int(blk.get("type", -1)) != 1:
                continue
            bb = blk.get("bbox")
            if not bb or len(bb) != 4:
                continue
            bx0, by0, bx1, by1 = map(float, bb)
            bw = bx1 - bx0
            bh = by1 - by0
            if bw < 20.0 or bh < 20.0:
                continue
            cell_bbox = (bx0, by0, bx1, by1)
            ia = _intersect_area(cell_bbox, region_bbox)
            if ia <= 0.0:
                continue
            ca = max(1.0, bw * bh)
            ra = max(1.0, (rx1 - rx0) * (ry1 - ry0))
            if (ia / ca) < 0.25 and (ia / ra) < 0.005:
                continue
            cx = (bx0 + bx1) / 2.0
            cy = (by0 + by1) / 2.0
            candidates.append(
                {
                    "bbox": cell_bbox,
                    "cx": cx,
                    "cy": cy,
                    "w": bw,
                    "h": bh,
                }
            )

        if not candidates:
            return []

        candidates.sort(key=lambda c: (c["cy"], c["cx"]))
        hs = sorted([float(c["h"]) for c in candidates if c.get("h") is not None])
        median_h = float(np.median(hs)) if hs else 24.0
        row_tol = max(12.0, median_h * 0.60)

        rows: List[List[dict]] = []
        for c in candidates:
            if not rows:
                rows.append([c])
                continue
            last_row = rows[-1]
            last_cy = float(np.mean([r["cy"] for r in last_row]))
            if abs(float(c["cy"]) - last_cy) <= row_tol:
                last_row.append(c)
            else:
                rows.append([c])

        ordered: List[dict] = []
        for ridx, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda x: x["cx"])
            for sidx, c in enumerate(row_sorted):
                ordered.append(
                    {
                        "source_row_index": ridx,
                        "source_slot_index": sidx,
                        "cell_id": f"R{ridx + 1}S{sidx + 1}",
                        "bbox": c["bbox"],
                        "cx": c["cx"],
                        "cy": c["cy"],
                        "w": c["w"],
                        "h": c["h"],
                    }
                )
        return ordered

    def _map_mid_band_slots_to_image_cells(
        slots: List[FullPalletMidBandSlot],
        source_cells: List[dict],
    ) -> Dict[str, Optional[dict]]:
        ordered_slots = sorted(slots, key=lambda s: (int(s.row_index), int(s.slot_in_row), int(s.slot_order)))
        mapping: Dict[str, Optional[dict]] = {}
        for idx, slot in enumerate(ordered_slots):
            mapping[slot.slot_id] = source_cells[idx] if idx < len(source_cells) else None
        return mapping

    def _build_mid_band_row_strip_lookup(
        source_cells: List[dict],
        expected_rows: int,
    ) -> Dict[int, dict]:
        if expected_rows <= 0:
            return {}
        strip_candidates: List[dict] = []
        for cell in source_cells:
            bbox = cell.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = map(float, bbox)
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            aspect = (w / h) if h > 0.0 else 99.0
            if w >= 120.0 and h >= 36.0 and aspect >= 1.65:
                row_cell = dict(cell)
                row_cell["strip_aspect"] = aspect
                strip_candidates.append(row_cell)
        if len(strip_candidates) < expected_rows:
            return {}
        strip_candidates.sort(key=lambda c: (float(c.get("cy", 0.0)), float(c.get("cx", 0.0))))
        return {idx: cell for idx, cell in enumerate(strip_candidates[:expected_rows])}

    def _slice_mid_band_row_strip_cell(
        row_strip_cell: dict,
        row_index: int,
        col_index: int,
        col_count: int,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], List[str]]:
        bbox = row_strip_cell.get("bbox")
        reasons: List[str] = ["source_cell_is_row_strip"]
        if not bbox or len(bbox) != 4 or col_count <= 0:
            reasons.append("invalid_row_strip_slice_inputs")
            return None, reasons
        sx0, sy0, sx1, sy1 = map(float, bbox)
        sw = max(0.0, sx1 - sx0)
        sh = max(0.0, sy1 - sy0)
        if sw < 30.0 or sh < 20.0:
            reasons.append("row_strip_too_small")
            return None, reasons

        bounded_col = max(0, min(int(col_count) - 1, int(col_index)))
        cell_w = sw / float(col_count)
        x_pad = min(3.5, max(1.0, cell_w * 0.06))
        y_pad = min(4.0, max(1.0, sh * 0.025))
        x0 = sx0 + bounded_col * cell_w + x_pad
        x1 = sx0 + (bounded_col + 1) * cell_w - x_pad
        y0 = sy0 + y_pad
        y1 = sy1 - y_pad
        if x1 <= x0 + 6.0 or y1 <= y0 + 6.0:
            reasons.append("row_strip_slice_collapsed")
            return None, reasons
        if bounded_col != int(col_index):
            reasons.append("column_index_clamped")
        if int(row_index) != int(row_strip_cell.get("source_row_index", row_index)):
            reasons.append("render_row_mapped_to_source_strip_order")
        return (float(x0), float(y0), float(x1), float(y1)), reasons

    def _render_page_pixmap_image(
        image_page: Optional[fitz.Page],
        zoom: float = 3.0,
    ) -> Tuple[Optional[Image.Image], float]:
        if image_page is None:
            return None, zoom
        pix = image_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), zoom

    def _tighten_pixmap_region_to_visible_art(
        page_img: Optional[Image.Image],
        page_zoom: float,
        region_bbox: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float, float], str]:
        if page_img is None:
            return region_bbox, "source_strip_not_tightened_no_pixmap"
        x0, y0, x1, y1 = map(float, region_bbox)
        ix0 = max(0, int(math.floor(x0 * page_zoom)))
        iy0 = max(0, int(math.floor(y0 * page_zoom)))
        ix1 = min(page_img.width, int(math.ceil(x1 * page_zoom)))
        iy1 = min(page_img.height, int(math.ceil(y1 * page_zoom)))
        if ix1 <= ix0 + 20 or iy1 <= iy0 + 20:
            return region_bbox, "source_strip_not_tightened_too_small"

        arr = np.asarray(page_img.crop((ix0, iy0, ix1, iy1)).convert("RGB"))
        min_rgb = np.min(arr, axis=2)
        max_rgb = np.max(arr, axis=2)
        saturation = max_rgb - min_rgb
        # Exclude pale page/ruler artifacts while retaining colored artwork and dark text.
        art_mask = (min_rgb < 245) & ((saturation > 18) | (min_rgb < 120))
        if not bool(np.any(art_mask)):
            return region_bbox, "source_strip_not_tightened_no_art_mask"

        h_px, w_px = art_mask.shape
        col_counts = np.sum(art_mask, axis=0)
        row_counts = np.sum(art_mask, axis=1)
        col_threshold = max(2, int(h_px * 0.025))
        row_threshold = max(2, int(w_px * 0.020))
        xs = np.where(col_counts >= col_threshold)[0]
        ys = np.where(row_counts >= row_threshold)[0]
        if len(xs) == 0 or len(ys) == 0:
            return region_bbox, "source_strip_not_tightened_sparse_art_mask"

        margin_x = int(round(5.0 * page_zoom))
        margin_y = int(round(3.0 * page_zoom))
        tx0 = max(0, int(xs.min()) - margin_x)
        tx1 = min(w_px, int(xs.max()) + 1 + margin_x)
        ty0 = max(0, int(ys.min()) - margin_y)
        ty1 = min(h_px, int(ys.max()) + 1 + margin_y)
        tightened = (
            float((ix0 + tx0) / page_zoom),
            float((iy0 + ty0) / page_zoom),
            float((ix0 + tx1) / page_zoom),
            float((iy0 + ty1) / page_zoom),
        )
        tw = tightened[2] - tightened[0]
        th = tightened[3] - tightened[1]
        ow = x1 - x0
        oh = y1 - y0
        if tw < ow * 0.62 or th < oh * 0.70:
            return region_bbox, "source_strip_tighten_rejected_too_aggressive"
        return tightened, "source_strip_tightened_to_visible_art"

    def _detect_bd_mid_band_visual_slot_grid(
        page_img: Optional[Image.Image],
        page_zoom: float,
        region_bbox: Tuple[float, float, float, float],
    ) -> Optional[List[List[Tuple[float, float, float, float]]]]:
        if page_img is None:
            return None
        x0, y0, x1, y1 = map(float, region_bbox)
        ix0 = max(0, int(math.floor(x0 * page_zoom)))
        iy0 = max(0, int(math.floor(y0 * page_zoom)))
        ix1 = min(page_img.width, int(math.ceil(x1 * page_zoom)))
        iy1 = min(page_img.height, int(math.ceil(y1 * page_zoom)))
        if ix1 <= ix0 + 20 or iy1 <= iy0 + 20:
            return None

        arr = np.asarray(page_img.crop((ix0, iy0, ix1, iy1)).convert("RGB"))
        min_rgb = np.min(arr, axis=2)
        max_rgb = np.max(arr, axis=2)
        saturation = max_rgb - min_rgb
        art_mask = (min_rgb < 245) & ((saturation > 18) | (min_rgb < 120))
        h_px, w_px = art_mask.shape

        def _segments_from_projection(values: np.ndarray, threshold: float, min_len_px: int) -> List[Tuple[int, int]]:
            active = values >= threshold
            segments: List[Tuple[int, int]] = []
            i = 0
            while i < len(active):
                if not bool(active[i]):
                    i += 1
                    continue
                j = i + 1
                while j < len(active) and bool(active[j]):
                    j += 1
                if (j - i) >= min_len_px:
                    segments.append((i, j))
                i = j
            return segments

        col_counts = np.sum(art_mask, axis=0)
        col_smooth = np.convolve(col_counts, np.ones(9) / 9.0, mode="same")
        x_segments = _segments_from_projection(
            col_smooth,
            threshold=max(2.0, h_px * 0.035),
            min_len_px=max(6, int(page_zoom * 5.0)),
        )
        if len(x_segments) < 8:
            return None
        if len(x_segments) > 8:
            x_segments = sorted(
                sorted(x_segments, key=lambda s: (s[1] - s[0]), reverse=True)[:8],
                key=lambda s: s[0],
            )

        row_counts = np.sum(art_mask, axis=1)
        row_smooth = np.convolve(row_counts, np.ones(9) / 9.0, mode="same")
        y_segments = _segments_from_projection(
            row_smooth,
            threshold=max(2.0, w_px * 0.025),
            min_len_px=max(6, int(page_zoom * 8.0)),
        )
        if len(y_segments) == 2 and (y_segments[1][1] - y_segments[1][0]) > (y_segments[0][1] - y_segments[0][0]) * 1.45:
            y0b, y1b = y_segments[1]
            mid = int(round((y0b + y1b) / 2.0))
            y_segments = [y_segments[0], (y0b, mid), (mid, y1b)]
        if len(y_segments) < 3:
            return None
        if len(y_segments) > 3:
            y_segments = sorted(
                sorted(y_segments, key=lambda s: (s[1] - s[0]), reverse=True)[:3],
                key=lambda s: s[0],
            )

        x_margin = int(round(page_zoom * 1.2))
        y_margin = int(round(page_zoom * 1.0))
        x_boxes = [
            (
                float((ix0 + max(0, sx - x_margin)) / page_zoom),
                float((ix0 + min(w_px, ex + x_margin)) / page_zoom),
            )
            for sx, ex in x_segments[:8]
        ]
        y_boxes = [
            (
                float((iy0 + max(0, sy - y_margin)) / page_zoom),
                float((iy0 + min(h_px, ey + y_margin)) / page_zoom),
            )
            for sy, ey in y_segments[:3]
        ]
        return [[(xx0, yy0, xx1, yy1) for xx0, xx1 in x_boxes] for yy0, yy1 in y_boxes]

    def _build_bd_mid_band_pixmap_grid(
        source_cells: List[dict],
        page_width: float,
        page_height: float,
        page_img: Optional[Image.Image],
        page_zoom: float,
    ) -> Tuple[Optional[List[List[Tuple[float, float, float, float]]]], Optional[Tuple[float, float, float, float]], str]:
        strip_candidates: List[dict] = []
        for cell in source_cells:
            bbox = cell.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = map(float, bbox)
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            aspect = (w / h) if h > 0.0 else 99.0
            if w >= 120.0 and h >= 70.0 and aspect >= 1.65:
                strip_candidates.append({"bbox": (x0, y0, x1, y1), "y0": y0, "cy": (y0 + y1) / 2.0})
        if not strip_candidates:
            return None, None, "no_wide_pixmap_source_strip"

        strip_candidates.sort(key=lambda c: (float(c["y0"]), float(c["cy"])))
        middle_candidates = [c for c in strip_candidates if 200.0 <= float(c["y0"]) <= 320.0]
        chosen = middle_candidates[0] if middle_candidates else (strip_candidates[1] if len(strip_candidates) >= 2 else strip_candidates[0])
        region, tighten_source = _tighten_pixmap_region_to_visible_art(page_img, page_zoom, chosen["bbox"])
        visual_grid = _detect_bd_mid_band_visual_slot_grid(page_img, page_zoom, region)
        if visual_grid:
            return visual_grid, region, f"bd_middle_band_visual_segments:{tighten_source}"
        grid = _build_anchor_mid_band_slot_grid(region, page_width, page_height)
        if not grid:
            return None, region, "pixmap_source_strip_grid_failed"
        return grid, region, f"bd_middle_band_source_strip:{tighten_source}"

    def _build_ac_mid_band_pixmap_cells(
        source_cells: List[dict],
        expected_count: int,
    ) -> Tuple[List[dict], Optional[Tuple[float, float, float, float]], str]:
        if expected_count <= 0:
            return [], None, "no_expected_ac_pixmap_cells"
        candidates: List[dict] = []
        for cell in source_cells:
            bbox = cell.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = map(float, bbox)
            w = x1 - x0
            h = y1 - y0
            if 18.0 <= w <= 48.0 and 24.0 <= h <= 58.0 and 235.0 <= y0 <= 370.0:
                c = dict(cell)
                c["bbox"] = (x0, y0, x1, y1)
                c["cx"] = (x0 + x1) / 2.0
                c["cy"] = (y0 + y1) / 2.0
                c["w"] = w
                c["h"] = h
                candidates.append(c)
        if not candidates:
            return [], None, "no_ac_mid_band_individual_cells"

        candidates.sort(key=lambda c: (float(c["cy"]), float(c["cx"])))
        hs = [float(c["h"]) for c in candidates]
        row_tol = max(10.0, float(np.median(hs)) * 0.55) if hs else 16.0
        rows: List[List[dict]] = []
        for c in candidates:
            if not rows:
                rows.append([c])
                continue
            row_cy = float(np.mean([r["cy"] for r in rows[-1]]))
            if abs(float(c["cy"]) - row_cy) <= row_tol:
                rows[-1].append(c)
            else:
                rows.append([c])

        visual_rows = [sorted(row, key=lambda c: float(c["cx"])) for row in rows if len(row) >= 6]
        visual_rows = sorted(visual_rows, key=lambda row: float(np.mean([c["cy"] for c in row])))
        needed_rows = int(math.ceil(expected_count / 8.0))
        selected_cells: List[dict] = []
        for row in visual_rows[:needed_rows]:
            selected_cells.extend(row[:8])
        selected_cells = selected_cells[:expected_count]
        if len(selected_cells) < expected_count:
            return selected_cells, None, "ac_pixmap_cell_shortage"

        x0 = min(float(c["bbox"][0]) for c in selected_cells)
        y0 = min(float(c["bbox"][1]) for c in selected_cells)
        x1 = max(float(c["bbox"][2]) for c in selected_cells)
        y1 = max(float(c["bbox"][3]) for c in selected_cells)
        return selected_cells, (x0, y0, x1, y1), "ac_individual_pixmap_cells_source_order"

    def _pad_mid_band_pixmap_slot_bbox(
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = map(float, bbox)
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        x_pad = min(3.0, max(1.0, w * 0.055))
        y_pad = min(2.6, max(0.8, h * 0.055))
        return (x0 + x_pad, y0 + y_pad, x1 - x_pad, y1 - y_pad)

    def _crop_mid_band_pixmap_slot(
        page_img: Optional[Image.Image],
        page_zoom: float,
        page_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[Optional[Image.Image], Optional[Tuple[float, float, float, float]], Dict[str, object]]:
        debug_info: Dict[str, object] = {
            "suspicious_crop": True,
            "suspicious_reason": ["pixmap_crop_not_attempted"],
            "whitespace_score": None,
        }
        if page_img is None or page_bbox is None:
            return None, None, debug_info

        px0, py0, px1, py1 = page_bbox
        ix0 = max(0, int(math.floor(float(px0) * page_zoom)))
        iy0 = max(0, int(math.floor(float(py0) * page_zoom)))
        ix1 = min(page_img.width, int(math.ceil(float(px1) * page_zoom)))
        iy1 = min(page_img.height, int(math.ceil(float(py1) * page_zoom)))
        if ix1 <= ix0 + 12 or iy1 <= iy0 + 12:
            debug_info["suspicious_reason"] = ["pixmap_slot_bbox_too_small"]
            return None, None, debug_info

        raw = page_img.crop((ix0, iy0, ix1, iy1)).convert("RGB")
        arr = np.asarray(raw)
        h_px, w_px = arr.shape[:2]
        edge = max(2, min(18, int(min(w_px, h_px) * 0.06)))
        samples = np.concatenate(
            [
                arr[:edge, :, :].reshape(-1, 3),
                arr[-edge:, :, :].reshape(-1, 3),
                arr[:, :edge, :].reshape(-1, 3),
                arr[:, -edge:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(samples, axis=0)
        diff = np.max(np.abs(arr.astype(np.int16) - bg.astype(np.int16)), axis=2)
        not_bg = diff > 16
        not_near_white = np.min(arr, axis=2) < 246
        content = not_bg & not_near_white
        content_ratio = float(np.mean(content)) if content.size else 0.0
        whitespace_score = round(1.0 - content_ratio, 4)
        debug_info["whitespace_score"] = whitespace_score
        ys, xs = np.where(content)
        if len(xs) == 0 or len(ys) == 0:
            debug_info["suspicious_reason"] = ["pixmap_slot_mostly_empty"]
            return None, None, debug_info

        left = max(0, int(xs.min()) - edge)
        right = min(w_px, int(xs.max()) + 1 + edge)
        top = max(0, int(ys.min()) - edge)
        bottom = min(h_px, int(ys.max()) + 1 + edge)
        crop_w = right - left
        crop_h = bottom - top
        aspect = (crop_w / crop_h) if crop_h > 0 else 99.0
        reasons: List[str] = []
        if content_ratio < 0.015:
            reasons.append("pixmap_slot_mostly_whitespace")
        if crop_w < max(24, w_px * 0.24):
            reasons.append("pixmap_content_too_narrow")
        if crop_h < max(24, h_px * 0.24):
            reasons.append("pixmap_content_too_short")
        if aspect < 0.22 or aspect > 4.2:
            reasons.append("pixmap_content_aspect_out_of_range")
        if crop_w > w_px * 0.98 and crop_h > h_px * 0.98 and content_ratio < 0.08:
            reasons.append("pixmap_untrimmed_sparse_slot")

        if reasons:
            debug_info["suspicious_reason"] = reasons
            return None, None, debug_info

        cleaned = raw.crop((left, top, right, bottom)).convert("RGBA")
        chosen_bbox = (
            float((ix0 + left) / page_zoom),
            float((iy0 + top) / page_zoom),
            float((ix0 + right) / page_zoom),
            float((iy0 + bottom) / page_zoom),
        )
        debug_info.update(
            {
                "suspicious_crop": False,
                "suspicious_reason": [],
                "content_ratio": round(content_ratio, 4),
                "raw_pixel_size": [int(w_px), int(h_px)],
                "crop_pixel_size": [int(cleaned.width), int(cleaned.height)],
            }
        )
        return cleaned, chosen_bbox, debug_info

    def _inset_bonus_source_bbox(
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, object]]:
        x0, y0, x1, y1 = map(float, bbox)
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        inset_x = min(2.0, max(0.7, w * 0.035))
        inset_y = min(1.8, max(0.5, h * 0.025))
        cleaned = (x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y)
        if cleaned[2] <= cleaned[0] + 4.0 or cleaned[3] <= cleaned[1] + 4.0:
            return bbox, {
                "source_inset_applied": False,
                "source_inset_x": 0.0,
                "source_inset_y": 0.0,
                "source_inset_rejected": True,
            }
        return cleaned, {
            "source_inset_applied": True,
            "source_inset_x": round(float(inset_x), 2),
            "source_inset_y": round(float(inset_y), 2),
            "source_inset_rejected": False,
        }

    def _sanitize_bonus_crop_image(
        img: Optional[Image.Image],
    ) -> Tuple[Optional[Image.Image], Dict[str, object]]:
        debug_info: Dict[str, object] = {
            "sanitized_crop_used": False,
            "neighboring_edge_contamination_detected": False,
            "image_edge_trim_bbox": None,
            "image_edge_trim_px": [0, 0, 0, 0],
        }
        if img is None:
            return None, debug_info
        try:
            src = img.convert("RGBA")
            arr = np.asarray(src.convert("RGB"))
            h_px, w_px = arr.shape[:2]
            if w_px < 30 or h_px < 30:
                return src, debug_info

            edge_w = max(2, min(8, int(w_px * 0.035)))
            edge_h = max(2, min(8, int(h_px * 0.030)))
            min_rgb = np.min(arr, axis=2)
            max_rgb = np.max(arr, axis=2)
            saturation = max_rgb - min_rgb
            ink = (min_rgb < 245) & ((saturation > 18) | (min_rgb < 120))

            left_density = float(np.mean(ink[:, :edge_w]))
            right_density = float(np.mean(ink[:, -edge_w:]))
            top_density = float(np.mean(ink[:edge_h, :]))
            bottom_density = float(np.mean(ink[-edge_h:, :]))
            center_density = (
                float(np.mean(ink[edge_h:-edge_h, edge_w:-edge_w]))
                if h_px > 2 * edge_h and w_px > 2 * edge_w
                else 0.0
            )

            trim_left = edge_w if left_density > max(0.42, center_density * 1.9) else 0
            trim_right = edge_w if right_density > max(0.42, center_density * 1.9) else 0
            trim_top = edge_h if top_density > max(0.48, center_density * 2.1) else 0
            trim_bottom = edge_h if bottom_density > max(0.48, center_density * 2.1) else 0

            max_trim_x = int(w_px * 0.055)
            max_trim_y = int(h_px * 0.045)
            trim_left = min(trim_left, max_trim_x)
            trim_right = min(trim_right, max_trim_x)
            trim_top = min(trim_top, max_trim_y)
            trim_bottom = min(trim_bottom, max_trim_y)

            if trim_left or trim_right or trim_top or trim_bottom:
                l = trim_left
                t = trim_top
                r = w_px - trim_right
                b = h_px - trim_bottom
                if r > l + max(18, w_px * 0.80) and b > t + max(18, h_px * 0.84):
                    src = src.crop((l, t, r, b))
                    debug_info.update(
                        {
                            "sanitized_crop_used": True,
                            "neighboring_edge_contamination_detected": True,
                            "image_edge_trim_bbox": [l, t, r, b],
                            "image_edge_trim_px": [trim_left, trim_top, trim_right, trim_bottom],
                        }
                    )
        except Exception:
            return img, debug_info
        return src, debug_info

    def _validate_mid_band_mapped_cell_bbox(
        mapped_bbox: Optional[Tuple[float, float, float, float]],
        fallback_bbox: Optional[Tuple[float, float, float, float]],
        strict_side_guardrails: bool,
    ) -> Tuple[bool, List[str]]:
        if mapped_bbox is None:
            return False, ["missing_mapped_bbox"]
        mx0, my0, mx1, my1 = mapped_bbox
        mw = max(0.0, float(mx1 - mx0))
        mh = max(0.0, float(my1 - my0))
        reasons: List[str] = []

        if mw < 16.0 or mh < 16.0:
            reasons.append("tiny_bbox")

        aspect = (mw / mh) if mh > 0.0 else 99.0
        if aspect > 3.6 or aspect < 0.28:
            reasons.append("strip_aspect")

        if fallback_bbox is not None:
            fx0, fy0, fx1, fy1 = fallback_bbox
            fw = max(1.0, float(fx1 - fx0))
            fh = max(1.0, float(fy1 - fy0))
            fcx = (float(fx0) + float(fx1)) / 2.0
            fcy = (float(fy0) + float(fy1)) / 2.0
            mcx = (float(mx0) + float(mx1)) / 2.0
            mcy = (float(my0) + float(my1)) / 2.0
            d = ((mcx - fcx) ** 2 + (mcy - fcy) ** 2) ** 0.5
            if strict_side_guardrails and d > max(44.0, fw * 1.7, fh * 1.7):
                reasons.append("off_expected_slot_geometry")

            area_ratio = (mw * mh) / max(1.0, fw * fh)
            if strict_side_guardrails:
                if mw > fw * 1.95:
                    reasons.append("too_wide_for_single_card")
                if mh > fh * 1.95:
                    reasons.append("too_tall_for_single_card")
                if mw < fw * 0.55:
                    reasons.append("too_narrow_for_single_card")
                if mh < fh * 0.55:
                    reasons.append("too_short_for_single_card")
                if area_ratio > 2.8:
                    reasons.append("likely_multi_card_group")
                if area_ratio < 0.35:
                    reasons.append("likely_partial_strip")
            else:
                if area_ratio > 4.5:
                    reasons.append("oversized_area")
                if area_ratio < 0.22:
                    reasons.append("undersized_area")

        return len(reasons) == 0, reasons

    def _resolve_mid_band_slot(
        page: FullPalletPage,
        slot: FullPalletMidBandSlot,
    ) -> Tuple[Optional[MatrixRow], Dict[str, object]]:
        match, trace = _resolve_mid_band_slot_no_position(slot)
        if match is not None:
            trace["excel_lookup_succeeded"] = True
            return match, trace

        side_key = (page.side_letter, int(slot.row_index), int(slot.slot_in_row))
        side_candidates = mid_slot_lookup_by_side.get(side_key, [])
        pos_match, pos_debug = _resolve_mid_slot_position_candidates(slot, side_candidates)
        trace.update(pos_debug)
        if pos_match is not None:
            trace["fallback_path"] = "template_position_side"
            trace["position_fallback_scope"] = "side"
            trace["excel_lookup_succeeded"] = True
            return pos_match, trace

        global_key = (int(slot.row_index), int(slot.slot_in_row))
        global_candidates = mid_slot_lookup_global.get(global_key, [])
        pos_global, pos_global_debug = _resolve_mid_slot_position_candidates(slot, global_candidates)
        trace["fallback_candidate_upcs"] = pos_global_debug.get("fallback_candidate_upcs", [])
        trace["fallback_similarity_scores"] = pos_global_debug.get("fallback_similarity_scores", {})
        trace["chosen_candidate"] = pos_global_debug.get("chosen_candidate")
        if pos_global is not None:
            trace["fallback_path"] = "template_position_global"
            trace["position_fallback_scope"] = "global"
            trace["excel_lookup_succeeded"] = True
            return pos_global, trace

        trace["position_fallback_scope"] = "none"
        trace["excel_lookup_succeeded"] = False
        return None, trace

    def _build_mid_band_candidate_record(
        p: FullPalletPage,
        slot: FullPalletMidBandSlot,
        source_index: int,
    ) -> dict:
        match, resolve_trace = _resolve_mid_band_slot(p, slot)
        return {
            "side": p.side_letter,
            "slot": slot,
            "source_index": int(source_index),
            "slot_id": slot.slot_id,
            "group": slot.block_name,
            "row_index": int(slot.row_index),
            "slot_in_row": int(slot.slot_in_row),
            "slot_order": int(slot.slot_order),
            "last5": _to_last5(slot.last5),
            "upc12": match.upc12 if match else None,
            "cpp": match.cpp_qty if match else None,
            "resolved_match": match,
            "resolved_name": match.display_name if match else (slot.parsed_name or slot.raw_label_text or "").strip(),
            "display_name": match.display_name if match else (slot.parsed_name or slot.raw_label_text or "").strip(),
            "resolve_fallback_path": resolve_trace.get("fallback_path"),
            "resolve_trace": resolve_trace,
        }

    def _clean_middle_grid_candidate_records(candidates: List[dict]) -> Tuple[List[dict], List[dict]]:
        cleaned: List[dict] = []
        rejected: List[dict] = []

        def _is_non_product_fixture_or_signage(candidate: dict) -> bool:
            slot = candidate.get("slot")
            if slot is None:
                return False
            match = candidate.get("resolved_match")
            text = " ".join(
                [
                    str(getattr(slot, "raw_label_text", "") or ""),
                    str(getattr(slot, "parsed_name", "") or ""),
                    str(getattr(slot, "block_name", "") or ""),
                    str(candidate.get("resolved_name") or ""),
                    str(candidate.get("display_name") or ""),
                    str(getattr(match, "display_name", "") or ""),
                ]
            ).upper()
            normalized = re.sub(r"[^A-Z0-9]+", " ", text)
            tokens = set(normalized.split())
            upc12 = str(candidate.get("upc12") or getattr(match, "upc12", "") or "").strip()

            phrase_markers = (
                "MARKETING MESSAGE",
                "FRAME SIGN",
                "GIFT CARD HOLDER",
                "GIFT CARD HOLDERS",
                "GIFT CARD D",
                "GIFT CARD IN NEW",
                "GIFT CARD D IN NEW",
                "GCI TALL",
                "GCI THIN",
                "GCI 3PK",
                "GCI 3 PK",
                "TALL LID",
                "THIN LID",
                "XL PRESENT",
                "XL ENV",
                "3 PK WRAP",
                "3 PK DIECUT",
                "3PK SHAPED",
                "2 PK",
                "BOX TRUCK",
                "GIFT CARD PKG",
                "GIFT CARD  PKG",
            )
            if any(marker in normalized for marker in phrase_markers):
                return True

            seasonal_fixture_tokens = {
                "SNOWFLAKE",
                "SNOWMAN",
                "HOHOHO",
                "REINDEER",
                "PEEKING",
                "HOLDER",
                "HOLDERS",
                "FIXTURE",
                "HEADER",
                "SIGN",
                "PKG",
                "LID",
                "WRAP",
                "DIECUT",
            }
            if upc12.startswith("084921908") and ("GCI" in tokens or bool(tokens & seasonal_fixture_tokens)):
                return True
            if "GCI" in tokens:
                return True
            if "PKG" in tokens:
                return True
            if "HEADER" in tokens or "SIGN" in tokens or "FIXTURE" in tokens:
                return True
            if "MARKETING" in tokens and "MESSAGE" in tokens:
                return True
            if "FRAME" in tokens and "SIGN" in tokens:
                return True
            if "GIFTCARD" in tokens or ("GIFT" in tokens and "CARD" in tokens and ("NEW" in tokens or "PKG" in tokens)):
                return True
            if "WM" in tokens and ("NEW" in tokens or "PKG" in tokens):
                return True
            if ("TALL" in tokens or "LID" in tokens) and ("BOX" in tokens or "HARDWARE" in tokens or "GCI" in tokens):
                return True

            return False

        for candidate in candidates:
            slot = candidate.get("slot")
            slot_id = str(candidate.get("slot_id") or getattr(slot, "slot_id", ""))
            raw_text = str(getattr(slot, "raw_label_text", "") or "")
            parsed_name = str(getattr(slot, "parsed_name", "") or "")
            last5 = _to_last5(candidate.get("last5") or getattr(slot, "last5", ""))
            upc12 = str(candidate.get("upc12") or "").strip()
            match = candidate.get("resolved_match")
            reason = ""

            if _is_non_product_fixture_or_signage(candidate):
                reason = "non_product_fixture_or_signage"
            elif match is None or not upc12:
                reason = "unresolved_no_matrix_match"
            elif not last5 and not upc12:
                reason = "missing_upc_and_last5"
            elif not str(candidate.get("resolved_name") or parsed_name or "").strip():
                reason = "missing_resolved_description"

            if reason:
                rejected.append(
                    {
                        "slot_id": slot_id,
                        "source_index": candidate.get("source_index"),
                        "row_index": candidate.get("row_index"),
                        "slot_in_row": candidate.get("slot_in_row"),
                        "last5": last5,
                        "upc12": upc12 or None,
                        "raw_label_text": raw_text,
                        "parsed_name": parsed_name,
                        "cpp": candidate.get("cpp"),
                        "resolve_fallback_path": candidate.get("resolve_fallback_path"),
                        "reason": reason,
                        "rejection_reason": reason,
                    }
                )
                continue

            cleaned.append(candidate)

        return cleaned, rejected

    def _build_middle_grid_cell_candidate_records(p: FullPalletPage) -> Tuple[List[dict], List[dict], int]:
        bonus_y = _marker_y(p, "bonus_strip")
        cells_above_bonus: List[CellData] = []
        for cell in p.cells:
            _cx, cy = _cell_center(cell)
            if bonus_y is not None and cy >= float(bonus_y) - 8.0:
                continue
            cells_above_bonus.append(cell)

        row_to_cells: Dict[int, List[CellData]] = {}
        for cell in cells_above_bonus:
            row_to_cells.setdefault(cell.row, []).append(cell)

        candidate_rows: List[Tuple[float, List[dict]]] = []
        rejected: List[dict] = []
        raw_candidate_count = 0
        for _row_id, row_cells in row_to_cells.items():
            row_records: List[dict] = []
            for cell in sorted(row_cells, key=lambda c_: _cell_center(c_)[0]):
                raw_candidate_count += 1
                match = resolve_full_pallet(cell.last5, cell.name, matrix_idx) if cell.last5 else None
                synthetic_slot = FullPalletMidBandSlot(
                    slot_id=f"{p.side_letter}-MB-CELL-R{cell.row}-C{cell.col}",
                    side_letter=p.side_letter,
                    row_index=0,
                    block_name="middle",
                    block_pos_index=0,
                    slot_order=0,
                    slot_in_row=0,
                    bbox=cell.bbox,
                    raw_label_text=cell.name,
                    parsed_name=cell.name,
                    last5=cell.last5,
                    qty=cell.qty,
                    extraction_bbox=cell.bbox,
                    accepted_words=[],
                    rejected_nearby_word_count=0,
                    resolved_upc12=match.upc12 if match else None,
                    resolved_display_name=match.display_name if match else None,
                    resolved_cpp_qty=match.cpp_qty if match else None,
                )
                record = {
                    "side": p.side_letter,
                    "slot": synthetic_slot,
                    "source_index": 0,
                    "slot_id": synthetic_slot.slot_id,
                    "group": "middle",
                    "row_index": 0,
                    "slot_in_row": 0,
                    "slot_order": 0,
                    "last5": _to_last5(cell.last5),
                    "upc12": match.upc12 if match else None,
                    "cpp": match.cpp_qty if match else None,
                    "resolved_match": match,
                    "resolved_name": match.display_name if match else cell.name.strip(),
                    "display_name": match.display_name if match else cell.name.strip(),
                    "resolve_fallback_path": "cell_last5_matrix" if match else "unresolved_cell",
                    "resolve_trace": {"fallback_path": "cell_last5_matrix" if match else "unresolved_cell"},
                }
                cleaned, rejected_one = _clean_middle_grid_candidate_records([record])
                if cleaned:
                    row_records.extend(cleaned)
                else:
                    rejected.extend(rejected_one)

            if row_records:
                row_y = float(np.mean([_cell_center(c_)[1] for c_ in row_cells]))
                candidate_rows.append((row_y, row_records))

        candidate_rows.sort(key=lambda item: item[0])
        selected_rows = candidate_rows[-3:]
        selected_records: List[dict] = []
        source_index = 0
        for row_index, (_row_y, row_records) in enumerate(selected_rows):
            row_records = sorted(row_records, key=lambda r: float(r["slot"].bbox[0]))
            for slot_in_row, record in enumerate(row_records[:8]):
                slot = record["slot"]
                updated_slot = FullPalletMidBandSlot(
                    slot_id=f"{p.side_letter}-MB-R{row_index + 1}-S{slot_in_row + 1}",
                    side_letter=p.side_letter,
                    row_index=row_index,
                    block_name=("left" if slot_in_row <= 1 else "center" if slot_in_row <= 5 else "right"),
                    block_pos_index=(slot_in_row if slot_in_row <= 1 else slot_in_row - 2 if slot_in_row <= 5 else slot_in_row - 6),
                    slot_order=source_index,
                    slot_in_row=slot_in_row,
                    bbox=slot.bbox,
                    raw_label_text=slot.raw_label_text,
                    parsed_name=slot.parsed_name,
                    last5=slot.last5,
                    qty=slot.qty,
                    extraction_bbox=slot.extraction_bbox,
                    accepted_words=slot.accepted_words,
                    rejected_nearby_word_count=slot.rejected_nearby_word_count,
                    resolved_upc12=slot.resolved_upc12,
                    resolved_display_name=slot.resolved_display_name,
                    resolved_cpp_qty=slot.resolved_cpp_qty,
                )
                selected_records.append(
                    {
                        **record,
                        "slot": updated_slot,
                        "source_index": source_index,
                        "slot_id": updated_slot.slot_id,
                        "row_index": row_index,
                        "slot_in_row": slot_in_row,
                        "slot_order": source_index,
                        "group": updated_slot.block_name,
                    }
                )
                source_index += 1

        return selected_records[:24], rejected, raw_candidate_count

    def _build_mid_band_profile_comparison(
        p: FullPalletPage,
        slots: List[FullPalletMidBandSlot],
    ) -> Dict[str, object]:
        profile = get_mid_band_physical_profile(p.side_letter)
        groups = [str(g) for g in profile.get("physical_groups", ["left", "center", "right"])]
        expected_counts = {
            str(k): int(v)
            for k, v in dict(profile.get("expected_max_per_group", {})).items()
            if v is not None
        }

        ordered_slots = sorted(slots, key=lambda s: (int(s.row_index), int(s.slot_in_row), int(s.slot_order)))
        candidates_by_group: Dict[str, List[dict]] = {g: [] for g in groups}
        for source_index, slot in enumerate(ordered_slots):
            candidate = _build_mid_band_candidate_record(p, slot, source_index)
            group = str(candidate.get("group") or "unknown")
            if group not in candidates_by_group:
                candidates_by_group[group] = []
            candidates_by_group[group].append(candidate)

        overage: Dict[str, int] = {}
        shortage: Dict[str, int] = {}
        candidate_upcs_by_group: Dict[str, List[Optional[str]]] = {}
        candidate_records = [candidate for rows in candidates_by_group.values() for candidate in rows]
        selection = select_mid_band_cards_for_display(p.side_letter, candidate_records, profile)
        row_aware_candidates_by_group: Dict[str, List[dict]] = {g: [] for g in groups}
        for group, rows in selection.selected_by_group.items():
            row_aware_candidates_by_group.setdefault(group, []).extend(rows)
        for group, rows in selection.omitted_by_group.items():
            row_aware_candidates_by_group.setdefault(group, []).extend(rows)
        detected_counts = {g: len(row_aware_candidates_by_group.get(g, [])) for g in row_aware_candidates_by_group.keys()}

        for group, candidates in row_aware_candidates_by_group.items():
            expected = expected_counts.get(group)
            candidate_upcs_by_group[group] = [c.get("upc12") for c in candidates]
            if expected is None:
                overage[group] = 0
                shortage[group] = 0
                continue

            overage[group] = max(0, len(candidates) - expected)
            shortage[group] = max(0, expected - len(candidates))

        def _debug_rows_by_group(rows_by_group: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
            return {
                group: [{k: v for k, v in row.items() if k != "slot"} for row in rows]
                for group, rows in rows_by_group.items()
            }

        return {
            "side": p.side_letter,
            "profile_name": profile.get("profile_name"),
            "detected_candidate_count": sum(detected_counts.values()),
            "detected_group_counts": detected_counts,
            "expected_group_counts": expected_counts,
            "expected_rows_per_group": profile.get("expected_rows_per_group", {}),
            "render_layout_hints": profile.get("render_layout_hints", {}),
            "overage_per_group": overage,
            "shortage_per_group": shortage,
            "candidate_upcs_by_group": candidate_upcs_by_group,
            "retained_if_expected_max_enforced": _debug_rows_by_group(selection.selected_by_group),
            "omitted_if_expected_max_enforced": _debug_rows_by_group(selection.omitted_by_group),
            "display_selection_debug": selection.debug_summary,
        }

    def _draw_canonical_mid_band_section(
        p: FullPalletPage,
        section: FullPalletMidBandSection,
        plan: Dict[str, float],
        sec_top: float,
        unresolved_bucket: List[str],
        missing_image_slots: List[str],
        content_x0: float,
        unresolved_debug_rows: Optional[List[dict]] = None,
        position_fallback_debug_rows: Optional[List[dict]] = None,
        layout_debug: Optional[Dict[str, object]] = None,
        layout_assignment_rows: Optional[List[dict]] = None,
        selection_debug: Optional[Dict[str, object]] = None,
        normalization_debug_rows: Optional[List[dict]] = None,
    ) -> Tuple[int, int, bool, float, int, int]:
        nonlocal rightmost_used, matched_cells, unmatched_cells

        y_cursor = sec_top
        if len(section.rows) == 0:
            return 0, 0, False, sec_top, 0, 0
        profile = get_mid_band_physical_profile(p.side_letter)
        candidate_slots = sorted(
            [s for r in section.rows for s in r.slots],
            key=lambda s: (int(s.row_index), int(s.slot_in_row), int(s.slot_order)),
        )
        candidate_records = [
            _build_mid_band_candidate_record(p, slot, source_index)
            for source_index, slot in enumerate(candidate_slots)
        ]
        selection = select_mid_band_cards_for_display(
            p.side_letter,
            candidate_records,
            profile,
        )
        if selection_debug is not None:
            selection_debug.update(selection.debug_summary)
        render_assignment_debug: List[dict] = []

        card_w = float(plan["card_w"])
        card_h = float(plan["card_h"])
        row_gutter = float(plan["row_gutter"])
        intra_gap = float(plan["intra_gap"])
        inter_gap = float(plan["inter_gap"])
        total_w = float(plan["total_w"])

        start_x = content_x0 + max(0.0, (content_w - total_w) / 2.0)
        row_xs = _mid_band_row_xs(
            start_x=start_x,
            card_w=card_w,
            cols=8,
            intra_gap=intra_gap,
            inter_gap=inter_gap,
            col_gap=intra_gap,
        )
        rightmost_used = max(rightmost_used, start_x + total_w)

        grid_top = y_cursor
        slots_drawn = 0
        overflow = bool(plan["overflow"])
        if layout_debug is not None:
            layout_debug.update(
                {
                    "side": p.side_letter,
                    "mode": "canonical_mid_band",
                    "candidate_card_count": len(candidate_slots),
                    "card_count": len(selection.selected_cards),
                    "columns_used": 8,
                    "rows_used": 3,
                    "card_w": card_w,
                    "card_h": card_h,
                    "h_gutter_intra": intra_gap,
                    "h_gutter_inter": inter_gap,
                    "h_gutter_uniform": None,
                    "v_gutter": row_gutter,
                    "origin_x": start_x,
                    "origin_y": grid_top,
                }
            )

        for selected_candidate in selection.selected_cards:
            slot = selected_candidate.get("slot")
            if slot is None:
                continue
            final_index = int(selected_candidate.get("final_index", slots_drawn))
            render_row = int(selected_candidate.get("selected_row", slot.row_index))
            render_col = int(selected_candidate.get("selected_col", slot.slot_in_row))
            x = row_xs[min(max(render_col, 0), len(row_xs) - 1)]
            y = grid_top - (render_row + 1) * card_h - render_row * row_gutter

            match = selected_candidate.get("resolved_match")
            resolve_trace = dict(selected_candidate.get("resolve_trace") or {})
            if match is None:
                match, resolve_trace = _resolve_mid_band_slot(p, slot)
            if (
                position_fallback_debug_rows is not None
                and str(resolve_trace.get("fallback_path", "")).startswith("template_position_")
            ):
                position_fallback_debug_rows.append(
                    {
                        "side": p.side_letter,
                        "slot_id": slot.slot_id,
                        "row_index": slot.row_index,
                        "slot_in_row": slot.slot_in_row,
                        "fallback_scope": resolve_trace.get("position_fallback_scope"),
                        "fallback_candidate_upcs": resolve_trace.get("fallback_candidate_upcs", []),
                        "fallback_similarity_scores": resolve_trace.get("fallback_similarity_scores", {}),
                        "chosen_candidate": resolve_trace.get("chosen_candidate"),
                    }
                )
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
                if unresolved_debug_rows is not None:
                    unresolved_debug_rows.append(
                        {
                            "page_index": p.page_index,
                            "side": p.side_letter,
                            "slot_id": slot.slot_id,
                            "row_index": slot.row_index,
                            "slot_in_row": slot.slot_in_row,
                            "block_name": slot.block_name,
                            "labels_last5": resolve_trace.get("labels_last5"),
                            "labels_text": (slot.raw_label_text or "").strip(),
                            "fallback_path": resolve_trace.get("fallback_path"),
                            "label_hint_upc_candidates": resolve_trace.get("label_hint_upc_candidates", []),
                            "label_hint_last5_candidates": resolve_trace.get("label_hint_last5_candidates", []),
                            "fallback_candidate_upcs": resolve_trace.get("fallback_candidate_upcs", []),
                            "fallback_similarity_scores": resolve_trace.get("fallback_similarity_scores", {}),
                            "chosen_candidate": resolve_trace.get("chosen_candidate"),
                            "position_fallback_scope": resolve_trace.get("position_fallback_scope"),
                            "excel_lookup_succeeded": bool(resolve_trace.get("excel_lookup_succeeded")),
                        }
                    )
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
            norm_debug = _draw_mid_band_card(
                c,
                x,
                y,
                card_w,
                card_h,
                img,
                upc_str,
                disp_name,
                cpp,
                section="mid_band",
                side=p.side_letter,
                final_index=final_index,
                row=render_row,
                col=render_col,
                source_crop_bbox=slot.bbox,
            )
            if normalization_debug_rows is not None:
                normalization_debug_rows.append(norm_debug)

            if debug_overlay:
                c.setFillColorRGB(0.35, 0.35, 0.35)
                c.setFont("Helvetica", 6)
                c.drawString(x + 2, y + card_h - 8, slot.slot_id)

            if layout_assignment_rows is not None:
                layout_assignment_rows.append(
                    {
                        "side": p.side_letter,
                        "final_index": final_index,
                        "slot_id": slot.slot_id,
                        "rendered_row": render_row,
                        "rendered_col": render_col,
                        "group": selected_candidate.get("selected_group"),
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "upc12": upc12,
                    }
                )
            render_assignment_debug.append(
                {
                    "final_index": final_index,
                    "row": render_row,
                    "col": render_col,
                    "group": selected_candidate.get("selected_group"),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "upc12": upc12,
                    "slot_id": slot.slot_id,
                    "image_draw_bbox": norm_debug.get("image_draw_bbox"),
                    "text_bbox": norm_debug.get("text_bbox"),
                    "overflow_or_bleed_detected": norm_debug.get("overflow_or_bleed_detected"),
                }
            )
            slots_drawn += 1

        if selection_debug is not None:
            render_assignment_debug.sort(key=lambda r: int(r.get("final_index", 0)))
            selected_ids = [r.get("slot_id") for r in selection_debug.get("selected_order", [])]
            rendered_ids = [r.get("slot_id") for r in render_assignment_debug]
            selection_debug["render_assignment"] = render_assignment_debug
            selection_debug["actual_selected_count"] = len(selection.selected_cards)
            selection_debug["final_render_order_matches_selected_order"] = rendered_ids == selected_ids

        sec_bottom = grid_top - 3 * card_h - 2 * row_gutter
        return 8, 3, overflow, sec_bottom, slots_drawn, 8

    def _draw_token_first_mid_band_section(
        p: FullPalletPage,
        section: FullPalletMidBandSection,
        plan: Dict[str, float],
        sec_top: float,
        unresolved_bucket: List[str],
        missing_image_slots: List[str],
        content_x0: float,
        unresolved_debug_rows: Optional[List[dict]] = None,
        position_fallback_debug_rows: Optional[List[dict]] = None,
        crop_debug_rows: Optional[List[dict]] = None,
        mapping_debug_rows: Optional[List[dict]] = None,
        detection_debug: Optional[Dict[str, object]] = None,
        layout_debug: Optional[Dict[str, object]] = None,
        layout_assignment_rows: Optional[List[dict]] = None,
        selection_debug: Optional[Dict[str, object]] = None,
        normalization_debug_rows: Optional[List[dict]] = None,
    ) -> Tuple[int, int, bool, float, int, int]:
        nonlocal rightmost_used, matched_cells, unmatched_cells

        all_candidate_slots: List[FullPalletMidBandSlot] = sorted(
            [s for r in section.rows for s in r.slots],
            key=lambda s: (int(s.row_index), int(s.slot_in_row), int(s.slot_order)),
        )
        profile = get_mid_band_physical_profile(p.side_letter)
        cleaned_candidate_records, rejected_middle_candidates, raw_middle_candidate_count = (
            _build_middle_grid_cell_candidate_records(p)
        )
        selection = select_mid_band_cards_for_display(
            p.side_letter,
            cleaned_candidate_records,
            profile,
        )
        all_slots: List[FullPalletMidBandSlot] = [
            candidate["slot"] for candidate in selection.selected_cards if candidate.get("slot") is not None
        ]
        if selection_debug is not None:
            selection_debug.update(selection.debug_summary)
            selection_debug["active_middle_band_path"] = "cell_middle_grid_above_bonus"
            selection_debug["raw_middle_candidate_count"] = raw_middle_candidate_count
            selection_debug["raw_candidate_count"] = raw_middle_candidate_count
            selection_debug["cleaned_middle_candidate_count"] = len(cleaned_candidate_records)
            selection_debug["clean_metadata_count"] = len(cleaned_candidate_records)
            selection_debug["rejected_middle_candidate_count"] = len(rejected_middle_candidates)
            selection_debug["rejected_candidate_count"] = len(rejected_middle_candidates)
            selection_debug["rejected_middle_candidates"] = rejected_middle_candidates
        if not all_slots:
            if detection_debug is not None:
                detection_debug.update(
                    {
                        "side": p.side_letter,
                        "active_mid_band_render_path": "token_first_mid_band",
                        "active_middle_band_path": "cell_middle_grid_above_bonus",
                        "raw_middle_candidate_count": raw_middle_candidate_count,
                        "raw_candidate_count": raw_middle_candidate_count,
                        "rejected_middle_candidate_count": len(rejected_middle_candidates),
                        "rejected_candidate_count": len(rejected_middle_candidates),
                        "rejected_middle_candidates": rejected_middle_candidates,
                        "cleaned_middle_candidate_count": 0,
                        "clean_metadata_count": 0,
                        "ordered_label_slot_count": 0,
                        "ordered_image_crop_count": 0,
                        "image_crop_count": 0,
                        "rendered_middle_slot_count": 0,
                        "rendered_count": 0,
                        "count_is_24": False,
                        "metadata_order_preview": [],
                        "image_order_preview": [],
                        "final_join_preview": [],
                        "labels_pdf_visual_crop_used": False,
                        "labels_pdf_visual_source_enabled": False,
                        "visual_source": "images_pdf",
                        "binding_mode": "image_pdf_crop_i_to_metadata_i",
                        "middle_band_binding_mode": "image_pdf_crop_i_to_clean_label_slot_i",
                        "slot_binding_mode": "image_pdf_crop_i_to_clean_label_slot_i",
                    }
                )
            return 0, 0, False, sec_top, 0, 0

        cols = int(plan.get("cols", 0))
        rows = int(plan.get("rows", 0))
        if cols <= 0:
            cols = min(8, max(1, len(all_slots)))
        rows = int(math.ceil(len(all_slots) / cols))

        card_w = float(plan["card_w"])
        card_h = float(plan["card_h"])
        row_gutter = float(plan["row_gutter"])
        col_gap = float(plan.get("col_gap", 6.0))
        intra_gap = float(plan.get("intra_gap", 6.0))
        inter_gap = float(plan.get("inter_gap", 14.0))
        total_w = float(plan["total_w"])

        start_x = content_x0 + max(0.0, (content_w - total_w) / 2.0)
        row_xs = _mid_band_row_xs(
            start_x=start_x,
            card_w=card_w,
            cols=cols,
            intra_gap=intra_gap,
            inter_gap=inter_gap,
            col_gap=col_gap,
        )
        rightmost_used = max(rightmost_used, start_x + total_w)
        grid_top = sec_top
        overflow = bool(plan["overflow"])
        slots_drawn = 0
        try:
            img_page = images_doc[p.page_index]
            page_width = float(img_page.rect.width)
            page_height = float(img_page.rect.height)
        except Exception:
            page_width = float(PAGE_W)
            page_height = float(BASE_PAGE_H)
            img_page = None

        region_bbox = _compute_mid_band_image_region_bbox(
            p=p,
            section=section,
            page_width=page_width,
            page_height=page_height,
        )
        source_cells = _detect_mid_band_image_cells(img_page, region_bbox) if img_page is not None else []
        slot_to_cell: Dict[str, Optional[dict]] = {}
        strict_side_guardrails = str(p.side_letter or "").upper() in {"B", "D"}
        ac_pixmap_enabled = str(p.side_letter or "").upper() in {"A", "C"}
        row_strip_lookup: Dict[int, dict] = {}
        pixmap_page_img: Optional[Image.Image] = None
        pixmap_page_zoom = 3.0
        pixmap_source_region_bbox: Optional[Tuple[float, float, float, float]] = None
        pixmap_grid_source = ""
        ac_pixmap_cells: List[dict] = []
        pixmap_flat_cells: List[dict] = []
        if strict_side_guardrails and cols == 8:
            try:
                pixmap_page_img, pixmap_page_zoom = _render_page_pixmap_image(img_page, zoom=3.0)
            except Exception:
                pixmap_page_img = None
            pixmap_slot_grid, pixmap_source_region_bbox, pixmap_grid_source = _build_bd_mid_band_pixmap_grid(
                source_cells,
                page_width,
                page_height,
                pixmap_page_img,
                pixmap_page_zoom,
            )
            if pixmap_slot_grid:
                for source_row, row in enumerate(pixmap_slot_grid):
                    for source_col, bbox in enumerate(row):
                        pixmap_flat_cells.append(
                            {
                                "bbox": bbox,
                                "cell_id": f"BD_R{source_row + 1}S{source_col + 1}",
                                "source_row_index": source_row,
                                "source_slot_index": source_col,
                            }
                        )
        elif ac_pixmap_enabled:
            try:
                pixmap_page_img, pixmap_page_zoom = _render_page_pixmap_image(img_page, zoom=3.0)
            except Exception:
                pixmap_page_img = None
            ac_pixmap_cells, pixmap_source_region_bbox, pixmap_grid_source = _build_ac_mid_band_pixmap_cells(
                source_cells,
                expected_count=len(all_slots),
            )
            pixmap_flat_cells = ac_pixmap_cells
            pixmap_slot_grid = None
        else:
            pixmap_slot_grid = None
        if detection_debug is not None:
            detection_debug["side"] = p.side_letter
            detection_debug["image_page_index"] = p.page_index
            detection_debug["mid_band_region_bbox_used"] = list(region_bbox)
            detection_debug["source_image_cell_count"] = len(source_cells)
            detection_debug["ordered_source_cell_ids"] = [str(c.get("cell_id", "")) for c in source_cells]
            detection_debug["strict_side_guardrails"] = strict_side_guardrails
            detection_debug["row_strip_fallback_available"] = bool(row_strip_lookup)
            detection_debug["row_strip_source_cell_ids"] = [
                str(cell.get("cell_id", "")) for _, cell in sorted(row_strip_lookup.items())
            ]
            detection_debug["pixmap_middle_band_available"] = bool(pixmap_page_img is not None)
            detection_debug["pixmap_source_region_bbox"] = list(pixmap_source_region_bbox) if pixmap_source_region_bbox else None
            detection_debug["pixmap_grid_source"] = pixmap_grid_source
            if ac_pixmap_enabled:
                detection_debug["ac_pixmap_crop_count"] = len(ac_pixmap_cells)
                detection_debug["ac_pixmap_crop_order"] = [
                    {
                        "source_index": int(i),
                        "source_cell_id": str(cell.get("cell_id", "")),
                        "source_bbox": list(cell.get("bbox")) if cell.get("bbox") else None,
                    }
                    for i, cell in enumerate(ac_pixmap_cells)
                ]
        if layout_debug is not None:
            layout_debug.update(
                {
                    "side": p.side_letter,
                    "mode": "token_first_mid_band",
                    "layout_mode": str(plan.get("layout_mode", "token_first")),
                    "candidate_card_count": len(all_candidate_slots),
                    "card_count": len(all_slots),
                    "columns_used": cols,
                    "rows_used": rows,
                    "card_w": card_w,
                    "card_h": card_h,
                    "h_gutter_intra": intra_gap if cols == 8 else None,
                    "h_gutter_inter": inter_gap if cols == 8 else None,
                    "h_gutter_uniform": col_gap if cols != 8 else None,
                    "v_gutter": row_gutter,
                    "origin_x": start_x,
                    "origin_y": grid_top,
                }
            )

        mapped_clean = 0
        mapped_fallback = 0
        mapped_empty = 0
        row_strip_slice_count = 0
        row_strip_slice_rejected_count = 0
        suspicious_crop_count = 0
        residual_bad_crop_warning_count = 0
        pixmap_crop_count = 0
        pixmap_fallback_count = 0
        pixmap_suspicious_count = 0
        exact_image_binding_count = 0
        last5_image_binding_count = 0
        unbound_image_count = 0
        prevented_wrong_image_count = 0
        render_assignment_debug: List[dict] = []
        image_crop_sources_by_slot: List[dict] = []
        blank_image_count = 0

        for idx, slot in enumerate(all_slots):
            selected_candidate = selection.selected_cards[idx] if idx < len(selection.selected_cards) else {}
            selected_source_index = int(selected_candidate.get("source_index", idx))
            ri = idx // cols
            ci = idx % cols
            x = row_xs[ci]
            y = grid_top - (ri + 1) * card_h - ri * row_gutter

            match = selected_candidate.get("resolved_match")
            resolve_trace = dict(selected_candidate.get("resolve_trace") or {})
            if match is None:
                match, resolve_trace = _resolve_mid_band_slot(p, slot)
            if (
                position_fallback_debug_rows is not None
                and str(resolve_trace.get("fallback_path", "")).startswith("template_position_")
            ):
                position_fallback_debug_rows.append(
                    {
                        "side": p.side_letter,
                        "slot_id": slot.slot_id,
                        "row_index": slot.row_index,
                        "slot_in_row": slot.slot_in_row,
                        "fallback_scope": resolve_trace.get("position_fallback_scope"),
                        "fallback_candidate_upcs": resolve_trace.get("fallback_candidate_upcs", []),
                        "fallback_similarity_scores": resolve_trace.get("fallback_similarity_scores", {}),
                        "chosen_candidate": resolve_trace.get("chosen_candidate"),
                    }
                )

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
                if unresolved_debug_rows is not None:
                    unresolved_debug_rows.append(
                        {
                            "page_index": p.page_index,
                            "side": p.side_letter,
                            "slot_id": slot.slot_id,
                            "row_index": slot.row_index,
                            "slot_in_row": slot.slot_in_row,
                            "block_name": slot.block_name,
                            "labels_last5": resolve_trace.get("labels_last5"),
                            "labels_text": (slot.raw_label_text or "").strip(),
                            "fallback_path": resolve_trace.get("fallback_path"),
                            "label_hint_upc_candidates": resolve_trace.get("label_hint_upc_candidates", []),
                            "label_hint_last5_candidates": resolve_trace.get("label_hint_last5_candidates", []),
                            "fallback_candidate_upcs": resolve_trace.get("fallback_candidate_upcs", []),
                            "fallback_similarity_scores": resolve_trace.get("fallback_similarity_scores", {}),
                            "chosen_candidate": resolve_trace.get("chosen_candidate"),
                            "position_fallback_scope": resolve_trace.get("position_fallback_scope"),
                            "excel_lookup_succeeded": bool(resolve_trace.get("excel_lookup_succeeded")),
                        }
                    )
                unmatched_cells += 1

            mapped_cell = slot_to_cell.get(slot.slot_id)
            candidate_crop_count = 1  # always include slot-geometry fallback candidate
            rejected_crop_reasons: List[str] = []
            constrained_by_slot_geometry = False
            fallback_reason = ""
            image_crop_bbox_used, _legacy_source, _legacy_grid = _derive_token_first_image_bbox(
                slot=slot,
                section=section,
                page_width=page_width,
                page_height=page_height,
            )
            crop_selection_path = "fallback_token_crop"
            crop_flagged_suspicious = False
            suspicious_reasons: List[str] = []
            fallback_used = False
            fallback_type = ""
            final_crop_slot_aligned = False
            neighboring_overlap_detected = False
            residual_bad_crop_warning = False
            row_strip_cell = row_strip_lookup.get(int(ri))
            row_strip_slice_bbox: Optional[Tuple[float, float, float, float]] = None
            row_strip_reasons: List[str] = []
            if row_strip_cell is not None:
                candidate_crop_count += 1
                row_strip_slice_count += 1
                row_strip_slice_rejected_count += 1
                row_strip_reasons = [
                    "row_strip_slicing_disabled",
                    "grouped_source_cell_not_safe_for_equal_slice",
                ]
                crop_flagged_suspicious = True
                suspicious_reasons = row_strip_reasons

            if mapped_cell is not None:
                candidate_crop_count += 1
                mapped_bbox = tuple(mapped_cell.get("bbox", slot.bbox))
                mapped_ok, reject_reasons = _validate_mid_band_mapped_cell_bbox(
                    mapped_bbox=mapped_bbox,
                    fallback_bbox=image_crop_bbox_used,
                    strict_side_guardrails=strict_side_guardrails,
                )
                mx0, my0, mx1, my1 = map(float, mapped_bbox)
                mw = max(0.0, mx1 - mx0)
                mh = max(0.0, my1 - my0)
                mapped_aspect = (mw / mh) if mh > 0.0 else 99.0
                if strict_side_guardrails and mw >= 120.0 and mh >= 36.0 and mapped_aspect >= 1.65:
                    mapped_ok = False
                    reject_reasons = list(reject_reasons) + ["grouped_row_strip_rejected"]
                if mapped_ok:
                    image_crop_bbox_used = mapped_bbox
                    image_crop_source = "image_cell_match"
                    used_slot_grid_crop_path = False
                    crop_selection_path = "image_cell_match"
                    mapped_clean += 1
                else:
                    rejected_crop_reasons = list(row_strip_reasons) + list(reject_reasons)
                    image_crop_source = "fallback_token_crop"
                    used_slot_grid_crop_path = bool(_legacy_grid)
                    constrained_by_slot_geometry = True
                    fallback_reason = "rejected_image_cell_candidate"
                    crop_selection_path = "fallback_token_crop"
                    fallback_used = True
                    fallback_type = "previous_fallback"
                    final_crop_slot_aligned = bool(_legacy_grid)
                    crop_flagged_suspicious = True
                    suspicious_reasons = list(rejected_crop_reasons)
                    mapped_fallback += 1
            elif image_crop_bbox_used:
                image_crop_source = "fallback_token_crop"
                used_slot_grid_crop_path = bool(_legacy_grid)
                constrained_by_slot_geometry = True
                fallback_reason = "no_source_image_cell_for_slot"
                rejected_crop_reasons = list(row_strip_reasons) + ["no_source_image_cell_for_slot"]
                crop_selection_path = "fallback_token_crop"
                fallback_used = True
                fallback_type = "previous_fallback"
                final_crop_slot_aligned = bool(_legacy_grid)
                crop_flagged_suspicious = True
                suspicious_reasons = list(rejected_crop_reasons)
                mapped_fallback += 1
            else:
                image_crop_bbox_used, _legacy_source, _legacy_grid = _derive_token_first_image_bbox(
                    slot=slot,
                    section=section,
                    page_width=page_width,
                    page_height=page_height,
                )
                if image_crop_bbox_used:
                    image_crop_source = "fallback_token_crop"
                    used_slot_grid_crop_path = bool(_legacy_grid)
                    constrained_by_slot_geometry = True
                    fallback_reason = "legacy_fallback_crop"
                    crop_selection_path = "fallback_token_crop"
                    fallback_used = True
                    fallback_type = "previous_fallback"
                    final_crop_slot_aligned = bool(_legacy_grid)
                    crop_flagged_suspicious = True
                    suspicious_reasons = list(row_strip_reasons) + [fallback_reason]
                    mapped_fallback += 1
                else:
                    image_crop_source = "fallback_none"
                    used_slot_grid_crop_path = False
                    constrained_by_slot_geometry = False
                    fallback_reason = "no_crop_candidate"
                    crop_selection_path = "fallback_none"
                    crop_flagged_suspicious = True
                    suspicious_reasons = [fallback_reason]
                    residual_bad_crop_warning = True
                    mapped_empty += 1
            if image_crop_source == "image_cell_match":
                chosen_crop_source_type = "individual_image_cell"
            elif image_crop_source == "fallback_token_crop":
                chosen_crop_source_type = "previous_fallback"
            elif image_crop_source == "fallback_none":
                chosen_crop_source_type = "none"
            else:
                chosen_crop_source_type = str(image_crop_source)
            crop_error: Optional[str] = None
            pixmap_debug: Dict[str, object] = {}
            pixmap_slot_bbox: Optional[Tuple[float, float, float, float]] = None
            pixmap_slot_bbox_before_padding: Optional[Tuple[float, float, float, float]] = None
            ac_pixmap_source_cell: Optional[dict] = None
            image_binding_method = "fallback_none"
            image_identity_confident = False
            wrong_image_fallback_prevented = False
            selected_image_source_id: Optional[str] = None
            img: Optional[Image.Image] = None
            image_source_index = int(idx)
            if (
                (ac_pixmap_enabled or strict_side_guardrails)
                and pixmap_page_img is not None
                and 0 <= image_source_index < len(pixmap_flat_cells)
            ):
                ac_pixmap_source_cell = pixmap_flat_cells[int(image_source_index)]
                source_bbox = ac_pixmap_source_cell.get("bbox")
                if source_bbox and len(source_bbox) == 4:
                    pixmap_slot_bbox_before_padding = tuple(map(float, source_bbox))
                    pixmap_slot_bbox = (
                        pixmap_slot_bbox_before_padding
                        if ac_pixmap_enabled
                        else _pad_mid_band_pixmap_slot_bbox(pixmap_slot_bbox_before_padding)
                    )
                    pix_img, pix_bbox, pixmap_debug = _crop_mid_band_pixmap_slot(
                        pixmap_page_img,
                        pixmap_page_zoom,
                        pixmap_slot_bbox,
                    )
                    if pix_img is not None and pix_bbox is not None:
                        img = pix_img
                        image_crop_bbox_used = pix_bbox
                        image_crop_source = "pixmap_middle_band"
                        chosen_crop_source_type = "pixmap_middle_band"
                        crop_selection_path = "image_pdf_crop_i_to_clean_label_slot_i"
                        fallback_used = False
                        fallback_type = ""
                        final_crop_slot_aligned = True
                        crop_flagged_suspicious = False
                        suspicious_reasons = []
                        image_binding_method = "ordered_clean_slot_index"
                        image_identity_confident = True
                        selected_image_source_id = str(ac_pixmap_source_cell.get("cell_id", ""))
                        pixmap_crop_count += 1
                    else:
                        pixmap_fallback_count += 1
                        pixmap_suspicious_count += 1
                        if pixmap_debug.get("suspicious_reason"):
                            rejected_crop_reasons = list(rejected_crop_reasons) + list(pixmap_debug.get("suspicious_reason", []))
                            suspicious_reasons = list(suspicious_reasons) + list(pixmap_debug.get("suspicious_reason", []))
                        crop_flagged_suspicious = True
                        fallback_used = True
                else:
                    pixmap_fallback_count += 1
                    pixmap_suspicious_count += 1
                    rejected_crop_reasons = list(rejected_crop_reasons) + ["missing_ac_pixmap_source_bbox"]
                    suspicious_reasons = list(suspicious_reasons) + ["missing_ac_pixmap_source_bbox"]
                    crop_flagged_suspicious = True
                    fallback_used = True
            elif ac_pixmap_enabled or strict_side_guardrails:
                rejected_crop_reasons = list(rejected_crop_reasons) + ["no_ordered_images_pdf_art_cell_for_clean_slot_index"]
                suspicious_reasons = list(suspicious_reasons) + ["no_ordered_images_pdf_art_cell_for_clean_slot_index"]
                crop_flagged_suspicious = True
                fallback_used = True

            if img is None and image_crop_source == "image_cell_match":
                image_binding_method = "exact_last5"
                image_identity_confident = True
                selected_image_source_id = mapped_cell.get("cell_id") if mapped_cell else None

            if img is None and not image_identity_confident:
                wrong_image_fallback_prevented = bool(image_crop_source != "fallback_none")
                image_crop_source = "fallback_none"
                chosen_crop_source_type = "none"
                crop_selection_path = "fallback_none_identity_unbound"
                fallback_reason = "no_confident_resolved_item_image_binding"
                rejected_crop_reasons = list(rejected_crop_reasons) + [fallback_reason]
                suspicious_reasons = list(suspicious_reasons) + [fallback_reason]
                crop_flagged_suspicious = True
                image_crop_bbox_used = None

            if img is None and image_identity_confident:
                try:
                    img = crop_image_cell(
                        images_doc,
                        p.page_index,
                        image_crop_bbox_used,
                        zoom=float(plan["crop_zoom"]),
                        inset=float(plan["crop_inset"]),
                    )
                except Exception as exc:
                    img = None
                    crop_error = str(exc)
                    if not residual_bad_crop_warning:
                        residual_bad_crop_warning_count += 1
                    residual_bad_crop_warning = True

            if crop_flagged_suspicious:
                suspicious_crop_count += 1
            if residual_bad_crop_warning and crop_error is None:
                residual_bad_crop_warning_count += 1
            if image_identity_confident:
                if image_binding_method == "exact_upc":
                    exact_image_binding_count += 1
                elif image_binding_method == "exact_last5":
                    last5_image_binding_count += 1
            else:
                unbound_image_count += 1
            if wrong_image_fallback_prevented:
                prevented_wrong_image_count += 1

            if img is None:
                missing_image_slots.append(slot.slot_id)
                blank_image_count += 1

            image_crop_sources_by_slot.append(
                {
                    "slot_index": int(idx),
                    "slot_id": slot.slot_id,
                    "artwork_crop_source": image_crop_source,
                    "crop_selection_path": crop_selection_path,
                    "source_cell_id": selected_image_source_id,
                    "crop_present": bool(img is not None),
                }
            )

            if debug and crop_debug_rows is not None:
                bw = float(image_crop_bbox_used[2] - image_crop_bbox_used[0]) if image_crop_bbox_used else 0.0
                bh = float(image_crop_bbox_used[3] - image_crop_bbox_used[1]) if image_crop_bbox_used else 0.0
                crop_aspect = round(bw / bh, 4) if bh > 0.0 else None
                crop_debug_rows.append(
                    {
                        "side": p.side_letter,
                        "page_index": p.page_index,
                        "slot_id": slot.slot_id,
                        "render_row": int(ri),
                        "render_col": int(ci),
                        "row_index": slot.row_index,
                        "slot_in_row": slot.slot_in_row,
                        "slot_order": slot.slot_order,
                        "last5": _to_last5(slot.last5),
                        "resolved_last5": _to_last5(upc12) if upc12 else _to_last5(slot.last5),
                        "resolved_upc12": upc12,
                        "resolved_display_name": disp_name,
                        "resolved_cpp": cpp,
                        "selected_source_index": int(selected_source_index),
                        "selected_image_source_id": selected_image_source_id,
                        "image_binding_method": image_binding_method,
                        "metadata_image_identity_matched": bool(image_identity_confident),
                        "wrong_image_fallback_prevented": bool(wrong_image_fallback_prevented),
                        "token_bbox": list(slot.bbox) if slot.bbox else None,
                        "slot_bbox": list(slot.bbox) if slot.bbox else None,
                        "extraction_bbox": list(slot.extraction_bbox) if slot.extraction_bbox else None,
                        "image_crop_bbox_used": list(image_crop_bbox_used) if image_crop_bbox_used else None,
                        "chosen_source_crop_bbox": list(image_crop_bbox_used) if image_crop_bbox_used else None,
                        "chosen_source_crop_width": bw,
                        "chosen_source_crop_height": bh,
                        "chosen_source_crop_aspect_ratio": crop_aspect,
                        "image_crop_source": image_crop_source,
                        "chosen_crop_source_type": chosen_crop_source_type,
                        "render_path_used": chosen_crop_source_type,
                        "pixmap_source_region_bbox": list(pixmap_source_region_bbox) if pixmap_source_region_bbox else None,
                        "pixmap_slot_bbox": list(pixmap_slot_bbox) if pixmap_slot_bbox else None,
                        "pixmap_slot_bbox_before_padding": list(pixmap_slot_bbox_before_padding) if pixmap_slot_bbox_before_padding else None,
                        "pixmap_slot_bbox_after_padding": list(pixmap_slot_bbox) if pixmap_slot_bbox else None,
                        "ac_pixmap_source_index": int(selected_source_index) if ac_pixmap_source_cell is not None else None,
                        "ac_pixmap_source_cell_id": ac_pixmap_source_cell.get("cell_id") if ac_pixmap_source_cell else None,
                        "ac_pixmap_source_bbox": list(ac_pixmap_source_cell.get("bbox")) if ac_pixmap_source_cell and ac_pixmap_source_cell.get("bbox") else None,
                        "pixmap_crop_pixel_width": int(getattr(img, "width")) if image_crop_source == "pixmap_middle_band" and img is not None else None,
                        "pixmap_crop_pixel_height": int(getattr(img, "height")) if image_crop_source == "pixmap_middle_band" and img is not None else None,
                        "pixmap_whitespace_score": pixmap_debug.get("whitespace_score"),
                        "pixmap_suspicious_crop": bool(pixmap_debug.get("suspicious_crop", False)),
                        "pixmap_suspicious_reason": pixmap_debug.get("suspicious_reason", []),
                        "crop_selection_path": crop_selection_path,
                        "crop_flagged_suspicious": bool(crop_flagged_suspicious),
                        "suspicious_reasons": suspicious_reasons,
                        "rejected_row_strip_slice": bool(row_strip_reasons),
                        "row_strip_slice_reject_reasons": row_strip_reasons,
                        "fallback_used": bool(fallback_used),
                        "fallback_type": fallback_type,
                        "final_crop_slot_aligned": bool(final_crop_slot_aligned),
                        "neighboring_overlap_detected": bool(neighboring_overlap_detected),
                        "residual_bad_crop_warning": bool(residual_bad_crop_warning),
                        "row_strip_source_cell_id": row_strip_cell.get("cell_id") if row_strip_cell else None,
                        "row_strip_source_bbox": list(row_strip_cell.get("bbox")) if row_strip_cell and row_strip_cell.get("bbox") else None,
                        "candidate_crop_count": candidate_crop_count,
                        "rejected_crop_reasons": rejected_crop_reasons,
                        "constrained_by_art_cell_geometry": bool(constrained_by_slot_geometry),
                        "fallback_reason": fallback_reason,
                        "used_slot_grid_crop_path": bool(used_slot_grid_crop_path),
                        "crop_path_used": image_crop_source,
                        "derived_bbox_width": bw,
                        "derived_bbox_height": bh,
                        "crop_zoom": float(plan["crop_zoom"]),
                        "crop_inset": float(plan["crop_inset"]),
                        "crop_success": bool(img is not None),
                        "crop_image_width": int(getattr(img, "width")) if img is not None and hasattr(img, "width") else None,
                        "crop_image_height": int(getattr(img, "height")) if img is not None and hasattr(img, "height") else None,
                        "tiny_bbox": bool(bw < 16.0 or bh < 16.0),
                        "large_bbox": bool(bw > 140.0 or bh > 90.0),
                        "crop_error": crop_error,
                    }
                )
            if debug and mapping_debug_rows is not None:
                mapping_debug_rows.append(
                    {
                        "side": p.side_letter,
                        "slot_id": slot.slot_id,
                        "row_index": slot.row_index,
                        "slot_in_row": slot.slot_in_row,
                        "slot_order": slot.slot_order,
                        "resolved_last5": _to_last5(slot.last5),
                        "resolved_upc12": upc12,
                        "resolved_name": disp_name,
                        "selected_source_index": int(selected_source_index),
                        "selected_image_source_id": selected_image_source_id,
                        "image_binding_method": image_binding_method,
                        "metadata_image_identity_matched": bool(image_identity_confident),
                        "wrong_image_fallback_prevented": bool(wrong_image_fallback_prevented),
                        "chosen_image_source_page": p.page_index,
                        "chosen_source_image_cell_id": (mapped_cell.get("cell_id") if mapped_cell else None),
                        "row_strip_source_cell_id": row_strip_cell.get("cell_id") if row_strip_cell else None,
                        "chosen_crop_bbox": list(image_crop_bbox_used) if image_crop_bbox_used else None,
                        "crop_source_type": image_crop_source,
                        "chosen_crop_source_type": chosen_crop_source_type,
                        "render_path_used": chosen_crop_source_type,
                        "pixmap_source_region_bbox": list(pixmap_source_region_bbox) if pixmap_source_region_bbox else None,
                        "pixmap_slot_bbox": list(pixmap_slot_bbox) if pixmap_slot_bbox else None,
                        "pixmap_slot_bbox_before_padding": list(pixmap_slot_bbox_before_padding) if pixmap_slot_bbox_before_padding else None,
                        "pixmap_slot_bbox_after_padding": list(pixmap_slot_bbox) if pixmap_slot_bbox else None,
                        "ac_pixmap_source_index": int(selected_source_index) if ac_pixmap_source_cell is not None else None,
                        "ac_pixmap_source_cell_id": ac_pixmap_source_cell.get("cell_id") if ac_pixmap_source_cell else None,
                        "ac_pixmap_source_bbox": list(ac_pixmap_source_cell.get("bbox")) if ac_pixmap_source_cell and ac_pixmap_source_cell.get("bbox") else None,
                        "pixmap_whitespace_score": pixmap_debug.get("whitespace_score"),
                        "pixmap_suspicious_crop": bool(pixmap_debug.get("suspicious_crop", False)),
                        "pixmap_suspicious_reason": pixmap_debug.get("suspicious_reason", []),
                        "crop_selection_path": crop_selection_path,
                        "crop_flagged_suspicious": bool(crop_flagged_suspicious),
                        "suspicious_reasons": suspicious_reasons,
                        "rejected_row_strip_slice": bool(row_strip_reasons),
                        "row_strip_slice_reject_reasons": row_strip_reasons,
                        "fallback_used": bool(fallback_used),
                        "fallback_type": fallback_type,
                        "final_crop_slot_aligned": bool(final_crop_slot_aligned),
                        "neighboring_overlap_detected": bool(neighboring_overlap_detected),
                        "residual_bad_crop_warning": bool(residual_bad_crop_warning),
                        "candidate_crop_count": candidate_crop_count,
                        "rejected_crop_reasons": rejected_crop_reasons,
                        "constrained_by_art_cell_geometry": bool(constrained_by_slot_geometry),
                        "fallback_reason": fallback_reason,
                        "rendered_row": int(ri),
                        "rendered_col": int(ci),
                        "mismatch_note": (
                            "row_strip_source_cell_rejected_using_previous_fallback"
                            if row_strip_reasons
                            else "no_source_image_cell_detected_for_slot"
                            if mapped_cell is None
                            else ("rejected_image_cell_candidate" if rejected_crop_reasons else "")
                        ),
                    }
                )

            if x < content_x0 - 0.001 or (x + card_w) > (content_x0 + content_w + 0.001):
                overflow = True

            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(*FILLED_STROKE)
            c.setLineWidth(0.75)
            c.rect(x, y, card_w, card_h, stroke=1, fill=1)
            norm_debug = _draw_mid_band_card(
                c,
                x,
                y,
                card_w,
                card_h,
                img,
                upc_str,
                disp_name,
                cpp,
                section="mid_band",
                side=p.side_letter,
                final_index=int(idx),
                row=int(ri),
                col=int(ci),
                source_crop_bbox=image_crop_bbox_used,
            )
            if normalization_debug_rows is not None:
                normalization_debug_rows.append(norm_debug)

            if debug_overlay:
                c.setFillColorRGB(0.35, 0.35, 0.35)
                c.setFont("Helvetica", 6)
                c.drawString(x + 2, y + card_h - 8, slot.slot_id)

            if layout_assignment_rows is not None:
                layout_assignment_rows.append(
                    {
                        "side": p.side_letter,
                        "final_index": int(idx),
                        "slot_id": slot.slot_id,
                        "rendered_row": int(ri),
                        "rendered_col": int(ci),
                        "group": selected_candidate.get("selected_group"),
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "upc12": upc12,
                    }
                )
            render_assignment_debug.append(
                {
                    "final_index": int(idx),
                    "row": int(ri),
                    "col": int(ci),
                    "group": selected_candidate.get("selected_group"),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "upc12": upc12,
                    "slot_id": slot.slot_id,
                    "image_draw_bbox": norm_debug.get("image_draw_bbox"),
                    "text_bbox": norm_debug.get("text_bbox"),
                    "overflow_or_bleed_detected": norm_debug.get("overflow_or_bleed_detected"),
                }
            )
            slots_drawn += 1

        if detection_debug is not None:
            rejected_middle_candidates = list(selection_debug.get("rejected_middle_candidates", [])) if selection_debug else []
            raw_middle_candidate_count = int(selection_debug.get("raw_middle_candidate_count", len(all_candidate_slots))) if selection_debug else len(all_candidate_slots)
            cleaned_middle_candidate_count = int(selection_debug.get("cleaned_middle_candidate_count", len(all_slots))) if selection_debug else len(all_slots)
            expected_middle_count = int(selection_debug.get("expected_selected_count", 24)) if selection_debug else 24
            missing_middle_count = max(0, expected_middle_count - slots_drawn)
            ordered_image_crop_count = max(0, min(expected_middle_count, len(pixmap_flat_cells)))
            metadata_order_preview = []
            image_order_preview = []
            final_join_preview = []
            for preview_index, candidate in enumerate(selection.selected_cards[:expected_middle_count]):
                preview_match = candidate.get("resolved_match")
                preview_slot = candidate.get("slot")
                preview_upc = (
                    getattr(preview_match, "upc12", None)
                    or candidate.get("upc12")
                    or getattr(preview_slot, "resolved_upc12", None)
                )
                preview_name = (
                    getattr(preview_match, "display_name", None)
                    or candidate.get("resolved_name")
                    or candidate.get("display_name")
                    or getattr(preview_slot, "resolved_display_name", None)
                    or getattr(preview_slot, "parsed_name", "")
                )
                preview_cpp = (
                    getattr(preview_match, "cpp_qty", None)
                    if preview_match is not None
                    else candidate.get("cpp")
                )
                image_cell = pixmap_flat_cells[preview_index] if preview_index < len(pixmap_flat_cells) else None
                metadata_order_preview.append(
                    {
                        "index": int(preview_index),
                        "upc": preview_upc,
                        "name": preview_name,
                        "cpp": preview_cpp,
                    }
                )
                image_order_preview.append(
                    {
                        "index": int(preview_index),
                        "source": "images_pdf",
                        "cell_id": str(image_cell.get("cell_id", "")) if image_cell else None,
                        "bbox": list(image_cell.get("bbox")) if image_cell and image_cell.get("bbox") else None,
                    }
                )
                final_join_preview.append(
                    {
                        "index": int(preview_index),
                        "metadata_upc": preview_upc,
                        "metadata_name": preview_name,
                        "image_source": "images_pdf" if image_cell else None,
                        "image_cell_id": str(image_cell.get("cell_id", "")) if image_cell else None,
                        "image_bbox": list(image_cell.get("bbox")) if image_cell and image_cell.get("bbox") else None,
                    }
                )
            detection_debug["active_mid_band_render_path"] = "token_first_mid_band"
            detection_debug["active_middle_band_path"] = "cell_middle_grid_above_bonus"
            detection_debug["side"] = p.side_letter
            detection_debug["visual_source"] = "images_pdf"
            detection_debug["labels_pdf_visual_crop_used"] = False
            detection_debug["labels_pdf_visual_source_enabled"] = False
            detection_debug["middle_band_binding_mode"] = "image_pdf_crop_i_to_clean_label_slot_i"
            detection_debug["slot_binding_mode"] = "image_pdf_crop_i_to_clean_label_slot_i"
            detection_debug["binding_mode"] = "image_pdf_crop_i_to_metadata_i"
            detection_debug["raw_middle_candidate_count"] = raw_middle_candidate_count
            detection_debug["raw_candidate_count"] = raw_middle_candidate_count
            detection_debug["cleaned_middle_candidate_count"] = cleaned_middle_candidate_count
            detection_debug["clean_metadata_count"] = cleaned_middle_candidate_count
            detection_debug["ordered_label_slot_count"] = len(all_slots)
            detection_debug["ordered_image_crop_count"] = ordered_image_crop_count
            detection_debug["image_crop_count"] = ordered_image_crop_count
            detection_debug["rendered_middle_slot_count"] = slots_drawn
            detection_debug["rendered_count"] = slots_drawn
            detection_debug["missing_middle_slots"] = missing_middle_count
            detection_debug["count_is_24"] = bool(
                len(all_slots) == expected_middle_count
                and ordered_image_crop_count == expected_middle_count
                and slots_drawn == expected_middle_count
            )
            detection_debug["rejected_middle_candidate_count"] = len(rejected_middle_candidates)
            detection_debug["rejected_candidate_count"] = len(rejected_middle_candidates)
            detection_debug["rejected_middle_candidates"] = rejected_middle_candidates
            detection_debug["metadata_order_preview"] = metadata_order_preview
            detection_debug["image_order_preview"] = image_order_preview
            detection_debug["final_join_preview"] = final_join_preview
            detection_debug["image_crop_sources_by_slot"] = image_crop_sources_by_slot
            detection_debug["invalid_label_text_crop_count"] = 0
            detection_debug["blank_image_count"] = blank_image_count
            detection_debug["resolved_mid_band_slot_count"] = len(all_slots)
            detection_debug["candidate_mid_band_slot_count"] = len(all_candidate_slots)
            detection_debug["mapped_clean_count"] = mapped_clean
            detection_debug["mapped_fallback_count"] = mapped_fallback
            detection_debug["mapped_empty_count"] = mapped_empty
            detection_debug["row_strip_slice_count"] = row_strip_slice_count
            detection_debug["row_strip_slice_rejected_count"] = row_strip_slice_rejected_count
            detection_debug["individual_image_cell_used_count"] = mapped_clean
            detection_debug["rendered_page_detected_cell_used_count"] = pixmap_crop_count
            detection_debug["pixmap_crop_count"] = pixmap_crop_count
            detection_debug["pixmap_fallback_count"] = pixmap_fallback_count
            detection_debug["pixmap_suspicious_crop_count"] = pixmap_suspicious_count
            detection_debug["previous_fallback_used_count"] = mapped_fallback
            detection_debug["suspicious_crop_count"] = suspicious_crop_count
            detection_debug["fallback_used_count"] = mapped_fallback
            detection_debug["residual_bad_crop_warning_count"] = residual_bad_crop_warning_count
            detection_debug["image_binding_summary"] = {
                "total_middle_band_slots": len(all_slots),
                "slots_with_exact_image_binding": exact_image_binding_count,
                "slots_with_last5_based_binding": last5_image_binding_count,
                "slots_with_no_confident_image_binding": unbound_image_count,
                "slots_where_legacy_positional_fallback_was_prevented": prevented_wrong_image_count,
            }
        if selection_debug is not None:
            selected_ids = [r.get("slot_id") for r in selection_debug.get("selected_order", [])]
            rendered_ids = [r.get("slot_id") for r in render_assignment_debug]
            selection_debug["render_assignment"] = render_assignment_debug
            selection_debug["actual_selected_count"] = len(selection.selected_cards)
            selection_debug["final_render_order_matches_selected_order"] = rendered_ids == selected_ids

        sec_bottom = grid_top - rows * card_h - max(0, rows - 1) * row_gutter
        return cols, rows, overflow, sec_bottom, slots_drawn, cols

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
                extraction_mode = (
                    "token_first"
                    if "token_first" in str(mid_band.section_id or "").lower()
                    else "template_slots"
                )
                st.write(
                    {
                        "side": pdata.side_letter,
                        "mid_band_extract_debug": {
                            "extract_path": extraction_mode,
                            "whitelist_only": extraction_mode != "token_first",
                            "present": True,
                            "shape_valid": mid_band.shape_valid,
                            "anchor_bbox": list(mid_band.anchor_bbox) if mid_band.anchor_bbox else None,
                            "slot_count": mid_band.slot_count,
                            "row_slot_counts": mid_band.row_slot_counts,
                            "row_block_grouping": mid_band.row_block_grouping,
                            "slot_ids_sorted": [s.slot_id for s in sorted(mid_slots_dbg, key=lambda s: (s.row_index, s.slot_in_row))],
                            "slot_last5": [_to_last5(s.last5) for s in sorted(mid_slots_dbg, key=lambda s: (s.row_index, s.slot_in_row))],
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

                mid_rows: List[dict] = []
                fallback_path_counts: Dict[str, int] = {}
                slots_with_last5 = 0
                slots_resolved = 0

                for slot in mid_slots_dbg:
                    resolved_row, resolve_trace = _resolve_mid_band_slot(pdata, slot)
                    slot_last5 = _to_last5(slot.last5)
                    if slot_last5:
                        slots_with_last5 += 1
                    if resolved_row is not None:
                        slots_resolved += 1

                    fallback_path = str(resolve_trace.get("fallback_path", "") or "")
                    fallback_path_counts[fallback_path] = fallback_path_counts.get(fallback_path, 0) + 1

                    mid_rows.append(
                        {
                            "side": pdata.side_letter,
                            "page_index": pdata.page_index,
                            "slot_id": slot.slot_id,
                            "row_index": slot.row_index,
                            "slot_in_row": slot.slot_in_row,
                            "block_name": slot.block_name,
                            "bbox": list(slot.bbox) if slot.bbox else None,
                            "extraction_bbox": list(slot.extraction_bbox) if slot.extraction_bbox else None,
                            "accepted_words": slot.accepted_words or [],
                            "rejected_nearby_word_count": int(slot.rejected_nearby_word_count or 0),
                            "raw_label_text": slot.raw_label_text,
                            "parsed_name": slot.parsed_name,
                            "slot_last5": slot_last5,
                            "resolve_fallback_path": fallback_path,
                            "resolved_upc12": resolved_row.upc12 if resolved_row else None,
                            "resolved_display_name": resolved_row.display_name if resolved_row else None,
                            "resolved_cpp": resolved_row.cpp_qty if resolved_row else None,
                            "label_hint_last5_candidates": resolve_trace.get("label_hint_last5_candidates", []),
                            "label_hint_upc_candidates": resolve_trace.get("label_hint_upc_candidates", []),
                            "excel_lookup_succeeded": bool(resolve_trace.get("excel_lookup_succeeded", resolved_row is not None)),
                        }
                    )

                total_slots = len(mid_slots_dbg)
                slots_missing_last5 = total_slots - slots_with_last5
                slots_unresolved = total_slots - slots_resolved

                st.write(f"FULL_PALLET mid-band slot debug - Side {pdata.side_letter}")
                st.write(
                    {
                        "total_slots": total_slots,
                        "slots_with_last5": slots_with_last5,
                        "slots_missing_last5": slots_missing_last5,
                        "slots_resolved": slots_resolved,
                        "slots_unresolved": slots_unresolved,
                        "fallback_path_counts": fallback_path_counts,
                    }
                )
                st.dataframe(pd.DataFrame(mid_rows), use_container_width=True)

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

            token_first_slot_count = len([s for r in mid_band.rows for s in r.slots]) if mid_band is not None else 0
            token_first_mid_band_ok = bool(mid_band is not None and token_first_slot_count > 0)
            canonical_main_plan = _measure_canonical_mid_band(content_w, include_bar=False)
            token_first_main_plan = _measure_token_first_mid_band(
                content_w,
                token_first_slot_count,
                include_bar=False,
            )
            if token_first_mid_band_ok:
                main_plan = token_first_main_plan
            else:
                main_plan = canonical_main_plan
            bonus_plan = _measure_section_shape(
                pdata,
                slot_map["bonus"]["row_ids"],
                "bonus",
                content_w,
                global_cols,
                global_gap_units,
                include_bar=bool(slot_map["bonus"]["row_ids"]),
            )

            products_block_h = float(main_plan["total_h"])
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
            PAGE_H = max(BASE_PAGE_H, (2 * MARGIN) + HEADER_H + FOOTER_HEIGHT + required_content_h)

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
            _draw_footer(c, PAGE_W, MARGIN, FOOTER_HEIGHT)

            cy0, cy1 = MARGIN + FOOTER_HEIGHT, PAGE_H - MARGIN - HEADER_H
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
            unresolved_mid_slot_debug: List[dict] = []
            position_fallback_mid_slot_debug: List[dict] = []
            token_first_crop_debug_rows: List[dict] = []
            token_first_mapping_debug_rows: List[dict] = []
            token_first_detection_debug: Dict[str, object] = {}
            mid_band_layout_debug: Dict[str, object] = {}
            mid_band_layout_assignments: List[dict] = []
            mid_band_selection_debug: Dict[str, object] = {}
            mid_band_normalization_debug_rows: List[dict] = []
            bonus_crop_debug_rows: List[dict] = []
            missing_main_images: List[str] = []
            missing_bonus_images: List[str] = []
            main_render_source = "token_first_mid_band" if token_first_mid_band_ok else "mid_band_template_placeholder"
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
                            "fallback": "token_first_mid_band" if token_first_mid_band_ok else "disabled_for_mid_band",
                            "render_mode": "token_first_mid_band" if token_first_mid_band_ok else "mid_band_template_placeholder",
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

            if token_first_mid_band_ok and mid_band is not None:
                (
                    main_cols,
                    main_rows_count,
                    main_over,
                    main_bottom,
                    _main_occ_count,
                    _main_sec_cols,
                ) = _draw_token_first_mid_band_section(
                    pdata,
                    mid_band,
                    main_plan,
                    products_top,
                    unresolved_main,
                    missing_main_images,
                    cx0,
                    unresolved_debug_rows=unresolved_mid_slot_debug,
                    position_fallback_debug_rows=position_fallback_mid_slot_debug,
                    crop_debug_rows=token_first_crop_debug_rows,
                    mapping_debug_rows=token_first_mapping_debug_rows,
                    detection_debug=token_first_detection_debug,
                    layout_debug=mid_band_layout_debug,
                    layout_assignment_rows=mid_band_layout_assignments,
                    selection_debug=mid_band_selection_debug,
                    normalization_debug_rows=mid_band_normalization_debug_rows,
                )
            elif canonical_mid_band_ok and mid_band is not None:
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
                    main_plan,
                    products_top,
                    unresolved_main,
                    missing_main_images,
                    cx0,
                    unresolved_debug_rows=unresolved_mid_slot_debug,
                    position_fallback_debug_rows=position_fallback_mid_slot_debug,
                    layout_debug=mid_band_layout_debug,
                    layout_assignment_rows=mid_band_layout_assignments,
                    selection_debug=mid_band_selection_debug,
                    normalization_debug_rows=mid_band_normalization_debug_rows,
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
                    main_plan,
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
                    normalization_debug_rows=mid_band_normalization_debug_rows,
                    bonus_crop_debug_rows=bonus_crop_debug_rows,
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
            elif token_first_mid_band_ok and mid_band is not None:
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
            def _normalization_summary(section_name: str) -> Dict[str, object]:
                rows = [r for r in mid_band_normalization_debug_rows if r.get("section") == section_name]
                return {
                    "side": pdata.side_letter,
                    "section": section_name,
                    "rendered_count": len(rows),
                    "rendered_mid_band_count": len(rows) if section_name == "mid_band" else 0,
                    "normalized_count": sum(1 for r in rows if bool(r.get("normalization_applied"))),
                    "trim_applied_count": sum(1 for r in rows if bool(r.get("trim_applied"))),
                    "thin_crop_count": sum(1 for r in rows if bool(r.get("thin_crop_detected"))),
                    "whitespace_heavy_count": sum(1 for r in rows if bool(r.get("excessive_whitespace_detected"))),
                    "overflow_before_clamp_count": sum(
                        1 for r in rows if bool(r.get("overflow_before_clamp"))
                    ),
                    "clamped_to_fit_count": sum(1 for r in rows if bool(r.get("clamped_to_fit"))),
                    "overflow_or_bleed_warning_count": sum(
                        1
                        for r in rows
                        if bool(r.get("residual_bleed_warning", r.get("overflow_or_bleed_detected")))
                    ),
                    "trim_applied_indices": [
                        r.get("final_index") for r in rows if bool(r.get("trim_applied"))
                    ],
                    "clamped_to_fit_indices": [
                        r.get("final_index") for r in rows if bool(r.get("clamped_to_fit"))
                    ],
                    "overflow_or_bleed_indices": [
                        r.get("final_index") for r in rows if bool(r.get("overflow_or_bleed_detected"))
                    ],
                }

            product_normalization_summary = {
                "side": pdata.side_letter,
                "by_section": {
                    "mid_band": _normalization_summary("mid_band"),
                    "bonus": _normalization_summary("bonus"),
                },
            }
            mid_band_normalization_summary = product_normalization_summary["by_section"]["mid_band"]
            mid_band_profile_comparison = (
                _build_mid_band_profile_comparison(pdata, main_slots)
                if main_slots
                else {
                    "side": pdata.side_letter,
                    "profile_name": get_mid_band_physical_profile(pdata.side_letter).get("profile_name"),
                    "detected_candidate_count": 0,
                    "detected_group_counts": {},
                    "expected_group_counts": get_mid_band_physical_profile(pdata.side_letter).get(
                        "expected_max_per_group", {}
                    ),
                    "overage_per_group": {},
                    "shortage_per_group": {},
                    "candidate_upcs_by_group": {},
                    "retained_if_expected_max_enforced": {},
                    "omitted_if_expected_max_enforced": {},
                }
            )

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
                            "token_first_slot_count": token_first_slot_count if token_first_mid_band_ok else 0,
                            "token_first_columns": int(main_plan.get("cols", 0)) if token_first_mid_band_ok else 0,
                            "token_first_rows": int(main_plan.get("rows", 0)) if token_first_mid_band_ok else 0,
                            "token_first_codes": main_last5_codes if token_first_mid_band_ok else [],
                            "token_first_unresolved_slots": [r.get("slot_id") for r in unresolved_mid_slot_debug] if token_first_mid_band_ok else [],
                            "physical_profile_comparison": mid_band_profile_comparison,
                            "display_selection_debug": mid_band_selection_debug,
                            "image_normalization_summary": mid_band_normalization_summary,
                            "product_normalization_summary": product_normalization_summary,
                            "layout_debug": mid_band_layout_debug,
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
                        "mid_unresolved_slot_debug": unresolved_mid_slot_debug,
                        "mid_position_fallback_debug": position_fallback_mid_slot_debug,
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
                st.write(
                    {
                        "FULL_PALLET_mid_band_physical_profile_comparison": {
                            "side": mid_band_profile_comparison.get("side"),
                            "profile_name": mid_band_profile_comparison.get("profile_name"),
                            "detected_candidate_count": mid_band_profile_comparison.get("detected_candidate_count"),
                            "detected_group_counts": mid_band_profile_comparison.get("detected_group_counts"),
                            "expected_group_counts": mid_band_profile_comparison.get("expected_group_counts"),
                            "overage_per_group": mid_band_profile_comparison.get("overage_per_group"),
                            "shortage_per_group": mid_band_profile_comparison.get("shortage_per_group"),
                            "candidate_upcs_by_group": mid_band_profile_comparison.get("candidate_upcs_by_group"),
                            "expected_selected_count": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("expected_selected_count"),
                            "row_clusters_detected": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("row_clusters_detected"),
                            "selected_row_col_assignment": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("selected_row_col_assignment"),
                            "selected_group_assignment_per_row": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("selected_group_assignment_per_row"),
                            "selected_upcs_by_group": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("selected_upcs_by_group"),
                            "selected_slot_ids_by_group": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("selected_slot_ids_by_group"),
                            "omitted_upcs_by_group": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("omitted_upcs_by_group"),
                            "omitted_slot_ids_by_group": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("omitted_slot_ids_by_group"),
                            "omit_reasons_by_slot": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("omit_reasons_by_slot"),
                            "shortage_after_row_aware_selection": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("shortage_after_row_aware_selection"),
                            "overage_after_row_aware_selection": dict(
                                mid_band_profile_comparison.get("display_selection_debug", {})
                            ).get("overage_after_row_aware_selection"),
                            "render_layout_hints": mid_band_profile_comparison.get("render_layout_hints"),
                        }
                    }
                )
                st.write(
                    {
                        "FULL_PALLET_mid_band_final_render_selection": {
                            "side": pdata.side_letter,
                            "candidate_count": mid_band_selection_debug.get("candidate_count"),
                            "candidate_order": mid_band_selection_debug.get("candidate_order", []),
                            "selected_count": mid_band_selection_debug.get("selected_count"),
                            "selected_order": mid_band_selection_debug.get("selected_order", []),
                            "omitted_count": mid_band_selection_debug.get("omitted_count"),
                            "omitted_order": mid_band_selection_debug.get("omitted_order", []),
                            "render_assignment": mid_band_selection_debug.get("render_assignment", []),
                            "image_normalization_summary": mid_band_normalization_summary,
                            "product_normalization_summary": product_normalization_summary,
                            "summary": {
                                "expected_selected_count": mid_band_selection_debug.get("expected_selected_count"),
                                "actual_selected_count": mid_band_selection_debug.get("actual_selected_count"),
                                "shortage_or_overage": mid_band_selection_debug.get("shortage_or_overage"),
                                "final_render_order_matches_selected_order": mid_band_selection_debug.get(
                                    "final_render_order_matches_selected_order"
                                ),
                            },
                        }
                    }
                )
                if mid_band_normalization_debug_rows:
                    st.write(
                        {
                            "FULL_PALLET_product_image_normalization_summary": product_normalization_summary,
                        }
                    )
                    st.dataframe(pd.DataFrame(mid_band_normalization_debug_rows), use_container_width=True)
                retained_profile_rows = [
                    {k: v for k, v in {**row, "profile_action": "retain"}.items() if k != "slot"}
                    for group_rows in dict(
                        mid_band_profile_comparison.get("retained_if_expected_max_enforced", {})
                    ).values()
                    for row in group_rows
                ]
                omitted_profile_rows = [
                    {k: v for k, v in {**row, "profile_action": "omit"}.items() if k != "slot"}
                    for group_rows in dict(
                        mid_band_profile_comparison.get("omitted_if_expected_max_enforced", {})
                    ).values()
                    for row in group_rows
                ]
                if retained_profile_rows or omitted_profile_rows:
                    st.dataframe(
                        pd.DataFrame(retained_profile_rows + omitted_profile_rows),
                        use_container_width=True,
                    )
                if mid_band_layout_debug:
                    st.write(
                        {
                            "mid_band_layout_summary": {
                                "side": pdata.side_letter,
                                "mid_band_card_count": mid_band_layout_debug.get("card_count"),
                                "columns_used": mid_band_layout_debug.get("columns_used"),
                                "rows_used": mid_band_layout_debug.get("rows_used"),
                                "card_box_width": mid_band_layout_debug.get("card_w"),
                                "card_box_height": mid_band_layout_debug.get("card_h"),
                                "horizontal_gutter_intra": mid_band_layout_debug.get("h_gutter_intra"),
                                "horizontal_gutter_inter": mid_band_layout_debug.get("h_gutter_inter"),
                                "horizontal_gutter_uniform": mid_band_layout_debug.get("h_gutter_uniform"),
                                "vertical_gutter": mid_band_layout_debug.get("v_gutter"),
                                "mid_band_origin_x": mid_band_layout_debug.get("origin_x"),
                                "mid_band_origin_y": mid_band_layout_debug.get("origin_y"),
                                "layout_mode": mid_band_layout_debug.get("layout_mode", mid_band_layout_debug.get("mode")),
                            }
                        }
                    )
                if bonus_crop_debug_rows:
                    st.write(
                        {
                            "FULL_PALLET_bonus_pixmap_render_debug": {
                                "side": pdata.side_letter,
                                "bonus_slot_count": len(bonus_crop_debug_rows),
                                "pixmap_crop_count": sum(
                                    1 for r in bonus_crop_debug_rows if r.get("render_path_used") == "pixmap_bonus"
                                ),
                                "fallback_count": sum(1 for r in bonus_crop_debug_rows if bool(r.get("fallback_used"))),
                                "suspicious_crop_count": sum(
                                    1 for r in bonus_crop_debug_rows if bool(r.get("suspicious_crop"))
                                ),
                                "blank_or_missing_crop_count": sum(
                                    1 for r in bonus_crop_debug_rows if bool(r.get("blank_or_missing_crop"))
                                ),
                                "clean_crop_count": sum(
                                    1
                                    for r in bonus_crop_debug_rows
                                    if not bool(r.get("suspicious_crop"))
                                    and not bool(r.get("blank_or_missing_crop"))
                                ),
                                "sanitized_crop_count": sum(
                                    1 for r in bonus_crop_debug_rows if bool(r.get("sanitized_crop_used"))
                                ),
                                "edge_contamination_detected_count": sum(
                                    1
                                    for r in bonus_crop_debug_rows
                                    if bool(r.get("neighboring_edge_contamination_detected"))
                                ),
                                "average_source_inset_x": round(
                                    float(np.mean([float(r.get("source_inset_x") or 0.0) for r in bonus_crop_debug_rows])),
                                    2,
                                ),
                                "average_source_inset_y": round(
                                    float(np.mean([float(r.get("source_inset_y") or 0.0) for r in bonus_crop_debug_rows])),
                                    2,
                                ),
                                "contain_fit_used": True,
                                "slot_to_crop_assignment": [
                                    {
                                        "slot_index": r.get("slot_index"),
                                        "slot_id": r.get("slot_id"),
                                        "row": r.get("row"),
                                        "col": r.get("col"),
                                        "upc12": r.get("upc12"),
                                        "render_path_used": r.get("render_path_used"),
                                        "source_crop_bbox": r.get("source_crop_bbox"),
                                        "suspicious_crop": r.get("suspicious_crop"),
                                        "suspicious_reason": r.get("suspicious_reason"),
                                    }
                                    for r in bonus_crop_debug_rows
                                ],
                            }
                        }
                    )
                    st.dataframe(pd.DataFrame(bonus_crop_debug_rows), use_container_width=True)
                if mid_band_layout_assignments:
                    st.dataframe(pd.DataFrame(mid_band_layout_assignments), use_container_width=True)
                if token_first_mid_band_ok:
                    crop_success_count = sum(1 for r in token_first_crop_debug_rows if bool(r.get("crop_success")))
                    crop_failure_count = len(token_first_crop_debug_rows) - crop_success_count
                    image_cell_match_count = sum(
                        1 for r in token_first_crop_debug_rows if str(r.get("image_crop_source", "")) == "image_cell_match"
                    )
                    row_strip_slice_count = sum(
                        1 for r in token_first_crop_debug_rows if bool(r.get("row_strip_source_cell_id"))
                    )
                    row_strip_slice_rejected_count = sum(
                        1 for r in token_first_crop_debug_rows if bool(r.get("rejected_row_strip_slice"))
                    )
                    fallback_crop_count = sum(
                        1
                        for r in token_first_crop_debug_rows
                        if str(r.get("image_crop_source", "")).startswith("fallback_")
                    )
                    suspicious_crop_count = sum(
                        1 for r in token_first_crop_debug_rows if bool(r.get("crop_flagged_suspicious"))
                    )
                    residual_bad_crop_warning_count = sum(
                        1 for r in token_first_crop_debug_rows if bool(r.get("residual_bad_crop_warning"))
                    )
                    pixmap_crop_count = sum(
                        1 for r in token_first_crop_debug_rows if str(r.get("image_crop_source", "")) == "pixmap_middle_band"
                    )
                    pixmap_fallback_count = sum(
                        1
                        for r in token_first_crop_debug_rows
                        if bool(r.get("pixmap_slot_bbox")) and str(r.get("image_crop_source", "")) != "pixmap_middle_band"
                    )
                    empty_crop_count = sum(
                        1 for r in token_first_crop_debug_rows if str(r.get("image_crop_source", "")) == "fallback_none"
                    )
                    tiny_bbox_count = sum(1 for r in token_first_crop_debug_rows if bool(r.get("tiny_bbox")))
                    large_bbox_count = sum(1 for r in token_first_crop_debug_rows if bool(r.get("large_bbox")))
                    widths = [float(r["crop_image_width"]) for r in token_first_crop_debug_rows if r.get("crop_image_width") is not None]
                    heights = [float(r["crop_image_height"]) for r in token_first_crop_debug_rows if r.get("crop_image_height") is not None]
                    avg_crop_width = round(float(np.mean(widths)), 2) if widths else None
                    avg_crop_height = round(float(np.mean(heights)), 2) if heights else None
                    st.write(f"FULL_PALLET token-first mid-band crop debug - Side {pdata.side_letter}")
                    st.write(
                        {
                            "token_first_slot_count": len(token_first_crop_debug_rows),
                            "new_crop_path_count": image_cell_match_count,
                            "row_strip_slice_attempted_count": row_strip_slice_count,
                            "row_strip_slice_rejected_count": row_strip_slice_rejected_count,
                            "fallback_crop_path_count": fallback_crop_count,
                            "empty_crop_path_count": empty_crop_count,
                            "pixmap_middle_band_crop_count": pixmap_crop_count,
                            "pixmap_middle_band_fallback_count": pixmap_fallback_count,
                            "suspicious_crop_count": suspicious_crop_count,
                            "residual_bad_crop_warning_count": residual_bad_crop_warning_count,
                            "image_binding_summary": token_first_detection_debug.get("image_binding_summary", {}),
                            "crop_success_count": crop_success_count,
                            "crop_failure_count": crop_failure_count,
                            "tiny_bbox_count": tiny_bbox_count,
                            "large_bbox_count": large_bbox_count,
                            "avg_crop_width": avg_crop_width,
                            "avg_crop_height": avg_crop_height,
                        }
                    )
                    st.write(
                        {
                            "token_first_detection_summary": {
                                "side": token_first_detection_debug.get("side", pdata.side_letter),
                                "image_page_index": token_first_detection_debug.get("image_page_index", pdata.page_index),
                                "mid_band_region_bbox_used": token_first_detection_debug.get("mid_band_region_bbox_used"),
                                "source_image_cell_count": token_first_detection_debug.get("source_image_cell_count", 0),
                                "ordered_source_cell_ids": token_first_detection_debug.get("ordered_source_cell_ids", []),
                                "strict_side_guardrails": bool(token_first_detection_debug.get("strict_side_guardrails")),
                                "row_strip_fallback_available": bool(token_first_detection_debug.get("row_strip_fallback_available")),
                                "row_strip_source_cell_ids": token_first_detection_debug.get("row_strip_source_cell_ids", []),
                                "pixmap_source_region_bbox": token_first_detection_debug.get("pixmap_source_region_bbox"),
                                "pixmap_grid_source": token_first_detection_debug.get("pixmap_grid_source"),
                                "active_mid_band_render_path": token_first_detection_debug.get("active_mid_band_render_path"),
                                "visual_source": token_first_detection_debug.get("visual_source"),
                                "labels_pdf_visual_crop_used": token_first_detection_debug.get("labels_pdf_visual_crop_used"),
                                "labels_pdf_visual_source_enabled": token_first_detection_debug.get("labels_pdf_visual_source_enabled"),
                                "middle_band_binding_mode": token_first_detection_debug.get("middle_band_binding_mode"),
                                "raw_middle_candidate_count": token_first_detection_debug.get("raw_middle_candidate_count"),
                                "cleaned_middle_candidate_count": token_first_detection_debug.get("cleaned_middle_candidate_count"),
                                "ordered_label_slot_count": token_first_detection_debug.get("ordered_label_slot_count"),
                                "ordered_image_crop_count": token_first_detection_debug.get("ordered_image_crop_count"),
                                "rendered_middle_slot_count": token_first_detection_debug.get("rendered_middle_slot_count"),
                                "missing_middle_slots": token_first_detection_debug.get("missing_middle_slots"),
                                "rejected_middle_candidate_count": token_first_detection_debug.get("rejected_middle_candidate_count"),
                                "rejected_middle_candidates": token_first_detection_debug.get("rejected_middle_candidates", []),
                                "image_crop_sources_by_slot": token_first_detection_debug.get("image_crop_sources_by_slot", []),
                                "blank_image_count": token_first_detection_debug.get("blank_image_count"),
                            },
                            "token_first_mapping_summary": {
                                "side": pdata.side_letter,
                                "resolved_mid_band_slot_count": token_first_detection_debug.get("resolved_mid_band_slot_count", len(token_first_mapping_debug_rows)),
                                "source_image_cell_count": token_first_detection_debug.get("source_image_cell_count", 0),
                                "mapped_cleanly": token_first_detection_debug.get("mapped_clean_count", 0),
                                "using_fallback": token_first_detection_debug.get("mapped_fallback_count", 0),
                                "empty_unrendered": token_first_detection_debug.get("mapped_empty_count", 0),
                                "row_strip_slices_attempted": token_first_detection_debug.get("row_strip_slice_count", 0),
                                "row_strip_slices_rejected": token_first_detection_debug.get("row_strip_slice_rejected_count", 0),
                                "individual_image_cells_used": token_first_detection_debug.get("individual_image_cell_used_count", 0),
                                "rendered_page_detected_cells_used": token_first_detection_debug.get("rendered_page_detected_cell_used_count", 0),
                                "pixmap_crop_count": token_first_detection_debug.get("pixmap_crop_count", 0),
                                "pixmap_fallback_count": token_first_detection_debug.get("pixmap_fallback_count", 0),
                                "pixmap_suspicious_crop_count": token_first_detection_debug.get("pixmap_suspicious_crop_count", 0),
                                "previous_fallback_used": token_first_detection_debug.get("previous_fallback_used_count", 0),
                                "suspicious_crop_count": token_first_detection_debug.get("suspicious_crop_count", 0),
                                "residual_bad_crop_warning_count": token_first_detection_debug.get("residual_bad_crop_warning_count", 0),
                                "image_binding_summary": token_first_detection_debug.get("image_binding_summary", {}),
                            },
                        }
                    )
                    st.dataframe(pd.DataFrame(token_first_crop_debug_rows), use_container_width=True)
                    if token_first_mapping_debug_rows:
                        st.dataframe(pd.DataFrame(token_first_mapping_debug_rows), use_container_width=True)

            c.showPage()

        c.save()
        return buf.getvalue()
    finally:
        images_doc.close()
