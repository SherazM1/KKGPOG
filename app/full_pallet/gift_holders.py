from __future__ import annotations

import io
import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl import load_workbook
from PIL import Image

from app.shared.constants import DIGITS_RE, IMAGE_ANCHOR_ROW_0BASED
from app.shared.models import GiftHolder
from app.shared.text_utils import _coerce_int, _norm_header


def _coerce_item_no(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"\.0$", "", str(v).strip())
    s = DIGITS_RE.sub("", s)
    return s.zfill(6) if s else None



def _img_anchor_col(img) -> Optional[int]:
    try:
        anchor = getattr(img, "anchor", None)
        if anchor is None:
            return None
        _from = getattr(anchor, "_from", None)
        if _from is not None:
            return int(_from.col)
        frm = getattr(anchor, "from_", None)
        if frm is not None:
            return int(frm.col)
    except Exception:
        return None
    return None



def _img_bytes_and_ext(img) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        raw = None
        if hasattr(img, "_data"):
            raw = img._data()
        elif hasattr(img, "ref"):
            ref = img.ref
            if isinstance(ref, bytes):
                raw = ref
            elif hasattr(ref, "read"):
                raw = ref.read()
        if not raw:
            return None, None

        fmt = str(getattr(img, "format", "") or "").strip().lower()
        ext = fmt if fmt in {"png", "jpg", "jpeg", "gif", "bmp"} else "png"

        try:
            im = Image.open(io.BytesIO(raw)).convert("RGBA")
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue(), "png"
        except Exception:
            return raw, ext
    except Exception:
        return None, None



def _extract_ws_images_by_col(ws) -> Dict[int, Tuple[bytes, str]]:
    """
    Map 0-based Excel column index -> first embedded image found in that column.
    """
    out: Dict[int, Tuple[bytes, str]] = {}
    for img in getattr(ws, "_images", []) or []:
        col = _img_anchor_col(img)
        if col is None:
            continue
        img_bytes, img_ext = _img_bytes_and_ext(img)
        if not img_bytes:
            continue
        out.setdefault(col, (img_bytes, img_ext or "png"))
    return out



def _nearest_image_for_col(
    images_by_col: Dict[int, Tuple[bytes, str]],
    target_col: int,
    max_distance: int = 3,
) -> Tuple[Optional[bytes], Optional[str]]:
    best: Optional[Tuple[int, Tuple[bytes, str]]] = None
    for col, payload in images_by_col.items():
        dist = abs(int(col) - int(target_col))
        if dist > max_distance:
            continue
        if best is None or dist < best[0]:
            best = (dist, payload)
    return best[1] if best else (None, None)



def _image_for_col_span(
    images_by_col: Dict[int, Tuple[bytes, str]],
    start_col: int,
    end_col: int,
) -> Tuple[Optional[bytes], Optional[str]]:
    in_span = [
        (col, payload)
        for col, payload in images_by_col.items()
        if start_col <= int(col) <= end_col
    ]
    if in_span:
        # Prefer the left-most image in the slot span; there should normally be one.
        in_span.sort(key=lambda x: x[0])
        return in_span[0][1]
    center_col = (start_col + end_col) // 2
    return _nearest_image_for_col(images_by_col, center_col, max_distance=2)



def _first_nonempty_in_span(ws, row_idx: int, start_col: int, end_col: int) -> Optional[str]:
    if row_idx < 0:
        return None
    for c in range(start_col, end_col + 1):
        val = ws.cell(row=row_idx + 1, column=c + 1).value
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return None



def _first_int_in_span(ws, row_idx: int, start_col: int, end_col: int) -> Optional[int]:
    if row_idx < 0:
        return None
    for c in range(start_col, end_col + 1):
        val = ws.cell(row=row_idx + 1, column=c + 1).value
        if val is None:
            continue
        text = str(val).strip()
        if not text:
            continue
        digits = re.sub(r"[^\d]", "", text)
        if digits:
            try:
                return int(digits)
            except Exception:
                pass
    return None



def _extract_top_holder_slots(
    ws,
    images_by_col: Dict[int, Tuple[bytes, str]],
    qty_row: int,
    item_row: int,
    desc_row: int,
) -> List[dict]:
    """
    Extract holder slots from the top FULL PALLET block:
    POCKET/PEG headers -> image -> qty/item/description below.
    """
    header_row = IMAGE_ANCHOR_ROW_0BASED
    if header_row < 1:
        return []

    max_col = ws.max_column
    slot_starts: List[Tuple[int, str]] = []

    for cidx in range(max_col):
        raw = ws.cell(row=header_row + 1, column=cidx + 1).value
        text = str(raw).strip() if raw is not None else ""
        text_u = text.upper()
        if text_u.startswith("POCKET ") or text_u.startswith("PEG "):
            slot_starts.append((cidx, text))
        elif "MARKETING MESSAGE PANEL" in text_u:
            slot_starts.append((cidx, "__BREAK__"))

    if not slot_starts:
        return []

    slots: List[dict] = []
    real_headers: List[Tuple[int, str, int]] = []
    segment_index = 0
    for c, h in slot_starts:
        if h == "__BREAK__":
            segment_index += 1
            continue
        real_headers.append((c, h, segment_index))

    for i, (start_col, header_text, seg_idx) in enumerate(real_headers):
        next_boundaries = [
            c for c, _ in slot_starts
            if c > start_col
        ]
        end_col = (min(next_boundaries) - 1) if next_boundaries else (max_col - 1)
        if end_col < start_col:
            end_col = start_col

        item_no = _first_nonempty_in_span(ws, item_row, start_col, end_col)
        if not item_no:
            continue

        qty = _first_int_in_span(ws, qty_row, start_col, end_col)
        name = _first_nonempty_in_span(ws, desc_row, start_col, end_col) or ""

        img_bytes, img_ext = _image_for_col_span(images_by_col, start_col, end_col)

        slots.append(
            {
                "header": header_text,
                "slot_order": i,
                "segment_index": seg_idx,
                "start_col": start_col,
                "end_col": end_col,
                "item_no": re.sub(r"[^\d]", "", str(item_no)),
                "qty": qty,
                "name": name,
                "image_bytes": img_bytes,
                "image_ext": img_ext,
            }
        )

    return slots



def _partition_top_slots_into_sides(top_slots: List[dict]) -> Dict[str, List[dict]]:
    """
    Partition top holder slots into ordered side segments using
    MARKETING MESSAGE PANEL-delimited segment indices from extraction.
    """
    by_side: Dict[str, List[dict]] = {s: [] for s in "ABCD"}
    if not top_slots:
        return by_side

    slots_ordered = sorted(
        top_slots,
        key=lambda s: (
            int(s.get("segment_index", 0)),
            int(s.get("start_col", 0)),
            int(s.get("end_col", 0)),
        ),
    )
    segment_ids = sorted({int(s.get("segment_index", 0)) for s in slots_ordered})
    segment_to_side = {seg_id: "ABCD"[idx] for idx, seg_id in enumerate(segment_ids[:4])}

    for slot in slots_ordered:
        seg_id = int(slot.get("segment_index", 0))
        side = segment_to_side.get(seg_id)
        if side is None:
            continue
        by_side[side].append(slot)

    for side in "ABCD":
        by_side[side].sort(
            key=lambda s: (
                int(s.get("start_col", 0)),
                int(s.get("end_col", 0)),
            )
        )
    return by_side



def load_gift_card_holders(gift_bytes: bytes) -> Dict[str, List[GiftHolder]]:
    """Parse POG workbook holder section and return ordered per-side holders."""

    try:
        xls = pd.read_excel(io.BytesIO(gift_bytes), sheet_name=None, header=None)
    except Exception as e:
        raise ValueError(f"Unable to read Gift Card Holders workbook: {e}")

    def find_header_row(df: pd.DataFrame, tokens: List[str], max_rows: int = 120) -> int:
        for i in range(min(len(df), max_rows)):
            row = df.iloc[i].astype(str).fillna("").str.upper().tolist()
            if all(any(tok in c for c in row) for tok in tokens):
                return i
        return -1

    def normalize_headers(raw_headers: List[object]) -> List[str]:
        headers: List[str] = []
        seen: Dict[str, int] = {}
        for v in raw_headers:
            base = _norm_header(v)
            n = seen.get(base, 0) + 1
            seen[base] = n
            headers.append(base if n == 1 else f"{base}_{n}")
        return headers

    def pick_col(headers: List[str], tokens: List[str]) -> Optional[str]:
        up = [h.upper() for h in headers]
        for tok in tokens:
            for i, h in enumerate(up):
                if tok in h:
                    return headers[i]
        return None

    def parse_side(value: object, current_side: str) -> str:
        text = str(value or "").upper()
        m = re.search(r"SIDE\s*([A-D])", text)
        if m:
            return m.group(1)
        m = re.fullmatch(r"\s*([A-D])\s*", text)
        if m:
            return m.group(1)
        return current_side

    full_pallet_sheet = next(
        (name for name in xls.keys() if "FULL" in name.upper() and "PALLET" in name.upper()),
        None,
    )
    if full_pallet_sheet is None:
        raise ValueError("FULL PALLET sheet/table missing in workbook.")

    try:
        wb = load_workbook(io.BytesIO(gift_bytes), data_only=True)
    except Exception as e:
        raise ValueError(f"Unable to open Gift Card Holders workbook images: {e}")

    full_pallet_ws = None
    for ws in wb.worksheets:
        title_u = str(ws.title or "").upper()
        if "FULL" in title_u and "PALLET" in title_u:
            full_pallet_ws = ws
            break

    if full_pallet_ws is None:
        raise ValueError("Could not find FULL PALLET sheet in workbook.")

    images_by_col = _extract_ws_images_by_col(full_pallet_ws)

    desc_map: Dict[str, str] = {}
    for _, sheet_df in xls.items():
        hdr = find_header_row(sheet_df, ["ITEM", "DESCRIPTION"])
        if hdr < 0:
            continue
        cols = normalize_headers(sheet_df.iloc[hdr].tolist())
        data = sheet_df.iloc[hdr + 1 :].copy()
        data.columns = cols
        item_lookup_col = pick_col(cols, ["ITEM", "ITEM_#", "SKU"])
        desc_lookup_col = pick_col(cols, ["DESCRIPTION", "DESC", "NAME"])
        if not item_lookup_col or not desc_lookup_col:
            continue
        for _, row in data.iterrows():
            item_no = _coerce_item_no(row.get(item_lookup_col))
            if not item_no:
                continue
            desc = str(row.get(desc_lookup_col, "") or "").strip()
            if desc:
                desc_map[item_no] = desc

    full_df_raw = xls[full_pallet_sheet].copy()

    item_row = -1
    qty_row = -1
    desc_row = -1
    for i in range(len(full_df_raw)):
        row_vals = [str(v or "").strip().upper() for v in full_df_raw.iloc[i].tolist()]
        if item_row < 0 and any("ITEM #" in v or v == "ITEM" for v in row_vals):
            numeric_items = sum(1 for v in row_vals if _coerce_item_no(v))
            if numeric_items >= 4:
                item_row = i
        if item_row >= 0 and i <= item_row and qty_row < 0:
            if any(v == "QTY" or v.startswith("QTY") for v in row_vals):
                qty_row = i
        if item_row >= 0 and i >= item_row and desc_row < 0:
            if any("DESCRIPTION" in v for v in row_vals):
                desc_row = i
        if item_row >= 0 and qty_row >= 0 and desc_row >= 0:
            break

    top_slots = _extract_top_holder_slots(
        full_pallet_ws,
        images_by_col=images_by_col,
        qty_row=qty_row,
        item_row=item_row,
        desc_row=desc_row,
    )

    holders: Dict[str, List[GiftHolder]] = {s: [] for s in "ABCD"}
    if top_slots:
        slots_by_side = _partition_top_slots_into_sides(top_slots)
        for side in "ABCD":
            for slot in slots_by_side[side]:
                item_no = str(slot["item_no"]).strip()
                if not item_no:
                    continue
                holders[side].append(
                    GiftHolder(
                        side=side,
                        item_no=item_no,
                        name=str(slot["name"] or "").strip(),
                        qty=slot["qty"],
                        image_bytes=slot["image_bytes"],
                        image_ext=slot["image_ext"],
                        slot_label=str(slot["header"] or "").strip() or None,
                        slot_order=int(slot["slot_order"]),
                        slot_start_col=int(slot["start_col"]),
                        slot_end_col=int(slot["end_col"]),
                    )
                )
        return holders

    header_row = find_header_row(full_df_raw, ["ITEM", "QTY"])
    if header_row < 0:
        raise ValueError("FULL PALLET holder table missing ITEM/QTY columns.")

    headers = normalize_headers(full_df_raw.iloc[header_row].tolist())
    full_df = full_df_raw.iloc[header_row + 1 :].copy()
    full_df.columns = headers

    item_col = pick_col(headers, ["ITEM", "ITEM_#", "SKU"])
    qty_col = pick_col(headers, ["QTY", "QUANTITY"])
    side_col = pick_col(headers, ["SIDE"])
    if item_col is None or qty_col is None:
        raise ValueError("FULL PALLET holder table missing ITEM/QTY columns.")

    current_side = "A"
    for _, row in full_df.iterrows():
        side_source = row.get(side_col) if side_col else " "
        current_side = parse_side(side_source, current_side)

        item_no = _coerce_item_no(row.get(item_col))
        if not item_no:
            row_text = " ".join(str(v or "") for v in row.tolist())
            current_side = parse_side(row_text, current_side)
            continue

        qty = _coerce_int(row.get(qty_col))
        name = desc_map.get(item_no) or "(missing description)"
        side = current_side if current_side in "ABCD" else "A"
        holders.setdefault(side, []).append(
                GiftHolder(
                    side=side,
                    item_no=item_no,
                    name=name,
                    qty=qty,
                    image_bytes=None,
                    image_ext=None,
                )
            )

    if not any(holders.values()):
        raise ValueError("No holder Item # rows found in FULL PALLET table.")

    non_empty = [s for s in "ABCD" if holders.get(s)]
    if non_empty == ["A"]:
        base_list = holders["A"]
        for s in "BCD":
            holders[s] = [
                GiftHolder(
                    side=s,
                    item_no=h.item_no,
                    name=h.name,
                    qty=h.qty,
                    image_bytes=h.image_bytes,
                    image_ext=h.image_ext,
                    slot_label=h.slot_label,
                    slot_order=h.slot_order,
                    slot_start_col=h.slot_start_col,
                    slot_end_col=h.slot_end_col,
                )
                for h in base_list
            ]
    return holders
