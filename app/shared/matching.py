from __future__ import annotations



import difflib

import io

from typing import Dict, List, Optional



import pandas as pd

import streamlit as st



from app.shared.models import MatrixRow

from app.shared.text_utils import (

    _coerce_int,

    _coerce_upc12,

    _find_header_row,

    _norm_header,

    _norm_name,

    _pick_col,

    _pick_col_optional,

    _to_last5,

)



def load_matrix_index(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    df_raw = pd.read_excel(io.BytesIO(matrix_bytes), header=None)

    hrow = _find_header_row(df_raw)

    headers: List[str] = []
    seen: Dict[str, int] = {}
    for v in df_raw.iloc[hrow].tolist():
        base = _norm_header(v)
        n = seen.get(base, 0)
        seen[base] = n + 1
        headers.append(base if n == 0 else f"{base}_{n+1}")

    df = df_raw.iloc[hrow + 1 :].copy()
    df.columns = headers

    upc_col = _pick_col(headers, ["UPC"], 0)
    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"], 1 if len(headers) > 1 else 0)
    cpp_col = _pick_col_optional(headers, ["CPP"])

    df["__upc12"] = df[upc_col].map(_coerce_upc12)
    df["__name"] = df[name_col].astype(str).fillna("")

    if cpp_col and cpp_col in df.columns:
        df["__cpp"] = df[cpp_col].map(_coerce_int)
    else:
        df["__cpp"] = None

    df = df[df["__upc12"].notna()].copy()
    df["__last5"] = df["__upc12"].str[-5:]
    df["__norm"] = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    for _, r in df.iterrows():
        last5 = str(r["__last5"])
        cpp_val = r.get("__cpp")
        cpp = None if pd.isna(cpp_val) else int(cpp_val) if cpp_val is not None else None
        display_name = str(r["__name"]).strip()
        idx.setdefault(last5, []).append(
            MatrixRow(
                upc12=str(r["__upc12"]),
                norm_name=str(r["__norm"]),
                display_name=display_name,
                cpp_qty=cpp,
            )
        )
    return idx



def load_full_pallet_matrix_index(matrix_bytes: bytes) -> Dict[str, List[MatrixRow]]:
    """Full-pallet matrix index.

    Index key  = last 5 digits of the 11-digit UPC column (column header "UPC").
    Stored upc12 = 11-digit UPC zero-padded to 12 chars.

    The labels PDF encodes each product as the last 5 digits of its 11-digit UPC,
    so indexing on that column is required for correct resolution.  The zero-padded
    11-digit value matches what the reference planogram displays as the product UPC.
    """
    df_raw = pd.read_excel(io.BytesIO(matrix_bytes), header=None)
    hrow = _find_header_row(df_raw)

    headers: List[str] = []
    seen: Dict[str, int] = {}
    for v in df_raw.iloc[hrow].tolist():
        base = _norm_header(v)
        n = seen.get(base, 0)
        seen[base] = n + 1
        headers.append(base if n == 0 else f"{base}_{n+1}")

    df = df_raw.iloc[hrow + 1 :].copy()
    df.columns = headers

    name_col = _pick_col(headers, ["NAME", "DESCRIPTION"], 1 if len(headers) > 1 else 0)
    cpp_col = _pick_col_optional(headers, ["CPP"])

    # ── FIX A: use ONLY the 11-digit UPC column as the index key ─────────────
    # The labels PDF encodes items as the last 5 digits of the 11-digit UPC.
    # Using the 12-digit UPC column would give a different last-5 (check digit
    # appended) and break all lookups.  Prefer the exact header "UPC"; fall back
    # to the first column whose normalised name contains "UPC".
    upc11_col: Optional[str] = None
    for h in headers:
        if h.upper() == "UPC":
            upc11_col = h
            break
    if upc11_col is None:
        upc11_col = _pick_col(headers, ["UPC"], 0)

    df["__name"] = df[name_col].astype(str).fillna("")
    if cpp_col and cpp_col in df.columns:
        df["__cpp"] = df[cpp_col].map(_coerce_int)
    else:
        df["__cpp"] = None
    df["__norm"] = df["__name"].map(_norm_name)

    idx: Dict[str, List[MatrixRow]] = {}
    seen_pairs: set = set()
    for _, r in df.iterrows():
        display_name = str(r["__name"]).strip()
        cpp_val = r.get("__cpp")
        cpp = None if pd.isna(cpp_val) else int(cpp_val) if cpp_val is not None else None

        raw_upc11 = r.get(upc11_col)
        if raw_upc11 is None:
            continue
        upc11_str = re.sub(r"\.0$", "", str(raw_upc11).strip())
        digits11 = re.sub(r"[^0-9]", "", upc11_str)
        if len(digits11) < 5:
            continue
        last5 = digits11[-5:]
        # Store as 11-digit UPC zero-padded to 12 chars — this is the format
        # shown in the reference planogram output (e.g. "019674209969").
        upc12_display = digits11.zfill(12)

        pair = (last5, upc12_display)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        idx.setdefault(last5, []).append(
            MatrixRow(
                upc12=upc12_display,
                norm_name=str(r["__norm"]),
                display_name=display_name,
                cpp_qty=cpp,
            )
        )
    return idx



def resolve_full_pallet(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[MatrixRow]:
    key = _to_last5(last5)
    rows = idx.get(key, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    target = _norm_name(label_name)
    return max(rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio())



def _resolve(last5: str, label_name: str, idx: Dict[str, List[MatrixRow]]) -> Optional[MatrixRow]:
    rows = idx.get(last5, [])
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    target = _norm_name(label_name)
    return max(
        rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio()
    )
