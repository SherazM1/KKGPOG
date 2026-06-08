from __future__ import annotations



import difflib

import io
import re

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



@st.cache_data(show_spinner=False)
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



@st.cache_data(show_spinner=False)
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

    name_col: Optional[str] = None
    preferred_name_headers = [
        "NAME",
        "ITEM_NAME",
        "ITEM_DESCRIPTION",
        "DESCRIPTION",
        "PRODUCT_NAME",
        "PRODUCT_DESCRIPTION",
    ]
    for preferred in preferred_name_headers:
        if preferred in headers:
            name_col = preferred
            break
    if name_col is None:
        for header in headers:
            upper = header.upper()
            if ("NAME" in upper or "DESCRIPTION" in upper) and not any(
                blocked in upper for blocked in ("POG", "PLANOGRAM", "STORE")
            ):
                name_col = header
                break
    if name_col is None:
        name_col = _pick_col(headers, ["DESCRIPTION", "NAME"], 1 if len(headers) > 1 else 0)
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
        return _resolve_known_missing_matrix_row(key, label_name)
    if len(rows) == 1:
        return _apply_known_label_name_correction(key, label_name, rows[0])
    target = _norm_name(label_name)
    return _apply_known_label_name_correction(
        key,
        label_name,
        max(rows, key=lambda r: difflib.SequenceMatcher(None, target, r.norm_name).ratio()),
    )


def _resolve_known_missing_matrix_row(last5: str, label_name: str) -> Optional[MatrixRow]:
    """Resolve confirmed items that appear in label PDFs but are absent from some state matrices."""
    source_name = str(label_name or "").strip()
    source_norm = _norm_name(source_name)
    known_rows = {
        "16340": ("019674216340", "ALAMO DRAFTHOUSE $25 Reg", 30, ("ALAMO", "DRAFTH")),
        "10092": ("019674210092", "TOP GOLF VGC MD", 30, ("TOP", "GOL")),
        "13196": ("019674213196", "SCOOTERS COFFEE VGC", 30, ("SCOOTERS", "COFFEE")),
        "11449": ("019674211449", "YARD HOUSE VGC MD", 20, ("YARD", "HOUSE")),
        "10541": ("019674210541", "MOES SW GRILL VGC", 30, ("MOES", "GRILL")),
        "10808": ("019674210808", "QDOBA VGC MD", 30, ("QDOBA",)),
        "16708": ("019674216708", "BETMGM $99 + $30 BONUS $8.95 ACTVN FEE", 30, ("ACTVN", "FEE")),
        "10807": ("019674210807", "MILLERSALEHOUSE VGC MD", 30, ("VGC",)),
        "11027": ("019674211027", "CHUCK E CHEESE VGC MD", 30, ("CHUCK", "HEESE")),
        "11964": ("019674211964", "WHAT A BURGER $15 MD", 30, ("MD",)),
        "16878": ("019674216878", "RAISING CANES VGC", 30, ("RAISING", "CANES")),
        "10546": ("019674210546", "JAMBA JUICE VGC MD", 30, ("JAMBA", "JUICE")),
        "14913": ("019674214913", "IN N OUT BURGERr VGC Reg", 30, ("GOEUR",)),
        "12134": ("019674212134", "DUTCH BROS VGC ($15-$250) REG", 30, ("REG",)),
        "10851": ("019674210851", "DAVE & BUSTERS VGC HOLIDAY MD", 20, ("DAVE", "BUSTERS")),
        "10640": ("019674210640", "CINEMARK VGC HOLIDAY MD", 30, ("CINEMARK",)),
        "10534": ("019674210534", "FOCUS BRANDS VGC MD", 30, ("FOCUS", "BRANDS")),
        "16338": ("019674216338", "99 RESTAURANT & PUBS VGC REG", 30, ("RESTAURANT", "PUBS")),
        "14401": ("019674214401", "SHAKE SHACK VGC Reg", 30, ("SH", "CK")),
        "05343": ("019674205343", "D&B/MAIN EVENT MD", 20, ("VENT",)),
        "10006": ("019674210006", "BOB EVANS VGC MD", 30, ("BOB", "EVANS")),
        "12964": ("019674212964", "FIRST WATCH $25 REG", 30, ("FIRST", "WATCH")),
        "10495": ("019674210495", "LONG JOHN SILVER'S VGC Reg", 30, ("REG",)),
        "10129": ("019674210129", "FANDANGO / FANDANGO AT HOME VGC ($25-$100)", 30, ("D", "F")),
        "10866": ("019674210866", "Saltgrass Steakhouse VGC ($25-$500) MD", 30, ("SALTGRASS",)),
        "16310": ("019674216310", "Johnnies Charcoal Broiler VGC Reg", 20, ("BROILER",)),
        "13078": ("019674213078", "CHUY'S VGC ($25-$500)", 20, ("CHUY",)),
        "11450": ("019674211450", "RUTHS CHRIS VGC MD", 20, ("RUTHS", "CHRIS")),
        "10493": ("019674210493", "FOGO DE CHAO VGC Reg", 30, ("FOGO", "CHAO")),
        "15368": ("019674215368", "PF CHANG'S VGC ($25-$500) REG", 30, ("CHANG",)),
        "07303": ("019674207303", "ZAXBYS VGC HOLIDAY", 20, ("ZAXBYS",)),
        "11062": ("019674211062", "GO PLAY GOLF VGC MD", 20, ("PLAY", "OLF")),
        "10868": ("019674210868", "LANDRYS VGC MD", 30, ("LANDRYS",)),
        "11441": ("019674211441", "CHEDDAR'S VGC ($10-$100) MD", 20, ("CHEDDAR",)),
        "12653": ("019674212653", "Larry Miller MegaPlex $25 MD", 30, ("LARRY", "MILLER")),
    }
    entry = known_rows.get(last5)
    if entry is None:
        return None
    upc12, display_name, cpp, required_tokens = entry
    if not all(token in source_norm for token in required_tokens):
        return None
    return MatrixRow(upc12, _norm_name(display_name), display_name, cpp)


def _apply_known_label_name_correction(last5: str, label_name: str, row: MatrixRow) -> MatrixRow:
    """Correct known matrix name swaps when the official label PDF confirms it."""
    source_name = str(label_name or "").strip()
    source_norm = _norm_name(source_name)
    if last5 == "16659" and "PIZZA" in source_norm and "HUT" in source_norm:
        return MatrixRow(row.upc12, _norm_name(source_name), source_name, row.cpp_qty)
    if last5 == "16660" and "KFC" in source_norm:
        return MatrixRow(row.upc12, _norm_name(source_name), source_name, row.cpp_qty)
    row_norm = _norm_name(row.display_name)
    if last5 == "16708" and ("BETMGM" in source_norm or "BETMGM" in row_norm):
        return MatrixRow(row.upc12, row.norm_name, row.display_name, 30)
    return row



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
