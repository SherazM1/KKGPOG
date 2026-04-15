from __future__ import annotations

import io
import os
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from app.sams_club.price_strip_models import SamsPriceStripBuildResult, SamsPriceStripRow, SamsPriceStripSegment

_PRICE_STRIP_SHEET = "Price Strip Data"
_CONTENT_WARNING_FIELDS: tuple[tuple[str, str], ...] = (
    ("retail", "Retail"),
    ("brand", "Brand"),
    ("item_number", "Item Number"),
    ("desc_1", "Desc 1"),
    ("desc_2", "Desc 2"),
)
_REQUIRED_GROUP_FIELDS: tuple[str, ...] = ("pog", "side", "row", "column")
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pog": ("POG", "pog"),
    "item_number": ("Item Number", "item_number", "item"),
    "brand": ("Brand", "brand"),
    "desc_1": ("Desc 1", "desc_1", "desc1"),
    "desc_2": ("Desc 2", "desc_2", "desc2"),
    "retail": ("Retail", "retail", "price"),
    "side": ("Side", "side"),
    "row": ("Row", "row"),
    "column": ("Column", "column", "col"),
    "length": ("Length", "length"),
    "data_on_bottom_left": ("Data on bottom left", "data on bottom left", "bottom left"),
}


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _canonical_header(value: str) -> str:
    cleaned = _collapse_spaces(str(value).strip().lower())
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = _collapse_spaces(cleaned)
    return cleaned.replace(" ", "")


def _coerce_uploaded_bytes(source_file: Any) -> tuple[bytes, str]:
    if isinstance(source_file, (bytes, bytearray)):
        return bytes(source_file), "uploaded.xlsx"

    if hasattr(source_file, "getvalue"):
        filename = getattr(source_file, "name", "uploaded.xlsx")
        return bytes(source_file.getvalue()), str(filename)

    if hasattr(source_file, "read"):
        filename = getattr(source_file, "name", "uploaded.xlsx")
        data = source_file.read()
        if hasattr(source_file, "seek"):
            try:
                source_file.seek(0)
            except Exception:
                pass
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Uploaded workbook stream did not return bytes.")
        return bytes(data), str(filename)

    raise TypeError("Unsupported workbook source type. Provide a file path, bytes, or uploaded file object.")


def _read_price_strip_sheet(source_file: Any) -> pd.DataFrame:
    if isinstance(source_file, (str, os.PathLike)):
        return pd.read_excel(str(source_file), sheet_name=_PRICE_STRIP_SHEET, dtype=object)

    payload, _ = _coerce_uploaded_bytes(source_file)
    return pd.read_excel(io.BytesIO(payload), sheet_name=_PRICE_STRIP_SHEET, dtype=object)


def _build_mapping(column_names: list[str]) -> dict[str, str]:
    canonical_source: dict[str, str] = {}
    for col in column_names:
        key = _canonical_header(col)
        if key and key not in canonical_source:
            canonical_source[key] = col

    mapping: dict[str, str] = {}
    for logical_key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = _canonical_header(alias)
            if alias_key in canonical_source:
                mapping[logical_key] = canonical_source[alias_key]
                break
    return mapping


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def build_sams_price_strip_rows(source_file: Any) -> SamsPriceStripBuildResult:
    warnings: list[str] = []
    errors: list[str] = []

    try:
        df = _read_price_strip_sheet(source_file)
    except ValueError as exc:
        return SamsPriceStripBuildResult(
            errors=[f"Unable to read '{_PRICE_STRIP_SHEET}' sheet: {exc}"],
            debug={"sheet_name": _PRICE_STRIP_SHEET},
        )
    except Exception as exc:
        return SamsPriceStripBuildResult(
            errors=[f"Price strip workbook read failed: {exc}"],
            debug={"sheet_name": _PRICE_STRIP_SHEET},
        )

    mapping = _build_mapping([str(c) for c in df.columns])
    missing_group_columns = [field for field in _REQUIRED_GROUP_FIELDS if field not in mapping]
    if missing_group_columns:
        return SamsPriceStripBuildResult(
            errors=[
                "Missing required grouping columns in 'Price Strip Data': "
                + ", ".join(missing_group_columns)
                + ". Required: POG, Side, Row, Column."
            ],
            debug={"sheet_name": _PRICE_STRIP_SHEET, "column_mapping": mapping},
        )

    grouped_segments: dict[tuple[str, int, int], list[SamsPriceStripSegment]] = defaultdict(list)
    extracted_count = len(df.index)
    included_count = 0
    skipped_count = 0

    for idx, row_data in enumerate(df.to_dict(orient="records")):
        pog = _as_text(row_data.get(mapping.get("pog", "")))
        side = _as_int(row_data.get(mapping.get("side", "")))
        row_value = _as_int(row_data.get(mapping.get("row", "")))
        column = _as_int(row_data.get(mapping.get("column", "")))

        if pog == "" or side is None or row_value is None or column is None or side <= 0 or row_value <= 0 or column <= 0:
            skipped_count += 1
            warnings.append(
                f"Record {idx} skipped: invalid grouping fields (POG={pog or '(blank)'}, Side={side}, Row={row_value}, Column={column})."
            )
            continue

        segment_warnings: list[str] = []
        values = {
            "item_number": _as_text(row_data.get(mapping.get("item_number", ""))),
            "brand": _as_text(row_data.get(mapping.get("brand", ""))),
            "desc_1": _as_text(row_data.get(mapping.get("desc_1", ""))),
            "desc_2": _as_text(row_data.get(mapping.get("desc_2", ""))),
            "retail": _as_text(row_data.get(mapping.get("retail", ""))),
            "length": _as_text(row_data.get(mapping.get("length", ""))),
            "data_on_bottom_left": _as_text(row_data.get(mapping.get("data_on_bottom_left", ""))),
        }

        for logical_key, label in _CONTENT_WARNING_FIELDS:
            if values[logical_key] == "":
                message = f"Record {idx}: missing {label} (POG={pog}, Side={side}, Row={row_value}, Column={column})."
                warnings.append(message)
                segment_warnings.append(message)

        segment = SamsPriceStripSegment(
            pog=pog,
            side=side,
            row=row_value,
            column=column,
            item_number=values["item_number"],
            brand=values["brand"],
            desc_1=values["desc_1"],
            desc_2=values["desc_2"],
            retail=values["retail"],
            length=values["length"],
            data_on_bottom_left=values["data_on_bottom_left"],
            warnings=segment_warnings,
        )
        grouped_segments[(pog, side, row_value)].append(segment)
        included_count += 1

    strip_rows: list[SamsPriceStripRow] = []
    segments_per_group: dict[str, int] = {}
    for key in sorted(grouped_segments.keys(), key=lambda x: (x[0], x[1], x[2])):
        pog, side, row_value = key
        segments = sorted(grouped_segments[key], key=lambda seg: seg.column)
        footer_candidates = [seg.data_on_bottom_left for seg in segments if seg.data_on_bottom_left.strip()]
        footer_text = footer_candidates[0] if footer_candidates else f"Side: {side}, Row: {row_value} - POG: {pog}"
        group_warnings: list[str] = []

        unique_footer_values = sorted({value.strip() for value in footer_candidates if value.strip()})
        if len(unique_footer_values) > 1:
            msg = (
                f"Group POG={pog}, Side={side}, Row={row_value}: multiple 'Data on bottom left' values found; using first by column."
            )
            warnings.append(msg)
            group_warnings.append(msg)

        strip_rows.append(
            SamsPriceStripRow(
                pog=pog,
                side=side,
                row=row_value,
                segments=segments,
                footer_text=footer_text,
                warnings=group_warnings,
            )
        )
        segments_per_group[f"{pog} | Side {side} | Row {row_value}"] = len(segments)

    debug = {
        "sheet_name": _PRICE_STRIP_SHEET,
        "column_mapping": mapping,
        "detected_strip_groups": [{"pog": row.pog, "side": row.side, "row": row.row} for row in strip_rows],
        "strip_group_count": len(strip_rows),
        "segments_per_strip_row": segments_per_group,
        "warnings": warnings.copy(),
    }
    return SamsPriceStripBuildResult(
        strip_rows=strip_rows,
        extracted_record_count=extracted_count,
        included_segment_count=included_count,
        skipped_segment_count=skipped_count,
        warnings=warnings,
        errors=errors,
        debug=debug,
    )
