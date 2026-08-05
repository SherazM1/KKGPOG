from __future__ import annotations

import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.sams_club.price_strip_models import SamsPriceStripBuildResult, SamsPriceStripRow, SamsPriceStripSegment
from app.sams_club.holiday_price_strips import is_sams_holiday_template, map_holiday_rows_to_strips

_PRICE_STRIP_SHEET = "Price Strip Data"
_CONTENT_WARNING_FIELDS: tuple[tuple[str, str], ...] = (
    ("retail", "Retail"),
    ("brand", "Brand"),
    ("item_number", "Item Number"),
    ("desc_1", "Desc 1"),
    ("desc_2", "Desc 2"),
)
_REQUIRED_GROUP_FIELDS: tuple[str, ...] = ("pog", "side", "row", "column")
_EXPECTED_PRODUCTION_FIELDS: tuple[str, ...] = (
    "pog",
    "item_number",
    "brand",
    "desc_1",
    "desc_2",
    "retail",
    "side",
    "row",
    "column",
    "length",
    "data_on_bottom_left",
)
_REQUIRED_HEADER_LABELS: dict[str, str] = {
    "pog": "POG",
    "item_number": "Item Number",
    "brand": "Brand",
    "desc_1": "Desc 1",
    "desc_2": "Desc 2",
    "retail": "Retail",
    "side": "Side",
    "row": "Row",
    "column": "Column",
    "length": "Length",
    "data_on_bottom_left": "Data on bottom left",
}
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


@dataclass(frozen=True)
class _WorksheetCandidate:
    sheet_name: str
    df: pd.DataFrame
    mapping: dict[str, str]
    missing_fields: tuple[str, ...]
    headers_found: tuple[str, ...]
    populated_rows: int
    exact_name_match: bool
    fuzzy_name_match: bool

    @property
    def header_match_count(self) -> int:
        return len(_EXPECTED_PRODUCTION_FIELDS) - len(self.missing_fields)

    @property
    def has_complete_schema(self) -> bool:
        return len(self.missing_fields) == 0

    @property
    def is_non_empty(self) -> bool:
        return self.populated_rows > 0 or len(self.headers_found) > 0


class _PriceStripWorksheetNotFound(ValueError):
    def __init__(self, message: str, debug: dict[str, Any]) -> None:
        super().__init__(message)
        self.debug = debug


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_sheet_name(value: str) -> str:
    cleaned = _collapse_spaces(str(value).lower())
    cleaned = re.sub(r"[^\w\s]+", " ", cleaned)
    cleaned = _collapse_spaces(cleaned)
    return cleaned


def _compact_sheet_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_sheet_name(value))


def _has_price_strip_data_words(sheet_name: str) -> bool:
    compact = _compact_sheet_name(sheet_name)
    return all(word in compact for word in ("price", "strip", "data"))


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


def _open_excel_file(source_file: Any) -> pd.ExcelFile:
    if isinstance(source_file, (str, os.PathLike)):
        return pd.ExcelFile(str(source_file))

    payload, _ = _coerce_uploaded_bytes(source_file)
    return pd.ExcelFile(io.BytesIO(payload))


def _populated_row_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    populated_cells = df.map(lambda value: not pd.isna(value) and str(value).strip() != "")
    return int(populated_cells.any(axis=1).sum())


def _missing_expected_fields(mapping: dict[str, str]) -> tuple[str, ...]:
    return tuple(field for field in _EXPECTED_PRODUCTION_FIELDS if field not in mapping)


def _candidate_rank(candidate: _WorksheetCandidate) -> tuple[int, int, int, int]:
    return (
        candidate.header_match_count,
        candidate.populated_rows,
        int(candidate.fuzzy_name_match),
        int(candidate.exact_name_match),
    )


def _best_candidate(candidates: list[_WorksheetCandidate]) -> _WorksheetCandidate:
    return max(candidates, key=_candidate_rank)


def _read_price_strip_sheet(source_file: Any) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    workbook = _open_excel_file(source_file)
    try:
        candidates: list[_WorksheetCandidate] = []
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
            mapping = _build_mapping([str(c) for c in df.columns])
            candidates.append(
                _WorksheetCandidate(
                    sheet_name=str(sheet_name),
                    df=df,
                    mapping=mapping,
                    missing_fields=_missing_expected_fields(mapping),
                    headers_found=tuple(str(c) for c in df.columns),
                    populated_rows=_populated_row_count(df),
                    exact_name_match=str(sheet_name) == _PRICE_STRIP_SHEET,
                    fuzzy_name_match=_has_price_strip_data_words(str(sheet_name)),
                )
            )

        complete = [candidate for candidate in candidates if candidate.has_complete_schema]
        selection_reason = ""
        selected: _WorksheetCandidate | None = None

        exact_matches = [candidate for candidate in complete if candidate.exact_name_match]
        if exact_matches:
            selected = _best_candidate(exact_matches)
            selection_reason = "exact worksheet-name match"

        if selected is None:
            fuzzy_matches = [candidate for candidate in complete if candidate.fuzzy_name_match]
            if fuzzy_matches:
                selected = _best_candidate(fuzzy_matches)
                selection_reason = "normalized worksheet-name match"

        if selected is None and complete:
            selected = _best_candidate(complete)
            selection_reason = "header schema match"

        if selected is None:
            non_empty = [candidate for candidate in candidates if candidate.is_non_empty]
            if len(non_empty) == 1 and non_empty[0].has_complete_schema:
                selected = non_empty[0]
                selection_reason = "single non-empty worksheet"

        if selected is not None:
            return (
                selected.df,
                selected.sheet_name,
                {
                    "sheet_name": selected.sheet_name,
                    "sheet_selection_reason": selection_reason,
                    "sheet_names_found": [candidate.sheet_name for candidate in candidates],
                    "column_mapping": selected.mapping,
                },
            )

        closest = _best_candidate(candidates) if candidates else None
        required_headers = [_REQUIRED_HEADER_LABELS[field] for field in _EXPECTED_PRODUCTION_FIELDS]
        missing_headers = (
            [_REQUIRED_HEADER_LABELS[field] for field in closest.missing_fields]
            if closest is not None
            else required_headers
        )
        headers_found = list(closest.headers_found) if closest is not None else []
        closest_name = closest.sheet_name if closest is not None else "(none)"
        sheets_found = [candidate.sheet_name for candidate in candidates]
        message = (
            "Could not locate a usable price-strip worksheet. "
            f"Sheets found: {', '.join(sheets_found) if sheets_found else '(none)'}. "
            f"Required headers: {', '.join(required_headers)}. "
            f"Closest match: {closest_name}. "
            f"Headers found on closest match: {', '.join(headers_found) if headers_found else '(none)'}. "
            f"Missing required columns: {', '.join(missing_headers)}."
        )
        raise _PriceStripWorksheetNotFound(
            message,
            {
                "sheet_name": None,
                "sheet_names_found": sheets_found,
                "required_headers": required_headers,
                "closest_match": closest_name,
                "headers_found": headers_found,
                "missing_required_columns": missing_headers,
            },
        )
    finally:
        close = getattr(workbook, "close", None)
        if callable(close):
            close()


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


def build_sams_price_strip_rows(
    source_file: Any,
    template_name: str | None = None,
) -> SamsPriceStripBuildResult:
    warnings: list[str] = []
    errors: list[str] = []

    try:
        df, selected_sheet_name, sheet_debug = _read_price_strip_sheet(source_file)
    except _PriceStripWorksheetNotFound as exc:
        return SamsPriceStripBuildResult(
            errors=[str(exc)],
            debug=exc.debug,
        )
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
                f"Missing required grouping columns in '{selected_sheet_name}': "
                + ", ".join(missing_group_columns)
                + ". Required: POG, Side, Row, Column."
            ],
            debug={**sheet_debug, "column_mapping": mapping},
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
            is_empty=False,
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

    active_template = template_name or ""
    if is_sams_holiday_template(active_template):
        strip_rows, holiday_warnings = map_holiday_rows_to_strips(strip_rows)
        warnings.extend(holiday_warnings)
        segments_per_group = {
            f"{row.pog} | Side {row.side} | Row {row.row}": len(row.segments)
            for row in strip_rows
        }

    debug = {
        "sheet_name": selected_sheet_name,
        "column_mapping": mapping,
        "sheet_selection_reason": sheet_debug.get("sheet_selection_reason"),
        "sheet_names_found": sheet_debug.get("sheet_names_found", []),
        "template_name": active_template,
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
