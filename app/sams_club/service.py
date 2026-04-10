from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.sams_club.extract_access import extract_master_pog_records
from app.sams_club.models import SamsPlanogram, SamsRow, SamsSidePage, SamsSlot
from app.sams_club.validate import (
    side_column_limit,
    validate_column,
    validate_row,
    validate_side,
    validate_slot_key_uniqueness,
)

_MISSING_FIELD_WARNINGS: tuple[tuple[str, str], ...] = (
    ("retail", "missing retail"),
    ("upc", "missing upc"),
    ("cpp", "missing cpp"),
    ("file_path", "missing file_path"),
    ("description", "missing description"),
)


@dataclass
class SamsBuildResult:
    """Build result payload for the Sam's Club structure pipeline."""

    planogram: SamsPlanogram
    extracted_record_count: int
    normalized_record_count: int
    detected_pogs: list[str] = field(default_factory=list)
    selected_pog: str = ""
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_text(value: Any) -> str:
    if value is None:
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
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pog": _as_text(record.get("pog")),
        "side": _as_int(record.get("side")),
        "row": _as_int(record.get("row")),
        "column": _as_int(record.get("column")),
        "item_number": _as_text(record.get("item_number")),
        "retail": _as_text(record.get("retail")),
        "brand": _as_text(record.get("brand")),
        "desc_1": _as_text(record.get("desc_1")),
        "desc_2": _as_text(record.get("desc_2")),
        "upc": _as_text(record.get("upc")),
        "cpp": _as_text(record.get("cpp")),
        "file_path": _as_text(record.get("file_path")),
        "description": _as_text(record.get("description")),
    }


def _choose_selected_pog(detected_pogs: list[str], selected_pog: str | None, warnings: list[str]) -> str:
    if not detected_pogs:
        return ""
    if selected_pog and selected_pog in detected_pogs:
        return selected_pog
    if selected_pog and selected_pog not in detected_pogs:
        warnings.append(f"Selected POG '{selected_pog}' not found; defaulting to '{detected_pogs[0]}'.")
    if selected_pog is None and len(detected_pogs) > 1:
        warnings.append(f"Multiple POGs detected; defaulting to '{detected_pogs[0]}'.")
    return detected_pogs[0]


def detect_sams_pogs(access_file: Any) -> tuple[list[str], list[str]]:
    """Read Access records and return detected POG identifiers for UI selection."""
    warnings: list[str] = []
    try:
        records = extract_master_pog_records(access_file)
    except Exception as exc:
        return [], [f"Access extraction failed: {exc}"]

    pogs = sorted({_as_text(record.get("pog")) for record in records if _as_text(record.get("pog"))})
    if not pogs:
        warnings.append("No POG values found in Access records.")
    return pogs, warnings


def build_sams_planogram_structure(
    access_file: Any,
    excel_file: Any = None,
    selected_pog: str | None = None,
) -> SamsBuildResult:
    """
    Build a populated Sam's Club planogram structure from Access records.

    The pipeline is Access-first and currently structure-only (no PDF rendering).
    """
    warnings: list[str] = []
    if excel_file is not None:
        warnings.append("Excel support input received; integration is not implemented yet.")

    try:
        raw_records = extract_master_pog_records(access_file)
    except Exception as exc:
        warnings.append(f"Access extraction failed: {exc}")
        return SamsBuildResult(
            planogram=SamsPlanogram(pog="", side_pages=[], warnings=warnings.copy()),
            extracted_record_count=0,
            normalized_record_count=0,
            detected_pogs=[],
            selected_pog="",
            warnings=warnings,
            debug={
                "detected_pogs": [],
                "side_counts": {},
                "rows_per_side": {},
                "populated_columns_per_row": {},
                "warnings": warnings.copy(),
            },
        )

    normalized_records: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_records):
        normalized = _normalize_record(raw)
        if normalized["pog"] == "":
            warnings.append(f"Record {idx} skipped: missing pog.")
            continue
        if normalized["side"] is None or not validate_side(normalized["side"]):
            warnings.append(f"Record {idx} skipped: invalid side '{raw.get('side')}'.")
            continue
        if normalized["row"] is None or not validate_row(normalized["row"]):
            warnings.append(f"Record {idx} skipped: invalid row '{raw.get('row')}'.")
            continue
        if normalized["column"] is None or normalized["column"] <= 0:
            warnings.append(f"Record {idx} skipped: invalid column '{raw.get('column')}'.")
            continue
        if not validate_column(normalized["side"], normalized["column"]):
            max_cols = side_column_limit(normalized["side"])
            warnings.append(
                f"Record {idx} skipped: column {normalized['column']} exceeds side {normalized['side']} max {max_cols}."
            )
            continue
        normalized_records.append(normalized)

    unique_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, Any, Any, Any]] = set()
    for record in normalized_records:
        key = (record["pog"], record["side"], record["row"], record["column"])
        if key in seen_keys:
            warnings.append(f"Duplicate slot key skipped: {key}.")
            continue
        seen_keys.add(key)
        unique_records.append(record)

    _, duplicate_keys = validate_slot_key_uniqueness(unique_records)
    if duplicate_keys:
        warnings.append(f"Unexpected duplicate keys after filtering: {duplicate_keys}.")

    detected_pogs = sorted({record["pog"] for record in unique_records})
    if not detected_pogs:
        warnings.append("No valid POG records were available after normalization and validation.")
    chosen_pog = _choose_selected_pog(detected_pogs, selected_pog, warnings)
    selected_records = [record for record in unique_records if record["pog"] == chosen_pog] if chosen_pog else []

    side_rows: dict[int, dict[int, list[SamsSlot]]] = {}
    row_column_debug: dict[str, dict[str, int]] = {}
    side_counts: dict[str, int] = {}
    rows_per_side: dict[str, int] = {}

    for record in selected_records:
        slot_warnings: list[str] = []
        for field_name, label in _MISSING_FIELD_WARNINGS:
            if _as_text(record.get(field_name)) == "":
                message = (
                    f"{label}: pog={record['pog']} side={record['side']} row={record['row']} column={record['column']}"
                )
                slot_warnings.append(message)
                warnings.append(message)

        slot = SamsSlot(
            pog=record["pog"],
            side=record["side"],
            row=record["row"],
            column=record["column"],
            item_number=record["item_number"],
            retail=record["retail"],
            brand=record["brand"],
            desc_1=record["desc_1"],
            desc_2=record["desc_2"],
            upc=record["upc"],
            cpp=record["cpp"],
            file_path=record["file_path"],
            description=record["description"],
            warnings=slot_warnings,
        )

        side_map = side_rows.setdefault(slot.side, {})
        row_slots = side_map.setdefault(slot.row, [])
        row_slots.append(slot)

    side_pages: list[SamsSidePage] = []
    for side in sorted(side_rows):
        side_map = side_rows[side]
        rows: list[SamsRow] = []
        for row_number in sorted(side_map):
            row_slots = sorted(side_map[row_number], key=lambda slot: slot.column)
            populated_columns = len({slot.column for slot in row_slots})
            rows.append(
                SamsRow(
                    side=side,
                    row_number=row_number,
                    column_limit=side_column_limit(side),
                    populated_column_count=populated_columns,
                    slots=row_slots,
                )
            )
            row_column_debug.setdefault(str(side), {})[str(row_number)] = populated_columns

        total_slots = sum(len(row.slots) for row in rows)
        side_pages.append(
            SamsSidePage(
                pog=chosen_pog,
                side=side,
                column_limit=side_column_limit(side),
                rows=rows,
                total_rows=len(rows),
                total_slots=total_slots,
                warnings=[],
            )
        )

        side_counts[str(side)] = total_slots
        rows_per_side[str(side)] = len(rows)

    planogram = SamsPlanogram(
        pog=chosen_pog,
        side_pages=side_pages,
        warnings=warnings.copy(),
    )
    debug = {
        "detected_pogs": detected_pogs,
        "side_counts": side_counts,
        "rows_per_side": rows_per_side,
        "populated_columns_per_row": row_column_debug,
        "warnings": warnings.copy(),
    }

    return SamsBuildResult(
        planogram=planogram,
        extracted_record_count=len(raw_records),
        normalized_record_count=len(selected_records),
        detected_pogs=detected_pogs,
        selected_pog=chosen_pog,
        warnings=warnings,
        debug=debug,
    )
