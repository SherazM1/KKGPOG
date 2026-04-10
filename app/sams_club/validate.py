from __future__ import annotations

from typing import Any, Iterable, Mapping


def side_column_limit(side: int) -> int:
    """Return the allowed column count for a Sam's Club side."""
    side_no = int(side)
    if side_no in (1, 3):
        return 8
    if side_no in (2, 4):
        return 10
    raise ValueError(f"Unsupported side: {side}")


def validate_side(side: int) -> bool:
    """Validate allowed Sam's Club side values."""
    try:
        return int(side) in (1, 2, 3, 4)
    except (TypeError, ValueError):
        return False


def validate_row(row: int) -> bool:
    """Validate that row is a positive integer."""
    try:
        return int(row) > 0
    except (TypeError, ValueError):
        return False


def validate_column(side: int, column: int) -> bool:
    """Validate a column value against the side's column limit."""
    try:
        col = int(column)
    except (TypeError, ValueError):
        return False

    if col < 1 or not validate_side(side):
        return False
    return col <= side_column_limit(int(side))


def validate_slot_key_uniqueness(
    records: Iterable[Mapping[str, Any]],
) -> tuple[bool, list[tuple[Any, Any, Any, Any]]]:
    """Check uniqueness of (pog, side, row, column) keys across records."""
    seen: set[tuple[Any, Any, Any, Any]] = set()
    duplicates: list[tuple[Any, Any, Any, Any]] = []
    for record in records:
        key = (record.get("pog"), record.get("side"), record.get("row"), record.get("column"))
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
    return len(duplicates) == 0, duplicates
