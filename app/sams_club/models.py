from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamsSlot:
    """Single slot-level record in a Sam's Club planogram."""

    pog: str = ""
    side: int = 0
    row: int = 0
    column: int = 0
    item_number: str = ""
    retail: str = ""
    brand: str = ""
    desc_1: str = ""
    desc_2: str = ""
    upc: str = ""
    cpp: str = ""
    file_path: str = ""
    resolved_image_path: str = ""
    image_resolution_source: str = "unresolved"
    description: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsRow:
    """Row-level grouping of Sam's Club slots."""

    side: int
    row_number: int
    column_limit: int
    populated_column_count: int = 0
    slots: list[SamsSlot] = field(default_factory=list)


@dataclass
class SamsSidePage:
    """One rendered side page in a Sam's Club planogram."""

    pog: str
    side: int
    column_limit: int
    rows: list[SamsRow] = field(default_factory=list)
    total_rows: int = 0
    total_slots: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsPlanogram:
    """Top-level Sam's Club planogram structure."""

    pog: str
    side_pages: list[SamsSidePage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
