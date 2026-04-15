from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamsPriceStripSegment:
    pog: str
    side: int
    row: int
    column: int
    item_number: str = ""
    brand: str = ""
    desc_1: str = ""
    desc_2: str = ""
    retail: str = ""
    length: str = ""
    data_on_bottom_left: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsPriceStripRow:
    pog: str
    side: int
    row: int
    segments: list[SamsPriceStripSegment] = field(default_factory=list)
    footer_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsPriceStripBuildResult:
    strip_rows: list[SamsPriceStripRow] = field(default_factory=list)
    extracted_record_count: int = 0
    included_segment_count: int = 0
    skipped_segment_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamsPriceStripPdfResult:
    pdf_bytes: bytes
    rendered_pages: int
    rendered_segments: int
    warnings: list[str] = field(default_factory=list)
