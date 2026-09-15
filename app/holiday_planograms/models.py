from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class HolidayProduct:
    product_name: str
    denomination: str
    product_upc: str
    pack_upc: str
    item_number: str
    nre_facings: int
    qtr_facings: int
    total_pegs: Optional[int]
    cpp: Optional[int]
    source_row: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BosPlanogramProduct:
    upc: str
    item_number: str
    name: str
    facings: int
    cpp: Optional[int]
    capacity: Optional[int]
    source_row: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceImage:
    name: str
    bytes_data: bytes
    width: int
    height: int
    anchor_row: int
    anchor_col: int


@dataclass(frozen=True)
class LayoutSlot:
    panel_index: int
    panel_name: str
    row: int
    column: int
    bbox: BBox
    center_x: float
    center_y: float
    width: float
    height: float
    active: bool = True
    detection_confidence: float = 1.0
    slot_type: str = "merchandise"
    classification_confidence: float = 1.0


@dataclass(frozen=True)
class LayoutPanel:
    panel_index: int
    panel_name: str
    bbox: BBox
    rows: int
    columns: int
    active_slots: int
    slots: list[LayoutSlot] = field(default_factory=list)


@dataclass(frozen=True)
class GeometryResult:
    panels: list[LayoutPanel]
    slots: list[LayoutSlot]
    warnings: list[str] = field(default_factory=list)
    debug_image_bytes: bytes = b""


@dataclass(frozen=True)
class ProductMatch:
    bos: BosPlanogramProduct
    d5: Optional[HolidayProduct]
    status: str
    method: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HolidayResolvedImage:
    image_path: str
    status: str
    source: str


@dataclass(frozen=True)
class HolidayPlacement:
    panel_index: int
    panel_name: str
    row: int
    column: int
    bbox: BBox
    reference_text: str
    product_upc: str
    pack_upc: str
    item_number: str
    product_name: str
    denomination: str
    cpp: Optional[int]
    capacity: Optional[int]
    image_path: str
    placement_status: str
    image_status: str
    image_resolution_source: str
    warnings: list[str] = field(default_factory=list)
    normalized_reference_text: str = ""
    match_method: str = ""
    match_confidence: str = ""


@dataclass(frozen=True)
class HolidayQpResult:
    bos_sheet: str
    d5_sheet: str
    reference_image: Optional[ReferenceImage]
    bos_products: list[BosPlanogramProduct]
    d5_products: list[HolidayProduct]
    d5_qp_products: list[HolidayProduct]
    product_matches: list[ProductMatch]
    geometry: Optional[GeometryResult]
    placements: list[HolidayPlacement]
    product_qa_rows: list[dict[str, Any]]
    placement_qa_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HolidayNreReference:
    configuration: str
    sheet_name: str
    title_text: str
    title_row: int
    title_col: int
    image: ReferenceImage
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HolidayNreResult:
    configuration: str
    bos_sheet: str
    d5_sheet: str
    reference: Optional[HolidayNreReference]
    d5_products: list[HolidayProduct]
    d5_nre_products: list[HolidayProduct]
    geometry: Optional[GeometryResult]
    placements: list[HolidayPlacement]
    product_qa_rows: list[dict[str, Any]]
    placement_qa_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HolidayNreConfigurationSummary:
    configuration: str
    sheet_name: str
    reference_image_name: str
    reference_width: int
    reference_height: int
    panel_count: int
    row_count: int
    column_count: int
    active_slots: int
    grid_positions: int
    physical_detected: int
    merchandise_slots: int
    filler_slots: int
    ambiguous_slots: int
    expected_nre_facings: int
    difference: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HolidayQpRenderResult:
    pdf_bytes: bytes
    preview_png_bytes: bytes
    panels_rendered: int
    slots_rendered: int
    image_slots: int
    placeholder_slots: int
    skipped_slots: int
    missing_image_rows: list[dict[str, Any]]
    render_rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
