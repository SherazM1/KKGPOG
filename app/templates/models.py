"""Top-left fixture coordinates, in inches or explicitly uncalibrated template space.

All dimensions within a template use its declared unit. Reference pixels and PDF
points are deliberately unsupported. See docs/geometry_contract.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class GeometryUnit(str, Enum):
    INCHES = "in"
    NORMALIZED = "normalized"


class SlotRole(str, Enum):
    PRODUCT = "PRODUCT"
    FILLER = "FILLER"
    HEADER = "HEADER"
    SIGNAGE = "SIGNAGE"
    PRICE_POCKET = "PRICE_POCKET"
    RESERVED = "RESERVED"


class PresentationHint(str, Enum):
    """Optional future grouping intent; never a coordinate transform."""
    GRID = "grid"
    SINGLE_ROW = "single_row"
    CUSTOM_ROWS = "custom_rows"
    SECTION = "section"
    PRESERVE_RELATIVE = "preserve_relative"


@dataclass(frozen=True)
class Bounds:
    """A facing-local rectangular envelope, in the template's declared unit."""
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Region:
    """Explicit ID-based slot membership, separate from physical slot storage.

    Bounds are an optional envelope, not occupied space or a local origin.
    Shared membership requires explicit consent from every participating region.
    """
    id: str
    name: str
    role: SlotRole
    order: int
    slot_ids: tuple[str, ...] = ()
    bounds: Bounds | None = None
    presentation_hint: PresentationHint | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    allow_shared_slots: bool = False

    def __post_init__(self):
        object.__setattr__(self, "slot_ids", tuple(self.slot_ids))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Slot:
    id: str
    role: SlotRole
    x: float
    y: float
    width: float
    height: float
    order: int
    row: int | None = None
    column: int | None = None
    section_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Facing:
    id: str
    name: str
    width: float
    height: float
    order: int
    slots: tuple[Slot, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "regions", tuple(self.regions))

    def ordered_slots(self) -> tuple[Slot, ...]:
        """Valid templates have unique orders; IDs provide a deterministic tie-break."""
        return tuple(sorted(self.slots, key=lambda slot: (slot.order, slot.id)))

    def ordered_regions(self) -> tuple[Region, ...]:
        return tuple(sorted(self.regions, key=lambda region: (region.order, region.id)))

    def slots_for_region(self, region_id: str) -> tuple[Slot, ...]:
        region = next((region for region in self.regions if region.id == region_id), None)
        if region is None:
            raise KeyError(f"Unknown region: {region_id}")
        by_id = {slot.id: slot for slot in self.slots}
        return tuple(sorted((by_id[slot_id] for slot_id in region.slot_ids),
                            key=lambda slot: (slot.order, slot.id)))

    def regions_for_slot(self, slot_id: str) -> tuple[Region, ...]:
        if not any(slot.id == slot_id for slot in self.slots):
            raise KeyError(f"Unknown slot: {slot_id}")
        return tuple(region for region in self.ordered_regions() if slot_id in region.slot_ids)


@dataclass(frozen=True)
class DisplayTemplate:
    id: str
    name: str
    version: str
    template_family: str
    unit: GeometryUnit
    locked: bool
    facings: tuple[Facing, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "facings", tuple(self.facings))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def ordered_facings(self) -> tuple[Facing, ...]:
        return tuple(sorted(self.facings, key=lambda facing: (facing.order, facing.id)))
