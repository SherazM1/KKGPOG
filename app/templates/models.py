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

    def __post_init__(self):
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def ordered_slots(self) -> tuple[Slot, ...]:
        """Valid templates have unique orders; IDs provide a deterministic tie-break."""
        return tuple(sorted(self.slots, key=lambda slot: (slot.order, slot.id)))


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
