"""Pure session-draft operations. No registry mutation, I/O, or production imports."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from uuid import uuid4

from app.templates.models import DisplayTemplate, Facing, GeometryUnit, PresentationHint, Region, Slot
from app.templates.models import SlotRole
from app.templates.validation import EPSILON, validate_template


@dataclass(frozen=True)
class TemplateDraft:
    template: DisplayTemplate
    next_id: int = 1
    revision: int = 0

    def __post_init__(self):
        if self.template.locked:
            raise ValueError("Locked templates cannot be edited as drafts.")


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Enter a {label}.")
    return value.strip()


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return value


def _id(draft, kind):
    return f"{draft.template.id}.{kind}{draft.next_id:04d}"


def new_draft(name, family, version, unit) -> TemplateDraft:
    unit = GeometryUnit(unit)
    return TemplateDraft(DisplayTemplate(
        id=f"draft_{uuid4().hex}", name=_name(name, "template name"),
        template_family=_name(family, "display type"), version=_name(version, "version"),
        unit=unit, locked=False, facings=(), metadata={
            "origin": "Created manually in this session; not saved.",
            "calibration": "Uncalibrated template space." if unit == GeometryUnit.NORMALIZED else
                           "Dimensions entered in inches; measurement accuracy requires operator verification.",
        },
    ))


def get_facing(draft, facing_id):
    for facing in draft.template.facings:
        if facing.id == facing_id:
            return facing
    raise KeyError("The selected facing is no longer in this draft.")


def _region(facing, region_id):
    for region in facing.regions:
        if region.id == region_id:
            return region
    raise KeyError("Choose a region on this facing.")


def _changed(draft, facing, used_ids=0):
    return replace(draft, template=replace(draft.template, facings=tuple(
        facing if item.id == facing.id else item for item in draft.template.facings
    )), next_id=draft.next_id + used_ids, revision=draft.revision + 1)


def add_facing(draft, name, width, height, order=None):
    width, height = _number(width, "Facing width", True), _number(height, "Facing height", True)
    facing = Facing(_id(draft, "facing"), _name(name, "facing name"), width, height,
                    order if order is not None else max((item.order for item in draft.template.facings), default=0) + 1)
    return replace(draft, template=replace(draft.template, facings=(*draft.template.facings, facing)),
                   next_id=draft.next_id + 1, revision=draft.revision + 1)


def update_facing(draft, facing_id, *, name, width, height, order):
    facing = get_facing(draft, facing_id)
    # Bounds/order violations remain visible draft validation issues, not clamped values.
    return _changed(draft, replace(facing, name=_name(name, "facing name"),
                                  width=_number(width, "Facing width"),
                                  height=_number(height, "Facing height"), order=order))


def remove_facing(draft, facing_id):
    get_facing(draft, facing_id)
    return replace(draft, template=replace(draft.template, facings=tuple(
        item for item in draft.template.facings if item.id != facing_id
    )), revision=draft.revision + 1)


def add_region(draft, facing_id, name, role=SlotRole.PRODUCT, hint=None, order=None):
    facing = get_facing(draft, facing_id)
    region = Region(_id(draft, "region"), _name(name, "region name"), SlotRole(role),
                    order if order is not None else max((item.order for item in facing.regions), default=0) + 1,
                    presentation_hint=PresentationHint(hint) if hint is not None else None)
    return _changed(draft, replace(facing, regions=(*facing.regions, region)), 1)


def update_region(draft, facing_id, region_id, *, name, role, hint, order):
    facing = get_facing(draft, facing_id)
    region = replace(_region(facing, region_id), name=_name(name, "region name"), role=SlotRole(role),
                     presentation_hint=PresentationHint(hint) if hint is not None else None, order=order)
    return _changed(draft, replace(facing, regions=tuple(
        region if item.id == region_id else item for item in facing.regions
    )))


def remove_region(draft, facing_id, region_id):
    facing = get_facing(draft, facing_id)
    removed = _region(facing, region_id)
    regions = tuple(item for item in facing.regions if item.id != region_id)
    retained_members = {slot_id for item in regions for slot_id in item.slot_ids}
    removed_members = set(removed.slot_ids) - retained_members
    return _changed(draft, replace(facing, regions=regions, slots=tuple(
        slot for slot in facing.slots if slot.id not in removed_members
    )))


def suggest_slot(facing):
    """Prefer a free small rectangle; never infer a physical card size."""
    width, height = facing.width * .20, facing.height * .15
    for row in range(5):
        for column in range(4):
            x, y = facing.width * (.025 + column * .25), facing.height * (.025 + row * .20)
            if not any(min(x + width, slot.x + slot.width) - max(x, slot.x) > EPSILON
                       and min(y + height, slot.y + slot.height) - max(y, slot.y) > EPSILON
                       for slot in facing.slots):
                return x, y, width, height
    return facing.width * .025, facing.height * .025, width, height


def add_slot(draft, facing_id, region_id, *, x=None, y=None, width=None, height=None, order=None):
    facing = get_facing(draft, facing_id)
    region = _region(facing, region_id)
    defaults = suggest_slot(facing) if any(value is None for value in (x, y, width, height)) else (None,) * 4
    values = [default if value is None else _number(value, label)
              for value, default, label in zip((x, y, width, height), defaults, ("Left", "Top", "Width", "Height"))]
    slot = Slot(_id(draft, "slot"), region.role, *values,
                order if order is not None else max((item.order for item in facing.slots), default=0) + 1)
    regions = tuple(replace(item, slot_ids=(*item.slot_ids, slot.id)) if item.id == region_id else item
                    for item in facing.regions)
    return _changed(draft, replace(facing, slots=(*facing.slots, slot), regions=regions), 1)


def create_grid(draft, facing_id, region_id, *, rows, columns, x, y, width, height, horizontal_gap=0, vertical_gap=0):
    facing = get_facing(draft, facing_id)
    region = _region(facing, region_id)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (rows, columns)):
        raise ValueError("Rows and positions per row must be positive whole numbers.")
    for value, label in ((x, "Left"), (y, "Top"), (horizontal_gap, "Horizontal gap"), (vertical_gap, "Vertical gap")):
        if _number(value, label) < 0:
            raise ValueError(f"{label} cannot be negative.")
    _number(width, "Grid width", True)
    _number(height, "Grid height", True)
    cell_width = (width - (columns - 1) * horizontal_gap) / columns
    cell_height = (height - (rows - 1) * vertical_gap) / rows
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("The gaps leave no room for positions. Reduce the gaps or increase the overall size.")
    if x + width > facing.width + EPSILON or y + height > facing.height + EPSILON:
        raise ValueError("The grid extends beyond the facing. Adjust its start or overall size.")
    # Each cell is materialized immediately; there is no grid geometry dependency.
    slots = []
    first_order = max((item.order for item in facing.slots), default=0) + 1
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            slots.append(Slot(f"{draft.template.id}.slot{draft.next_id + index:04d}", region.role,
                              x + column * (cell_width + horizontal_gap),
                              y + row * (cell_height + vertical_gap), cell_width, cell_height,
                              first_order + index, row=row + 1, column=column + 1))
    regions = tuple(replace(item, slot_ids=(*item.slot_ids, *(slot.id for slot in slots)))
                    if item.id == region_id else item for item in facing.regions)
    return _changed(draft, replace(facing, slots=(*facing.slots, *slots), regions=regions), len(slots))


def update_slot(draft, facing_id, slot_id, *, x, y, width, height, order, region_id):
    facing = get_facing(draft, facing_id)
    _region(facing, region_id)
    selected = next((slot for slot in facing.slots if slot.id == slot_id), None)
    if selected is None:
        raise KeyError("The selected position is no longer in this draft.")
    values = [_number(value, label) for value, label in zip((x, y, width, height), ("Left", "Top", "Width", "Height"))]
    selected = replace(selected, x=values[0], y=values[1], width=values[2], height=values[3], order=order)
    regions = tuple(replace(item, slot_ids=tuple(member for member in item.slot_ids if member != slot_id)
                            + ((slot_id,) if item.id == region_id else ())) for item in facing.regions)
    return _changed(draft, replace(facing, slots=tuple(
        selected if slot.id == slot_id else slot for slot in facing.slots
    ), regions=regions))


def remove_slot(draft, facing_id, slot_id):
    facing = get_facing(draft, facing_id)
    if not any(slot.id == slot_id for slot in facing.slots):
        raise KeyError("The selected position is no longer in this draft.")
    return _changed(draft, replace(facing, slots=tuple(slot for slot in facing.slots if slot.id != slot_id),
                                  regions=tuple(replace(region, slot_ids=tuple(
                                      member for member in region.slot_ids if member != slot_id
                                  )) for region in facing.regions)))


def reset_draft(draft):
    """Clear layout, keep basics, and do not recycle previously issued IDs."""
    return replace(draft, template=replace(draft.template, facings=()), revision=draft.revision + 1)


def validate_draft(draft):
    return validate_template(draft.template)
