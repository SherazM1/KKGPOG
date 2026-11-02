"""Operator defaults/buttons composed from the unchanged draft operations."""
from dataclasses import replace

from app.templates import draft as edits
from app.templates.validation import EPSILON


def _section(facing, region_id):
    return next(region for region in facing.regions if region.id == region_id)


def _area(facing, region):
    if region.bounds is not None:
        return region.bounds.x, region.bounds.y, region.bounds.width, region.bounds.height
    return 0.0, 0.0, facing.width, facing.height


def _open_origin(facing, region, width, height, margin=.025):
    x0, y0, area_width, area_height = _area(facing, region)
    gap_x, gap_y = area_width * margin, area_height * margin
    left, top = x0 + gap_x, y0 + gap_y
    right, bottom = x0 + area_width - gap_x - width, y0 + area_height - gap_y - height
    xs, ys = {left}, {top}
    for slot in facing.slots:
        xs.update((slot.x + slot.width + gap_x, slot.x - width - gap_x))
        ys.update((slot.y + slot.height + gap_y, slot.y - height - gap_y))
    for y in sorted(ys):
        for x in sorted(xs):
            if x < left - EPSILON or x > right + EPSILON or y < top - EPSILON or y > bottom + EPSILON:
                continue
            if not any(min(x + width, slot.x + slot.width) - max(x, slot.x) > EPSILON
                       and min(y + height, slot.y + slot.height) - max(y, slot.y) > EPSILON
                       for slot in facing.slots):
                return x, y
    raise ValueError("There is not enough open space in this section. Make room by moving or resizing boxes.")


def grid_defaults(facing, region_id, rows, columns):
    """Supply safe dimensions to the existing grid API, never a new grid model."""
    region = _section(facing, region_id)
    _, _, area_width, area_height = _area(facing, region)
    for width_ratio, height_ratio in ((.9, .4), (.9, .25), (.6, .3), (.45, .3), (.3, .2)):
        width, height = area_width * width_ratio, area_height * height_ratio
        try:
            x, y = _open_origin(facing, region, width, height, margin=.05)
        except ValueError:
            continue
        return dict(x=x, y=y, width=width, height=height,
                    horizontal_gap=min(area_width * .02, width / (columns * 8)),
                    vertical_gap=min(area_height * .02, height / (rows * 8)))
    raise ValueError("There is not enough open space for this grid. Make room or use Advanced grid settings.")


def add_box(draft, facing_id, region_id):
    facing = edits.get_facing(draft, facing_id)
    region = _section(facing, region_id)
    _, _, area_width, area_height = _area(facing, region)
    width, height = area_width * .2, area_height * .15
    x, y = _open_origin(facing, region, width, height)
    return edits.add_slot(draft, facing_id, region_id, x=x, y=y, width=width, height=height)


def duplicate_box(draft, facing_id, region_id, slot_id):
    facing = edits.get_facing(draft, facing_id)
    source = next(slot for slot in facing.slots_for_region(region_id) if slot.id == slot_id)
    x, y = _open_origin(facing, _section(facing, region_id), source.width, source.height)
    updated = edits.add_slot(draft, facing_id, region_id, x=x, y=y, width=source.width, height=source.height)
    face = edits.get_facing(updated, facing_id)
    copied = replace(face.slots[-1], role=source.role, row=source.row, column=source.column, metadata=source.metadata)
    face = replace(face, slots=(*face.slots[:-1], copied))
    return replace(updated, template=replace(updated.template, facings=tuple(
        face if item.id == face.id else item for item in updated.template.facings)))


def adjust_box(draft, facing_id, region_id, slot_id, action):
    facing = edits.get_facing(draft, facing_id)
    slot = next(slot for slot in facing.slots_for_region(region_id) if slot.id == slot_id)
    x0, y0, area_width, area_height = _area(facing, _section(facing, region_id))
    right, bottom = x0 + area_width, y0 + area_height
    dx, dy = area_width * .02, area_height * .02
    x, y, width, height = slot.x, slot.y, slot.width, slot.height
    if action == "left":
        x = max(x0, x - dx)
    elif action == "right":
        x = max(x0, min(right - width, x + dx))
    elif action == "up":
        y = max(y0, y - dy)
    elif action == "down":
        y = max(y0, min(bottom - height, y + dy))
    elif action == "wider":
        width = min(right - x, width + dx)
    elif action == "narrower":
        width = max(area_width * .01, width - dx)
    elif action == "taller":
        height = min(bottom - y, height + dy)
    elif action == "shorter":
        height = max(area_height * .01, height - dy)
    else:
        raise ValueError("Unknown box adjustment.")
    return edits.update_slot(draft, facing_id, slot_id, x=x, y=y, width=width, height=height,
                             order=slot.order, region_id=region_id)
