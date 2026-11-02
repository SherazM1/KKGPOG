"""Canvas projection/event boundary. The existing TemplateDraft is authoritative."""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from math import isclose, isfinite
from xml.etree import ElementTree as ET

from app.templates import draft as edits


SELECTION_EPOCH_KEY = "template_builder:canvas_selection_epoch"


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("Canvas coordinates must be finite numbers.")
    return float(value)


@dataclass(frozen=True)
class CanvasTransform:
    origin_x: float
    origin_y: float
    scale: float

    @classmethod
    def from_svg(cls, svg, facing):
        root = ET.fromstring(svg)
        boundary = next(node for node in root.iter() if node.get("data-kind") == "facing")
        scale = float(boundary.get("width")) / facing.width
        if not isfinite(scale) or scale <= 0 or not isclose(
                float(boundary.get("height")) / facing.height, scale, rel_tol=1e-5):
            raise ValueError("The canvas must use the same scale on both axes.")
        return cls(float(boundary.get("x")), float(boundary.get("y")), scale)

    def to_template(self, rectangle):
        if not isinstance(rectangle, Mapping) or set(rectangle) != {"x", "y", "width", "height"}:
            raise ValueError("The canvas edit is missing its box measurements.")
        x, y, width, height = (_number(rectangle[field]) for field in ("x", "y", "width", "height"))
        if width <= 0 or height <= 0:
            raise ValueError("A box must have a width and height greater than zero.")
        return dict(x=(x - self.origin_x) / self.scale, y=(y - self.origin_y) / self.scale,
                    width=width / self.scale, height=height / self.scale)

    def to_canvas(self, slot):
        return dict(x=self.origin_x + slot.x * self.scale, y=self.origin_y + slot.y * self.scale,
                    width=slot.width * self.scale, height=slot.height * self.scale)


@dataclass(frozen=True)
class CanvasEdit:
    draft: edits.TemplateDraft
    slot_id: str | None = None
    region_id: str | None = None
    accepted: bool = False
    changed: bool = False
    error: str | None = None


def apply_event(draft, facing_id, event, transform, active_region_id=None):
    """Translate one gesture; never accept replacement models, IDs or membership."""
    try:
        if not isinstance(event, Mapping) or event.get("kind") not in ("select", "drag", "resize"):
            raise ValueError("This canvas action is not supported.")
        if not isinstance(event.get("event_id"), str) or not event["event_id"]:
            raise ValueError("The canvas edit is missing its event ID.")
        if event.get("template_id") != draft.template.id or event.get("facing_id") != facing_id:
            raise ValueError("This canvas edit belongs to another layout or side.")
        revision = event.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision != draft.revision:
            raise ValueError("The layout changed before this edit arrived. Please try again.")
        facing = edits.get_facing(draft, facing_id)
        slot = next((slot for slot in facing.slots if slot.id == event.get("slot_id")), None)
        if slot is None:
            raise ValueError("This box is no longer on the active side.")
        regions = facing.regions_for_slot(slot.id)
        if not regions:
            raise ValueError("Assign this box to a section in Advanced settings before canvas editing.")
        region_id = active_region_id if any(item.id == active_region_id for item in regions) else regions[0].id
        if event["kind"] == "select":
            return CanvasEdit(draft, slot.id, region_id, accepted=True)
        geometry = transform.to_template(event.get("rect"))
        if event["kind"] == "drag":
            geometry.update(width=slot.width, height=slot.height)
        geometry = {field: getattr(slot, field) if isclose(getattr(slot, field), value,
                    rel_tol=1e-9, abs_tol=1e-9) else value for field, value in geometry.items()}
        if all(isclose(getattr(slot, field), value, rel_tol=1e-9, abs_tol=1e-9) for field, value in geometry.items()):
            return CanvasEdit(draft, slot.id, region_id, accepted=True)
        updated = edits.update_slot(draft, facing_id, slot.id, **geometry, order=slot.order, region_id=region_id)
        # update_slot normally makes a single assignment. A canvas gesture must
        # preserve even explicitly shared memberships, including their ordering.
        face = replace(edits.get_facing(updated, facing_id), regions=facing.regions)
        updated = replace(updated, template=replace(updated.template, facings=tuple(
            face if item.id == facing_id else item for item in updated.template.facings)))
        return CanvasEdit(updated, slot.id, region_id, accepted=True, changed=True)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return CanvasEdit(draft, error=str(exc).strip("'"))


def consume_event(state, *, component_key, draft_key, selection_key, facing_id, transform):
    """Component callback: apply once before controls render; acknowledge even rejection."""
    event = state.get(component_key)
    token = hashlib.sha256(json.dumps(event, sort_keys=True).encode("utf-8")).hexdigest()
    if state.get(f"{component_key}:last") == token:
        return
    state[f"{component_key}:last"] = token
    state[f"{component_key}:ack"] = event.get("event_id") if isinstance(event, Mapping) else None
    draft = state.get(draft_key)
    if draft is None:
        return
    selection = dict(state.get(selection_key, {}))
    if selection.get("facing") not in (None, facing_id):
        state[f"{component_key}:notice"] = "The active side changed. Please try your edit on the current side."
        return
    result = apply_event(draft, facing_id, event, transform, selection.get("region"))
    if result.error:
        state[f"{component_key}:notice"] = result.error
        return
    state[draft_key] = result.draft
    next_selection = dict(facing=facing_id, region=result.region_id, slot=result.slot_id)
    if selection != next_selection:
        state[SELECTION_EPOCH_KEY] = state.get(SELECTION_EPOCH_KEY, 0) + 1
    state[selection_key] = next_selection
