"""Pure structural QA. No Streamlit, reference detection, or output rendering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.templates.models import DisplayTemplate, GeometryUnit, PresentationHint, SlotRole


# Ignore only floating-point boundary noise, in the declared template unit.
EPSILON = 1e-9


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    facing_id: str | None = None
    slot_ids: tuple[str, ...] = ()
    region_id: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == Severity.WARNING)

    @property
    def valid(self) -> bool:
        return not self.errors


def _finite(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def _positive_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_template(template: DisplayTemplate) -> ValidationResult:
    issues = []

    def report(code, message, facing=None, slots=(), severity=Severity.ERROR, region=None):
        issues.append(ValidationIssue(severity, code, message, facing, slots, region))

    for field in ("id", "name", "version", "template_family"):
        if not _text(getattr(template, field)):
            report("template_field", f"Template {field} must be nonempty text.")
    if template.unit not in (GeometryUnit.INCHES, GeometryUnit.NORMALIZED):
        report("unit", "Use inches (in) or explicitly uncalibrated normalized coordinates.")
    if not isinstance(template.locked, bool):
        report("locked", "Template locking must be true or false.")
    if not template.facings:
        report("facings", "A template needs at least one facing.")

    facing_ids, facing_orders, slot_ids, region_ids = set(), set(), set(), set()
    for facing in template.facings:
        if not _text(facing.id):
            report("facing_id", "Every facing needs a nonempty ID.")
        elif facing.id in facing_ids:
            report("duplicate_facing_id", f"Facing ID {facing.id} is repeated.", facing.id)
        else:
            facing_ids.add(facing.id)
        if not _text(facing.name):
            report("facing_name", "Every facing needs a name.", facing.id)
        if not _positive_integer(facing.order):
            report("facing_order", "Facing order must be a positive integer.", facing.id)
        elif facing.order in facing_orders:
            report("duplicate_facing_order", "Facing order values must be unique.", facing.id)
        else:
            facing_orders.add(facing.order)
        facing_size_valid = all(_finite(value) and value > 0 for value in (facing.width, facing.height))
        if not facing_size_valid:
            report("facing_dimensions", "Facing width and height must be finite and positive.", facing.id)

        slot_orders, rectangles = set(), []
        for slot in facing.slots:
            scope = (slot.id,)
            if not _text(slot.id):
                report("slot_id", "Every slot needs a nonempty ID.", facing.id)
            elif slot.id in slot_ids:
                report("duplicate_slot_id", f"Slot ID {slot.id} is repeated within the template.", facing.id, scope)
            else:
                slot_ids.add(slot.id)
            if slot.role not in tuple(SlotRole):
                report("role", f"Slot {slot.id} has an unsupported role.", facing.id, scope)
            if not _positive_integer(slot.order):
                report("slot_order", "Slot order must be a positive integer.", facing.id, scope)
            elif slot.order in slot_orders:
                report("duplicate_slot_order", "Slot order values must be unique within a facing.", facing.id, scope)
            else:
                slot_orders.add(slot.order)
            for field in ("row", "column"):
                if getattr(slot, field) is not None and not _positive_integer(getattr(slot, field)):
                    report("grid_index", f"Optional {field} must be a positive integer.", facing.id, scope)
            if slot.section_id is not None and not _text(slot.section_id):
                report("section_id", "Optional section ID must be nonempty text.", facing.id, scope)
            if not all(_finite(value) for value in (slot.x, slot.y, slot.width, slot.height)):
                report("slot_coordinates", "Slot coordinates and sizes must be finite numbers.", facing.id, scope)
                continue
            if slot.width <= 0 or slot.height <= 0:
                report("slot_dimensions", "Slot width and height must be positive.", facing.id, scope)
                continue
            if slot.x < 0 or slot.y < 0:
                report("slot_bounds", "Slot origin must be inside its facing.", facing.id, scope)
            if facing_size_valid and (slot.x + slot.width > facing.width + EPSILON
                                      or slot.y + slot.height > facing.height + EPSILON):
                report("slot_bounds", "Slot extends beyond its facing.", facing.id, scope)
            rectangles.append(slot)

        _validate_regions(facing, report, region_ids, facing_size_valid)

        for index, left in enumerate(rectangles):
            for right in rectangles[index + 1:]:
                width = min(left.x + left.width, right.x + right.width) - max(left.x, right.x)
                height = min(left.y + left.height, right.y + right.height) - max(left.y, right.y)
                if width > EPSILON and height > EPSILON:
                    products = left.role == right.role == SlotRole.PRODUCT
                    pair = tuple(sorted((left.id, right.id)))
                    report("product_overlap" if products else "slot_overlap",
                           f"Slots {pair[0]} and {pair[1]} overlap.", facing.id, pair,
                           Severity.ERROR if products else Severity.WARNING)

    # Stable diagnostics even when storage/list order changes.
    return ValidationResult(tuple(sorted(issues, key=lambda issue: (
        issue.facing_id or "", issue.region_id or "", issue.slot_ids, issue.code, issue.message,
    ))))


def _validate_regions(facing, report, template_region_ids, facing_size_valid):
    by_id = {slot.id: slot for slot in facing.slots if _text(slot.id)}
    memberships = {slot_id: [] for slot_id in by_id}
    orders = set()
    for region in facing.regions:
        def issue(code, message, slots=()):
            report(code, message, facing.id, slots, region=region.id)

        if not _text(region.id):
            issue("region_id", "Every region needs a nonempty ID.")
        elif region.id in template_region_ids:
            issue("duplicate_region_id", f"Region ID {region.id} is repeated within the template.")
        else:
            template_region_ids.add(region.id)
        if not _text(region.name):
            issue("region_name", "Every region needs a display name.")
        if region.role not in tuple(SlotRole):
            issue("region_role", f"Region {region.id} has an unsupported role.")
        if not _positive_integer(region.order):
            issue("region_order", "Region order must be a positive integer.")
        elif region.order in orders:
            issue("duplicate_region_order", "Region order values must be unique within a facing.")
        else:
            orders.add(region.order)
        if region.presentation_hint is not None and region.presentation_hint not in tuple(PresentationHint):
            issue("presentation_hint", f"Region {region.id} has an unsupported presentation hint.")
        if not isinstance(region.allow_shared_slots, bool):
            issue("shared_membership_flag", "Shared slot membership must be explicitly true or false.")

        bounds = region.bounds
        bounds_valid = bounds is not None and all(_finite(value) for value in (
            bounds.x, bounds.y, bounds.width, bounds.height,
        )) and bounds.width > 0 and bounds.height > 0
        if bounds is not None:
            if not bounds_valid:
                issue("region_dimensions", "Region bounds must have finite coordinates and positive sizes.")
            elif bounds.x < 0 or bounds.y < 0 or (facing_size_valid and (
                bounds.x + bounds.width > facing.width + EPSILON
                or bounds.y + bounds.height > facing.height + EPSILON
            )):
                issue("region_bounds", "Region bounds extend beyond the facing.")

        seen = set()
        for slot_id in region.slot_ids:
            if not _text(slot_id) or slot_id not in by_id:
                issue("region_slot_reference", f"Region {region.id} references an unknown facing slot: {slot_id}.")
                continue
            if slot_id in seen:
                issue("duplicate_region_slot", f"Region {region.id} repeats slot {slot_id}.", (slot_id,))
                continue
            seen.add(slot_id)
            memberships[slot_id].append(region)
            slot = by_id[slot_id]
            if bounds_valid and all(_finite(value) for value in (slot.x, slot.y, slot.width, slot.height)):
                if (slot.x < bounds.x - EPSILON or slot.y < bounds.y - EPSILON
                        or slot.x + slot.width > bounds.x + bounds.width + EPSILON
                        or slot.y + slot.height > bounds.y + bounds.height + EPSILON):
                    issue("slot_region_bounds", f"Slot {slot_id} extends beyond region {region.id}.", (slot_id,))

    if facing.regions:
        for slot_id, regions in memberships.items():
            if not regions:
                report("unassigned_region_slot", f"Slot {slot_id} has no region membership.", facing.id, (slot_id,))
            elif len(regions) > 1 and not all(region.allow_shared_slots is True for region in regions):
                names = ", ".join(sorted(region.id for region in regions))
                report("multiple_region_membership", f"Slot {slot_id} belongs to multiple regions: {names}.",
                       facing.id, (slot_id,))
