"""Pure structural QA. No Streamlit, reference detection, or output rendering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.templates.models import DisplayTemplate, GeometryUnit, SlotRole


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

    def report(code, message, facing=None, slots=(), severity=Severity.ERROR):
        issues.append(ValidationIssue(severity, code, message, facing, slots))

    for field in ("id", "name", "version", "template_family"):
        if not _text(getattr(template, field)):
            report("template_field", f"Template {field} must be nonempty text.")
    if template.unit not in (GeometryUnit.INCHES, GeometryUnit.NORMALIZED):
        report("unit", "Use inches (in) or explicitly uncalibrated normalized coordinates.")
    if not isinstance(template.locked, bool):
        report("locked", "Template locking must be true or false.")
    if not template.facings:
        report("facings", "A template needs at least one facing.")

    facing_ids, facing_orders, slot_ids = set(), set(), set()
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
        issue.facing_id or "", issue.slot_ids, issue.code, issue.message,
    ))))
