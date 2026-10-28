"""Small built-in catalog. No persistence or production renderer imports."""
from app.templates.defaults import holiday_nre, holiday_qp_2026
from app.templates.models import DisplayTemplate, Facing


_BUILDERS = (holiday_qp_2026, holiday_nre)


def get_templates() -> tuple[DisplayTemplate, ...]:
    """Fresh, immutable definitions in explicit catalog order."""
    return tuple(builder() for builder in _BUILDERS)


def get_template(template_id: str) -> DisplayTemplate:
    for template in get_templates():
        if template.id == template_id:
            return template
    raise KeyError(f"Unknown display template: {template_id}")


def get_facings(template_id: str) -> tuple[Facing, ...]:
    return get_template(template_id).ordered_facings()
