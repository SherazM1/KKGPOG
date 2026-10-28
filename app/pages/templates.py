"""Read-only default-template inspection, independent of product/job workflows."""
from collections import Counter

import streamlit as st

from app.pages._inputs import selectbox
from app.templates.models import GeometryUnit, SlotRole
from app.templates.preview import geometry_svg
from app.templates.registry import get_facings, get_template, get_templates
from app.templates.validation import validate_template


def render():
    st.subheader("Templates")
    st.caption("Inspect blank display layouts. This viewer does not edit or save templates.")
    available = get_templates()
    names = {template.id: template.name for template in available}
    template_id = selectbox("Template", list(names), format_func=names.get, key="template_viewer_template")
    template = get_template(template_id)
    facings = get_facings(template_id)
    facing_names = {facing.id: facing.name for facing in facings}
    facing_id = selectbox("Facing", list(facing_names), format_func=facing_names.get,
                          key=f"template_viewer_facing:{template.id}")
    facing = next(facing for facing in facings if facing.id == facing_id)

    st.caption(f"Version {template.version} · " + ("System template (locked)" if template.locked else "Versioned baseline"))
    if template.unit == GeometryUnit.NORMALIZED:
        st.info("Structure only. Physical dimensions, spacing and proportions are not calibrated. This is not a to-scale fixture drawing.")
    else:
        st.caption("Dimensions are in inches. The preview is scaled to fit the screen.")

    product_count = sum(slot.role == SlotRole.PRODUCT for slot in facing.slots)
    total_products = sum(slot.role == SlotRole.PRODUCT for item in facings for slot in item.slots)
    first, second, third = st.columns(3)
    first.metric("Product positions on this facing", product_count)
    second.metric("Facings", len(facings))
    third.metric("Total product positions", total_products)
    st.write(facing.metadata.get("structure", f"{len(facing.slots)} positions"))

    validation = validate_template(template)
    for issue in validation.errors:
        st.error(issue.message)
    for issue in validation.warnings:
        st.warning(issue.message)
    if validation.valid:
        st.caption("Structure checks passed: identifiers, order, bounds and overlaps.")
        st.image(geometry_svg(template, facing.id), width="stretch")
        st.caption("Numbers show the explicit slot order. Dashed outlines mark sections where applicable.")

    with st.expander("Advanced / Geometry details"):
        st.write({
            "Template ID": template.id, "Family": template.template_family,
            "Version": template.version, "Unit": template.unit.value,
            "Origin": "Top-left; X right, Y down", "Facing ID": facing.id,
            "Facing width": facing.width, "Facing height": facing.height,
            "Roles on this facing": dict(Counter(SlotRole(slot.role).value for slot in facing.slots)),
        })
        for label, value in template.metadata.items():
            st.write(f"{label.replace('_', ' ').capitalize()}: {value}")
        st.dataframe([
            {"Order": slot.order, "Slot ID": slot.id, "Role": SlotRole(slot.role).value,
             "Row": slot.row, "Column": slot.column, "Section": slot.section_id,
             "X": slot.x, "Y": slot.y, "Width": slot.width, "Height": slot.height}
            for slot in facing.ordered_slots()
        ], hide_index=True, width="stretch")
