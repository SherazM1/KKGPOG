"""Read-only default-template inspection, independent of product/job workflows."""
from collections import Counter

import streamlit as st

from app.pages._inputs import selectbox
from app.templates.models import GeometryUnit, PresentationHint, SlotRole
from app.templates.preview import geometry_svg
from app.templates.registry import get_facings, get_template, get_templates
from app.templates.validation import validate_template


def render():
    from app.pages import template_builder
    if st.session_state.get(template_builder.MODE_KEY) == "builder":
        template_builder.render()
        return
    st.subheader("Templates")
    if st.button("Create New Template", key="template_builder:create"):
        st.session_state[template_builder.MODE_KEY] = "builder"
        st.session_state[template_builder.NEW_KEY] = True
        st.rerun()
    if st.session_state.get(template_builder.DRAFT_KEY) is not None:
        if st.button("Resume draft", key="template_builder:resume"):
            st.session_state[template_builder.MODE_KEY] = "builder"
            st.session_state[template_builder.NEW_KEY] = False
            st.rerun()
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
    if "count_assumptions" in template.metadata:
        st.caption(template.metadata["count_assumptions"])

    validation = validate_template(template)
    for issue in validation.errors:
        st.error(issue.message)
    for issue in validation.warnings:
        st.warning(issue.message)
    if validation.valid:
        st.caption("Structure checks passed: identifiers, order, bounds and overlaps.")
        st.image(geometry_svg(template, facing.id), width="content")
        st.caption("Numbers show slot order. Colored outlines are region envelopes; dashed section outlines appear where applicable.")
        if facing.regions:
            hints = {
                PresentationHint.GRID: "Grid", PresentationHint.SINGLE_ROW: "Single row",
                PresentationHint.CUSTOM_ROWS: "Custom rows", PresentationHint.SECTION: "Grouped section",
                PresentationHint.PRESERVE_RELATIVE: "Keep relative positions",
            }
            st.caption("Suggested grouping describes a future planogram arrangement. It does not move the physical boxes shown above.")
            st.table([
                {"Region": region.name, "Positions": len(region.slot_ids),
                 "Suggested grouping": hints.get(region.presentation_hint, "Not specified")}
                for region in facing.ordered_regions()
            ])

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
             "Regions": ", ".join(region.id for region in facing.regions_for_slot(slot.id)),
             "X": slot.x, "Y": slot.y, "Width": slot.width, "Height": slot.height}
            for slot in facing.ordered_slots()
        ], hide_index=True, width="stretch")
        if facing.regions:
            st.dataframe([
                {"Region ID": region.id, "Order": region.order, "Role": SlotRole(region.role).value,
                 "Presentation hint": region.presentation_hint.value if region.presentation_hint else None,
                 "Envelope": str(region.bounds) if region.bounds else "Derived from members"}
                for region in facing.ordered_regions()
            ], hide_index=True, width="stretch")
