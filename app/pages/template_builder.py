"""Guided session-only geometry builder; never writes templates or invokes production."""
import streamlit as st

from app.templates import draft as edits
from app.templates.models import GeometryUnit, PresentationHint, SlotRole
from app.templates.preview import geometry_svg


DRAFT_KEY = "template_builder:draft"
MODE_KEY = "template_builder:mode"
NEW_KEY = "template_builder:new"
SELECTION_KEY = "template_builder:selection"


def _apply(action, pick=None, rerun=True):
    before = st.session_state[DRAFT_KEY]
    try:
        updated = action(before)
    except (ValueError, KeyError) as exc:
        st.error(str(exc).strip("'"))
        return
    selection = dict(st.session_state.get(SELECTION_KEY, {}))
    if pick == "facing":
        selection = {"facing": updated.template.facings[-1].id}
    elif pick:
        facing = edits.get_facing(updated, selection["facing"])
        selection[pick] = (facing.regions if pick == "region" else facing.slots)[-1].id
    st.session_state[DRAFT_KEY] = updated
    st.session_state[SELECTION_KEY] = selection
    if rerun:
        st.rerun()


def _choose(label, items, prefix, kind, format_func):
    selection = dict(st.session_state.get(SELECTION_KEY, {}))
    ids = [item.id for item in items]
    current = selection.get(kind)
    selected = st.selectbox(label, ids, index=ids.index(current) if current in ids else 0,
                            format_func=format_func, key=f"{prefix}:select_{kind}")
    if kind == "facing" and current != selected:
        selection = {}
    selection[kind] = selected
    st.session_state[SELECTION_KEY] = selection
    return next(item for item in items if item.id == selected)


def _role_hint(prefix, role=SlotRole.PRODUCT, hint=None):
    role = st.selectbox("Region type", list(SlotRole), index=list(SlotRole).index(role) if role in list(SlotRole) else 0,
                        format_func=lambda item: item.value.replace("_", " ").title(), key=f"{prefix}:role")
    hints = [None, *PresentationHint]
    hint = st.selectbox("Suggested planogram grouping", hints, index=hints.index(hint) if hint in hints else 0,
                        format_func=lambda item: "No preference" if item is None else item.value.replace("_", " ").title(),
                        key=f"{prefix}:hint")
    return role, hint


def _geometry(prefix, x, y, width, height):
    first, second = st.columns(2)
    width = first.number_input("Position width", min_value=0.0, value=max(0.0, float(width)), format="%.4f", key=f"{prefix}:width")
    height = second.number_input("Position height", min_value=0.0, value=max(0.0, float(height)), format="%.4f", key=f"{prefix}:height")
    with st.expander("Placement from the top-left"):
        x = st.number_input("Distance from left", min_value=0.0, value=max(0.0, float(x)), format="%.4f", key=f"{prefix}:x")
        y = st.number_input("Distance from top", min_value=0.0, value=max(0.0, float(y)), format="%.4f", key=f"{prefix}:y")
    return x, y, width, height


def _basics():
    if st.session_state.get(DRAFT_KEY) is not None:
        st.info("Starting a new draft replaces the current session draft.")
    with st.form("template_builder:basics"):
        name = st.text_input("Template name", "Untitled Display")
        family = st.text_input("Family / display type", "Custom")
        version = st.text_input("Version", "1.0")
        unit = st.selectbox("Geometry unit", list(GeometryUnit), index=1,
                            format_func=lambda value: "Inches" if value == GeometryUnit.INCHES else "Uncalibrated (normalized)")
        st.caption("The chosen unit applies to the whole draft. Units are never converted automatically.")
        submitted = st.form_submit_button("Start blank draft", type="primary")
    if submitted:
        try:
            st.session_state[DRAFT_KEY] = edits.new_draft(name, family, version, unit)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state[NEW_KEY] = False
        st.session_state[SELECTION_KEY] = {}
        st.rerun()


def _facings(draft, prefix):
    with st.expander("1. Facings", expanded=not draft.template.facings):
        with st.form(f"{prefix}:add_facing"):
            st.write("Add a facing")
            name = st.text_input("New facing name", f"Facing {len(draft.template.facings) + 1}")
            default = 0.0 if draft.template.unit == GeometryUnit.INCHES else 1.0
            width = st.number_input("New facing width", min_value=0.0, value=default, format="%.4f")
            height = st.number_input("New facing height", min_value=0.0, value=default, format="%.4f")
            if draft.template.unit == GeometryUnit.INCHES:
                st.caption("Enter the measured physical width and height in inches.")
            added = st.form_submit_button("Add facing")
        if added:
            _apply(lambda current: edits.add_facing(current, name, width, height), "facing")
        if not draft.template.facings:
            return None
        facings = draft.template.ordered_facings()
        names = {item.id: item.name for item in facings}
        facing = _choose("Facing to work on", facings, prefix, "facing", names.get)
        with st.form(f"{prefix}:{facing.id}:edit_facing"):
            name = st.text_input("Facing name", facing.name)
            order = st.number_input("Facing order", min_value=0, value=max(0, facing.order), step=1)
            width = st.number_input("Facing width", min_value=0.0, value=max(0.0, float(facing.width)), format="%.4f")
            height = st.number_input("Facing height", min_value=0.0, value=max(0.0, float(facing.height)), format="%.4f")
            applied = st.form_submit_button("Apply facing changes")
        if applied:
            _apply(lambda current: edits.update_facing(current, facing.id, name=name, width=width, height=height, order=order))
        st.caption("Removing a facing also removes its regions and positions from this draft.")
        st.button("Remove facing", key=f"{prefix}:remove_facing", on_click=_apply,
                  args=(lambda current: edits.remove_facing(current, facing.id),), kwargs={"rerun": False})
        return facing


def _regions(draft, facing, prefix):
    with st.expander("2. Regions", expanded=not facing.regions):
        st.caption("Regions group physical positions. Suggested grouping does not move their geometry.")
        with st.form(f"{prefix}:add_region"):
            name = st.text_input("New region name", "Main area")
            role, hint = _role_hint(f"{prefix}:add_region")
            added = st.form_submit_button("Add region")
        if added:
            _apply(lambda current: edits.add_region(current, facing.id, name, role, hint), "region")
        if not facing.regions:
            return None
        regions = facing.ordered_regions()
        names = {item.id: item.name for item in regions}
        region = _choose("Region to work on", regions, prefix, "region", names.get)
        with st.form(f"{prefix}:{region.id}:edit_region"):
            name = st.text_input("Region name", region.name)
            order = st.number_input("Region order", min_value=0, value=max(0, region.order), step=1)
            role, hint = _role_hint(f"{prefix}:{region.id}:edit_region", region.role, region.presentation_hint)
            applied = st.form_submit_button("Apply region changes")
        if applied:
            _apply(lambda current: edits.update_region(current, facing.id, region.id, name=name, role=role, hint=hint, order=order))
        st.caption("Removing a region also removes positions that belong only to it.")
        st.button("Remove region", key=f"{prefix}:remove_region", on_click=_apply,
                  args=(lambda current: edits.remove_region(current, facing.id, region.id),), kwargs={"rerun": False})
        return region


def _positions(draft, facing, region, prefix):
    if facing.width <= 0 or facing.height <= 0:
        st.info("Set positive facing dimensions before adding or editing positions.")
        return
    with st.expander("3. Positions", expanded=True):
        st.caption(f"Adding positions to {region.name}. Apply each form to update the preview.")
        single, grid, edit = st.tabs(["Add position", "Create grid", "Edit position"])
        with single, st.form(f"{prefix}:add_slot"):
            values = _geometry(f"{prefix}:add_slot", *edits.suggest_slot(facing))
            added = st.form_submit_button("Add position")
        if added:
            _apply(lambda current: edits.add_slot(current, facing.id, region.id,
                                                 **dict(zip(("x", "y", "width", "height"), values))), "slot")
        with grid, st.form(f"{prefix}:grid"):
            rows = st.number_input("Rows", min_value=1, max_value=50, value=2, step=1)
            columns = st.number_input("Positions per row", min_value=1, max_value=50, value=3, step=1)
            width = st.number_input("Overall grid width", min_value=0.0, value=facing.width * .9, format="%.4f")
            height = st.number_input("Overall grid height", min_value=0.0, value=facing.height * .4, format="%.4f")
            horizontal_gap = st.number_input("Space between columns", min_value=0.0, value=facing.width * .02, format="%.4f")
            vertical_gap = st.number_input("Space between rows", min_value=0.0, value=facing.height * .02, format="%.4f")
            with st.expander("Grid start from the top-left"):
                x = st.number_input("Grid distance from left", min_value=0.0, value=facing.width * .05, format="%.4f")
                y = st.number_input("Grid distance from top", min_value=0.0, value=facing.height * .05, format="%.4f")
            created = st.form_submit_button("Create grid")
        if created:
            _apply(lambda current: edits.create_grid(current, facing.id, region.id, rows=rows, columns=columns,
                                                     x=x, y=y, width=width, height=height,
                                                     horizontal_gap=horizontal_gap, vertical_gap=vertical_gap), "slot")
        with edit:
            if not facing.slots:
                st.caption("Add a position or grid first.")
                return
            slots = facing.ordered_slots()
            labels = {slot.id: f"Position {slot.order}" for slot in slots}
            selected = _choose("Position to edit", slots, prefix, "slot", labels.get)
            with st.form(f"{prefix}:{selected.id}:edit_slot"):
                values = _geometry(f"{prefix}:{selected.id}:edit_slot", selected.x, selected.y, selected.width, selected.height)
                order = st.number_input("Position order", min_value=0, value=max(0, selected.order), step=1)
                regions = facing.ordered_regions()
                region_ids = [item.id for item in regions]
                names = {item.id: item.name for item in regions}
                members = facing.regions_for_slot(selected.id)
                target = st.selectbox("Belongs to region", region_ids, format_func=names.get,
                                      index=region_ids.index(members[0].id) if members else 0)
                applied = st.form_submit_button("Apply position changes")
            if applied:
                _apply(lambda current: edits.update_slot(current, facing.id, selected.id, order=order, region_id=target,
                                                         **dict(zip(("x", "y", "width", "height"), values))))
            st.button("Remove position", key=f"{prefix}:remove_slot", on_click=_apply,
                      args=(lambda current: edits.remove_slot(current, facing.id, selected.id),), kwargs={"rerun": False})


def _preview(draft, facing):
    st.subheader("Draft preview")
    if not draft.template.facings:
        st.info("Add your first facing, then add a region and positions.")
        return
    validation = edits.validate_draft(draft)
    names = {item.id: item.name for item in draft.template.facings}
    for item in draft.template.facings:
        names.update({region.id: region.name for region in item.regions})
        names.update({slot.id: f"position {slot.order}" for slot in item.slots})
    for issue in validation.issues:
        message = issue.message
        for item_id in sorted(names, key=len, reverse=True):
            message = message.replace(item_id, names[item_id])
        (st.error if issue.severity == "error" else st.warning)(message)
    if validation.valid:
        st.success("Draft geometry checks passed.")
        if facing:
            st.image(geometry_svg(draft.template, facing.id), width="content")
            st.caption("Empty regions have an outline once positions are added. Numbers show position order.")
    else:
        st.caption("Fix the geometry checks to show the preview.")
    with st.expander("Advanced / Draft details"):
        st.write({"Template ID": draft.template.id, "Unit": draft.template.unit.value,
                  "Revision": draft.revision, "Validation errors": len(validation.errors)})
        if facing:
            st.dataframe([{"ID": slot.id, "Order": slot.order, "X": slot.x, "Y": slot.y,
                           "Width": slot.width, "Height": slot.height,
                           "Regions": ", ".join(region.name for region in facing.regions_for_slot(slot.id))}
                          for slot in facing.ordered_slots()], hide_index=True, width="stretch")


def render():
    st.subheader("Template Builder")
    st.caption("Build a draft for this session. Nothing is permanently saved or sent to planogram generation.")
    draft = st.session_state.get(DRAFT_KEY)
    back, reset, clear = st.columns(3)
    if back.button("Back to templates"):
        st.session_state[MODE_KEY] = "viewer"
        st.rerun()
    if draft is not None:
        if reset.button("Reset draft"):
            st.session_state[NEW_KEY] = False
            st.session_state[SELECTION_KEY] = {}
            _apply(edits.reset_draft)
        if clear.button("Clear draft"):
            st.session_state.pop(DRAFT_KEY, None)
            st.session_state.pop(SELECTION_KEY, None)
            st.session_state[NEW_KEY] = True
            st.rerun()
    if draft is None or st.session_state.get(NEW_KEY):
        _basics()
        return
    template = draft.template
    st.write(f"{template.name} · {template.template_family} · Version {template.version}")
    if template.unit == GeometryUnit.NORMALIZED:
        st.info("Uncalibrated draft. Dimensions and proportions are illustrative, not approved physical measurements.")
    else:
        st.info("All dimensions are in inches. Use measured facing dimensions; the draft does not verify measurements.")
    st.caption("Reset draft clears the layout and keeps its basics. Clear draft starts over. Position IDs stay fixed.")
    prefix = f"builder:{template.id}:{draft.revision}"
    controls, preview = st.columns([1.1, 1])
    with controls:
        facing = _facings(draft, prefix)
        if facing:
            region = _regions(draft, facing, f"{prefix}:{facing.id}")
            if region:
                _positions(draft, facing, region, f"{prefix}:{facing.id}")
    with preview:
        _preview(draft, facing)
