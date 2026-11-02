"""Guided session-only geometry builder; never writes templates or invokes production."""
import re
from xml.etree import ElementTree as ET

import streamlit as st

from app.pages import _builder_actions as actions
from app.pages._canvas_events import SELECTION_EPOCH_KEY
from app.pages._template_canvas import render_canvas

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
                            format_func=format_func, key=f"{prefix}:select_{kind}:{st.session_state.get(SELECTION_EPOCH_KEY, 0)}")
    if kind == "facing" and current != selected:
        selection = {}
    selection[kind] = selected
    st.session_state[SELECTION_KEY] = selection
    return next(item for item in items if item.id == selected)


def _role_hint(prefix, role=SlotRole.PRODUCT, hint=None):
    role = st.selectbox("Section type", list(SlotRole), index=list(SlotRole).index(role) if role in list(SlotRole) else 0,
                        format_func=lambda item: item.value.replace("_", " ").title(), key=f"{prefix}:role")
    hints = [None, *PresentationHint]
    hint = st.selectbox("Planogram grouping preference", hints, index=hints.index(hint) if hint in hints else 0,
                        format_func=lambda item: "No preference" if item is None else item.value.replace("_", " ").title(),
                        key=f"{prefix}:hint")
    st.caption("Optional settings for how this section may be used later. They do not rearrange its boxes.")
    return role, hint


def _geometry(prefix, x, y, width, height):
    first, second = st.columns(2)
    width = first.number_input("Box width", min_value=0.0, value=max(0.0, float(width)), format="%.4f", key=f"{prefix}:width")
    height = second.number_input("Box height", min_value=0.0, value=max(0.0, float(height)), format="%.4f", key=f"{prefix}:height")
    x = first.number_input("X (from left)", min_value=0.0, value=max(0.0, float(x)), format="%.4f", key=f"{prefix}:x")
    y = second.number_input("Y (from top)", min_value=0.0, value=max(0.0, float(y)), format="%.4f", key=f"{prefix}:y")
    return x, y, width, height


def _basics():
    st.markdown("### 1. Name your template")
    if st.session_state.get(DRAFT_KEY) is not None:
        st.info("Starting a new draft replaces the current session draft.")
    with st.form("template_builder:basics"):
        name = st.text_input("Template name", "Untitled Display")
        with st.expander("Advanced template settings"):
            family = st.text_input("Family / display type", "Custom")
            version = st.text_input("Version", "1.0")
            unit = st.selectbox("Measurement units", list(GeometryUnit), index=1,
                                format_func=lambda value: "Measured in inches" if value == GeometryUnit.INCHES else "Layout sketch (unmeasured)")
            st.caption("Units stay fixed for the draft and are never converted automatically.")
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
    with st.expander("2. Sides", expanded=not draft.template.facings):
        with st.form(f"{prefix}:add_facing"):
            name = st.text_input("New side name", f"Side {len(draft.template.facings) + 1}")
            default = 0.0 if draft.template.unit == GeometryUnit.INCHES else 1.0
            with st.expander("Advanced side settings"):
                width = st.number_input("New side width", min_value=0.0, value=default, format="%.4f")
                height = st.number_input("New side height", min_value=0.0, value=default, format="%.4f")
            if draft.template.unit == GeometryUnit.INCHES:
                st.caption("Enter measured width and height in Advanced side settings before adding this side.")
            added = st.form_submit_button("Add Side")
        if added:
            _apply(lambda current: edits.add_facing(current, name, width, height), "facing")
        if not draft.template.facings:
            return None
        facings = draft.template.ordered_facings()
        names = {item.id: item.name for item in facings}
        facing = _choose("Side", facings, prefix, "facing", names.get)
        with st.expander("Side options"):
            with st.form(f"{prefix}:{facing.id}:edit_facing"):
                name = st.text_input("Side name", facing.name)
                with st.expander("Advanced side settings"):
                    order = st.number_input("Side order", min_value=0, value=max(0, facing.order), step=1)
                    width = st.number_input("Side width", min_value=0.0, value=max(0.0, float(facing.width)), format="%.4f")
                    height = st.number_input("Side height", min_value=0.0, value=max(0.0, float(facing.height)), format="%.4f")
                applied = st.form_submit_button("Apply side changes")
            if applied:
                _apply(lambda current: edits.update_facing(current, facing.id, name=name, width=width, height=height, order=order))
            st.caption("Deleting this side also deletes its sections and boxes from the draft.")
            st.button("Delete Side", key=f"{prefix}:remove_facing", on_click=_apply,
                      args=(lambda current: edits.remove_facing(current, facing.id),), kwargs={"rerun": False})
        return facing


def _regions(draft, facing, prefix):
    with st.expander("3. Sections", expanded=not facing.regions):
        with st.form(f"{prefix}:add_region"):
            name = st.text_input("New section name", "Main Cards")
            with st.expander("Advanced section settings"):
                role, hint = _role_hint(f"{prefix}:add_region")
            added = st.form_submit_button("Add Section")
        if added:
            _apply(lambda current: edits.add_region(current, facing.id, name, role, hint), "region")
        if not facing.regions:
            return None
        regions = facing.ordered_regions()
        names = {item.id: item.name for item in regions}
        region = _choose("Section", regions, prefix, "region", names.get)
        with st.expander("Section options"):
            with st.form(f"{prefix}:{region.id}:edit_region"):
                name = st.text_input("Section name", region.name)
                with st.expander("Advanced section settings"):
                    order = st.number_input("Section order", min_value=0, value=max(0, region.order), step=1)
                    role, hint = _role_hint(f"{prefix}:{region.id}:edit_region", region.role, region.presentation_hint)
                applied = st.form_submit_button("Apply section changes")
            if applied:
                _apply(lambda current: edits.update_region(current, facing.id, region.id, name=name, role=role, hint=hint, order=order))
            st.caption("Deleting this section also deletes boxes that belong only to it.")
            st.button("Delete Section", key=f"{prefix}:remove_region", on_click=_apply,
                      args=(lambda current: edits.remove_region(current, facing.id, region.id),), kwargs={"rerun": False})
        return region


def _grid(draft, facing, region, prefix):
    with st.expander("Create a grid", expanded=not region.slot_ids):
        with st.form(f"{prefix}:grid"):
            left, right = st.columns(2)
            rows = left.number_input("Rows", min_value=1, max_value=50, value=2, step=1)
            columns = right.number_input("Columns", min_value=1, max_value=50, value=3, step=1)
            try:
                defaults = actions.grid_defaults(facing, region.id, rows, columns)
            except ValueError:
                defaults = dict(x=0.0, y=0.0, width=facing.width * .9, height=facing.height * .4,
                                horizontal_gap=0.0, vertical_gap=0.0)
            with st.expander("Advanced grid settings"):
                custom = st.checkbox("Use custom grid settings", key=f"{prefix}:custom_grid")
                st.caption("Check this option to apply the measurements below instead of automatic sizing.")
                width = st.number_input("Total grid width", min_value=0.0, value=defaults["width"], format="%.4f")
                height = st.number_input("Total grid height", min_value=0.0, value=defaults["height"], format="%.4f")
                x = st.number_input("Starting X", min_value=0.0, value=defaults["x"], format="%.4f")
                y = st.number_input("Starting Y", min_value=0.0, value=defaults["y"], format="%.4f")
                horizontal_gap = st.number_input("Horizontal gap", min_value=0.0, value=defaults["horizontal_gap"], format="%.4f")
                vertical_gap = st.number_input("Vertical gap", min_value=0.0, value=defaults["vertical_gap"], format="%.4f")
            created = st.form_submit_button("Create Grid")
        if created:
            settings = dict(x=x, y=y, width=width, height=height, horizontal_gap=horizontal_gap, vertical_gap=vertical_gap)
            _apply(lambda current: edits.create_grid(current, facing.id, region.id, rows=rows, columns=columns,
                **(settings if custom else actions.grid_defaults(edits.get_facing(current, facing.id), region.id, rows, columns))), "slot")


def _edit_box(draft, facing, region, prefix):
    # Let validation explain bad references instead of crashing the controls.
    slots = tuple(slot for slot in facing.ordered_slots() if slot.id in region.slot_ids)
    if not slots:
        st.caption("No boxes in this section yet. Add a box or create a grid.")
        return
    labels = {slot.id: f"Box {slot.order}" for slot in slots}
    selected = _choose("Box", slots, prefix, "slot", labels.get)
    duplicate, delete = st.columns(2)
    duplicate.button("Duplicate Box", key=f"{prefix}:duplicate_box", on_click=_apply,
                     args=(lambda current: actions.duplicate_box(current, facing.id, region.id, selected.id), "slot"),
                     kwargs={"rerun": False})
    delete.button("Delete Box", key=f"{prefix}:remove_slot", on_click=_apply,
                  args=(lambda current: edits.remove_slot(current, facing.id, selected.id),), kwargs={"rerun": False})
    with st.expander("Move or resize", expanded=True):
        for row in ((("Move left", "left"), ("Move right", "right"), ("Move up", "up"), ("Move down", "down")),
                    (("Wider", "wider"), ("Narrower", "narrower"), ("Taller", "taller"), ("Shorter", "shorter"))):
            buttons = st.columns(4)
            for button, (label, action) in zip(buttons, row):
                button.button(label, key=f"{prefix}:{selected.id}:{action}", on_click=_apply,
                    args=(lambda current, action=action: actions.adjust_box(current, facing.id, region.id, selected.id, action),),
                    kwargs={"rerun": False})
    with st.expander("Advanced box geometry"):
        geometry_key = f"{prefix}:{selected.id}:edit_slot"
        with st.form(f"{geometry_key}:form"):
            _geometry(geometry_key, selected.x, selected.y, selected.width, selected.height)
            st.number_input("Box order", min_value=0, value=max(0, selected.order), step=1, key=f"{geometry_key}:order")
            regions = facing.ordered_regions()
            names = {item.id: item.name for item in regions}
            st.selectbox("Box section", list(names), format_func=names.get,
                         index=list(names).index(region.id), key=f"{geometry_key}:region")
            def update():
                values = {field: st.session_state[f"{geometry_key}:{field}"] for field in ("x", "y", "width", "height", "order")}
                target = st.session_state[f"{geometry_key}:region"]
                _apply(lambda current: edits.update_slot(current, facing.id, selected.id, region_id=target, **values), rerun=False)
            st.form_submit_button("Apply box changes", on_click=update)


def _positions(draft, facing, region, prefix):
    if facing.width <= 0 or facing.height <= 0:
        st.info("Set a positive side width and height in Advanced side settings before adding boxes.")
        return
    with st.expander("4. Boxes", expanded=True):
        st.caption(f"Working in section: {region.name}. New boxes are added here automatically.")
        st.button("Add Box", type="primary", key=f"{prefix}:add_box", on_click=_apply,
                  args=(lambda current: actions.add_box(current, facing.id, region.id), "slot"), kwargs={"rerun": False})
        _grid(draft, facing, region, prefix)
        _edit_box(draft, facing, region, prefix)


def _operator_svg(template, facing, *, allow_layout_errors=False):
    # Keep the shared geometry renderer unchanged; adapt only visible labels/tooltips.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(geometry_svg(template, facing.id, allow_layout_errors=allow_layout_errors))
    for node in root.iter():
        if node.text and node.text.startswith("Uncalibrated structure"):
            node.text = "Layout sketch - sizes are not measured"
        if node.get("data-slot-id"):
            slot = next(slot for slot in facing.slots if slot.id == node.get("data-slot-id"))
            for child in node:
                if child.tag.endswith("title"):
                    child.text = f"Box {slot.order}"
        if node.get("data-region-id"):
            region = next(region for region in facing.regions if region.id == node.get("data-region-id"))
            for child in node:
                if child.tag.endswith("title"):
                    child.text = f"Section: {region.name}"
    return ET.tostring(root, encoding="unicode")


def _preview(draft, facing):
    st.subheader("5. Review")
    if not draft.template.facings:
        st.info("Add a side, then a section and boxes.")
        return
    validation = edits.validate_draft(draft)
    names = {item.id: item.name for item in draft.template.facings}
    for item in draft.template.facings:
        names.update({region.id: region.name for region in item.regions})
        names.update({slot.id: f"box {slot.order}" for slot in item.slots})
    for issue in validation.issues:
        message = issue.message
        for item_id in sorted(names, key=len, reverse=True):
            message = message.replace(item_id, names[item_id])
        for old, new in (("slot", "box"), ("region", "section"), ("facing", "side")):
            message = re.sub(rf"\b{old}s?\b", lambda match: new + ("es" if new == "box" and match[0].lower().endswith("s") else "s" if match[0].lower().endswith("s") else ""), message, flags=re.IGNORECASE)
        message = message.replace("presentation hint", "grouping preference")
        (st.error if issue.severity == "error" else st.warning)(message)
    if validation.valid:
        st.success("Layout checks passed.")
        if facing:
            st.caption(f"{facing.name} | Sections: {len(facing.regions)} | Boxes: {len(facing.slots)}")
            with st.expander("Advanced static preview"):
                st.image(_operator_svg(draft.template, facing), width="content")
            st.caption("Section colors and box numbers help you review the layout. Empty sections gain outlines when boxes are added.")
    else:
        st.caption("Fix the layout checks using the canvas, move/resize buttons or Advanced settings.")
    if facing:
        try:
            svg = _operator_svg(draft.template, facing, allow_layout_errors=True)
        except ValueError:
            st.caption("Use Advanced settings to repair invalid dimensions or section references before canvas editing.")
        else:
            st.caption("Click a box to select it. Drag to move; drag its handles to resize. Duplicate and Delete act on the selected box.")
            render_canvas(svg, draft, facing, draft_key=DRAFT_KEY, selection_key=SELECTION_KEY)
    def done():
        st.session_state[MODE_KEY] = "viewer"
    st.button("Done", type="primary", disabled=not validation.valid, on_click=done)
    st.caption("Done returns to Templates. Your draft stays in this session; it is not permanently saved.")


def _draft_details(draft, facing):
    with st.expander("Advanced draft details"):
        st.write({"Template ID": draft.template.id, "Unit": draft.template.unit.value,
                  "Revision": draft.revision, "Validation errors": len(edits.validate_draft(draft).errors)})
        if facing:
            st.dataframe([{"ID": slot.id, "Order": slot.order, "X": slot.x, "Y": slot.y,
                           "Width": slot.width, "Height": slot.height,
                           "Sections": ", ".join(region.name for region in facing.regions_for_slot(slot.id))}
                          for slot in facing.ordered_slots()], hide_index=True, width="stretch")
        st.caption("Reset draft clears the layout but keeps its name and setup. Clear draft starts over.")
        reset, clear = st.columns(2)
        def reset_action():
            st.session_state[NEW_KEY] = False
            st.session_state[SELECTION_KEY] = {}
            _apply(edits.reset_draft, rerun=False)
        def clear_action():
            st.session_state.pop(DRAFT_KEY, None)
            st.session_state.pop(SELECTION_KEY, None)
            st.session_state[NEW_KEY] = True
        reset.button("Reset draft", on_click=reset_action)
        clear.button("Clear draft", on_click=clear_action)


def render():
    st.subheader("Template Builder")
    st.caption("Name your template, add sides and sections, then add and arrange boxes. Drafts last for this session only.")
    draft = st.session_state.get(DRAFT_KEY)
    if st.button("Back to templates"):
        st.session_state[MODE_KEY] = "viewer"
        st.rerun()
    if draft is None or st.session_state.get(NEW_KEY):
        _basics()
        return
    template = draft.template
    st.write(template.name)
    if template.unit == GeometryUnit.NORMALIZED:
        st.info("Layout sketch: sizes and proportions are not approved physical measurements.")
    else:
        st.info("Measurements are in inches. Use the actual side dimensions in Advanced side settings.")
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
    _draft_details(draft, facing)
