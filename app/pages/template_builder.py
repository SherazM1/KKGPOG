"""Guided session-only geometry builder; never writes templates or invokes production."""
import re
from dataclasses import replace
from datetime import datetime
from functools import partial
from hashlib import sha256
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import streamlit as st

from app.pages import _builder_actions as actions
from app.pages import _direct_builder as direct
from app.pages._direct_canvas import canvas_svg
from app.pages._canvas_events import ACTIVE_SIDE_KEY, SELECTION_EPOCH_KEY
from app.pages._template_canvas import render_canvas

from app.templates import draft as edits
from app.templates.models import GeometryUnit, PresentationHint, SlotRole
from app.templates.preview import geometry_svg
from app.templates.print_style import (DEFAULT_HEADER_COLOR, PDF_IDENTIFIER_FIELDS,
                                       normalize_text_fields, section_header_color,
                                       section_preserves_spacing)


DRAFT_KEY = "template_builder:draft"
MODE_KEY = "template_builder:mode"
NEW_KEY = "template_builder:new"
SELECTION_KEY = "template_builder:selection"
PDF_KEY = "template_builder:pdf_export"
PDF_READY_KEY = "template_builder:pdf_ready"
PDF_FIELDS_KEY = "template_builder:pdf_text_fields"
PDF_LAYOUT_VERSION = 25


def side_selection_key(facing_id):
    return f"{SELECTION_KEY}:{facing_id}"


def _apply(action, pick=None, rerun=True, *, facing_id=None):
    before = st.session_state[DRAFT_KEY]
    try:
        updated = action(before)
    except (ValueError, KeyError) as exc:
        st.error(str(exc).strip("'"))
        return
    key = side_selection_key(facing_id) if facing_id else SELECTION_KEY
    selection = dict(st.session_state.get(key, {}))
    if pick == "facing":
        facing_id = updated.template.facings[-1].id
        st.session_state[ACTIVE_SIDE_KEY] = facing_id
        key = side_selection_key(facing_id)
        selection = {"facing": facing_id}
    elif pick == "clear_boxes":
        selection.pop("slot", None)
        selection.pop("slots", None)
        st.session_state[SELECTION_EPOCH_KEY] = (
            st.session_state.get(SELECTION_EPOCH_KEY, 0) + 1
        )
    elif pick:
        facing = edits.get_facing(updated, facing_id)
        selection[pick] = (facing.regions if pick == "region" else facing.slots)[-1].id
        if pick == "region":
            selection.pop("slot", None)
            selection.pop("slots", None)
        else:
            selection['slots'] = (selection['slot'],)
    if facing_id and any(f.id==facing_id for f in updated.template.facings) and 'slots' in selection:
        valid = {slot.id for slot in edits.get_facing(updated,facing_id).slots}
        selection['slots'] = tuple(sid for sid in selection['slots'] if sid in valid)
    st.session_state[DRAFT_KEY] = updated
    st.session_state[key] = selection
    sides = updated.template.ordered_facings()
    if not sides:
        st.session_state.pop(ACTIVE_SIDE_KEY, None)
    elif st.session_state.get(ACTIVE_SIDE_KEY) not in {side.id for side in sides}:
        previous_ids = [side.id for side in before.template.ordered_facings()]
        previous = st.session_state.get(ACTIVE_SIDE_KEY)
        index = previous_ids.index(previous) if previous in previous_ids else 0
        st.session_state[ACTIVE_SIDE_KEY] = sides[min(index, len(sides) - 1)].id
    if not before.template.facings and updated.template.facings:
        # Change the sidebar preference once, leaving its manual toggle usable.
        st.session_state["app:sidebar_state"] = "collapsed"
    if rerun:
        st.rerun()


def _choose(label, items, prefix, kind, format_func, *, facing_id):
    key = side_selection_key(facing_id)
    selection = dict(st.session_state.get(key, {}))
    ids = [item.id for item in items]
    current = selection.get(kind)
    
    index = ids.index(current) if current in ids else 0
    picker_key = f"{prefix}:select_{kind}:{st.session_state.get(SELECTION_EPOCH_KEY, 0)}"

    def choose_box():
        chosen = st.session_state[picker_key]
        updated_selection = dict(st.session_state.get(key, {}))
        updated_selection[kind] = chosen
        if kind == "slot":
            updated_selection["slots"] = (chosen,) if chosen else ()
        else:
            updated_selection.pop("slots", None)
        st.session_state[key] = updated_selection

    selected = st.selectbox(
        label,
        ids,
        index=index,
        placeholder=f"Select a {label.lower()}",
        format_func=format_func,
        key=picker_key,
        on_change=choose_box,
    )
    # Default the editor to the first box without selecting it on the canvas.
    # Explicit dropdown changes and canvas clicks still select a box normally.
    return next((item for item in items if item.id == selected), None)


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
    with st.form("template_builder:basics", border=False):
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
        st.session_state.pop(ACTIVE_SIDE_KEY, None)
        st.rerun()


def _side_letter(number):
    label = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _next_side_name(draft):
    used = {side.name.strip().casefold() for side in draft.template.facings}
    number = 1
    while f"side {_side_letter(number)}".casefold() in used:
        number += 1
    return f"Side {_side_letter(number)}"


def _add_side(draft):
    prefix = f"builder:{draft.template.id}:add_facing_letters:{len(draft.template.facings)}"
    with st.container():
        st.markdown("**Add a side**")
        with st.form(prefix, border=False):
            st.text_input("New side name", _next_side_name(draft), key=f"{prefix}:name")
            default = 0.0 if draft.template.unit == GeometryUnit.INCHES else 1.0
            with st.expander("Advanced side settings"):
                st.number_input("New side width", min_value=0.0, value=default, format="%.4f", key=f"{prefix}:width")
                st.number_input("New side height", min_value=0.0, value=default, format="%.4f", key=f"{prefix}:height")
            if draft.template.unit == GeometryUnit.INCHES:
                st.caption("Enter measured width and height in Advanced side settings before adding this side.")
            def add():
                values = {field: st.session_state[f"{prefix}:{field}"] for field in ("name", "width", "height")}
                _apply(lambda current: edits.add_facing(current, **values), "facing", rerun=False)
            st.form_submit_button("Add Side", on_click=add)


def _side_options(draft, facing, prefix):
    apply = partial(_apply, facing_id=facing.id)
    st.markdown("**Side options**")
    _add_side(draft)
    with st.container():
        with st.form(f"{prefix}:duplicate_side", border=False):
            st.text_input("Copy side name", f"{facing.name} Copy", key=f"{prefix}:copy_name")
            def duplicate():
                apply(lambda current: edits.duplicate_facing(current, facing.id,
                    st.session_state[f"{prefix}:copy_name"]), "facing", rerun=False)
            st.form_submit_button("Duplicate Side", on_click=duplicate)
    with st.container():
        with st.form(f"{prefix}:edit_facing", border=False):
            st.text_input("Side name", facing.name, key=f"{prefix}:side_name")
            with st.expander("Advanced side settings"):
                st.number_input("Side order", min_value=0, value=max(0, facing.order), step=1, key=f"{prefix}:side_order")
                st.number_input("Side width", min_value=0.0, value=max(0.0, float(facing.width)), format="%.4f", key=f"{prefix}:side_width")
                st.number_input("Side height", min_value=0.0, value=max(0.0, float(facing.height)), format="%.4f", key=f"{prefix}:side_height")
            def update():
                values = {field: st.session_state[f"{prefix}:side_{field}"] for field in ("name", "width", "height", "order")}
                apply(lambda current: edits.update_facing(current, facing.id, **values), rerun=False)
            st.form_submit_button("Apply side changes", on_click=update)
        st.caption("Deleting this side also deletes its sections and boxes from the draft.")
        st.button("Delete Side", key=f"{prefix}:remove_facing", on_click=apply,
                  args=(lambda current: edits.remove_facing(current, facing.id),), kwargs={"rerun": False})


def _select_region(facing_id, region_id):
    key = side_selection_key(facing_id)
    st.session_state[key] = dict(facing=facing_id, region=region_id)
    st.session_state[SELECTION_EPOCH_KEY] = st.session_state.get(SELECTION_EPOCH_KEY, 0) + 1


def _regions(draft, facing, prefix):
    apply = partial(_apply, facing_id=facing.id)
    key = side_selection_key(facing.id)
    selection = dict(st.session_state.get(key, {"facing": facing.id}))
    regions = facing.ordered_regions()
    selected_id = selection.get("region")
    if selected_id not in {region.id for region in regions}:
        selected_id = regions[0].id if regions else None
        selection = dict(facing=facing.id, region=selected_id)
    selected_region = next((region for region in regions if region.id == selected_id), None)
    if selected_region is None or selection.get("slot") not in selected_region.slot_ids:
        selection.pop("slot", None)
    st.markdown("**Sections**")
    with st.container(horizontal=True):
        for item in regions:
            st.button(item.name, key=f"{prefix}:section_chip:{item.id}",
                      type="primary" if item.id == selected_id else "secondary",
                      on_click=_select_region, args=(facing.id, item.id))
    st.session_state[key] = selection
    with st.container():
        with st.form(f"{prefix}:add_region", border=False):
            name = st.text_input("New section name", "Main Cards")
            header_color = st.color_picker("Section header color", DEFAULT_HEADER_COLOR,
                                           key=f"{prefix}:add_region:header_color",
                                           help="Colors the full-width section header band in the PDF.")
            preserve_spacing = st.checkbox("Keep vertical spacing in PDF", value=True,
                key=f"{prefix}:add_region:preserve_spacing",
                help="Keep your row and section gaps. Uncheck to use the gaps for taller boxes in this section.")
            with st.expander("Advanced section settings"):
                role, hint = _role_hint(f"{prefix}:add_region")
            added = st.form_submit_button("Add Section")
        if added:
            apply(lambda current: direct.add_section(current, facing.id, name, role, hint,
                header_color=header_color, preserve_spacing=preserve_spacing), "region")
    if not regions:
        st.caption("Add a section to this side, then add boxes or create a grid.")
        return None
    region = next(item for item in regions if item.id == selected_id)
    st.markdown(f"**Build: {region.name}**")
    _positions(draft, facing, region, f"{prefix}:{region.id}")

    st.divider()
    st.markdown("**Manage section**")
    with st.container():
        with st.form(f'{prefix}:{region.id}:duplicate_section', border=False):
            copy_key = f'{prefix}:{region.id}:copy_section_name'
            st.text_input('Copy section name',f'{region.name} Copy',key=copy_key)
            def duplicate_section():
                apply(lambda current: direct.duplicate_section(current,facing.id,region.id,
                    st.session_state[copy_key]),'region',rerun=False)
            st.form_submit_button('Create Section Copy',on_click=duplicate_section)
    with st.container():
        st.caption(f'Delete {region.name} and its boxes from {facing.name}?')
        st.button('Confirm Delete Section',key=f'{prefix}:{region.id}:remove_region',on_click=apply,
                  args=(lambda current: edits.remove_region(current,facing.id,region.id),),kwargs={'rerun':False})
    with st.container():
        with st.form(f"{prefix}:{region.id}:edit_region", border=False):
            name = st.text_input("Section name", region.name)
            header_color = st.color_picker("Section header color", section_header_color(region),
                                           key=f"{prefix}:{region.id}:header_color",
                                           help="Colors the full-width section header band in the PDF.")
            preserve_spacing = st.checkbox("Keep vertical spacing in PDF",
                value=section_preserves_spacing(region), key=f"{prefix}:{region.id}:preserve_spacing",
                help="Keep your row and section gaps. Uncheck to use the gaps for taller boxes in this section.")
            with st.expander("Advanced section settings"):
                order = st.number_input("Section order", min_value=0, value=max(0, region.order), step=1)
                role, hint = _role_hint(f"{prefix}:{region.id}:edit_region", region.role, region.presentation_hint)
            applied = st.form_submit_button("Apply section changes")
        if applied:
            apply(lambda current: edits.update_region(current, facing.id, region.id, name=name,
                                                      role=role, hint=hint, order=order,
                                                      header_color=header_color,
                                                      preserve_spacing=preserve_spacing))
    return region


def _grid(draft, facing, region, prefix):
    apply = partial(_apply, facing_id=facing.id)
    with st.container():
        with st.form(f"{prefix}:grid", border=False):
            left, right = st.columns(2)
            rows = left.number_input("Rows", min_value=1, max_value=50, value=2, step=1)
            columns = right.number_input("Columns", min_value=1, max_value=50, value=3, step=1)
            try:
                defaults = actions.grid_defaults(facing, region.id, rows, columns)
            except ValueError:
                defaults = dict(x=0.0, y=0.0, width=facing.width * .9, height=facing.height * .4,
                                horizontal_gap=0.0, vertical_gap=0.0)
            with st.expander("Advanced grid settings"):
                st.caption('Technical compatibility only. Normal grids use the fixed box size and automatic spacing.')
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
            apply(lambda current: edits.create_grid(current, facing.id, region.id, rows=rows, columns=columns, **settings)
                if custom else direct.create_grid(current,facing.id,region.id,rows=rows,columns=columns), "clear_boxes",)


def _edit_box(draft, facing, region, prefix):
    apply = partial(_apply, facing_id=facing.id)
    # Let validation explain bad references instead of crashing the controls.
    slots = tuple(slot for slot in facing.ordered_slots() if slot.id in region.slot_ids)
    if not slots:
        st.caption("No boxes in this section yet. Add a box or create a grid.")
        return
    labels = {slot.id: f"Box {slot.order}" for slot in slots}
    selected = _choose("Box", slots, prefix, "slot", labels.get, facing_id=facing.id)
    
    if selected is None:
        st.caption("Select a box on the canvas or from the Box list to edit it.")
        return
    fields_key = f"{PDF_FIELDS_KEY}:{draft.template.id}"
    fields = normalize_text_fields(st.session_state.get(fields_key))
    text_key = f"{prefix}:{selected.id}:pdf_text:{draft.revision}"
    
    with st.expander("Text for this box"):
        with st.form(f"{text_key}:form", border=False):
            values = {}
            for field in fields:
                values[field] = st.text_input(
                    field,
                    value=selected.metadata.get(f"pdf_text:{field}", ""),
                    key=f"{text_key}:{field}",
                )
                
            saved = st.form_submit_button("Save box text")
            
        if saved:
            apply(lambda current: edits.set_slot_pdf_text(
                current, facing.id, selected.id, values
                )
            )
    duplicate, delete = st.columns(2)
    duplicate.button("Duplicate Box", key=f"{prefix}:duplicate_box", on_click=apply, width="stretch",
                     args=(lambda current: direct.duplicate_box(current, facing.id, region.id, selected.id), "slot"),
                     kwargs={"rerun": False})
    delete.button("Delete Box", key=f"{prefix}:remove_slot", on_click=apply, width="stretch",
                  args=(lambda current: edits.remove_slot(current, facing.id, selected.id),), kwargs={"rerun": False})
    with st.container():
        for row in (
            (("Move left", "left"), ("Move right", "right")),
            (("Move up", "up"), ("Move down", "down")),
        ):
            buttons = st.columns(2)
            for button, (label, action) in zip(buttons, row):
                button.button(label, key=f"{prefix}:{selected.id}:{action}", on_click=apply, width="stretch",
                    args=(lambda current, action=action: actions.adjust_box(current, facing.id, region.id, selected.id, action),),
                    kwargs={"rerun": False})
    with st.expander("Advanced box geometry"):
        st.caption('Technical compatibility only. Normal boxes have a fixed size; use the canvas to arrange them.')
        geometry_key = f"{prefix}:{selected.id}:edit_slot"
        with st.form(f"{geometry_key}:form", border=False):
            _geometry(geometry_key, selected.x, selected.y, selected.width, selected.height)
            st.number_input("Box order", min_value=0, value=max(0, selected.order), step=1, key=f"{geometry_key}:order")
            regions = facing.ordered_regions()
            names = {item.id: item.name for item in regions}
            st.selectbox("Box section", list(names), format_func=names.get,
                         index=list(names).index(region.id), key=f"{geometry_key}:region")
            def update():
                values = {field: st.session_state[f"{geometry_key}:{field}"] for field in ("x", "y", "width", "height", "order")}
                target = st.session_state[f"{geometry_key}:region"]
                apply(lambda current: edits.update_slot(current, facing.id, selected.id, region_id=target, **values), rerun=False)
            st.form_submit_button("Apply box changes", on_click=update)


def _pdf_text_fields_picker(draft):
    fields_key = f"{PDF_FIELDS_KEY}:{draft.template.id}"
    picker_key = f"{fields_key}:third_line"
    def save_fields():
        st.session_state[fields_key] = normalize_text_fields(st.session_state[picker_key])
        st.session_state.pop(PDF_KEY, None)
    current = normalize_text_fields(st.session_state.get(fields_key))[-1]
    st.caption("Description and CPP are always printed on each PDF box.")
    selected = st.selectbox(
        "Third PDF text line", PDF_IDENTIFIER_FIELDS,
        index=PDF_IDENTIFIER_FIELDS.index(current),
        key=picker_key, on_change=save_fields,
        help="Choose UPC, ID#, or Item #. Applies to every PDF box in this template.",
    )
    st.session_state[fields_key] = normalize_text_fields(selected)


def _positions(draft, facing, region, prefix):
    apply = partial(_apply, facing_id=facing.id)
    if facing.width <= 0 or facing.height <= 0:
        st.info("Set a positive side width and height in Advanced side settings before adding boxes.")
        return
    _grid(draft, facing, region, prefix)

    st.button("Add Box", type="primary", key=f"{prefix}:add_box", on_click=apply,
              args=(lambda current: direct.add_box(current, facing.id, region.id), "slot"), kwargs={"rerun": False})

    _pdf_text_fields_picker(draft)
    with st.container():
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


def _readable_issue(issue, template):
    names = {item.id: item.name for item in template.facings}
    for item in template.facings:
        names.update({region.id: region.name for region in item.regions})
        names.update({slot.id: f"box {slot.order}" for slot in item.slots})
    message = issue.message
    for item_id in sorted(names, key=len, reverse=True):
        message = message.replace(item_id, names[item_id])
    for old, new in (("slot", "box"), ("region", "section"), ("facing", "side")):
        message = re.sub(rf"\b{old}s?\b", lambda match: new + ("es" if new == "box" and match[0].lower().endswith("s") else "s" if match[0].lower().endswith("s") else ""), message, flags=re.IGNORECASE)
    return message.replace("presentation hint", "grouping preference")


def _side_template(draft, facing):
    # One invalid Side must not hide another Side's canvas.
    return replace(draft.template, facings=(facing,))


def _preview(draft, facing):
    template = _side_template(draft, facing)
    validation = edits.validate_draft(replace(draft, template=template))
    spatial = {"product_overlap", "slot_bounds", "slot_region_bounds"}
    overlaps = {sid for issue in validation.errors if issue.code == "product_overlap" for sid in issue.slot_ids}
    outside = {sid for issue in validation.errors if issue.code in ("slot_bounds", "slot_region_bounds") for sid in issue.slot_ids}
    if overlaps:
        st.error(f"{len(overlaps)} boxes overlap. Drag them apart to separate them.")
    if outside:
        st.error(f"{len(outside)} box{'es are' if len(outside) != 1 else ' is'} outside this side or its section (beyond its bounds).")
    for issue in validation.errors:
        if issue.code not in spatial:
            st.error(_readable_issue(issue, template))
    if validation.valid:
        st.caption("Layout checks passed.")
    if validation.issues:
        with st.expander("Advanced validation details"):
            for issue in validation.issues:
                st.caption(f"{issue.code}: {issue.message}")
    st.caption(f"{facing.name} | Sections: {len(facing.regions)} | Boxes: {len(facing.slots)}")

    try:
        svg = canvas_svg(replace(draft,template=template), facing)
    except ValueError:
        st.caption("Use Advanced settings to repair invalid dimensions or section references before canvas editing.")
        svg = None
    render_canvas(svg, draft, facing, draft_key=DRAFT_KEY, selection_key=side_selection_key(facing.id))
    if svg:
        st.caption("Fixed-size boxes: drag to arrange. Drag empty space or use Select boxes to select a group.")
        st.caption("Scroll the page to see the full side. Layout sketches expand downward as you build.")


def render_pdf_export(draft):
    """Offer the current draft PDF here and after Done; invalidate it on edits."""
    completed = (draft.template.id, draft.revision)
    
    if st.session_state.get(PDF_READY_KEY) != completed:
        st.caption("Click Done in the builder to enable PDF export.")
        return
    
    fields_key = f"{PDF_FIELDS_KEY}:{draft.template.id}"
    text_fields = normalize_text_fields(st.session_state.get(fields_key))
    signature = (draft.template.id, draft.revision, PDF_LAYOUT_VERSION, text_fields)
    cached = st.session_state.get(PDF_KEY)
    if cached and cached['signature'] != signature:
        st.session_state.pop(PDF_KEY, None)
        cached = None
    valid = edits.validate_draft(draft).valid
    st.caption("Print each Side on its own portrait page, using each section's vertical-spacing setting.")
    if st.button("Create Template PDF", key="template_builder:create_pdf", disabled=not valid):
        from app.templates.render_pdf import render_template_pdf
        with st.spinner("Preparing template PDF..."):
            try:
                pdf = render_template_pdf(draft.template, text_fields=text_fields)
            except (ValueError, OSError) as exc:
                st.error(f"Unable to create the template PDF: {exc}")
            else:
                created = datetime.now(ZoneInfo("America/Chicago"))
                cached = dict(signature=signature, pdf=pdf,
                              download_stamp=created.strftime("%Y%m%d_%H%M%S"),
                              created_label=created.strftime("%b %d, %Y at %I:%M:%S %p"))
                st.session_state[PDF_KEY] = cached
    if cached and valid:
        filename = re.sub(r'[^\w.-]+', '_', draft.template.name).strip('._') or 'template'
        stamp = f"_{cached['download_stamp']}" if cached.get('download_stamp') else ''
        if cached.get('created_label'):
            st.caption(f"PDF prepared {cached['created_label']} (Central time).")
        st.download_button("Download Template PDF", cached['pdf'], file_name=f'{filename}_all_sides{stamp}.pdf',
                           mime='application/pdf', key="template_builder:download_pdf", on_click="ignore")
    if not valid:
        st.caption("Complete the side layouts and resolve layout checks to enable PDF export.")


def _review(draft):
    validation = edits.validate_draft(draft)
    if validation.valid:
        st.caption("All sides pass layout checks.")
    elif draft.template.facings:
        st.warning(f"{len(validation.errors)} layout checks need attention. Check the affected Side cards.")
    with st.expander("Review All Sides"):
        for facing in draft.template.ordered_facings():
            st.markdown(f"**{facing.name}**")
            try:
                st.image(_operator_svg(_side_template(draft, facing), facing), width="content")
            except ValueError:
                st.caption("Resolve this side's layout checks to show its static review.")
        for issue in validation.issues:
            if issue.facing_id is None or issue.code.startswith("duplicate_"):
                st.caption(_readable_issue(issue, draft.template))
    def done():
        st.session_state[PDF_READY_KEY] = (
            draft.template.id,
            draft.revision,
        )
        st.session_state[MODE_KEY] = "viewer"
    
    st.button(
        "Done",
        type="primary",
        disabled=not validation.valid,
        on_click=done,
    )

def _draft_details(draft):
    with st.expander("Advanced draft details"):
        st.write({"Template ID": draft.template.id, "Unit": draft.template.unit.value,
                  "Revision": draft.revision, "Validation errors": len(edits.validate_draft(draft).errors)})
        for facing in draft.template.ordered_facings():
            st.caption(facing.name)
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
            st.session_state.pop(ACTIVE_SIDE_KEY, None)
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
    sides = template.ordered_facings()
    if not sides:
        setup, _ = st.columns([1.4, 2.2], gap="medium")
        with setup:
            _add_side(draft)
    if sides:
        facing = next((side for side in sides if side.id == st.session_state.get(ACTIVE_SIDE_KEY)), sides[0])
        st.session_state[ACTIVE_SIDE_KEY] = facing.id
        stamp = sha256(repr(facing).encode("utf-8")).hexdigest()[:12]
        prefix = f"builder:{template.id}:{facing.id}:{stamp}"
        with st.container(border=True, key=f"side_workspace:{facing.id}"):
            st.subheader(facing.name)
            controls, canvas = st.columns([1.4, 2.2], gap="medium")

            with controls:
                _regions(draft, facing, prefix)

                selection = st.session_state.get(
                    side_selection_key(facing.id), {}
                )
                selected = selection.get(
                    "slots",
                    (selection["slot"],) if selection.get("slot") else (),
                )
                st.button(
                    "Delete Selected",
                    key=f"{facing.id}:delete_selected",
                    width="stretch",
                    disabled=not selected,
                    on_click=_apply,
                    args=(
                        lambda current: edits.remove_slots(
                            current, facing.id, selected
                        ),
                    ),
                    kwargs={"facing_id": facing.id, "rerun": False},
                )

                _side_options(draft, facing, prefix)

            with canvas:
                _preview(draft, facing)
    if not template.facings:
        st.info("Add a side, then a section and boxes.")
    _review(draft)
    _draft_details(draft)
