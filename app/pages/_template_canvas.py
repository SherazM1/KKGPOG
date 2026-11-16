"""Small vanilla-JS component, isolated from production renderers and persistence."""
from pathlib import Path
from dataclasses import replace

import streamlit as st
import streamlit.components.v1 as components

from app.pages._canvas_events import CanvasTransform, consume_event
from app.pages._direct_canvas import canvas_svg


_component = components.declare_component(
    # Version the asset URL so open sessions pick up canvas/toolbar updates on rerun.
    "template_canvas_side_previews", path=Path(__file__).resolve().parents[1] / "components" / "template_canvas",
)


def render_canvas(svg, draft, facing, *, draft_key, selection_key):
    key = f"template_canvas:{draft.template.id}:{facing.id}"
    selection = st.session_state.get(selection_key, {})
    transform = CanvasTransform.from_svg(svg, facing) if svg else None
    notice = st.session_state.pop(f"{key}:notice", None)
    if notice:
        st.warning(notice)

    def changed():
        consume_event(st.session_state, component_key=key, draft_key=draft_key,
                      selection_key=selection_key, facing_id=facing.id, transform=transform, allow_resize=False)

    previews = []
    for side in draft.template.ordered_facings():
        try:
            preview = canvas_svg(replace(draft, template=replace(draft.template, facings=(side,))), side)
        except ValueError:
            preview = None
        previews.append(dict(id=side.id, name=side.name, svg=preview, box_count=len(side.slots)))

    _component(
        svg=svg, side_previews=previews, template_id=draft.template.id, facing_id=facing.id, revision=draft.revision,
        selected_box_id=selection.get("slot"), active_section_id=selection.get("region"),
        selected_box_ids=list(selection.get('slots', (selection['slot'],) if selection.get('slot') else ())),
        boxes={slot.id: {"number": slot.order, "sections": [region.name for region in facing.regions_for_slot(slot.id)]}
               for slot in facing.slots},
        ack_id=st.session_state.get(f"{key}:ack"), default=None, key=key, on_change=changed,
    )
