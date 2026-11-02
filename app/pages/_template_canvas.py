"""Small vanilla-JS component, isolated from production renderers and persistence."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.pages._canvas_events import CanvasTransform, consume_event


_component = components.declare_component(
    "template_canvas", path=Path(__file__).resolve().parents[1] / "components" / "template_canvas",
)


def render_canvas(svg, draft, facing, *, draft_key, selection_key):
    key = f"template_canvas:{draft.template.id}:{facing.id}"
    selection = st.session_state.get(selection_key, {})
    transform = CanvasTransform.from_svg(svg, facing)
    notice = st.session_state.pop(f"{key}:notice", None)
    if notice:
        st.warning(notice)

    def changed():
        consume_event(st.session_state, component_key=key, draft_key=draft_key,
                      selection_key=selection_key, facing_id=facing.id, transform=transform)

    _component(
        svg=svg, template_id=draft.template.id, facing_id=facing.id, revision=draft.revision,
        selected_box_id=selection.get("slot"), active_section_id=selection.get("region"),
        boxes={slot.id: {"number": slot.order, "sections": [region.name for region in facing.regions_for_slot(slot.id)]}
               for slot in facing.slots},
        ack_id=st.session_state.get(f"{key}:ack"), default=None, key=key, on_change=changed,
    )
