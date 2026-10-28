"""Application shell. Workflow UI and result ownership live in app.pages."""
from importlib import import_module

import streamlit as st


WORKFLOWS = {
    "Standard Display": "standard_display",
    "Full Pallet": "full_pallet",
    "Holiday Quarter Pallet": "holiday_qp",
    "Holiday NRE": "holiday_nre",
    "Sam's Planogram": "sams_planogram",
    "Sam's Price Strips": "sams_price_strips",
    "Image Review / Mapping": "image_review",
    "Templates": "templates",
}


def main():
    st.set_page_config(page_title="Planogram Generator", layout="wide")
    st.title("Planogram Generator")
    with st.sidebar:
        selected = st.selectbox("Workflow", list(WORKFLOWS), key="active_workflow")
    st.session_state["workflow_navigation_changed"] = st.session_state.get("previous_workflow") != selected
    st.session_state["previous_workflow"] = selected
    import_module(f"app.pages.{WORKFLOWS[selected]}").render()


if __name__ == "__main__":
    main()
