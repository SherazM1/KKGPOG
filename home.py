"""Application shell. Workflow UI and result ownership live in app.pages."""
from importlib import import_module, reload
from pathlib import Path
import sys

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


def _refresh_template_workflow():
    """Pick up changed template code without restarting session-only drafts.

    Imported page modules can stay cached after Streamlit reruns home.py.
    Reload only the template dependencies, leaving model types and the existing
    draft instances and geometry in session state intact.
    """
    module_names = (
        "app.templates.image_compare",
        "app.pages._pog_comparison",
        "app.sams_club.price_strip_models",
        "app.sams_club.holiday_price_strips",
        "app.sams_club.generic_price_strips",
        "app.sams_club.render_generic_price_strips",
        "app.sams_club.render_price_strips_html",
        "app.templates.print_style",
        "app.templates.draft",
        "app.pages._direct_builder",
        "app.templates.render_pdf",
        "app.templates.eps_preview",
        "app.pages.template_builder",
        "app.pages._price_strip_sheet",
        "app.pages.price_strip_builder",
        "app.pages.templates",
    )
    root = Path(__file__).resolve().parent
    signature = []
    for name in module_names:
        source = root.joinpath(*name.split('.')).with_suffix('.py')
        stat = source.stat()
        signature.append((name, stat.st_mtime_ns, stat.st_size))
    signature = tuple(signature)
    if st.session_state.get("app:template_source_signature") == signature:
        return
    for name in module_names:
        module = sys.modules.get(name)
        if module is not None:
            reload(module)
    st.session_state["app:template_source_signature"] = signature
    st.session_state.pop("template_builder:pdf_export", None)
    st.session_state.pop("blank_price_strips:holiday_pdf", None)
    builder = sys.modules.get('app.pages.price_strip_builder')
    if builder is not None:
        builder._live_strip_preview.clear()
        builder._eps_preview.clear()


def _sync_sidebar():
    """Apply startup/first-Side requests once, then leave the toggle to the user.

    Page config alone can be overridden by Streamlit's saved browser preference.
    Use the normal toggle for these explicit transitions, without hiding it.
    """
    state = st.session_state.get("app:sidebar_state", "expanded")
    if st.session_state.get("app:sidebar_applied") == state:
        return
    st.session_state["app:sidebar_applied"] = state
    expanded = "true" if state == "expanded" else "false"
    st.html(
        """<script>
        (() => {
            const doc = document;
            const desired = EXPANDED;
            let attempts = 0;
            const timer = setInterval(() => {
                const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                if (++attempts > 100) { clearInterval(timer); return; }
                if (!sidebar) return;
                const open = sidebar.getAttribute('aria-expanded') === 'true';
                if (open === desired) { clearInterval(timer); return; }
                const selector = desired
                    ? '[data-testid="stExpandSidebarButton"]'
                    : '[data-testid="stSidebarCollapseButton"]';
                const control = doc.querySelector(selector);
                const button = control?.matches('button')
                    ? control : control?.querySelector('button');
                if (button) { button.click(); clearInterval(timer); }
            }, 50);
            window.addEventListener('pagehide', () => clearInterval(timer), {once: true});
        })();
        </script>""".replace("EXPANDED", expanded),
        unsafe_allow_javascript=True,
    )

def _collapse_sidebar_on_selection():
    st.session_state["app:sidebar_state"] = "collapsed"
    st.session_state.pop("app:sidebar_applied", None)
    
    
def main():
    st.set_page_config(
        page_title="Planogram Generator",
        layout="wide",
        initial_sidebar_state=st.session_state.get("app:sidebar_state", "expanded"),
    )
    st.title("Planogram Generator")
    with st.sidebar:
        selected = st.selectbox("Workflow", 
            list(WORKFLOWS),
            key="active_workflow",
            on_change=_collapse_sidebar_on_selection,
        )
    _sync_sidebar()
    st.session_state["workflow_navigation_changed"] = st.session_state.get("previous_workflow") != selected
    st.session_state["previous_workflow"] = selected
    if selected == "Templates":
        _refresh_template_workflow()
    import_module(f"app.pages.{WORKFLOWS[selected]}").render()
    

if __name__ == "__main__":
    main()
