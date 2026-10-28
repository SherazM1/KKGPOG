"""Retain active inputs when Streamlit removes widgets on another workflow."""
import streamlit as st


def upload_file(label, *, key, **options):
    uploaded = st.file_uploader(label, key=key, **options)
    saved_key = f"inputs:{key}"
    present_key = f"inputs:{key}:present"
    if uploaded:
        st.session_state[saved_key] = uploaded
    elif st.session_state.get(present_key) and not st.session_state.get("workflow_navigation_changed"):
        st.session_state.pop(saved_key, None)
    st.session_state[present_key] = bool(uploaded)
    active = uploaded or st.session_state.get(saved_key)
    if active and not uploaded:
        files = active if isinstance(active, list) else [active]
        st.caption("Active file: " + ", ".join(file.name for file in files))
        if st.button("Clear active file", key=f"clear:{key}"):
            st.session_state.pop(saved_key, None)
            st.rerun()
    return active


def text_input(label, value="", *, key, **options):
    result = st.text_input(label, value=st.session_state.get(f"inputs:{key}", value),
                           key=key, **options)
    st.session_state[f"inputs:{key}"] = result
    return result


def selectbox(label, options, *, key, index=0, **kwargs):
    previous = st.session_state.get(f"inputs:{key}")
    if previous in options:
        index = list(options).index(previous)
    result = st.selectbox(label, options, index=index, key=key, **kwargs)
    st.session_state[f"inputs:{key}"] = result
    return result
