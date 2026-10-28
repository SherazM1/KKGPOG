"""Independent Sam's strip and Price Pocket workflow."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input, selectbox
import pandas as pd

from app.shared.workflow_state import fingerprint, file_revision, sync_inputs, mark_current

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import is_sams_holiday_template

def _build_sams_price_strip_rows_compat(source_file: object, template_name: str | None = None):
    try:
        return build_sams_price_strip_rows(source_file, template_name=template_name)
    except TypeError as exc:
        if "unexpected keyword argument 'template_name'" not in str(exc):
            raise
        return build_sams_price_strip_rows(source_file)


# Renderer toggle: set to True to use the new HTML/Playwright renderer, False for the old ReportLab renderer
USE_HTML_PRICE_STRIP_RENDERER = True

def _load_sams_price_strip_renderer(template_name: str | None = None):
    if USE_HTML_PRICE_STRIP_RENDERER or is_sams_holiday_template(template_name):
        from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf
    else:
        from app.sams_club.render_price_strips import render_sams_price_strips_pdf
    return render_sams_price_strips_pdf


def render():
    st.subheader("Sam's Price Strips")
    with st.sidebar:
        st.header("Inputs")
        sams_excel_file = upload_file(
            "Sam's Price Strip Source (.xlsx, .docx, .pdf)",
            type=["xlsx", "docx", "pdf"],
            help="Workbooks use the existing Price Strip Data schema. DOCX/PDF accept labeled fields or tables for the selected mode.",
            key="sams_excel_file",
        )
        sams_price_strip_template = selectbox(
            "Sam's Price Strip Template",
            ["Standard", "Sam's Holiday", "Price Pocket"],
            index=0,
            key="sams_price_strip_template",
        )
        sams_price_strip_calibration = st.checkbox(
            "Sam's Holiday calibration guides",
            value=False,
            key="sams_price_strip_calibration",
            help="Draws strip boundary, slot boundaries, centerlines, and slot labels on Holiday price strips.",
        )
        generate_sams_price_strips = st.button(
            "Generate Sam's Price Strips",
            type="primary",
            width="stretch",
            key="generate_sams_price_strips",
        )
    current = fingerprint(source=sams_excel_file, template=sams_price_strip_template, calibration=sams_price_strip_calibration, html_renderer=USE_HTML_PRICE_STRIP_RENDERER, layout=file_revision("assets/templates/sams_price_strip_layout.json"))
    if sync_inputs(st.session_state, "sams_strips", current, ["sams_price_strip_build_result", "sams_price_strip_pdf_result"]):
        st.info("Inputs changed. Generate an updated output.")
    if generate_sams_price_strips:
        mark_current(st.session_state, "sams_strips")
    if sams_price_strip_template == "Price Pocket":
        if not sams_excel_file:
            st.info("Upload XLSX, DOCX or PDF with Name, Price, Item and UPC. Quantity defaults to 1; Type defaults to GIFT CARD.")
            return
        from dataclasses import asdict
        from app.sams_club.price_strip_inputs import parse_source
        from app.sams_club.price_pocket import pocket_records, render_price_pocket_pdf
        try:
            records = pocket_records(parse_source(sams_excel_file))
            st.dataframe([asdict(record) for record in records], width="stretch")
            if generate_sams_price_strips:
                st.session_state["sams_price_strip_pdf_result"] = None
                st.session_state["sams_price_strip_pdf_result"] = render_price_pocket_pdf(records)
            pocket_pdf = st.session_state.get("sams_price_strip_pdf_result")
            if pocket_pdf:
                st.success(f"Rendered {pocket_pdf.rendered_segments} Price Pocket strips, 8.5 × 4 inches each.")
                for warning in pocket_pdf.warnings:
                    st.warning(warning)
                st.download_button("Download Sam's Price Strips PDF", pocket_pdf.pdf_bytes,
                                   file_name="sams_price_pocket.pdf", mime="application/pdf",
                                   key="download_sams_price_strips_pdf")
        except Exception as exc:
            st.session_state["sams_price_strip_pdf_result"] = None
            st.error(f"Price Pocket generation failed: {type(exc).__name__}: {str(exc) or 'The renderer could not complete the PDF. Check the terminal traceback for details.'}")
        return
    if not sams_excel_file:
        st.info("Upload a Sam's price strip source (.xlsx, .docx or .pdf).")
    else:
        if sams_excel_file.name.lower().endswith((".docx", ".pdf")):
            st.caption(
                "Document inputs use product fields; Excel worksheet columns are not required. "
                "Without explicit positions, Standard places one product per 11 × 2.45 inch strip; "
                "Sam's Holiday fills six-position strips in document order. "
                "Choose Price Pocket for 8.5 × 4 inch cards."
            )
        if st.session_state.get("sams_price_strip_build_result") is None:
            st.session_state["sams_price_strip_build_result"] = _build_sams_price_strip_rows_compat(
                sams_excel_file, template_name=sams_price_strip_template)
        preview_build = st.session_state["sams_price_strip_build_result"]
        if not preview_build.errors:
            from dataclasses import asdict
            st.dataframe([asdict(segment) for row in preview_build.strip_rows for segment in row.segments],
                         width="stretch")
        if generate_sams_price_strips:
            with st.spinner("Building Sam's price strip groups..."):
                strip_build = _build_sams_price_strip_rows_compat(
                    sams_excel_file,
                    template_name=sams_price_strip_template,
                )
            st.session_state["sams_price_strip_build_result"] = strip_build
            if strip_build.errors:
                st.session_state["sams_price_strip_pdf_result"] = None
            else:
                with st.spinner("Rendering Sam's price strips PDF..."):
                    render_sams_price_strips_pdf = _load_sams_price_strip_renderer(sams_price_strip_template)
                    st.session_state["sams_price_strip_pdf_result"] = None
                    try:
                        st.session_state["sams_price_strip_pdf_result"] = render_sams_price_strips_pdf(
                            strip_build.strip_rows,
                            generated_by="Kendal King",
                            template_name=sams_price_strip_template,
                            calibration=sams_price_strip_calibration,
                        )
                    except Exception as exc:
                        st.error(f"Price strip rendering failed: {exc}")

        strip_build = st.session_state.get("sams_price_strip_build_result")
        if strip_build is not None:
            if strip_build.errors:
                for error in strip_build.errors:
                    st.error(error)
            else:
                st.success(
                    f"Detected {strip_build.debug.get('strip_group_count', 0)} strip group(s) "
                    f"from {strip_build.extracted_record_count} source record(s)."
                )
                st.write("Detected strip groups:", strip_build.debug.get("detected_strip_groups", []))
                st.write("Template:", strip_build.debug.get("template_name", ""))
                st.write("(POG, Side, Row) group count:", strip_build.debug.get("strip_group_count", 0))
                st.write("Segment count per strip row:", strip_build.debug.get("segments_per_strip_row", {}))
                st.write("Included segment count:", strip_build.included_segment_count)
                st.write("Skipped segment count:", strip_build.skipped_segment_count)
                if strip_build.warnings:
                    with st.expander("Price strip warnings"):
                        for warning in strip_build.warnings:
                            st.write(f"- {warning}")

                strip_pdf = st.session_state.get("sams_price_strip_pdf_result")
                if strip_pdf is not None:
                    st.success(
                        f"Rendered {strip_pdf.rendered_pages} strip page(s) with {strip_pdf.rendered_segments} segment block(s)."
                    )
                    with st.expander("Renderer diagnostics"):
                        renderer_module = "app.sams_club.render_price_strips_html" if USE_HTML_PRICE_STRIP_RENDERER else "app.sams_club.render_price_strips"
                        st.caption(
                            f"Active Sam's strip renderer: {renderer_module}.render_sams_price_strips_pdf"
                        )
                    if strip_pdf.warnings:
                        with st.expander("Price strip render warnings"):
                            for warning in strip_pdf.warnings:
                                st.write(f"- {warning}")
                    st.download_button(
                        "Download Sam's Price Strips PDF",
                        strip_pdf.pdf_bytes,
                        file_name="sams_price_strips.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key="download_sams_price_strips_pdf",
                    )
    return

