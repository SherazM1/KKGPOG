"""Holiday Quarter Pallet QA and rendering controller."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input, selectbox
import pandas as pd

from app.shared.workflow_state import sync_inputs, mark_current, holiday_fingerprint

def render():
    with st.sidebar:
        st.header("Inputs")
        holiday_bos_file = upload_file("BOS Planogram Workbook (.xlsx)", type=["xlsx"], key="holiday_qp_bos")
        holiday_assortment_file = upload_file("Holiday Assortment Workbook (.xlsx)", type=["xlsx"], key="holiday_qp_assortment")
        holiday_image_folder = text_input("Local Image Folder", value=r"Z:\Kendal King\Images", key="holiday_qp_images")
        holiday_show_geometry_debug = st.checkbox("Show geometry debug", key="holiday_qp_debug")
        generate = st.button("Build Holiday QP QA", type="primary", width="stretch")
    if sync_inputs(st.session_state, "holiday_qp", holiday_fingerprint(holiday_bos_file, holiday_assortment_file, holiday_image_folder, None), ["holiday_qp_result", "holiday_qp_render_result"]):
        st.info("Inputs changed. Rebuild to generate an updated output.")
    st.subheader("Holiday Quarter Pallet QA")
    if not (holiday_bos_file and holiday_assortment_file):
        st.info("Upload BOS Planogram Workbook + Holiday Assortment Workbook to build the Holiday QP QA preview.")
        return

    try:
        from app.holiday_planograms.service import build_holiday_qp_qa
        from app.holiday_planograms.render_qp import render_holiday_qp
    except ModuleNotFoundError as e:
        st.error(f"Holiday Quarter Pallet dependency missing: {e.name}. Please install project requirements.")
        return

    if generate or st.session_state.get("holiday_qp_result") is None:
        if not generate:
            st.info("Click 'Build Holiday QP QA' to resolve the Holiday Quarter Pallet placement model.")
            return
        st.session_state["holiday_qp_result"] = None
        st.session_state["holiday_qp_render_result"] = None
        with st.spinner("Building Holiday Quarter Pallet QA model..."):
            st.session_state["holiday_qp_result"] = build_holiday_qp_qa(
                holiday_bos_file,
                holiday_assortment_file,
                holiday_image_folder,
            )
            st.session_state["holiday_qp_render_result"] = None
        mark_current(st.session_state, "holiday_qp")
    result = st.session_state["holiday_qp_result"]

    if result.errors:
        for error in result.errors:
            st.error(error)
        return

    summary = result.summary
    ref_col, geo_col, bos_col, d5_col, match_col, image_col = st.columns(6)
    ref_col.metric("Reference Image", "Found" if result.reference_image else "Missing")
    geo_col.metric("Active Slots", summary.get("active_slots", 0))
    bos_col.metric("BOS Facings", summary.get("bos_facings", 0))
    d5_col.metric("D5 QP Facings", summary.get("d5_qp_facings", 0))
    match_col.metric("Matched Products", f"{summary.get('matched_products', 0)} / {summary.get('bos_products', 0)}")
    image_col.metric("Images", f"{summary.get('placements_with_images', 0)} / {len(result.placements)}")

    st.markdown("### Summary")
    st.dataframe(
        pd.DataFrame(
            [
                {"Area": "Reference", "Metric": "BOS QP sheet", "Value": summary.get("bos_sheet", "")},
                {
                    "Area": "Reference",
                    "Metric": "Planogram image",
                    "Value": f"{summary.get('reference_image_width', 0)}x{summary.get('reference_image_height', 0)}",
                },
                {"Area": "Geometry", "Metric": "Panels", "Value": summary.get("panel_count", 0)},
                {"Area": "Geometry", "Metric": "Rows per panel", "Value": ", ".join(map(str, summary.get("rows_per_panel", [])))},
                {"Area": "Geometry", "Metric": "Columns per panel", "Value": ", ".join(map(str, summary.get("columns_per_panel", [])))},
                {"Area": "Geometry", "Metric": "Active slots", "Value": summary.get("active_slots", 0)},
                {"Area": "BOS Table", "Metric": "Products", "Value": summary.get("bos_products", 0)},
                {"Area": "BOS Table", "Metric": "Facings", "Value": summary.get("bos_facings", 0)},
                {"Area": "D5 Assortment", "Metric": "Sheet", "Value": summary.get("d5_sheet", "")},
                {"Area": "D5 Assortment", "Metric": "Total products", "Value": summary.get("d5_total_products", 0)},
                {"Area": "D5 Assortment", "Metric": "QP products", "Value": summary.get("d5_qp_products", 0)},
                {"Area": "Product Matching", "Metric": "Item-number matches", "Value": summary.get("item_number_matches", 0)},
                {"Area": "Product Matching", "Metric": "UPC matches", "Value": summary.get("upc_matches", 0)},
                {"Area": "Product Matching", "Metric": "Fallback matches", "Value": summary.get("fallback_matches", 0)},
                {"Area": "Images", "Metric": "Local folder", "Value": summary.get("image_folder", "")},
                {"Area": "Images", "Metric": "Folder exists", "Value": summary.get("image_folder_exists", False)},
                {"Area": "Images", "Metric": "Files indexed", "Value": summary.get("image_files_indexed", 0)},
                {"Area": "Images", "Metric": "Source breakdown", "Value": str(summary.get("image_source_breakdown", {}))},
            ]
        ).astype({"Value": str}),
        width="stretch",
        hide_index=True,
    )

    if result.warnings:
        st.warning("Holiday QP QA generated warnings.")
        with st.expander("Warnings"):
            for warning in result.warnings:
                st.write(warning)

    st.markdown("### Physical Placements")
    st.dataframe(pd.DataFrame(result.placement_qa_rows), width="stretch", height=420)

    st.markdown("### Product QA")
    st.dataframe(pd.DataFrame(result.product_qa_rows), width="stretch", height=360)

    if holiday_show_geometry_debug and result.geometry and result.geometry.debug_image_bytes:
        st.markdown("### Geometry Debug")
        st.image(result.geometry.debug_image_bytes, caption="Detected panels and active slot rectangles")

    st.markdown("### Render")
    render_qp = st.button("Render Holiday QP", type="primary", width="stretch", key="render_holiday_qp")
    if render_qp:
        st.session_state["holiday_qp_render_result"] = None
        try:
            with st.spinner("Rendering Holiday Quarter Pallet..."):
                st.session_state["holiday_qp_render_result"] = render_holiday_qp(result.placements)
        except Exception as e:
            st.error(f"Unable to render Holiday Quarter Pallet: {e}")
            return

    render_result = st.session_state.get("holiday_qp_render_result")
    if render_result is not None:
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Panels Rendered", render_result.panels_rendered)
        r2.metric("Slots Rendered", render_result.slots_rendered)
        r3.metric("Image Slots", render_result.image_slots)
        r4.metric("Placeholder Slots", render_result.placeholder_slots)
        r5.metric("Skipped Slots", render_result.skipped_slots)

        image_missing_products = len(render_result.missing_image_rows)
        st.caption(
            f"Missing images: {image_missing_products} unique product(s), "
            f"{render_result.placeholder_slots} placeholder-backed placement(s)."
        )
        st.image(render_result.preview_png_bytes, caption="Holiday Quarter Pallet render preview")
        st.download_button(
            "Download Holiday QP PDF",
            render_result.pdf_bytes,
            file_name="holiday_quarter_pallet.pdf",
            mime="application/pdf",
            width="stretch",
            key="download_holiday_qp_pdf",
        )

        if render_result.missing_image_rows:
            missing_df = pd.DataFrame(render_result.missing_image_rows)
            st.markdown("### Missing Image Report")
            st.dataframe(missing_df, width="stretch", height=300)
            st.download_button(
                "Download Missing Image Report CSV",
                missing_df.to_csv(index=False).encode("utf-8"),
                file_name="holiday_qp_missing_images.csv",
                mime="text/csv",
                width="stretch",
                key="download_holiday_qp_missing_images",
            )
        if render_result.warnings:
            with st.expander("Render warnings"):
                for warning in render_result.warnings:
                    st.write(warning)
    return
