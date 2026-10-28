"""Holiday NRE configuration, QA, and rendering controller."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input, selectbox
import pandas as pd

from app.shared.workflow_state import sync_inputs, mark_current, holiday_fingerprint

def render():
    with st.sidebar:
        st.header("Inputs")
        holiday_nre_configuration = selectbox("NRE Configuration", ["4x78", "3x78", "3x60", "4x60"], key="holiday_nre_configuration")
        holiday_bos_file = upload_file("BOS Planogram Workbook (.xlsx)", type=["xlsx"], key="holiday_nre_bos")
        holiday_assortment_file = upload_file("Holiday Assortment Workbook (.xlsx)", type=["xlsx"], key="holiday_nre_assortment")
        holiday_image_folder = text_input("Local Image Folder", value=r"Z:\Kendal King\Images", key="holiday_nre_images")
        holiday_show_geometry_debug = st.checkbox("Show geometry debug", key="holiday_nre_debug")
        analyze_all_holiday_nre = st.button("Analyze All NRE Configurations", width="stretch")
        generate = st.button("Build Holiday NRE QA", type="primary", width="stretch")
    # Source changes invalidate every configuration; selection changes only the selected job.
    source_results = ["holiday_nre_all_configurations"] + [
        key for config in ("4x78", "3x78", "3x60", "4x60")
        for key in (f"holiday_nre_result_{config}", f"holiday_nre_render_result_{config}")
    ]
    sources_stale = sync_inputs(
        st.session_state, "holiday_nre_sources",
        holiday_fingerprint(holiday_bos_file, holiday_assortment_file, holiday_image_folder),
        source_results,
    )
    selected_stale = sync_inputs(
        st.session_state, "holiday_nre",
        holiday_fingerprint(holiday_bos_file, holiday_assortment_file,
                            holiday_image_folder, holiday_nre_configuration),
        [f"holiday_nre_result_{holiday_nre_configuration}",
         f"holiday_nre_render_result_{holiday_nre_configuration}"],
    )
    if sources_stale or selected_stale:
        st.info("Inputs changed. Rebuild to generate an updated output.")
    st.subheader("Holiday NRE QA")
    if not (holiday_bos_file and holiday_assortment_file):
        st.info("Upload BOS Planogram Workbook + Holiday Assortment Workbook to build the Holiday NRE QA preview.")
        return

    try:
        from app.holiday_planograms.service import analyze_holiday_nre_configurations, build_holiday_nre_qa
        from app.holiday_planograms.render_nre import render_holiday_nre
    except ModuleNotFoundError as e:
        st.error(f"Holiday NRE dependency missing: {e.name}. Please install project requirements.")
        return

    if analyze_all_holiday_nre:
        with st.spinner("Analyzing all Holiday NRE configurations..."):
            st.session_state["holiday_nre_all_configurations"] = analyze_holiday_nre_configurations(
                holiday_bos_file,
                holiday_assortment_file,
            )

    all_configurations = st.session_state.get("holiday_nre_all_configurations")
    if all_configurations:
        st.markdown("### All NRE Configurations")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Configuration": row.configuration,
                        "Selected Sheet": row.sheet_name,
                        "Selected Embedded Image": row.reference_image_name,
                        "Image Dimensions": f"{row.reference_width}x{row.reference_height}",
                        "Panel Count": row.panel_count,
                        "Rows": row.row_count,
                        "Columns": row.column_count,
                        "Grid Positions": row.grid_positions,
                        "Physical Detected": row.physical_detected,
                        "Merchandise": row.merchandise_slots,
                        "Filler": row.filler_slots,
                        "Ambiguous": row.ambiguous_slots,
                        "D5 NRE Facings": row.expected_nre_facings,
                        "Difference": row.difference,
                        "Warnings": "; ".join(row.warnings),
                        "Errors": "; ".join(row.errors),
                    }
                    for row in all_configurations
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    result_key = f"holiday_nre_result_{holiday_nre_configuration}"
    if generate or st.session_state.get(result_key) is None:
        if not generate:
            st.info("Click 'Build Holiday NRE QA' to analyze the selected Holiday NRE configuration.")
            return
        st.session_state[result_key] = None
        st.session_state[f"holiday_nre_render_result_{holiday_nre_configuration}"] = None
        with st.spinner("Building Holiday NRE QA model..."):
            st.session_state[result_key] = build_holiday_nre_qa(
                holiday_bos_file,
                holiday_assortment_file,
                holiday_nre_configuration,
                holiday_image_folder,
            )
            st.session_state[f"holiday_nre_render_result_{holiday_nre_configuration}"] = None
        mark_current(st.session_state, "holiday_nre")
        mark_current(st.session_state, "holiday_nre_sources")
    result = st.session_state[result_key]

    if result.errors:
        for error in result.errors:
            st.error(error)
        return

    summary = result.summary
    ref_col, config_col, geo_col, d5_col, match_col, image_col = st.columns(6)
    ref_col.metric("Reference Found", "Yes" if result.reference else "No")
    config_col.metric("Configuration", summary.get("configuration", holiday_nre_configuration))
    geo_col.metric("Merchandise Slots", summary.get("merchandise_slots", 0))
    d5_col.metric("D5 NRE Facings", summary.get("d5_nre_facings", 0))
    match_col.metric("Resolved Placements", summary.get("resolved_placements", 0))
    image_col.metric("Images", f"{summary.get('unique_products_with_images', 0)} / {summary.get('d5_nre_products', 0)}")

    st.markdown("### Summary")
    st.dataframe(
        pd.DataFrame(
            [
                {"Area": "Reference", "Metric": "BOS NRE sheet", "Value": summary.get("bos_sheet", "")},
                {"Area": "Reference", "Metric": "Selected configuration", "Value": summary.get("configuration", "")},
                {"Area": "Reference", "Metric": "Reference found", "Value": summary.get("reference_found", False)},
                {"Area": "Reference", "Metric": "Embedded image", "Value": summary.get("reference_image_name", "")},
                {
                    "Area": "Reference",
                    "Metric": "Planogram image",
                    "Value": f"{summary.get('reference_image_width', 0)}x{summary.get('reference_image_height', 0)}",
                },
                {"Area": "Geometry", "Metric": "Panels", "Value": summary.get("panel_count", 0)},
                {"Area": "Geometry", "Metric": "Rows per panel", "Value": ", ".join(map(str, summary.get("rows_per_panel", [])))},
                {"Area": "Geometry", "Metric": "Columns per panel", "Value": ", ".join(map(str, summary.get("columns_per_panel", [])))},
                {"Area": "Geometry", "Metric": "Physical cells detected", "Value": summary.get("physical_cells_detected", 0)},
                {"Area": "Geometry", "Metric": "Theoretical grid positions", "Value": summary.get("theoretical_grid_positions", 0)},
                {"Area": "Geometry", "Metric": "Merchandise slots", "Value": summary.get("merchandise_slots", 0)},
                {"Area": "Geometry", "Metric": "Filler slots", "Value": summary.get("filler_slots", 0)},
                {"Area": "Geometry", "Metric": "Ambiguous slots", "Value": summary.get("ambiguous_slots", 0)},
                {"Area": "Geometry", "Metric": "Merchandise difference", "Value": summary.get("merchandise_difference", 0)},
                {"Area": "D5 Assortment", "Metric": "Sheet", "Value": summary.get("d5_sheet", "")},
                {"Area": "D5 Assortment", "Metric": "NRE products", "Value": summary.get("d5_nre_products", 0)},
                {"Area": "D5 Assortment", "Metric": "NRE facings", "Value": summary.get("d5_nre_facings", 0)},
                {"Area": "Placement", "Metric": "Resolved placements", "Value": summary.get("resolved_placements", 0)},
                {"Area": "Placement", "Metric": "Unresolved placements", "Value": summary.get("unresolved_placements", 0)},
                {"Area": "Placement", "Metric": "Non-product placements", "Value": summary.get("non_product_placements", 0)},
                {"Area": "Placement", "Metric": "Ambiguous placements", "Value": summary.get("ambiguous_placements", 0)},
                {"Area": "Placement", "Metric": "Ambiguous reference text", "Value": summary.get("ambiguous_reference_text_placements", 0)},
                {"Area": "Product QA", "Metric": "Facing mismatches", "Value": summary.get("facing_mismatches", 0)},
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
        st.warning("Holiday NRE QA generated warnings.")
        with st.expander("Warnings"):
            for warning in result.warnings:
                st.write(warning)

    st.markdown("### Physical Placements")
    st.dataframe(pd.DataFrame(result.placement_qa_rows), width="stretch", height=420)

    st.markdown("### Product QA")
    st.dataframe(pd.DataFrame(result.product_qa_rows), width="stretch", height=360)

    if holiday_show_geometry_debug and result.geometry and result.geometry.debug_image_bytes:
        st.markdown("### Geometry Debug")
        st.image(result.geometry.debug_image_bytes, caption=f"Holiday NRE {result.configuration} detected slot rectangles")

    st.markdown("### Render")
    render_key = f"holiday_nre_render_result_{holiday_nre_configuration}"
    render_nre = st.button("Render Holiday NRE", type="primary", width="stretch", key="render_holiday_nre")
    if render_nre:
        st.session_state[render_key] = None
        try:
            with st.spinner("Rendering Holiday NRE..."):
                st.session_state[render_key] = render_holiday_nre(result.placements, result.configuration)
        except Exception as e:
            st.error(f"Unable to render Holiday NRE: {e}")
            return

    render_result = st.session_state.get(render_key)
    if render_result is not None:
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Pages Rendered", render_result.panels_rendered)
        r2.metric("Slots Rendered", render_result.slots_rendered)
        r3.metric("Image Cards", render_result.image_slots)
        r4.metric("Placeholder Cards", render_result.placeholder_slots)
        r5.metric("Filler Blocks", render_result.skipped_slots)

        st.image(render_result.preview_png_bytes, caption=f"Holiday NRE {result.configuration} render preview")
        st.download_button(
            "Download Holiday NRE PDF",
            render_result.pdf_bytes,
            file_name=f"holiday_nre_{result.configuration.lower()}.pdf",
            mime="application/pdf",
            width="stretch",
            key="download_holiday_nre_pdf",
        )

        if render_result.missing_image_rows:
            missing_df = pd.DataFrame(render_result.missing_image_rows)
            st.markdown("### Missing Image Report")
            st.dataframe(missing_df, width="stretch", height=300)
            st.download_button(
                "Download Missing Image Report CSV",
                missing_df.to_csv(index=False).encode("utf-8"),
                file_name=f"holiday_nre_{result.configuration.lower()}_missing_images.csv",
                mime="text/csv",
                width="stretch",
                key="download_holiday_nre_missing_images",
            )
    return
