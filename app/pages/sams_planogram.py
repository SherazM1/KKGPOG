"""Sam's planogram lifecycle; domain matching and rendering stay unchanged."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input, selectbox
import pandas as pd

from app.shared.workflow_state import fingerprint, file_revision, sync_inputs, mark_current, sams_fingerprint

from app.sams_club.service import build_sams_planogram_structure, detect_sams_pogs
from app.sams_club.render_planogram import render_sams_planogram_pdf

MAPPINGS_PATH = "unresolved/manual_image_mappings.csv"


def input_fingerprint(context):
    return sams_fingerprint(context["source"], context["selected_pog"], context["image_root"],
                            context["image_zip"], context["catalog"], file_revision(MAPPINGS_PATH))


def rebuild_job(context):
    current = input_fingerprint(context)
    sync_inputs(st.session_state, "sams_planogram", current, ["sams_build_result", "sams_pdf_result"])
    st.session_state["sams_build_result"] = None
    st.session_state["sams_pdf_result"] = None
    result = build_sams_planogram_structure(
        context["source"], None, image_zip_file=context["image_zip"],
        selected_pog=context["selected_pog"], local_image_root=context["image_root"],
        ocr_catalog_file=context["catalog"],
    )
    st.session_state["sams_build_result"] = result
    st.session_state["sams_review_result"] = result
    st.session_state["sams_review_context"] = context
    mark_current(st.session_state, "sams_planogram")
    return result


def render():
    st.subheader("Sam's Planogram")
    sams_selected_pog = None
    with st.sidebar:
        st.header("Inputs")
        sams_main_source_file = upload_file(
            "Sam's Main Source (.accdb, .mdb, .xlsx, .csv)",
            type=["accdb", "mdb", "xlsx", "csv"],
            help="Upload either the Access database or an exported Master POG Query file.",
            key="sams_main_source_file",
        )
        sams_image_zip_file = upload_file(
            "Sam's Image ZIP (optional fallback)",
            type=["zip"],
            help=(
                "Optional fallback image source for Sam's card images. "
                "The app first tries each record's file_path, then ZIP filename fallback."
            ),
            key="sams_image_zip_file",
        )
        sams_ocr_catalog_file = upload_file(
            "OCR Image Catalog (optional CSV)",
            type=["csv"],
            help=(
                "Optional image-search catalog used only for unresolved Sam's images. "
                "Direct UPC matches and saved mappings are applied before OCR review."
            ),
            key="sams_ocr_catalog_file",
        )
        if sams_ocr_catalog_file is not None:
            st.session_state["sams_recent_ocr_catalog_name"] = sams_ocr_catalog_file.name
        sams_local_image_root = text_input(
            "Local Sam's Image Folder",
            key="sams_local_image_root",
            value=r"Z:\Kendal King\Images",
            help="Optional local image folder. Images will be indexed recursively.",
        )
        if sams_main_source_file:
            detected_pogs, detect_warnings = detect_sams_pogs(sams_main_source_file)
            if detect_warnings:
                for warn in detect_warnings:
                    st.warning(warn)
            if detected_pogs:
                sams_selected_pog = selectbox(
                    "Selected POG",
                    detected_pogs,
                    index=0,
                    key="sams_selected_pog",
                )
        build_sams = st.button("Build Sam's Planogram Structure", type="primary", width="stretch")
    context = dict(source=sams_main_source_file, selected_pog=sams_selected_pog,
                   image_root=sams_local_image_root, image_zip=sams_image_zip_file,
                   catalog=sams_ocr_catalog_file)
    current = input_fingerprint(context)
    sync_inputs(st.session_state, "sams_review_inputs",
                sams_fingerprint(sams_main_source_file, sams_selected_pog, sams_local_image_root,
                                 sams_image_zip_file, sams_ocr_catalog_file, None),
                ["sams_review_result", "sams_review_context"])
    if sync_inputs(st.session_state, "sams_planogram", current,
                   ["sams_build_result", "sams_pdf_result"]):
        st.info("Inputs changed. Rebuild to generate an updated output.")
    for key in ("sams_pdf_title", "sams_pdf_title_seed"):
        st.session_state.setdefault(key, st.session_state.get(f"inputs:{key}", ""))
    if build_sams and sams_main_source_file:
        with st.spinner("Building Sam's Club structure..."):
            rebuild_job(context)
    result = st.session_state.get("sams_build_result")
    if result is None:
        st.info("Click 'Build Sam's Planogram Structure' to run the completed-planogram workflow.")
    else:
        st.success("Sam's Club structure build complete.")
        for warning in result.warnings:
            st.warning(warning)

        image_debug = result.debug.get("image_resolution", {})
        local_image_diagnostics = {
            "supplied_local_image_root": sams_local_image_root,
            "normalized_local_image_root": image_debug.get("local_image_root", ""),
            "local_image_root_exists": image_debug.get("local_image_root_exists", False),
            "indexed_image_count": image_debug.get("indexed_image_count", 0),
            "lookup_key_count": image_debug.get("lookup_key_count", 0),
            "local_index_warnings": image_debug.get("local_index_warnings", []),
            "records_with_upc": result.debug.get("records_with_upc", 0),
            "records_with_item_number": result.debug.get("records_with_item_number", 0),
            "resolved_by_local_upc": image_debug.get("resolved_by_local_upc", 0),
            "resolved_by_local_item_number": image_debug.get("resolved_by_local_item_number", 0),
            "resolved_by_ocr_filename_upc": image_debug.get("resolved_by_ocr_filename_upc", 0),
            "resolved_by_ocr_upc_variant": image_debug.get("resolved_by_ocr_upc_variant", 0),
            "real_unresolved_products": image_debug.get("unresolved", 0),
            "intentional_blank_positions": image_debug.get("intentional_blank_positions", 0),
            "gci_pending_image_positions": image_debug.get("gci_pending_image_positions", 0),
            "ocr_catalog_loaded": image_debug.get("ocr_catalog_loaded", False),
            "ocr_catalog_source": image_debug.get("ocr_catalog_source", ""),
            "ocr_catalog_rows_read": image_debug.get("ocr_catalog_rows_read", 0),
            "ocr_catalog_valid_rows": image_debug.get("ocr_catalog_valid_rows", 0),
            "ocr_catalog_invalid_rows": image_debug.get("ocr_catalog_invalid_rows", 0),
            "ocr_catalog_candidates_available": image_debug.get("ocr_catalog_candidates_available", 0),
            "first_five_unresolved_examples": image_debug.get("unresolved_examples", [])[:5],
            "first_five_resolution_samples": image_debug.get("debug_sample", [])[:5],
            "startup_probe_190199709997": image_debug.get("startup_probe", {}),
        }
        if (
            image_debug.get("local_image_root_exists")
            and image_debug.get("indexed_image_count", 0) == 0
        ):
            st.error(
                f"No images were indexed from {image_debug.get('local_image_root') or sams_local_image_root}. "
                "Verify that the application process can access the mapped Z: drive. "
                "If Z: is unavailable inside Python, enter the real UNC path for the image folder."
            )
        with st.expander("Image folder diagnostics"):
            st.markdown("#### Local Image Folder Diagnostics")
            st.json(local_image_diagnostics)
        st.markdown("#### Planogram Image Summary")
        st.write(
            {
                "total_positioned_products": image_debug.get("total_positioned_products", image_debug.get("total_slots", 0)),
                "resolved_images": image_debug.get("resolved_images", 0),
                "direct_upc_matches": image_debug.get("direct_upc_matches", 0),
                "saved_manual_matches": image_debug.get("saved_manual_matches", 0),
                "ocr_approved_matches": image_debug.get("ocr_approved_matches", 0),
                "real_unresolved_products": image_debug.get("unresolved", 0),
                "intentional_blank_positions": image_debug.get("intentional_blank_positions", 0),
                "gci_pending_image_positions": image_debug.get("gci_pending_image_positions", 0),
            }
        )
        if not image_debug.get("ocr_catalog_loaded"):
            recent_catalog = st.session_state.get("sams_recent_ocr_catalog_name", "")
            if recent_catalog:
                st.info(f"No OCR catalog is loaded. Most recent catalog: {recent_catalog}.")
            else:
                st.info("No OCR catalog is loaded. Direct UPC, saved mappings, local folder, and ZIP fallback still run.")

        st.caption("Use Image Review / Mapping to resolve missing images and save selections.")

        detected_title_seed = (result.selected_pog or result.planogram.pog or "").strip()
        current_title_value = (st.session_state.get("sams_pdf_title") or "").strip()
        previous_title_seed = (st.session_state.get("sams_pdf_title_seed") or "").strip()
        if (not current_title_value) or current_title_value == previous_title_seed:
            st.session_state["sams_pdf_title"] = detected_title_seed
        st.session_state["sams_pdf_title_seed"] = detected_title_seed
        sams_pdf_title = st.text_input(
            "Sam's Sheet/Page Title",
            key="sams_pdf_title",
            help="This title is used in the Sam's completed planogram PDF header.",
        )

        st.session_state["inputs:sams_pdf_title"] = sams_pdf_title
        if sync_inputs(st.session_state, "sams_planogram_pdf", fingerprint(build=current, title=sams_pdf_title), ["sams_pdf_result"]):
            st.info("Title or inputs changed. Generate an updated PDF.")

        if not result.planogram.side_pages:
            st.warning("No populated side pages are available for PDF rendering.")
        else:
            generate_sams_pdf = st.button(
                "Generate Completed Sam's Planogram PDF",
                type="primary",
                width="stretch",
                key="generate_sams_pdf",
            )
            if generate_sams_pdf:
                mark_current(st.session_state, "sams_planogram_pdf")
                st.session_state["sams_pdf_result"] = None
                with st.spinner("Rendering Sam's completed planogram PDF..."):
                    st.session_state["sams_pdf_result"] = render_sams_planogram_pdf(
                        result.planogram,
                        generated_by="Kendal King",
                        title_override=sams_pdf_title,
                    )

        pdf_result = st.session_state.get("sams_pdf_result")
        if pdf_result is not None:
            st.success(
                f"Rendered {pdf_result.rendered_slots} slots across {len(result.planogram.side_pages)} side page(s)."
            )
            if pdf_result.warnings:
                st.warning(
                    f"{pdf_result.missing_image_slots} cards rendered without image. "
                    "Missing images were replaced with placeholders."
                )
                with st.expander("Image load warnings"):
                    for warning in pdf_result.warnings:
                        st.write(f"- {warning}")

            sanitized_pog = (result.selected_pog or "sams").replace(" ", "_")
            st.download_button(
                "Download Completed Sam's Planogram PDF",
                pdf_result.pdf_bytes,
                file_name=f"{sanitized_pog}_completed_planogram.pdf",
                mime="application/pdf",
                width="stretch",
                key="download_sams_pdf",
            )

        with st.expander("Advanced planogram diagnostics"):
            st.write("Detected source type:", result.debug.get("source_type", "unknown"))
            st.write("Normalized column mapping used:", result.debug.get("column_mapping", {}))
            st.write("Detected POGs:", result.detected_pogs)
            st.write("Selected POG:", result.selected_pog or "(none)")
            st.write("Sides found:", result.debug.get("sides_found", []))
            st.write("Side counts:", result.debug.get("side_counts", {}))
            st.write("Rows per side:", result.debug.get("rows_per_side", {}))
            st.write("Populated columns per row:", result.debug.get("populated_columns_per_row", {}))
            image_debug = result.debug.get("image_resolution", {})
            st.write("Sam's image zip uploaded:", "yes" if image_debug.get("image_zip_uploaded") else "no")
            st.write("Sam's total slots:", image_debug.get("total_slots", 0))
            st.write("Resolved images:", image_debug.get("resolved_images", 0))
            st.write("Intentional blanks:", image_debug.get("intentional_blank_positions", 0))
            st.write("GCI pending images:", image_debug.get("gci_pending_image_positions", 0))
            st.write("Resolved by original path:", image_debug.get("resolved_by_original_path", 0))
            st.write("Resolved by zip basename:", image_debug.get("resolved_by_zip_basename", 0))
            st.write("Resolved by zip UPC:", image_debug.get("resolved_by_zip_upc", 0))
            st.write("Real unresolved products:", image_debug.get("unresolved", 0))
            unresolved_examples = image_debug.get("unresolved_examples", [])
            if unresolved_examples:
                with st.expander("Unresolved image examples (up to 10)"):
                    for example in unresolved_examples:
                        st.write(
                            "- "
                            f"side={example.get('side')} row={example.get('row')} column={example.get('column')} "
                            f"upc={example.get('upc')} file_path={example.get('file_path')}"
                        )
            st.write("Warnings:", result.debug.get("warnings", []))
            st.write("Errors:", result.debug.get("errors", []))
            st.json(result.to_dict())

