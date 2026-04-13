from __future__ import annotations

import pandas as pd
import streamlit as st

from app.sams_club.render_planogram import render_sams_planogram_pdf
from app.sams_club.service import build_sams_planogram_structure, detect_sams_pogs
from app.shared.constants import DISPLAY_FULL_PALLET, DISPLAY_SAMS_CLUB, DISPLAY_STANDARD, N_COLS


def main() -> None:
    st.set_page_config(page_title="Planogram Generator", layout="wide")
    st.title("Planogram Generator")

    matrix_file = None
    labels_pdf = None
    images_pdf = None
    title_prefix = "POG"
    out_name = "pog_export.pdf"
    generate = False
    sams_main_source_file = None
    sams_excel_file = None
    sams_image_zip_file = None
    sams_selected_pog = None
    build_sams = False

    with st.sidebar:
        st.header("Configuration")
        display_type = st.selectbox(
            "Display type",
            [DISPLAY_STANDARD, DISPLAY_FULL_PALLET, DISPLAY_SAMS_CLUB],
            index=0,
        )

        if display_type == DISPLAY_SAMS_CLUB:
            st.divider()
            sams_main_source_file = st.file_uploader(
                "Sam's Main Source (.accdb, .mdb, .xlsx, .csv)",
                type=["accdb", "mdb", "xlsx", "csv"],
                help="Upload either the Access database or an exported Master POG Query file.",
                key="sams_main_source_file",
            )
            sams_excel_file = st.file_uploader(
                "Sam's Excel Workbook (optional support file)",
                type=["xlsx"],
                key="sams_excel_file",
            )
            sams_image_zip_file = st.file_uploader(
                "Sam's Image ZIP (optional fallback)",
                type=["zip"],
                help=(
                    "Optional fallback image source for Sam's card images. "
                    "The app first tries each record's file_path, then ZIP filename fallback."
                ),
                key="sams_image_zip_file",
            )
            if sams_main_source_file:
                detected_pogs, detect_warnings = detect_sams_pogs(sams_main_source_file)
                if detect_warnings:
                    for warn in detect_warnings:
                        st.warning(warn)
                if detected_pogs:
                    sams_selected_pog = st.selectbox(
                        "Selected POG",
                        detected_pogs,
                        index=0,
                        key="sams_selected_pog",
                    )
            build_sams = st.button("Build Sam's Planogram Structure", type="primary", use_container_width=True)
        else:
            st.divider()
            matrix_file = st.file_uploader("Matrix Excel (.xlsx)", type=["xlsx"])
            labels_pdf = st.file_uploader("Labels PDF", type=["pdf"])
            images_pdf = st.file_uploader("Images PDF", type=["pdf"])

            st.divider()
            title_prefix = st.text_input("PDF title prefix", "POG")
            out_name = st.text_input("Output filename", "pog_export.pdf")
            generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if display_type == DISPLAY_SAMS_CLUB:
        st.subheader("Sam's Club Planogram Display")
        st.info("Build the populated Sam's structure, then generate one completed PDF page per side.")

        if not sams_main_source_file:
            st.warning("Upload a Sam's main source file (.accdb, .mdb, .xlsx, or .csv) to begin.")
            return

        st.caption("Optional support file accepted: Excel workbook (.xlsx).")

        if "sams_build_result" not in st.session_state:
            st.session_state["sams_build_result"] = None
        if "sams_pdf_result" not in st.session_state:
            st.session_state["sams_pdf_result"] = None

        if build_sams:
            with st.spinner("Building Sam's Club structure..."):
                result = build_sams_planogram_structure(
                    sams_main_source_file,
                    sams_excel_file,
                    image_zip_file=sams_image_zip_file,
                    selected_pog=sams_selected_pog,
                )
            st.session_state["sams_build_result"] = result
            st.session_state["sams_pdf_result"] = None

        result = st.session_state.get("sams_build_result")
        if result is None:
            st.info("Click 'Build Sam's Planogram Structure' to run the scaffold workflow.")
            return

        st.success("Sam's Club structure build complete.")
        for warning in result.warnings:
            st.warning(warning)

        if not result.planogram.side_pages:
            st.warning("No populated side pages are available for PDF rendering.")
        else:
            generate_sams_pdf = st.button(
                "Generate Completed Sam's Planogram PDF",
                type="primary",
                use_container_width=True,
                key="generate_sams_pdf",
            )
            if generate_sams_pdf:
                with st.spinner("Rendering Sam's completed planogram PDF..."):
                    st.session_state["sams_pdf_result"] = render_sams_planogram_pdf(
                        result.planogram,
                        generated_by="Kendal King",
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
                use_container_width=True,
                key="download_sams_pdf",
            )

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
        st.write("Resolved by original path:", image_debug.get("resolved_by_original_path", 0))
        st.write("Resolved by zip basename:", image_debug.get("resolved_by_zip_basename", 0))
        st.write("Resolved by zip UPC:", image_debug.get("resolved_by_zip_upc", 0))
        st.write("Unresolved:", image_debug.get("unresolved", 0))
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
        return

    if not (matrix_file and labels_pdf and images_pdf):
        st.info("Upload Matrix XLSX + Labels PDF + Images PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    images_bytes = images_pdf.getvalue()

    if display_type == DISPLAY_STANDARD:
        from app.standard_display.service import prepare_standard_display, render_standard_display_pdf

        pages, matrix_idx, rows = prepare_standard_display(matrix_bytes, labels_bytes, N_COLS)
        if not pages:
            st.error("No 5-digit UPC tokens found in Labels PDF.")
            return

        st.subheader(f"Detected {len(pages)} side(s)")

        st.dataframe(
            pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
            use_container_width=True,
            height=420,
        )

        if generate:
            with st.spinner("Rendering PDF..."):
                pdf = render_standard_display_pdf(pages, images_bytes, matrix_idx, title_prefix)
            st.success("Done.")
            st.download_button(
                "Download Planogram PDF",
                pdf,
                file_name=out_name if out_name.endswith(".pdf") else f"{out_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    else:
        try:
            from app.full_pallet.service import (
                build_full_pallet_rows,
                load_full_pallet_matrix,
                parse_full_pallet_pages,
                parse_gift_holders,
                parse_ppt_cards,
                render_full_pallet_display_pdf,
                validate_ppt_cards,
            )
        except ModuleNotFoundError as e:
            st.error(f"Full Pallet mode dependency missing: {e.name}. Please install project requirements.")
            return

        pptx_file = st.file_uploader("Top Cards Blueprint (.pptx)", type=["pptx"])
        gift_file = st.file_uploader("2025 D82 POG Workbook (.xlsx)", type=["xlsx"])
        ppt_cpp_global = st.number_input("PPT Cards CPP (Global)", min_value=0, value=0, step=1)
        show_debug = st.checkbox("Show debug details")
        show_layout_overlay = st.checkbox("Show Full Pallet layout overlay")

        if not (pptx_file and gift_file):
            st.info("Upload PPTX + POG XLSX for Full Pallet mode.")
            return

        try:
            ppt_cards = parse_ppt_cards(pptx_file.getvalue())
        except ImportError:
            st.error(
                "python-pptx is not installed. Full Pallet mode requires python-pptx to parse the Top Cards Blueprint."
            )
            return
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error("Unable to parse Top Cards Blueprint (.pptx). Please verify the file.")
            return

        ppt_issues = validate_ppt_cards(ppt_cards)
        if ppt_issues:
            st.error("Top Cards Blueprint validation failed:\n" + "\n".join(f"- {msg}" for msg in ppt_issues))
            return

        if ppt_cpp_global in (None, 0):
            st.warning("PPT CPP global is 0 or empty. PPT cards will render with CPP: 0.")

        try:
            gift_holders = parse_gift_holders(gift_file.getvalue())
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error(f"Unable to parse POG workbook holder table: {e}")
            return

        try:
            fp_matrix_idx = load_full_pallet_matrix(matrix_bytes)
        except Exception as e:
            if show_debug:
                st.exception(e)
            st.error("Unable to parse D82 Item List for Full Pallet matching.")
            return

        fp_pages = parse_full_pallet_pages(labels_bytes)
        if not fp_pages:
            st.error("No product cells detected in Labels PDF.")
            return

        st.subheader(f"Detected {len(fp_pages)} side(s) - one output page per side")
        rows = build_full_pallet_rows(fp_pages, fp_matrix_idx)

        st.dataframe(
            pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
            use_container_width=True,
            height=420,
        )

        if generate:
            with st.spinner("Rendering PDF..."):
                try:
                    pdf = render_full_pallet_display_pdf(
                        fp_pages,
                        images_bytes,
                        fp_matrix_idx,
                        title_prefix,
                        ppt_cards,
                        gift_holders,
                        ppt_cpp_global,
                        show_debug,
                        show_layout_overlay,
                    )
                except Exception as e:
                    if show_debug:
                        st.exception(e)
                    else:
                        st.error("Error generating Full Pallet PDF. Enable debug for details.")
                    return
            st.success("Done.")
            st.download_button(
                "Download Planogram PDF",
                pdf,
                file_name=out_name if out_name.endswith(".pdf") else f"{out_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
