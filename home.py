from __future__ import annotations

import pandas as pd
import streamlit as st

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
    sams_access_file = None
    sams_excel_file = None
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
            sams_access_file = st.file_uploader(
                "Sam's Access Database (.accdb/.mdb)",
                type=["accdb", "mdb"],
                key="sams_access_file",
            )
            sams_excel_file = st.file_uploader(
                "Sam's Excel Workbook (optional support file)",
                type=["xlsx"],
                key="sams_excel_file",
            )
            if sams_access_file:
                detected_pogs, detect_warnings = detect_sams_pogs(sams_access_file)
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
        st.info("New workflow in progress. Structure-only scaffold is available; PDF rendering is not implemented yet.")

        if not sams_access_file:
            st.warning("Upload a Sam's Access database file (.accdb/.mdb) to begin.")
            return

        st.caption("Optional support file accepted: Excel workbook (.xlsx).")

        if build_sams:
            with st.spinner("Building placeholder Sam's Club structure..."):
                result = build_sams_planogram_structure(
                    sams_access_file,
                    sams_excel_file,
                    selected_pog=sams_selected_pog,
                )
            st.success("Sam's Club structure build complete.")
            for warning in result.warnings:
                st.warning(warning)
            st.write("Detected POGs:", result.detected_pogs)
            st.write("Selected POG:", result.selected_pog or "(none)")
            st.write("Side counts:", result.debug.get("side_counts", {}))
            st.write("Rows per side:", result.debug.get("rows_per_side", {}))
            st.write("Populated columns per row:", result.debug.get("populated_columns_per_row", {}))
            st.write("Warnings:", result.debug.get("warnings", []))
            st.json(result.to_dict())
        else:
            st.info("Click 'Build Sam's Planogram Structure' to run the scaffold workflow.")
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
