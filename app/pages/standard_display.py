"""Standard Display input, review, generation, and downloads."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input
import pandas as pd

from app.shared.constants import DISPLAY_STANDARD
from app.pages._display_inputs import prepare_images

def render():
    st.subheader('Standard Display')
    display_type = DISPLAY_STANDARD
    images_pdf = None
    with st.sidebar:
        st.header("Inputs")
        matrix_file = upload_file("Matrix Excel (.xlsx)", type=["xlsx"], key="standard_display_matrix")
        labels_pdf = upload_file("Labels PDF", type=["pdf"], key="standard_display_labels")
        images_pdf = upload_file(
            "Card Images PDF / UPC Image ZIPs (optional)",
            type=["pdf", "zip", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            help=(
                "Optional. Upload one or more PDFs for existing page-crop behavior, "
                "or ZIPs/images named by UPC to place images by resolved product."
            ),
            key='standard_display_images_pdf',
        )

        title_prefix = text_input("PDF title prefix", "POG", key="standard_display_title")
        out_name = text_input("Output filename", "pog_export.pdf", key="standard_display_filename")
        generate = st.button("Generate POG PDF", type="primary", width="stretch")

    if not (matrix_file and labels_pdf):
        st.info("Upload Matrix XLSX + Labels PDF to begin.")
        return
    prepared = prepare_images(matrix_file, labels_pdf, images_pdf, display_type)
    if prepared is None:
        return
    matrix_bytes, labels_bytes, images_bytes, named_image_index, image_source_has_pdf, image_aliases = prepared
    from app.standard_display.service import prepare_standard_display, render_standard_display_pdf

    pages, matrix_idx, rows = prepare_standard_display(matrix_bytes, labels_bytes)
    if not pages:
        st.error("No 5-digit UPC tokens found in Labels PDF.")
        return

    st.subheader(f"Detected {len(pages)} side(s)")

    st.dataframe(
        pd.DataFrame(rows).sort_values(["Side", "Row", "Col"]),
        width="stretch",
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
            width="stretch",
        )
