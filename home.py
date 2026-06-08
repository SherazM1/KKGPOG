from __future__ import annotations

import pandas as pd
import streamlit as st
import difflib
import re
from typing import Optional

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.render_planogram import render_sams_planogram_pdf
from app.sams_club.service import build_sams_planogram_structure, detect_sams_pogs
from app.shared.constants import DISPLAY_FULL_PALLET, DISPLAY_SAMS_CLUB, DISPLAY_STANDARD, N_COLS
from app.shared import upload_utils as _upload_utils

NamedImageIndex = _upload_utils.NamedImageIndex
blank_images_pdf_from_labels = _upload_utils.blank_images_pdf_from_labels
build_named_image_index = _upload_utils.build_named_image_index
images_upload_to_pdf_bytes = _upload_utils.images_upload_to_pdf_bytes
upc_a_from_11 = _upload_utils.upc_a_from_11
build_named_image_index_from_folder = getattr(
    _upload_utils,
    "build_named_image_index_from_folder",
    lambda _folder_path: NamedImageIndex(),
)

try:
    upc_near_match_reason = _upload_utils.upc_near_match_reason
except AttributeError:
    def upc_near_match_reason(target_upc: object, image_upc: object) -> Optional[str]:
        target = re.sub(r"[^0-9]", "", str(target_upc or "")).lstrip("0")
        image = re.sub(r"[^0-9]", "", str(image_upc or "")).lstrip("0")
        if not target or not image:
            return None
        target_core = target[:11] if len(target) == 12 else target
        image_core = image[:11] if len(image) == 12 else image
        if target_core == image_core:
            return "same UPC core/check-digit variant"
        return None


def _numeric_near_image_match(index: NamedImageIndex, raw_upc: object):
    target_digits = re.sub(r"[^0-9]", "", str(raw_upc or "")).lstrip("0")
    numeric_upcs = list(getattr(index, "numeric_upcs", []) or [])
    if not target_digits or not numeric_upcs:
        return "", ""
    if len(target_digits) > 12:
        target_digits = target_digits[-12:]
    target_core = target_digits[:11] if len(target_digits) == 12 else target_digits
    if len(target_core) < 10:
        return "", ""

    candidates = []
    for image_upc, payload in numeric_upcs:
        image_digits = re.sub(r"[^0-9]", "", str(image_upc or "")).lstrip("0")
        if len(image_digits) > 12:
            image_digits = image_digits[-12:]
        image_core = image_digits[:11] if len(image_digits) == 12 else image_digits
        if len(image_core) < 10:
            continue
        shared_prefix = 0
        for left, right in zip(target_core, image_core):
            if left != right:
                break
            shared_prefix += 1
        if shared_prefix < 9:
            continue
        try:
            distance = abs(int(target_core) - int(image_core))
        except ValueError:
            continue
        if (shared_prefix >= 10 and distance <= 250) or distance <= 100:
            candidates.append((-shared_prefix, distance, image_upc, str(payload)))

    if not candidates:
        return "", ""
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_prefix = -candidates[0][0]
    best_distance = candidates[0][1]
    if best_distance > 250 or (best_prefix < 10 and best_distance > 100):
        return "", ""
    return f"numeric-near:{candidates[0][2]}:prefix{best_prefix}:distance{best_distance}", candidates[0][3]


def _upload_contains_pdf(uploaded: object) -> bool:
    if uploaded is None:
        return False
    files = uploaded if isinstance(uploaded, (list, tuple)) else [uploaded]
    for file_obj in files:
        name = str(getattr(file_obj, "name", "") or "").lower()
        if name.endswith(".pdf"):
            return True
    return False


# Renderer toggle: set to True to use the new HTML/Playwright renderer, False for the old ReportLab renderer
USE_HTML_PRICE_STRIP_RENDERER = True

def _load_sams_price_strip_renderer():
    if USE_HTML_PRICE_STRIP_RENDERER:
        from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf
    else:
        from app.sams_club.render_price_strips import render_sams_price_strips_pdf
    return render_sams_price_strips_pdf


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
    generate_sams_price_strips = False
    pptx_file = None
    gift_file = None
    ppt_cpp_global = 0
    show_debug = False
    show_layout_overlay = False
    image_library_path = ""
    image_alias_file = None

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
                "Sam's Pricing Workbook (.xlsx)",
                type=["xlsx"],
                help="Used by the Sam's price strip workflow from the 'Price Strip Data' sheet.",
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
            generate_sams_price_strips = st.button(
                "Generate Sam's Price Strips",
                use_container_width=True,
                key="generate_sams_price_strips",
            )
        else:
            st.divider()
            matrix_file = st.file_uploader("Matrix Excel (.xlsx)", type=["xlsx"])
            labels_pdf = st.file_uploader("Labels PDF", type=["pdf"])
            images_pdf = st.file_uploader(
                "Card Images PDF / UPC Image ZIPs (optional)",
                type=["pdf", "zip", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                accept_multiple_files=True,
                help=(
                    "Optional. Upload one or more PDFs for existing page-crop behavior, "
                    "or ZIPs/images named by UPC to place images by resolved product."
                ),
            )

            if display_type == DISPLAY_FULL_PALLET:
                image_library_path = st.text_input(
                    "Local UPC Image Library Folder (optional)",
                    value="",
                    help=(
                        "Paste a local folder path containing UPC-named card images. "
                        "The app scans it recursively and only uses images needed by the planogram."
                    ),
                )
                image_alias_file = st.file_uploader(
                    "Approved Image UPC Alias File (.csv, .xlsx)",
                    type=["csv", "xlsx"],
                    help=(
                        "Optional. Upload approved mappings with current_upc and image_upc columns. "
                        "Only approved rows are used for rendering."
                    ),
                )
                pptx_file = st.file_uploader(
                    "Top Cards Blueprint (.pptx, .ppt, .pdf, images, ZIP)",
                    type=["pptx", "ppt", "pdf", "zip", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                    help="PPTX uses the existing PowerPoint parser. PDF/images/ZIP use the visual fallback in the same side/page order.",
                )
                gift_file = st.file_uploader("2025 D82 POG Workbook (.xlsx)", type=["xlsx"])
                ppt_cpp_global = st.number_input("PPT Cards CPP (Global)", min_value=0, value=0, step=1)
                show_debug = st.checkbox("Show debug details")
                show_layout_overlay = st.checkbox("Show Full Pallet layout overlay")

            st.divider()
            title_prefix = st.text_input("PDF title prefix", "POG")
            out_name = st.text_input("Output filename", "pog_export.pdf")
            generate = st.button("Generate POG PDF", type="primary", use_container_width=True)

    if display_type == DISPLAY_SAMS_CLUB:
        st.subheader("Sam's Club Planogram Display")
        st.info("Completed planogram and price strips run as separate Sam's outputs.")

        if "sams_build_result" not in st.session_state:
            st.session_state["sams_build_result"] = None
        if "sams_pdf_result" not in st.session_state:
            st.session_state["sams_pdf_result"] = None
        if "sams_pdf_title" not in st.session_state:
            st.session_state["sams_pdf_title"] = ""
        if "sams_pdf_title_seed" not in st.session_state:
            st.session_state["sams_pdf_title_seed"] = ""
        if "sams_price_strip_build_result" not in st.session_state:
            st.session_state["sams_price_strip_build_result"] = None
        if "sams_price_strip_pdf_result" not in st.session_state:
            st.session_state["sams_price_strip_pdf_result"] = None

        st.markdown("### Completed Planogram")
        if not sams_main_source_file:
            st.warning("Upload a Sam's main source file (.accdb, .mdb, .xlsx, or .csv) to use completed planogram flow.")

        if build_sams and sams_main_source_file:
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
            st.info("Click 'Build Sam's Planogram Structure' to run the completed-planogram workflow.")
        else:
            st.success("Sam's Club structure build complete.")
            for warning in result.warnings:
                st.warning(warning)

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

        st.markdown("---")
        st.markdown("### Price Strips")
        if not sams_excel_file:
            st.info("Upload Sam's Pricing Workbook (.xlsx) to generate price strips from 'Price Strip Data'.")
        else:
            if generate_sams_price_strips:
                with st.spinner("Building Sam's price strip groups..."):
                    strip_build = build_sams_price_strip_rows(sams_excel_file)
                st.session_state["sams_price_strip_build_result"] = strip_build
                if strip_build.errors:
                    st.session_state["sams_price_strip_pdf_result"] = None
                else:
                    with st.spinner("Rendering Sam's price strips PDF..."):
                        render_sams_price_strips_pdf = _load_sams_price_strip_renderer()
                        st.session_state["sams_price_strip_pdf_result"] = render_sams_price_strips_pdf(
                            strip_build.strip_rows,
                            generated_by="Kendal King",
                        )

            strip_build = st.session_state.get("sams_price_strip_build_result")
            if strip_build is not None:
                if strip_build.errors:
                    for error in strip_build.errors:
                        st.error(error)
                else:
                    st.success(
                        f"Detected {strip_build.debug.get('strip_group_count', 0)} strip group(s) "
                        f"from {strip_build.extracted_record_count} workbook record(s)."
                    )
                    st.write("Detected strip groups:", strip_build.debug.get("detected_strip_groups", []))
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
                            use_container_width=True,
                            key="download_sams_price_strips_pdf",
                        )
        return

    if not (matrix_file and labels_pdf):
        st.info("Upload Matrix XLSX + Labels PDF to begin.")
        return

    matrix_bytes = matrix_file.getvalue()
    labels_bytes = labels_pdf.getvalue()
    image_aliases: dict[str, str] = {}
    if image_alias_file is not None:
        try:
            alias_name = str(image_alias_file.name or "").lower()
            if alias_name.endswith(".csv"):
                alias_df = pd.read_csv(image_alias_file, dtype=str)
            else:
                alias_df = pd.read_excel(image_alias_file, dtype=str)
            alias_cols = {re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_"): col for col in alias_df.columns}
            current_col = alias_cols.get("current_upc") or alias_cols.get("planogram_upc") or alias_cols.get("display_upc")
            image_col = alias_cols.get("image_upc") or alias_cols.get("old_upc") or alias_cols.get("alias_upc")
            status_col = alias_cols.get("status")
            if current_col and image_col:
                for _, row in alias_df.iterrows():
                    status = str(row.get(status_col, "approved") if status_col else "approved").strip().lower()
                    if status not in {"approved", "approve", "yes", "y", "use", "1", "true"}:
                        continue
                    current = re.sub(r"[^0-9]", "", str(row.get(current_col, "") or ""))
                    image = re.sub(r"[^0-9]", "", str(row.get(image_col, "") or ""))
                    if current and image:
                        image_aliases[current] = image
                        image_aliases[current.lstrip("0")] = image
                st.caption(f"Loaded {len(set(image_aliases.values()))} approved image UPC alias(es).")
            else:
                st.warning("Alias file needs current_upc and image_upc columns.")
        except Exception as e:
            st.error(f"Unable to read image alias file: {e}")
            return
    named_image_index = NamedImageIndex()
    image_source_has_pdf = _upload_contains_pdf(images_pdf)
    if image_source_has_pdf and display_type == DISPLAY_FULL_PALLET:
        st.caption("Using uploaded image PDF as the card image source. Local UPC image library matching is ignored for this run.")
    if image_library_path and not image_source_has_pdf:
        named_image_index = build_named_image_index_from_folder(image_library_path)
        if display_type == DISPLAY_FULL_PALLET:
            if named_image_index.indexed_images:
                st.caption(f"Indexed {named_image_index.indexed_images} UPC-named image file(s) from the local library.")
            else:
                st.warning("No UPC-named images were found in the local image library folder.")
    if images_pdf:
        try:
            images_bytes = images_upload_to_pdf_bytes(images_pdf, labels_bytes)
            if not image_source_has_pdf:
                uploaded_image_index = build_named_image_index(images_pdf)
                named_image_index.images.update(uploaded_image_index.images)
                for image_name, entries in getattr(uploaded_image_index, "names", {}).items():
                    named_image_index.names.setdefault(image_name, []).extend(entries)
                if not hasattr(named_image_index, "numeric_upcs"):
                    named_image_index.numeric_upcs = []
                named_image_index.numeric_upcs.extend(getattr(uploaded_image_index, "numeric_upcs", []) or [])
                named_image_index.indexed_images += uploaded_image_index.indexed_images
                named_image_index.duplicate_keys += uploaded_image_index.duplicate_keys
                named_image_index.ignored_files += uploaded_image_index.ignored_files
        except Exception as e:
            st.error(f"Unable to read image source upload: {e}")
            return
    else:
        images_bytes = blank_images_pdf_from_labels(labels_bytes)

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
                build_full_pallet_label_audit,
                build_full_pallet_rows,
                load_full_pallet_matrix,
                parse_full_pallet_pages,
                parse_gift_holders,
                parse_ppt_cards,
                render_full_pallet_display_pdf,
                validate_ppt_cards,
            )
            from app.full_pallet.ppt_visual import load_visual_ppt_cards
        except ModuleNotFoundError as e:
            st.error(f"Full Pallet mode dependency missing: {e.name}. Please install project requirements.")
            return

        if not (pptx_file and gift_file):
            st.info("Upload Top Cards Blueprint + POG XLSX for Full Pallet mode.")
            return

        try:
            ppt_ext = f".{str(pptx_file.name).rsplit('.', 1)[-1].lower()}" if "." in str(pptx_file.name) else ""
            if ppt_ext == ".pptx":
                ppt_cards = parse_ppt_cards(pptx_file.getvalue())
            elif ppt_ext == ".ppt":
                st.error("Legacy .ppt files are not supported by the parser. Save or export the PowerPoint as .pptx or PDF.")
                return
            else:
                ppt_cards = load_visual_ppt_cards(pptx_file)
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

        label_audit_rows = build_full_pallet_label_audit(fp_pages, fp_matrix_idx)
        label_audit_df = pd.DataFrame(label_audit_rows)
        resolved_label_count = int((label_audit_df["Status"] == "RESOLVED").sum()) if not label_audit_df.empty else 0
        total_label_count = int(len(label_audit_df))
        unresolved_label_df = label_audit_df[label_audit_df["Status"] != "RESOLVED"] if not label_audit_df.empty else label_audit_df
        side_counts = label_audit_df.groupby("Side").size().to_dict() if not label_audit_df.empty else {}
        expected_side_count = max(side_counts.values()) if side_counts else 0
        side_count_issues = {
            side: count
            for side, count in side_counts.items()
            if expected_side_count and count != expected_side_count
        }
        if total_label_count and resolved_label_count == total_label_count and not side_count_issues:
            st.success(f"Label/UPC resolution passed: {resolved_label_count} of {total_label_count} parsed label cells resolved.")
        else:
            st.warning(
                f"Label/UPC resolution needs review: {resolved_label_count} of {total_label_count} parsed label cells resolved."
            )
            if side_count_issues:
                st.warning(f"Uneven label counts by side: {side_count_issues}")
        st.download_button(
            "Download Full Pallet Label Placement Audit CSV",
            label_audit_df.to_csv(index=False).encode("utf-8"),
            file_name="full_pallet_label_placement_audit.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if unresolved_label_df is not None and not unresolved_label_df.empty:
            with st.expander("Unresolved label cells"):
                st.dataframe(unresolved_label_df, use_container_width=True, height=260)

        if named_image_index.images:
            image_report_rows = []
            suggested_alias_rows = []
            seen_required = set()
            for page in fp_pages:
                for cell in page.cells:
                    if not cell.last5:
                        continue
                    from app.shared.matching import resolve_full_pallet

                    match = resolve_full_pallet(cell.last5, cell.name, fp_matrix_idx)
                    if match is None:
                        continue
                    upc = str(match.upc12 or "").strip()
                    if upc and upc not in seen_required:
                        seen_required.add(upc)
                        stripped = upc.lstrip("0")
                        expected_image_upc = upc_a_from_11(stripped) if len(stripped) == 11 else upc
                        alias_upc = image_aliases.get(upc) or image_aliases.get(stripped)
                        alias_expected_upc = ""
                        if alias_upc:
                            alias_stripped = alias_upc.lstrip("0")
                            alias_expected_upc = upc_a_from_11(alias_stripped) if len(alias_stripped) == 11 else alias_upc
                        keys = [
                            expected_image_upc,
                            upc,
                            stripped,
                            alias_expected_upc,
                            alias_upc or "",
                            stripped[-5:] if len(stripped) >= 5 else "",
                        ]
                        matched_key = next((key for key in keys if key and key in named_image_index.images), "")
                        matched_file = str(named_image_index.images.get(matched_key, "")) if matched_key else ""
                        match_status = "FOUND" if matched_key else "MISSING"
                        target_name = re.sub(r"[^A-Z0-9 ]+", " ", str(match.display_name or "").upper())
                        target_name = re.sub(r"\s+", " ", target_name).strip()
                        if not matched_key and target_name and named_image_index.names:
                            near_matches = []
                            for image_name, entries in named_image_index.names.items():
                                image_norm = re.sub(r"[^A-Z0-9 ]+", " ", str(image_name or "").upper())
                                image_norm = re.sub(r"\s+", " ", image_norm).strip()
                                if not image_norm:
                                    continue
                                ratio = difflib.SequenceMatcher(None, target_name, image_norm).ratio()
                                if target_name in image_norm or image_norm in target_name:
                                    ratio = max(ratio, 0.88)
                                if ratio < 0.78:
                                    continue
                                for image_upc, payload in entries:
                                    reason = upc_near_match_reason(upc, image_upc)
                                    if reason:
                                        near_matches.append((ratio, image_upc, str(payload), reason))
                            near_matches.sort(key=lambda item: item[0], reverse=True)
                            if near_matches:
                                top_ratio, image_upc, payload, reason = near_matches[0]
                                second_ratio = near_matches[1][0] if len(near_matches) > 1 else 0.0
                                if top_ratio >= 0.86 or top_ratio - second_ratio >= 0.08:
                                    matched_key = f"name+near-upc:{image_upc}"
                                    matched_file = payload
                                    match_status = "FOUND_NEAR"
                        if not matched_key:
                            numeric_key, numeric_file = _numeric_near_image_match(named_image_index, expected_image_upc or upc)
                            if numeric_key:
                                matched_key = numeric_key
                                matched_file = numeric_file
                                match_status = "REVIEW_NUMERIC_NEAR"
                        image_report_rows.append(
                            {
                                "Status": match_status,
                                "Display UPC": upc,
                                "Expected Image UPC": expected_image_upc,
                                "Approved Alias Image UPC": alias_upc or "",
                                "Last5": stripped[-5:] if len(stripped) >= 5 else "",
                                "Card Name": match.display_name,
                                "Matched Key": matched_key,
                                "Matched File": matched_file,
                            }
                        )
                        if match_status == "REVIEW_NUMERIC_NEAR":
                            candidate_upc_match = re.search(r"numeric-near:(\d+)", matched_key)
                            candidate_upc = candidate_upc_match.group(1) if candidate_upc_match else ""
                            suggested_alias_rows.append(
                                {
                                    "status": "review",
                                    "current_upc": upc,
                                    "current_name": match.display_name,
                                    "image_upc": candidate_upc,
                                    "suggested_image_upc": candidate_upc,
                                    "suggested_image_name": "",
                                    "confidence": 0.5,
                                    "matched_file": matched_file,
                                    "reason": "review only: numeric-near candidate; not used for rendering unless approved",
                                }
                            )
                        if match_status == "MISSING" and named_image_index.names:
                            suggestions = []
                            for image_name, entries in named_image_index.names.items():
                                image_norm = re.sub(r"[^A-Z0-9 ]+", " ", str(image_name or "").upper())
                                image_norm = re.sub(r"\s+", " ", image_norm).strip()
                                if not image_norm:
                                    continue
                                ratio = difflib.SequenceMatcher(None, target_name, image_norm).ratio()
                                if target_name in image_norm or image_norm in target_name:
                                    ratio = max(ratio, 0.88)
                                if ratio < 0.72:
                                    continue
                                for image_upc, payload in entries:
                                    near_reason = upc_near_match_reason(upc, image_upc)
                                    if near_reason:
                                        ratio = max(ratio, 0.86)
                                    suggestions.append((ratio, image_upc, image_name, str(payload), near_reason))
                            suggestions.sort(key=lambda item: item[0], reverse=True)
                            for ratio, image_upc, image_name, payload, near_reason in suggestions[:5]:
                                suggested_alias_rows.append(
                                    {
                                        "status": "review",
                                        "current_upc": upc,
                                        "current_name": match.display_name,
                                        "image_upc": image_upc,
                                        "suggested_image_upc": image_upc,
                                        "suggested_image_name": image_name,
                                        "confidence": round(float(ratio), 4),
                                        "matched_file": payload,
                                        "reason": near_reason or "filename name similarity; review before approval",
                                    }
                                )

            if suggested_alias_rows:
                deduped_suggestions: dict[tuple[str, str], dict] = {}
                for row in suggested_alias_rows:
                    key = (str(row["current_upc"]), str(row["suggested_image_upc"]))
                    existing = deduped_suggestions.get(key)
                    if existing is None or float(row["confidence"]) > float(existing["confidence"]):
                        deduped_suggestions[key] = row

                by_current: dict[str, list[dict]] = {}
                for row in deduped_suggestions.values():
                    by_current.setdefault(str(row["current_upc"]), []).append(row)

                cleaned_suggestion_rows = []
                for current_upc, rows_for_upc in by_current.items():
                    rows_for_upc.sort(key=lambda row: float(row["confidence"]), reverse=True)
                    top_conf = float(rows_for_upc[0]["confidence"]) if rows_for_upc else 0.0
                    second_conf = float(rows_for_upc[1]["confidence"]) if len(rows_for_upc) > 1 else 0.0
                    for idx, row in enumerate(rows_for_upc):
                        conf = float(row["confidence"])
                        reason_text = str(row.get("reason") or "")
                        if "review only: numeric-near candidate" in reason_text:
                            row["status"] = "review"
                            row["reason"] = reason_text
                        elif (
                            idx == 0
                            and "near UPC core" in reason_text
                            and conf >= 0.88
                            and (len(rows_for_upc) == 1 or top_conf - second_conf >= 0.04)
                        ):
                            row["status"] = "approved"
                            row["reason"] = f"auto-approved: {reason_text} plus filename name match"
                        elif idx == 0 and conf >= 0.92 and (len(rows_for_upc) == 1 or top_conf - second_conf >= 0.025):
                            row["status"] = "approved"
                            row["reason"] = "auto-approved: single high-confidence filename name match; UPC alias still required"
                        elif conf >= 0.84:
                            row["status"] = "review"
                            row["reason"] = f"review: {reason_text}" if reason_text else "review: plausible filename name match"
                        else:
                            row["status"] = "ignore"
                            row["reason"] = "ignore: low filename name similarity"
                        cleaned_suggestion_rows.append(row)
                suggested_alias_rows = cleaned_suggestion_rows

            matched_image_count = sum(1 for row in image_report_rows if str(row["Status"]).startswith("FOUND"))
            missing_image_rows = [row for row in image_report_rows if row["Status"] == "MISSING"]
            review_image_count = sum(1 for row in image_report_rows if str(row["Status"]).startswith("REVIEW"))
            st.caption(
                f"Local image library safely matches {matched_image_count} of {len(image_report_rows)} unique planogram UPC(s); {review_image_count} numeric candidate(s) need review."
            )
            if image_report_rows:
                report_df = pd.DataFrame(image_report_rows).sort_values(["Status", "Display UPC"])
                st.download_button(
                    "Download Local Image Match Report CSV",
                    report_df.to_csv(index=False).encode("utf-8"),
                    file_name="local_image_match_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if suggested_alias_rows:
                suggestion_df = pd.DataFrame(suggested_alias_rows).sort_values(
                    ["confidence", "current_upc"],
                    ascending=[False, True],
                )
                st.download_button(
                    "Download Suggested Image Alias CSV",
                    suggestion_df.to_csv(index=False).encode("utf-8"),
                    file_name="suggested_image_aliases.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if missing_image_rows:
                with st.expander("Missing local images by UPC"):
                    for row in missing_image_rows[:300]:
                        st.write(
                            f"{row['Display UPC']} | expected image UPC {row['Expected Image UPC']} | {row['Card Name']}"
                        )
                    if len(missing_image_rows) > 300:
                        st.write(f"...and {len(missing_image_rows) - 300} more.")

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
                        labels_bytes,
                        fp_matrix_idx,
                        title_prefix,
                        ppt_cards,
                        gift_holders,
                        ppt_cpp_global,
                        show_debug,
                        show_layout_overlay,
                        None if image_source_has_pdf else (named_image_index if named_image_index.images else None),
                        image_aliases if image_aliases else None,
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
