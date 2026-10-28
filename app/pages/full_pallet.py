"""Full Pallet orchestration, including existing-PDF image transfer."""
from __future__ import annotations

import streamlit as st
from app.pages._inputs import upload_file, text_input
import pandas as pd

import re
import difflib
from app.shared.constants import DISPLAY_FULL_PALLET
from app.pages._display_inputs import prepare_images, _numeric_near_image_match, upc_a_from_11, upc_near_match_reason

def render():
    st.subheader('Full Pallet')
    display_type = DISPLAY_FULL_PALLET
    images_pdf = None
    with st.sidebar:
        st.header("Inputs")
        matrix_file = upload_file("Matrix Excel (.xlsx)", type=["xlsx"], key="full_pallet_matrix")
        labels_pdf = upload_file("Labels PDF", type=["pdf"], key="full_pallet_labels")
        pptx_file = upload_file(
            "Top Cards Blueprint (.pptx, .ppt, .pdf, images, ZIP)",
            type=["pptx", "ppt", "pdf", "zip", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            help="PPTX uses the existing PowerPoint parser. PDF/images/ZIP use the visual fallback in the same side/page order.",
            key='full_pallet_pptx_file',
        )
        gift_file = upload_file("2025 D82 POG Workbook (.xlsx)", type=["xlsx"], key='full_pallet_gift_file')
        ppt_cpp_global = st.number_input("PPT Cards CPP (Global)", min_value=0, value=0, step=1)
        show_debug = st.checkbox("Show debug details")
        show_layout_overlay = st.checkbox("Show Full Pallet layout overlay")

        st.divider()
        st.markdown("### Add Images to Existing Label POG")
        image_transfer_label_pdf = upload_file(
            "Generated Label-Only POG PDF",
            type=["pdf"],
            key="image_transfer_label_pdf",
            help="Upload the already-generated label POG. Existing labels and CPPs stay untouched.",
        )
        image_transfer_source_pdf = upload_file(
            "Official Image PDF",
            type=["pdf"],
            key="image_transfer_source_pdf",
            help="Upload the matching official planogram PDF with card images.",
        )
        image_transfer_middle = st.checkbox("Transfer middle images", value=True)
        image_transfer_bonus = st.checkbox("Transfer bonus images", value=True)
        run_image_transfer = st.button(
            "Add Images to Existing Label POG",
            width="stretch",
            key="run_full_pallet_image_transfer",
        )

        title_prefix = text_input("PDF title prefix", "POG", key="full_pallet_title")
        out_name = text_input("Output filename", "pog_export.pdf", key="full_pallet_filename")
        generate = st.button("Generate POG PDF", type="primary", width="stretch")

    if run_image_transfer:
        if not (image_transfer_label_pdf and image_transfer_source_pdf):
            st.warning("Upload both the generated label-only POG PDF and the official image PDF.")
            return
        if not (image_transfer_middle or image_transfer_bonus):
            st.warning("Select at least one section to transfer: middle or bonus.")
            return
        try:
            from app.full_pallet.image_transfer import add_images_to_existing_label_pog
        except ModuleNotFoundError as e:
            st.error(f"Image transfer dependency missing: {e.name}. Please install project requirements.")
            return

        with st.spinner("Adding middle/bonus images to existing label POG..."):
            try:
                transfer_result = add_images_to_existing_label_pog(
                    image_transfer_label_pdf.getvalue(),
                    image_transfer_source_pdf.getvalue(),
                    include_middle=bool(image_transfer_middle),
                    include_bonus=bool(image_transfer_bonus),
                )
            except Exception as e:
                if show_debug:
                    st.exception(e)
                else:
                    st.error("Unable to add images to the label POG. Enable debug for details.")
                return

        audit_df = pd.DataFrame(transfer_result.audit_rows)
        inserted_count = int((audit_df["Status"] == "INSERTED").sum()) if not audit_df.empty else 0
        skipped_count = int((audit_df["Status"] != "INSERTED").sum()) if not audit_df.empty else 0
        st.success(f"Image transfer complete: {inserted_count} slot image(s) inserted, {skipped_count} slot(s) skipped.")
        st.download_button(
            "Download POG With Images",
            transfer_result.pdf_bytes,
            file_name="pog_with_images.pdf",
            mime="application/pdf",
            width="stretch",
        )
        st.download_button(
            "Download Image Transfer Audit CSV",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="image_transfer_audit.csv",
            mime="text/csv",
            width="stretch",
        )
        if skipped_count:
            with st.expander("Skipped image slots"):
                st.dataframe(audit_df[audit_df["Status"] != "INSERTED"], width="stretch", height=260)
        return


    if not (matrix_file and labels_pdf):
        st.info("Upload Matrix XLSX + Labels PDF to begin.")
        return
    prepared = prepare_images(matrix_file, labels_pdf, images_pdf, display_type)
    if prepared is None:
        return
    matrix_bytes, labels_bytes, images_bytes, named_image_index, image_source_has_pdf, image_aliases = prepared
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
        width="stretch",
    )
    if unresolved_label_df is not None and not unresolved_label_df.empty:
        with st.expander("Unresolved label cells"):
            st.dataframe(unresolved_label_df, width="stretch", height=260)

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
                width="stretch",
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
                width="stretch",
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
        width="stretch",
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
            width="stretch",
        )
