"""Review a built Sam's job and persist mappings using the existing policy."""
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

from app.sams_club.ocr_image_resolution import append_manual_image_mapping, remove_manual_image_mapping, preview_image_status

def invalidate_output():
    st.session_state["sams_build_result"] = None
    st.session_state["sams_pdf_result"] = None
    st.session_state["sams_planogram:stale"] = True


def render():
    st.subheader("Image Review / Mapping")
    context = st.session_state.get("sams_review_context")
    result = st.session_state.get("sams_review_result")
    if not context or result is None:
        st.info("Build a Sam's Planogram first, then return here to review its images.")
        return
    st.caption(f"Reviewing Sam's planogram: {result.selected_pog or result.planogram.pog}")
    st.info("Saved mapping changes require a planogram rebuild before generating an updated PDF.")
    from app.pages.sams_planogram import rebuild_job
    def rebuild():
        rebuild_job(context)
    image_debug = result.debug.get("image_resolution", {})
    candidate_path = Path("unresolved/candidate_matches.csv")
    manual_mapping_path = Path("unresolved/manual_image_mappings.csv")
    ocr_candidates = image_debug.get("ocr_candidates", [])
    if ocr_candidates:
        st.markdown("#### Review Missing Images")
        unresolved_keys = {
            (
                str(candidate.get("unresolved_upc", "")),
                str(candidate.get("item_number", "")),
            )
            for candidate in ocr_candidates
        }
        accepted_count = image_debug.get("manual_mapping_approved_count", 0)
        st.write(
            {
                "products_needing_review": len(unresolved_keys),
                "accepted_saved_mappings": accepted_count,
                "still_unresolved": image_debug.get("unresolved", 0),
                "candidate_rows": len(ocr_candidates),
            }
        )
        grouped_ocr_candidates: dict[tuple[str, str], list[dict[str, object]]] = {}
        for candidate in ocr_candidates:
            grouped_ocr_candidates.setdefault(
                (
                    str(candidate.get("unresolved_upc", "")),
                    str(candidate.get("item_number", "")),
                ),
                [],
            ).append(candidate)
        review_groups = list(grouped_ocr_candidates.items())
        review_index_key = "sams_ocr_review_index"
        current_review_index = int(st.session_state.get(review_index_key, 0) or 0)
        current_review_index = max(0, min(current_review_index, len(review_groups) - 1))
        nav_cols = st.columns([1, 1, 4])
        if nav_cols[0].button("Previous", key="sams_ocr_previous"):
            st.session_state[review_index_key] = max(0, current_review_index - 1)
            st.rerun()
        if nav_cols[1].button("Next", key="sams_ocr_next"):
            st.session_state[review_index_key] = min(len(review_groups) - 1, current_review_index + 1)
            st.rerun()
        nav_cols[2].write(
            f"Product {current_review_index + 1} of {len(review_groups)}"
        )
        st.session_state[review_index_key] = current_review_index

        for group_index, ((upc_value, item_value), group_rows) in [
            (current_review_index + 1, review_groups[current_review_index])
        ]:
            label = f"{group_index} of {len(grouped_ocr_candidates)} | UPC {upc_value or '-'} | Item {item_value or '-'}"
            with st.expander(label):
                product_name = str(group_rows[0].get("product_name", ""))
                st.write(
                    {
                        "POG": group_rows[0].get("pog", result.selected_pog),
                        "side": group_rows[0].get("side", ""),
                        "row": group_rows[0].get("row", ""),
                        "column": group_rows[0].get("column", ""),
                        "item_number": item_value,
                        "UPC": upc_value,
                        "12_digit_UPC": group_rows[0].get("upc12", ""),
                        "product": product_name,
                        "description": group_rows[0].get("description", ""),
                        "CPP": group_rows[0].get("cpp", ""),
                        "expected_brand": group_rows[0].get("expected_brand", ""),
                        "expected_denomination": group_rows[0].get("expected_denomination", ""),
                        "expected_pack_quantity": group_rows[0].get("expected_pack_quantity", ""),
                    }
                )
                for candidate in sorted(group_rows, key=lambda row: int(row.get("candidate_rank", 0) or 0)):
                    candidate_path_text = str(candidate.get("candidate_file_path", ""))
                    preview_ok, preview_error = preview_image_status(candidate_path_text)
                    cols = st.columns([1, 2])
                    if preview_ok:
                        cols[0].image(candidate_path_text, caption=str(candidate.get("candidate_filename", "")), width=160)
                    else:
                        cols[0].warning(preview_error)
                    cols[1].write(
                        {
                            "rank": candidate.get("candidate_rank", ""),
                            "file_path": candidate_path_text,
                            "filename_upc": candidate.get("candidate_filename_upc", ""),
                            "detected_brand": candidate.get("detected_brand", ""),
                            "detected_denomination": candidate.get("detected_denomination", ""),
                            "detected_pack_quantity": candidate.get("detected_pack_quantity", ""),
                            "confidence_score": candidate.get("confidence_score", ""),
                            "confidence_label": candidate.get("confidence_label", ""),
                            "reasons": candidate.get("reasons", ""),
                            "detected_text": str(candidate.get("detected_text", ""))[:300],
                        }
                    )
                    accept_key = f"sams_ocr_accept_{group_index}_{candidate.get('candidate_rank', '')}"
                    reject_key = f"sams_ocr_reject_{group_index}_{candidate.get('candidate_rank', '')}"
                    accept_col, reject_col = st.columns(2)
                    if accept_col.button("Approve", key=accept_key):
                        append_manual_image_mapping(
                            manual_mapping_path,
                            upc=upc_value,
                            original_upc=upc_value,
                            item_number=item_value,
                            file_path=candidate_path_text,
                            filename_upc=candidate.get("candidate_filename_upc", ""),
                            source="OCR_ACCEPTED",
                            notes=str(candidate.get("reasons", "")),
                        )
                        invalidate_output()
                        st.success("Saved approved fallback mapping. Rebuild before generating the PDF.")
                    if reject_col.button("Reject candidate", key=reject_key):
                        st.info("Candidate rejected for this review session.")
                manual_path = st.text_input(
                    "Browse/manual image path",
                    key=f"sams_manual_path_{group_index}",
                    help="Paste a verified local image path to save as a manual fallback for this product.",
                )
                manual_cols = st.columns(3)
                if manual_cols[0].button("Save manual image", key=f"sams_manual_save_{group_index}"):
                    if Path(manual_path).is_file():
                        append_manual_image_mapping(
                            manual_mapping_path,
                            upc=upc_value,
                            original_upc=upc_value,
                            item_number=item_value,
                            file_path=manual_path,
                            source="MANUAL_BROWSE",
                            notes="Saved from Streamlit review",
                        )
                        invalidate_output()
                        st.success("Saved manual image mapping. Rebuild before generating the PDF.")
                    else:
                        st.error("Manual image path does not exist.")
                if manual_cols[1].button("Clear existing selection", key=f"sams_manual_clear_{group_index}"):
                    removed = remove_manual_image_mapping(manual_mapping_path, upc_value, item_value)
                    invalidate_output()
                    st.info(f"Removed {removed} saved mapping(s).")
                if manual_cols[2].button("Skip for now", key=f"sams_manual_skip_{group_index}"):
                    st.info("Skipped for now.")
    if candidate_path.exists():
        candidate_df = pd.read_csv(candidate_path).fillna("")
        if not candidate_df.empty:
            st.markdown("#### Unresolved Image Candidate Review")
            grouped_candidates = list(
                candidate_df.groupby(["unresolved_upc", "item_number"], dropna=False)
            )
            for group_index, ((upc_value, item_value), group_df) in enumerate(grouped_candidates[:10]):
                label = f"UPC {upc_value or '-'} | Item {item_value or '-'}"
                with st.expander(label):
                    for row_index, candidate in group_df.sort_values("candidate_rank").head(5).iterrows():
                        st.write(
                            {
                                "candidate_rank": candidate.get("candidate_rank", ""),
                                "candidate_file_path": candidate.get("candidate_file_path", ""),
                                "detected_text": candidate.get("detected_text", ""),
                                "detected_denomination": candidate.get("detected_denomination", ""),
                                "total_score": candidate.get("total_score", ""),
                                "confidence": candidate.get("confidence", ""),
                                "review_status": candidate.get("review_status", ""),
                            }
                        )
                        approve_key = f"sams_candidate_approve_{group_index}_{row_index}"
                        reject_key = f"sams_candidate_reject_{group_index}_{row_index}"
                        approve_col, reject_col = st.columns(2)
                        if approve_col.button("Approve", key=approve_key):
                            manual_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                            existing_manual = (
                                pd.read_csv(manual_mapping_path).fillna("")
                                if manual_mapping_path.exists()
                                else pd.DataFrame(
                                    columns=["UPC", "Item Number", "file_path", "approved", "source", "notes"]
                                )
                            )
                            new_mapping = pd.DataFrame(
                                [
                                    {
                                        "UPC": candidate.get("unresolved_upc", ""),
                                        "Item Number": candidate.get("item_number", ""),
                                        "file_path": candidate.get("candidate_file_path", ""),
                                        "approved": "true",
                                        "source": "streamlit_ocr_review",
                                        "notes": candidate.get("review_notes", ""),
                                    }
                                ]
                            )
                            pd.concat([existing_manual, new_mapping], ignore_index=True).drop_duplicates(
                                subset=["UPC", "Item Number", "file_path"],
                                keep="last",
                            ).to_csv(manual_mapping_path, index=False)
                            candidate_df.loc[row_index, "review_status"] = "approved"
                            candidate_df.to_csv(candidate_path, index=False)
                            invalidate_output()
                            st.success("Saved approved image mapping.")
                        if reject_col.button("Reject", key=reject_key):
                            candidate_df.loc[row_index, "review_status"] = "rejected"
                            candidate_df.to_csv(candidate_path, index=False)
                            st.info("Candidate rejected.")



    if not ocr_candidates and not candidate_path.exists():
        st.info("No unresolved image candidates are available for this build.")
    if st.button("Rebuild Sam's Planogram With Saved Mappings", type="primary", key="rebuild_sams_review"):
        rebuild()
        st.rerun()
