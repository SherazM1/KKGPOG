from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

import re
from typing import Optional
from app.shared.constants import DISPLAY_FULL_PALLET

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



def prepare_images(matrix_file, labels_pdf, images_pdf, display_type, image_alias_file=None, image_library_path=""):
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

    return matrix_bytes, labels_bytes, images_bytes, named_image_index, image_source_has_pdf, image_aliases
