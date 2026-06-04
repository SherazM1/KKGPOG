from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import fitz
from PIL import Image


SUPPORTED_UPLOAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class NamedImageIndex:
    images: Dict[str, Any] = field(default_factory=dict)
    indexed_images: int = 0
    duplicate_keys: int = 0
    ignored_files: int = 0


def _uploaded_name(uploaded: Any, fallback: str = "") -> str:
    return str(getattr(uploaded, "name", fallback) or fallback)


def _uploaded_bytes(uploaded: Any) -> bytes:
    if isinstance(uploaded, bytes):
        return uploaded
    if isinstance(uploaded, bytearray):
        return bytes(uploaded)
    if hasattr(uploaded, "getvalue"):
        return bytes(uploaded.getvalue())
    raise TypeError("Unsupported uploaded file object.")


def _image_entries_from_zip(payload: bytes) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename.lower()):
            if info.is_dir():
                continue
            name = info.filename
            if Path(name).suffix.lower() not in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
                continue
            entries.append((name, archive.read(info)))
    return entries


def _image_entries_from_uploads(uploaded_files: Sequence[Any]) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []
    for uploaded in uploaded_files:
        name = _uploaded_name(uploaded, "uploaded")
        ext = Path(name).suffix.lower()
        payload = _uploaded_bytes(uploaded)
        if ext == ".zip":
            entries.extend(_image_entries_from_zip(payload))
        elif ext in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
            entries.append((name, payload))
    return entries


def _keys_from_image_name(name: str) -> List[str]:
    stem = Path(name).stem
    digit_tokens = re.findall(r"\d{5,14}", stem)
    keys: List[str] = []
    seen: set[str] = set()
    for token in digit_tokens:
        digits = re.sub(r"[^0-9]", "", token)
        candidates = [digits, digits[-5:]]
        if len(digits) in {11, 12}:
            candidates.append(digits.zfill(12))
        if len(digits) > 12:
            candidates.append(digits[-12:])
        for candidate in candidates:
            if candidate and candidate not in seen:
                keys.append(candidate)
                seen.add(candidate)
    return keys


def build_named_image_index(uploaded: Any) -> NamedImageIndex:
    uploaded_files = coerce_uploaded_file_list(uploaded)
    result = NamedImageIndex()
    for uploaded_file in uploaded_files:
        name = _uploaded_name(uploaded_file, "uploaded")
        ext = Path(name).suffix.lower()
        payload = _uploaded_bytes(uploaded_file)
        entries: List[Tuple[str, bytes]] = []
        if ext == ".zip":
            entries = _image_entries_from_zip(payload)
        elif ext in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
            entries = [(name, payload)]
        else:
            result.ignored_files += 1
            continue

        for entry_name, entry_payload in entries:
            keys = _keys_from_image_name(entry_name)
            if not keys:
                result.ignored_files += 1
                continue
            result.indexed_images += 1
            for key in keys:
                if key in result.images:
                    result.duplicate_keys += 1
                    continue
                result.images[key] = entry_payload
    return result


def build_named_image_index_from_folder(folder_path: str) -> NamedImageIndex:
    result = NamedImageIndex()
    root = Path(str(folder_path or "").strip().strip('"'))
    if not root.exists() or not root.is_dir():
        return result

    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
            result.ignored_files += 1
            continue
        keys = _keys_from_image_name(str(path.name))
        if not keys:
            result.ignored_files += 1
            continue
        result.indexed_images += 1
        path_text = str(path)
        for key in keys:
            if key in result.images:
                result.duplicate_keys += 1
                continue
            result.images[key] = path_text
    return result


def _label_page_sizes(labels_pdf_bytes: bytes) -> List[Tuple[float, float]]:
    sizes: List[Tuple[float, float]] = []
    labels_doc = fitz.open(stream=labels_pdf_bytes, filetype="pdf")
    try:
        for page in labels_doc:
            rect = page.rect
            sizes.append((float(rect.width), float(rect.height)))
    finally:
        labels_doc.close()
    return sizes


def blank_images_pdf_from_labels(labels_pdf_bytes: bytes) -> bytes:
    """Create a blank image-source PDF with pages aligned to the labels PDF."""
    page_sizes = _label_page_sizes(labels_pdf_bytes)
    out_doc = fitz.open()
    try:
        if not page_sizes:
            out_doc.new_page(width=612.0, height=792.0)
        for page_w, page_h in page_sizes:
            out_doc.new_page(width=page_w, height=page_h)
        return out_doc.tobytes()
    finally:
        out_doc.close()


def images_upload_to_pdf_bytes(uploaded: Any, labels_pdf_bytes: bytes) -> bytes:
    """Return PDF bytes for an image source upload without changing crop semantics.

    PDF uploads pass through unchanged. Image and ZIP uploads are converted into a
    page-per-image PDF, using the labels PDF page sizes so existing bbox cropping
    remains in the same coordinate space.
    """
    uploaded_files = list(uploaded) if isinstance(uploaded, list) else [uploaded]
    if not uploaded_files:
        raise ValueError("No image source files were uploaded.")

    image_entries = _image_entries_from_uploads(uploaded_files)
    page_sizes = _label_page_sizes(labels_pdf_bytes)
    out_doc = fitz.open()
    try:
        for uploaded_file in uploaded_files:
            name = _uploaded_name(uploaded_file, "")
            if Path(name).suffix.lower() != ".pdf":
                continue
            src = fitz.open(stream=_uploaded_bytes(uploaded_file), filetype="pdf")
            try:
                out_doc.insert_pdf(src)
            finally:
                src.close()

        for idx, (_name, payload) in enumerate(image_entries):
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                fallback_w, fallback_h = image.size
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                normalized = io.BytesIO()
                image.save(normalized, format="PNG")
                image_bytes = normalized.getvalue()

            if idx < len(page_sizes):
                page_w, page_h = page_sizes[idx]
            elif page_sizes:
                page_w, page_h = page_sizes[-1]
            else:
                page_w, page_h = float(fallback_w), float(fallback_h)

            page = out_doc.new_page(width=page_w, height=page_h)
            page.insert_image(page.rect, stream=image_bytes, keep_proportion=False)
        if out_doc.page_count == 0:
            raise ValueError("Upload a PDF, image file, or ZIP containing image files.")
        return out_doc.tobytes()
    finally:
        out_doc.close()


def coerce_uploaded_file_list(uploaded: Any) -> List[Any]:
    if uploaded is None:
        return []
    if isinstance(uploaded, list):
        return uploaded
    if isinstance(uploaded, tuple):
        return list(uploaded)
    return [uploaded]
