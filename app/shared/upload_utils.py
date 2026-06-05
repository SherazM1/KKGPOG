from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SUPPORTED_UPLOAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_NAMED_IMAGE_EXTENSIONS = SUPPORTED_UPLOAD_IMAGE_EXTENSIONS | {".pdf"}


@dataclass
class NamedImageIndex:
    images: Dict[str, Any] = field(default_factory=dict)
    names: Dict[str, List[Tuple[str, Any]]] = field(default_factory=dict)
    numeric_upcs: List[Tuple[str, Any]] = field(default_factory=list)
    ambiguous_keys: set[str] = field(default_factory=set)
    indexed_images: int = 0
    duplicate_keys: int = 0
    ignored_files: int = 0


def upc_a_check_digit(upc11: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(upc11 or ""))
    if len(digits) != 11:
        return ""
    total = sum(int(d) * (3 if idx % 2 == 0 else 1) for idx, d in enumerate(digits))
    return str((10 - (total % 10)) % 10)


def upc_a_from_11(upc11: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(upc11 or ""))
    check = upc_a_check_digit(digits)
    return f"{digits}{check}" if check else ""


def upc_digit_variants(raw: Optional[str]) -> List[str]:
    digits = re.sub(r"[^0-9]", "", str(raw or ""))
    if not digits:
        return []
    variants = [
        digits,
        digits.lstrip("0"),
        digits.zfill(12) if len(digits) in {11, 12} else "",
        digits[-11:] if len(digits) in {11, 12} else "",
        digits[-12:] if len(digits) > 12 else "",
    ]
    stripped = digits.lstrip("0")
    if len(stripped) == 11:
        variants.append(upc_a_from_11(stripped))
    if len(digits) == 11:
        variants.append(upc_a_from_11(digits))

    result: List[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant and variant not in seen:
            result.append(variant)
            seen.add(variant)
    return result


def upc_near_match_reason(target_upc: Optional[str], image_upc: Optional[str]) -> Optional[str]:
    target_variants = [v for v in upc_digit_variants(target_upc) if len(v) >= 10]
    image_variants = [v for v in upc_digit_variants(image_upc) if len(v) >= 10]
    if not target_variants or not image_variants:
        return None

    def _core(value: str) -> str:
        digits = re.sub(r"[^0-9]", "", str(value or "")).lstrip("0")
        if len(digits) == 12:
            return digits[:11]
        return digits

    target_cores = {_core(v) for v in target_variants if len(_core(v)) >= 10}
    image_cores = {_core(v) for v in image_variants if len(_core(v)) >= 10}
    if target_cores & image_cores:
        return "same UPC core/check-digit variant"

    def _one_edit_apart(a: str, b: str) -> bool:
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            return sum(1 for x, y in zip(a, b) if x != y) <= 1
        if len(a) > len(b):
            a, b = b, a
        i = j = edits = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                edits += 1
                if edits > 1:
                    return False
                j += 1
        return True

    for target in target_cores:
        for image in image_cores:
            if len(target) >= 10 and len(image) >= 10:
                if _one_edit_apart(target, image):
                    return "near UPC core: one digit changed/added/dropped"
                shorter, longer = (target, image) if len(target) <= len(image) else (image, target)
                if len(shorter) >= 10 and shorter in longer:
                    return "near UPC core: extra leading/trailing digit"
                if len(target) >= 11 and len(image) >= 11 and target[1:10] == image[1:10]:
                    return "near UPC core: matching middle digits"
    return None


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
        candidates = upc_digit_variants(digits) + [digits[-5:]]
        for candidate in candidates:
            if candidate and candidate not in seen:
                keys.append(candidate)
                seen.add(candidate)
    return keys


def _name_key_from_image_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"(?i)\bUPC\b", " ", stem)
    stem = re.sub(r"\d{5,14}", " ", stem)
    stem = re.sub(r"\([^)]*\)", " ", stem)
    stem = re.sub(r"\$?\d+(?:\.\d{2})?", " ", stem)
    stem = re.sub(r"[^A-Za-z0-9 ]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().upper()


def _store_image_key(result: NamedImageIndex, key: str, value: Any) -> None:
    if not key or key in result.ambiguous_keys:
        return
    if key in result.images:
        result.duplicate_keys += 1
        if len(key) <= 5:
            result.images.pop(key, None)
            result.ambiguous_keys.add(key)
        return
    result.images[key] = value


def _store_name_key(result: NamedImageIndex, name_key: str, upc_keys: List[str], value: Any) -> None:
    if not name_key or not upc_keys:
        return
    strong_upc_keys = [key for key in upc_keys if len(key) >= 11]
    if not strong_upc_keys:
        return
    result.names.setdefault(name_key, []).append((strong_upc_keys[0], value))


def _store_numeric_upcs(result: NamedImageIndex, upc_keys: List[str], value: Any) -> None:
    seen_local: set[str] = set()
    for key in upc_keys:
        digits = re.sub(r"[^0-9]", "", str(key or ""))
        if len(digits) < 10 or digits in seen_local:
            continue
        result.numeric_upcs.append((digits, value))
        seen_local.add(digits)


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
                _store_image_key(result, key, entry_payload)
            _store_numeric_upcs(result, keys, entry_payload)
            _store_name_key(result, _name_key_from_image_name(entry_name), keys, entry_payload)
    return result


def build_named_image_index_from_folder(folder_path: str) -> NamedImageIndex:
    result = NamedImageIndex()
    root = Path(str(folder_path or "").strip().strip('"'))
    if not root.exists() or not root.is_dir():
        return result

    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_NAMED_IMAGE_EXTENSIONS:
            result.ignored_files += 1
            continue
        keys = _keys_from_image_name(str(path.name))
        if not keys:
            result.ignored_files += 1
            continue
        result.indexed_images += 1
        path_text = str(path)
        for key in keys:
            _store_image_key(result, key, path_text)
        _store_numeric_upcs(result, keys, path_text)
        _store_name_key(result, _name_key_from_image_name(path.name), keys, path_text)
    return result


def _label_page_sizes(labels_pdf_bytes: bytes) -> List[Tuple[float, float]]:
    import fitz

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
    import fitz

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
    import fitz
    from PIL import Image

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
