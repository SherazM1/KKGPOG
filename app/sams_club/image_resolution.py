from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SOURCE_ORIGINAL_PATH = "original_path"
SOURCE_ZIP_BASENAME = "zip_basename"
SOURCE_ZIP_UPC = "zip_upc"
SOURCE_UNRESOLVED = "unresolved"


@dataclass
class SamsImageZipIndex:
    uploaded: bool = False
    zip_name: str = ""
    extracted_dir: str = ""
    indexed_images: int = 0
    duplicate_filename_count: int = 0
    index: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsResolvedImage:
    resolved_path: str = ""
    source: str = SOURCE_UNRESOLVED


def _coerce_uploaded_bytes(source_file: Any) -> tuple[bytes, str]:
    if source_file is None:
        return b"", ""
    if isinstance(source_file, (bytes, bytearray)):
        return bytes(source_file), "images.zip"
    if hasattr(source_file, "getvalue"):
        payload = source_file.getvalue()
        filename = str(getattr(source_file, "name", "images.zip") or "images.zip")
        return bytes(payload), filename
    raise TypeError("Unsupported image zip file type. Provide bytes or uploaded file object.")


def _safe_extract_zip(payload: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zip_ref:
        for member in zip_ref.infolist():
            member_name = member.filename
            if not member_name:
                continue
            target_path = (destination / member_name).resolve()
            if destination not in target_path.parents and target_path != destination:
                continue
            zip_ref.extract(member, destination)


def build_sams_image_zip_index(image_zip_file: Any) -> SamsImageZipIndex:
    result = SamsImageZipIndex()
    if image_zip_file is None:
        return result

    payload, zip_name = _coerce_uploaded_bytes(image_zip_file)
    result.uploaded = True
    result.zip_name = zip_name

    if not payload:
        result.warnings.append("Sam's image zip was uploaded but empty.")
        return result

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="sams_img_zip_"))
        result.extracted_dir = str(temp_dir)
        _safe_extract_zip(payload, temp_dir)
    except zipfile.BadZipFile:
        result.warnings.append("Sam's image zip is not a valid ZIP archive.")
        return result
    except Exception as exc:  # noqa: BLE001 - non-fatal zip processing path
        result.warnings.append(f"Sam's image zip processing failed: {exc}")
        return result

    for file_path in temp_dir.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        name_lower = file_path.name.lower()
        if name_lower in result.index:
            result.duplicate_filename_count += 1
            continue
        result.index[name_lower] = str(file_path)
        result.indexed_images += 1

    return result


def _can_open_image(path_text: str) -> bool:
    if not path_text:
        return False
    try:
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 - non-fatal validation path
        return False


def _basename(path_text: str) -> str:
    text = (path_text or "").strip()
    if not text:
        return ""
    return Path(text).name.strip()


def _upc_candidates(upc_text: str) -> list[str]:
    upc = (upc_text or "").strip()
    if not upc:
        return []

    candidates: list[str] = []
    for ext in (".jpg", ".jpeg", ".png"):
        candidates.append(f"{upc}{ext}")

    digits_only = "".join(ch for ch in upc if ch.isdigit())
    if digits_only and digits_only != upc:
        for ext in (".jpg", ".jpeg", ".png"):
            candidates.append(f"{digits_only}{ext}")

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_sams_image_path(file_path: str, upc: str, zip_index: SamsImageZipIndex | None) -> SamsResolvedImage:
    original = (file_path or "").strip()
    if original and _can_open_image(original):
        return SamsResolvedImage(resolved_path=original, source=SOURCE_ORIGINAL_PATH)

    if zip_index is None or not zip_index.index:
        return SamsResolvedImage()

    base = _basename(original)
    if base:
        by_basename = zip_index.index.get(base.lower())
        if by_basename:
            return SamsResolvedImage(resolved_path=by_basename, source=SOURCE_ZIP_BASENAME)

    for candidate in _upc_candidates(upc):
        resolved = zip_index.index.get(candidate.lower())
        if resolved:
            return SamsResolvedImage(resolved_path=resolved, source=SOURCE_ZIP_UPC)

    return SamsResolvedImage()
