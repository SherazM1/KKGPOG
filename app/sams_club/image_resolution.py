# app/sams_club/image_resolution.py

from __future__ import annotations

import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_ORIGINAL_PATH = "original_path"
SOURCE_LOCAL_BASENAME = "local_basename"
SOURCE_LOCAL_UPC = "local_upc"
SOURCE_LOCAL_ITEM_NUMBER = "local_item_number"
SOURCE_ZIP_BASENAME = "zip_basename"
SOURCE_ZIP_UPC = "zip_upc"
SOURCE_ZIP_ITEM_NUMBER = "zip_item_number"
SOURCE_UNRESOLVED = "unresolved"


@dataclass
class SamsImageIndex:
    source_name: str = ""
    root_dir: str = ""
    indexed_images: int = 0
    duplicate_key_count: int = 0
    index: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsImageZipIndex(SamsImageIndex):
    uploaded: bool = False
    zip_name: str = ""
    extracted_dir: str = ""


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
        filename = str(
            getattr(source_file, "name", "images.zip") or "images.zip"
        )
        return bytes(payload), filename

    raise TypeError(
        "Unsupported image ZIP file type. "
        "Provide bytes or an uploaded file object."
    )


def _safe_extract_zip(payload: bytes, destination: Path) -> None:
    destination = destination.resolve()

    with zipfile.ZipFile(io.BytesIO(payload), "r") as zip_ref:
        for member in zip_ref.infolist():
            member_name = member.filename
            if not member_name:
                continue

            target_path = (destination / member_name).resolve()
            if target_path != destination and destination not in target_path.parents:
                continue

            zip_ref.extract(member, destination)


def _digits_only(value: str) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "".join(character for character in text if character.isdigit())


def _identifier_keys(value: str) -> list[str]:
    digits = _digits_only(value)
    if not digits:
        return []

    candidates = [digits]

    stripped = digits.lstrip("0")
    if stripped:
        candidates.append(stripped)

    for length in (14, 13, 12, 11):
        if len(digits) >= length:
            suffix = digits[-length:]
            candidates.append(suffix)

            suffix_stripped = suffix.lstrip("0")
            if suffix_stripped:
                candidates.append(suffix_stripped)

    return _unique_lowercase(candidates)


def _filename_keys(file_path: Path) -> list[str]:
    keys = [file_path.name.lower(), file_path.stem.lower()]
    keys.extend(_identifier_keys(file_path.stem))

    for match in re.finditer(r"\d+", file_path.stem):
        keys.extend(_identifier_keys(match.group(0)))

    return _unique_lowercase(keys)


def _unique_lowercase(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        key = str(value or "").strip().lower()
        if not key or key in seen:
            continue

        seen.add(key)
        result.append(key)

    return result


def _add_image_to_index(
    image_index: SamsImageIndex,
    file_path: Path,
) -> None:
    path_text = str(file_path)

    for key in _filename_keys(file_path):
        existing = image_index.index.get(key)

        if existing and existing != path_text:
            image_index.duplicate_key_count += 1
            continue

        image_index.index[key] = path_text


def _index_image_files(
    root_dir: Path,
    image_index: SamsImageIndex,
) -> None:
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        _add_image_to_index(image_index, file_path)
        image_index.indexed_images += 1


def build_sams_local_image_index(
    image_root: str | Path,
) -> SamsImageIndex:
    root_text = str(image_root or "").strip().strip("\"'")
    root_text = os.path.expandvars(root_text)
    root_text = root_text.replace("\\", os.sep).replace("/", os.sep)
    root = Path(root_text).expanduser()

    result = SamsImageIndex(
        source_name="local_folder",
        root_dir=str(root),
    )

    if not root.exists():
        result.warnings.append(
            f"Sam's image folder does not exist: {root}"
        )
        return result

    if not root.is_dir():
        result.warnings.append(
            f"Sam's image path is not a directory: {root}"
        )
        return result

    try:
        _index_image_files(root, result)
    except OSError as exc:
        result.warnings.append(
            f"Sam's image folder could not be indexed: {exc}"
        )

    if result.indexed_images == 0:
        result.warnings.append(
            f"No images were indexed from {root}. Verify that the application process can access the mapped drive. "
            "If this is a Windows mapped drive such as Z:, use the real UNC path instead."
        )

    return result


def build_sams_image_zip_index(
    image_zip_file: Any,
) -> SamsImageZipIndex:
    result = SamsImageZipIndex(source_name="zip")

    if image_zip_file is None:
        return result

    payload, zip_name = _coerce_uploaded_bytes(image_zip_file)
    result.uploaded = True
    result.zip_name = zip_name

    if not payload:
        result.warnings.append(
            "Sam's image ZIP was uploaded but empty."
        )
        return result

    try:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="sams_img_zip_")
        )
        result.extracted_dir = str(temp_dir)
        result.root_dir = str(temp_dir)

        _safe_extract_zip(payload, temp_dir)
        _index_image_files(temp_dir, result)
    except zipfile.BadZipFile:
        result.warnings.append(
            "Sam's image ZIP is not a valid ZIP archive."
        )
    except Exception as exc:
        result.warnings.append(
            f"Sam's image ZIP processing failed: {exc}"
        )

    return result


def _can_open_image(path_text: str) -> bool:
    if not path_text:
        return False

    try:
        path = Path(path_text)

        if not path.exists() or not path.is_file():
            return False

        with Image.open(path) as image:
            image.verify()

        return True
    except Exception:
        return False


def _basename(path_text: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""

    return Path(text).name.strip()


def _lookup_by_basename(
    original_path: str,
    image_index: SamsImageIndex | None,
) -> str:
    if image_index is None or not image_index.index:
        return ""

    basename = _basename(original_path)
    if not basename:
        return ""

    return image_index.index.get(basename.lower(), "")


def _lookup_by_identifier(
    identifier: str,
    image_index: SamsImageIndex | None,
) -> str:
    if image_index is None or not image_index.index:
        return ""

    for key in _identifier_keys(identifier):
        resolved = image_index.index.get(key)
        if resolved:
            return resolved

    return ""


def _resolve_from_index(
    original_path: str,
    upc: str,
    item_number: str,
    image_index: SamsImageIndex | None,
    basename_source: str,
    upc_source: str,
    item_number_source: str,
) -> SamsResolvedImage:
    by_basename = _lookup_by_basename(
        original_path,
        image_index,
    )
    if by_basename:
        return SamsResolvedImage(
            resolved_path=by_basename,
            source=basename_source,
        )

    by_upc = _lookup_by_identifier(upc, image_index)
    if by_upc:
        return SamsResolvedImage(
            resolved_path=by_upc,
            source=upc_source,
        )

    by_item_number = _lookup_by_identifier(
        item_number,
        image_index,
    )
    if by_item_number:
        return SamsResolvedImage(
            resolved_path=by_item_number,
            source=item_number_source,
        )

    return SamsResolvedImage()


def resolve_sams_image_path(
    file_path: str,
    upc: str,
    item_number: str = "",
    zip_index: SamsImageZipIndex | None = None,
    local_index: SamsImageIndex | None = None,
) -> SamsResolvedImage:
    original = str(file_path or "").strip()

    if original and _can_open_image(original):
        return SamsResolvedImage(
            resolved_path=original,
            source=SOURCE_ORIGINAL_PATH,
        )

    local_result = _resolve_from_index(
        original_path=original,
        upc=upc,
        item_number=item_number,
        image_index=local_index,
        basename_source=SOURCE_LOCAL_BASENAME,
        upc_source=SOURCE_LOCAL_UPC,
        item_number_source=SOURCE_LOCAL_ITEM_NUMBER,
    )
    if local_result.resolved_path:
        return local_result

    zip_result = _resolve_from_index(
        original_path=original,
        upc=upc,
        item_number=item_number,
        image_index=zip_index,
        basename_source=SOURCE_ZIP_BASENAME,
        upc_source=SOURCE_ZIP_UPC,
        item_number_source=SOURCE_ZIP_ITEM_NUMBER,
    )
    if zip_result.resolved_path:
        return zip_result

    return SamsResolvedImage()
