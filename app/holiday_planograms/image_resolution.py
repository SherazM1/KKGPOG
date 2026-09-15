from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.holiday_planograms.models import HolidayResolvedImage
from app.holiday_planograms.normalization import identifier_variants


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class HolidayImageIndex:
    root_dir: str = ""
    folder_exists: bool = False
    folder_is_dir: bool = False
    indexed_files: int = 0
    duplicate_keys: int = 0
    index: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _keys_from_path(path: Path) -> list[str]:
    keys = [path.name.lower(), path.stem.lower()]
    for token in identifier_variants(path.stem):
        keys.append(token.lower())
    return list(dict.fromkeys(key for key in keys if key))


def build_holiday_image_index(folder_path: str | None) -> HolidayImageIndex:
    root_text = str(folder_path or "").strip().strip("\"'")
    root_text = os.path.expandvars(root_text)
    root = Path(root_text).expanduser()
    result = HolidayImageIndex(root_dir=str(root))
    result.folder_exists = root.exists()
    result.folder_is_dir = root.is_dir()
    if not result.folder_exists:
        result.warnings.append(f"Local image folder is unavailable: {root}.")
        return result
    if not result.folder_is_dir:
        result.warnings.append(f"Local image path is not a directory: {root}.")
        return result
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        result.indexed_files += 1
        path_text = str(path)
        for key in _keys_from_path(path):
            existing = result.index.get(key)
            if existing and existing != path_text:
                result.duplicate_keys += 1
                continue
            result.index[key] = path_text
    return result


def resolve_holiday_image(
    product_upc: str,
    pack_upc: str,
    item_number: str,
    image_index: HolidayImageIndex,
) -> HolidayResolvedImage:
    for source, identifier in (
        ("local_product_upc", product_upc),
        ("local_pack_upc", pack_upc),
        ("local_item_number", item_number),
    ):
        for key in identifier_variants(identifier):
            path = image_index.index.get(key.lower())
            if path:
                return HolidayResolvedImage(image_path=path, status="resolved", source=source)
    return HolidayResolvedImage(image_path="", status="missing", source="unresolved")
