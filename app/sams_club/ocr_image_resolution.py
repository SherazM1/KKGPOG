from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from app.sams_club.image_resolution import (
    _digits_only,
    _identifier_keys,
)

SOURCE_OCR_FILENAME_UPC = "ocr_filename_upc"
SOURCE_OCR_UPC_VARIANT = "ocr_upc_variant"

OCR_CATALOG_COLUMNS = (
    "file_path",
    "filename",
    "filename_upc",
    "width",
    "height",
    "detected_text",
    "normalized_text",
    "detected_brand",
    "detected_denomination",
    "detected_pack_quantity",
    "catalog_status",
    "catalog_error",
)

OCR_CONFIDENCE_HIGH = 75
OCR_CONFIDENCE_MEDIUM = 45
MANUAL_MAPPING_COLUMNS = (
    "UPC",
    "original_upc",
    "Item Number",
    "file_path",
    "selected_image_filename",
    "selected_filename_upc",
    "approved",
    "source",
    "date_created",
    "date_updated",
    "notes",
)
GENERIC_TOKENS = {
    "GIFT",
    "CARD",
    "CARDS",
    "EGIFT",
    "HOLIDAY",
    "VARIABLE",
    "ASSORTED",
    "VALUE",
    "DIGITAL",
}


@dataclass
class SamsOcrCatalogEntry:
    file_path: str = ""
    filename: str = ""
    filename_upc: str = ""
    width: str = ""
    height: str = ""
    detected_text: str = ""
    normalized_text: str = ""
    detected_brand: str = ""
    detected_denomination: str = ""
    detected_pack_quantity: str = ""
    catalog_status: str = ""
    catalog_error: str = ""
    filename_upc_keys: list[str] = field(default_factory=list)
    text_tokens: set[str] = field(default_factory=set)


@dataclass
class SamsOcrCatalogIndex:
    loaded: bool = False
    source_name: str = ""
    rows_read: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    missing_file_rows: int = 0
    failed_rows_ignored: int = 0
    entries: list[SamsOcrCatalogEntry] = field(default_factory=list)
    exact_filename_upc: dict[str, str] = field(default_factory=dict)
    variant_filename_upc: dict[str, str] = field(default_factory=dict)
    by_brand: dict[str, list[int]] = field(default_factory=dict)
    by_denomination: dict[str, list[int]] = field(default_factory=dict)
    by_pack_quantity: dict[str, list[int]] = field(default_factory=dict)
    by_token: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SamsOcrCandidate:
    unresolved_upc: str
    item_number: str
    product_name: str
    pog: str
    side: str
    row: str
    column: str
    upc12: str
    description: str
    cpp: str
    expected_brand: str
    expected_denomination: str
    expected_pack_quantity: str
    candidate_rank: int
    candidate_file_path: str
    candidate_filename: str
    candidate_filename_upc: str
    detected_brand: str
    detected_denomination: str
    detected_pack_quantity: str
    detected_text: str
    brand_score: int
    denomination_score: int
    pack_score: int
    token_score: int
    filename_score: int
    total_score: int
    confidence_score: int
    confidence_label: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unresolved_upc": self.unresolved_upc,
            "item_number": self.item_number,
            "product_name": self.product_name,
            "pog": self.pog,
            "side": self.side,
            "row": self.row,
            "column": self.column,
            "upc12": self.upc12,
            "description": self.description,
            "cpp": self.cpp,
            "expected_brand": self.expected_brand,
            "expected_denomination": self.expected_denomination,
            "expected_pack_quantity": self.expected_pack_quantity,
            "candidate_rank": self.candidate_rank,
            "candidate_file_path": self.candidate_file_path,
            "candidate_filename": self.candidate_filename,
            "candidate_filename_upc": self.candidate_filename_upc,
            "detected_brand": self.detected_brand,
            "detected_denomination": self.detected_denomination,
            "detected_pack_quantity": self.detected_pack_quantity,
            "detected_text": self.detected_text,
            "brand_score": self.brand_score,
            "denomination_score": self.denomination_score,
            "pack_score": self.pack_score,
            "token_score": self.token_score,
            "filename_score": self.filename_score,
            "total_score": self.total_score,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "reasons": "; ".join(self.reasons),
            "review_status": "needs_review",
            "review_notes": "",
        }


def normalize_upc(value: Any) -> str:
    return _digits_only(str(value or "").strip())


def upc_comparison_keys(
    upc: Any,
    check_digit: Any = None,
    upc12: Any = None,
) -> list[str]:
    keys: list[str] = []
    for value in (upc, upc12):
        keys.extend(_identifier_keys(normalize_upc(value)))

    body = normalize_upc(upc)
    check = normalize_upc(check_digit)
    if len(body) == 11 and len(check) == 1:
        keys.extend(_identifier_keys(f"{body}{check}"))

    return _unique(keys)


def load_sams_ocr_catalog(catalog_source: Any) -> SamsOcrCatalogIndex:
    result = SamsOcrCatalogIndex()
    if catalog_source is None:
        return result

    try:
        rows, source_name = _read_catalog_rows(catalog_source)
    except Exception as exc:
        result.loaded = True
        result.warnings.append(f"OCR image catalog could not be read: {exc}")
        return result

    result.loaded = True
    result.source_name = source_name
    result.rows_read = len(rows)

    for row in rows:
        entry = _catalog_entry_from_row(row)
        if not entry.file_path:
            result.invalid_rows += 1
            continue

        if not Path(entry.file_path).is_file():
            result.invalid_rows += 1
            result.missing_file_rows += 1
            continue

        if _failed_without_useful_metadata(entry):
            result.invalid_rows += 1
            result.failed_rows_ignored += 1
            continue

        _add_entry(result, entry)

    return result


def resolve_by_ocr_catalog_upc(
    upc: Any,
    catalog_index: SamsOcrCatalogIndex | None,
) -> tuple[str, str, str]:
    if catalog_index is None or not catalog_index.entries:
        return "", "", ""

    exact_key = normalize_upc(upc)
    if exact_key:
        path = catalog_index.exact_filename_upc.get(exact_key)
        if path:
            return path, SOURCE_OCR_FILENAME_UPC, exact_key

    for key in _identifier_keys(normalize_upc(upc)):
        path = catalog_index.variant_filename_upc.get(key)
        if path:
            source = (
                SOURCE_OCR_FILENAME_UPC
                if catalog_index.exact_filename_upc.get(key) == path
                and key == exact_key
                else SOURCE_OCR_UPC_VARIANT
            )
            return path, source, key

    return "", "", ""


def build_ocr_candidates_for_records(
    records: Iterable[dict[str, Any]],
    catalog_index: SamsOcrCatalogIndex | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if catalog_index is None or not catalog_index.entries:
        return []

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for record in records:
        key = (
            normalize_upc(record.get("upc")),
            normalize_upc(record.get("item_number")),
        )
        if key in seen:
            continue
        seen.add(key)

        scored = [
            score_ocr_candidate(record, catalog_index.entries[index])
            for index in _candidate_entry_indexes(record, catalog_index)
        ]
        scored.sort(
            key=lambda candidate: (
                candidate.confidence_score,
                candidate.total_score,
                candidate.filename_score,
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(scored[:limit], start=1):
            candidate.candidate_rank = rank
            output.append(candidate.to_dict())

    return output


def score_ocr_candidate(
    product: dict[str, Any],
    image: SamsOcrCatalogEntry,
) -> SamsOcrCandidate:
    product_name = _product_text(product)
    product_tokens = _tokens(product_name)
    image_tokens = image.text_tokens | _tokens(image.filename)
    expected_brand = _normalize_brand(product.get("brand") or product_name)
    detected_brand = _normalize_brand(image.detected_brand or image.detected_text)
    expected_denomination = _detect_denomination(product_name)
    detected_denomination = _normalize_denomination(image.detected_denomination) or _detect_denomination(image.detected_text)
    expected_pack = _detect_pack_quantity(product_name)
    detected_pack = _normalize_pack_quantity(image.detected_pack_quantity) or _detect_pack_quantity(image.detected_text)
    reasons: list[str] = []

    brand_score = 0
    if expected_brand and detected_brand:
        if expected_brand == detected_brand:
            brand_score = 35
            reasons.append("brand match")
        else:
            brand_score = -45
            reasons.append("brand conflict")

    denomination_score = 0
    if expected_denomination and detected_denomination:
        if expected_denomination == detected_denomination:
            denomination_score = 35
            reasons.append("denomination match")
        else:
            denomination_score = -45
            reasons.append("denomination conflict")

    pack_score = 0
    if expected_pack and detected_pack:
        if expected_pack == detected_pack:
            pack_score = 20
            reasons.append("pack match")
        else:
            pack_score = -25
            reasons.append("pack conflict")

    overlap = product_tokens & image_tokens
    token_score = min(25, len(overlap) * 5)
    if token_score:
        reasons.append(f"shared tokens: {', '.join(sorted(overlap)[:5])}")

    filename_score = 0
    product_keys = set(upc_comparison_keys(product.get("upc"))) | set(
        upc_comparison_keys(product.get("item_number"))
    )
    image_keys = set(image.filename_upc_keys)
    if product_keys and image_keys and product_keys & image_keys:
        filename_score = 30
        reasons.append("filename identifier support")
    elif product_tokens & _tokens(image.filename):
        filename_score = 8
        reasons.append("filename text support")

    total = brand_score + denomination_score + pack_score + token_score + filename_score
    confidence_score = max(0, min(100, total))
    if confidence_score >= OCR_CONFIDENCE_HIGH:
        confidence_label = "High"
    elif confidence_score >= OCR_CONFIDENCE_MEDIUM:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return SamsOcrCandidate(
        unresolved_upc=normalize_upc(product.get("upc")),
        item_number=normalize_upc(product.get("item_number")),
        product_name=product_name,
        pog=str(product.get("pog", "") or "").strip(),
        side=str(product.get("side", "") or "").strip(),
        row=str(product.get("row", "") or "").strip(),
        column=str(product.get("column", "") or "").strip(),
        upc12=normalize_upc(product.get("upc12")),
        description=str(product.get("description") or product_name or "").strip(),
        cpp=str(product.get("cpp", "") or "").strip(),
        expected_brand=expected_brand,
        expected_denomination=expected_denomination,
        expected_pack_quantity=expected_pack,
        candidate_rank=0,
        candidate_file_path=image.file_path,
        candidate_filename=image.filename,
        candidate_filename_upc=image.filename_upc,
        detected_brand=detected_brand,
        detected_denomination=detected_denomination,
        detected_pack_quantity=detected_pack,
        detected_text=image.detected_text,
        brand_score=brand_score,
        denomination_score=denomination_score,
        pack_score=pack_score,
        token_score=token_score,
        filename_score=filename_score,
        total_score=total,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        reasons=reasons,
    )


def append_manual_image_mapping(
    mapping_path: str | Path,
    upc: Any,
    item_number: Any,
    file_path: str,
    source: str,
    original_upc: Any = "",
    filename_upc: Any = "",
    notes: str = "",
) -> None:
    path = Path(mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_mapping_rows(path)
    normalized_upc = normalize_upc(upc)
    normalized_item = normalize_upc(item_number)
    now = datetime.now().isoformat(timespec="seconds")
    created = now

    for row in rows:
        if (
            normalize_upc(row.get("UPC")) == normalized_upc
            and normalize_upc(row.get("Item Number")) == normalized_item
        ):
            created = str(row.get("date_created") or now)
            row.update(
                _mapping_row(
                    normalized_upc,
                    original_upc,
                    normalized_item,
                    file_path,
                    source,
                    filename_upc,
                    created,
                    now,
                    notes,
                )
            )
            _write_mapping_rows(path, rows)
            return

    rows.append(
        _mapping_row(
            normalized_upc,
            original_upc,
            normalized_item,
            file_path,
            source,
            filename_upc,
            created,
            now,
            notes,
        )
    )
    _write_mapping_rows(path, rows)


def remove_manual_image_mapping(
    mapping_path: str | Path,
    upc: Any,
    item_number: Any = "",
) -> int:
    path = Path(mapping_path)
    rows = _read_existing_mapping_rows(path)
    normalized_upc = normalize_upc(upc)
    normalized_item = normalize_upc(item_number)
    kept = [
        row
        for row in rows
        if not (
            normalize_upc(row.get("UPC")) == normalized_upc
            and (
                not normalized_item
                or normalize_upc(row.get("Item Number")) == normalized_item
            )
        )
    ]
    removed = len(rows) - len(kept)
    if removed:
        _write_mapping_rows(path, kept)
    return removed


def preview_image_status(path_text: str, max_pixels: int = 20_000_000) -> tuple[bool, str]:
    if not path_text:
        return False, "No image path supplied."
    path = Path(path_text)
    if not path.is_file():
        return False, f"Image file does not exist: {path}"
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > max_pixels:
                return False, f"Image preview is too large: {image.width}x{image.height}."
            image.verify()
        return True, ""
    except Exception as exc:
        return False, f"Image preview unavailable: {type(exc).__name__}: {exc}"


def _read_catalog_rows(catalog_source: Any) -> tuple[list[dict[str, str]], str]:
    if isinstance(catalog_source, (str, Path)):
        path = Path(catalog_source)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), str(path)

    if hasattr(catalog_source, "getvalue"):
        payload = bytes(catalog_source.getvalue())
        name = str(getattr(catalog_source, "name", "image_catalog.csv") or "image_catalog.csv")
        text = payload.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text))), name

    if hasattr(catalog_source, "read"):
        payload = catalog_source.read()
        if hasattr(catalog_source, "seek"):
            catalog_source.seek(0)
        name = str(getattr(catalog_source, "name", "image_catalog.csv") or "image_catalog.csv")
        text = bytes(payload).decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text))), name

    raise TypeError("Unsupported OCR catalog input.")


def _catalog_entry_from_row(row: dict[str, Any]) -> SamsOcrCatalogEntry:
    lookup = {str(key or "").strip().lower(): value for key, value in row.items()}

    def get(name: str) -> str:
        return str(lookup.get(name.lower(), "") or "").strip()

    file_path = get("file_path")
    filename = get("filename") or Path(file_path).name
    filename_upc = normalize_upc(get("filename_upc") or filename)
    text = get("detected_text")
    normalized_text = _normalize_text(get("normalized_text") or text)

    return SamsOcrCatalogEntry(
        file_path=file_path,
        filename=filename,
        filename_upc=filename_upc,
        width=get("width"),
        height=get("height"),
        detected_text=text,
        normalized_text=normalized_text,
        detected_brand=get("detected_brand"),
        detected_denomination=get("detected_denomination"),
        detected_pack_quantity=get("detected_pack_quantity"),
        catalog_status=get("catalog_status"),
        catalog_error=get("catalog_error"),
        filename_upc_keys=upc_comparison_keys(filename_upc),
        text_tokens=_tokens(" ".join([text, normalized_text, filename])),
    )


def _add_entry(index: SamsOcrCatalogIndex, entry: SamsOcrCatalogEntry) -> None:
    entry_index = len(index.entries)
    index.entries.append(entry)
    index.valid_rows += 1

    exact_key = normalize_upc(entry.filename_upc)
    if exact_key:
        index.exact_filename_upc.setdefault(exact_key, entry.file_path)
    for key in entry.filename_upc_keys:
        index.variant_filename_upc.setdefault(key, entry.file_path)

    brand = _normalize_brand(entry.detected_brand or entry.detected_text)
    if brand:
        index.by_brand.setdefault(brand, []).append(entry_index)
    denomination = _normalize_denomination(entry.detected_denomination) or _detect_denomination(entry.detected_text)
    if denomination:
        index.by_denomination.setdefault(denomination, []).append(entry_index)
    pack = _normalize_pack_quantity(entry.detected_pack_quantity) or _detect_pack_quantity(entry.detected_text)
    if pack:
        index.by_pack_quantity.setdefault(pack, []).append(entry_index)
    for token in entry.text_tokens:
        index.by_token.setdefault(token, []).append(entry_index)


def _candidate_entry_indexes(
    product: dict[str, Any],
    catalog_index: SamsOcrCatalogIndex,
) -> list[int]:
    candidate_indexes: set[int] = set()
    product_text = _product_text(product)
    brand = _normalize_brand(product.get("brand") or product_text)
    denomination = _detect_denomination(product_text)
    pack = _detect_pack_quantity(product_text)

    if brand:
        candidate_indexes.update(catalog_index.by_brand.get(brand, []))
    if denomination:
        candidate_indexes.update(catalog_index.by_denomination.get(denomination, []))
    if pack:
        candidate_indexes.update(catalog_index.by_pack_quantity.get(pack, []))
    for token in _tokens(product_text):
        candidate_indexes.update(catalog_index.by_token.get(token, []))

    if not candidate_indexes:
        candidate_indexes.update(range(len(catalog_index.entries)))

    return sorted(candidate_indexes)


def _failed_without_useful_metadata(entry: SamsOcrCatalogEntry) -> bool:
    status = entry.catalog_status.strip().lower()
    if status not in {"error", "failed", "fail"}:
        return False
    return not any(
        [
            entry.filename_upc,
            entry.detected_text,
            entry.normalized_text,
            entry.detected_brand,
            entry.detected_denomination,
            entry.detected_pack_quantity,
        ]
    )


def _product_text(product: dict[str, Any]) -> str:
    return " ".join(
        str(product.get(field_name, "") or "").strip()
        for field_name in (
            "brand",
            "description",
            "desc_1",
            "desc_2",
            "card_name",
            "name",
            "retail",
        )
        if str(product.get(field_name, "") or "").strip()
    )


def _normalize_text(value: Any) -> str:
    text = str(value or "").upper()
    text = re.sub(r"\$(\d+)\s*\.\s*00\b", r"$\1", text)
    text = re.sub(r"[^A-Z0-9$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in _normalize_text(value).split()
        if len(token) >= 3 and token not in GENERIC_TOKENS and not token.isdigit()
    }


def _normalize_brand(value: Any) -> str:
    tokens = _normalize_text(value)
    brand_aliases = {
        "PLAYSTATION": ("PLAYSTATION", "PSN"),
        "XBOX": ("XBOX",),
        "APPLEBEES": ("APPLEBEE", "APPLEBEES"),
        "MCDONALDS": ("MCDONALD", "MCDONALDS"),
        "DUNKIN": ("DUNKIN", "DUNKIN DONUTS"),
        "AMC": ("AMC",),
        "NINTENDO": ("NINTENDO",),
        "ROBLOX": ("ROBLOX",),
        "STEAM": ("STEAM",),
        "VISA": ("VISA",),
        "MASTERCARD": ("MASTERCARD", "MASTER CARD"),
        "DISNEY": ("DISNEY",),
        "STARBUCKS": ("STARBUCKS",),
        "PANERA": ("PANERA",),
        "OUTBACK": ("OUTBACK",),
        "DOORDASH": ("DOORDASH", "DOOR DASH"),
        "UBER": ("UBER",),
        "BUFFALO WILD WINGS": ("BUFFALO WILD WINGS", "BWW"),
        "TEXAS ROADHOUSE": ("TEXAS ROADHOUSE",),
        "DARDEN": ("DARDEN", "OLIVE GARDEN"),
    }
    padded = f" {tokens} "
    for brand, aliases in brand_aliases.items():
        if any(f" {alias} " in padded for alias in aliases):
            return brand
    return tokens.split(" ")[0] if tokens else ""


def _detect_denomination(value: Any) -> str:
    normalized = _normalize_text(value)
    range_match = re.search(r"\$(\d{1,3})\s*(?:TO|-)\s*\$?(\d{1,3})", normalized)
    if range_match:
        return f"${range_match.group(1)}-${range_match.group(2)}"
    multi_match = re.search(r"\b([23456])\s*X\s*\$?(\d{1,3})\b", normalized)
    if multi_match:
        return f"{multi_match.group(1)} X ${multi_match.group(2)}"
    amount_match = re.search(r"\$(\d{1,3})\b", normalized)
    if amount_match:
        return f"${amount_match.group(1)}"
    return ""


def _normalize_denomination(value: Any) -> str:
    return _detect_denomination(value)


def _detect_pack_quantity(value: Any) -> str:
    normalized = _normalize_text(value)
    multi_match = re.search(r"\b([23456])\s*X\s*\$?\d{1,3}\b", normalized)
    if multi_match:
        return multi_match.group(1)
    pack_match = re.search(r"\b([23456])\s*(?:PACK|PK)\b", normalized)
    if pack_match:
        return pack_match.group(1)
    return ""


def _normalize_pack_quantity(value: Any) -> str:
    digits = normalize_upc(value)
    if len(digits) == 1:
        return digits
    return _detect_pack_quantity(value)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = str(value or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def _read_existing_mapping_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _mapping_row(
    upc: str,
    original_upc: Any,
    item_number: str,
    file_path: str,
    source: str,
    filename_upc: Any,
    created: str,
    updated: str,
    notes: str,
) -> dict[str, str]:
    selected_path = Path(file_path)
    return {
        "UPC": upc,
        "original_upc": str(original_upc or "").strip(),
        "Item Number": item_number,
        "file_path": str(file_path),
        "selected_image_filename": selected_path.name,
        "selected_filename_upc": normalize_upc(filename_upc) or normalize_upc(selected_path.stem),
        "approved": "true",
        "source": source,
        "date_created": created,
        "date_updated": updated,
        "notes": notes,
    }


def _write_mapping_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_MAPPING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANUAL_MAPPING_COLUMNS})
