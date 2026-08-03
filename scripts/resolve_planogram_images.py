from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sams_club.image_resolution import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SOURCE_UNRESOLVED,
    _calculate_upca_check_digit,
    _identifier_keys,
    build_sams_local_image_index,
    load_sams_manual_image_mappings,
    resolve_sams_image_path,
)

REVIEW_COLUMNS = [
    "review_group_id",
    "unresolved_upc",
    "calculated_upc12",
    "item_number",
    "product_name",
    "description",
    "expected_brand",
    "expected_denomination",
    "expected_pack_quantity",
    "cpp",
    "occurrence_count",
    "pogs",
    "positions",
    "candidate_rank",
    "candidate_file_path",
    "candidate_filename",
    "candidate_filename_upc",
    "detected_brand",
    "detected_denomination",
    "detected_pack_quantity",
    "detected_text",
    "identifier_score",
    "brand_score",
    "denomination_score",
    "pack_score",
    "text_score",
    "total_score",
    "confidence",
    "score_reasons",
    "selected",
    "approved",
    "review_notes",
]

MAPPING_COLUMNS = ["UPC", "Item Number", "file_path", "approved", "source", "notes"]
TRUTHY = {"TRUE", "YES", "Y", "1", "APPROVED", "SELECTED"}
KNOWN_BLANK_IDENTIFIERS = {"SAMTEMP6"}
GENERIC_TOKENS = {
    "GIFT",
    "CARD",
    "CARDS",
    "HOLIDAY",
    "DIGITAL",
    "EGIFT",
    "VALUE",
    "VARIABLE",
    "ASSORTED",
    "ONLINE",
    "EXCHANGE",
    "BARCODE",
}

FIELD_ALIASES = {
    "pog": ("POG", "POG NAME", "PLANOGRAM", "PLANOGRAM ID"),
    "side": ("SIDE", "SEGMENT"),
    "row": ("ROW",),
    "column": ("COLUMN", "COL"),
    "item_number": ("ITEM NUMBER", "MERCHANT SKU", "ITEM", "SKU"),
    "upc": ("UPC", "UPC 11", "UPC11"),
    "upc12": ("12 DIGIT UPC", "UPC12", "UPC 12"),
    "product_name": ("NAME", "PRODUCT NAME", "DESCRIPTION", "DESC 1", "DESC 2", "POSITION REPORT NAME", "PALLET PRODUCT"),
    "description": ("DESCRIPTION", "DESC 1", "DESC 2", "PRODUCT DESC", "PALLET PRODUCT"),
    "brand": ("BRAND", "VENDOR"),
    "cpp": ("CPP", "CARDS PER PEG", "QTY PER FACING"),
    "merchant_category": ("MERCHANT CATEGORY", "CATEGORY", "SECTION"),
    "segment": ("MERCHANT SEGMENT", "SEGMENT", "SECTION"),
    "section": ("SECTION",),
    "intentional_blank": ("INTENTIONAL BLANK", "INTENTIONAL_BLANK"),
    "image_status": ("IMAGE STATUS", "IMAGE_STATUS", "STATUS"),
    "file_path": ("FILE_PATH", "FILE PATH", "IMAGE PATH", "FILEPATH"),
}

REQUIRED_FIELDS = ("pog", "side", "row", "column", "item_number", "upc", "product_name", "cpp")

BRAND_ALIASES = {
    "PLAYSTATION": ("PLAYSTATION", "PSN", "PS"),
    "XBOX": ("XBOX",),
    "APPLEBEES": ("APPLEBEE", "APPLEBEES", "APPLEBEE'S"),
    "MCDONALDS": ("MCDONALD", "MCDONALDS", "MCDONALD'S"),
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
}


@dataclass
class ProductGroup:
    group_id: str
    upc: str
    upc12: str
    item_number: str
    product_name: str
    description: str
    brand: str
    cpp: str
    merchant_category: str = ""
    segment: str = ""
    positions: list[str] = field(default_factory=list)
    pogs: set[str] = field(default_factory=set)


@dataclass
class CatalogEntry:
    file_path: str
    filename: str
    filename_upc: str = ""
    detected_text: str = ""
    normalized_text: str = ""
    detected_brand: str = ""
    detected_denomination: str = ""
    detected_pack_quantity: str = ""
    filename_keys: set[str] = field(default_factory=set)
    text_keys: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)


@dataclass
class CatalogIndex:
    rows_loaded: int = 0
    valid_image_rows: int = 0
    invalid_paths: int = 0
    error_rows: int = 0
    usable_ocr_rows: int = 0
    entries: list[CatalogEntry] = field(default_factory=list)
    by_identifier: dict[str, set[int]] = field(default_factory=dict)
    by_brand: dict[str, set[int]] = field(default_factory=dict)
    by_denomination: dict[str, set[int]] = field(default_factory=dict)
    by_pack: dict[str, set[int]] = field(default_factory=dict)
    by_token: dict[str, set[int]] = field(default_factory=dict)


@dataclass
class ReviewSummary:
    workbook_rows: int = 0
    unique_products: int = 0
    intentional_blanks_excluded: int = 0
    gci_rows_excluded: int = 0
    products_already_resolved: int = 0
    unique_products_needing_review: int = 0
    candidates_written: int = 0
    high_confidence_candidate_groups: int = 0
    medium_confidence_candidate_groups: int = 0
    low_confidence_only_groups: int = 0
    products_with_no_candidates: int = 0
    output_path: str = ""


@dataclass
class ApplySummary:
    approved_rows_read: int = 0
    mappings_added: int = 0
    mappings_updated: int = 0
    invalid_approvals_skipped: int = 0
    duplicate_approvals_rejected: int = 0
    backup_path: str = ""
    mapping_output_path: str = ""


def normalize_text(value: Any) -> str:
    text = str(value or "").upper().replace("_", " ")
    text = re.sub(r"\$(\d+)\s*\.\s*00\b", r"$\1", text)
    text = re.sub(r"[^A-Z0-9$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"[^0-9]", "", text)


def truthy(value: Any) -> bool:
    return normalize_text(value) in TRUTHY


def calculated_upc12(upc: Any) -> str:
    digits = normalize_identifier(upc)
    if len(digits) == 11:
        check = _calculate_upca_check_digit(digits)
        return f"{digits}{check}" if check else ""
    if len(digits) >= 12:
        return digits[-12:]
    return ""


def identifier_keys(value: Any) -> set[str]:
    return set(_identifier_keys(normalize_identifier(value)))


def canonical_header(value: Any) -> str:
    return normalize_text(value).replace(" ", "")


def build_column_mapping(columns: Iterable[Any]) -> dict[str, str]:
    canonical = {canonical_header(column): str(column) for column in columns}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for logical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            column = canonical.get(canonical_header(alias))
            if column and column not in used:
                mapping[logical] = column
                used.add(column)
                break
    return mapping


def load_workbook_records(path: Path) -> tuple[list[dict[str, str]], dict[str, str], str]:
    if not path.exists():
        raise FileNotFoundError(f"Workbook missing: {path}")

    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        raise ValueError(f"Unable to read workbook: {path}: {exc}") from exc

    try:
        best_missing = list(REQUIRED_FIELDS)
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(workbook, sheet_name=sheet_name, dtype=str).fillna("")
            mapping = build_column_mapping(df.columns)
            missing = [field for field in REQUIRED_FIELDS if field not in mapping]
            if not missing:
                return normalize_rows(df, mapping), mapping, sheet_name
            if len(missing) < len(best_missing):
                best_missing = missing
    finally:
        workbook.close()

    raise ValueError(
        "No usable worksheet found. Missing required columns: "
        + ", ".join(best_missing)
    )


def normalize_rows(df: pd.DataFrame, mapping: dict[str, str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw in df.to_dict(orient="records"):
        record: dict[str, str] = {}
        for logical in FIELD_ALIASES:
            source = mapping.get(logical)
            record[logical] = str(raw.get(source, "") if source else "").strip()
        record["upc"] = normalize_identifier(record.get("upc"))
        record["upc12"] = normalize_identifier(record.get("upc12"))
        if not record["upc12"]:
            record["upc12"] = calculated_upc12(record["upc"])
        record["item_number"] = normalize_identifier(record.get("item_number")) or record.get("item_number", "")
        records.append(record)
    return records


def product_text(record: dict[str, str] | ProductGroup) -> str:
    if isinstance(record, ProductGroup):
        values = [
            record.product_name,
            record.description,
            record.brand,
            record.merchant_category,
            record.segment,
        ]
    else:
        values = [
            record.get("product_name", ""),
            record.get("description", ""),
            record.get("brand", ""),
            record.get("merchant_category", ""),
            record.get("segment", ""),
        ]
    return " ".join(value for value in values if str(value or "").strip())


def normalize_brand(value: Any) -> str:
    normalized = normalize_text(value)
    padded = f" {normalized} "
    for brand, aliases in BRAND_ALIASES.items():
        if any(f" {normalize_text(alias)} " in padded for alias in aliases):
            return brand
    return normalized.split(" ")[0] if normalized else ""


def normalize_denomination(value: Any) -> str:
    normalized = normalize_text(value)
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


def normalize_pack_quantity(value: Any) -> str:
    normalized = normalize_text(value)
    multi_match = re.search(r"\b([23456])\s*X\s*\$?\d{1,3}\b", normalized)
    if multi_match:
        return multi_match.group(1)
    pack_match = re.search(r"\b([23456])\s*(?:PACK|PK)\b", normalized)
    if pack_match:
        return pack_match.group(1)
    digits = normalize_identifier(value)
    return digits if len(digits) == 1 else ""


def useful_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in GENERIC_TOKENS and not token.isdigit()
    }


def denomination_compatible(expected: str, detected: str) -> bool:
    if not expected or not detected:
        return False
    if expected == detected:
        return True
    range_match = re.fullmatch(r"\$(\d{1,3})-\$(\d{1,3})", expected)
    value_match = re.fullmatch(r"\$(\d{1,3})", detected)
    if range_match and value_match:
        value = int(value_match.group(1))
        return int(range_match.group(1)) <= value <= int(range_match.group(2))
    range_match = re.fullmatch(r"\$(\d{1,3})-\$(\d{1,3})", detected)
    value_match = re.fullmatch(r"\$(\d{1,3})", expected)
    if range_match and value_match:
        value = int(value_match.group(1))
        return int(range_match.group(1)) <= value <= int(range_match.group(2))
    return False


def is_intentional_blank(record: dict[str, str]) -> bool:
    if truthy(record.get("intentional_blank")):
        return True
    if normalize_text(record.get("image_status")) == "INTENTIONAL BLANK":
        return True
    for field_name in ("merchant_category", "segment", "section"):
        if normalize_text(record.get(field_name)) == "SAMS FP GFT":
            return True
    identifiers = {
        normalize_text(record.get("item_number")),
        normalize_text(record.get("upc")),
        normalize_text(record.get("upc12")),
    }
    return bool(identifiers & KNOWN_BLANK_IDENTIFIERS)


def is_gci(record: dict[str, str]) -> bool:
    if normalize_text(record.get("image_status")) == "GCI IMAGE PENDING":
        return True
    for field_name in ("merchant_category", "segment"):
        if "GCI" in normalize_text(record.get(field_name)).split():
            return True
    return "GCI" in normalize_text(product_text(record)).split()


def image_readable(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def supported_image_path(path_text: str) -> bool:
    path = Path(path_text)
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and path.is_file()


def load_catalog(path: Path) -> CatalogIndex:
    if not path.exists():
        raise FileNotFoundError(f"Catalog missing: {path}")

    index = CatalogIndex()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise ValueError(f"Malformed CSV: {path}: {exc}") from exc

    index.rows_loaded = len(rows)
    for row in rows:
        try:
            entry = catalog_entry_from_row(row)
            status = normalize_text(row.get("catalog_status", ""))
            if status in {"ERROR", "FAILED", "FAIL"}:
                index.error_rows += 1
                if not any(
                    [
                        entry.filename_upc,
                        entry.detected_text,
                        entry.normalized_text,
                        entry.detected_brand,
                        entry.detected_denomination,
                        entry.detected_pack_quantity,
                    ]
                ):
                    continue
            if not entry.file_path or not supported_image_path(entry.file_path):
                index.invalid_paths += 1
                continue
            add_catalog_entry(index, entry)
        except Exception:
            index.invalid_paths += 1
            continue
    return index


def catalog_entry_from_row(row: dict[str, Any]) -> CatalogEntry:
    lookup = {str(key or "").strip().lower(): value for key, value in row.items()}

    def get(name: str) -> str:
        return str(lookup.get(name.lower(), "") or "").strip()

    file_path = get("file_path")
    filename = get("filename") or Path(file_path).name
    detected_text = get("detected_text")
    normalized = get("normalized_text") or normalize_text(detected_text)
    filename_upc = normalize_identifier(get("filename_upc") or filename)
    combined_text = " ".join([filename, filename_upc, detected_text, normalized])
    return CatalogEntry(
        file_path=file_path,
        filename=filename,
        filename_upc=filename_upc,
        detected_text=detected_text,
        normalized_text=normalized,
        detected_brand=normalize_brand(get("detected_brand") or detected_text),
        detected_denomination=normalize_denomination(get("detected_denomination") or detected_text),
        detected_pack_quantity=normalize_pack_quantity(get("detected_pack_quantity") or detected_text),
        filename_keys=identifier_keys(filename_upc),
        text_keys=identifier_keys(combined_text),
        tokens=useful_tokens(combined_text),
    )


def add_catalog_entry(index: CatalogIndex, entry: CatalogEntry) -> None:
    entry_index = len(index.entries)
    index.entries.append(entry)
    index.valid_image_rows += 1
    if entry.detected_text or entry.normalized_text or entry.detected_brand:
        index.usable_ocr_rows += 1
    for key in entry.filename_keys | entry.text_keys:
        index.by_identifier.setdefault(key, set()).add(entry_index)
    if entry.detected_brand:
        index.by_brand.setdefault(entry.detected_brand, set()).add(entry_index)
    if entry.detected_denomination:
        index.by_denomination.setdefault(entry.detected_denomination, set()).add(entry_index)
    if entry.detected_pack_quantity:
        index.by_pack.setdefault(entry.detected_pack_quantity, set()).add(entry_index)
    for token in entry.tokens:
        index.by_token.setdefault(token, set()).add(entry_index)


def dedupe_products(records: Iterable[dict[str, str]]) -> dict[tuple[str, str], ProductGroup]:
    groups: dict[tuple[str, str], ProductGroup] = {}
    for record in records:
        key = (normalize_identifier(record.get("upc")), normalize_identifier(record.get("item_number")))
        if key == ("", ""):
            continue
        group = groups.get(key)
        if group is None:
            group = ProductGroup(
                group_id=f"RG{len(groups) + 1:05d}",
                upc=key[0],
                upc12=record.get("upc12", "") or calculated_upc12(key[0]),
                item_number=key[1],
                product_name=record.get("product_name", ""),
                description=record.get("description", ""),
                brand=record.get("brand", ""),
                cpp=record.get("cpp", ""),
                merchant_category=record.get("merchant_category", ""),
                segment=record.get("segment", ""),
            )
            groups[key] = group
        group.pogs.add(record.get("pog", ""))
        group.positions.append(
            f"{record.get('pog', '')}:S{record.get('side', '')}:R{record.get('row', '')}:C{record.get('column', '')}"
        )
    return groups


def candidate_pool(product: ProductGroup, catalog: CatalogIndex, minimum: int = 20) -> list[int]:
    pool: set[int] = set()
    product_keys = identifier_keys(product.upc) | identifier_keys(product.upc12) | identifier_keys(product.item_number)
    for key in product_keys:
        pool.update(catalog.by_identifier.get(key, set()))
    if len(pool) >= minimum:
        return sorted(pool)

    brand = normalize_brand(product.brand or product_text(product))
    denomination = normalize_denomination(product_text(product))
    pack = normalize_pack_quantity(product_text(product))
    if brand:
        pool.update(catalog.by_brand.get(brand, set()))
    if len(pool) >= minimum:
        return sorted(pool)
    if denomination:
        pool.update(catalog.by_denomination.get(denomination, set()))
    if pack:
        pool.update(catalog.by_pack.get(pack, set()))
    if len(pool) >= minimum:
        return sorted(pool)
    for token in useful_tokens(product_text(product)):
        pool.update(catalog.by_token.get(token, set()))
    if not pool:
        pool.update(range(len(catalog.entries)))
    return sorted(pool)


def score_candidate(product: ProductGroup, image: CatalogEntry) -> dict[str, Any]:
    expected_text = product_text(product)
    expected_brand = normalize_brand(product.brand or expected_text)
    expected_denomination = normalize_denomination(expected_text)
    expected_pack = normalize_pack_quantity(expected_text)
    product_keys = identifier_keys(product.upc)
    calculated_keys = identifier_keys(product.upc12 or calculated_upc12(product.upc))
    item_keys = identifier_keys(product.item_number)
    image_filename_keys = image.filename_keys
    image_all_keys = image.filename_keys | image.text_keys
    reasons: list[str] = []

    identifier_score = 0
    if product.upc and normalize_identifier(image.filename_upc) == product.upc:
        identifier_score = 100
        reasons.append("exact normalized UPC")
    elif calculated_keys and calculated_keys & image_filename_keys:
        identifier_score = 95
        reasons.append("calculated UPC12 match")
    elif item_keys and item_keys & image_all_keys:
        identifier_score = 90
        reasons.append("exact item number support")
    elif (product_keys | calculated_keys) and (product_keys | calculated_keys) & image_filename_keys:
        identifier_score = 85
        reasons.append("filename UPC variant")
    elif product.upc and image.filename_upc and image.filename_upc.endswith(product.upc[-6:]):
        identifier_score = 8
        reasons.append("last-six support")
    elif product.upc and image.filename_upc and image.filename_upc.endswith(product.upc[-5:]):
        identifier_score = 4
        reasons.append("last-five support")

    brand_score = 0
    if expected_brand and image.detected_brand:
        if expected_brand == image.detected_brand:
            brand_score = 40
            reasons.append("brand match")
        else:
            brand_score = -60
            reasons.append("brand conflict")

    denomination_score = 0
    if expected_denomination and image.detected_denomination:
        if expected_denomination == image.detected_denomination:
            denomination_score = 40
            reasons.append("denomination match")
        elif denomination_compatible(expected_denomination, image.detected_denomination):
            denomination_score = 25
            reasons.append("compatible denomination")
        else:
            denomination_score = -60
            reasons.append("denomination conflict")

    pack_score = 0
    if expected_pack and image.detected_pack_quantity:
        if expected_pack == image.detected_pack_quantity:
            pack_score = 25
            reasons.append("pack match")
        else:
            pack_score = -35
            reasons.append("pack conflict")

    overlap = useful_tokens(expected_text) & image.tokens
    text_score = min(35, len(overlap) * 7)
    if text_score:
        reasons.append("token overlap: " + ", ".join(sorted(overlap)[:5]))
    expected_norm = normalize_text(expected_text)
    image_norm = normalize_text(" ".join([image.detected_text, image.filename]))
    distinctive = [token for token in expected_norm.split() if token not in GENERIC_TOKENS and len(token) >= 5]
    if distinctive and " ".join(distinctive[:2]) in image_norm:
        text_score += 20
        reasons.append("distinctive phrase match")

    total = identifier_score + brand_score + denomination_score + pack_score + text_score
    major_conflict = brand_score <= -60 or denomination_score <= -60 or pack_score <= -35
    if total >= 90 and not major_conflict:
        confidence = "High"
    elif total >= 55:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "unresolved_upc": product.upc,
        "calculated_upc12": product.upc12 or calculated_upc12(product.upc),
        "item_number": product.item_number,
        "product_name": product.product_name,
        "description": product.description,
        "expected_brand": expected_brand,
        "expected_denomination": expected_denomination,
        "expected_pack_quantity": expected_pack,
        "cpp": product.cpp,
        "occurrence_count": len(product.positions),
        "pogs": " | ".join(sorted(filter(None, product.pogs))),
        "positions": " | ".join(product.positions),
        "candidate_file_path": image.file_path,
        "candidate_filename": image.filename,
        "candidate_filename_upc": image.filename_upc,
        "detected_brand": image.detected_brand,
        "detected_denomination": image.detected_denomination,
        "detected_pack_quantity": image.detected_pack_quantity,
        "detected_text": image.detected_text,
        "identifier_score": identifier_score,
        "brand_score": brand_score,
        "denomination_score": denomination_score,
        "pack_score": pack_score,
        "text_score": text_score,
        "total_score": total,
        "confidence": confidence,
        "score_reasons": "; ".join(reasons),
        "selected": "FALSE",
        "approved": "FALSE",
        "review_notes": "",
    }


def product_already_resolved(product: ProductGroup, local_index: Any, manual_index: Any) -> bool:
    resolution = resolve_sams_image_path(
        file_path="",
        upc=product.upc,
        item_number=product.item_number,
        local_index=local_index,
        manual_index=manual_index,
    )
    return resolution.source != SOURCE_UNRESOLVED


def explicit_path_resolved(records: Iterable[dict[str, str]], key: tuple[str, str]) -> bool:
    for record in records:
        if (normalize_identifier(record.get("upc")), normalize_identifier(record.get("item_number"))) != key:
            continue
        path_text = record.get("file_path", "")
        if path_text and Path(path_text).is_file() and image_readable(path_text):
            return True
    return False


def build_review(
    workbook_path: Path,
    catalog_path: Path,
    image_root: Path,
    mappings_path: Path,
    output_path: Path,
) -> tuple[list[dict[str, Any]], ReviewSummary, CatalogIndex]:
    if not image_root.exists() or not image_root.is_dir():
        raise FileNotFoundError(f"Image root missing: {image_root}")

    records, _mapping, _sheet = load_workbook_records(workbook_path)
    catalog = load_catalog(catalog_path)
    local_index = build_sams_local_image_index(image_root)
    manual_index = load_sams_manual_image_mappings(mappings_path)

    normal_records: list[dict[str, str]] = []
    summary = ReviewSummary(workbook_rows=len(records), output_path=str(output_path))
    for record in records:
        if is_intentional_blank(record):
            summary.intentional_blanks_excluded += 1
        elif is_gci(record):
            summary.gci_rows_excluded += 1
        else:
            normal_records.append(record)

    all_groups = dedupe_products(normal_records)
    summary.unique_products = len(all_groups)
    review_groups: list[ProductGroup] = []
    records_by_key = {
        key: [r for r in normal_records if (normalize_identifier(r.get("upc")), normalize_identifier(r.get("item_number"))) == key]
        for key in all_groups
    }
    for key, product in all_groups.items():
        if explicit_path_resolved(records_by_key[key], key) or product_already_resolved(product, local_index, manual_index):
            summary.products_already_resolved += 1
            continue
        review_groups.append(product)

    rows: list[dict[str, Any]] = []
    for product in review_groups:
        scored = [score_candidate(product, catalog.entries[index]) for index in candidate_pool(product, catalog)]
        scored.sort(key=lambda row: (float(row["total_score"]), float(row["identifier_score"])), reverse=True)
        if not scored:
            summary.products_with_no_candidates += 1
            continue
        top = scored[:5]
        labels = [row["confidence"] for row in top]
        if "High" in labels:
            summary.high_confidence_candidate_groups += 1
        elif "Medium" in labels:
            summary.medium_confidence_candidate_groups += 1
        else:
            summary.low_confidence_only_groups += 1
        for rank, row in enumerate(top, start=1):
            row = row.copy()
            row["review_group_id"] = product.group_id
            row["candidate_rank"] = rank
            rows.append(row)

    summary.unique_products_needing_review = len(review_groups)
    summary.candidates_written = len(rows)
    write_review_csv(rows, output_path)
    return rows, summary, catalog


def write_review_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REVIEW_COLUMNS})


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise ValueError(f"Malformed CSV: {path}: {exc}") from exc


def apply_approved(review_path: Path, mappings_path: Path) -> ApplySummary:
    if not review_path.exists():
        raise FileNotFoundError(f"Review CSV missing: {review_path}")

    review_rows = read_csv(review_path)
    summary = ApplySummary(mapping_output_path=str(mappings_path))
    approved_by_group: dict[str, list[dict[str, str]]] = {}
    for row in review_rows:
        if truthy(row.get("approved")) and truthy(row.get("selected")):
            summary.approved_rows_read += 1
            approved_by_group.setdefault(row.get("review_group_id", ""), []).append(row)

    valid_rows: list[dict[str, str]] = []
    for group_rows in approved_by_group.values():
        if len(group_rows) > 1:
            summary.duplicate_approvals_rejected += len(group_rows)
            continue
        row = group_rows[0]
        candidate_path = row.get("candidate_file_path", "")
        if not Path(candidate_path).is_file() or not image_readable(candidate_path):
            summary.invalid_approvals_skipped += 1
            continue
        valid_rows.append(row)

    existing = read_csv(mappings_path) if mappings_path.exists() else []
    now = datetime.now().isoformat(timespec="seconds")
    existing_by_key = {
        (normalize_identifier(row.get("UPC")), normalize_identifier(row.get("Item Number"))): row
        for row in existing
        if normalize_identifier(row.get("UPC")) or normalize_identifier(row.get("Item Number"))
    }

    for row in valid_rows:
        key = (normalize_identifier(row.get("unresolved_upc")), normalize_identifier(row.get("item_number")))
        notes = (
            f"product={row.get('product_name', '')}; "
            f"confidence={row.get('confidence', '')}; "
            f"score={row.get('total_score', '')}; "
            f"review_date={now}; "
            f"{row.get('review_notes', '')}"
        ).strip()
        mapping_row = {
            "UPC": key[0],
            "Item Number": key[1],
            "file_path": row.get("candidate_file_path", ""),
            "approved": "TRUE",
            "source": "OFFLINE_OCR_REVIEW",
            "notes": notes,
        }
        if key in existing_by_key:
            existing_by_key[key].update(mapping_row)
            summary.mappings_updated += 1
        else:
            existing.append(mapping_row)
            existing_by_key[key] = mapping_row
            summary.mappings_added += 1

    if mappings_path.exists():
        backup_path = mappings_path.with_name(
            f"{mappings_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak{mappings_path.suffix}"
        )
        shutil.copy2(mappings_path, backup_path)
        summary.backup_path = str(backup_path)

    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{mappings_path.stem}.", suffix=".tmp", dir=str(mappings_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
            writer.writeheader()
            for row in existing:
                writer.writerow({column: row.get(column, "") for column in MAPPING_COLUMNS})
        os.replace(temp_path, mappings_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return summary


def print_review_summary(summary: ReviewSummary, catalog: CatalogIndex) -> None:
    print(f"workbook rows: {summary.workbook_rows}")
    print(f"unique products: {summary.unique_products}")
    print(f"intentional blanks excluded: {summary.intentional_blanks_excluded}")
    print(f"GCI rows excluded: {summary.gci_rows_excluded}")
    print(f"products already resolved: {summary.products_already_resolved}")
    print(f"unique products needing review: {summary.unique_products_needing_review}")
    print(f"candidates written: {summary.candidates_written}")
    print(f"high-confidence candidate groups: {summary.high_confidence_candidate_groups}")
    print(f"medium-confidence candidate groups: {summary.medium_confidence_candidate_groups}")
    print(f"low-confidence-only groups: {summary.low_confidence_only_groups}")
    print(f"products with no candidates: {summary.products_with_no_candidates}")
    print(f"catalog rows loaded: {catalog.rows_loaded}")
    print(f"valid image rows: {catalog.valid_image_rows}")
    print(f"invalid paths: {catalog.invalid_paths}")
    print(f"error rows: {catalog.error_rows}")
    print(f"usable OCR rows: {catalog.usable_ocr_rows}")
    print(f"output path: {summary.output_path}")


def print_apply_summary(summary: ApplySummary) -> None:
    print(f"approved rows read: {summary.approved_rows_read}")
    print(f"mappings added: {summary.mappings_added}")
    print(f"mappings updated: {summary.mappings_updated}")
    print(f"invalid approvals skipped: {summary.invalid_approvals_skipped}")
    print(f"duplicate approvals rejected: {summary.duplicate_approvals_rejected}")
    print(f"backup path: {summary.backup_path}")
    print(f"mapping output path: {summary.mapping_output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve Sam's Club planogram images from the terminal.")
    parser.add_argument("--workbook", help="Merged Sam's planogram workbook.")
    parser.add_argument("--catalog", help="OCR image catalog CSV.")
    parser.add_argument("--images", help="Local image library root.")
    parser.add_argument("--mappings", default="unresolved/manual_image_mappings.csv", help="Manual mapping CSV.")
    parser.add_argument("--output", default="unresolved/planogram_image_review.csv", help="Review CSV output.")
    parser.add_argument("--review", help="Review CSV with selected approved candidates.")
    parser.add_argument("--apply-approved", action="store_true", help="Apply approved review rows to mapping CSV.")
    args = parser.parse_args(argv)

    try:
        if args.apply_approved:
            if not args.review:
                parser.error("--review is required with --apply-approved")
            summary = apply_approved(Path(args.review), Path(args.mappings))
            print_apply_summary(summary)
            return 0

        missing = [name for name in ("workbook", "catalog", "images") if not getattr(args, name)]
        if missing:
            parser.error("Missing required arguments: " + ", ".join(f"--{name}" for name in missing))
        rows, summary, catalog = build_review(
            Path(args.workbook),
            Path(args.catalog),
            Path(args.images),
            Path(args.mappings),
            Path(args.output),
        )
        print_review_summary(summary, catalog)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
