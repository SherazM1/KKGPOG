from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sams_club.image_resolution import _calculate_upca_check_digit, _identifier_keys
from scripts.catalog_sams_images import (
    BRAND_ALIASES,
    detect_brand,
    detect_denomination,
    detect_pack_quantity,
    normalize_text,
)

CANDIDATE_COLUMNS = [
    "unresolved_upc",
    "calculated_upc12",
    "item_number",
    "product_name",
    "expected_brand",
    "expected_denomination",
    "expected_pack_quantity",
    "candidate_rank",
    "candidate_file_path",
    "candidate_filename_upc",
    "detected_brand",
    "detected_denomination",
    "detected_pack_quantity",
    "detected_text",
    "brand_score",
    "denomination_score",
    "pack_score",
    "keyword_score",
    "identifier_score",
    "total_score",
    "confidence",
    "review_status",
    "review_notes",
]

MANUAL_COLUMNS = ["UPC", "Item Number", "file_path", "approved", "source", "notes"]


def _row_get(row: dict[str, str], *names: str) -> str:
    lookup = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _digits(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "".join(ch for ch in text if ch.isdigit())


def calculated_upc12(upc: str) -> str:
    digits = _digits(upc)
    if len(digits) == 11:
        return digits + _calculate_upca_check_digit(digits)
    if len(digits) >= 12:
        return digits[-12:]
    return ""


def product_text(row: dict[str, str]) -> str:
    return " ".join(
        filter(
            None,
            [
                _row_get(row, "Product Name", "product_name", "Description"),
                _row_get(row, "Description 1", "desc_1", "Description 2", "desc_2"),
                _row_get(row, "Brand"),
                _row_get(row, "denomination"),
            ],
        )
    )


def normalize_brand(value: str) -> str:
    detected = detect_brand(value)
    if detected:
        return detected
    normalized = normalize_text(value)
    for brand, aliases in BRAND_ALIASES.items():
        if brand in normalized or any(alias.strip() in normalized for alias in aliases):
            return brand
    return normalized.split(" ")[0] if normalized else ""


def keywords_for_text(value: str) -> set[str]:
    stop = {"gift", "card", "cards", "digital", "egift", "the", "and", "for", "with"}
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 4 and token not in stop and not token.isdigit()
    }


def unresolved_key(row: dict[str, str]) -> tuple[str, str]:
    return (
        _digits(_row_get(row, "UPC", "UPC12", "upc", "unresolved_upc")),
        _digits(_row_get(row, "Item Number", "item_number")),
    )


def read_unresolved_products(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        status = _row_get(row, "status", "image_resolution_source", "resolution_method")
        if status and "unresolved" not in status.lower():
            continue
        key = unresolved_key(row)
        if key == ("", ""):
            continue
        deduped.setdefault(key, row)
    return list(deduped.values())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def score_candidate(product: dict[str, str], image: dict[str, str]) -> dict[str, Any]:
    upc = _digits(_row_get(product, "UPC", "UPC12", "upc", "unresolved_upc"))
    item_number = _digits(_row_get(product, "Item Number", "item_number"))
    name = product_text(product)
    expected_brand = normalize_brand(_row_get(product, "Brand") or name)
    expected_denomination = detect_denomination(
        _row_get(product, "denomination") or name
    )
    expected_pack = detect_pack_quantity(name)
    detected_brand = normalize_brand(image.get("detected_brand") or image.get("detected_text", ""))
    detected_denomination = image.get("detected_denomination", "")
    detected_pack = image.get("detected_pack_quantity", "")

    image_identifier = _digits(image.get("filename_upc") or image.get("filename"))
    identifier_score = 0
    product_keys = set(_identifier_keys(upc)) | set(_identifier_keys(item_number))
    image_keys = set(_identifier_keys(image_identifier))
    if product_keys and image_keys and product_keys & image_keys:
        identifier_score = 100
    elif upc and image_identifier and image_identifier.endswith(upc[-6:]):
        identifier_score = 8
    elif upc and image_identifier and image_identifier.endswith(upc[-5:]):
        identifier_score = 5

    brand_score = 0
    if expected_brand and detected_brand:
        brand_score = 45 if expected_brand == detected_brand else -45

    denomination_score = 0
    if expected_denomination and detected_denomination:
        denomination_score = 35 if expected_denomination == detected_denomination else -35

    pack_score = 0
    if expected_pack and detected_pack:
        pack_score = 20 if expected_pack == detected_pack else -20

    product_keywords = keywords_for_text(name)
    image_keywords = keywords_for_text(
        " ".join(
            [
                image.get("detected_text", ""),
                image.get("normalized_text", ""),
                image.get("filename", ""),
            ]
        )
    )
    keyword_score = min(25, len(product_keywords & image_keywords) * 5)

    total = brand_score + denomination_score + pack_score + keyword_score + identifier_score
    if (
        brand_score > 0
        and denomination_score > 0
        and brand_score + denomination_score + pack_score >= 80
    ):
        confidence = "high"
    elif brand_score > 0 and (denomination_score >= 0 or keyword_score >= 10):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "unresolved_upc": upc,
        "calculated_upc12": calculated_upc12(upc),
        "item_number": item_number,
        "product_name": name,
        "expected_brand": expected_brand,
        "expected_denomination": expected_denomination,
        "expected_pack_quantity": expected_pack,
        "candidate_file_path": image.get("file_path", ""),
        "candidate_filename_upc": image.get("filename_upc", ""),
        "detected_brand": detected_brand,
        "detected_denomination": detected_denomination,
        "detected_pack_quantity": detected_pack,
        "detected_text": image.get("detected_text", ""),
        "brand_score": brand_score,
        "denomination_score": denomination_score,
        "pack_score": pack_score,
        "keyword_score": keyword_score,
        "identifier_score": identifier_score,
        "total_score": total,
        "confidence": confidence,
        "review_status": "needs_review",
        "review_notes": "",
    }


def build_candidate_matches(
    unresolved_rows: list[dict[str, str]],
    catalog_rows: list[dict[str, str]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for product in unresolved_rows:
        scored = [score_candidate(product, image) for image in catalog_rows]
        scored.sort(key=lambda row: float(row["total_score"]), reverse=True)
        for rank, row in enumerate(scored[:limit], start=1):
            row = row.copy()
            row["candidate_rank"] = rank
            output.append(row)
    return output


def write_candidate_matches(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CANDIDATE_COLUMNS})


def ensure_manual_mapping_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS).writeheader()


def apply_approved_candidates(candidate_path: Path, mapping_path: Path) -> int:
    ensure_manual_mapping_file(mapping_path)
    existing = read_csv_rows(mapping_path)
    existing_keys = {
        (_digits(row.get("UPC", "")), _digits(row.get("Item Number", "")))
        for row in existing
    }
    additions: list[dict[str, str]] = []
    for row in read_csv_rows(candidate_path):
        if str(row.get("review_status", "")).strip().lower() != "approved":
            continue
        key = (_digits(row.get("unresolved_upc", "")), _digits(row.get("item_number", "")))
        if key in existing_keys:
            continue
        additions.append(
            {
                "UPC": row.get("unresolved_upc", ""),
                "Item Number": row.get("item_number", ""),
                "file_path": row.get("candidate_file_path", ""),
                "approved": "true",
                "source": "ocr_candidate",
                "notes": row.get("review_notes", ""),
            }
        )
        existing_keys.add(key)

    if additions:
        with mapping_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
            writer.writerows(additions)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Match unresolved Sam's products to OCR image catalog candidates.")
    parser.add_argument("--unresolved", required=False, help="Input unresolved product CSV.")
    parser.add_argument("--catalog", required=False, help="Input image_catalog.csv.")
    parser.add_argument("--output", default="unresolved/candidate_matches.csv")
    parser.add_argument("--manual-mapping", default="unresolved/manual_image_mappings.csv")
    parser.add_argument("--apply-approved", action="store_true")
    args = parser.parse_args()

    if args.apply_approved:
        count = apply_approved_candidates(Path(args.output), Path(args.manual_mapping))
        print(f"Applied {count} approved candidate mapping(s) to {args.manual_mapping}")
        return 0

    if not args.unresolved or not args.catalog:
        parser.error("--unresolved and --catalog are required unless --apply-approved is used.")

    unresolved_rows = read_unresolved_products(Path(args.unresolved))
    catalog_rows = read_csv_rows(Path(args.catalog))
    candidate_rows = build_candidate_matches(unresolved_rows, catalog_rows)
    write_candidate_matches(candidate_rows, Path(args.output))
    ensure_manual_mapping_file(Path(args.manual_mapping))
    print(f"Wrote {len(candidate_rows)} candidate row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
