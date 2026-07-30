# scripts/catalog_sams_images.py

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sams_club.image_resolution import (
    SUPPORTED_IMAGE_EXTENSIONS,
    _digits_only,
)

CATALOG_COLUMNS = [
    "file_path",
    "filename",
    "filename_upc",
    "file_size",
    "modified_time",
    "width",
    "height",
    "detected_text",
    "normalized_text",
    "detected_brand",
    "detected_denomination",
    "detected_pack_quantity",
    "catalog_status",
    "catalog_error",
]

BRAND_ALIASES = {
    "playstation": ("playstation", "psn", "ps "),
    "xbox": ("xbox",),
    "applebees": ("applebee", "applebees"),
    "mcdonalds": ("mcdonald", "mcdonalds"),
    "dunkin": ("dunkin", "dunkin donuts"),
    "amc": ("amc",),
    "nintendo": ("nintendo",),
    "roblox": ("roblox",),
    "steam": ("steam",),
    "visa": ("visa",),
    "mastercard": ("mastercard", "master card"),
    "disney": ("disney",),
    "starbucks": ("starbucks",),
    "panera": ("panera",),
    "outback": ("outback",),
    "doordash": ("doordash", "door dash"),
    "uber": ("uber",),
    "buffalo wild wings": ("buffalo wild wings", "bww"),
    "texas roadhouse": ("texas roadhouse",),
}

DEFAULT_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_brand(text: str) -> str:
    normalized = f" {normalize_text(text)} "

    for brand, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            if f" {alias.strip()} " in normalized:
                return brand

    return ""


def detect_denomination(text: str) -> str:
    normalized = normalize_text(text)

    range_match = re.search(
        r"\$(\d{1,3})\s*\$?\s*(?:to|-)\s*\$?(\d{1,3})",
        normalized,
    )
    if range_match:
        return f"${range_match.group(1)}-${range_match.group(2)}"

    multi_match = re.search(
        r"\b([234])\s*x\s*\$?(\d{1,3})\b",
        normalized,
    )
    if multi_match:
        return f"{multi_match.group(1)} x ${multi_match.group(2)}"

    amount_match = re.search(
        r"\$(5|10|15|20|25|30|40|45|50|75|100)\b",
        normalized,
    )
    if amount_match:
        return f"${amount_match.group(1)}"

    return ""


def detect_pack_quantity(text: str) -> str:
    normalized = normalize_text(text)

    multi_match = re.search(
        r"\b([234])\s*x\s*\$?\d{1,3}\b",
        normalized,
    )
    if multi_match:
        return multi_match.group(1)

    pack_match = re.search(
        r"\b([234])\s*(?:pack|pk)\b",
        normalized,
    )
    if pack_match:
        return pack_match.group(1)

    return ""


def filename_upc_for_path(path: Path) -> str:
    sequences = re.findall(r"\d+", path.stem)

    if not sequences:
        return ""

    longest_sequence = max(sequences, key=len)
    return _digits_only(longest_sequence)


def resolve_tesseract_path() -> Path:
    configured_path = os.environ.get("TESSERACT_CMD", "").strip().strip("\"'")

    if configured_path:
        path = Path(configured_path).expanduser()

        if path.is_file():
            return path

        raise RuntimeError(
            "TESSERACT_CMD points to a missing file: "
            f"{path}"
        )

    path_from_environment = shutil.which("tesseract")

    if path_from_environment:
        return Path(path_from_environment)

    for path in DEFAULT_TESSERACT_PATHS:
        if path.is_file():
            return path

    expected_paths = ", ".join(str(path) for path in DEFAULT_TESSERACT_PATHS)

    raise RuntimeError(
        "Tesseract OCR was not found. Set TESSERACT_CMD to the full "
        f"tesseract.exe path. Checked: {expected_paths}"
    )


def configure_pytesseract() -> Any:
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "Unable to import pytesseract. "
            "Run: .\\.venv\\Scripts\\python.exe -m pip install "
            f"--upgrade pytesseract pillow. Import error: {exc}"
        ) from exc

    tesseract_path = resolve_tesseract_path()
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

    return pytesseract


def prepare_ocr_variants(image: Image.Image) -> list[Image.Image]:
    base = ImageOps.exif_transpose(image).convert("RGB")

    if min(base.size) < 700:
        scale = max(2, int(700 / max(1, min(base.size))))
        base = base.resize(
            (base.width * scale, base.height * scale),
            Image.Resampling.LANCZOS,
        )

    grayscale = base.convert("L")
    contrasted = ImageOps.autocontrast(grayscale)
    thresholded = contrasted.point(
        lambda pixel: 255 if pixel >= 160 else 0,
    )

    return [
        base,
        contrasted,
        thresholded,
    ]


def _ocr_image(image: Image.Image) -> str:
    pytesseract = configure_pytesseract()
    variants = prepare_ocr_variants(image)

    seen: set[str] = set()
    lines: list[str] = []

    for variant in variants:
        try:
            text = pytesseract.image_to_string(
                variant,
                config="--oem 3 --psm 6",
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract could not be started. Current executable: "
                f"{pytesseract.pytesseract.tesseract_cmd}"
            ) from exc
        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                f"Tesseract OCR failed: {exc}"
            ) from exc

        for line in text.splitlines():
            cleaned = line.strip()
            normalized_line = cleaned.lower()

            if cleaned and normalized_line not in seen:
                seen.add(normalized_line)
                lines.append(cleaned)

    return "\n".join(lines)


def load_existing_catalog(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            return {
                row.get("file_path", ""): row
                for row in csv.DictReader(handle)
                if row.get("file_path")
            }
    except (OSError, csv.Error):
        return {}


def cached_row_is_valid(
    cached: dict[str, str] | None,
    file_size: str,
    modified_time: str,
) -> bool:
    if not cached:
        return False

    return (
        cached.get("catalog_status") == "ok"
        and cached.get("file_size") == file_size
        and cached.get("modified_time") == modified_time
    )


def create_catalog_row(
    path: Path,
    file_size: str,
    modified_time: str,
) -> dict[str, str]:
    return {
        "file_path": str(path),
        "filename": path.name,
        "filename_upc": filename_upc_for_path(path),
        "file_size": file_size,
        "modified_time": modified_time,
        "width": "",
        "height": "",
        "detected_text": "",
        "normalized_text": "",
        "detected_brand": "",
        "detected_denomination": "",
        "detected_pack_quantity": "",
        "catalog_status": "ok",
        "catalog_error": "",
    }


def catalog_image(
    path: Path,
    cached: dict[str, str] | None = None,
) -> tuple[dict[str, str], bool]:
    stat = path.stat()
    file_size = str(stat.st_size)
    modified_time = str(stat.st_mtime)

    if cached_row_is_valid(cached, file_size, modified_time):
        row = {
            column: cached.get(column, "")
            for column in CATALOG_COLUMNS
        }
        return row, True

    row = create_catalog_row(
        path=path,
        file_size=file_size,
        modified_time=modified_time,
    )

    try:
        with Image.open(path) as image:
            transposed_image = ImageOps.exif_transpose(image)

            row["width"] = str(transposed_image.width)
            row["height"] = str(transposed_image.height)

            detected_text = _ocr_image(transposed_image)
            normalized = normalize_text(detected_text)

            row["detected_text"] = detected_text
            row["normalized_text"] = normalized
            row["detected_brand"] = detect_brand(normalized)
            row["detected_denomination"] = detect_denomination(
                normalized
            )
            row["detected_pack_quantity"] = detect_pack_quantity(
                normalized
            )
    except Exception as exc:
        row["catalog_status"] = "error"
        row["catalog_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return row, False


def iter_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )


def write_rows(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CATALOG_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    temporary_path.replace(output_path)


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return (
            f"{hours:02d}:{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    return f"{minutes:02d}:{remaining_seconds:02d}"


def print_progress(
    processed: int,
    total: int,
    ocr_count: int,
    cached_count: int,
    error_count: int,
    started_at: float,
) -> None:
    elapsed = format_elapsed(time.monotonic() - started_at)

    print(
        f"Processed {processed}/{total} | "
        f"OCR: {ocr_count} | "
        f"Cached: {cached_count} | "
        f"Errors: {error_count} | "
        f"Elapsed: {elapsed}",
        flush=True,
    )


def write_catalog(
    images_root: Path,
    output_path: Path,
    checkpoint_every: int,
) -> list[dict[str, str]]:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached_rows = load_existing_catalog(output_path)
    image_paths = iter_image_paths(images_root)

    rows: list[dict[str, str]] = []
    started_at = time.monotonic()

    ocr_count = 0
    cached_count = 0
    error_count = 0

    total = len(image_paths)

    print(
        f"Found {total} image(s) under {images_root}",
        flush=True,
    )
    print(
        f"Using Tesseract: {resolve_tesseract_path()}",
        flush=True,
    )

    for index, path in enumerate(image_paths, start=1):
        cached = cached_rows.get(str(path))

        try:
            row, used_cache = catalog_image(
                path,
                cached,
            )
        except Exception as exc:
            stat = path.stat()

            row = create_catalog_row(
                path=path,
                file_size=str(stat.st_size),
                modified_time=str(stat.st_mtime),
            )
            row["catalog_status"] = "error"
            row["catalog_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            used_cache = False

        rows.append(row)

        if used_cache:
            cached_count += 1
        else:
            ocr_count += 1

        if row["catalog_status"] == "error":
            error_count += 1
            print(
                f"ERROR: {path.name}: "
                f"{row['catalog_error']}",
                flush=True,
            )

        should_checkpoint = (
            index % checkpoint_every == 0
            or index == total
        )

        if should_checkpoint:
            write_rows(output_path, rows)
            print_progress(
                processed=index,
                total=total,
                ocr_count=ocr_count,
                cached_count=cached_count,
                error_count=error_count,
                started_at=started_at,
            )

    if not image_paths:
        write_rows(output_path, [])

    return rows


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Value must be an integer."
        ) from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "Value must be at least 1."
        )

    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Catalog Sam's local image library with "
            "cached OCR text."
        )
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Root image folder to scan recursively.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image_catalog.csv path.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=positive_integer,
        default=10,
        help=(
            "Save the CSV and print progress after this "
            "many images. Default: 10."
        ),
    )

    args = parser.parse_args()

    images_root = Path(
        str(args.images).strip().strip("\"'")
    ).expanduser()

    output_path = Path(
        str(args.output).strip().strip("\"'")
    ).expanduser()

    if not images_root.is_dir():
        print(
            "Image folder not found or inaccessible: "
            f"{images_root}",
            file=sys.stderr,
        )
        return 2

    try:
        rows = write_catalog(
            images_root=images_root,
            output_path=output_path,
            checkpoint_every=args.checkpoint_every,
        )
    except Exception as exc:
        print(
            f"Catalog failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    successful_count = sum(
        row["catalog_status"] == "ok"
        for row in rows
    )
    error_count = len(rows) - successful_count

    print(
        f"Cataloged {len(rows)} image(s) to {output_path}",
        flush=True,
    )
    print(
        f"Successful: {successful_count} | "
        f"Errors: {error_count}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())