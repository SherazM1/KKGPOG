from __future__ import annotations

import io
from typing import Any

import pandas as pd

from app.holiday_planograms.models import HolidayProduct
from app.holiday_planograms.normalization import canonical_header, digits_only, int_value, text_value


ASSORTMENT_SHEET_HINTS = ("assortmentlistd5holidays", "assortment", "holidays")

_ALIASES: dict[str, tuple[str, ...]] = {
    "product_name": ("product name",),
    "denomination": ("denom / load range", "denom", "load range"),
    "product_upc": ("product 12 digit upc", "product upc", "upc"),
    "pack_upc": ("pack upc",),
    "item_number": ("wm item number", "walmart item number", "item number"),
    "nre_facings": ("nre",),
    "qtr_facings": ("qtr pallet", "quarter pallet"),
    "total_pegs": ("total # of pegs", "total pegs"),
    "cpp": ("cpp",),
}


def _coerce_source(source_file: Any) -> bytes | str:
    if isinstance(source_file, (str, bytes)):
        return source_file
    if hasattr(source_file, "getvalue"):
        return bytes(source_file.getvalue())
    if hasattr(source_file, "read"):
        data = source_file.read()
        if hasattr(source_file, "seek"):
            source_file.seek(0)
        return bytes(data)
    raise TypeError("Unsupported assortment workbook source.")


def select_sheet(sheet_names: list[str], hints: tuple[str, ...] = ASSORTMENT_SHEET_HINTS) -> str:
    scored: list[tuple[int, str]] = []
    for name in sheet_names:
        compact = canonical_header(name)
        score = sum(1 for hint in hints if hint in compact)
        if score:
            scored.append((score, name))
    if not scored:
        raise ValueError("Could not find a Holiday assortment worksheet.")
    scored.sort(key=lambda item: (-item[0], item[1]))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        tied = ", ".join(name for score, name in scored if score == scored[0][0])
        raise ValueError(f"Multiple equally strong assortment worksheets found: {tied}.")
    return scored[0][1]


def _build_mapping(columns: list[str]) -> dict[str, str]:
    source = {canonical_header(column): column for column in columns}
    mapping: dict[str, str] = {}
    for key, aliases in _ALIASES.items():
        for alias in aliases:
            found = source.get(canonical_header(alias))
            if found:
                mapping[key] = found
                break
    return mapping


def load_holiday_assortment(source_file: Any) -> tuple[str, list[HolidayProduct], list[str]]:
    warnings: list[str] = []
    source = _coerce_source(source_file)
    workbook = pd.ExcelFile(io.BytesIO(source) if isinstance(source, bytes) else source)
    try:
        sheet_name = select_sheet([str(name) for name in workbook.sheet_names])
        df = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
    finally:
        workbook.close()
    mapping = _build_mapping([str(column) for column in df.columns])
    required = ("product_name", "product_upc", "item_number", "qtr_facings", "cpp")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError("Missing required assortment columns: " + ", ".join(missing) + ".")

    products: list[HolidayProduct] = []
    for index, row in enumerate(df.to_dict(orient="records"), start=2):
        if not any(text_value(value) for value in row.values()):
            continue
        qtr = int_value(row.get(mapping.get("qtr_facings", ""))) or 0
        nre = int_value(row.get(mapping.get("nre_facings", ""))) or 0
        total = int_value(row.get(mapping.get("total_pegs", "")))
        cpp = int_value(row.get(mapping.get("cpp", "")))
        name = text_value(row.get(mapping["product_name"]))
        if not name:
            warnings.append(f"Assortment row {index}: blank product name.")
        products.append(
            HolidayProduct(
                product_name=name,
                denomination=text_value(row.get(mapping.get("denomination", ""))),
                product_upc=digits_only(row.get(mapping["product_upc"])),
                pack_upc=digits_only(row.get(mapping.get("pack_upc", ""))),
                item_number=digits_only(row.get(mapping["item_number"])),
                nre_facings=nre,
                qtr_facings=qtr,
                total_pegs=total,
                cpp=cpp,
                source_row=index,
                raw=row,
            )
        )
    return sheet_name, products, warnings
