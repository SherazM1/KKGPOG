from __future__ import annotations

import io
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

NORMALIZED_KEYS = (
    "pog",
    "side",
    "row",
    "column",
    "item_number",
    "retail",
    "brand",
    "desc_1",
    "desc_2",
    "upc",
    "cpp",
    "file_path",
    "description",
)

REQUIRED_LOGICAL_FIELDS = ("pog", "item_number", "side", "row", "column")

TABULAR_ALIASES: dict[str, tuple[str, ...]] = {
    "pog": ("pog", "mod name", "pog name"),
    "item_number": ("item number", "item_number", "item"),
    "side": ("side",),
    "row": ("row",),
    "column": ("column", "col"),
    "retail": ("retail", "price", "value offer"),
    "brand": ("brand",),
    "desc_1": ("desc 1", "desc1", "sign desc 1"),
    "desc_2": ("desc 2", "desc2", "sign desc 2"),
    "upc": ("upc", "12 digit upc"),
    "cpp": ("cpp",),
    "file_path": ("file path", "filepath", "image path"),
    "description": ("description", "product desc"),
}

ACCESS_ALIASES: dict[str, tuple[str, ...]] = {
    "pog": ("pog", "planogram", "planogram_id", "pog_id", "mod name", "pog name"),
    "side": ("side", "side_no", "side_number", "page", "panel"),
    "row": ("row", "row_no", "row_number", "shelf", "shelf_row"),
    "column": ("column", "col", "column_no", "column_number", "slot", "position"),
    "item_number": ("item_number", "item", "item_no", "itemnum", "sku", "sku_number", "item number"),
    "retail": ("retail", "retail_price", "price", "unit_price", "value offer"),
    "brand": ("brand", "vendor", "manufacturer"),
    "desc_1": ("desc_1", "desc1", "description_1", "short_desc", "description1", "desc 1", "sign desc 1"),
    "desc_2": ("desc_2", "desc2", "description_2", "long_desc_2", "description2", "desc 2", "sign desc 2"),
    "upc": ("upc", "upc12", "upc_code", "barcode", "12 digit upc"),
    "cpp": ("cpp", "case_pack", "casepack", "pack_qty"),
    "file_path": ("file_path", "filepath", "image_path", "asset_path", "path", "file path", "image path"),
    "description": ("description", "item_description", "long_description", "desc_full", "product desc"),
}


@dataclass
class SamsSourceExtractionResult:
    """Extraction output plus source metadata for Sam's debug/validation."""

    source_type: str
    records: list[dict[str, Any]] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _canonical_header(value: str) -> str:
    cleaned = _collapse_spaces(str(value).strip().lower())
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = _collapse_spaces(cleaned)
    return cleaned.replace(" ", "")


def _detect_extension(source_file: Any) -> str:
    if isinstance(source_file, (str, os.PathLike)):
        return Path(str(source_file)).suffix.lower()
    filename = getattr(source_file, "name", "")
    return Path(str(filename)).suffix.lower()


def _coerce_uploaded_bytes(source_file: Any) -> tuple[bytes, str]:
    if isinstance(source_file, (bytes, bytearray)):
        return bytes(source_file), "uploaded.bin"

    if hasattr(source_file, "getvalue"):
        filename = getattr(source_file, "name", "uploaded.bin")
        return bytes(source_file.getvalue()), str(filename)

    if hasattr(source_file, "read"):
        filename = getattr(source_file, "name", "uploaded.bin")
        data = source_file.read()
        if hasattr(source_file, "seek"):
            try:
                source_file.seek(0)
            except Exception:
                pass
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Uploaded source stream did not return bytes.")
        return bytes(data), str(filename)

    raise TypeError("Unsupported source file type. Provide a file path, bytes, or uploaded file object.")


@contextmanager
def _resolve_access_path(source_file: Any) -> Iterable[str]:
    if isinstance(source_file, (str, os.PathLike)):
        path = str(source_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Access file not found: {path}")
        yield path
        return

    payload, source_name = _coerce_uploaded_bytes(source_file)
    suffix = Path(source_name).suffix.lower()
    if suffix not in (".accdb", ".mdb"):
        suffix = ".accdb"

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            temp_path = tmp.name
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _get_access_connection(path: str) -> Any:
    try:
        import pyodbc
    except Exception as exc:
        raise RuntimeError(
            "Direct Access extraction is unavailable in this environment. Upload an exported .xlsx or .csv fallback."
        ) from exc

    drivers = [driver for driver in pyodbc.drivers() if "access" in driver.lower()]
    if not drivers:
        raise RuntimeError(
            "Direct Access extraction is unavailable in this environment. Upload an exported .xlsx or .csv fallback."
        )

    errors: list[str] = []
    for driver in drivers:
        connection_string = f"DRIVER={{{driver}}};DBQ={path};"
        try:
            return pyodbc.connect(connection_string, autocommit=True)
        except Exception as exc:
            errors.append(f"{driver}: {exc}")

    raise RuntimeError("Unable to connect to Access database. " + " | ".join(errors))


def _build_mapping(column_names: list[str], aliases: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    canonical_source: dict[str, str] = {}
    for col in column_names:
        key = _canonical_header(col)
        if key and key not in canonical_source:
            canonical_source[key] = col

    mapping: dict[str, str] = {}
    for logical_key in NORMALIZED_KEYS:
        alias_list = aliases.get(logical_key, ())
        for alias in alias_list:
            alias_key = _canonical_header(alias)
            if alias_key in canonical_source:
                mapping[logical_key] = canonical_source[alias_key]
                break
    return mapping


def _normalize_from_mapping(row: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for logical_key in NORMALIZED_KEYS:
        source_col = mapping.get(logical_key)
        value = row.get(source_col) if source_col else None
        if pd.isna(value):
            value = None
        normalized[logical_key] = value
    return normalized


def _extract_from_access(source_file: Any) -> SamsSourceExtractionResult:
    result = SamsSourceExtractionResult(source_type="access")
    try:
        with _resolve_access_path(source_file) as access_path:
            connection = _get_access_connection(access_path)
            try:
                cursor = connection.cursor()
                table_names: list[str] = []
                for row in cursor.tables(tableType="TABLE"):
                    name = getattr(row, "table_name", None)
                    if name and not str(name).lower().startswith("msys"):
                        table_names.append(str(name))

                records: list[dict[str, Any]] = []
                for table_name in table_names:
                    try:
                        table_df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", connection)
                    except Exception:
                        continue
                    if table_df.empty:
                        continue
                    table_mapping = _build_mapping([str(c) for c in table_df.columns], ACCESS_ALIASES)
                    for logical_key, source_col in table_mapping.items():
                        result.column_mapping.setdefault(logical_key, source_col)
                    for row_dict in table_df.to_dict(orient="records"):
                        normalized = _normalize_from_mapping(row_dict, table_mapping)
                        if any(v is not None and str(v).strip() != "" for v in normalized.values()):
                            records.append(normalized)

                result.records = records
            finally:
                connection.close()
    except RuntimeError as exc:
        message = str(exc)
        result.warnings.append(message)
        if not message.startswith("Direct Access extraction is unavailable"):
            result.errors.append(message)
    except Exception as exc:
        result.errors.append(f"Access extraction failed: {exc}")
    return result


def _read_dataframe_from_source(source_file: Any, extension: str) -> pd.DataFrame:
    if isinstance(source_file, (str, os.PathLike)):
        path = str(source_file)
        if extension == ".xlsx":
            return pd.read_excel(path)
        return pd.read_csv(path)

    payload, _ = _coerce_uploaded_bytes(source_file)
    buffer = io.BytesIO(payload)
    if extension == ".xlsx":
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def _extract_from_tabular(source_file: Any, extension: str) -> SamsSourceExtractionResult:
    source_type = "xlsx" if extension == ".xlsx" else "csv"
    result = SamsSourceExtractionResult(source_type=source_type)
    try:
        df = _read_dataframe_from_source(source_file, extension)
    except Exception as exc:
        result.errors.append(f"Unable to read {source_type.upper()} source: {exc}")
        return result

    mapping = _build_mapping([str(c) for c in df.columns], TABULAR_ALIASES)
    result.column_mapping = mapping

    missing_required = [field for field in REQUIRED_LOGICAL_FIELDS if field not in mapping]
    if missing_required:
        result.errors.append(
            "Missing required columns for fallback source: "
            + ", ".join(missing_required)
            + ". Required fields: pog, item_number, side, row, column."
        )
        return result

    records: list[dict[str, Any]] = []
    for row_dict in df.to_dict(orient="records"):
        normalized = _normalize_from_mapping(row_dict, mapping)
        if any(v is not None and str(v).strip() != "" for v in normalized.values()):
            records.append(normalized)
    result.records = records
    return result


def extract_master_pog_source(source_file: Any) -> SamsSourceExtractionResult:
    """
    Extract normalized records from Sam's main source file.

    Supported types: .accdb, .mdb, .xlsx, .csv.
    """
    extension = _detect_extension(source_file)
    if extension in (".accdb", ".mdb"):
        return _extract_from_access(source_file)
    if extension in (".xlsx", ".csv"):
        return _extract_from_tabular(source_file, extension)
    return SamsSourceExtractionResult(
        source_type="unknown",
        errors=[
            f"Unsupported source extension '{extension or '(none)'}'. Use .accdb, .mdb, .xlsx, or .csv."
        ],
    )


def extract_master_pog_records(source_file: Any) -> list[dict[str, Any]]:
    """
    Backward-compatible record-only extractor.

    Returned records use:
    pog, side, row, column, item_number, retail, brand, desc_1, desc_2, upc, cpp, file_path, description.
    """
    return extract_master_pog_source(source_file).records
