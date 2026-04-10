from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

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

ALIASES: dict[str, tuple[str, ...]] = {
    "pog": ("pog", "planogram", "planogram_id", "pog_id"),
    "side": ("side", "side_no", "side_number", "page", "panel"),
    "row": ("row", "row_no", "row_number", "shelf", "shelf_row"),
    "column": ("column", "col", "column_no", "column_number", "slot", "position"),
    "item_number": ("item_number", "item", "item_no", "itemnum", "sku", "sku_number"),
    "retail": ("retail", "retail_price", "price", "unit_price"),
    "brand": ("brand", "vendor", "manufacturer"),
    "desc_1": ("desc_1", "desc1", "description_1", "short_desc", "description1"),
    "desc_2": ("desc_2", "desc2", "description_2", "long_desc_2", "description2"),
    "upc": ("upc", "upc12", "upc_code", "barcode"),
    "cpp": ("cpp", "case_pack", "casepack", "pack_qty"),
    "file_path": ("file_path", "filepath", "image_path", "asset_path", "path"),
    "description": ("description", "item_description", "long_description", "desc_full"),
}


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_row_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical_source: dict[str, Any] = {}
    for key, value in row.items():
        canonical_source[_canonical_key(str(key))] = value

    normalized: dict[str, Any] = {key: None for key in NORMALIZED_KEYS}
    for target_key, aliases in ALIASES.items():
        for alias in aliases:
            match_key = _canonical_key(alias)
            if match_key in canonical_source:
                normalized[target_key] = canonical_source[match_key]
                break
    return normalized


def _coerce_uploaded_bytes(access_file: Any) -> tuple[bytes, str]:
    if isinstance(access_file, (bytes, bytearray)):
        return bytes(access_file), "uploaded.accdb"

    if hasattr(access_file, "getvalue"):
        filename = getattr(access_file, "name", "uploaded.accdb")
        return bytes(access_file.getvalue()), str(filename)

    if hasattr(access_file, "read"):
        filename = getattr(access_file, "name", "uploaded.accdb")
        data = access_file.read()
        if hasattr(access_file, "seek"):
            try:
                access_file.seek(0)
            except Exception:
                pass
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Uploaded Access file stream did not return bytes.")
        return bytes(data), str(filename)

    raise TypeError("Unsupported Access file type. Provide a file path, bytes, or uploaded file object.")


@contextmanager
def _resolve_access_path(access_file: Any) -> Iterable[str]:
    if isinstance(access_file, (str, os.PathLike)):
        path = str(access_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Access file not found: {path}")
        yield path
        return

    payload, source_name = _coerce_uploaded_bytes(access_file)
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
            "Access extraction requires optional dependency 'pyodbc' and an installed Microsoft Access ODBC driver."
        ) from exc

    drivers = [driver for driver in pyodbc.drivers() if "access" in driver.lower()]
    if not drivers:
        raise RuntimeError("No Microsoft Access ODBC driver found for pyodbc.")

    errors: list[str] = []
    for driver in drivers:
        connection_string = f"DRIVER={{{driver}}};DBQ={path};"
        try:
            return pyodbc.connect(connection_string, autocommit=True)
        except Exception as exc:
            errors.append(f"{driver}: {exc}")

    raise RuntimeError("Unable to connect to Access database. " + " | ".join(errors))


def _read_table_names(connection: Any) -> list[str]:
    cursor = connection.cursor()
    table_names: list[str] = []
    for row in cursor.tables(tableType="TABLE"):
        name = getattr(row, "table_name", None)
        if not name:
            continue
        if str(name).lower().startswith("msys"):
            continue
        table_names.append(str(name))
    return table_names


def _fetch_rows_from_table(connection: Any, table_name: str) -> list[dict[str, Any]]:
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM [{table_name}]")
    column_names = [str(col[0]) for col in cursor.description] if cursor.description else []
    if not column_names:
        return []

    rows: list[dict[str, Any]] = []
    for db_row in cursor.fetchall():
        raw = {column_names[idx]: db_row[idx] for idx in range(len(column_names))}
        normalized = _normalize_row_dict(raw)
        if any(value is not None and str(value).strip() != "" for value in normalized.values()):
            rows.append(normalized)
    return rows


def extract_master_pog_records(access_file: Any) -> list[dict[str, Any]]:
    """
    Extract normalized raw records from an Access database.

    Returned records use the intermediate shape:
    pog, side, row, column, item_number, retail, brand, desc_1, desc_2, upc, cpp, file_path, description.
    """
    with _resolve_access_path(access_file) as access_path:
        connection = _get_access_connection(access_path)
        try:
            records: list[dict[str, Any]] = []
            table_names = _read_table_names(connection)
            for table_name in table_names:
                try:
                    records.extend(_fetch_rows_from_table(connection, table_name))
                except Exception:
                    continue
            return records
        finally:
            connection.close()
