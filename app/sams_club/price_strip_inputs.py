"""Shared structured document input; native production workbooks keep their parser."""
from __future__ import annotations

import io
import re
import zipfile
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


ALIASES = {
    "name": ("name", "product name", "description", "title"),
    "retail": ("price", "retail", "retail price"),
    "quantity": ("quantity", "qty", "count"),
    "type": ("type", "product type"),
    "item_number": ("item", "item #", "item number", "wm item", "sam's item"),
    "upc": ("upc", "upc #", "upc number"),
    "pog": ("pog",), "side": ("side",), "row": ("row",),
    "column": ("column", "col"), "brand": ("brand",),
    "desc_1": ("desc 1", "desc1"), "desc_2": ("desc 2", "desc2"),
    "length": ("length",), "data_on_bottom_left": ("data on bottom left", "bottom left"),
}
LOOKUP = {_key(alias): field for field, aliases in ALIASES.items() for alias in (*aliases, field)}


def normalize_records(records):
    result = []
    for number, record in enumerate(records, 1):
        normalized = {}
        for label, value in record.items():
            field = LOOKUP.get(_key(label))
            if field is None:
                continue
            value = "" if value is None or pd.isna(value) else str(value).strip()
            if field in normalized and normalized[field] != value:
                raise ValueError(f"Record {number}: conflicting values for {field}.")
            normalized[field] = value
        if any(normalized.values()):
            result.append(normalized)
    if not result:
        raise ValueError("No recognizable price strip records found. Use labeled fields or a table with headers.")
    return result


def parse_text(text):
    records, current = [], {}
    headers = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        table_line = line[1:-1] if line.startswith("|") and line.endswith("|") else line
        cells = [cell.strip() for cell in re.split(r"\s*\|\s*|\t+|\s{2,}", table_line)]
        if len(cells) > 1 and all(_key(c) in LOOKUP for c in cells):
            if current:
                records.append(current)
                current = {}
            headers = cells
            continue
        match = re.match(r"^([^:]+):\s*(.*)$", line)
        if match and _key(match[1]) in LOOKUP:
            headers = None
            field = LOOKUP[_key(match[1])]
            if field in current:
                records.append(current)
                current = {}
            current[field] = match[2]
        elif headers:
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                continue
            if len(cells) != len(headers):
                raise ValueError(f"Table row has {len(cells)} cells; expected {len(headers)}: {line}")
            records.append(dict(zip(headers, cells)))
        else:
            raise ValueError(f"Unrecognized input line: {line}. Use labeled fields or a table.")
    if current:
        records.append(current)
    return normalize_records(records)


def source_suffix(source):
    return Path(source if isinstance(source, (str, Path)) else getattr(source, "name", "source.xlsx")).suffix.lower()


def parse_source(source):
    suffix = source_suffix(source)
    if isinstance(source, (str, Path)):
        payload = Path(source).read_bytes()
    elif isinstance(source, bytes):
        payload = source
    elif hasattr(source, "getvalue"):
        payload = source.getvalue()
    else:
        payload = source.read()
        source.seek(0)
    if suffix == ".xlsx":
        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            candidates = []
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet, dtype=str).fillna("")
                fields = {LOOKUP.get(_key(c)) for c in frame.columns}
                if {"name", "retail", "item_number", "upc"} <= fields:
                    candidates.append((sheet, frame))
            if len(candidates) != 1:
                raise ValueError("Price Pocket requires exactly one worksheet with Name, Price, Item and UPC headers.")
            return normalize_records(candidates[0][1].to_dict("records"))
    if suffix == ".docx":
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        lines = []
        def content(node):
            return "".join("\n" if n.tag.endswith("}br") else "\t" if n.tag.endswith("}tab") else n.text or "" for n in node.iter() if n.tag.rsplit("}", 1)[-1] in ("t", "br", "tab"))
        for node in root.find("w:body", ns):
            if node.tag.endswith("}p"):
                lines.append(content(node))
            elif node.tag.endswith("}tbl"):
                for row in node.findall("w:tr", ns):
                    lines.append(" | ".join(content(cell) for cell in row.findall("w:tc", ns)))
        return parse_text("\n".join(lines))
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(payload)) as document:
            parts = []
            for page in document.pages:
                tables = page.find_tables()
                # Remove table regions from text so records are never duplicated.
                text_page = page.filter(lambda obj: not any(t.bbox[0] <= obj.get("x0", -1) < t.bbox[2] and t.bbox[1] <= obj.get("top", -1) < t.bbox[3] for t in tables))
                parts.append(text_page.extract_text(layout=False) or "")
                for table in tables:
                    parts.extend(" | ".join(cell or "" for cell in row) for row in table.extract())
        if not any(p.strip() for p in parts):
            raise ValueError("PDF has no extractable text. Provide a text PDF or DOCX/XLSX; scanned PDFs require OCR.")
        return parse_text("\n".join(parts))
    raise ValueError("Supported price strip inputs: .xlsx, .docx, .pdf.")


def validate_price(value, field="price"):
    if not re.fullmatch(r"\$?\s*(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?", str(value)):
        raise ValueError(f"Invalid {field}: {value!r}; use a nonnegative amount with at most two decimal places.")
    return Decimal(str(value).replace("$", "").replace(",", "").strip())


def legacy_document_frame(source, template_name=None):
    from app.sams_club.extract_price_strips import _EXPECTED_PRODUCTION_FIELDS
    from app.sams_club.holiday_price_strips import is_sams_holiday_template
    from app.sams_club.render_price_strips import parse_strip_length
    records = parse_source(source)
    errors = []
    positions = set()
    lengths = {}
    for i, record in enumerate(records, 1):
        # Explicit equivalent document fields may supply the existing description lines.
        if "name" in record and "desc_1" not in record:
            record["desc_1"] = record["name"]
        if "type" in record and "desc_2" not in record:
            record["desc_2"] = record["type"]
        missing = [f for f in _EXPECTED_PRODUCTION_FIELDS if f not in record or (f in ("pog", "side", "row", "column", "item_number", "retail", "brand", "desc_1") and not record[f])]
        if missing:
            errors.append(f"Record {i}: missing required fields: {', '.join(missing)}")
        for field in ("side", "row", "column"):
            if field in record and not re.fullmatch(r"[1-9]\d*", record[field]):
                errors.append(f"Record {i}: {field} must be a positive integer.")
        position = tuple(record.get(f) for f in ("pog", "side", "row", "column"))
        if position in positions:
            errors.append(f"Record {i}: duplicate POG/Side/Row/Column position.")
        positions.add(position)
        if not is_sams_holiday_template(template_name):
            length = parse_strip_length(record.get("length", ""))
            if length is None:
                errors.append(f"Record {i}: missing or invalid length; provide width x height in inches.")
            group = position[:3]
            if group in lengths and lengths[group] != length:
                errors.append(f"Record {i}: conflicting length for the same strip row.")
            lengths[group] = length
        if record.get("retail"):
            try:
                validate_price(record["retail"], "retail")
            except ValueError as exc:
                errors.append(f"Record {i}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    return pd.DataFrame(records)
