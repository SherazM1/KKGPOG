from __future__ import annotations

import math
import re
from typing import Any, Optional

import pandas as pd


def collapse_spaces(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_header(value: Any) -> str:
    text = collapse_spaces(value).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return collapse_spaces(text).replace(" ", "")


def digits_only(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return re.sub(r"[^0-9]", "", text)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return collapse_spaces(text)


def int_value(value: Any, *, allow_zero: bool = True) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        number = int(value)
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        number = int(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        if not parsed.is_integer():
            return None
        number = int(parsed)
    if number < 0:
        return None
    if number == 0 and not allow_zero:
        return None
    return number


def normalized_name(value: Any) -> str:
    text = text_value(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return collapse_spaces(text)


def upc_a_check_digit(upc11: str) -> str:
    digits = digits_only(upc11)
    if len(digits) != 11:
        return ""
    total = sum(int(digit) * (3 if index % 2 == 0 else 1) for index, digit in enumerate(digits))
    return str((10 - (total % 10)) % 10)


def identifier_variants(value: Any) -> list[str]:
    digits = digits_only(value)
    if not digits:
        return []
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if not candidate:
            return
        candidates.append(candidate)
        stripped = candidate.lstrip("0")
        if stripped:
            candidates.append(stripped)
        if len(candidate) == 11:
            check = upc_a_check_digit(candidate)
            if check:
                candidates.append(candidate + check)
        if len(candidate) == 12:
            candidates.append(candidate[:11])

    add(digits)
    for length in (14, 13, 12, 11):
        if len(digits) >= length:
            add(digits[-length:])

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out
