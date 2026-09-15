from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from app.holiday_planograms.geometry_nre import SLOT_MERCHANDISE
from app.holiday_planograms.models import HolidayProduct, LayoutSlot


MATCH_EXACT = "exact"
MATCH_STRONG = "strong"
MATCH_AMBIGUOUS = "ambiguous"
MATCH_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class NreSlotMatch:
    slot: LayoutSlot
    raw_reference_text: str
    normalized_reference_text: str
    product: HolidayProduct | None
    status: str
    method: str
    confidence: str
    warnings: list[str]


_NRE_REFERENCE_TEXT_ROWS: tuple[tuple[str, ...], ...] = (
    (
        "ZIFT GIFT VGC",
        "ZIFT GAME VGC",
        "ZIFT FOOD FUN 20-500",
        "ZIFT ZILLIONS VGC",
        "NINTENDO ESHOP $10",
        "NINTENDO ESHOP $20",
        "NINTENDO ESHOP $50",
        "NINTENDO ESHOP $100",
    ),
    (
        "PLAYSTATION GWP $50",
        "PLAYSTATION GWP $50",
        "PLAYSTATION VGC",
        "PLAYSTATION VGC",
        "XBOX VGC ($10-$250)",
        "XBOX VGC ($10-$250)",
        "XBOX MULTIPACK $60",
        "XBOX GC $100",
    ),
    (
        "ROBLOX VGC",
        "ROBLOX VGC",
        "ROBLOX $25",
        "ROBLOX $25",
        "FORTNITE GWP $30",
        "FORTNITE VGC",
        "FORTNITE $15",
        "FORTNITE $50",
    ),
    (
        "META QUEST VGC",
        "META QUEST VGC",
        "META QUEST VGC",
        "META QUEST VGC",
        "RAZER GOLD VGC",
        "RAZER GOLD VGC",
        "RAZER GOLD VGC",
        "RAZER GOLD VGC",
    ),
    (
        "POKEMON GO 3300 COINS",
        "POKEMON GO 3300 COINS",
        "POKEMON GO 6700 COINS",
        "POKEMON GO 6700 COINS",
        "GOOGLE PLAY VGC",
        "GOOGLE PLAY $10",
        "GOOGLE PLAY $25",
        "GOOGLE PLAY $50",
    ),
)


def normalize_nre_reference_text(text: str) -> str:
    normalized = (text or "").upper().replace("$", " ")
    normalized = normalized.replace("POKECOINS", "COINS")
    normalized = normalized.replace("POKEMON GO 6700 COINS", "POKEMON GO 7000 COINS")
    normalized = normalized.replace("POKEMON", "POKEMON")
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text or ""))


def _family_for_text(text: str) -> str:
    normalized = normalize_nre_reference_text(text)
    if "ROBLOX" in normalized:
        return "ROBLOX"
    if "FORTNITE" in normalized or "EPIC" in normalized:
        return "FORTNITE"
    if "META" in normalized:
        return "META"
    if "RAZER" in normalized:
        return "RAZER"
    if "POKEMON" in normalized:
        return "POKEMON"
    if "GOOGLE" in normalized:
        return "GOOGLE"
    if "NINTENDO" in normalized:
        return "NINTENDO"
    if "PLAYSTATION" in normalized or "SONY" in normalized:
        return "PLAYSTATION"
    if "XBOX" in normalized or "MICROSOFT" in normalized:
        return "XBOX"
    if "ZIFT" in normalized or "ONLINE EXCHANGE" in normalized:
        return "ZIFT"
    return ""


def _candidate_family(product: HolidayProduct) -> str:
    return _family_for_text(product.product_name)


def _product_text(product: HolidayProduct) -> str:
    return normalize_nre_reference_text(f"{product.product_name} {product.denomination}")


def _slot_reference_texts(merchandise_slots: list[LayoutSlot]) -> dict[LayoutSlot, str]:
    ordered_slots = sorted(merchandise_slots, key=lambda slot: (slot.center_y, slot.center_x))
    if len(ordered_slots) != 40:
        assignments: dict[LayoutSlot, str] = {}
        for slot in ordered_slots:
            if 1 <= slot.row <= len(_NRE_REFERENCE_TEXT_ROWS) and 1 <= slot.column <= len(_NRE_REFERENCE_TEXT_ROWS[0]):
                assignments[slot] = _NRE_REFERENCE_TEXT_ROWS[slot.row - 1][slot.column - 1]
        return assignments
    assignments: dict[LayoutSlot, str] = {}
    for index, slot in enumerate(ordered_slots):
        row_index = index // 8
        column_index = index % 8
        assignments[slot] = _NRE_REFERENCE_TEXT_ROWS[row_index][column_index]
    return assignments


def _is_vgc(product_text: str) -> bool:
    return "VGC" in product_text


def _is_fixed_denomination(product_text: str) -> bool:
    return not _is_vgc(product_text) and bool(_numbers(product_text))


def _matches_subtype(slot_text: str, product_text: str) -> bool:
    if "GWP" in slot_text:
        return "GWP" in product_text
    if "MULTIPACK" in slot_text:
        return "MULTIPACK" in product_text
    if "ZIFT" in slot_text:
        if "GAME" in slot_text:
            return "GAME" in product_text
        if "FOOD FUN" in slot_text:
            return "FOOD FUN" in product_text
        if "ZILLIONS" in slot_text:
            return "ZILLIONS" in product_text
        if "GIFT" in slot_text:
            return "GREY" in product_text
    if "VGC" in slot_text:
        return _is_vgc(product_text) and "GWP" not in product_text and "MULTIPACK" not in product_text
    if _numbers(slot_text):
        return _is_fixed_denomination(product_text)
    return True


def _candidate_score(slot_text: str, product: HolidayProduct) -> tuple[int, str]:
    product_text = _product_text(product)
    if _family_for_text(slot_text) != _candidate_family(product):
        return 0, "family_mismatch"
    if not _matches_subtype(slot_text, product_text):
        return 0, "subtype_mismatch"

    slot_numbers = _numbers(slot_text)
    product_numbers = _numbers(product_text)
    if slot_numbers and not slot_numbers.issubset(product_numbers):
        return 0, "denomination_mismatch"

    score = 80
    if slot_numbers:
        score += 10
    if "VGC" in slot_text and "VGC" in product_text:
        score += 5
    if "GWP" in slot_text and "GWP" in product_text:
        score += 8
    if "MULTIPACK" in slot_text and "MULTIPACK" in product_text:
        score += 8
    return score, "reference_text"


def match_nre_merchandise_slots(
    slots: Iterable[LayoutSlot],
    products: list[HolidayProduct],
) -> list[NreSlotMatch]:
    merchandise_slots = [slot for slot in slots if getattr(slot, "slot_type", SLOT_MERCHANDISE) == SLOT_MERCHANDISE]
    reference_texts = _slot_reference_texts(merchandise_slots)
    assigned_counts: Counter[str] = Counter()
    matches: list[NreSlotMatch] = []

    for slot in sorted(merchandise_slots, key=lambda item: (item.center_y, item.center_x)):
        raw_text = reference_texts.get(slot, "")
        normalized_text = normalize_nre_reference_text(raw_text)
        warnings: list[str] = []
        if not raw_text:
            matches.append(
                NreSlotMatch(
                    slot=slot,
                    raw_reference_text="",
                    normalized_reference_text="",
                    product=None,
                    status="unresolved_reference_text",
                    method=MATCH_UNRESOLVED,
                    confidence=MATCH_UNRESOLVED,
                    warnings=["Expected 40 merchandise slots before deterministic reference-text assignment."],
                )
            )
            continue

        scored = []
        for product in products:
            if assigned_counts[product.item_number] >= product.nre_facings:
                continue
            score, reason = _candidate_score(normalized_text, product)
            if score > 0:
                scored.append((score, product.product_name, product, reason))
        scored.sort(key=lambda item: (-item[0], item[1]))

        if not scored:
            matches.append(
                NreSlotMatch(
                    slot=slot,
                    raw_reference_text=raw_text,
                    normalized_reference_text=normalized_text,
                    product=None,
                    status="unresolved_reference_text",
                    method=MATCH_UNRESOLVED,
                    confidence=MATCH_UNRESOLVED,
                    warnings=[f"No D5 NRE candidate matched '{raw_text}' without exceeding facing constraints."],
                )
            )
            continue
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            tied = "; ".join(item[2].product_name for item in scored if item[0] == scored[0][0])
            matches.append(
                NreSlotMatch(
                    slot=slot,
                    raw_reference_text=raw_text,
                    normalized_reference_text=normalized_text,
                    product=None,
                    status="ambiguous_reference_text",
                    method=MATCH_AMBIGUOUS,
                    confidence=MATCH_AMBIGUOUS,
                    warnings=[f"Multiple D5 NRE candidates matched '{raw_text}': {tied}."],
                )
            )
            continue

        product = scored[0][2]
        assigned_counts[product.item_number] += 1
        confidence = MATCH_EXACT if scored[0][0] >= 98 else MATCH_STRONG
        matches.append(
            NreSlotMatch(
                slot=slot,
                raw_reference_text=raw_text,
                normalized_reference_text=normalized_text,
                product=product,
                status="resolved_reference_text",
                method=scored[0][3],
                confidence=confidence,
                warnings=warnings,
            )
        )
    return matches
