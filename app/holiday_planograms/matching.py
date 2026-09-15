from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from app.holiday_planograms.models import BosPlanogramProduct, HolidayProduct, ProductMatch
from app.holiday_planograms.normalization import identifier_variants, normalized_name


def match_bos_to_d5(
    bos_products: Iterable[BosPlanogramProduct],
    d5_qp_products: Iterable[HolidayProduct],
) -> list[ProductMatch]:
    d5_items: dict[str, HolidayProduct] = {}
    d5_upcs: dict[str, HolidayProduct] = {}
    d5_products = list(d5_qp_products)
    for product in d5_products:
        if product.item_number:
            d5_items.setdefault(product.item_number, product)
        for key in identifier_variants(product.product_upc):
            d5_upcs.setdefault(key, product)
        for key in identifier_variants(product.pack_upc):
            d5_upcs.setdefault(key, product)

    matches: list[ProductMatch] = []
    for bos in bos_products:
        warnings: list[str] = []
        d5 = d5_items.get(bos.item_number) if bos.item_number else None
        method = "item_number" if d5 else ""
        if d5 is None:
            for key in identifier_variants(bos.upc):
                if key in d5_upcs:
                    d5 = d5_upcs[key]
                    method = "upc"
                    break
        if d5 is None:
            bos_name = normalized_name(bos.name)
            ranked = sorted(
                ((SequenceMatcher(None, bos_name, normalized_name(product.product_name)).ratio(), product) for product in d5_products),
                key=lambda item: item[0],
                reverse=True,
            )
            if ranked and ranked[0][0] >= 0.88:
                d5 = ranked[0][1]
                method = "name_fallback"
                warnings.append("Matched by product-name fallback; verify manually.")
        if d5 is None:
            matches.append(ProductMatch(bos=bos, d5=None, status="unmatched", method="unresolved", warnings=[]))
            continue
        if bos.facings != d5.qtr_facings:
            warnings.append(f"Facing mismatch: BOS={bos.facings}, D5={d5.qtr_facings}.")
        if bos.cpp is not None and d5.cpp is not None and bos.cpp != d5.cpp:
            warnings.append(f"CPP mismatch: BOS={bos.cpp}, D5={d5.cpp}.")
        if bos.item_number and d5.item_number and bos.item_number != d5.item_number:
            warnings.append(f"Item mismatch: BOS={bos.item_number}, D5={d5.item_number}.")
        matches.append(ProductMatch(bos=bos, d5=d5, status="matched", method=method, warnings=warnings))
    return matches


def occurrence_counts_by_item(placements: Iterable[object]) -> Counter[str]:
    return Counter(str(getattr(placement, "item_number", "") or "") for placement in placements if getattr(placement, "item_number", ""))
