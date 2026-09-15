from __future__ import annotations

from collections import Counter
from typing import Any

from app.holiday_planograms.extract_assortment import load_holiday_assortment
from app.holiday_planograms.extract_bos import load_bos_qp
from app.holiday_planograms.extract_nre import NRE_CONFIGURATIONS, load_bos_nre_reference
from app.holiday_planograms.geometry import detect_geometry
from app.holiday_planograms.geometry_nre import detect_nre_geometry
from app.holiday_planograms.image_resolution import build_holiday_image_index, resolve_holiday_image
from app.holiday_planograms.matching import match_bos_to_d5
from app.holiday_planograms.matching_nre import match_nre_merchandise_slots, normalize_nre_reference_text
from app.holiday_planograms.models import (
    HolidayNreConfigurationSummary,
    HolidayNreResult,
    HolidayPlacement,
    HolidayProduct,
    HolidayQpResult,
    ProductMatch,
)


def _expand_bos_products(matches: list[ProductMatch]) -> list[ProductMatch]:
    expanded: list[ProductMatch] = []
    for match in matches:
        for _ in range(max(0, int(match.bos.facings))):
            expanded.append(match)
    return expanded


def _build_placements(matches: list[ProductMatch], slots: list[Any], image_index: Any) -> list[HolidayPlacement]:
    placements: list[HolidayPlacement] = []
    expanded = _expand_bos_products(matches)
    for index, slot in enumerate(slots):
        warnings: list[str] = []
        if index >= len(expanded):
            placements.append(
                HolidayPlacement(
                    panel_index=slot.panel_index,
                    panel_name=slot.panel_name,
                    row=slot.row,
                    column=slot.column,
                    bbox=slot.bbox,
                    reference_text="",
                    product_upc="",
                    pack_upc="",
                    item_number="",
                    product_name="",
                    denomination="",
                    cpp=None,
                    capacity=None,
                    image_path="",
                    placement_status="unassigned",
                    image_status="missing",
                    image_resolution_source="unresolved",
                    warnings=["No BOS facing record available for this detected slot."],
                )
            )
            continue
        match = expanded[index]
        d5 = match.d5
        if match.method != "item_number":
            warnings.append(f"Product match method: {match.method}.")
        warnings.extend(match.warnings)
        if not d5:
            warnings.append("No matching D5 enrichment record.")
        product_upc = d5.product_upc if d5 else match.bos.upc
        pack_upc = d5.pack_upc if d5 else ""
        item_number = d5.item_number if d5 else match.bos.item_number
        image = resolve_holiday_image(product_upc, pack_upc, item_number, image_index)
        placements.append(
            HolidayPlacement(
                panel_index=slot.panel_index,
                panel_name=slot.panel_name,
                row=slot.row,
                column=slot.column,
                bbox=slot.bbox,
                reference_text="",
                product_upc=product_upc,
                pack_upc=pack_upc,
                item_number=item_number,
                product_name=d5.product_name if d5 else match.bos.name,
                denomination=d5.denomination if d5 else "",
                cpp=d5.cpp if d5 and d5.cpp is not None else match.bos.cpp,
                capacity=match.bos.capacity,
                image_path=image.image_path,
                placement_status="resolved" if match.status == "matched" else "unresolved_product",
                image_status=image.status,
                image_resolution_source=image.source,
                warnings=warnings,
            )
        )
    return placements


def _placement_rows(placements: list[HolidayPlacement]) -> list[dict[str, Any]]:
    return [
        {
            "Panel": placement.panel_name,
            "Row": placement.row,
            "Column": placement.column,
            "Reference Text": placement.reference_text,
            "Product Name": placement.product_name,
            "Denom / Load Range": placement.denomination,
            "UPC": placement.product_upc,
            "Pack UPC": placement.pack_upc,
            "WM Item Number": placement.item_number,
            "CPP": placement.cpp,
            "Capacity": placement.capacity,
            "Placement Status": placement.placement_status,
            "Image Status": placement.image_status,
            "Image Resolution Source": placement.image_resolution_source,
            "Image Path": placement.image_path,
            "Warnings": "; ".join(placement.warnings),
        }
        for placement in placements
    ]


def _product_rows(matches: list[ProductMatch], placements: list[HolidayPlacement]) -> list[dict[str, Any]]:
    count_by_item = Counter(placement.item_number for placement in placements if placement.item_number)
    image_by_item: dict[str, str] = {}
    for placement in placements:
        if placement.item_number:
            image_by_item.setdefault(placement.item_number, placement.image_status)
            if placement.image_status == "resolved":
                image_by_item[placement.item_number] = "resolved"
    rows: list[dict[str, Any]] = []
    for match in matches:
        d5 = match.d5
        item = d5.item_number if d5 else match.bos.item_number
        warnings = list(match.warnings)
        detected = count_by_item.get(item, 0)
        if detected != match.bos.facings:
            warnings.append(f"Detected slot count {detected} does not equal BOS facings {match.bos.facings}.")
        rows.append(
            {
                "Product": d5.product_name if d5 else match.bos.name,
                "BOS UPC": match.bos.upc,
                "D5 UPC": d5.product_upc if d5 else "",
                "BOS ID": match.bos.item_number,
                "D5 WM Item Number": d5.item_number if d5 else "",
                "BOS Facings": match.bos.facings,
                "D5 QTR Facings": d5.qtr_facings if d5 else None,
                "BOS CPP": match.bos.cpp,
                "D5 CPP": d5.cpp if d5 else None,
                "Detected Slot Count": detected,
                "Product Match Status": match.status,
                "Match Method": match.method,
                "Image Match Status": image_by_item.get(item, "missing"),
                "Warnings": "; ".join(warnings),
            }
        )
    return rows


def _nre_placement_rows(placements: list[HolidayPlacement], configuration: str) -> list[dict[str, Any]]:
    return [
        {
            "Configuration": configuration,
            "Panel": placement.panel_name,
            "Row": placement.row,
            "Column": placement.column,
            "Slot Type": "filler" if placement.placement_status == "non_product" else "merchandise",
            "Raw Reference Text": "" if placement.placement_status == "non_product" else placement.reference_text,
            "Normalized Reference Text": placement.normalized_reference_text,
            "Product Name": placement.product_name,
            "Product UPC": placement.product_upc,
            "WM Item Number": placement.item_number,
            "CPP": placement.cpp,
            "Expected Product Facings": placement.capacity,
            "Placement Status": placement.placement_status,
            "Match Method": placement.match_method,
            "Match Confidence": placement.match_confidence,
            "Image Status": placement.image_status,
            "Image Resolution Source": placement.image_resolution_source,
            "Image Path": placement.image_path,
            "Warnings": "; ".join(placement.warnings),
        }
        for placement in placements
    ]


def _nre_product_rows(products: list[HolidayProduct], placements: list[HolidayPlacement]) -> list[dict[str, Any]]:
    count_by_item = Counter(placement.item_number for placement in placements if placement.item_number)
    image_by_item: dict[str, str] = {}
    for placement in placements:
        if placement.item_number:
            image_by_item.setdefault(placement.item_number, placement.image_status)
            if placement.image_status == "resolved":
                image_by_item[placement.item_number] = "resolved"

    rows: list[dict[str, Any]] = []
    for product in products:
        resolved = count_by_item.get(product.item_number, 0)
        difference = resolved - product.nre_facings
        rows.append(
            {
                "Product Name": product.product_name,
                "Product UPC": product.product_upc,
                "WM Item Number": product.item_number,
                "D5 NRE Facings": product.nre_facings,
                "Resolved Occurrences": resolved,
                "Difference": difference,
                "CPP": product.cpp,
                "Match Status": "matched" if difference == 0 else "mismatch",
                "Image Match Status": image_by_item.get(product.item_number, "not_applicable"),
                "Warnings": "" if difference == 0 else f"Resolved occurrence difference: {difference}.",
            }
        )
    return rows


def _build_nre_placements(
    slots: list[Any],
    products: list[HolidayProduct],
    image_index: Any,
    configuration: str,
) -> list[HolidayPlacement]:
    placements: list[HolidayPlacement] = []
    matches_by_slot = {match.slot: match for match in match_nre_merchandise_slots(slots, products)}
    for slot in slots:
        slot_type = getattr(slot, "slot_type", "merchandise")
        if slot_type == "filler":
            placements.append(
                HolidayPlacement(
                    panel_index=slot.panel_index,
                    panel_name=slot.panel_name,
                    row=slot.row,
                    column=slot.column,
                    bbox=slot.bbox,
                    reference_text="filler",
                    product_upc="",
                    pack_upc="",
                    item_number="",
                    product_name="",
                    denomination="",
                    cpp=None,
                    capacity=None,
                    image_path="",
                    placement_status="non_product",
                    image_status="not_applicable",
                    image_resolution_source="unresolved",
                    warnings=[f"Holiday NRE {configuration}: filler/non-product fixture cell."],
                    match_method="non_product",
                    match_confidence="non_product",
                )
            )
            continue
        elif slot_type == "ambiguous":
            placement_status = "ambiguous_slot"
            image_status = "missing"
            warning = f"Holiday NRE {configuration}: slot classification is ambiguous."
            reference_text = slot_type
            normalized_reference_text = ""
            match_method = "ambiguous_slot"
            match_confidence = "ambiguous"
        else:
            match = matches_by_slot.get(slot)
            if match and match.product:
                product = match.product
                image = resolve_holiday_image(product.product_upc, product.pack_upc, product.item_number, image_index)
                placements.append(
                    HolidayPlacement(
                        panel_index=slot.panel_index,
                        panel_name=slot.panel_name,
                        row=slot.row,
                        column=slot.column,
                        bbox=slot.bbox,
                        reference_text=match.raw_reference_text,
                        product_upc=product.product_upc,
                        pack_upc=product.pack_upc,
                        item_number=product.item_number,
                        product_name=product.product_name,
                        denomination=product.denomination,
                        cpp=product.cpp,
                        capacity=product.nre_facings,
                        image_path=image.image_path,
                        placement_status=match.status,
                        image_status=image.status,
                        image_resolution_source=image.source,
                        warnings=match.warnings,
                        normalized_reference_text=match.normalized_reference_text,
                        match_method=match.method,
                        match_confidence=match.confidence,
                    )
                )
                continue
            placement_status = match.status if match else "unresolved_reference_text"
            image_status = "missing"
            warning = "; ".join(match.warnings) if match else f"Holiday NRE {configuration}: no reference-text match."
            reference_text = match.raw_reference_text if match else ""
            normalized_reference_text = match.normalized_reference_text if match else ""
            match_method = match.method if match else "unresolved"
            match_confidence = match.confidence if match else "unresolved"
        placements.append(
            HolidayPlacement(
                panel_index=slot.panel_index,
                panel_name=slot.panel_name,
                row=slot.row,
                column=slot.column,
                bbox=slot.bbox,
                reference_text=reference_text if slot_type == "merchandise" else slot_type,
                product_upc="",
                pack_upc="",
                item_number="",
                product_name="",
                denomination="",
                cpp=None,
                capacity=None,
                image_path="",
                placement_status=placement_status,
                image_status=image_status,
                image_resolution_source="unresolved",
                warnings=[warning] if warning else [],
                normalized_reference_text=normalized_reference_text,
                match_method=match_method,
                match_confidence=match_confidence,
            )
        )
    return placements


def build_holiday_nre_qa(
    bos_workbook: Any,
    assortment_workbook: Any,
    configuration: str,
    local_image_folder: str | None = None,
) -> HolidayNreResult:
    warnings: list[str] = []
    errors: list[str] = []
    bos_sheet = ""
    d5_sheet = ""
    reference = None
    d5_products: list[HolidayProduct] = []
    d5_nre_products: list[HolidayProduct] = []
    geometry = None
    placements: list[HolidayPlacement] = []

    try:
        bos_sheet, reference, reference_warnings = load_bos_nre_reference(bos_workbook, configuration)
        warnings.extend(reference_warnings)
    except Exception as exc:
        errors.append(str(exc))

    try:
        d5_sheet, d5_products, d5_warnings = load_holiday_assortment(assortment_workbook)
        warnings.extend(d5_warnings)
        d5_nre_products = [product for product in d5_products if product.nre_facings > 0]
    except Exception as exc:
        errors.append(str(exc))

    image_index = build_holiday_image_index(local_image_folder)
    warnings.extend(image_index.warnings)

    product_image_status: dict[str, str] = {}
    product_image_source: dict[str, str] = {}
    product_image_path: dict[str, str] = {}
    for product in d5_nre_products:
        image = resolve_holiday_image(product.product_upc, product.pack_upc, product.item_number, image_index)
        product_image_status[product.item_number] = image.status
        product_image_source[product.item_number] = image.source
        product_image_path[product.item_number] = image.image_path

    if reference is not None:
        geometry = detect_nre_geometry(reference.image)
        warnings.extend(geometry.warnings)
        placements = _build_nre_placements(geometry.slots, d5_nre_products, image_index, reference.configuration)

    product_rows = _nre_product_rows(d5_nre_products, placements)
    for row in product_rows:
        item = str(row.get("WM Item Number", ""))
        row["Image Match Status"] = product_image_status.get(item, row["Image Match Status"])
        if product_image_path.get(item):
            row["Image Path"] = product_image_path[item]
        row["Image Resolution Source"] = product_image_source.get(item, "unresolved")

    detected_slots = len(geometry.slots) if geometry else 0
    grid_positions = sum(panel.rows * panel.columns for panel in geometry.panels) if geometry else 0
    merchandise_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "merchandise") if geometry else 0
    filler_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "filler") if geometry else 0
    ambiguous_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "ambiguous") if geometry else 0
    expected_facings = sum(product.nre_facings for product in d5_nre_products)
    image_breakdown = Counter(placement.image_resolution_source for placement in placements if placement.placement_status == "resolved_reference_text")
    resolved_counts = Counter(placement.item_number for placement in placements if placement.placement_status == "resolved_reference_text" and placement.item_number)
    facing_mismatches = sum(1 for product in d5_nre_products if resolved_counts.get(product.item_number, 0) != product.nre_facings)
    summary = {
        "configuration": configuration,
        "bos_sheet": bos_sheet,
        "d5_sheet": d5_sheet,
        "reference_found": reference is not None,
        "reference_image_name": reference.image.name if reference else "",
        "reference_image_width": reference.image.width if reference else 0,
        "reference_image_height": reference.image.height if reference else 0,
        "panel_count": len(geometry.panels) if geometry else 0,
        "rows_per_panel": [panel.rows for panel in geometry.panels] if geometry else [],
        "columns_per_panel": [panel.columns for panel in geometry.panels] if geometry else [],
        "active_slots": merchandise_slots,
        "physical_cells_detected": detected_slots,
        "theoretical_grid_positions": grid_positions,
        "merchandise_slots": merchandise_slots,
        "filler_slots": filler_slots,
        "ambiguous_slots": ambiguous_slots,
        "d5_total_products": len(d5_products),
        "d5_nre_products": len(d5_nre_products),
        "d5_nre_facings": expected_facings,
        "detected_expected_difference": merchandise_slots - expected_facings,
        "merchandise_difference": merchandise_slots - expected_facings,
        "matched_products": len([product for product in d5_nre_products if resolved_counts.get(product.item_number, 0) > 0]),
        "products_with_exact_occurrence_counts": len([product for product in d5_nre_products if resolved_counts.get(product.item_number, 0) == product.nre_facings]),
        "facing_mismatches": facing_mismatches,
        "resolved_placements": sum(1 for placement in placements if placement.placement_status == "resolved_reference_text"),
        "unresolved_placements": sum(1 for placement in placements if placement.placement_status == "unresolved_reference_text"),
        "ambiguous_reference_text_placements": sum(1 for placement in placements if placement.placement_status == "ambiguous_reference_text"),
        "non_product_placements": sum(1 for placement in placements if placement.placement_status == "non_product"),
        "ambiguous_placements": sum(1 for placement in placements if placement.placement_status == "ambiguous_slot"),
        "image_folder": image_index.root_dir,
        "image_folder_exists": image_index.folder_exists,
        "image_folder_is_dir": image_index.folder_is_dir,
        "image_files_indexed": image_index.indexed_files,
        "unique_products_with_images": len({placement.item_number for placement in placements if placement.image_status == "resolved" and placement.item_number}),
        "placements_with_images": sum(1 for placement in placements if placement.image_status == "resolved"),
        "image_source_breakdown": dict(image_breakdown),
    }
    return HolidayNreResult(
        configuration=configuration,
        bos_sheet=bos_sheet,
        d5_sheet=d5_sheet,
        reference=reference,
        d5_products=d5_products,
        d5_nre_products=d5_nre_products,
        geometry=geometry,
        placements=placements,
        product_qa_rows=product_rows,
        placement_qa_rows=_nre_placement_rows(placements, configuration),
        summary=summary,
        warnings=warnings,
        errors=errors,
    )


def analyze_holiday_nre_configurations(
    bos_workbook: Any,
    assortment_workbook: Any,
) -> list[HolidayNreConfigurationSummary]:
    try:
        _d5_sheet, d5_products, _warnings = load_holiday_assortment(assortment_workbook)
        expected_facings = sum(product.nre_facings for product in d5_products if product.nre_facings > 0)
    except Exception:
        expected_facings = 0

    rows: list[HolidayNreConfigurationSummary] = []
    for configuration in NRE_CONFIGURATIONS:
        warnings: list[str] = []
        errors: list[str] = []
        sheet_name = ""
        reference_image_name = ""
        reference_width = 0
        reference_height = 0
        panel_count = 0
        row_count = 0
        column_count = 0
        active_slots = 0
        grid_positions = 0
        merchandise_slots = 0
        filler_slots = 0
        ambiguous_slots = 0
        try:
            sheet_name, reference, reference_warnings = load_bos_nre_reference(bos_workbook, configuration)
            warnings.extend(reference_warnings)
            if reference is not None:
                reference_image_name = reference.image.name
                reference_width = reference.image.width
                reference_height = reference.image.height
                geometry = detect_nre_geometry(reference.image)
                warnings.extend(geometry.warnings)
                panel_count = len(geometry.panels)
                row_count = max((panel.rows for panel in geometry.panels), default=0)
                column_count = max((panel.columns for panel in geometry.panels), default=0)
                active_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "merchandise")
                grid_positions = sum(panel.rows * panel.columns for panel in geometry.panels)
                merchandise_slots = active_slots
                filler_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "filler")
                ambiguous_slots = sum(1 for slot in geometry.slots if getattr(slot, "slot_type", "") == "ambiguous")
        except Exception as exc:
            errors.append(str(exc))
        rows.append(
            HolidayNreConfigurationSummary(
                configuration=configuration,
                sheet_name=sheet_name,
                reference_image_name=reference_image_name,
                reference_width=reference_width,
                reference_height=reference_height,
                panel_count=panel_count,
                row_count=row_count,
                column_count=column_count,
                active_slots=active_slots,
                grid_positions=grid_positions,
                physical_detected=merchandise_slots + filler_slots + ambiguous_slots,
                merchandise_slots=merchandise_slots,
                filler_slots=filler_slots,
                ambiguous_slots=ambiguous_slots,
                expected_nre_facings=expected_facings,
                difference=merchandise_slots - expected_facings,
                warnings=warnings,
                errors=errors,
            )
        )
    return rows


def build_holiday_qp_qa(
    bos_workbook: Any,
    assortment_workbook: Any,
    local_image_folder: str | None = None,
) -> HolidayQpResult:
    warnings: list[str] = []
    errors: list[str] = []
    bos_sheet = ""
    d5_sheet = ""
    bos_products = []
    d5_products = []
    d5_qp_products = []
    matches: list[ProductMatch] = []
    geometry = None
    placements: list[HolidayPlacement] = []
    reference_image = None

    try:
        bos_sheet, bos_products, reference_image, bos_warnings = load_bos_qp(bos_workbook)
        warnings.extend(bos_warnings)
    except Exception as exc:
        errors.append(str(exc))
    try:
        d5_sheet, d5_products, d5_warnings = load_holiday_assortment(assortment_workbook)
        warnings.extend(d5_warnings)
        d5_qp_products = [product for product in d5_products if product.qtr_facings > 0]
    except Exception as exc:
        errors.append(str(exc))

    if reference_image is not None:
        geometry = detect_geometry(reference_image)
        warnings.extend(geometry.warnings)

    image_index = build_holiday_image_index(local_image_folder)
    warnings.extend(image_index.warnings)

    if bos_products and d5_qp_products:
        matches = match_bos_to_d5(bos_products, d5_qp_products)
    if geometry is not None and matches:
        placements = _build_placements(matches, geometry.slots, image_index)

    bos_facings = sum(product.facings for product in bos_products)
    d5_qp_facings = sum(product.qtr_facings for product in d5_qp_products)
    if geometry is not None and len(geometry.slots) != bos_facings:
        warnings.append(f"Detected {len(geometry.slots)} slots but BOS requires {bos_facings} facings.")
    if bos_facings != d5_qp_facings:
        warnings.append(f"BOS facings {bos_facings} do not equal D5 QP facings {d5_qp_facings}.")

    product_rows = _product_rows(matches, placements)
    placement_rows = _placement_rows(placements)
    image_breakdown = Counter(placement.image_resolution_source for placement in placements)
    summary = {
        "bos_sheet": bos_sheet,
        "d5_sheet": d5_sheet,
        "bos_products": len(bos_products),
        "bos_facings": bos_facings,
        "d5_total_products": len(d5_products),
        "d5_qp_products": len(d5_qp_products),
        "d5_qp_facings": d5_qp_facings,
        "reference_image_width": reference_image.width if reference_image else 0,
        "reference_image_height": reference_image.height if reference_image else 0,
        "panel_count": len(geometry.panels) if geometry else 0,
        "rows_per_panel": [panel.rows for panel in geometry.panels] if geometry else [],
        "columns_per_panel": [panel.columns for panel in geometry.panels] if geometry else [],
        "active_slots": len(geometry.slots) if geometry else 0,
        "matched_products": sum(1 for match in matches if match.status == "matched"),
        "item_number_matches": sum(1 for match in matches if match.method == "item_number"),
        "upc_matches": sum(1 for match in matches if match.method == "upc"),
        "fallback_matches": sum(1 for match in matches if match.method == "name_fallback"),
        "unmatched_products": sum(1 for match in matches if match.status != "matched"),
        "resolved_placements": sum(1 for placement in placements if placement.placement_status == "resolved"),
        "unresolved_placements": sum(1 for placement in placements if placement.placement_status != "resolved"),
        "image_folder": image_index.root_dir,
        "image_folder_exists": image_index.folder_exists,
        "image_folder_is_dir": image_index.folder_is_dir,
        "image_files_indexed": image_index.indexed_files,
        "placements_with_images": sum(1 for placement in placements if placement.image_status == "resolved"),
        "unique_products_with_images": len({placement.item_number for placement in placements if placement.image_status == "resolved" and placement.item_number}),
        "image_source_breakdown": dict(image_breakdown),
    }
    return HolidayQpResult(
        bos_sheet=bos_sheet,
        d5_sheet=d5_sheet,
        reference_image=reference_image,
        bos_products=bos_products,
        d5_products=d5_products,
        d5_qp_products=d5_qp_products,
        product_matches=matches,
        geometry=geometry,
        placements=placements,
        product_qa_rows=product_rows,
        placement_qa_rows=placement_rows,
        summary=summary,
        warnings=warnings,
        errors=errors,
    )
