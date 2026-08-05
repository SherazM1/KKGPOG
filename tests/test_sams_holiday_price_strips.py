from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import io
import html as html_lib
import re

import pandas as pd

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import (
    SAMS_HOLIDAY_TEMPLATE,
    SAMS_HOLIDAY_TEMPLATE_NAME,
    expand_holiday_row_to_slots,
    holiday_geometry_for_side,
    holiday_slot_centers_pt,
    map_holiday_rows_to_strips,
)
from app.sams_club.price_strip_models import SamsPriceStripRow, SamsPriceStripSegment
from app.sams_club.render_price_strips_html import (
    _build_full_html,
    _calculate_centered_price_group_left,
    _calculate_holiday_description_box,
    _estimate_text_width,
    _estimate_price_object_width,
    fit_holiday_description_text,
    _normalize_price_parts,
    compute_strip_canvas,
)


def _valid_price_strip_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "POG": "POG1",
        "Item Number": "ITEM1",
        "Brand": "BRAND",
        "Desc 1": "DESC 1",
        "Desc 2": "DESC 2",
        "Retail": "25.00",
        "Side": 1,
        "Row": 1,
        "Column": 1,
        "Length": "6",
        "Data on bottom left": "Side: 1, Row: 1 - POG: POG1",
    }
    row.update(overrides)
    return row


def _workbook_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    payload = io.BytesIO()
    with pd.ExcelWriter(payload) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return payload.getvalue()


def _holiday_description_fields(html_text: str) -> list[str]:
    matches = re.findall(
        r'<div class="field (?:brand-field|desc-field)" style="[^"]*">\s*([^<]*?)\s*</div>',
        html_text,
    )
    return [html_lib.unescape(match.strip()) for match in matches[:3]]


def _row(side: int, row: int, columns: list[int]) -> SamsPriceStripRow:
    return SamsPriceStripRow(
        pog="POG",
        side=side,
        row=row,
        segments=[
            SamsPriceStripSegment(
                pog="POG",
                side=side,
                row=row,
                column=column,
                item_number=f"ITEM{column}",
                brand="BRAND",
                desc_1="DESC 1",
                desc_2="DESC 2",
                retail="25.00",
            )
            for column in columns
        ],
    )


def _pog_row(pog: str, side: int, row: int, columns: list[int]) -> SamsPriceStripRow:
    row_data = _row(side, row, columns)
    return SamsPriceStripRow(
        pog=pog,
        side=row_data.side,
        row=row_data.row,
        segments=[
            SamsPriceStripSegment(
                pog=pog,
                side=segment.side,
                row=segment.row,
                column=segment.column,
                item_number=segment.item_number,
                brand=segment.brand,
                desc_1=segment.desc_1,
                desc_2=segment.desc_2,
                retail=segment.retail,
            )
            for segment in row_data.segments
        ],
    )


class SamsHolidayPriceStripTests(unittest.TestCase):
    def assert_points_close(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual, expected, places=5)

    def test_holiday_dimensions_and_slot_counts(self) -> None:
        thin = holiday_geometry_for_side(1)
        wide = holiday_geometry_for_side(2)

        self.assert_points_close(thin.width_pt, 2178.0)
        self.assert_points_close(thin.height_pt, 247.5)
        self.assertEqual(thin.slot_count, 6)
        self.assert_points_close(wide.width_pt, 2767.5)
        self.assert_points_close(wide.height_pt, 247.5)
        self.assertEqual(wide.slot_count, 8)

    def test_holiday_slot_centers(self) -> None:
        thin_expected = [181.5, 544.5, 907.5, 1270.5, 1633.5, 1996.5]
        wide_expected = [
            172.96875,
            518.90625,
            864.84375,
            1210.78125,
            1556.71875,
            1902.65625,
            2248.59375,
            2594.53125,
        ]

        self.assertEqual(holiday_slot_centers_pt(holiday_geometry_for_side(1)), thin_expected)
        self.assertEqual(holiday_slot_centers_pt(holiday_geometry_for_side(2)), wide_expected)

    def test_side_selection(self) -> None:
        self.assertEqual(holiday_geometry_for_side(1).designation, "thin")
        self.assertEqual(holiday_geometry_for_side(3).designation, "thin")
        self.assertEqual(holiday_geometry_for_side(2).designation, "wide")
        self.assertEqual(holiday_geometry_for_side(4).designation, "wide")

    def test_thin_rows_map_to_five_six_position_strips(self) -> None:
        rows = [_row(1, row, [1, 2, 3, 4, 5, 6]) for row in range(1, 6)]

        mapped, warnings = map_holiday_rows_to_strips(rows)

        self.assertEqual(warnings, [])
        self.assertEqual(len(mapped), 5)
        self.assertTrue(all(len(row.segments) == 6 for row in mapped))

    def test_wide_cr80_row_is_excluded(self) -> None:
        rows = [_row(2, 1, list(range(1, 10)))]
        rows.extend(_row(2, row, list(range(1, 9))) for row in range(2, 6))

        mapped, warnings = map_holiday_rows_to_strips(rows)

        self.assertEqual(warnings, [])
        self.assertEqual(len(mapped), 4)
        self.assertEqual([row.row for row in mapped], [2, 3, 4, 5])
        self.assertTrue(all(len(row.segments) == 8 for row in mapped))

    def test_wide_side_starting_at_row_two_keeps_packaging_row(self) -> None:
        rows = [_pog_row("National", 2, row, list(range(1, 9))) for row in range(2, 6)]
        rows.extend(_pog_row("National SLFCHK", 4, row, list(range(1, 9))) for row in range(2, 6))

        mapped, warnings = map_holiday_rows_to_strips(rows)

        self.assertEqual(warnings, [])
        self.assertIn(("National", 2, 2), [(row.pog, row.side, row.row) for row in mapped])
        self.assertIn(("National SLFCHK", 4, 2), [(row.pog, row.side, row.row) for row in mapped])
        self.assertEqual(
            [(row.pog, row.side, row.row) for row in mapped],
            [
                ("National", 2, 2),
                ("National", 2, 3),
                ("National", 2, 4),
                ("National", 2, 5),
                ("National SLFCHK", 4, 2),
                ("National SLFCHK", 4, 3),
                ("National SLFCHK", 4, 4),
                ("National SLFCHK", 4, 5),
            ],
        )

    def test_holiday_rows_are_ordered_by_complete_planogram_then_side_then_row(self) -> None:
        rows: list[SamsPriceStripRow] = []
        for pog in ("POG 1", "POG 2"):
            rows.extend(_pog_row(pog, 1, row, list(range(1, 7))) for row in range(1, 6))
            if pog == "POG 1":
                rows.append(_pog_row(pog, 2, 1, list(range(1, 10))))
            rows.extend(_pog_row(pog, 2, row, list(range(1, 9))) for row in range(2, 6))
            rows.extend(_pog_row(pog, 3, row, list(range(1, 7))) for row in range(1, 6))
            if pog == "POG 1":
                rows.append(_pog_row(pog, 4, 1, list(range(1, 10))))
            rows.extend(_pog_row(pog, 4, row, list(range(1, 9))) for row in range(2, 6))

        interleaved_rows = [
            row
            for side in (1, 2, 3, 4)
            for row_number in range(1, 6)
            for row in rows
            if row.side == side and row.row == row_number
        ]

        mapped, warnings = map_holiday_rows_to_strips(interleaved_rows)

        self.assertEqual(warnings, [])
        self.assertEqual(
            [(row.pog, row.side, row.row) for row in mapped],
            [
                ("POG 1", 1, 1),
                ("POG 1", 1, 2),
                ("POG 1", 1, 3),
                ("POG 1", 1, 4),
                ("POG 1", 1, 5),
                ("POG 1", 2, 2),
                ("POG 1", 2, 3),
                ("POG 1", 2, 4),
                ("POG 1", 2, 5),
                ("POG 1", 3, 1),
                ("POG 1", 3, 2),
                ("POG 1", 3, 3),
                ("POG 1", 3, 4),
                ("POG 1", 3, 5),
                ("POG 1", 4, 2),
                ("POG 1", 4, 3),
                ("POG 1", 4, 4),
                ("POG 1", 4, 5),
                ("POG 2", 1, 1),
                ("POG 2", 1, 2),
                ("POG 2", 1, 3),
                ("POG 2", 1, 4),
                ("POG 2", 1, 5),
                ("POG 2", 2, 2),
                ("POG 2", 2, 3),
                ("POG 2", 2, 4),
                ("POG 2", 2, 5),
                ("POG 2", 3, 1),
                ("POG 2", 3, 2),
                ("POG 2", 3, 3),
                ("POG 2", 3, 4),
                ("POG 2", 3, 5),
                ("POG 2", 4, 2),
                ("POG 2", 4, 3),
                ("POG 2", 4, 4),
                ("POG 2", 4, 5),
            ],
        )

    def test_empty_positions_do_not_shift_later_prices(self) -> None:
        expanded = expand_holiday_row_to_slots(_row(1, 1, [1, 3, 6]))

        self.assertEqual([segment.column for segment in expanded.segments], [1, 2, 3, 4, 5, 6])
        self.assertTrue(expanded.segments[1].is_empty)
        self.assertEqual(expanded.segments[2].item_number, "ITEM3")
        self.assertEqual(expanded.segments[5].item_number, "ITEM6")

    def test_incorrect_counts_produce_actionable_validation_failure(self) -> None:
        mapped, warnings = map_holiday_rows_to_strips([_row(2, 1, list(range(1, 10)))])

        self.assertEqual(mapped, [])
        self.assertTrue(
            any("expected 4 priced rows of 8 positions after excluding the CR80 row" in warning for warning in warnings)
        )

    def test_compute_strip_canvas_uses_exact_holiday_trim_size(self) -> None:
        warnings: list[str] = []
        thin_w, thin_h, _ = compute_strip_canvas(_row(1, 1, [1]), warnings, SAMS_HOLIDAY_TEMPLATE_NAME)
        wide_w, wide_h, _ = compute_strip_canvas(_row(2, 1, [1]), warnings, SAMS_HOLIDAY_TEMPLATE_NAME)

        self.assert_points_close(thin_w, 2178.0)
        self.assert_points_close(thin_h, 247.5)
        self.assert_points_close(wide_w, 2767.5)
        self.assert_points_close(wide_h, 247.5)

    def test_build_sams_price_strip_rows_accepts_holiday_template_keyword(self) -> None:
        workbook_rows = []
        for row in range(1, 6):
            workbook_rows.append(
                _valid_price_strip_row(
                    Row=row,
                    **{
                        "Item Number": f"ITEM{row}",
                        "Data on bottom left": f"Side: 1, Row: {row} - POG: POG1",
                    },
                )
            )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as handle:
            workbook_path = Path(handle.name)

        try:
            with pd.ExcelWriter(workbook_path) as writer:
                pd.DataFrame(workbook_rows).to_excel(writer, sheet_name="Price Strip Data", index=False)

            result = build_sams_price_strip_rows(workbook_path, template_name=SAMS_HOLIDAY_TEMPLATE_NAME)

            self.assertEqual(result.errors, [])
            self.assertEqual(result.debug["template_name"], SAMS_HOLIDAY_TEMPLATE_NAME)
            self.assertEqual(len(result.strip_rows), 5)
            self.assertTrue(all(len(row.segments) == 6 for row in result.strip_rows))
        finally:
            workbook_path.unlink(missing_ok=True)

    def test_price_strip_import_selects_exact_price_strip_data_sheet(self) -> None:
        workbook = _workbook_bytes({"Price Strip Data": pd.DataFrame([_valid_price_strip_row()])})

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "Price Strip Data")
        self.assertEqual(result.included_segment_count, 1)

    def test_price_strip_import_accepts_lowercase_sheet_name(self) -> None:
        workbook = _workbook_bytes({"price strip data": pd.DataFrame([_valid_price_strip_row()])})

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "price strip data")
        self.assertEqual(result.debug["sheet_selection_reason"], "normalized worksheet-name match")

    def test_price_strip_import_accepts_trailing_space_sheet_name(self) -> None:
        workbook = _workbook_bytes({"Price Strip Data ": pd.DataFrame([_valid_price_strip_row()])})

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "Price Strip Data ")

    def test_price_strip_import_accepts_single_sheet1_with_valid_headers(self) -> None:
        workbook = _workbook_bytes({"Sheet1": pd.DataFrame([_valid_price_strip_row()])})

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "Sheet1")
        self.assertEqual(result.debug["sheet_selection_reason"], "header schema match")

    def test_price_strip_import_ignores_readme_when_valid_production_sheet_exists(self) -> None:
        workbook = _workbook_bytes(
            {
                "README": pd.DataFrame([{"Notes": "Use the production worksheet."}]),
                "Production": pd.DataFrame([_valid_price_strip_row()]),
            }
        )

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "Production")

    def test_price_strip_import_accepts_uppercase_desc_headers(self) -> None:
        row = _valid_price_strip_row()
        row["DESC 1"] = row.pop("Desc 1")
        row["DESC 2"] = row.pop("Desc 2")
        workbook = _workbook_bytes({"Price Strip Data": pd.DataFrame([row])})

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.strip_rows[0].segments[0].desc_1, "DESC 1")
        self.assertEqual(result.strip_rows[0].segments[0].desc_2, "DESC 2")

    def test_price_strip_import_reports_invalid_workbook_without_matching_headers(self) -> None:
        workbook = _workbook_bytes(
            {
                "Sheet1": pd.DataFrame([{"POG": "POG1", "Side": 1}]),
                "README": pd.DataFrame([{"Notes": "No production data here."}]),
            }
        )

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(len(result.errors), 1)
        self.assertIn("Could not locate a usable price-strip worksheet.", result.errors[0])
        self.assertIn("Sheets found: Sheet1, README", result.errors[0])
        self.assertIn("Closest match: Sheet1", result.errors[0])
        self.assertIn("Missing required columns:", result.errors[0])
        self.assertIn("Row", result.errors[0])
        self.assertIn("Column", result.errors[0])

    def test_price_strip_import_selects_complete_schema_when_multiple_sheets_exist(self) -> None:
        workbook = _workbook_bytes(
            {
                "Almost": pd.DataFrame([{key: value for key, value in _valid_price_strip_row().items() if key != "Column"}]),
                "Production": pd.DataFrame([_valid_price_strip_row(POG="GOOD")]),
            }
        )

        result = build_sams_price_strip_rows(workbook)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.debug["sheet_name"], "Production")
        self.assertEqual(result.strip_rows[0].pog, "GOOD")

    def test_full_price_group_midpoint_equals_slot_center(self) -> None:
        center_x = holiday_slot_centers_pt(holiday_geometry_for_side(1))[0]
        width = _estimate_price_object_width(
            "25",
            "00",
            SAMS_HOLIDAY_TEMPLATE.price_dollar_sign_font_size_pt,
            SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt,
            SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt,
            0.6,
            0.6,
        )
        left_x = _calculate_centered_price_group_left(center_x, width)

        self.assert_points_close(left_x + (width / 2.0), center_x)

    def test_holiday_description_uses_bounded_single_digit_shift(self) -> None:
        single_digit_prices = ["1.23", "4.98", "5.98"]
        stable_prices = ["24.13", "51.25", "153.78"]
        center_x = holiday_slot_centers_pt(holiday_geometry_for_side(1))[0]
        slot_width = 2178.0 / 6.0

        observed: dict[str, tuple[float, float, float, float]] = {}

        for price in single_digit_prices + stable_prices:
            row = SamsPriceStripRow(
                pog="POG",
                side=1,
                row=1,
                segments=[
                    SamsPriceStripSegment(
                        pog="POG",
                        side=1,
                        row=1,
                        column=1,
                        brand="BRAND",
                        desc_1="DESCRIPTION",
                        desc_2="$20-$500",
                        retail=price,
                    )
                ],
            )
            warnings: list[str] = []
            html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
            ticket_left = float(re.search(r'<div class="ticket" style="left: ([0-9.]+)pt;', html).group(1))
            price_left = float(re.search(r'<div class="price" style="left: ([0-9.]+)pt;', html).group(1))
            desc_left = float(re.search(r'class="field brand-field" style="left: ([0-9.]+)pt;', html).group(1))
            desc_width = float(re.search(r'class="field brand-field" style="left: [0-9.]+pt; top: [0-9.]+pt; width: ([0-9.]+)pt;', html).group(1))
            dollars, cents = _normalize_price_parts(price)
            group_width = _estimate_price_object_width(
                dollars,
                cents,
                SAMS_HOLIDAY_TEMPLATE.price_dollar_sign_font_size_pt,
                SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt,
                SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt,
                0.6,
                0.6,
            )
            group_left = ticket_left + price_left

            self.assert_points_close(group_left + (group_width / 2.0), center_x)
            self.assertGreaterEqual(ticket_left + desc_left, ticket_left + SAMS_HOLIDAY_TEMPLATE.description_inset_pt)
            self.assertLessEqual(
                ticket_left + desc_left,
                ticket_left + SAMS_HOLIDAY_TEMPLATE.description_inset_pt + SAMS_HOLIDAY_TEMPLATE.maximum_description_shift_pt,
            )
            self.assertLessEqual(ticket_left + desc_left + desc_width, ticket_left + slot_width - SAMS_HOLIDAY_TEMPLATE.sku_inset_pt)
            observed[price] = (ticket_left, desc_left, desc_width, price_left)

        expected_shift = SAMS_HOLIDAY_TEMPLATE.single_digit_description_shift_pt
        for price in single_digit_prices:
            self.assertAlmostEqual(observed[price][1], SAMS_HOLIDAY_TEMPLATE.description_inset_pt + expected_shift)
        for price in stable_prices:
            self.assertAlmostEqual(observed[price][1], SAMS_HOLIDAY_TEMPLATE.description_inset_pt)

    def test_holiday_final_markup_uses_target_typography_and_uppercase(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(
                    pog="POG",
                    side=1,
                    row=1,
                    column=1,
                    brand="Zift Holiday",
                    desc_1="Gift Purple",
                    desc_2="Gift For You",
                    retail="47.88",
                    item_number="990000000",
                )
            ],
        )
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
        item_match = re.search(r'class="field item-field" style="left: ([0-9.]+)pt; top: [0-9.]+pt; width: ([0-9.]+)pt; font-size: ([0-9.]+)pt;', html)

        self.assertIn("ZIFT HOLIDAY", html)
        self.assertIn("GIFT PURPLE", html)
        self.assertIn("GIFT FOR YOU", html)
        self.assertNotIn("Zift Holiday", html)
        self.assertNotIn("Gift Purple", html)
        self.assertNotIn("...", html)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.description_font_size_pt, 18.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.price_dollar_sign_font_size_pt, 38.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.brand_top_pt, 45.5)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.description_top_pt, 60.9)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.pack_range_top_pt, 76.3)
        self.assertAlmostEqual(
            SAMS_HOLIDAY_TEMPLATE.description_top_pt - SAMS_HOLIDAY_TEMPLATE.brand_top_pt,
            15.4,
        )
        self.assertAlmostEqual(
            SAMS_HOLIDAY_TEMPLATE.pack_range_top_pt - SAMS_HOLIDAY_TEMPLATE.description_top_pt,
            15.4,
        )
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt, 110.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt, 44.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.price_cents_translate_y_pt, -6.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.footer_font_size_pt, 8.0)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.sku_font_size_pt, 9.0)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.price_dollar_sign_font_size_pt}pt;", html)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt}pt;", html)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt}pt;", html)
        self.assertIn(f"transform: translateY({SAMS_HOLIDAY_TEMPLATE.price_cents_translate_y_pt}pt);", html)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.footer_font_size_pt}pt;", html)
        self.assertAlmostEqual(float(item_match.group(3)), SAMS_HOLIDAY_TEMPLATE.sku_font_size_pt)

    def test_holiday_cents_vertical_offset_is_fixed_across_prices(self) -> None:
        prices = ["5.98", "22.50", "38.58", "78.58", "153.78", "201.25"]
        observed_price_lefts: list[float] = []
        observed_transforms: set[str] = set()

        for price in prices:
            row = SamsPriceStripRow(
                pog="POG",
                side=1,
                row=1,
                segments=[
                    SamsPriceStripSegment(
                        pog="POG",
                        side=1,
                        row=1,
                        column=1,
                        brand="BRAND",
                        desc_1="DESCRIPTION",
                        desc_2="PACK",
                        retail=price,
                    )
                ],
            )
            warnings: list[str] = []
            html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
            price_left = float(re.search(r'<div class="price" style="left: ([0-9.]+)pt;', html).group(1))
            transform = re.search(r'\.cents \{[^}]*transform: translateY\((-?[0-9.]+)pt\);', html).group(1)
            cents_font = re.search(r'\.cents \{[^}]*font-size: ([0-9.]+)pt;', html).group(1)

            dollars, cents = _normalize_price_parts(price)
            group_width = _estimate_price_object_width(
                dollars,
                cents,
                SAMS_HOLIDAY_TEMPLATE.price_dollar_sign_font_size_pt,
                SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt,
                SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt,
                0.6,
                0.6,
            )
            ticket_left = float(re.search(r'<div class="ticket" style="left: ([0-9.]+)pt;', html).group(1))
            center_x = holiday_slot_centers_pt(holiday_geometry_for_side(1))[0]

            observed_price_lefts.append(price_left)
            observed_transforms.add(transform)
            self.assertAlmostEqual(float(cents_font), SAMS_HOLIDAY_TEMPLATE.price_cents_font_size_pt)
            self.assert_points_close(ticket_left + price_left + (group_width / 2.0), center_x)

        self.assertEqual(observed_transforms, {str(SAMS_HOLIDAY_TEMPLATE.price_cents_translate_y_pt)})

    def test_holiday_footer_and_sku_x_are_fixed_across_prices(self) -> None:
        prices = ["1.23", "5.98", "24.13", "51.25", "153.78"]
        footer_lefts: list[float] = []
        sku_rights: list[float] = []

        for price in prices:
            row = SamsPriceStripRow(
                pog="POG",
                side=1,
                row=1,
                segments=[
                    SamsPriceStripSegment(
                        pog="POG",
                        side=1,
                        row=1,
                        column=1,
                        brand="BRAND",
                        desc_1="DESCRIPTION",
                        desc_2="PACK",
                        retail=price,
                        item_number="990000000",
                    )
                ],
            )
            warnings: list[str] = []
            html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
            footer_lefts.append(float(re.search(r'\.footer \{[^}]*left: ([0-9.]+)pt;', html).group(1)))
            item_match = re.search(r'class="field item-field" style="left: ([0-9.]+)pt; top: [0-9.]+pt; width: ([0-9.]+)pt;', html)
            sku_rights.append(float(item_match.group(1)) + float(item_match.group(2)))

        self.assertEqual(len(set(footer_lefts)), 1)
        self.assertEqual(len({round(value, 5) for value in sku_rights}), 1)
        self.assertAlmostEqual(footer_lefts[0], SAMS_HOLIDAY_TEMPLATE.footer_inset_pt)
        self.assertAlmostEqual(sku_rights[0], (2178.0 / 6.0) - SAMS_HOLIDAY_TEMPLATE.sku_inset_pt)
        self.assertAlmostEqual(SAMS_HOLIDAY_TEMPLATE.sku_inset_pt, 24.0)

    def test_holiday_long_three_line_description_stays_inside_slot(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(
                    pog="POG",
                    side=1,
                    row=1,
                    column=1,
                    brand="",
                    desc_1="Golden Corral $30 Multipack",
                    desc_2="$75 (3 X $25) + $10 BONUS",
                    retail="153.78",
                )
            ],
        )
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

        ticket_left = float(re.search(r'<div class="ticket" style="left: ([0-9.]+)pt;', html).group(1))
        brand_match = re.search(r'class="field brand-field" style="left: ([0-9.]+)pt; top: [0-9.]+pt; width: ([0-9.]+)pt;', html)
        desc_left = float(brand_match.group(1))
        desc_width = float(brand_match.group(2))

        self.assertGreaterEqual(ticket_left + desc_left, ticket_left)
        self.assertLessEqual(ticket_left + desc_left + desc_width, ticket_left + (2178.0 / 6.0) - SAMS_HOLIDAY_TEMPLATE.sku_inset_pt)
        self.assertNotIn("...", html)
        self.assertEqual(
            _holiday_description_fields(html),
            ["", "GOLDEN CORRAL $30 MULTIPACK", "$75 (3 X $25) + $10 BONUS"],
        )
        for token in ["GOLDEN", "CORRAL", "$30", "MULTIPACK", "$75", "BONUS"]:
            self.assertIn(token, html)

    def test_holiday_description_fields_render_as_explicit_source_lines(self) -> None:
        cases = [
            (
                "Logans",
                "LOGAN'S MP $100",
                "$100 (2 X $50)",
                "GIFTCARDS",
                ["LOGAN'S MP $100", "$100 (2 X $50)", "GIFTCARDS"],
            ),
            (
                "MasterCard",
                "MasterCard $75 Multipack",
                "$75 (3 X $25)",
                "Gift Cards",
                ["MASTERCARD $75 MULTIPACK", "$75 (3 X $25)", "GIFT CARDS"],
            ),
            (
                "AMC",
                "AMC Theatres",
                "1 X $30",
                "Gift Card",
                ["AMC THEATRES", "1 X $30", "GIFT CARD"],
            ),
            (
                "LongDesc",
                "Golden Corral $30 Multipack",
                "$75 (3 X $25) + $10 Bonus",
                "Gift Cards",
                [
                    "GOLDEN CORRAL $30 MULTIPACK",
                    "$75 (3 X $25) + $10 BONUS",
                    "GIFT CARDS",
                ],
            ),
        ]

        for _label, brand, desc_1, desc_2, expected_lines in cases:
            row = SamsPriceStripRow(
                pog="POG",
                side=1,
                row=1,
                segments=[
                    SamsPriceStripSegment(
                        pog="POG",
                        side=1,
                        row=1,
                        column=1,
                        brand=brand,
                        desc_1=desc_1,
                        desc_2=desc_2,
                        retail="24.13",
                    )
                ],
            )
            warnings: list[str] = []
            html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

            self.assertEqual(_holiday_description_fields(html), expected_lines)
            self.assertNotIn(f"{expected_lines[0]} {expected_lines[1]}", html)
            self.assertNotIn(f"{expected_lines[1]} {expected_lines[2]}", html)
            self.assertEqual(len(re.findall(r'class="field brand-field"', html)), 1)
            self.assertEqual(len(re.findall(r'class="field desc-field"', html)), 2)

    def test_holiday_long_descriptions_are_uppercase_and_not_truncated_with_ellipsis(self) -> None:
        descriptions = [
            "MasterCard $75 Multipack",
            "Starbucks $40 Multipack",
            "Olive Garden $75 Multipack",
            "Golden Corral $30 Multipack",
            "$75 (3 X $25) + $10 BONUS",
        ]

        for description in descriptions:
            row = SamsPriceStripRow(
                pog="POG",
                side=1,
                row=1,
                segments=[
                    SamsPriceStripSegment(
                        pog="POG",
                        side=1,
                        row=1,
                        column=1,
                        brand="",
                        desc_1=description,
                        desc_2="",
                        retail="24.13",
                    )
                ],
            )
            warnings: list[str] = []
            html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

            self.assertNotIn("...", html)
            for token in description.split():
                self.assertIn(token.upper(), html)

    def test_holiday_uppercases_description_lines_without_mutating_source(self) -> None:
        segment = SamsPriceStripSegment(
            pog="POG",
            side=1,
            row=1,
            column=1,
            brand="Mixed Brand",
            desc_1="Mixed Description",
            desc_2="Range Pack",
            retail="24.13",
        )
        row = SamsPriceStripRow(pog="POG", side=1, row=1, segments=[segment])
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

        self.assertIn("MIXED BRAND", html)
        self.assertIn("MIXED DESCRIPTION", html)
        self.assertIn("RANGE PACK", html)
        self.assertEqual(segment.brand, "Mixed Brand")
        self.assertEqual(segment.desc_1, "Mixed Description")
        self.assertEqual(segment.desc_2, "Range Pack")

    def test_holiday_short_description_coordinates_and_font_remain_unchanged(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(
                    pog="POG",
                    side=1,
                    row=1,
                    column=1,
                    brand="BRAND",
                    desc_1="SHORT DESC",
                    desc_2="PACK",
                    retail="24.13",
                )
            ],
        )
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
        brand_match = re.search(
            r'class="field brand-field" style="left: ([0-9.]+)pt; top: ([0-9.]+)pt; width: ([0-9.]+)pt; font-size: ([0-9.]+)pt;',
            html,
        )

        self.assertAlmostEqual(float(brand_match.group(1)), SAMS_HOLIDAY_TEMPLATE.description_inset_pt)
        self.assertAlmostEqual(float(brand_match.group(2)), SAMS_HOLIDAY_TEMPLATE.brand_top_pt)
        self.assertAlmostEqual(float(brand_match.group(3)), SAMS_HOLIDAY_TEMPLATE.description_box_width_pt)
        self.assertAlmostEqual(float(brand_match.group(4)), SAMS_HOLIDAY_TEMPLATE.description_font_size_pt)

    def test_holiday_footer_font_size_increases_without_moving_anchor(self) -> None:
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(_row(1, 1, [1]))], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
        footer_match = re.search(r'\.footer \{[^}]*left: ([0-9.]+)pt;[^}]*font-size: ([0-9.]+)pt;', html)

        self.assertAlmostEqual(float(footer_match.group(1)), SAMS_HOLIDAY_TEMPLATE.footer_inset_pt)
        self.assertAlmostEqual(float(footer_match.group(2)), SAMS_HOLIDAY_TEMPLATE.footer_font_size_pt)

    def test_holiday_description_fitting_keeps_source_lines_and_scales_independently(self) -> None:
        left, width = _calculate_holiday_description_box(0.0, 2178.0 / 6.0, "24")

        fit = fit_holiday_description_text(
            "MASTERCARD $75 MULTIPACK",
            "$75 (3 X $25)",
            "GIFT CARDS",
            0.0,
            2178.0 / 6.0,
            left,
            SAMS_HOLIDAY_TEMPLATE.brand_top_pt,
            width,
            SAMS_HOLIDAY_TEMPLATE.description_font_size_pt,
            63.72,
        )

        self.assertEqual(fit.lines, ("MASTERCARD $75 MULTIPACK", "$75 (3 X $25)", "GIFT CARDS"))
        self.assertAlmostEqual(fit.font_size, SAMS_HOLIDAY_TEMPLATE.description_font_size_pt)
        self.assertEqual(fit.top, SAMS_HOLIDAY_TEMPLATE.brand_top_pt)
        self.assertTrue(all(0 < scale <= 1.0 for scale in fit.line_scales))
        self.assertLess(fit.line_scales[0], 1.0)

    def test_holiday_long_single_word_scales_without_splitting(self) -> None:
        left, width = _calculate_holiday_description_box(0.0, 2178.0 / 6.0, "24")
        fit = fit_holiday_description_text(
            "",
            "SUPERCALIFRAGILISTICBONUSCARD",
            "",
            0.0,
            2178.0 / 6.0,
            left,
            SAMS_HOLIDAY_TEMPLATE.brand_top_pt,
            width,
            SAMS_HOLIDAY_TEMPLATE.description_font_size_pt,
            63.72,
        )

        self.assertEqual(fit.lines, ("", "SUPERCALIFRAGILISTICBONUSCARD", ""))
        self.assertLess(fit.line_scales[1], 1.0)
        self.assertNotIn("...", " ".join(fit.lines))

    def test_holiday_unfittable_description_raises_clear_error(self) -> None:
        left, width = _calculate_holiday_description_box(0.0, 2178.0 / 6.0, "24")

        with self.assertRaisesRegex(ValueError, "does not fit within the slot without truncation"):
            fit_holiday_description_text(
                "",
                " ".join(["OVERLONG"] * 80),
                "",
                0.0,
                2178.0 / 6.0,
                left,
                SAMS_HOLIDAY_TEMPLATE.brand_top_pt,
                width,
                SAMS_HOLIDAY_TEMPLATE.description_font_size_pt,
                63.72,
            )

    def test_non_holiday_rendering_does_not_uppercase_or_change_footer_font(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(
                    pog="POG",
                    side=1,
                    row=1,
                    column=1,
                    brand="Mixed Brand",
                    desc_1="Mixed Description",
                    desc_2="Range Pack",
                    retail="24.13",
                )
            ],
        )
        warnings: list[str] = []
        html = _build_full_html([row], warnings, None, False)[0]
        footer_match = re.search(r'\.footer \{[^}]*font-size: ([0-9.]+)pt;', html)

        self.assertIn("Mixed Brand", html)
        self.assertIn("Mixed Description", html)
        self.assertIn("Range Pack", html)
        self.assertAlmostEqual(float(footer_match.group(1)), 8.0)

    def test_holiday_end_to_end_output_file_preserves_final_markup_targets(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(
                    pog="POG",
                    side=1,
                    row=1,
                    column=1,
                    brand="Zift Holiday",
                    desc_1="Gift Purple",
                    desc_2="Gift For You",
                    retail="47.88",
                    item_number="990000000",
                )
            ],
        )
        expanded = expand_holiday_row_to_slots(row)
        warnings: list[str] = []
        html = _build_full_html([expanded], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as handle:
            output_path = Path(handle.name)
            handle.write(html)

        try:
            generated_output = output_path.read_text(encoding="utf-8")
        finally:
            output_path.unlink(missing_ok=True)

        self.assertIn("ZIFT HOLIDAY", generated_output)
        self.assertIn("GIFT PURPLE", generated_output)
        self.assertNotIn("Zift Holiday", generated_output)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.description_font_size_pt}pt;", generated_output)
        self.assertIn(f"font-size: {SAMS_HOLIDAY_TEMPLATE.price_dollars_font_size_pt}pt;", generated_output)

    def test_production_mode_does_not_draw_calibration_guides(self) -> None:
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(_row(1, 1, [1, 2, 3]))], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

        self.assertNotIn("holiday-calibration", html)

    def test_calibration_mode_draws_guides_at_all_slot_centers(self) -> None:
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(_row(2, 2, [1, 2, 3, 4]))], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, True)[0]

        self.assertIn("holiday-calibration", html)
        for center in holiday_slot_centers_pt(holiday_geometry_for_side(2)):
            self.assertIn(f"left:{center}pt", html)

    def test_holiday_text_y_anchors_do_not_depend_on_content(self) -> None:
        row = SamsPriceStripRow(
            pog="POG",
            side=1,
            row=1,
            segments=[
                SamsPriceStripSegment(pog="POG", side=1, row=1, column=1, brand="BRAND", desc_1="ONE LINE", desc_2="", retail="25.00"),
                SamsPriceStripSegment(pog="POG", side=1, row=1, column=2, brand="BRAND", desc_1="TWO LINE", desc_2="$20-$500", retail="25.00"),
                SamsPriceStripSegment(pog="POG", side=1, row=1, column=3, brand="", desc_1="", desc_2="", retail="25.00"),
            ],
        )
        warnings: list[str] = []
        html = _build_full_html([expand_holiday_row_to_slots(row)], warnings, SAMS_HOLIDAY_TEMPLATE_NAME, False)[0]

        brand_tops = re.findall(r'class="field brand-field" style="[^"]* top: ([0-9.]+)pt;', html)
        desc_tops = re.findall(r'class="field desc-field" style="[^"]* top: ([0-9.]+)pt;', html)

        self.assertGreaterEqual(len(brand_tops), 3)
        self.assertEqual(set(brand_tops[:3]), {str(SAMS_HOLIDAY_TEMPLATE.brand_top_pt)})
        self.assertGreaterEqual(len(desc_tops), 6)
        self.assertEqual(set(desc_tops[0::2][:3]), {str(SAMS_HOLIDAY_TEMPLATE.description_top_pt)})
        self.assertEqual(set(desc_tops[1::2][:3]), {str(SAMS_HOLIDAY_TEMPLATE.pack_range_top_pt)})


if __name__ == "__main__":
    unittest.main()
