from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

import pandas as pd

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import (
    SAMS_HOLIDAY_TEMPLATE_NAME,
    expand_holiday_row_to_slots,
    holiday_geometry_for_side,
    holiday_slot_centers_pt,
    map_holiday_rows_to_strips,
)
from app.sams_club.price_strip_models import SamsPriceStripRow, SamsPriceStripSegment
from app.sams_club.render_price_strips_html import (
    _build_full_html,
    _calculate_centered_dollars_left,
    _estimate_text_width,
    compute_strip_canvas,
)


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
                {
                    "POG": "POG1",
                    "Side": 1,
                    "Row": row,
                    "Column": 1,
                    "Item Number": f"ITEM{row}",
                    "Brand": "BRAND",
                    "Desc 1": "DESC 1",
                    "Desc 2": "DESC 2",
                    "Retail": "25.00",
                }
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

    def test_main_amount_midpoint_equals_slot_center(self) -> None:
        center_x = holiday_slot_centers_pt(holiday_geometry_for_side(1))[0]
        left_x = _calculate_centered_dollars_left(center_x, "25", 90.0)
        width = _estimate_text_width("25", 90.0, "semibold")

        self.assert_points_close(left_x + (width / 2.0), center_x)

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
        self.assertEqual(set(brand_tops[:3]), {"59.5"})
        self.assertGreaterEqual(len(desc_tops), 6)
        self.assertEqual(set(desc_tops[0::2][:3]), {"74.9"})
        self.assertEqual(set(desc_tops[1::2][:3]), {"90.3"})


if __name__ == "__main__":
    unittest.main()
