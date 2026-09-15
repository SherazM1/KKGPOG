from __future__ import annotations

import io
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.holiday_planograms.extract_assortment import load_holiday_assortment
from app.holiday_planograms.extract_bos import load_bos_qp
from app.holiday_planograms.geometry import detect_geometry
from app.holiday_planograms.matching import match_bos_to_d5
from app.holiday_planograms.models import HolidayPlacement, ReferenceImage
from app.holiday_planograms.normalization import digits_only, identifier_variants, int_value
from app.holiday_planograms.render_qp import (
    _compute_grid_layout,
    _compute_page_dimensions_for_panel,
    _contain_image_rect,
    _preferred_max_card_width,
    render_holiday_qp,
)
from app.holiday_planograms.service import build_holiday_qp_qa


BOS_WORKBOOK = Path.home() / "Downloads" / "BOS Holiday2026_Planograms.xlsx"
D5_WORKBOOK = Path.home() / "Downloads" / "D5_Holiday_Assortment_0505.xlsx"


def _require_supplied_workbooks() -> None:
    if not BOS_WORKBOOK.exists() or not D5_WORKBOOK.exists():
        raise unittest.SkipTest("Supplied Holiday 2026 workbooks are not available.")


class HolidayQpTests(unittest.TestCase):
    def test_d5_parser_supplied_workbook_counts(self) -> None:
        _require_supplied_workbooks()

        sheet, products, _warnings = load_holiday_assortment(str(D5_WORKBOOK))
        qp_products = [product for product in products if product.qtr_facings > 0]

        self.assertEqual(sheet, "AssortmentList_D5_Holidays")
        self.assertEqual(len(products), 34)
        self.assertEqual(len(qp_products), 22)
        self.assertEqual(sum(product.qtr_facings for product in qp_products), 60)

    def test_bos_qp_table_supplied_workbook_counts(self) -> None:
        _require_supplied_workbooks()

        sheet, products, reference_image, warnings = load_bos_qp(str(BOS_WORKBOOK))

        self.assertEqual(sheet, "2026 QTR PALLET HOLIDAY")
        self.assertEqual(len(products), 22)
        self.assertEqual(sum(product.facings for product in products), 60)
        self.assertIsNotNone(reference_image)
        self.assertFalse([warning for warning in warnings if "required" in warning.lower()])

    def test_bos_to_d5_supplied_workbook_matches_by_item_number(self) -> None:
        _require_supplied_workbooks()

        _sheet, bos_products, _image, _warnings = load_bos_qp(str(BOS_WORKBOOK))
        _d5_sheet, d5_products, _d5_warnings = load_holiday_assortment(str(D5_WORKBOOK))
        matches = match_bos_to_d5(bos_products, [product for product in d5_products if product.qtr_facings > 0])

        self.assertEqual(len(matches), 22)
        self.assertEqual(sum(1 for match in matches if match.status == "matched"), 22)
        self.assertEqual(sum(1 for match in matches if match.method == "item_number"), 22)

    def test_upc_normalization_variants(self) -> None:
        self.assertEqual(digits_only("00123456789.0"), "00123456789")
        self.assertEqual(digits_only(196742078664.0), "196742078664")
        variants = identifier_variants("019674207866")
        self.assertIn("019674207866", variants)
        self.assertIn("19674207866", variants)
        self.assertIn("196742078664", variants)
        self.assertEqual(digits_only(""), "")

    def test_facing_normalization(self) -> None:
        self.assertEqual(int_value(4), 4)
        self.assertEqual(int_value(4.0), 4)
        self.assertIsNone(int_value(""))
        self.assertIsNone(int_value(-1))
        self.assertIsNone(int_value(1.5))

    def test_geometry_synthetic_grid_keeps_missing_slots_inactive(self) -> None:
        image = Image.new("RGB", (500, 260), "white")
        draw = ImageDraw.Draw(image)
        for panel in range(2):
            x_offset = 20 + panel * 240
            for row in range(3):
                for col in range(4):
                    if panel == 1 and row == 2 and col == 3:
                        continue
                    x0 = x_offset + col * 45
                    y0 = 30 + row * 65
                    draw.rectangle((x0, y0, x0 + 28, y0 + 45), outline="black", width=1)
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        result = detect_geometry(
            ReferenceImage(
                name="synthetic",
                bytes_data=buf.getvalue(),
                width=image.width,
                height=image.height,
                anchor_row=0,
                anchor_col=0,
            )
        )

        self.assertEqual(len(result.panels), 2)
        self.assertEqual(len(result.slots), 23)
        self.assertEqual([panel.active_slots for panel in result.panels], [12, 11])

    def test_current_qp_integration_supplied_reference_geometry(self) -> None:
        _require_supplied_workbooks()

        _sheet, _products, reference_image, _warnings = load_bos_qp(str(BOS_WORKBOOK))
        self.assertIsNotNone(reference_image)
        result = detect_geometry(reference_image)

        self.assertEqual(len(result.panels), 4)
        self.assertEqual(len(result.slots), 60)
        self.assertEqual([panel.rows for panel in result.panels], [5, 5, 5, 5])
        self.assertEqual([panel.columns for panel in result.panels], [3, 3, 3, 3])

    def test_current_qp_service_supplied_workbooks(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_qp_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), r"Z:\Kendal King\Images")

        self.assertEqual(result.errors, [])
        self.assertEqual(result.summary["bos_products"], 22)
        self.assertEqual(result.summary["bos_facings"], 60)
        self.assertEqual(result.summary["d5_qp_products"], 22)
        self.assertEqual(result.summary["d5_qp_facings"], 60)
        self.assertEqual(result.summary["active_slots"], 60)
        self.assertEqual(result.summary["resolved_placements"], 60)

    def test_render_uses_image_for_resolved_path_and_placeholder_for_missing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "card.png"
            Image.new("RGB", (80, 120), "green").save(image_path)
            placements = [
                self._placement(1, 1, image_path=str(image_path), image_status="resolved"),
                self._placement(1, 2, item_number="2", product_upc="222222222222", image_status="missing"),
            ]

            result = render_holiday_qp(placements)

        self.assertGreater(len(result.pdf_bytes), 0)
        self.assertGreater(len(result.preview_png_bytes), 0)
        self.assertEqual(result.slots_rendered, 2)
        self.assertEqual(result.image_slots, 1)
        self.assertEqual(result.placeholder_slots, 1)
        self.assertEqual(result.skipped_slots, 0)
        self.assertEqual([row["Content Type"] for row in result.render_rows], ["image", "placeholder"])
        self.assertEqual(
            [row["Metadata Fields"] for row in result.render_rows],
            ["Price; CPP; UPC; ITEM; Product Name", "Price; CPP; UPC; ITEM; Product Name"],
        )
        self.assertEqual(result.render_rows[0]["UPC"], "111111111111")
        self.assertEqual(result.render_rows[0]["ITEM"], "123")
        self.assertEqual(result.render_rows[0]["Price"], "$10")
        self.assertNotIn("Denom / Load Range", result.render_rows[0])
        self.assertNotIn("Denom / Load Range", result.render_rows[1])
        self.assertNotIn("Product 12 Digit UPC", result.render_rows[0])

        import fitz

        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        self.assertIn("$10", text)
        self.assertIn("UPC 111111111111", text)
        self.assertIn("ITEM 123", text)
        self.assertIn("Test Product", text)
        self.assertIn("CPP 10", text)
        self.assertNotIn("Product 12 Digit UPC", text)
        self.assertNotIn("Product Name", text)
        self.assertNotIn("DENOM", text)
        self.assertNotIn("LOAD RANGE", text)

    def test_render_placeholder_missing_report_preserves_d5_metadata(self) -> None:
        placement = self._placement(
            1,
            1,
            product_name="Xbox Microsoft Gift Card VGC ($10-$250)",
            denomination="$10 - $250",
            product_upc="196742040135",
            item_number="662209621",
            cpp=20,
            image_status="missing",
        )

        result = render_holiday_qp([placement])

        self.assertEqual(result.placeholder_slots, 1)
        self.assertEqual(result.missing_image_rows[0]["Product Name"], "Xbox Microsoft Gift Card VGC ($10-$250)")
        self.assertEqual(result.missing_image_rows[0]["Denom / Load Range"], "$10 - $250")
        self.assertEqual(result.missing_image_rows[0]["Product 12 Digit UPC"], "196742040135")
        self.assertEqual(result.missing_image_rows[0]["WM Item Number"], "662209621")
        self.assertEqual(result.missing_image_rows[0]["QP Facings"], 1)

    def test_render_counts_sixty_placements(self) -> None:
        placements = [
            self._placement(
                ((index // 15) + 1),
                ((index % 15) // 3) + 1,
                column=((index % 3) + 1),
                item_number=str(1000 + index),
                product_upc=str(196742000000 + index),
                image_status="missing",
            )
            for index in range(60)
        ]

        result = render_holiday_qp(placements)

        self.assertEqual(result.panels_rendered, 4)
        self.assertEqual(result.slots_rendered, 60)
        self.assertEqual(result.image_slots, 0)
        self.assertEqual(result.placeholder_slots, 60)
        import fitz

        doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
        self.assertEqual(doc.page_count, 4)

    def test_bad_resolved_image_path_falls_back_to_placeholder(self) -> None:
        placement = self._placement(1, 1, image_path="Z:/missing/card.png", image_status="resolved")

        result = render_holiday_qp([placement])

        self.assertEqual(result.image_slots, 0)
        self.assertEqual(result.placeholder_slots, 1)
        self.assertEqual(result.render_rows[0]["Image Status"], "placeholder_bad_image")
        self.assertEqual(result.missing_image_rows[0]["Image Status"], "placeholder_bad_image")

    def test_image_containment_centers_portrait_and_landscape_sources(self) -> None:
        self.assertEqual(_contain_image_rect(100, 200, 0, 0, 200, 200), (50.0, 0.0, 100.0, 200.0))
        self.assertEqual(_contain_image_rect(200, 100, 0, 0, 200, 200), (0.0, 50.0, 200.0, 100.0))

    def test_qp_grid_layout_caps_card_width_and_centers_three_columns(self) -> None:
        placements = [
            self._placement(1, ((index // 3) + 1), column=((index % 3) + 1), image_status="missing")
            for index in range(15)
        ]

        metrics = _compute_page_dimensions_for_panel(placements)
        grid = _compute_grid_layout(metrics, 5, 3)
        max_card_w = _preferred_max_card_width(metrics.page_w, 3)

        self.assertLessEqual(grid.card_w, max_card_w)
        self.assertAlmostEqual(grid.grid_w, (3 * grid.card_w) + (2 * grid.col_gap))
        self.assertAlmostEqual(grid.grid_left, (metrics.page_w - grid.grid_w) / 2)
        self.assertGreater(grid.grid_left, metrics.margin)

    def _placement(
        self,
        panel: int,
        row: int,
        *,
        column: int = 1,
        product_name: str = "Test Product",
        denomination: str = "$10",
        product_upc: str = "111111111111",
        pack_upc: str = "999999999999",
        item_number: str = "123",
        cpp: int = 10,
        image_path: str = "",
        image_status: str = "missing",
    ) -> HolidayPlacement:
        return HolidayPlacement(
            panel_index=panel,
            panel_name=f"Panel {panel}",
            row=row,
            column=column,
            bbox=(0.0, 0.0, 40.0, 60.0),
            reference_text="",
            product_upc=product_upc,
            pack_upc=pack_upc,
            item_number=item_number,
            product_name=product_name,
            denomination=denomination,
            cpp=cpp,
            capacity=40,
            image_path=image_path,
            placement_status="resolved",
            image_status=image_status,
            image_resolution_source="local_product_upc" if image_status == "resolved" else "unresolved",
            warnings=[],
        )


if __name__ == "__main__":
    unittest.main()
