from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.holiday_planograms.extract_assortment import load_holiday_assortment
from app.holiday_planograms.extract_nre import NRE_CONFIGURATIONS, load_bos_nre_reference
from app.holiday_planograms.geometry_nre import (
    SLOT_AMBIGUOUS,
    SLOT_FILLER,
    SLOT_MERCHANDISE,
    classify_nre_slot,
    detect_nre_geometry,
)
from app.holiday_planograms.matching_nre import match_nre_merchandise_slots
from app.holiday_planograms.models import HolidayProduct, LayoutSlot, ReferenceImage
from app.holiday_planograms.render_nre import _template_for_configuration, render_holiday_nre
from app.holiday_planograms.service import analyze_holiday_nre_configurations, build_holiday_nre_qa


BOS_WORKBOOK = Path.home() / "Downloads" / "BOS Holiday2026_Planograms.xlsx"
D5_WORKBOOK = Path.home() / "Downloads" / "D5_Holiday_Assortment_0505.xlsx"


def _require_supplied_workbooks() -> None:
    if not BOS_WORKBOOK.exists() or not D5_WORKBOOK.exists():
        raise unittest.SkipTest("Supplied Holiday 2026 workbooks are not available.")


class HolidayNreTests(unittest.TestCase):
    def _product(
        self,
        name: str,
        *,
        denom: str = "",
        item: str = "1",
        upc: str = "111111111111",
        facings: int = 1,
        cpp: int = 20,
    ) -> HolidayProduct:
        return HolidayProduct(
            product_name=name,
            denomination=denom,
            product_upc=upc,
            pack_upc="",
            item_number=item,
            nre_facings=facings,
            qtr_facings=0,
            total_pegs=None,
            cpp=cpp,
            source_row=2,
            raw={},
        )

    def _merch_slot(self, index: int, *, slot_type: str = SLOT_MERCHANDISE) -> LayoutSlot:
        return LayoutSlot(
            panel_index=1,
            panel_name="panel_1",
            row=(index // 8) + 1,
            column=(index % 8) + 1,
            bbox=(float(index), 0.0, float(index + 1), 1.0),
            center_x=float(index),
            center_y=float(index // 8),
            width=1.0,
            height=1.0,
            slot_type=slot_type,
        )

    def test_d5_nre_filter_supplied_workbook_counts(self) -> None:
        _require_supplied_workbooks()

        sheet, products, warnings = load_holiday_assortment(str(D5_WORKBOOK))
        nre_products = [product for product in products if product.nre_facings > 0]

        self.assertEqual(sheet, "AssortmentList_D5_Holidays")
        self.assertEqual(warnings, [])
        self.assertEqual(len(nre_products), 27)
        self.assertEqual(sum(product.nre_facings for product in nre_products), 40)

    def test_nre_reference_selection_is_configuration_specific(self) -> None:
        _require_supplied_workbooks()

        references = {}
        for configuration in NRE_CONFIGURATIONS:
            sheet, reference, warnings = load_bos_nre_reference(str(BOS_WORKBOOK), configuration)
            self.assertEqual(sheet, "2026 NRE HOLIDAY")
            self.assertEqual(warnings, [])
            self.assertIsNotNone(reference)
            references[configuration] = reference

        self.assertEqual(references["4x78"].image.name, "worksheet_image_2")
        self.assertEqual(references["3x78"].image.name, "worksheet_image_1")
        self.assertEqual(references["3x60"].image.name, "worksheet_image_3")
        self.assertEqual(references["4x60"].image.name, "worksheet_image_4")
        self.assertEqual(len({reference.image.name for reference in references.values()}), 4)

    def test_analyze_all_nre_configurations_supplied_workbook(self) -> None:
        _require_supplied_workbooks()

        rows = analyze_holiday_nre_configurations(str(BOS_WORKBOOK), str(D5_WORKBOOK))
        by_config = {row.configuration: row for row in rows}

        self.assertEqual(set(by_config), set(NRE_CONFIGURATIONS))
        self.assertEqual(by_config["4x78"].physical_detected, 60)
        self.assertEqual(by_config["3x78"].physical_detected, 48)
        self.assertEqual(by_config["3x60"].physical_detected, 44)
        self.assertEqual(by_config["4x60"].physical_detected, 50)
        self.assertTrue(all(row.merchandise_slots == 40 for row in rows))
        self.assertEqual(by_config["4x78"].filler_slots, 20)
        self.assertEqual(by_config["3x78"].filler_slots, 8)
        self.assertEqual(by_config["3x60"].filler_slots, 4)
        self.assertEqual(by_config["4x60"].filler_slots, 10)
        self.assertTrue(all(row.ambiguous_slots == 0 for row in rows))
        self.assertTrue(all(row.expected_nre_facings == 40 for row in rows))
        self.assertTrue(all(row.difference == 0 for row in rows))

    def test_nre_slot_crop_classifier_distinguishes_filler_merchandise_and_ambiguous(self) -> None:
        filler = Image.new("RGB", (48, 64), (70, 220, 100))
        draw = ImageDraw.Draw(filler)
        draw.rectangle((0, 0, 47, 63), outline="black", width=1)

        merchandise = Image.new("RGB", (48, 64), "white")
        draw = ImageDraw.Draw(merchandise)
        draw.rectangle((0, 0, 47, 63), outline=(200, 0, 0), width=1)
        draw.text((8, 18), "CARD", fill="black")

        ambiguous = Image.new("RGB", (48, 64), "white")
        draw = ImageDraw.Draw(ambiguous)
        draw.rectangle((12, 12, 29, 29), fill=(90, 190, 95))

        self.assertEqual(classify_nre_slot(filler)[0], SLOT_FILLER)
        self.assertEqual(classify_nre_slot(merchandise)[0], SLOT_MERCHANDISE)
        self.assertEqual(classify_nre_slot(ambiguous)[0], SLOT_AMBIGUOUS)

    def test_nre_matching_resolves_exact_brand_and_denomination(self) -> None:
        products = [self._product("Roblox Gift Card $25", denom="25", item="roblox25", facings=2)]
        slots = [self._merch_slot(index) for index in range(17, 24)]

        matches = match_nre_merchandise_slots(slots, products)
        resolved = [match for match in matches if match.product is not None]

        self.assertEqual(len(resolved), 2)
        self.assertTrue(all(match.raw_reference_text == "ROBLOX $25" for match in resolved))
        self.assertTrue(all(match.product and match.product.item_number == "roblox25" for match in resolved))

    def test_nre_matching_keeps_vgc_distinct_from_fixed_denomination(self) -> None:
        products = [
            self._product("Roblox Gift Card VGC ($10-$250)", denom="$10 - $250", item="robloxvgc", facings=2),
            self._product("Roblox Gift Card $25", denom="25", item="roblox25", facings=2),
        ]
        slots = [self._merch_slot(index) for index in range(16, 24)]

        matches = match_nre_merchandise_slots(slots, products)
        by_text = {match.raw_reference_text: match.product.item_number for match in matches if match.product}

        self.assertEqual(by_text["ROBLOX VGC"], "robloxvgc")
        self.assertEqual(by_text["ROBLOX $25"], "roblox25")

    def test_nre_matching_keeps_gwp_distinct_from_ordinary_playstation(self) -> None:
        products = [
            self._product("Sony PlayStation Store GWP $50", denom="50", item="psgwp", facings=2),
            self._product("Sony PlayStation Store Gift Card VGC ($10-$250)", denom="$10 - $250", item="psvgc", facings=2),
        ]
        slots = [self._merch_slot(index) for index in range(8, 16)]

        matches = match_nre_merchandise_slots(slots, products)
        resolved = [match for match in matches if match.product]

        self.assertEqual([match.product.item_number for match in resolved[:2]], ["psgwp", "psgwp"])
        self.assertEqual([match.product.item_number for match in resolved[2:4]], ["psvgc", "psvgc"])

    def test_nre_matching_does_not_exceed_facing_constraints(self) -> None:
        products = [self._product("Meta Quest Gift Card VGC ($15-$100)", denom="$15 - $100", item="meta", facings=2)]
        slots = [self._merch_slot(index) for index in range(24, 28)]

        matches = match_nre_merchandise_slots(slots, products)

        self.assertEqual(sum(1 for match in matches if match.product), 2)
        self.assertEqual(sum(1 for match in matches if match.product is None), 2)

    def test_nre_matching_ignores_filler_slots(self) -> None:
        products = [self._product("Roblox Gift Card $25", denom="25", item="roblox25", facings=2)]
        slots = [self._merch_slot(18), self._merch_slot(19), self._merch_slot(20, slot_type=SLOT_FILLER)]

        matches = match_nre_merchandise_slots(slots, products)

        self.assertEqual(len(matches), 2)
        self.assertTrue(all(match.slot.slot_type == SLOT_MERCHANDISE for match in matches))

    def test_nre_matching_leaves_ambiguous_text_unresolved(self) -> None:
        products = [
            self._product("Meta Quest Gift Card VGC ($15-$100)", denom="$15 - $100", item="meta1", facings=1),
            self._product("Meta Quest Bonus Gift Card VGC ($15-$100)", denom="$15 - $100", item="meta2", facings=1),
        ]
        slots = [self._merch_slot(24)]

        matches = match_nre_merchandise_slots(slots, products)

        self.assertEqual(matches[0].status, "ambiguous_reference_text")
        self.assertIsNone(matches[0].product)

    def test_nre_geometry_synthetic_grid_detects_rows_columns_and_missing_cell(self) -> None:
        image = Image.new("RGB", (260, 240), "white")
        draw = ImageDraw.Draw(image)
        for row in range(3):
            for col in range(4):
                if row == 2 and col == 3:
                    continue
                x0 = 24 + col * 36
                y0 = 24 + row * 64
                draw.rectangle((x0, y0, x0 + 34, y0 + 52), outline="black", width=1)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = detect_nre_geometry(
            ReferenceImage(
                name="synthetic_nre",
                bytes_data=buffer.getvalue(),
                width=image.width,
                height=image.height,
                anchor_row=0,
                anchor_col=0,
            )
        )

        self.assertEqual(len(result.slots), 11)
        counts = {
            slot_type: sum(1 for slot in result.slots if slot.slot_type == slot_type)
            for slot_type in (SLOT_MERCHANDISE, SLOT_FILLER, SLOT_AMBIGUOUS)
        }
        self.assertEqual(sum(counts.values()), len(result.slots))
        self.assertEqual(len(result.panels), 1)
        self.assertEqual(result.panels[0].rows, 3)
        self.assertEqual(result.panels[0].columns, 4)
        self.assertIn("missing/irregular cells", result.warnings[0])
        self.assertGreater(len(result.debug_image_bytes), 0)

    def test_nre_placement_qa_classifies_merchandise_and_filler_without_guessing(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "3x60", local_image_folder="")

        self.assertEqual(result.errors, [])
        self.assertEqual(result.summary["d5_nre_products"], 27)
        self.assertEqual(result.summary["d5_nre_facings"], 40)
        self.assertEqual(result.summary["active_slots"], 40)
        self.assertEqual(result.summary["physical_cells_detected"], 44)
        self.assertEqual(result.summary["theoretical_grid_positions"], 48)
        self.assertEqual(result.summary["merchandise_slots"], 40)
        self.assertEqual(result.summary["filler_slots"], 4)
        self.assertEqual(result.summary["ambiguous_slots"], 0)
        self.assertEqual(result.summary["merchandise_difference"], 0)
        self.assertEqual(result.summary["resolved_placements"], 40)
        self.assertEqual(result.summary["unresolved_placements"], 0)
        self.assertEqual(result.summary["facing_mismatches"], 0)
        self.assertEqual(result.summary["non_product_placements"], 4)
        statuses = {row["Placement Status"] for row in result.placement_qa_rows}
        self.assertIn("resolved_reference_text", statuses)
        self.assertIn("non_product", statuses)
        filler_row = next(row for row in result.placement_qa_rows if row["Placement Status"] == "non_product")
        self.assertEqual(filler_row["Slot Type"], SLOT_FILLER)
        self.assertEqual(filler_row["Image Status"], "not_applicable")

    def test_nre_product_image_fallback_missing_folder_is_non_blocking(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "4x60", local_image_folder="Z:/missing/nre/images")

        self.assertEqual(result.errors, [])
        self.assertFalse(result.summary["image_folder_exists"])
        self.assertEqual(result.summary["unique_products_with_images"], 0)
        self.assertTrue(result.product_qa_rows)
        self.assertEqual(result.product_qa_rows[0]["Image Match Status"], "missing")

    def test_nre_product_image_resolution_reuses_holiday_identity_hierarchy(self) -> None:
        _require_supplied_workbooks()

        _sheet, products, _warnings = load_holiday_assortment(str(D5_WORKBOOK))
        product = next(item for item in products if item.nre_facings > 0)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / f"{product.product_upc}.png"
            Image.new("RGB", (50, 70), "green").save(image_path)

            result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "4x78", local_image_folder=temp_dir)

        row = next(item for item in result.product_qa_rows if item["Product UPC"] == product.product_upc)
        self.assertEqual(row["Image Match Status"], "resolved")
        self.assertEqual(row["Image Resolution Source"], "local_product_upc")

    def test_nre_renderer_outputs_one_page_for_selected_configuration(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "4x78", local_image_folder="")
        render_result = render_holiday_nre(result.placements, result.configuration)

        self.assertGreater(len(render_result.pdf_bytes), 0)
        self.assertGreater(len(render_result.preview_png_bytes), 0)
        self.assertEqual(render_result.panels_rendered, 1)
        self.assertEqual(render_result.slots_rendered, 60)
        self.assertEqual(render_result.skipped_slots, 20)
        self.assertEqual(render_result.placeholder_slots, 40)
        import fitz

        with fitz.open(stream=render_result.pdf_bytes, filetype="pdf") as doc:
            self.assertEqual(doc.page_count, 1)

    def test_nre_renderer_templates_define_expected_heading_order_for_each_config(self) -> None:
        expected = [
            "ZIFT",
            "NINTENDO",
            "PLAYSTATION",
            "XBOX",
            "ROBLOX",
            "FORTNITE",
            "META",
            "RAZER GOLD",
            "POKEMON GO",
            "GOOGLE PLAY",
        ]

        for configuration in NRE_CONFIGURATIONS:
            template = _template_for_configuration(configuration)
            self.assertEqual([heading.group_name for heading in template.headings], expected)

    def test_nre_renderer_template_slots_do_not_overlap(self) -> None:
        for configuration in NRE_CONFIGURATIONS:
            template = _template_for_configuration(configuration)
            slots = template.slots
            for left_index, left in enumerate(slots):
                lx1, ly1, lx2, ly2 = left.x, left.y, left.x + left.w, left.y + left.h
                for right in slots[left_index + 1 :]:
                    rx1, ry1, rx2, ry2 = right.x, right.y, right.x + right.w, right.y + right.h
                    overlaps = lx1 < rx2 and lx2 > rx1 and ly1 < ry2 and ly2 > ry1
                    self.assertFalse(overlaps, f"{configuration} template slots overlap: {left} vs {right}")

    def test_nre_renderer_assigns_merchandise_cards_to_template_groups(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "4x78", local_image_folder="")
        render_result = render_holiday_nre(result.placements, result.configuration)

        group_rows = [row for row in render_result.render_rows if row["Slot Type"] == SLOT_MERCHANDISE]
        by_group = {
            group: [row for row in group_rows if row["Group"] == group]
            for group in ["ZIFT", "NINTENDO", "PLAYSTATION", "XBOX", "ROBLOX", "FORTNITE", "META", "RAZER GOLD", "POKEMON GO", "GOOGLE PLAY"]
        }
        self.assertTrue(all(len(rows) == 4 for rows in by_group.values()))
        self.assertTrue(all("PLAYSTATION" in row["Reference Text"] for row in by_group["PLAYSTATION"]))
        self.assertTrue(all("XBOX" in row["Reference Text"] for row in by_group["XBOX"]))
        self.assertTrue(all("ROBLOX" in row["Reference Text"] for row in by_group["ROBLOX"]))
        self.assertTrue(all("FORTNITE" in row["Reference Text"] for row in by_group["FORTNITE"]))

    def test_nre_renderer_merchandise_cards_include_required_metadata(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "3x78", local_image_folder="")
        render_result = render_holiday_nre(result.placements, result.configuration)

        merchandise_row = next(row for row in render_result.render_rows if row["Slot Type"] == SLOT_MERCHANDISE)
        self.assertTrue(str(merchandise_row["Price"]))
        self.assertTrue(str(merchandise_row["CPP"]).startswith("CPP "))
        self.assertTrue(str(merchandise_row["UPC"]).isdigit())
        self.assertTrue(str(merchandise_row["ITEM"]).isdigit())
        self.assertTrue(str(merchandise_row["Product Name"]))
        self.assertEqual(merchandise_row["Metadata Fields"], "Price; CPP; UPC; ITEM; Product Name")

    def test_nre_renderer_filler_blocks_preserve_footprint_without_metadata(self) -> None:
        _require_supplied_workbooks()

        result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "3x60", local_image_folder="")
        render_result = render_holiday_nre(result.placements, result.configuration)

        filler_rows = [row for row in render_result.render_rows if row["Slot Type"] == SLOT_FILLER]
        self.assertEqual(len(filler_rows), 4)
        for row in filler_rows:
            self.assertEqual(row["Content Type"], "filler")
            self.assertEqual(row["Product Name"], "")
            self.assertEqual(row["UPC"], "")
            self.assertEqual(row["ITEM"], "")
            self.assertEqual(row["CPP"], "")
            self.assertEqual(row["Image Status"], "not_applicable")
            _x, _y, width, height = [float(value) for value in str(row["Card Rect"]).split(",")]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)

    def test_nre_renderer_resolved_image_is_contained_and_preserves_aspect_ratio(self) -> None:
        _require_supplied_workbooks()

        _sheet, products, _warnings = load_holiday_assortment(str(D5_WORKBOOK))
        product = next(item for item in products if item.nre_facings > 0)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / f"{product.product_upc}.png"
            Image.new("RGB", (40, 80), "blue").save(image_path)
            result = build_holiday_nre_qa(str(BOS_WORKBOOK), str(D5_WORKBOOK), "4x60", local_image_folder=temp_dir)
            render_result = render_holiday_nre(result.placements, result.configuration)

        image_rows = [row for row in render_result.render_rows if row["Content Type"] == "image"]
        self.assertEqual(len(image_rows), product.nre_facings)
        row = image_rows[0]
        box_x, box_y, box_w, box_h = [float(value) for value in str(row["Image Box Rect"]).split(",")]
        draw_x, draw_y, draw_w, draw_h = [float(value) for value in str(row["Image Draw Rect"]).split(",")]
        self.assertGreaterEqual(draw_x, box_x)
        self.assertGreaterEqual(draw_y, box_y)
        self.assertLessEqual(draw_x + draw_w, box_x + box_w + 0.01)
        self.assertLessEqual(draw_y + draw_h, box_y + box_h + 0.01)
        self.assertAlmostEqual(draw_w / draw_h, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
