from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from app.sams_club.extract_access import _build_mapping, _normalize_upc_value, extract_master_pog_source
from app.sams_club.image_resolution import (
    SOURCE_LOCAL_ITEM_NUMBER,
    SOURCE_LOCAL_UPC,
    SOURCE_UNRESOLVED,
    SamsImageIndex,
    _calculate_upca_check_digit,
    _identifier_keys,
    _lookup_by_identifier,
    build_sams_local_image_index,
)
from app.sams_club.models import SamsPlanogram, SamsRow, SamsSidePage, SamsSlot
from app.sams_club.render_planogram import render_sams_planogram_pdf
from app.sams_club.service import build_sams_planogram_structure


class SamsExcelUpcTests(unittest.TestCase):
    def _write_workbook(
        self,
        path: Path,
        upc_header: str,
        upc_value: object,
        item_number: str = "12345",
        file_path: str | None = None,
    ) -> None:
        row = {
            "POG": "POG1",
            "Side": 1,
            "Row": 1,
            "Column": 1,
            "Item Number": item_number,
            upc_header: upc_value,
            "Retail": "9.99",
            "CPP": "1",
            "Description": "Test product",
        }
        if file_path is not None:
            row["file_path"] = file_path
        pd.DataFrame([row]).to_excel(path, index=False)

    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (10, 10), "white").save(path)

    def test_upc_header_named_upc_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook = Path(temp_dir) / "source.xlsx"
            self._write_workbook(workbook, "UPC", "87458605402")

            result = extract_master_pog_source(workbook)

        self.assertEqual(result.records[0]["upc"], "87458605402")
        self.assertEqual(result.column_mapping["upc"], "UPC")

    def test_upc_header_named_upc_11_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook = Path(temp_dir) / "source.xlsx"
            self._write_workbook(workbook, "UPC 11", "87458605402")

            result = extract_master_pog_source(workbook)

        self.assertEqual(result.records[0]["upc"], "87458605402")
        self.assertEqual(result.column_mapping["upc"], "UPC 11")

    def test_upc_alias_headers_are_canonicalized(self) -> None:
        for header in ("UPC11", "upc", "upc_11"):
            with self.subTest(header=header):
                mapping = _build_mapping(
                    ["POG", "Item Number", "Side", "Row", "Column", header],
                    {
                        "pog": ("pog",),
                        "item_number": ("item number",),
                        "side": ("side",),
                        "row": ("row",),
                        "column": ("column",),
                        "upc": ("upc", "upc 11", "upc11", "upc_11"),
                    },
                )
                self.assertEqual(mapping["upc"], header)

    def test_numeric_upc_values_normalize_to_digits(self) -> None:
        self.assertEqual(_normalize_upc_value(87458605402), "87458605402")
        self.assertEqual(_normalize_upc_value(87458605402.0), "87458605402")

    def test_text_upc_preserves_leading_zero(self) -> None:
        self.assertEqual(_normalize_upc_value("087458605402"), "087458605402")

    def test_upc11_generates_correct_upc12_check_digit(self) -> None:
        self.assertEqual(_calculate_upca_check_digit("19674217114"), "3")
        self.assertEqual(_calculate_upca_check_digit("19674217113"), "6")
        self.assertEqual(_calculate_upca_check_digit("19674208510"), "5")
        self.assertIn("196742171143", _identifier_keys("19674217114"))

    def test_upc12_generates_upc11_body(self) -> None:
        keys = _identifier_keys("196742171143")

        self.assertIn("196742171143", keys)
        self.assertIn("19674217114", keys)

    def test_invalid_upc_identifier_does_not_raise(self) -> None:
        self.assertEqual(_calculate_upca_check_digit("not-a-upc"), "")
        self.assertEqual(_identifier_keys("not-a-upc"), [])

    def test_missing_file_path_resolves_local_image_by_upc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_workbook(workbook, "UPC", "087458605402")
            self._write_image(image_dir / "087458605402.png")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=str(image_dir),
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.upc, "087458605402")
        self.assertEqual(slot.image_resolution_source, SOURCE_LOCAL_UPC)
        self.assertTrue(slot.resolved_image_path.endswith("087458605402.png"))
        self.assertEqual(result.debug["excel_records_read"], 1)
        self.assertEqual(result.debug["records_with_upc"], 1)
        self.assertEqual(result.debug["records_without_upc"], 0)
        self.assertEqual(result.debug["image_resolution"]["resolved_by_local_upc"], 1)
        self.assertEqual(result.debug["image_resolution"]["unresolved"], 0)

    def test_recursive_uppercase_prefixed_upc_with_trailing_decimal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_dir = root / "images"
            nested_dir = image_dir / "Gift Cards"
            nested_dir.mkdir(parents=True)
            self._write_workbook(workbook, "UPC", "087458605402.0")
            self._write_image(nested_dir / "100 0087458605402.JPG")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=f'"{image_dir}"',
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.upc, "087458605402")
        self.assertEqual(slot.image_resolution_source, SOURCE_LOCAL_UPC)
        self.assertTrue(slot.resolved_image_path.endswith("100 0087458605402.JPG"))
        self.assertTrue(result.debug["image_resolution"]["local_image_root_exists"])
        self.assertEqual(result.debug["image_resolution"]["indexed_image_count"], 1)

    def test_known_identifier_image_is_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            image_path = image_dir / "190199709997.jpg"
            self._write_image(image_path)

            image_index = build_sams_local_image_index(image_dir)
            generated_keys = _identifier_keys("190199709997")
            matched_path = _lookup_by_identifier("190199709997", image_index)
            matched_path_exists = Path(matched_path).is_file()

        self.assertIn("190199709997", generated_keys)
        self.assertEqual(matched_path, str(image_path))
        self.assertTrue(matched_path_exists)

    def test_11_digit_upc_matches_longer_filename_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            image_path = image_dir / "100 0019674217114.JPG"
            self._write_image(image_path)

            image_index = build_sams_local_image_index(image_dir)
            matched_path = _lookup_by_identifier("19674217114", image_index)

        self.assertEqual(matched_path, str(image_path))

    def test_upc11_matches_upc12_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            image_path = image_dir / "196742171143.jpg"
            self._write_image(image_path)

            image_index = build_sams_local_image_index(image_dir)
            matched_path = _lookup_by_identifier("19674217114", image_index)

        self.assertEqual(matched_path, str(image_path))

    def test_upc11_matches_14_digit_gtin_filename_with_upca_check_digit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            image_path = image_dir / "00196742171143.JPG"
            self._write_image(image_path)

            image_index = build_sams_local_image_index(image_dir)
            matched_path = _lookup_by_identifier("19674217114", image_index)

        self.assertEqual(matched_path, str(image_path))

    def test_nonexistent_generated_file_path_is_ignored_for_upc_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_dir = root / "images"
            nested_dir = image_dir / "Gift Cards"
            nested_dir.mkdir(parents=True)
            generated_path = str(image_dir / "19674217114.jpg")
            self._write_workbook(
                workbook,
                "UPC",
                "19674217114",
                file_path=generated_path,
            )
            self._write_image(nested_dir / "100 0019674217114.JPG")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=str(image_dir),
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        sample = result.debug["image_resolution"]["debug_sample"][0]
        self.assertEqual(slot.file_path, "")
        self.assertEqual(slot.image_resolution_source, SOURCE_LOCAL_UPC)
        self.assertTrue(slot.resolved_image_path.endswith("100 0019674217114.JPG"))
        self.assertEqual(sample["supplied_file_path"], generated_path)
        self.assertFalse(sample["supplied_file_path_exists"])
        self.assertIn("196742171143", sample["generated_upc_keys"])
        self.assertIn(sample["matched_index_key"], sample["generated_upc_keys"])
        self.assertEqual(sample["resolved_path"], slot.resolved_image_path)
        self.assertEqual(sample["resolution_method"], SOURCE_LOCAL_UPC)

    def test_fallback_resolution_by_item_number_after_upc_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_workbook(workbook, "UPC", "111111111111", item_number="98765")
            self._write_image(image_dir / "98765.png")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=str(image_dir),
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.image_resolution_source, SOURCE_LOCAL_ITEM_NUMBER)
        self.assertTrue(slot.resolved_image_path.endswith("98765.png"))
        self.assertEqual(result.debug["records_with_upc"], 1)
        self.assertEqual(result.debug["records_without_upc"], 0)
        self.assertEqual(result.debug["records_with_item_number"], 1)
        self.assertEqual(result.debug["image_resolution"]["resolved_by_local_item_number"], 1)

    def test_missing_local_root_records_unresolved_without_file_path_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            missing_root = root / "missing-images"
            self._write_workbook(workbook, "UPC", "087458605402")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=str(missing_root),
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.image_resolution_source, SOURCE_UNRESOLVED)
        self.assertFalse(result.debug["image_resolution"]["local_image_root_exists"])
        self.assertEqual(result.debug["image_resolution"]["indexed_image_count"], 0)
        self.assertEqual(result.debug["image_resolution"]["unresolved"], 1)

        pdf_result = render_sams_planogram_pdf(result.planogram)
        warning_text = "\n".join(pdf_result.warnings)
        self.assertIn(
            "no local or ZIP match for UPC=087458605402, Item Number=12345",
            warning_text,
        )
        self.assertNotIn("missing file_path", warning_text)

    def test_ui_local_image_root_reaches_service_index_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            supplied_root = root / "images"
            supplied_root.mkdir()
            self._write_workbook(workbook, "UPC", "190199709997")
            captured_roots: list[object] = []

            def fake_build_sams_local_image_index(image_root: object) -> SamsImageIndex:
                captured_roots.append(image_root)
                return SamsImageIndex(root_dir=str(supplied_root))

            with patch(
                "app.sams_club.service.build_sams_local_image_index",
                fake_build_sams_local_image_index,
            ):
                build_sams_planogram_structure(
                    workbook,
                    selected_pog="POG1",
                    local_image_root=str(supplied_root),
                )

        self.assertEqual(captured_roots, [str(supplied_root)])

    def test_zero_indexed_images_produces_useful_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            empty_image_dir = root / "images"
            empty_image_dir.mkdir()
            self._write_workbook(workbook, "UPC", "190199709997")

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root=str(empty_image_dir),
            )

        self.assertEqual(result.debug["image_resolution"]["indexed_image_count"], 0)
        self.assertTrue(
            any("No images were indexed from" in warning for warning in result.warnings)
        )

    def test_renderer_uses_resolved_image_path_before_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "resolved.png"
            self._write_image(image_path)
            slot = SamsSlot(
                pog="POG1",
                side=1,
                row=1,
                column=1,
                item_number="12345",
                upc="190199709997",
                file_path=str(Path(temp_dir) / "missing.png"),
                resolved_image_path=str(image_path),
            )
            planogram = SamsPlanogram(
                pog="POG1",
                side_pages=[
                    SamsSidePage(
                        pog="POG1",
                        side=1,
                        column_limit=1,
                        rows=[
                            SamsRow(
                                side=1,
                                row_number=1,
                                column_limit=1,
                                populated_column_count=1,
                                slots=[slot],
                            )
                        ],
                    )
                ],
            )

            result = render_sams_planogram_pdf(planogram)

        self.assertEqual(result.missing_image_slots, 0)
        self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
