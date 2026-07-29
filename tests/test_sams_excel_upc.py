from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from app.sams_club.extract_access import _build_mapping, _normalize_upc_value, extract_master_pog_source
from app.sams_club.image_resolution import SOURCE_LOCAL_ITEM_NUMBER, SOURCE_LOCAL_UPC, SOURCE_UNRESOLVED
from app.sams_club.render_planogram import render_sams_planogram_pdf
from app.sams_club.service import build_sams_planogram_structure


class SamsExcelUpcTests(unittest.TestCase):
    def _write_workbook(self, path: Path, upc_header: str, upc_value: object, item_number: str = "12345") -> None:
        pd.DataFrame(
            [
                {
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
            ]
        ).to_excel(path, index=False)

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


if __name__ == "__main__":
    unittest.main()
