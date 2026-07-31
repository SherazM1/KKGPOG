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
    SOURCE_MANUAL_UPC,
    SOURCE_OCR_FILENAME_UPC,
    SOURCE_OCR_UPC_VARIANT,
    SOURCE_UNRESOLVED,
    SamsImageIndex,
    SamsManualImageMappingIndex,
    _calculate_upca_check_digit,
    _identifier_keys,
    _lookup_by_identifier,
    build_sams_local_image_index,
    load_sams_manual_image_mappings,
    resolve_sams_image_path,
)
from app.sams_club.models import SamsPlanogram, SamsRow, SamsSidePage, SamsSlot
from app.sams_club.ocr_image_resolution import (
    append_manual_image_mapping,
    load_sams_ocr_catalog,
    preview_image_status,
    remove_manual_image_mapping,
    score_ocr_candidate,
    upc_comparison_keys,
)
from app.sams_club.render_planogram import render_sams_planogram_pdf
from app.sams_club.service import build_sams_planogram_structure
from scripts.catalog_sams_images import catalog_image
from scripts.match_sams_unresolved import build_candidate_matches


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

    def _write_corrupt_file(self, path: Path) -> None:
        path.write_bytes(b"not an image")

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

    def test_check_digit_and_12_digit_upc_headers_are_preserved(self) -> None:
        mapping = _build_mapping(
            ["POG", "Item Number", "Side", "Row", "Column", "Check Digit", "12 Digit UPC"],
            {
                "pog": ("pog",),
                "item_number": ("item number",),
                "side": ("side",),
                "row": ("row",),
                "column": ("column",),
                "check_digit": ("check digit", "check_digit"),
                "upc12": ("12 digit upc", "upc12"),
            },
        )

        self.assertEqual(mapping["check_digit"], "Check Digit")
        self.assertEqual(mapping["upc12"], "12 Digit UPC")

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

    def test_catalog_handles_uppercase_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "PlayStation 50.JPG"
            self._write_image(image_path)
            cached = {
                "file_path": str(image_path),
                "filename": image_path.name,
                "filename_upc": "50",
                "file_size": str(image_path.stat().st_size),
                "modified_time": str(image_path.stat().st_mtime),
                "width": "10",
                "height": "10",
                "detected_text": "PlayStation $50",
                "normalized_text": "playstation $50",
                "detected_brand": "playstation",
                "detected_denomination": "$50",
                "detected_pack_quantity": "",
                "catalog_status": "ok",
                "catalog_error": "",
            }

            row, _used_cache = catalog_image(image_path, cached)

        self.assertEqual(row["catalog_status"], "ok")
        self.assertEqual(row["filename"], "PlayStation 50.JPG")

    def test_catalog_handles_corrupt_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "broken.JPG"
            self._write_corrupt_file(image_path)

            row, _used_cache = catalog_image(image_path)

        self.assertEqual(row["catalog_status"], "error")
        self.assertTrue(row["catalog_error"])

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

    def test_playstation_50_ranks_above_other_brands(self) -> None:
        unresolved_rows = [
            {
                "status": "unresolved",
                "UPC": "11111111111",
                "Item Number": "123",
                "Brand": "PSN",
                "Description": "PlayStation Store $50 Gift Card",
            }
        ]
        catalog_rows = [
            {
                "file_path": "steam.jpg",
                "filename": "steam.jpg",
                "filename_upc": "",
                "detected_text": "Steam $50",
                "normalized_text": "steam $50",
                "detected_brand": "steam",
                "detected_denomination": "$50",
                "detected_pack_quantity": "",
            },
            {
                "file_path": "ps.jpg",
                "filename": "ps.jpg",
                "filename_upc": "",
                "detected_text": "PlayStation $50",
                "normalized_text": "playstation $50",
                "detected_brand": "playstation",
                "detected_denomination": "$50",
                "detected_pack_quantity": "",
            },
        ]

        rows = build_candidate_matches(unresolved_rows, catalog_rows)

        self.assertEqual(rows[0]["candidate_file_path"], "ps.jpg")
        self.assertGreater(rows[0]["total_score"], rows[1]["total_score"])

    def test_conflicting_denominations_are_penalized(self) -> None:
        unresolved_rows = [
            {
                "status": "unresolved",
                "UPC": "11111111111",
                "Item Number": "123",
                "Brand": "PlayStation",
                "Description": "PlayStation Store $50 Gift Card",
            }
        ]
        catalog_rows = [
            {
                "file_path": "ps25.jpg",
                "filename": "ps25.jpg",
                "filename_upc": "",
                "detected_text": "PlayStation $25",
                "normalized_text": "playstation $25",
                "detected_brand": "playstation",
                "detected_denomination": "$25",
                "detected_pack_quantity": "",
            }
        ]

        rows = build_candidate_matches(unresolved_rows, catalog_rows)

        self.assertLess(rows[0]["denomination_score"], 0)

    def test_repeated_unresolved_positions_are_deduplicated_by_matcher(self) -> None:
        unresolved_rows = [
            {"status": "unresolved", "UPC": "111", "Item Number": "222", "Description": "PlayStation $50"},
            {"status": "unresolved", "UPC": "111", "Item Number": "222", "Description": "PlayStation $50"},
        ]
        catalog_rows = [
            {
                "file_path": "ps.jpg",
                "filename": "ps.jpg",
                "filename_upc": "",
                "detected_text": "PlayStation $50",
                "normalized_text": "playstation $50",
                "detected_brand": "playstation",
                "detected_denomination": "$50",
                "detected_pack_quantity": "",
            }
        ]

        rows = build_candidate_matches(unresolved_rows[:1], catalog_rows)

        self.assertEqual(len(rows), 1)

    def test_approved_mapping_resolves_all_repeated_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_path = root / "approved.png"
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {"POG": "POG1", "Side": 1, "Row": 1, "Column": 1, "Item Number": "12345", "UPC": "19674217114"},
                    {"POG": "POG1", "Side": 1, "Row": 1, "Column": 2, "Item Number": "12345", "UPC": "19674217114"},
                ]
            ).to_excel(workbook, index=False)
            manual_index = SamsManualImageMappingIndex(
                by_upc={"19674217114": str(image_path)},
                approved_count=1,
            )

            with patch(
                "app.sams_club.service.load_sams_manual_image_mappings",
                lambda: manual_index,
            ):
                result = build_sams_planogram_structure(
                    workbook,
                    selected_pog="POG1",
                    local_image_root="",
                )

        slots = result.planogram.side_pages[0].rows[0].slots
        self.assertEqual([slot.image_resolution_source for slot in slots], [SOURCE_MANUAL_UPC, SOURCE_MANUAL_UPC])
        self.assertEqual([slot.resolved_image_path for slot in slots], [str(image_path), str(image_path)])

    def test_invalid_manual_mapping_falls_through_normally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "manual.csv"
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            local_image = image_dir / "190199709997.png"
            self._write_image(local_image)
            mapping_path.write_text(
                "UPC,Item Number,file_path,approved,source,notes\n"
                f"190199709997,,{Path(temp_dir) / 'missing.png'},true,test,\n",
                encoding="utf-8",
            )
            manual_index = load_sams_manual_image_mappings(mapping_path)
            local_index = build_sams_local_image_index(image_dir)

            resolution = resolve_sams_image_path(
                file_path="",
                upc="190199709997",
                item_number="",
                local_index=local_index,
                manual_index=manual_index,
            )

        self.assertEqual(resolution.source, SOURCE_LOCAL_UPC)
        self.assertEqual(resolution.resolved_path, str(local_image))

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

    def test_ocr_upc_comparison_keys_include_upc11_check_digit(self) -> None:
        keys = upc_comparison_keys("19674217114")

        self.assertIn("19674217114", keys)
        self.assertIn("196742171143", keys)

    def test_ocr_catalog_exact_12_digit_upc_match_resolves_before_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            catalog_path = root / "image_catalog.csv"
            image_path = root / "196742171143.jpg"
            self._write_workbook(workbook, "UPC", "196742171143")
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {
                        "file_path": str(image_path),
                        "filename": image_path.name,
                        "filename_upc": "196742171143",
                        "detected_text": "Visa $150",
                        "normalized_text": "VISA $150",
                        "detected_brand": "Visa",
                        "detected_denomination": "$150",
                        "detected_pack_quantity": "",
                        "catalog_status": "ok",
                        "catalog_error": "",
                    }
                ]
            ).to_csv(catalog_path, index=False)

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root="",
                ocr_catalog_file=catalog_path,
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.image_resolution_source, SOURCE_OCR_FILENAME_UPC)
        self.assertEqual(slot.resolved_image_path, str(image_path))
        self.assertEqual(result.debug["image_resolution"]["resolved_by_ocr_filename_upc"], 1)
        self.assertEqual(result.debug["image_resolution"]["ocr_catalog_candidates_available"], 0)

    def test_ocr_catalog_upc11_matches_upc12_filename_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            catalog_path = root / "image_catalog.csv"
            image_path = root / "196742171143.jpg"
            self._write_workbook(workbook, "UPC", "19674217114")
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {
                        "file_path": str(image_path),
                        "filename": image_path.name,
                        "filename_upc": "196742171143",
                        "detected_text": "",
                        "normalized_text": "",
                        "detected_brand": "",
                        "detected_denomination": "",
                        "detected_pack_quantity": "",
                        "catalog_status": "ok",
                        "catalog_error": "",
                    }
                ]
            ).to_csv(catalog_path, index=False)

            result = build_sams_planogram_structure(
                workbook,
                selected_pog="POG1",
                local_image_root="",
                ocr_catalog_file=catalog_path,
            )

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.image_resolution_source, SOURCE_OCR_UPC_VARIANT)
        self.assertEqual(slot.resolved_image_path, str(image_path))

    def test_ocr_catalog_handles_missing_columns_blank_ocr_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "valid.JPG"
            missing_path = root / "missing.JPG"
            catalog_path = root / "image_catalog.csv"
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {"file_path": str(image_path), "filename": "valid.JPG"},
                    {"file_path": str(missing_path), "filename": "missing.JPG"},
                    {"file_path": "", "filename": "blank.JPG"},
                ]
            ).to_csv(catalog_path, index=False)

            catalog = load_sams_ocr_catalog(catalog_path)

        self.assertTrue(catalog.loaded)
        self.assertEqual(catalog.rows_read, 3)
        self.assertEqual(catalog.valid_rows, 1)
        self.assertEqual(catalog.invalid_rows, 2)
        self.assertEqual(catalog.missing_file_rows, 1)

    def test_ocr_catalog_ignores_failed_rows_without_useful_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "failed.JPG"
            catalog_path = root / "image_catalog.csv"
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {
                        "file_path": str(image_path),
                        "filename": image_path.name,
                        "catalog_status": "error",
                        "catalog_error": "ocr failed",
                    }
                ]
            ).to_csv(catalog_path, index=False)

            catalog = load_sams_ocr_catalog(catalog_path)

        self.assertEqual(catalog.valid_rows, 0)
        self.assertEqual(catalog.failed_rows_ignored, 1)

    def test_ocr_scoring_brand_denomination_and_pack_penalties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good_image = root / "playstation.JPG"
            bad_image = root / "steam.JPG"
            catalog_path = root / "image_catalog.csv"
            self._write_image(good_image)
            self._write_image(bad_image)
            pd.DataFrame(
                [
                    {
                        "file_path": str(good_image),
                        "filename": good_image.name,
                        "detected_text": "PlayStation $50 2 x $25",
                        "detected_brand": "PlayStation",
                        "detected_denomination": "2 x $25",
                        "detected_pack_quantity": "2",
                        "catalog_status": "ok",
                    },
                    {
                        "file_path": str(bad_image),
                        "filename": bad_image.name,
                        "detected_text": "Steam $25 4 x $25",
                        "detected_brand": "Steam",
                        "detected_denomination": "$25",
                        "detected_pack_quantity": "4",
                        "catalog_status": "ok",
                    },
                ]
            ).to_csv(catalog_path, index=False)
            catalog = load_sams_ocr_catalog(catalog_path)
            product = {
                "upc": "111",
                "item_number": "222",
                "brand": "PlayStation",
                "description": "PlayStation Gift Card 2 x $25",
            }

            good_score = score_ocr_candidate(product, catalog.entries[0])
            bad_score = score_ocr_candidate(product, catalog.entries[1])

        self.assertEqual(good_score.confidence_label, "High")
        self.assertLess(bad_score.denomination_score, 0)
        self.assertLess(bad_score.pack_score, 0)
        self.assertGreater(good_score.total_score, bad_score.total_score)

    def test_saved_mapping_reuse_deleted_mapping_manual_browse_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_path = root / "approved.png"
            mapping_path = root / "manual.csv"
            self._write_workbook(workbook, "UPC", "19674217114")
            self._write_image(image_path)
            append_manual_image_mapping(
                mapping_path,
                upc="19674217114",
                original_upc="19674217114",
                item_number="12345",
                file_path=str(image_path),
                source="MANUAL_BROWSE",
            )
            manual_index = load_sams_manual_image_mappings(mapping_path)

            with patch(
                "app.sams_club.service.load_sams_manual_image_mappings",
                lambda: manual_index,
            ):
                result = build_sams_planogram_structure(
                    workbook,
                    selected_pog="POG1",
                    local_image_root="",
                )
            removed = remove_manual_image_mapping(mapping_path, "19674217114", "12345")
            image_path.unlink()
            deleted_index = load_sams_manual_image_mappings(mapping_path)

        slot = result.planogram.side_pages[0].rows[0].slots[0]
        self.assertEqual(slot.image_resolution_source, SOURCE_MANUAL_UPC)
        self.assertEqual(slot.resolved_image_path, str(image_path))
        self.assertEqual(removed, 1)
        self.assertEqual(deleted_index.approved_count, 0)

    def test_preview_status_handles_corrupt_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt_path = Path(temp_dir) / "bad.jpg"
            self._write_corrupt_file(corrupt_path)

            ok, message = preview_image_status(str(corrupt_path))

        self.assertFalse(ok)
        self.assertIn("Image preview unavailable", message)

    def test_pdf_generation_after_approved_fallback_preserves_cpp_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workbook = root / "source.xlsx"
            image_path = root / "approved.png"
            self._write_image(image_path)
            pd.DataFrame(
                [
                    {
                        "POG": "POG1",
                        "Side": 1,
                        "Row": 1,
                        "Column": 2,
                        "Item Number": "222",
                        "UPC": "22222222222",
                        "Retail": "9.99",
                        "CPP": "6",
                        "Description": "Second",
                    },
                    {
                        "POG": "POG1",
                        "Side": 1,
                        "Row": 1,
                        "Column": 1,
                        "Item Number": "111",
                        "UPC": "11111111111",
                        "Retail": "9.99",
                        "CPP": "2",
                        "Description": "First",
                    },
                ]
            ).to_excel(workbook, index=False)
            manual_index = SamsManualImageMappingIndex(
                by_upc={"11111111111": str(image_path), "22222222222": str(image_path)},
                approved_count=2,
            )

            with patch(
                "app.sams_club.service.load_sams_manual_image_mappings",
                lambda: manual_index,
            ):
                result = build_sams_planogram_structure(
                    workbook,
                    selected_pog="POG1",
                    local_image_root="",
                )
            pdf_result = render_sams_planogram_pdf(result.planogram)

        slots = result.planogram.side_pages[0].rows[0].slots
        self.assertEqual([slot.column for slot in slots], [1, 2])
        self.assertEqual([slot.cpp for slot in slots], ["2", "6"])
        self.assertGreater(len(pdf_result.pdf_bytes), 0)
        self.assertEqual(pdf_result.missing_image_slots, 0)


if __name__ == "__main__":
    unittest.main()
