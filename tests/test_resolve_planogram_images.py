from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts import resolve_planogram_images as rpi


class ResolvePlanogramImagesTests(unittest.TestCase):
    def _image(self, path: Path, color: str = "white") -> Path:
        Image.new("RGB", (12, 12), color).save(path)
        return path

    def _workbook(self, path: Path, rows: list[dict[str, object]]) -> Path:
        pd.DataFrame(rows).to_excel(path, index=False)
        return path

    def _catalog(self, path: Path, rows: list[dict[str, object]]) -> Path:
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _base_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "POG NAME": "POG1",
            "Segment": 1,
            "Row": 1,
            "Col": 1,
            "Merchant SKU": "12345",
            "UPC 11": "19674217114",
            "Name": "PlayStation Store 2 X $25 Gift Card",
            "Cards Per Peg": "6",
            "Brand": "PlayStation",
            "Description 2": "PlayStation Store 2 X $25",
        }
        row.update(overrides)
        return row

    def test_workbook_alias_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook = self._workbook(Path(temp_dir) / "source.xlsx", [self._base_row()])
            records, mapping, sheet = rpi.load_workbook_records(workbook)

        self.assertEqual(sheet, "Sheet1")
        self.assertEqual(mapping["pog"], "POG NAME")
        self.assertEqual(mapping["side"], "Segment")
        self.assertEqual(mapping["column"], "Col")
        self.assertEqual(mapping["item_number"], "Merchant SKU")
        self.assertEqual(mapping["cpp"], "Cards Per Peg")
        self.assertEqual(records[0]["upc"], "19674217114")

    def test_intentional_blank_exclusion(self) -> None:
        self.assertTrue(rpi.is_intentional_blank({"intentional_blank": "TRUE"}))
        self.assertTrue(rpi.is_intentional_blank({"image_status": "INTENTIONAL_BLANK"}))
        self.assertTrue(rpi.is_intentional_blank({"merchant_category": "SAMS FP GFT"}))

    def test_samtemp6_exclusion(self) -> None:
        self.assertTrue(rpi.is_intentional_blank({"item_number": "SAMTEMP6"}))

    def test_gci_exclusion(self) -> None:
        self.assertTrue(rpi.is_gci({"image_status": "GCI_IMAGE_PENDING"}))
        self.assertTrue(rpi.is_gci({"merchant_category": "SAMS GCI"}))
        self.assertTrue(rpi.is_gci({"product_name": "GCI display holder"}))

    def test_repeated_position_deduplication(self) -> None:
        groups = rpi.dedupe_products(
            [
                {"pog": "POG1", "side": "1", "row": "1", "column": "1", "upc": "111", "item_number": "222", "product_name": "A"},
                {"pog": "POG1", "side": "1", "row": "1", "column": "2", "upc": "111", "item_number": "222", "product_name": "A"},
            ]
        )

        self.assertEqual(len(groups), 1)
        group = next(iter(groups.values()))
        self.assertEqual(len(group.positions), 2)

    def test_existing_manual_mapping_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = self._image(root / "manual.png")
            mapping = root / "manual.csv"
            mapping.write_text(
                "UPC,Item Number,file_path,approved,source,notes\n"
                f"19674217114,12345,{image},TRUE,test,\n",
                encoding="utf-8",
            )
            product = rpi.ProductGroup("RG00001", "19674217114", "196742171143", "12345", "PlayStation $25", "", "", "6")
            manual_index = rpi.load_sams_manual_image_mappings(mapping)

            resolved = rpi.product_already_resolved(product, None, manual_index)

        self.assertTrue(resolved)

    def test_exact_upc_match(self) -> None:
        image = rpi.CatalogEntry(
            file_path="x.jpg",
            filename="19674217114.jpg",
            filename_upc="19674217114",
            detected_brand="PLAYSTATION",
            filename_keys=rpi.identifier_keys("19674217114"),
        )
        product = rpi.ProductGroup("RG00001", "19674217114", "196742171143", "12345", "PlayStation $25", "", "PlayStation", "6")

        score = rpi.score_candidate(product, image)

        self.assertEqual(score["identifier_score"], 100)

    def test_upc11_to_upc12_normalization(self) -> None:
        self.assertEqual(rpi.calculated_upc12("19674217114"), "196742171143")
        self.assertIn("196742171143", rpi.identifier_keys("19674217114"))

    def test_leading_zero_handling(self) -> None:
        keys = rpi.identifier_keys("087458605402")

        self.assertIn("087458605402", keys)
        self.assertIn("87458605402", keys)

    def test_item_number_filename_match(self) -> None:
        image = rpi.CatalogEntry(
            file_path="x.jpg",
            filename="12345.jpg",
            filename_upc="",
            detected_brand="",
            filename_keys=rpi.identifier_keys("12345"),
        )
        product = rpi.ProductGroup("RG00001", "11111111111", "", "12345", "Unknown", "", "", "9")

        score = rpi.score_candidate(product, image)

        self.assertEqual(score["identifier_score"], 90)

    def test_brand_match_scoring(self) -> None:
        image = rpi.CatalogEntry(file_path="x.jpg", filename="x.jpg", detected_brand="PLAYSTATION")
        product = rpi.ProductGroup("RG00001", "111", "", "222", "PSN $25", "", "", "1")

        self.assertEqual(rpi.score_candidate(product, image)["brand_score"], 40)

    def test_denomination_conflict_penalty(self) -> None:
        image = rpi.CatalogEntry(file_path="x.jpg", filename="x.jpg", detected_denomination="$50")
        product = rpi.ProductGroup("RG00001", "111", "", "222", "PlayStation $25", "", "", "1")

        self.assertEqual(rpi.score_candidate(product, image)["denomination_score"], -60)

    def test_pack_conflict_penalty(self) -> None:
        image = rpi.CatalogEntry(file_path="x.jpg", filename="x.jpg", detected_pack_quantity="4")
        product = rpi.ProductGroup("RG00001", "111", "", "222", "PlayStation 2 X $25", "", "", "1")

        self.assertEqual(rpi.score_candidate(product, image)["pack_score"], -35)

    def test_top_five_ranking(self) -> None:
        product = rpi.ProductGroup("RG00001", "11111111111", "", "222", "PlayStation $25", "", "PlayStation", "1")
        rows = []
        for idx in range(7):
            rows.append(
                rpi.score_candidate(
                    product,
                    rpi.CatalogEntry(
                        file_path=f"{idx}.jpg",
                        filename=f"{idx}.jpg",
                        detected_text="PlayStation $25" if idx == 6 else "Steam $5",
                        detected_brand="PLAYSTATION" if idx == 6 else "STEAM",
                        detected_denomination="$25" if idx == 6 else "$5",
                    ),
                )
            )
        rows.sort(key=lambda row: row["total_score"], reverse=True)

        self.assertEqual(len(rows[:5]), 5)
        self.assertEqual(rows[0]["candidate_file_path"], "6.jpg")

    def test_review_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "review.csv"
            rpi.write_review_csv([], output)
            with output.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))

        self.assertEqual(header, rpi.REVIEW_COLUMNS)

    def test_one_approved_row_creates_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = self._image(root / "candidate.jpg")
            review = root / "review.csv"
            mapping = root / "manual.csv"
            rpi.write_review_csv(
                [
                    {
                        "review_group_id": "RG00001",
                        "unresolved_upc": "111",
                        "item_number": "222",
                        "product_name": "Product",
                        "candidate_file_path": str(image),
                        "approved": "TRUE",
                        "selected": "TRUE",
                        "confidence": "High",
                        "total_score": "100",
                    }
                ],
                review,
            )

            summary = rpi.apply_approved(review, mapping)
            rows = rpi.read_csv(mapping)

        self.assertEqual(summary.mappings_added, 1)
        self.assertEqual(rows[0]["source"], "OFFLINE_OCR_REVIEW")

    def test_repeated_positions_require_one_mapping(self) -> None:
        groups = rpi.dedupe_products(
            [
                {"pog": "POG1", "side": "1", "row": "1", "column": "1", "upc": "111", "item_number": "222", "product_name": "A"},
                {"pog": "POG1", "side": "1", "row": "1", "column": "2", "upc": "111", "item_number": "222", "product_name": "A"},
            ]
        )

        self.assertEqual(len(groups), 1)

    def test_multiple_approvals_in_one_group_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image1 = self._image(root / "candidate1.jpg")
            image2 = self._image(root / "candidate2.jpg")
            review = root / "review.csv"
            mapping = root / "manual.csv"
            rows = [
                {"review_group_id": "RG00001", "unresolved_upc": "111", "item_number": "222", "candidate_file_path": str(image1), "approved": "TRUE", "selected": "TRUE"},
                {"review_group_id": "RG00001", "unresolved_upc": "111", "item_number": "222", "candidate_file_path": str(image2), "approved": "TRUE", "selected": "TRUE"},
            ]
            rpi.write_review_csv(rows, review)

            summary = rpi.apply_approved(review, mapping)

        self.assertEqual(summary.duplicate_approvals_rejected, 2)
        self.assertEqual(summary.mappings_added, 0)

    def test_invalid_file_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review = root / "review.csv"
            mapping = root / "manual.csv"
            rpi.write_review_csv(
                [{"review_group_id": "RG00001", "unresolved_upc": "111", "item_number": "222", "candidate_file_path": str(root / "missing.jpg"), "approved": "TRUE", "selected": "TRUE"}],
                review,
            )

            summary = rpi.apply_approved(review, mapping)

        self.assertEqual(summary.invalid_approvals_skipped, 1)

    def test_existing_mapping_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing_image = self._image(root / "existing.jpg")
            new_image = self._image(root / "new.jpg")
            mapping = root / "manual.csv"
            mapping.write_text(
                "UPC,Item Number,file_path,approved,source,notes\n"
                f"999,888,{existing_image},TRUE,OLD,keep\n",
                encoding="utf-8",
            )
            review = root / "review.csv"
            rpi.write_review_csv(
                [{"review_group_id": "RG00001", "unresolved_upc": "111", "item_number": "222", "candidate_file_path": str(new_image), "approved": "TRUE", "selected": "TRUE"}],
                review,
            )

            rpi.apply_approved(review, mapping)
            rows = rpi.read_csv(mapping)

        self.assertEqual(len(rows), 2)
        self.assertTrue(any(row["source"] == "OLD" for row in rows))

    def test_mapping_backup_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = self._image(root / "candidate.jpg")
            mapping = root / "manual.csv"
            mapping.write_text("UPC,Item Number,file_path,approved,source,notes\n", encoding="utf-8")
            review = root / "review.csv"
            rpi.write_review_csv(
                [{"review_group_id": "RG00001", "unresolved_upc": "111", "item_number": "222", "candidate_file_path": str(image), "approved": "TRUE", "selected": "TRUE"}],
                review,
            )

            summary = rpi.apply_approved(review, mapping)
            self.assertTrue(summary.backup_path)
            self.assertTrue(Path(summary.backup_path).exists())

    def test_cpp_is_preserved_but_not_used_in_scoring(self) -> None:
        image = rpi.CatalogEntry(file_path="x.jpg", filename="x.jpg", detected_text="CPP 99 PlayStation $25")
        a = rpi.ProductGroup("RG00001", "111", "", "222", "PlayStation $25", "", "PlayStation", "1")
        b = rpi.ProductGroup("RG00001", "111", "", "222", "PlayStation $25", "", "PlayStation", "999")

        score_a = rpi.score_candidate(a, image)
        score_b = rpi.score_candidate(b, image)

        self.assertEqual(score_a["total_score"], score_b["total_score"])
        self.assertEqual(score_a["cpp"], "1")

    def test_no_app_or_renderer_modules_are_referenced_by_script(self) -> None:
        source = Path(rpi.__file__).read_text(encoding="utf-8")

        self.assertNotIn("home.py", source)
        self.assertNotIn("render_planogram", source)
        self.assertNotIn("render_price_strips", source)
        self.assertNotIn("streamlit", source.lower())


if __name__ == "__main__":
    unittest.main()
