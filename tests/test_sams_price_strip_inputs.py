import io
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
import pandas as pd
from reportlab.pdfgen import canvas

from app.sams_club.price_strip_inputs import parse_source, parse_text
from app.sams_club.price_pocket import pocket_records, render_price_pocket_pdf
from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.render_price_strips_html import _build_full_html

BASELINE = """NAME: Sams GOM COC $60 Club 2026
PRICE: 60
QUANTITY: 1
TYPE: GIFT CARD
ITEM: 984323644
UPC: 874586044822

NAME: Sams GOM COC $120 Plus 2026
PRICE: 120
QUANTITY: 1
TYPE: GIFT CARD
ITEM: 984323645
UPC: 874586044839
"""


def source(payload, suffix):
    stream = io.BytesIO(payload)
    stream.name = "source" + suffix
    return stream


def docx(text=None, rows=None):
    def p(value):
        return f"<w:p><w:r><w:t>{escape(value)}</w:t></w:r></w:p>"
    body = "".join(p(line) for line in text.splitlines()) if text else "<w:tbl>" + "".join("<w:tr>" + "".join("<w:tc>" + p(cell) + "</w:tc>" for cell in row) + "</w:tr>" for row in rows) + "</w:tbl>"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + '</w:body></w:document>')
    return source(output.getvalue(), ".docx")


def pdf(text):
    output = io.BytesIO()
    c = canvas.Canvas(output)
    obj = c.beginText(30, 800)
    obj.setFont("Helvetica", 10)
    for line in text.splitlines():
        obj.textLine(line)
    c.drawText(obj)
    c.save()
    return source(output.getvalue(), ".pdf")


def xlsx(rows):
    output = io.BytesIO()
    pd.DataFrame(rows).to_excel(output, sheet_name="Price Strip Data", index=False)
    return source(output.getvalue(), ".xlsx")


class InputTests(unittest.TestCase):
    def test_pocket_uses_active_renderer_font_registration_and_fallback(self):
        from unittest.mock import patch
        from app.sams_club.price_pocket import _pocket_html
        from app.sams_club import render_price_strips_html as renderer
        records = pocket_records(parse_text(BASELINE))
        with patch.object(renderer, "sams_font_face_css", return_value="/* shared font registration */") as loader:
            content = _pocket_html(records, [])
            loader.assert_called_once()
        self.assertIn("/* shared font registration */", content)
        self.assertIn("font-family: " + renderer.SAMS_FONT_STACK, content)
        self.assertNotIn("Raleway", content)
        with patch.object(renderer, "_font_file_to_data_uri", side_effect=OSError("missing")):
            warnings = []
            content = _pocket_html(records, warnings)
        self.assertIn("Gibson font data URI load failed", warnings[0])
        self.assertIn('font-family: "Gibson", Arial, sans-serif', content)

    def test_all_formats_pocket(self):
        expected = parse_text(BASELINE)
        for upload in (docx(BASELINE), pdf(BASELINE), xlsx(expected)):
            records = pocket_records(parse_source(upload))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].item, "984323644")
            self.assertEqual(str(records[1].price), "120")

    def test_aliases_zeros_defaults_and_tables(self):
        text = "Product Name | Retail Price | WM Item | UPC #\nExample | 60 | 000123 | 001234567890"
        for upload in (docx(text), pdf(text), docx(rows=[line.split(" | ") for line in text.splitlines()])):
            record = pocket_records(parse_source(upload))[0]
            self.assertEqual(record.item, "000123")
            self.assertEqual(record.upc, "001234567890")
            self.assertEqual(record.quantity, 1)
            self.assertEqual(record.type, "GIFT CARD")
        records = parse_text("TITLE: A\nRETAIL: 12.50\nSAM'S ITEM: 001\nUPC NUMBER: 0002\nTITLE: B\nRETAIL: 15\nITEM #: 003\nUPC #: 0004")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["item_number"], "003")
        self.assertEqual(parse_source(xlsx(records))[0]["upc"], "0002")

    def test_missing_and_ambiguous_values(self):
        for upload in (docx("NAME: Test"), pdf("NAME: Test")):
            with self.assertRaisesRegex(ValueError, "retail, item_number, upc"):
                pocket_records(parse_source(upload))
        for price in ("1,20", "1e2", "NaN", "-2", "12.345", "60 or 120"):
            with self.assertRaisesRegex(ValueError, "Invalid price"):
                pocket_records([{"name": "Test", "price": price, "item": "001", "upc": "002"}])
        with self.assertRaisesRegex(ValueError, "quantity"):
            pocket_records([{"name": "Test", "price": "1", "item": "001", "upc": "002", "qty": "1.5"}])
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            parse_source(pdf(""))

    def test_pdf_grid_table_and_empty_table_cells(self):
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors
        output = io.BytesIO()
        table = Table([["Name", "Price", "Item", "UPC"], ["Example", "60", "001", "0002"]])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.black)]))
        SimpleDocTemplate(output).build([table])
        record = pocket_records(parse_source(source(output.getvalue(), ".pdf")))[0]
        self.assertEqual(record.upc, "0002")
        parsed = parse_source(docx(rows=[["Name", "Price", "Desc 2"], ["Example", "60", ""]]))
        self.assertEqual(parsed[0]["desc_2"], "")

    def test_document_geometry_errors_do_not_guess(self):
        row = {"POG": "TEST", "Side": "1.5", "Row": "1", "Column": "1", "Item Number": "001", "Brand": "B", "Desc 1": "D", "Desc 2": "", "Retail": "60", "Length": "", "Data on bottom left": ""}
        result = build_sams_price_strip_rows(docx("\n".join(f"{k}: {v}" for k, v in row.items())), "Standard")
        self.assertIn("side must be a positive integer", result.errors[0])
        self.assertIn("invalid length", result.errors[0])

    def test_existing_modes_share_document_path_and_workbook_results(self):
        row = {"POG": "TEST", "Side": "1", "Row": "1", "Column": "1", "Item Number": "001234", "Brand": "BRAND", "Desc 1": "DESCRIPTION", "Desc 2": "GIFT CARD", "Retail": "60", "Length": '30.75 x 3.5', "Data on bottom left": "TEST FOOTER"}
        labeled = "\n".join(f"{k}: {v}" for k, v in row.items())
        for mode in ("Standard", "Sam's Holiday", "Sam's Holiday 2026"):
            native = build_sams_price_strip_rows(xlsx([row]), template_name=mode)
            self.assertFalse(native.errors)
            for upload in (docx(labeled), pdf(labeled)):
                result = build_sams_price_strip_rows(upload, template_name=mode)
                self.assertFalse(result.errors, result.errors)
                self.assertEqual([asdict(r) for r in result.strip_rows], [asdict(r) for r in native.strip_rows])
                self.assertEqual(_build_full_html(result.strip_rows, [], mode, False), _build_full_html(native.strip_rows, [], mode, False))
            for upload in (docx(BASELINE), pdf(BASELINE)):
                result = build_sams_price_strip_rows(upload, template_name=mode)
                self.assertTrue(result.errors)
                self.assertIn("pog", result.errors[0])
                self.assertIn("column", result.errors[0])
                self.assertFalse(result.strip_rows)

    def test_pdf_vector_content_geometry_and_baseline_artifact(self):
        records = pocket_records(parse_text(BASELINE))
        result = render_price_pocket_pdf(records)
        self.assertEqual((result.rendered_pages, result.rendered_segments), (1, 2))
        with fitz.open(stream=result.pdf_bytes, filetype="pdf") as document:
            page = document[0]
            self.assertEqual(tuple(page.rect), (0, 0, 612, 576))
            self.assertFalse(page.get_images())
            for i, record in enumerate(records):
                region = fitz.Rect(0, i*288, 612, (i+1)*288)
                text = page.get_text(clip=region)
                for expected in (record.name, f"(1 X ${record.price})", "GIFT CARD", record.item, f"({record.upc})", "$", str(record.price), "00"):
                    self.assertIn(expected, text)
                self.assertNotIn("ITEM", text)
                self.assertNotIn("UPC", text)
                item_box = page.search_for(record.item)[0]
                upc_box = page.search_for(f"({record.upc})")[0]
                self.assertAlmostEqual(item_box.x0, 18, places=2)
                self.assertGreater(upc_box.y0, item_box.y0)
                for block in page.get_text("dict", clip=region)["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            self.assertTrue(region.contains(fitz.Rect(span["bbox"])), span)
        three = render_price_pocket_pdf(records + records[:1])
        self.assertEqual(three.rendered_pages, 2)


if __name__ == "__main__":
    unittest.main()
