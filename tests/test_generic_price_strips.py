"""Generic merge data uses the supplied drawing's dimensions and alignment."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import pymupdf
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.pages._price_strip_sheet import import_price_strip_sheet
from app.sams_club.generic_price_strips import GENERIC_SIZES, geometry_for_size
from app.templates.eps_preview import parse_eps, render_eps_preview


def workbook(records=None):
    book = Workbook()
    sheet = book.active
    sheet.append(['UPC12', 'Brand', 'Signing Desc 1', 'Signing Desc 2', 'Retail Price',
                  'Dollar', 'Cents', 'SIDE', 'ROW', 'POSITION', 'MESSAGE ON BOTTOM LEFT OF STRIP'])
    for values in records or [
        ['001234567890', 'Product A', '3 X $25', 'Gift Card', 0, 0, 0, 1, 1, 1, 'SIDE 1 ROW 1'],
        ['001234567890', 'Product A', '3 X $25', 'Gift Card', 71.98, 71, 98, 1, 1, 2, None],
        ['196742045383', 'Product B', '$100', 'Gift Card', None, 94, 98, 1, 1, 3, None],
        ['196742045383', 'Product B', '$100', 'Gift Card', 94.98, 94, 98, 3, 6, 3, 'SIDE 3 ROW 6'],
    ]:
        sheet.append(values)
    output = io.BytesIO()
    book.save(output)
    return output.getvalue()


class GenericPriceStripTests(unittest.TestCase):
    def test_schema_mapping_preserves_duplicate_upcs_zero_prices_gaps_and_footer(self):
        data = import_price_strip_sheet(workbook())
        self.assertEqual((data['format'], data['products'], data['rows']), ('generic', 4, 2))
        sides = data['pogs']['Generic Price Strips']
        self.assertEqual(set(sides), {1, 3})
        self.assertEqual(sides[3]['start'], 6)
        self.assertEqual(sides[1]['strip_size'], 'generic_11')
        row = sides[1]['rows'][0]
        self.assertEqual(row['footer_text'], 'SIDE 1 ROW 1')
        self.assertEqual(row['tickets'][0]['Item #'], '001234567890')
        self.assertEqual([t['Price'] for t in row['tickets']], ['0.00', '71.98', '94.98'])
        self.assertEqual(sides[3]['rows'][0]['tickets'][0]['Brand'], '')

    def test_bad_position_and_duplicate_position_rejected(self):
        record = ['123456789012', 'Product', '$25', 'Card', 22.5, 22, 50, 1, 1, 1, None]
        with self.assertRaisesRegex(ValueError, 'duplicate position'):
            import_price_strip_sheet(workbook([record, record]))
        record[9] = 7
        with self.assertRaisesRegex(ValueError, 'Position 1–6'):
            import_price_strip_sheet(workbook([record]))

    def test_baseline_shared_edges_recover_all_46_outlines(self):
        drawing = parse_eps(builder.GENERIC_EPS_PATH.read_bytes())
        self.assertEqual(len(drawing.cuts), 46)
        for size, expected in zip(builder.GENERIC_STRIP_SIZES, [12, 12, 10, 12]):
            g = geometry_for_size(size)
            self.assertEqual(sum(abs(r[2]-r[0]-g.width_pt) < .05 for r in drawing.cuts), expected)

    def test_clean_pdf_and_preview_at_each_supported_size(self):
        for size in builder.GENERIC_STRIP_SIZES:
            with self.subTest(size=size):
                g = geometry_for_size(size)
                tickets = builder._default_tickets(1, size)
                tickets[-1].update({'Brand':'LAST POSITION', 'Item #':'001234567890', 'Price':'94.98'})
                clean = builder._create_pdf('Generic', 1, 2, [tickets], strip_size=size)
                guided = builder._create_pdf('Generic', 1, 2, [tickets], strip_size=size, calibration=True)
                with pymupdf.open(stream=clean) as doc:
                    self.assertEqual(len(doc), 1)
                    page = doc[0]
                    self.assertAlmostEqual(page.rect.width, g.width_pt+36)
                    self.assertAlmostEqual(page.rect.height, g.height_pt+36, delta=.5)
                    self.assertIn('001234567890', page.get_text())
                    # Production keeps gray crop marks, but excludes colored guides.
                    self.assertFalse(any(d['color'] and max(d['color']) - min(d['color']) > .1
                                         for d in page.get_drawings()))
                    self.assertTrue(all(page.rect.contains(pymupdf.Rect(*span['bbox']))
                        for block in page.get_text('dict')['blocks'] if 'lines' in block
                        for line in block['lines'] for span in line['spans']))
                overview, detail = render_eps_preview(builder.GENERIC_EPS_PATH.read_bytes(), size,
                                                     guided, guides_in_pdf=True, row_number=2)
                self.assertTrue(overview.startswith(b'\x89PNG'))
                self.assertTrue(detail.startswith(b'\x89PNG'))

    def test_format_switch_restores_drafts_and_custom_eps(self):
        holiday = {1:{'start':1, 'strip_size':'thin', 'rows':[{'tickets':[{'Brand':'Keep me'}]}]}}
        state = {builder.ROWS_KEY:deepcopy(holiday), builder.EPS_KEY:{'name':'holiday.eps','bytes':b'eps'}}
        with patch.object(builder.st, 'session_state', state):
            builder._switch_format('generic')
            self.assertEqual(builder._side_rows(1)['strip_size'], 'generic_11')
            self.assertIn('Generic baseline', builder._active_eps()['name'])
            builder._switch_format('holiday')
            self.assertEqual(state[builder.ROWS_KEY], holiday)
            self.assertEqual(state[builder.EPS_KEY]['name'], 'holiday.eps')

    def test_generic_tickets_preserve_the_reference_font_and_spacing_ratios(self):
        def spans(pdf):
            with pymupdf.open(stream=pdf) as doc:
                return {span['text']:span for block in doc[0].get_text('dict')['blocks']
                        for line in block.get('lines', []) for span in line['spans']}
        reference_tickets = builder._default_tickets(1, 'thin')
        reference_tickets[0].update({'Brand':'Spacing Brand', 'Pack / value':'Line One',
                                     'Type':'Line Two', 'Price':'71.98', 'Item #':'001234567890'})
        reference = spans(builder._create_pdf('Reference', 1, 1, [reference_tickets], strip_size='thin'))
        for size in builder.GENERIC_STRIP_SIZES:
            with self.subTest(size=size):
                tickets = builder._default_tickets(1, size)
                tickets[0].update({'Brand':'Spacing Brand', 'Pack / value':'Line One',
                                   'Type':'Line Two', 'Price':'71.98', 'Item #':'001234567890'})
                pdf = builder._create_pdf('Spacing', 1, 1, [tickets], strip_size=size)
                actual = spans(pdf)
                brand = actual['SPACING BRAND']
                ref_brand = reference['SPACING BRAND']
                scale = brand['size'] / ref_brand['size']
                for label in ('LINE ONE', 'LINE TWO', '71', '001234567890'):
                    self.assertAlmostEqual(actual[label]['size'], reference[label]['size'] * scale, delta=.02)
                    self.assertAlmostEqual(actual[label]['origin'][1] - brand['origin'][1],
                        (reference[label]['origin'][1] - ref_brand['origin'][1]) * scale, delta=.25)
                    self.assertAlmostEqual(actual[label]['origin'][0] - brand['origin'][0],
                        (reference[label]['origin'][0] - ref_brand['origin'][0]) * scale, delta=.25)
                self.assertAlmostEqual(actual['LINE ONE']['origin'][0], brand['origin'][0], delta=.02)
                self.assertAlmostEqual(actual['LINE TWO']['origin'][0], brand['origin'][0], delta=.02)

    def test_upload_auto_switches_format_in_streamlit_and_generates_all_sides(self):
        app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from app.pages import price_strip_builder as builder
from tests.test_generic_price_strips import workbook
def upload():
    st.session_state[builder.SHEET_UPLOAD_KEY] = SimpleNamespace(name='Generic.xlsx', getvalue=workbook)
    builder._load_sheet_upload()
st.button('Load sample', on_click=upload)
builder.render()
''', default_timeout=15).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state[builder.FORMAT_KEY], 'generic')
        self.assertFalse(any(w.label == 'Strip dimensions' for w in app.selectbox))
        self.assertEqual(app.session_state[builder.SHEET_KEY]['pogs']['Generic Price Strips'][1]['strip_size'], 'generic_11')
        next(w for w in app.button if w.label == 'Create Price Strip PDF').click().run()
        self.assertFalse(app.exception)
        with pymupdf.open(stream=app.session_state[builder.PDF_KEY]['bytes']) as doc:
            self.assertEqual(len(doc), 2)
            self.assertIn('SIDE 3 ROW 6', doc[1].get_text())
        next(w for w in app.selectbox if w.label == 'Template format').select('holiday').run()
        self.assertFalse(app.exception)
        self.assertFalse(any(w.label == 'Create Price Strip PDF' for w in app.button))
        next(w for w in app.selectbox if w.label == 'Template format').select('generic').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state[builder.ROWS_KEY][1]['rows'][0]['tickets']), 3)
