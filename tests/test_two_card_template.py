"""The two-card template keeps two 8.5 x 4 panels and native Gibson prices."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.pages._price_strip_sheet import import_price_strip_sheet
from app.sams_club.generic_price_strips import geometry_for_size
from app.templates.eps_preview import parse_eps,strip_crosshair_layout,render_eps_preview


def workbook(count=2,side=1,prices=None):
    book = Workbook()
    sheet = book.active
    sheet.title = 'Price Strip Data'
    sheet.append(['POG','Item Number','Brand','Desc 1','Desc 2','Retail','Side','Row','Column'])
    for col in range(1,count+1):
        price = prices[col-1] if prices is not None else (0 if col==1 else 22.50)
        sheet.append(['Two Card Test',f'0012345678{col}','ZIFT HOLIDAY','1 X $25','GIFT CARD',price,side,3,col])
    output=io.BytesIO()
    book.save(output)
    book.close()
    return output.getvalue()


class TwoCardTemplateTests(unittest.TestCase):
    def test_baseline_dimensions_and_green_crosshairs(self):
        with patch.object(builder.st,'session_state',{builder.FORMAT_KEY:'two_card'}):
            baseline=builder._active_eps()
        g=geometry_for_size('generic_two_card')
        self.assertEqual((g.width_in,g.height_in,g.slot_count),(17,4,2))
        centers=strip_crosshair_layout(parse_eps(baseline['bytes']),'generic_two_card')
        self.assertEqual([point[0] for point in centers],[306,918])
        self.assertTrue(all(abs(point[1]-23.04)<.001 for point in centers))

    def test_import_requires_two_complete_prices_and_preserves_identifiers_and_zero(self):
        for side in (1,2,3,4):
            data=import_price_strip_sheet(workbook(side=side),template_format='two_card')
            self.assertEqual(data['format'],'two_card')
            draft=data['pogs']['Two Card Test'][side]
            self.assertEqual(draft['strip_size'],'generic_two_card')
            self.assertEqual(len(draft['rows'][0]['tickets']),2)
            self.assertEqual(draft['rows'][0]['tickets'][0]['Item #'],'00123456781')
            self.assertEqual(draft['rows'][0]['tickets'][0]['Price'],'0')
        for payload,message in [(workbook(1),'exactly 2 prices'),(workbook(3),'exceeds 2'),
                                (workbook(prices=[None,22.50]),'enter a valid price'),
                                (workbook(prices=[-1,22.50]),'nonnegative')]:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                import_price_strip_sheet(payload,template_format='two_card')

    def test_eight_and_ten_price_rule_checks_the_complete_row(self):
        for side,count in [(1,8),(2,10)]:
            data=import_price_strip_sheet(workbook(count,side),template_format='kk250133')
            self.assertEqual(len(data['pogs']['Two Card Test'][side]['rows'][0]['tickets']),count)
            with self.assertRaisesRegex(ValueError,'exactly 8 or 10 prices'):
                import_price_strip_sheet(workbook(count-1,side),template_format='kk250133')

    def test_plain_eps_reader_rejects_executable_commands(self):
        data=builder.TWO_CARD_EPS_PATH.read_bytes()
        with self.assertRaisesRegex(ValueError,'unsupported'):
            parse_eps(data.replace(b'gsave',b'gsave\n(secret) run',1))

    def test_pdf_output_has_exact_trim_size_both_prices_and_no_colored_guides(self):
        draft=import_price_strip_sheet(workbook(),template_format='two_card')['pogs']['Two Card Test']
        baseline=builder.TWO_CARD_EPS_PATH.read_bytes()
        clean=builder._create_all_sides_pdf('Two Card Test',draft,eps_data=baseline)
        with pymupdf.open(stream=clean,filetype='pdf') as pdf:
            self.assertEqual(len(pdf),1)
            self.assertAlmostEqual(pdf[0].rect.width,17*72+36,delta=.5)
            self.assertAlmostEqual(pdf[0].rect.height,4*72+36,delta=.5)
            self.assertIn('00123456781',pdf[0].get_text())
            self.assertIn('00123456782',pdf[0].get_text())
            self.assertFalse(any(p['color'] and max(p['color'])-min(p['color'])>.1 for p in pdf[0].get_drawings()))
            spans=[span for block in pdf[0].get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line['spans']]
            self.assertTrue(all(pdf[0].rect.contains(pymupdf.Rect(span['bbox'])) for span in spans))
        tickets=draft[1]['rows'][0]['tickets']
        guided=builder._create_pdf('Two Card Test',1,3,[tickets],strip_size='generic_two_card',eps_data=baseline,calibration=True)
        _,image=render_eps_preview(baseline,'generic_two_card',guided,price_above=True,guides_in_pdf=True)
        self.assertTrue(image.startswith(b'\x89PNG'))

    def test_ui_format_upload_preview_and_real_pdf_download(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow']='Templates'
        app.session_state['templates:selected_type']='2. Price Strips'
        state={builder.FORMAT_KEY:'two_card',builder.SHEET_UPLOAD_KEY:SimpleNamespace(name='two.xlsx',getvalue=workbook)}
        with patch.object(builder.st,'session_state',state):
            builder._load_sheet_upload()
        state.pop(builder.SHEET_UPLOAD_KEY)
        for key,value in state.items():
            app.session_state[key]=value
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state[builder.FORMAT_KEY],'two_card')
        self.assertTrue(any("Sam's Two Cards" in option for option in next(w for w in app.selectbox if w.label=='Template format').options))
        next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state[builder.PDF_KEY]['pages'],1)
        self.assertTrue(app.get('download_button'))
