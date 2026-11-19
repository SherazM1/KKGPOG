"""Eight/ten-price rows select geometry by their data, independently of side."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf
from openpyxl import Workbook
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.pages._price_strip_sheet import import_price_strip_sheet
from app.sams_club.generic_price_strips import geometry_for_size


def workbook(groups):
    book=Workbook()
    sheet=book.active
    sheet.title='Price Strip Data'
    sheet.append(['POG','Item Number','Brand','Desc 1','Desc 2','Retail','Side','Row','Column'])
    for pog,side,row,count in groups:
        for col in range(1,count+1):
            sheet.append([pog,f'001{side}{row}{col:02d}','ZIFT HOLIDAY','1 X $25','GIFT CARD',22.5,side,row,col])
    stream=io.BytesIO()
    book.save(stream)
    book.close()
    return stream.getvalue()


class Kk250133ImportTests(unittest.TestCase):
    def test_both_counts_work_on_any_side_and_same_side_rows_can_use_different_sizes(self):
        data=import_price_strip_sheet(workbook([('Mixed',1,1,10),('Mixed',1,5,8),('Mixed',2,2,8)]),template_format='kk250133')
        source=data['pogs']['Mixed']
        self.assertEqual([r['strip_size'] for r in source[1]['rows']],['generic_kk250133_long','generic_kk250133_short'])
        with patch.object(builder,'_render_strips',return_value=b'pdf') as render:
            builder._create_all_sides_pdf('Mixed',source)
        rows=render.call_args.args[0]
        self.assertEqual([(r.side,r.row,len(r.segments)) for r in rows],[(1,1,10),(1,5,8),(2,2,8)])
        self.assertEqual(rows[0].segments[-1].item_number,'0011110')

    def test_mixed_workbook_lists_only_complete_pogs_and_reports_omitted_groups(self):
        payload=workbook([('Legacy',1,1,6),('Legacy',2,1,8),('Test 8',1,1,8),('Test 10',1,1,10)])
        result=import_price_strip_sheet(payload,template_format='kk250133')
        self.assertEqual(set(result['pogs']),{'Test 8','Test 10'})
        self.assertEqual((result['products'],result['rows']),(18,2))
        self.assertTrue(any('Skipped POG "Legacy"' in warning for warning in result['warnings']))
        with self.assertRaisesRegex(ValueError,'exactly 8 or 10 prices'):
            import_price_strip_sheet(workbook([('Incomplete',1,1,7)]),template_format='kk250133')

    def test_partial_spacing_keeps_row_five_when_row_one_has_a_different_size(self):
        source=import_price_strip_sheet(workbook([('Mixed',1,1,10),('Mixed',1,5,8)]),template_format='kk250133')['pogs']['Mixed']
        g=geometry_for_size('generic_kk250133_short')
        stream=io.BytesIO()
        pdf=canvas.Canvas(stream,pagesize=(g.width_pt+36,g.height_pt+36))
        pdf.rect(18,18,g.width_pt,g.height_pt)
        pdf.setStrokeColorRGB(0,.7,0)
        from app.sams_club.generic_price_strips import GENERIC_SIZES
        for x in GENERIC_SIZES['generic_kk250133_short'][1]:
            pdf.line(18+x-9,18+g.height_pt/2,18+x+9,18+g.height_pt/2)
            pdf.line(18+x,18+g.height_pt/2-9,18+x,18+g.height_pt/2+9)
        pdf.showPage()
        pdf.save()
        filtered=builder._spacing_compatible_drafts(source,stream.getvalue())
        self.assertEqual(builder._row_numbers(filtered[1]),[5])
        self.assertEqual(filtered[1]['rows'][0]['strip_size'],'generic_kk250133_short')
        self.assertEqual(builder._row_numbers(source[1]),[1,5])

    def test_ui_import_and_pdf_generation_accept_ten_prices_on_side_one(self):
        payload=workbook([('Legacy',1,1,6),('Test 10',1,1,10),('Test 8',1,1,8)])
        state={builder.FORMAT_KEY:'kk250133',builder.SHEET_UPLOAD_KEY:SimpleNamespace(name='mixed.xlsx',getvalue=lambda:payload)}
        with patch.object(builder.st,'session_state',state):
            builder._load_sheet_upload()
        self.assertNotIn(builder.SHEET_ERROR_KEY,state)
        state.pop(builder.SHEET_UPLOAD_KEY)
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow']='Templates'
        app.session_state['templates:selected_type']='2. Price Strips'
        for key,value in state.items():
            app.session_state[key]=value
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any('Legacy' in warning.value for warning in app.warning))
        for pog,size,last in [('Test 10','generic_kk250133_long','0011110'),('Test 8','generic_kk250133_short','0011108')]:
            if app.session_state[builder.SHEET_KEY]['current']!=pog:
                next(w for w in app.selectbox if w.label=='POG from sheet').select(pog).run()
            next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            with pymupdf.open(stream=app.session_state[builder.PDF_KEY]['bytes'],filetype='pdf') as pdf:
                self.assertEqual(len(pdf),1)
                self.assertAlmostEqual(pdf[0].rect.width,geometry_for_size(size).width_pt+36,delta=.5)
                self.assertIn(last,pdf[0].get_text())
