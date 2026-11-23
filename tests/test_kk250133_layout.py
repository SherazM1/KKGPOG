"""Short strips use the available space for large Holiday-style typography."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.sams_club import render_price_strips_html as renderer
from app.shared.async_runtime import run_browser_job
from app.templates.eps_preview import parse_eps,strip_crosshair_layout,widest_eps_card_width
from tests.test_kk250133_import import workbook


def make_row(size,price='18.48',empty=False):
    tickets=builder._default_tickets(1,size)
    for i,ticket in enumerate(tickets):
        ticket.update({'Brand':'MORNING MILL','Pack / value':'PANCAKE MIX','Type':'32 OZ',
                       'Price':price,'Item #':f'8800000{i+1:02d}'})
    if empty:
        tickets[2]={key:'' for key in tickets[2]}
    return builder._build_strip('Layout test',1,1,tickets,size)


class Kk250133LayoutTests(unittest.TestCase):
    def test_large_prices_and_blank_description_lines_remain_inside_the_strip(self):
        for size in ('generic_kk250133_short','generic_kk250133_long'):
            tickets=builder._default_tickets(1,size)
            for ticket in tickets:
                ticket.update({'Brand':'','Pack / value':'','Type':'','Price':'99999.99','Item #':'001234567890'})
            result=builder._create_pdf('Large price',1,1,[tickets],strip_size=size)
            with pymupdf.open(stream=result,filetype='pdf') as pdf:
                self.assertIn('001234567890',pdf[0].get_text())
                self.assertIn('99999',pdf[0].get_text())
                spans=[span for block in pdf[0].get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line['spans']]
                self.assertTrue(all(pdf[0].rect.contains(pymupdf.Rect(span['bbox'])) for span in spans))

    def test_full_size_type_fits_without_collisions_and_tracks_eps_centers(self):
        async def measure():
            results=[]
            drawing=parse_eps(builder.KK250133_EPS_PATH.read_bytes())
            async with renderer.async_playwright() as pw:
                browser=await pw.chromium.launch()
                try:
                    page=await browser.new_page()
                    for size in ('generic_kk250133_short','generic_kk250133_long'):
                        for eps in (False,True):
                            row=make_row(size,empty=True)
                            if eps:
                                row=replace(row,eps_card_centers=strip_crosshair_layout(drawing,size),
                                            eps_card_width_pt=widest_eps_card_width(drawing)[0])
                            content=renderer._build_full_html([row],[],'Generic',False)[0]
                            await page.set_content(content)
                            await renderer.require_gibson_fonts(page)
                            await renderer.fit_compact_tickets(page)
                            results.append(await page.evaluate('''() => {
                                const trim=document.querySelector('.trim-artwork').getBoundingClientRect();
                                const unit=trim.width/Number(document.querySelector('.compact-ticket').dataset.stripWidth);
                                const ink=e=>{const r=document.createRange();r.selectNodeContents(e);const b=r.getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom};};
                                return [...document.querySelectorAll('.compact-ticket')].map(ticket=>{
                                    const price=ticket.querySelector('.price');
                                    return {text:parseFloat(getComputedStyle(ticket.querySelector('.brand-field')).fontSize)*.75,
                                            description:parseFloat(getComputedStyle(ticket.querySelector('.desc-field')).fontSize)*.75,
                                            sign:parseFloat(getComputedStyle(price.querySelector('.dollar-sign')).fontSize)*.75,
                                            cents:parseFloat(getComputedStyle(price.querySelector('.cents')).fontSize)*.75,
                                            itemFont:parseFloat(getComputedStyle(ticket.querySelector('.item-field')).fontSize)*.75,
                                            footerFont:parseFloat(getComputedStyle(document.querySelector('.footer')).fontSize)*.75,
                                            dollars:parseFloat(getComputedStyle(price.querySelector('.dollars')).fontSize)*.75,
                                            center:(price.getBoundingClientRect().left+price.getBoundingClientRect().right)/2,
                                            expected:trim.left+Number(ticket.dataset.anchorX)*unit,
                                            left:trim.left+Number(ticket.dataset.contentLeft)*unit,
                                            right:trim.left+Number(ticket.dataset.contentRight)*unit,
                                            footer:document.querySelector('.footer').getBoundingClientRect().top,
                                            textBoxes:[...ticket.querySelectorAll('.brand-field,.desc-field')].map(ink),
                                            priceBoxes:[...price.querySelectorAll('span')].map(ink),
                                            item:ink(ticket.querySelector('.item-field'))};
                                });
                            }'''))
                finally:
                    await browser.close()
            return results
        results=run_browser_job(measure)
        for entries,expected_count in zip(results,[7,7,9,9]):
            self.assertEqual(len(entries),expected_count)
            for item in entries:
                self.assertAlmostEqual(item['text'],20,delta=.1)
                self.assertAlmostEqual(item['description'],16,delta=.1)
                self.assertAlmostEqual(item['sign'],40,delta=.1)
                self.assertAlmostEqual(item['cents'],44,delta=.1)
                # Card-center bounds may require a small reduction beside the price.
                self.assertGreater(item['itemFont'],8)
                self.assertLessEqual(item['itemFont'],12.1)
                self.assertAlmostEqual(item['footerFont'],10,delta=.1)
                self.assertGreater(item['dollars'],75)
                self.assertLessEqual(item['dollars'],90)
                self.assertAlmostEqual(item['center'],item['expected'],delta=.1)
                text=item['textBoxes']
                self.assertLessEqual(text[0]['bottom'],text[1]['top']+.1)
                for box in item['priceBoxes']:
                    self.assertGreaterEqual(box['top'],text[-1]['bottom']-.1)
                    self.assertLessEqual(box['bottom'],item['footer']+.1)
                    self.assertGreaterEqual(box['left'],item['left']-.1)
                    self.assertLessEqual(box['right'],item['right']+.1)
                    overlaps=(box['left']<item['item']['right'] and box['right']>item['item']['left']
                              and box['top']<item['item']['bottom'] and box['bottom']>item['item']['top'])
                    self.assertFalse(overlaps)

    def test_compact_layout_applies_only_to_the_eight_ten_template(self):
        for size in ('thin','generic_11','generic_two_card'):
            content=renderer._build_full_html([make_row(size)],[],"Sam's Holiday" if size=='thin' else 'Generic',False)[0]
            self.assertNotIn('class="ticket compact-ticket"',content)

    def test_ui_has_no_font_controls_and_combines_native_size_pages(self):
        payload=workbook([('Eight',1,1,8),('Ten',1,1,10)])
        state={builder.FORMAT_KEY:'kk250133',builder.SHEET_UPLOAD_KEY:SimpleNamespace(name='both.xlsx',getvalue=lambda:payload)}
        with patch.object(builder.st,'session_state',state):
            builder._load_sheet_upload()
        state.pop(builder.SHEET_UPLOAD_KEY)
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=20)
        app.session_state['active_workflow']='Templates'
        app.session_state['templates:selected_type']='2. Price Strips'
        for key,value in state.items():
            app.session_state[key]=value
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.number_input)
        self.assertFalse(any('font options' in w.label for w in app.expander))
        self.assertFalse(any(w.label=='Include all POGs in this PDF' for w in app.checkbox))
        next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        with pymupdf.open(stream=app.session_state[builder.PDF_KEY]['bytes'],filetype='pdf') as doc:
            self.assertEqual(len(doc),2)
            for page,width in zip(doc,[30.6875,38.9375]):
                self.assertAlmostEqual(page.rect.width,width*72+36,delta=.5)
                self.assertAlmostEqual(page.rect.height,1.9375*72+36,delta=.5)
                self.assertGreater(len(page.get_text()),150)
