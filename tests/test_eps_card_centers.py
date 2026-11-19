"""New EPS artwork, rather than fixed pitches, aligns prices over their cards."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.pages import price_strip_builder as builder
from app.sams_club import render_price_strips_html as renderer
from app.shared.async_runtime import run_browser_job
from app.templates.eps_preview import parse_eps, strip_crosshair_layout, widest_eps_card_width, render_eps_preview


def artwork(offset=(0,0), missing=False, card_boxes=False):
    dx,dy = offset
    parts = ['%!PS-Adobe-3.0', '%%Creator: Adobe_Illustrator',
             f'%%BoundingBox: {-200+dx} {-200+dy} {3400+dx} {1400+dy}', '%%EndSetup']
    for x,y,width,count in ((100,600,2178,6), (100,100,2767.5,8)):
        x,y = x+dx,y+dy
        parts.extend(['0 0 0 1 (Artios_Cut) 0 X', f'{x} {y} m', f'{x+width} {y} L',
                      f'{x+width} {y+247.5} L', f'{x} {y+247.5} L', f'{x} {y} L', 'S', '1 0 1 0 X'])
        # Nonuniform centers and varying vertical card marks exercise the
        # distinction between card centers and the text's vertical placement.
        for column in range(count-int(missing)):
            cx = x+140+column*(width-280)/(count-1)
            cy = y+90+(column%2)*40
            parts.extend([f'{cx-9} {cy} m', f'{cx+9} {cy} L', 'S',
                          f'{cx} {cy-9} m', f'{cx} {cy+9} L', 'S'])
            if card_boxes:
                half = (144 if column == count-1 else 136.8)
                parts.extend(['1 0 0 0 X', f'{cx-half} {cy-80} m', f'{cx+half} {cy-80} L',
                              f'{cx+half} {cy+80} L', f'{cx-half} {cy+80} L',
                              f'{cx-half} {cy-80} L','S','1 0 1 0 X'])
    parts.append('%%PageTrailer')
    return '\n'.join(parts).encode()


class EpsCardCenterTests(unittest.TestCase):
    def test_one_wider_card_sets_the_width_for_all_rows_even_when_its_product_is_empty(self):
        data = artwork(card_boxes=True)
        self.assertEqual(widest_eps_card_width(parse_eps(data)),(288,'card outlines'))
        rows = []
        for size in ('thin','wide'):
            tickets = builder._default_tickets(1,size)
            tickets[-1] = {key:'' for key in tickets[-1]}
            rows.append(builder._build_strip('Width',1,1,tickets,size))
        with patch.object(renderer,'render_sams_price_strips_pdf',return_value=SimpleNamespace(pdf_bytes=b'pdf',warnings=[])) as render:
            builder._render_strips(rows,eps_data=data)
        rendered = render.call_args.args[0]
        self.assertEqual([row.eps_card_width_pt for row in rendered],[288,288])
        for row in rendered:
            content = renderer._build_full_html([row],[],"Sam's Holiday",True)[0]
            self.assertIn('data-reference-width="288.0"',content)
            self.assertNotIn('#00a633',content)  # Card marks belong to the EPS row below.

    def test_crosshair_spacing_supplies_width_when_eps_has_no_card_outlines(self):
        width,source = widest_eps_card_width(parse_eps(artwork()))
        self.assertEqual(source,'crosshair spacing')
        self.assertAlmostEqual(width,(2178-280)/5,places=3)

    def test_comparison_places_price_above_eps_and_aligns_the_red_line_with_green_mark(self):
        import io
        from PIL import Image
        from tests.test_eps_preview import print_pdf
        data = artwork()
        original_pdf = print_pdf()
        _,detail = render_eps_preview(data,'thin',original_pdf,price_above=True)
        image = Image.open(io.BytesIO(detail)).convert('RGB')
        unit = image.width/(2178+36)
        self.assertAlmostEqual(image.height/unit,2*247.5+12+36,delta=2)
        x = round((18+140)*unit)
        cy = round((18+247.5+12+(247.5-90))*unit)
        nearby = [image.getpixel((px,py)) for px in range(x-2,x+3) for py in range(cy-2,cy+3)]
        self.assertTrue(any(g>100 and g>r*1.5 and g>b*1.5 for r,g,b in nearby))
        guide = image.crop((x-2,round(100*unit),x+3,round(400*unit)))
        self.assertTrue(any(r>150 and g<120 and b<120 for r,g,b in guide.getdata()))
        top = image.crop((0,0,image.width,round((18+247.5)*unit)))
        bottom = image.crop((0,round((18+247.5+12)*unit),image.width,image.height))
        self.assertTrue(any(max(rgb)<100 for rgb in top.getdata()))  # Printed product text is above.
        self.assertFalse(any(max(rgb)<100 for rgb in bottom.getdata()))
        # Rendering a comparison never replaces or enlarges the printable PDF.
        import pymupdf
        with pymupdf.open(stream=original_pdf,filetype='pdf') as doc:
            self.assertEqual(tuple(doc[0].rect),(0,0,2214,283.5))

    def test_moving_the_whole_drawing_does_not_change_relative_card_centers(self):
        original = parse_eps(artwork())
        moved = parse_eps(artwork((-1200,2400)))
        for size,count in (('thin',6),('wide',8)):
            centers = strip_crosshair_layout(original,size)
            shifted = strip_crosshair_layout(moved,size)
            self.assertEqual(len(centers),count)
            for a,b in zip(centers,shifted):
                self.assertAlmostEqual(a[0],b[0],places=3)
                self.assertAlmostEqual(a[1],b[1],places=3)
            self.assertAlmostEqual(centers[0][0],140,places=3)

    def test_missing_mark_prevents_a_silent_return_to_fixed_spacing(self):
        with self.assertRaisesRegex(ValueError,'needs 6 green crosshairs.*5 were found'):
            strip_crosshair_layout(parse_eps(artwork(missing=True)),'thin')

    def test_plain_green_lines_are_not_crosshairs(self):
        data = artwork().replace(b'%%PageTrailer', b'1 0 1 0 X\n100 700 m\n2200 700 L\nS\n%%PageTrailer')
        self.assertEqual(len(strip_crosshair_layout(parse_eps(data),'thin')),6)

    def test_dark_green_marks_are_recognized(self):
        data = artwork().replace(b'1 0 1 0 X',b'1 0 1 .75 X')
        self.assertEqual(len(strip_crosshair_layout(parse_eps(data),'thin')),6)

    def test_repeated_generic_outlines_can_be_reused_for_more_spreadsheet_rows(self):
        drawing = parse_eps(builder.GENERIC_EPS_PATH.read_bytes())
        first = strip_crosshair_layout(drawing,'generic_11',1)
        extra = strip_crosshair_layout(drawing,'generic_11',20)
        for a,b in zip(first,extra):
            self.assertAlmostEqual(a[0],b[0],delta=.05)
            self.assertAlmostEqual(a[1],b[1],delta=.05)

    def test_preview_and_download_receive_eps_centers_without_mutating_product_data(self):
        strip = builder._build_strip('Original',1,1,builder._default_tickets(1,'thin'),'thin')
        original = repr(strip)
        result = SimpleNamespace(pdf_bytes=b'pdf',warnings=[])
        with patch.object(renderer,'render_sams_price_strips_pdf',return_value=result) as render:
            builder._render_strips([strip],eps_data=artwork())
        self.assertEqual(render.call_args.args[0][0].eps_card_centers,
                         strip_crosshair_layout(parse_eps(artwork()),'thin'))
        self.assertEqual(repr(strip),original)

    def test_artwork_uploaded_before_generic_workbook_is_retained(self):
        from tests.test_generic_price_strips import workbook
        custom = {'name':'chosen.eps','bytes':artwork()}
        spacing = {'name':'card-spacing.eps','bytes':artwork((50,100))}
        state = {builder.EPS_KEY:custom, builder.CARD_SPACING_KEY:spacing, builder.SHEET_UPLOAD_KEY:
                 SimpleNamespace(name='Generic.xlsx',getvalue=workbook)}
        with patch.object(builder.st,'session_state',state):
            builder._load_sheet_upload()
        self.assertEqual(state[builder.FORMAT_KEY],'generic')
        self.assertEqual(state[builder.EPS_KEY],custom)
        self.assertEqual(state[builder.CARD_SPACING_KEY],spacing)

    def test_new_card_spacing_invalidates_download_and_is_used_for_preview_and_export(self):
        import io
        from pathlib import Path
        from PIL import Image
        from streamlit.testing.v1 import AppTest
        from tests.test_price_strip_sheet import upload_state
        from tests.test_eps_preview import print_pdf
        blank = io.BytesIO()
        Image.new('RGB',(10,10),'white').save(blank,format='PNG')
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '2. Price Strips'
        for key,value in upload_state().items():
            app.session_state[key] = value
        app.run()
        with patch.object(builder,'_live_strip_preview',return_value=(blank.getvalue(),print_pdf())) as preview, \
                patch.object(builder,'_create_all_sides_pdf',return_value=b'pdf') as export:
            next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
            self.assertIn(builder.PDF_KEY,app.session_state)
            data = artwork()
            app.session_state[builder.CARD_SPACING_KEY] = {'name':'new.eps','bytes':data}
            app.run()
            self.assertNotIn(builder.PDF_KEY,app.session_state)
            self.assertEqual(preview.call_args.kwargs['eps_data'],data)
            next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
            self.assertEqual(export.call_args.kwargs['eps_data'],data)
            self.assertFalse(app.exception)

    def test_loaded_price_group_centers_match_the_actual_marks_and_empty_slots_do_not_shift_them(self):
        tickets = builder._default_tickets(1,'thin')
        tickets[2] = {key:'' for key in tickets[2]}
        row = builder._build_strip('Centers',1,1,tickets,'thin')
        centers = strip_crosshair_layout(parse_eps(artwork()),'thin')
        row = replace(row,eps_card_centers=centers,eps_card_width_pt=288)
        content = renderer._build_full_html([row],[],"Sam's Holiday",True)[0]
        async def measure():
            async with renderer.async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                try:
                    page = await browser.new_page()
                    await page.set_content(content)
                    await renderer.require_gibson_fonts(page)
                    await renderer.fit_eps_tickets(page)
                    return await page.evaluate('''() => {
                        const trim = document.querySelector('.trim-artwork').getBoundingClientRect();
                        return [...document.querySelectorAll('.eps-ticket')].map(ticket => {
                            const price = ticket.querySelector('.price').getBoundingClientRect();
                            const unit = trim.width/Number(ticket.dataset.stripWidth);
                            return {expected:Number(ticket.dataset.anchorX), actual:((price.left+price.right)/2-trim.left)/unit,
                                    top:(price.top-trim.top)/unit,
                                    width:ticket.getBoundingClientRect().width/unit};
                        });
                    }''')
                finally:
                    await browser.close()
        measured = run_browser_job(measure)
        self.assertEqual(len(measured),5)
        self.assertEqual([round(item['expected'],3) for item in measured],
                         [round(center[0],3) for i,center in enumerate(centers) if i!=2])
        for item in measured:
            self.assertAlmostEqual(item['actual'],item['expected'],delta=.05)
            self.assertAlmostEqual(item['width'],288,delta=.05)
        # Card-center height never becomes the vertical center of the label.
        self.assertLess(max(item['top'] for item in measured)-min(item['top'] for item in measured),.05)
