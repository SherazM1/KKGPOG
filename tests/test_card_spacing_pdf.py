"""Vector PDF card spacing uses the same geometry as the EPS equivalent."""
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf
from PIL import Image
from reportlab.pdfgen import canvas

from app.pages import price_strip_builder as builder
from app.sams_club import render_price_strips_html as renderer
from app.templates.eps_preview import (parse_card_spacing, strip_crosshair_layout,
                                      widest_eps_card_width, render_eps_preview, card_spacing_overflow, combine_card_spacing)
from tests.test_eps_card_centers import artwork
from tests.test_eps_preview import print_pdf


def spacing_pdf(*, pages=1, marks=True, boxes=False):
    drawing = parse_card_spacing(artwork(card_boxes=boxes))
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream,pagesize=(3600,1600))
    for index in range(pages):
        pdf.saveState()
        pdf.translate(200,-100)
        for commands,color,width,_ in drawing.paths:
            if not marks and color==(0,1,0):
                continue
            path = pdf.beginPath()
            for op,args in commands:
                {'m':path.moveTo,'L':path.lineTo,'c':path.curveTo}[op](*args)
            pdf.setStrokeColorRGB(*color)
            pdf.setLineWidth(width)
            pdf.drawPath(path,stroke=1,fill=0)
        pdf.restoreState()
        pdf.drawString(30,1550,f'Card organization page {index+1}')
        pdf.showPage()
    pdf.save()
    return stream.getvalue()


def top_cross_pdf(count=6,card_width=3.8,gap=1,inset=1.1):
    strip_width = 2178 if count==6 else 2767.5
    required = (count*card_width+(count-1)*gap)*72
    extra = 45.45 if required>strip_width else 0
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream,pagesize=(strip_width+2*extra,247.5))
    pdf.rect(extra+inset,inset,strip_width-2*inset,247.5-2*inset)
    first = extra+(strip_width-required)/2+card_width*36
    centers = [first+i*(card_width+gap)*72 for i in range(count)]
    for x in centers:
        pdf.setStrokeColorRGB(0,.49,.72)
        pdf.roundRect(x-card_width*36,41.76,card_width*72,167.04,4.5)
        pdf.rect(x-19.8,105.48,39.6,39.6)
        pdf.setStrokeColorRGB(0,.65,.318)
        pdf.line(x-7.56,208.8,x+7.56,208.8)
        pdf.line(x,201.24,x,216.36)
    pdf.showPage()
    pdf.save()
    return stream.getvalue(),[x-extra for x in centers]


class CardSpacingPdfTests(unittest.TestCase):
    def test_combined_pdf_layouts_preserve_centers_in_either_upload_order(self):
        sources = [{'name':f'{count}.pdf','bytes':top_cross_pdf(count=count)[0]} for count in (6,8)]
        for uploads in (sources,sources[::-1]):
            combined = parse_card_spacing(combine_card_spacing(uploads))
            for source,size in zip(sources,('thin','wide')):
                expected = strip_crosshair_layout(parse_card_spacing(source['bytes']),size)
                actual = strip_crosshair_layout(combined,size)
                for (a,b),(x,y) in zip(actual,expected):
                    self.assertAlmostEqual(a,x,delta=.005)
                    self.assertAlmostEqual(b,y,delta=.005)

    def test_eps_and_pdf_files_can_be_uploaded_together(self):
        combined = parse_card_spacing(combine_card_spacing([
            {'name':'generic.eps','bytes':builder.GENERIC_EPS_PATH.read_bytes()},
            {'name':'holiday.pdf','bytes':top_cross_pdf(count=8)[0]}]))
        self.assertEqual(len(strip_crosshair_layout(combined,'generic_11')),3)
        self.assertEqual(len(strip_crosshair_layout(combined,'wide')),8)

    def test_inset_border_and_rounded_cards_with_top_marks_keep_physical_positions(self):
        for count,width,gap in [(6,3.8,1),(6,4,1),(8,3.8,1),(8,4,.85),(8,4,1)]:
            with self.subTest(count=count,width=width,gap=gap):
                data,expected = top_cross_pdf(count,width,gap)
                drawing = parse_card_spacing(data)
                size = 'thin' if count==6 else 'wide'
                centers = strip_crosshair_layout(drawing,size,20)
                self.assertEqual(len(centers),count)
                for (x,y),wanted in zip(centers,expected):
                    self.assertAlmostEqual(x,wanted,delta=.005)
                    self.assertAlmostEqual(y,38.7,delta=.005)
                actual,source = widest_eps_card_width(drawing)
                self.assertEqual(source,'card outlines')
                self.assertAlmostEqual(actual,width*72,delta=.005)
                overhang = card_spacing_overflow(drawing,size)
                expected_overhang = 20.25 if count==8 and width==4 and gap==1 else 0
                for value in overhang:
                    self.assertAlmostEqual(value,expected_overhang,delta=.005)

    def test_a_substantially_wrong_border_is_not_rescaled_to_fit(self):
        data,_ = top_cross_pdf(inset=10)
        with self.assertRaisesRegex(ValueError,'matching the selected dimensions'):
            strip_crosshair_layout(parse_card_spacing(data),'thin')

    def test_pdf_and_eps_return_the_same_centers_and_widest_card_width(self):
        eps = parse_card_spacing(artwork(card_boxes=True))
        drawing = parse_card_spacing(spacing_pdf(boxes=True))
        for size in ('thin','wide'):
            for a,b in zip(strip_crosshair_layout(eps,size),strip_crosshair_layout(drawing,size)):
                for x,y in zip(a,b):
                    self.assertAlmostEqual(x,y,delta=.005)
        self.assertEqual(widest_eps_card_width(drawing),(288,'card outlines'))

    def test_all_pdf_pages_are_read_in_order_and_keep_original_artwork(self):
        data = spacing_pdf(pages=2)
        drawing = parse_card_spacing(data)
        self.assertEqual(len(drawing.cuts),4)
        self.assertEqual(strip_crosshair_layout(drawing,'wide',1),strip_crosshair_layout(drawing,'wide',2))
        with pymupdf.open(stream=drawing.preview_pdf,filetype='pdf') as doc:
            text = doc[0].get_text()
            self.assertIn('Card organization page 1',text)
            self.assertIn('Card organization page 2',text)

    def test_rotated_pdf_keeps_marks_in_the_displayed_coordinate_system(self):
        with pymupdf.open(stream=spacing_pdf(),filetype='pdf') as doc:
            doc[0].set_rotation(180)
            rotated = doc.tobytes()
        drawing = parse_card_spacing(rotated)
        original = strip_crosshair_layout(parse_card_spacing(spacing_pdf()),'thin')
        expected = [(2178-x,247.5-y) for x,y in reversed(original)]
        for a,b in zip(strip_crosshair_layout(drawing,'thin'),expected):
            for x,y in zip(a,b):
                self.assertAlmostEqual(x,y,delta=.005)

    def test_pdf_upload_reaches_the_preview_and_download_renderer(self):
        data = spacing_pdf()
        row = builder._build_strip('PDF spacing',1,1,builder._default_tickets(1),'thin')
        with patch.object(renderer,'render_sams_price_strips_pdf',return_value=SimpleNamespace(pdf_bytes=b'pdf',warnings=[])) as render:
            self.assertEqual(builder._render_strips([row],eps_data=data),b'pdf')
        rendered = render.call_args.args[0][0]
        self.assertEqual(rendered.eps_card_centers,strip_crosshair_layout(parse_card_spacing(data),'thin'))
        self.assertAlmostEqual(rendered.eps_card_width_pt,(2178-280)/5,delta=.005)
        self.assertEqual(row.eps_card_centers,())
        state = {builder.CARD_SPACING_UPLOAD_KEY:SimpleNamespace(name='cards.pdf',getvalue=lambda:data)}
        with patch.object(builder.st,'session_state',state):
            builder._save_card_spacing_upload()
        self.assertEqual(state[builder.CARD_SPACING_KEY],{'name':'cards.pdf','bytes':data})

    def test_pdf_comparison_keeps_the_price_strip_above_the_card_marks(self):
        overview,detail = render_eps_preview(spacing_pdf(),'thin',print_pdf(),price_above=True)
        self.assertTrue(overview.startswith(b'\x89PNG'))
        image = Image.open(io.BytesIO(detail)).convert('RGB')
        unit = image.width/2214
        self.assertAlmostEqual(image.height/unit,543,delta=2)
        x = round((18+140)*unit)
        y = round((18+247.5+12+247.5-90)*unit)
        self.assertTrue(any(g>100 and g>r*1.5 and g>b*1.5
                            for r,g,b in [image.getpixel((a,b)) for a in range(x-2,x+3) for b in range(y-2,y+3)]))

    def test_missing_vector_marks_invalid_encrypted_and_oversize_pdfs_report_errors(self):
        with self.assertRaisesRegex(ValueError,'green vector crosshairs'):
            parse_card_spacing(spacing_pdf(marks=False))
        with self.assertRaisesRegex(ValueError,'could not be opened'):
            parse_card_spacing(b'%PDF-1.4\ninvalid')
        with self.assertRaisesRegex(ValueError,'10 MB'):
            parse_card_spacing(b'%PDF-1.4\n'+b'0'*(10*1024*1024))
        with pymupdf.open(stream=spacing_pdf(),filetype='pdf') as doc:
            locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='secret')
        with self.assertRaisesRegex(ValueError,'password protection'):
            parse_card_spacing(locked)
