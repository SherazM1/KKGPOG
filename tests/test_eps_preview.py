"""Uploaded engineering outlines and live print previews stay aligned."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.templates.eps_preview import parse_eps, render_eps_preview


ROOT = Path(__file__).resolve().parents[1]


def example_eps(wide=True):
    rectangles = [(100, 600, 2278, 847.5)]
    if wide:
        rectangles.append((100, 100, 2867.5, 347.5))
    paths = []
    for x0, y0, x1, y1 in rectangles:
        paths.append(f'''2 w
0 0 0 1 (Artios_Cut) 0 X
{x0} {y0} m
{x1} {y0} L
{x1} {y1} L
{x0} {y1} L
{x0} {y0} L
S''')
    return ('''%!PS-Adobe-3.0
%%Creator: Adobe_Illustrator
%%BoundingBox: 0 0 3200 1000
%%EndSetup
%AI5_BeginLayer
0 To
1 0 0 1 100 950 0 Tp
1 .3 0 0 k
/_ArialMT 40 Tf
(Drawing label) Tx
''' + '\n'.join(paths) + '\n%AI5_EndLayer--\n').encode()


def print_pdf(width=2178):
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(width + 36, 283.5))
    pdf.drawString(54, 180, 'Printed product')
    pdf.showPage()
    pdf.save()
    return output.getvalue()


class EpsPreviewTests(unittest.TestCase):
    def test_bundled_baseline_covers_both_sizes_and_custom_removal_returns_to_it(self):
        with patch.object(builder.st, 'session_state', {}):
            baseline = builder._active_eps()
            drawing = parse_eps(baseline['bytes'])
        widths = sorted(round(rect[2] - rect[0], 1) for rect in drawing.cuts)
        self.assertEqual(widths, [2178, 2767.5])
        state = {builder.EPS_KEY:{'name':'custom.eps', 'bytes':example_eps()}}
        with patch.object(builder.st, 'session_state', state):
            self.assertEqual(builder._active_eps()['name'], 'custom.eps')
            state[builder.EPS_UPLOAD_KEY] = None
            builder._save_eps_upload()
            self.assertEqual(builder._active_eps(), baseline)

    def test_live_preview_uses_the_passed_products_and_guided_renderer(self):
        builder._live_strip_preview.clear()
        tickets = builder._default_tickets(1, 'thin')
        with patch.object(builder, '_create_pdf', return_value=print_pdf()) as create:
            image, pdf = builder._live_strip_preview('Live', 1, 1, tickets, 'thin')
        self.assertTrue(image.startswith(b'\x89PNG'))
        self.assertTrue(pdf.startswith(b'%PDF'))
        self.assertEqual(create.call_args.args[3], [tickets])
        self.assertTrue(create.call_args.kwargs['calibration'])

    def test_main_preview_uses_renderer_guides_without_changing_download_defaults(self):
        builder._eps_preview.clear()
        tickets = builder._default_tickets(1, 'thin')
        with patch.object(builder, '_create_pdf', return_value=b'pdf') as create, \
                patch('app.templates.eps_preview.render_eps_preview', return_value=(b'full', b'detail')) as render:
            self.assertEqual(builder._eps_preview(example_eps(), 'Guides', 1, 1, tickets, 'thin'),
                             (b'full', b'detail'))
        self.assertTrue(create.call_args.kwargs['calibration'])
        self.assertTrue(render.call_args.kwargs['guides_in_pdf'])
        renderer = SimpleNamespace(render_sams_price_strips_pdf=lambda rows, **kwargs:
            SimpleNamespace(pdf_bytes=b'pdf' if kwargs['calibration'] is False else b'guided', warnings=[]))
        with patch.dict('sys.modules', {'app.sams_club.render_price_strips_html':renderer}):
            self.assertEqual(builder._create_pdf('Clean', 1, 1, [tickets], strip_size='thin'), b'pdf')

    def test_preserves_vector_labels_and_both_cut_sizes(self):
        drawing = parse_eps(example_eps())
        self.assertEqual(len(drawing.cuts), 2)
        self.assertEqual(drawing.texts[0][0], 'Drawing label')
        self.assertEqual(drawing.cuts[0][2] - drawing.cuts[0][0], 2178)
        self.assertEqual(drawing.cuts[1][2] - drawing.cuts[1][0], 2767.5)

    def test_preview_contains_red_cut_blue_dividers_and_printed_data(self):
        overview, detail = render_eps_preview(example_eps(), 'thin', print_pdf())
        self.assertTrue(overview.startswith(b'\x89PNG'))
        image = Image.open(io.BytesIO(detail)).convert('RGB')
        colors = list(image.getdata())
        self.assertTrue(any(r > 150 and g < 80 and b < 100 for r, g, b in colors))
        self.assertTrue(any(b > 150 and r < 100 and g < 150 for r, g, b in colors))
        self.assertTrue(any(max(pixel) < 80 for pixel in colors))

    def test_size_mismatch_and_unsupported_postscript_report_errors(self):
        with self.assertRaisesRegex(ValueError, 'matching the selected dimensions'):
            render_eps_preview(example_eps(wide=False), 'wide', print_pdf(2767.5))
        with self.assertRaisesRegex(ValueError, 'unsupported artwork'):
            parse_eps(example_eps().replace(b'%AI5_BeginLayer', b'%AI5_BeginLayer\n(ignored) run'))
        with self.assertRaisesRegex(ValueError, 'Upload an ArtiosCAD'):
            parse_eps(b'not an EPS')
        with self.assertRaisesRegex(ValueError, '10 MB'):
            parse_eps(b'x' * (10 * 1024 * 1024 + 1))

    def test_upload_and_removal_store_the_actual_file_bytes(self):
        data = example_eps()
        state = {builder.EPS_UPLOAD_KEY: SimpleNamespace(name='drawing.eps', getvalue=lambda: data)}
        with patch.object(builder.st, 'session_state', state):
            builder._save_eps_upload()
            self.assertEqual(state[builder.EPS_KEY], {'name': 'drawing.eps', 'bytes': data})
            state[builder.EPS_UPLOAD_KEY] = None
            builder._save_eps_upload()
            self.assertNotIn(builder.EPS_KEY, state)


class EpsPreviewAppTests(unittest.TestCase):
    def start(self):
        from tests.test_price_strip_sheet import upload_state
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '2. Price Strips'
        for key, value in upload_state().items():
            app.session_state[key] = value
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_preview_tracks_imported_side_and_row_and_survives_navigation(self):
        app = self.start()
        self.assertTrue(any(w.label == 'Full EPS drawing' for w in app.expander))
        self.assertNotIn(builder.PDF_KEY, app.session_state)
        app.session_state[builder.EPS_KEY] = {'name':'drawing.eps', 'bytes':example_eps()}
        blank = io.BytesIO()
        Image.new('RGB', (10, 10), 'white').save(blank, format='PNG')
        with patch.object(builder, '_live_strip_preview', return_value=(blank.getvalue(), print_pdf())) as preview, \
                patch('app.templates.eps_preview.render_eps_preview', return_value=(blank.getvalue(), blank.getvalue())):
            app.run()
            self.assertEqual(preview.call_args.args[-1], 'thin')
            next(w for w in app.selectbox if w.label == 'Preview strip').select(1).run()
            self.assertEqual(preview.call_args.args[2], 4)
            next(w for w in app.selectbox if w.label == 'Preview strip').select(2).run()
            self.assertEqual(preview.call_args.args[1:3], (2, 2))
            self.assertEqual(preview.call_args.args[-1], 'wide')
            next(w for w in app.radio if w.label == 'Template type').set_value('1. POG Template').run()
            next(w for w in app.radio if w.label == 'Template type').set_value('2. Price Strips').run()
            self.assertEqual(app.session_state[builder.EPS_KEY]['name'], 'drawing.eps')
            self.assertFalse(app.exception)

    def test_invalid_eps_blocks_output_without_restoring_edit_controls(self):
        app = self.start()
        app.session_state[builder.EPS_KEY] = {'name':'bad.eps', 'bytes':b'bad'}
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('ArtiosCAD' in w.value for w in app.error))
        self.assertFalse(app.get('data_editor'))
        self.assertTrue(next(b for b in app.button if b.label == 'Create Price Strip PDF').disabled)
