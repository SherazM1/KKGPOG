"""A separate card organization EPS controls alignment, independently of strip artwork."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from tests.test_eps_card_centers import artwork
from tests.test_eps_preview import example_eps, print_pdf
from tests.test_price_strip_sheet import upload_state


class CardSpacingUploadTests(unittest.TestCase):
    def test_two_spacing_uploads_generate_one_all_sides_pdf_and_both_previews(self):
        import pymupdf
        from tests.test_card_spacing_pdf import top_cross_pdf
        uploads = [SimpleNamespace(name=f'{count}-cards.pdf',getvalue=lambda data=top_cross_pdf(count=count)[0]:data)
                   for count in (6,8)]
        state = {builder.CARD_SPACING_UPLOAD_KEY:uploads}
        with patch.object(builder.st,'session_state',state):
            builder._save_card_spacing_upload()
        self.assertNotIn(builder.CARD_SPACING_ERROR_KEY,state)
        combined = state[builder.CARD_SPACING_KEY]
        self.assertEqual(combined['name'],'6-cards.pdf, 8-cards.pdf')
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '2. Price Strips'
        for key,value in upload_state().items():
            app.session_state[key] = value
        app.session_state[builder.CARD_SPACING_KEY] = combined
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(any(w.label=='PDF output' for w in app.radio))
        self.assertEqual(len(next(w for w in app.selectbox if w.label=='Preview strip').options),3)
        next(w for w in app.selectbox if w.label=='Preview strip').select(2).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
        self.assertFalse(app.exception)
        result = app.session_state[builder.PDF_KEY]
        self.assertEqual((result['scope'],result['pages']),('all sides',3))
        with pymupdf.open(stream=result['bytes'],filetype='pdf') as pdf:
            self.assertEqual(len(pdf),3)
            self.assertEqual([round(page.rect.width,1) for page in pdf],[2214,2214,2803.9])
            for page,(side,row) in zip(pdf,[(1,2),(1,4),(2,2)]):
                self.assertIn(f'Custom footer POG A {side} {row}',page.get_text())
        # Removing the eight-card file switches back to explicit partial export.
        partial = {builder.CARD_SPACING_UPLOAD_KEY:uploads[:1]}
        with patch.object(builder.st,'session_state',partial):
            builder._save_card_spacing_upload()
        app.session_state[builder.CARD_SPACING_KEY] = partial[builder.CARD_SPACING_KEY]
        app.run()
        self.assertNotIn(builder.PDF_KEY,app.session_state)
        self.assertTrue(any(w.label=='PDF output' for w in app.radio))

    def test_conflicting_spacing_files_clear_the_old_download_and_report_the_size(self):
        from tests.test_card_spacing_pdf import top_cross_pdf
        state = {builder.PDF_KEY:b'old',builder.CARD_SPACING_KEY:{'name':'old','bytes':b'old'},
                 builder.CARD_SPACING_UPLOAD_KEY:[
                     SimpleNamespace(name='six-a.pdf',getvalue=lambda:top_cross_pdf(count=6,card_width=3.8)[0]),
                     SimpleNamespace(name='six-b.pdf',getvalue=lambda:top_cross_pdf(count=6,card_width=4)[0])]}
        with patch.object(builder.st,'session_state',state):
            builder._save_card_spacing_upload()
        self.assertNotIn(builder.PDF_KEY,state)
        self.assertNotIn(builder.CARD_SPACING_KEY,state)
        self.assertIn('both define the 6-position',state[builder.CARD_SPACING_ERROR_KEY])
        state[builder.CARD_SPACING_UPLOAD_KEY] = []
        with patch.object(builder.st,'session_state',state):
            builder._save_card_spacing_upload()
        self.assertNotIn(builder.CARD_SPACING_ERROR_KEY,state)

    def test_single_size_spacing_generates_matching_strips_and_all_side_export_is_explicitly_blocked(self):
        import pymupdf
        from tests.test_card_spacing_pdf import top_cross_pdf
        for count,expected_side,pages in [(6,1,2),(8,2,1)]:
            with self.subTest(count=count):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
                app.session_state['active_workflow'] = 'Templates'
                app.session_state['templates:selected_type'] = '2. Price Strips'
                for key,value in upload_state().items():
                    app.session_state[key] = value
                data,_ = top_cross_pdf(count=count)
                app.session_state[builder.CARD_SPACING_KEY] = {'name':f'{count}-cards.pdf','bytes':data}
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(next(w for w in app.radio if w.label=='PDF output').value,'matching strips only')
                self.assertTrue(all(f'Side {expected_side},' in option for option in next(w for w in app.selectbox if w.label=='Preview strip').options))
                button = next(w for w in app.button if w.label=='Create Price Strip PDF')
                self.assertFalse(button.disabled)
                button.click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                result = app.session_state[builder.PDF_KEY]
                self.assertEqual(result['pages'],pages)
                self.assertEqual(result['scope'],'matching strips only')
                with pymupdf.open(stream=result['bytes'],filetype='pdf') as doc:
                    self.assertEqual(len(doc),pages)
                    self.assertTrue(all(f'Custom footer POG A {expected_side}' in page.get_text() for page in doc))
                next(w for w in app.radio if w.label=='PDF output').set_value('all sides').run()
                self.assertNotIn(builder.PDF_KEY,app.session_state)
                self.assertTrue(next(w for w in app.button if w.label=='Create Price Strip PDF').disabled)
                self.assertTrue(any('every strip size' in w.value for w in app.warning))

    def test_partial_coverage_does_not_hide_malformed_marks_for_a_supported_size(self):
        from tests.test_price_strip_sheet import imported
        with self.assertRaisesRegex(ValueError,'needs 6 green crosshairs'):
            builder._spacing_compatible_drafts(imported()['pogs']['POG A'],artwork(missing=True))

    def test_invalid_spacing_identifies_the_upload_that_needs_correcting(self):
        with self.assertRaisesRegex(ValueError,'Card Spacing:.*ArtiosCAD'):
            builder._create_pdf('Invalid',1,1,[builder._default_tickets(1)],strip_size='thin',eps_data=b'invalid')

    def test_spacing_upload_and_removal_do_not_replace_the_strip_eps_or_sheet(self):
        strip = {'name':'outline.eps','bytes':example_eps()}
        sheet = {'current':'Original'}
        data = artwork()
        state = {builder.EPS_KEY:strip,builder.SHEET_KEY:sheet,builder.PDF_KEY:b'old',
                 builder.CARD_SPACING_UPLOAD_KEY:SimpleNamespace(name='cards.eps',getvalue=lambda:data)}
        with patch.object(builder.st,'session_state',state):
            builder._save_card_spacing_upload()
            self.assertEqual(state[builder.CARD_SPACING_KEY],{'name':'cards.eps','bytes':data})
            self.assertNotIn(builder.PDF_KEY,state)
            state[builder.CARD_SPACING_UPLOAD_KEY] = None
            state[builder.PDF_KEY] = b'old'
            builder._save_card_spacing_upload()
        self.assertNotIn(builder.CARD_SPACING_KEY,state)
        self.assertNotIn(builder.PDF_KEY,state)
        self.assertIs(state[builder.EPS_KEY],strip)
        self.assertIs(state[builder.SHEET_KEY],sheet)

    def test_format_navigation_restores_each_formats_own_spacing(self):
        first = {'name':'holiday.eps','bytes':artwork()}
        second = {'name':'generic.eps','bytes':b'generic'}
        state = {builder.CARD_SPACING_KEY:first}
        with patch.object(builder.st,'session_state',state):
            builder._switch_format('generic')
            self.assertNotIn(builder.CARD_SPACING_KEY,state)
            state[builder.CARD_SPACING_KEY] = second
            builder._switch_format('holiday')
            self.assertEqual(state[builder.CARD_SPACING_KEY],first)
            builder._switch_format('generic')
            self.assertEqual(state[builder.CARD_SPACING_KEY],second)

    def test_upload_order_and_only_spacing_eps_controls_preview_export_and_guides(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '2. Price Strips'
        for key,value in upload_state().items():
            app.session_state[key] = value
        app.run()
        blank = io.BytesIO()
        Image.new('RGB',(10,10),'white').save(blank,format='PNG')
        strip,spacing = example_eps(),artwork((500,100))
        app.session_state[builder.EPS_KEY] = {'name':'strip.eps','bytes':strip}
        with patch.object(builder,'_live_strip_preview',return_value=(blank.getvalue(),print_pdf())) as preview, \
                patch('app.templates.eps_preview.render_eps_preview',return_value=(blank.getvalue(),blank.getvalue())) as guide, \
                patch.object(builder,'_create_all_sides_pdf',return_value=b'pdf') as export:
            app.run()
            labels = [w.proto.label for w in app.get('file_uploader')]
            self.assertEqual(labels,['Upload a different EPS (optional)','Upload price-strip data (.xlsx)','Card Spacing'])
            spacing_widget = next(w for w in app.get('file_uploader') if w.proto.label=='Card Spacing')
            self.assertEqual(list(spacing_widget.proto.type),['.eps','.pdf'])
            self.assertTrue(spacing_widget.proto.multiple_files)
            self.assertIsNone(preview.call_args.kwargs['eps_data'])
            app.session_state[builder.CARD_SPACING_KEY] = {'name':'cards.eps','bytes':spacing}
            app.run()
            self.assertEqual(preview.call_args.kwargs['eps_data'],spacing)
            paired = [call for call in guide.call_args_list if call.kwargs.get('price_above')]
            self.assertEqual(paired[-1].args[0],spacing)
            self.assertEqual(guide.call_args.args[0],strip)
            next(w for w in app.button if w.label=='Create Price Strip PDF').click().run()
            self.assertEqual(export.call_args.kwargs['eps_data'],spacing)
            self.assertIn(builder.PDF_KEY,app.session_state)
            del app.session_state[builder.CARD_SPACING_KEY]
            app.run()
            self.assertNotIn(builder.PDF_KEY,app.session_state)
            self.assertIsNone(preview.call_args.kwargs['eps_data'])
            app.session_state[builder.CARD_SPACING_KEY] = {'name':'bad.eps','bytes':b'invalid'}
            preview.side_effect = ValueError('Invalid Card Spacing EPS')
            app.run()
            create_buttons = [w for w in app.button if w.label=='Create Price Strip PDF']
            self.assertTrue(all(button.disabled for button in create_buttons))
            self.assertNotIn(builder.PDF_KEY,app.session_state)
            self.assertTrue(any('Card Spacing' in w.value for w in app.error))
            self.assertFalse(app.exception)
