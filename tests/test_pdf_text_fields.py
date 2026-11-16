"""Description and CPP are required; the third PDF line remains selectable."""
from pathlib import Path
import unittest

import fitz
from streamlit.testing.v1 import AppTest

from app.pages.template_builder import (DRAFT_KEY, MODE_KEY, PDF_FIELDS_KEY, PDF_KEY,
                                         PDF_LAYOUT_VERSION, PDF_READY_KEY)
from app.templates.print_style import (DEFAULT_TEXT_FIELDS, PDF_IDENTIFIER_FIELDS,
                                       PDF_TEXT_FIELDS, normalize_text_fields)
from app.templates.render_pdf import render_template_pdf
from tests.test_pdf_sections import section_layout


ROOT = Path(__file__).resolve().parents[1]


class PdfTextFieldTests(unittest.TestCase):
    def test_default_lines_stay_description_cpp_and_upc(self):
        draft = section_layout()
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as doc:
            lines = doc[0].get_text().splitlines()
            for field in DEFAULT_TEXT_FIELDS:
                self.assertEqual(lines.count(f'{field}:'), 24)
            self.assertNotIn('ID#:', lines)
            self.assertNotIn('Item #:', lines)

    def test_each_identifier_keeps_description_and_cpp_inside_every_box(self):
        template = section_layout().template
        for identifier in PDF_IDENTIFIER_FIELDS:
            with self.subTest(identifier=identifier):
                with fitz.open(stream=render_template_pdf(template, text_fields=identifier),
                               filetype='pdf') as doc:
                    page = doc[0]
                    rectangles = [d['rect'] for d in page.get_drawings() if d['fill'] is None
                                  and any(item[0] == 're' for item in d['items'])]
                    self.assertEqual(len(rectangles), 48)
                    for outer, image in zip(rectangles[::2], rectangles[1::2]):
                        lines = page.get_text(clip=outer).splitlines()
                        for field in PDF_TEXT_FIELDS:
                            expected = field in ('Description', 'CPP', identifier)
                            self.assertEqual(lines.count(f'{field}:'), int(expected))
                        self.assertFalse(page.get_text(clip=image).strip())

    def test_empty_and_legacy_preferences_keep_mandatory_lines(self):
        self.assertEqual(normalize_text_fields(None), DEFAULT_TEXT_FIELDS)
        self.assertEqual(normalize_text_fields([]), DEFAULT_TEXT_FIELDS)
        self.assertEqual(normalize_text_fields(['ID', 'CPP', 'ID']), ('Description', 'CPP', 'ID#'))
        self.assertEqual(normalize_text_fields('Item #'), ('Description', 'CPP', 'Item #'))
        for invalid in ('Unknown', ['Unknown'], 123):
            with self.assertRaises(ValueError):
                normalize_text_fields(invalid)


class PdfTextFieldAppTests(unittest.TestCase):
    def start(self, legacy_fields=None):
        draft = section_layout()
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = draft
        app.session_state[PDF_READY_KEY] = (draft.template.id, draft.revision)
        if legacy_fields is not None:
            app.session_state[f'{PDF_FIELDS_KEY}:{draft.template.id}'] = legacy_fields
        app.run()
        self.assertFalse(app.exception)
        return app, draft

    def click(self, app, label):
        next(w for w in app.button if w.label == label).click().run()
        self.assertFalse(app.exception)

    def test_defaults_and_changes_invalidate_the_download_without_editing_the_draft(self):
        app, draft = self.start()
        selector = next(w for w in app.selectbox if w.label == 'Third PDF text line')
        self.assertEqual(selector.value, 'UPC')
        self.click(app, 'Done')
        self.click(app, 'Create Template PDF')
        self.assertTrue(app.get('download_button'))
        self.assertRegex(app.session_state[PDF_KEY]['download_stamp'], r'^\d{8}_\d{6}$')
        self.assertTrue(any('PDF prepared ' in c.value for c in app.caption))
        self.click(app, 'Resume draft')
        next(w for w in app.selectbox if w.label == 'Third PDF text line').select('ID#').run()
        self.assertFalse(app.get('download_button'))
        self.assertNotIn(PDF_KEY, app.session_state)
        self.assertEqual(app.session_state[DRAFT_KEY], draft)
        self.click(app, 'Done')
        self.click(app, 'Create Template PDF')
        with fitz.open(stream=app.session_state[PDF_KEY]['pdf'], filetype='pdf') as doc:
            for field in ('Description', 'CPP', 'ID#'):
                self.assertEqual(doc[0].get_text().splitlines().count(f'{field}:'), 24)
            self.assertNotIn('UPC:', doc[0].get_text())

    def test_selection_survives_resuming_the_draft_and_item_number_exports(self):
        app, _ = self.start()
        next(w for w in app.selectbox if w.label == 'Third PDF text line').select('Item #').run()
        self.click(app, 'Done')
        self.assertFalse(any(w.label == 'Third PDF text line' for w in app.selectbox))
        self.click(app, 'Resume draft')
        selector = next(w for w in app.selectbox if w.label == 'Third PDF text line')
        self.assertEqual(selector.value, 'Item #')
        self.click(app, 'Done')
        self.click(app, 'Create Template PDF')
        with fitz.open(stream=app.session_state[PDF_KEY]['pdf'], filetype='pdf') as doc:
            for field in ('Description', 'CPP', 'Item #'):
                self.assertEqual(doc[0].get_text().splitlines().count(f'{field}:'), 24)

    def test_old_optional_id_selection_migrates_to_id_number(self):
        app, _ = self.start(legacy_fields=('ID',))
        selector = next(w for w in app.selectbox if w.label == 'Third PDF text line')
        self.assertEqual(selector.value, 'ID#')
        self.assertFalse(app.multiselect)

    def test_old_layout_download_is_removed_before_regenerating(self):
        app, draft = self.start()
        self.click(app, 'Done')
        app.session_state[PDF_KEY] = dict(
            signature=(draft.template.id, draft.revision, PDF_LAYOUT_VERSION - 1, DEFAULT_TEXT_FIELDS),
            pdf=b'%PDF-old-layout',
        )
        app.run()
        self.assertNotIn(PDF_KEY, app.session_state)
        self.assertFalse(app.get('download_button'))
        self.click(app, 'Create Template PDF')
        self.assertTrue(app.get('download_button'))
