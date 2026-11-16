"""Changed template code refreshes without discarding the user's draft."""
import unittest
from unittest.mock import patch

import home
from app.templates import render_pdf
from tests.test_pdf_sections import section_layout


class TemplateReloadTests(unittest.TestCase):
    def test_stale_renderer_reloads_and_cached_pdf_is_removed_without_losing_draft(self):
        draft = section_layout()
        selection = {'region': draft.template.facings[0].regions[0].id}
        state = {
            'app:template_source_signature': ('old-source',),
            'template_builder:draft': draft,
            'template_builder:selection': selection,
            'template_builder:pdf_export': {'pdf': b'%PDF-old'},
        }
        configured_size = render_pdf.PDF_PAGE_SIZE
        render_pdf.PDF_PAGE_SIZE = (612, 792)
        with patch.object(home.st, 'session_state', state):
            home._refresh_template_workflow()
            self.assertEqual(render_pdf.PDF_PAGE_SIZE, configured_size)
            self.assertIs(state['template_builder:draft'], draft)
            self.assertIs(state['template_builder:selection'], selection)
            self.assertNotIn('template_builder:pdf_export', state)
            state['template_builder:pdf_export'] = {'pdf': b'%PDF-new'}
            with patch.object(home, 'reload') as reload_mock:
                home._refresh_template_workflow()
                reload_mock.assert_not_called()
            self.assertEqual(state['template_builder:pdf_export']['pdf'], b'%PDF-new')
