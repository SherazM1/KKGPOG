"""The catalog Quarter Pallet PDF includes all blank sides and a scale notice."""
from pathlib import Path
import unittest

import pymupdf
from streamlit.testing.v1 import AppTest

from app.templates.defaults import holiday_qp_2026
from app.templates.render_pdf import render_template_pdf


class QuarterPalletDownloadTests(unittest.TestCase):
    def test_all_four_named_sides_have_fifteen_blank_positions_and_scale_notice(self):
        template = holiday_qp_2026()
        data = render_template_pdf(template, use_facing_names=True,
                                   layout_note='Layout sketch - not to scale. Physical dimensions are not calibrated.')
        with pymupdf.open(stream=data, filetype='pdf') as document:
            self.assertEqual(len(document), 4)
            for page, side in zip(document, template.ordered_facings()):
                text = page.get_text()
                self.assertIn(side.name, text)
                self.assertIn(template.name, text)
                self.assertIn('Layout sketch - not to scale', text)
                self.assertEqual(text.count('Description:'), 15)
                self.assertEqual(text.count('CPP:'), 15)
                self.assertEqual(text.count('UPC:'), 15)
                rects = [d['rect'] for d in page.get_drawings()
                         if any(item[0] == 're' for item in d['items'])]
                self.assertEqual(len(rects), 30)
                self.assertTrue(all(page.rect.contains(rect) for rect in rects))
                self.assertTrue(all(rect.y0 > 53 for rect in rects))

    def test_catalog_selection_offers_pdf_without_requiring_a_draft(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'))
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '1. POG Template'
        app.session_state['template_viewer_template'] = 'holiday_qp_2026'
        app.run()
        self.assertFalse(app.exception)
        downloads = app.get('download_button')
        self.assertTrue(any(d.proto.label == 'Download Holiday Quarter Pallet Template' for d in downloads))
