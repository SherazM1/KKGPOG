"""Printed draft geometry, vertical pagination, and export result lifecycle."""
from dataclasses import replace
from datetime import date
from pathlib import Path
import unittest

import fitz
from streamlit.testing.v1 import AppTest

from app.pages import _direct_builder as direct
from app.pages.template_builder import DRAFT_KEY, MODE_KEY, PDF_KEY, PDF_LAYOUT_VERSION
from app.templates import draft as edits
from app.templates.render_pdf import render_template_pdf


ROOT = Path(__file__).resolve().parents[1]


def small_sides():
    draft = edits.new_draft('Print layout', 'Custom', '1', 'normalized')
    for name in ('Side Alpha', 'Side Beta', 'Side Gamma'):
        draft = edits.add_facing(draft, name, 2, .35)
        fid = draft.template.facings[-1].id
        draft = edits.add_region(draft, fid, 'Cards & Gifts')
        rid = draft.template.facings[-1].regions[0].id
        draft = edits.create_grid(draft, fid, rid, rows=1, columns=4,
                                  x=.1, y=.1, width=1.8, height=.15, horizontal_gap=.04)
    return draft


class TemplatePdfTests(unittest.TestCase):
    def test_each_side_has_its_own_portrait_page_in_display_order(self):
        draft = small_sides()
        template = replace(draft.template, facings=tuple(reversed(draft.template.facings)))
        document = fitz.open(stream=render_template_pdf(template), filetype='pdf')
        self.addCleanup(document.close)
        self.assertEqual(len(document), 3)
        for index, name in enumerate(('Side Alpha', 'Side Beta', 'Side Gamma')):
            page = document[index]
            self.assertLess(page.rect.width, page.rect.height)
            self.assertIn(template.name, page.get_text())
            self.assertIn(f'SIDE {index + 1}', page.get_text())
            self.assertNotIn('Cards & Gifts', page.get_text())
            self.assertNotIn('Layout sketch', page.get_text())
        self.assertEqual(template.facings, tuple(reversed(draft.template.facings)))

    def test_tall_sides_continue_on_portrait_pages_without_losing_boxes(self):
        draft = edits.new_draft('Tall layout', 'Custom', '1', 'normalized')
        for name in ('Front', 'Back', 'Left', 'Right'):
            draft = edits.add_facing(draft, name, 1, 1)
            fid = draft.template.facings[-1].id
            draft = direct.add_section(draft, fid, 'Main Cards')
            rid = draft.template.facings[-1].regions[0].id
            draft = direct.create_grid(draft, fid, rid, rows=12, columns=5)
        document = fitz.open(stream=render_template_pdf(draft.template), filetype='pdf')
        self.addCleanup(document.close)
        self.assertEqual(len(document), 4)
        for index, page in enumerate(document):
            self.assertLess(page.rect.width, page.rect.height)
            text = page.get_text()
            self.assertIn('Tall layout', text)
            self.assertIn(f'SIDE {index + 1}', text)
            self.assertIn('60', text.splitlines())
            self.assertIn(f'Page {index + 1}', text)
            for block in page.get_text('blocks'):
                self.assertGreaterEqual(block[0], 0)
                self.assertGreaterEqual(block[1], 0)
                self.assertLessEqual(block[2], page.rect.width)
                self.assertLessEqual(block[3], page.rect.height)

    def test_measured_export_preserves_box_geometry_without_printing_setup_notes(self):
        draft = edits.new_draft('Measured', 'Custom', '1', 'in')
        draft = edits.add_facing(draft, 'Measured Side', 12, 24)
        fid = draft.template.facings[0].id
        draft = edits.add_region(draft, fid, 'Cards')
        rid = draft.template.facings[0].regions[0].id
        draft = edits.add_slot(draft, fid, rid, x=2, y=4, width=4, height=8)
        before = draft.template
        document = fitz.open(stream=render_template_pdf(before), filetype='pdf')
        self.addCleanup(document.close)
        page = document[0]
        self.assertNotIn('Measurements in inches', page.get_text())
        rectangles = [drawing['rect'] for drawing in page.get_drawings()
                      if any(item[0] == 're' for item in drawing['items'])]
        self.assertTrue(rectangles)
        self.assertAlmostEqual(rectangles[0].width / rectangles[0].height, .5, places=4)
        self.assertEqual(draft.template, before)

    def test_small_and_large_boxes_fit_the_printable_area_in_both_directions(self):
        for width, height in ((.04, .02), (.02, .04), (600, 900), (900, 600)):
            with self.subTest(width=width, height=height):
                draft = edits.new_draft('Fit layout', 'Custom', '1', 'normalized')
                draft = edits.add_facing(draft, 'Side A', 2000, 2000)
                fid = draft.template.facings[0].id
                draft = edits.add_region(draft, fid, 'Cards')
                rid = draft.template.facings[0].regions[0].id
                draft = edits.add_slot(draft, fid, rid, x=100, y=200, width=width, height=height)
                with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as doc:
                    boxes = [d['rect'] for d in doc[0].get_drawings()
                             if any(item[0] == 're' for item in d['items'])]
                    self.assertEqual(len(boxes), 1)
                    box = boxes[0]
                    scale = min(528 / width, 648 / height)
                    self.assertAlmostEqual(box.width, width * scale, places=2)
                    self.assertAlmostEqual(box.height, height * scale, places=2)
                    self.assertGreaterEqual(box.x0, 42 - .01)
                    self.assertGreaterEqual(box.y0, 92 - .01)
                    self.assertLessEqual(box.x1, 570 + .01)
                    self.assertLessEqual(box.y1, 740 + .01)

    def test_fitting_keeps_relative_box_sizes_gaps_and_the_draft_unchanged(self):
        draft = edits.new_draft('Keep spacing', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side A', 100, 100)
        fid = draft.template.facings[0].id
        draft = edits.add_region(draft, fid, 'Cards')
        rid = draft.template.facings[0].regions[0].id
        for x, y, width, height in ((10, 20, 2, 3), (13, 20, 1, 2), (10, 24, 2, 3)):
            draft = edits.add_slot(draft, fid, rid, x=x, y=y, width=width, height=height)
        before = draft.template
        with fitz.open(stream=render_template_pdf(before), filetype='pdf') as doc:
            boxes = [d['rect'] for d in doc[0].get_drawings()
                     if any(item[0] == 're' for item in d['items'])]
            self.assertEqual(len(boxes), 3)
            scale = 648 / 7
            self.assertAlmostEqual(boxes[0].width, 2 * scale, places=3)
            self.assertAlmostEqual(boxes[1].width, scale, places=3)
            self.assertAlmostEqual(boxes[1].x0 - boxes[0].x1, scale, places=3)
            self.assertAlmostEqual(boxes[2].y0 - boxes[0].y1, scale, places=3)
        self.assertEqual(draft.template, before)

    def test_empty_and_invalid_templates_cannot_be_exported(self):
        empty = edits.new_draft('Empty', 'Custom', '1', 'normalized')
        with self.assertRaises(ValueError):
            render_template_pdf(empty.template)
        draft = small_sides()
        face = draft.template.facings[0]
        bad = replace(face, slots=(replace(face.slots[0], x=-1), *face.slots[1:]))
        with self.assertRaises(ValueError):
            render_template_pdf(replace(draft.template, facings=(bad, *draft.template.facings[1:])))

    def test_reference_header_and_footer_positions_and_explicit_side_labels(self):
        draft = small_sides()
        names = ('Front Display', 'Side B - Back Panel', 'Side #12')
        template = replace(draft.template, facings=tuple(replace(side, name=name)
                            for side, name in zip(draft.template.facings, names)))
        document = fitz.open(stream=render_template_pdf(template, created_on=date(2026, 11, 5)), filetype='pdf')
        self.addCleanup(document.close)
        for page, identifier in zip(document, ('SIDE 1', 'SIDE B', 'SIDE 12')):
            spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for span in line['spans']]
            title = next(span for span in spans if span['text'] == template.name)
            label = next(span for span in spans if span['text'] == identifier)
            created = next(span for span in spans if span['text'] == 'Date: 2026-11-05')
            generated = next(span for span in spans if span['text'] == 'Generated By Kendal King')
            self.assertLess(title['bbox'][0], page.rect.width / 2)
            self.assertLess(title['bbox'][1], 60)
            self.assertGreater(label['bbox'][0], page.rect.width / 2)
            self.assertAlmostEqual(title['bbox'][1], label['bbox'][1], delta=1)
            self.assertLess(title['bbox'][2], label['bbox'][0])
            self.assertGreater(created['bbox'][1], page.rect.height - 60)
            self.assertLess(created['bbox'][0], page.rect.width / 4)
            self.assertAlmostEqual((generated['bbox'][0] + generated['bbox'][2]) / 2,
                                   page.rect.width / 2, delta=1)
            self.assertAlmostEqual(created['bbox'][1], generated['bbox'][1], delta=1)

    def test_long_template_name_fits_without_overlapping_the_side_identifier(self):
        draft = small_sides()
        template = replace(draft.template, name='Extra long Template name ' * 20,
                           facings=(draft.template.facings[0],))
        document = fitz.open(stream=render_template_pdf(template), filetype='pdf')
        self.addCleanup(document.close)
        header = [span for block in document[0].get_text('dict')['blocks'] if 'lines' in block
                  for line in block['lines'] for span in line['spans'] if span['bbox'][1] < 60]
        title = next(span for span in header if span['text'].startswith('Extra long'))
        label = next(span for span in header if span['text'] == 'SIDE 1')
        self.assertLess(title['bbox'][2], label['bbox'][0] - 12)

    def test_only_neutral_unfilled_box_outlines_are_printed(self):
        draft = small_sides()
        document = fitz.open(stream=render_template_pdf(draft.template), filetype='pdf')
        self.addCleanup(document.close)
        for page, side in zip(document, draft.template.ordered_facings()):
            rectangles = [drawing for drawing in page.get_drawings()
                          if any(item[0] == 're' for item in drawing['items'])]
            self.assertEqual(len(rectangles), len(side.slots))
            self.assertTrue(all(drawing['fill'] is None for drawing in rectangles))
            self.assertEqual(len({drawing['color'] for drawing in rectangles}), 1)
            self.assertTrue(all(channel > .7 for channel in rectangles[0]['color']))
            text = page.get_text()
            for removed in ('Layout sketch', 'sections |', 'Sections:', 'Cards & Gifts'):
                self.assertNotIn(removed, text)

    def test_every_box_contains_its_number_and_three_writing_fields(self):
        draft = small_sides()
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            for page, side in zip(document, draft.template.ordered_facings()):
                rectangles = [d['rect'] for d in page.get_drawings()
                              if any(item[0] == 're' for item in d['items'])]
                for box, slot in zip(rectangles, side.ordered_slots()):
                    text = page.get_text(clip=box)
                    self.assertIn(str(slot.order), text.splitlines())
                    for label in ('Description:', 'CPP:', 'VPC:'):
                        self.assertEqual(text.splitlines().count(label), 1)

    def test_copy_labels_remain_visible_in_the_pdf(self):
        source = small_sides()
        for name, expected in (
            ('Side A Copy', 'SIDE A COPY'),
            ('Side 1 Copy 2', 'SIDE 1 COPY 2'),
            ('A Copy Copy', 'SIDE A COPY COPY'),
            ('Front Display Copy', 'SIDE 4 COPY'),
        ):
            with self.subTest(name=name):
                draft = edits.duplicate_facing(source, source.template.facings[0].id, name)
                document = fitz.open(stream=render_template_pdf(draft.template), filetype='pdf')
                try:
                    self.assertIn(expected, document[-1].get_text())
                    self.assertNotIn('COPY', document[0].get_text())
                finally:
                    document.close()


class TemplatePdfAppTests(unittest.TestCase):
    def start(self, draft):
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = draft
        app.run()
        self.assertFalse(app.exception)
        return app

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def downloads(self, app):
        return [item.proto.label for item in app.get('download_button')]

    def test_export_remains_available_after_done(self):
        draft = small_sides()
        app = self.start(draft)
        self.click(app, 'Create Template PDF')
        self.assertIn('Download Template PDF', self.downloads(app))
        self.assertTrue(app.session_state[PDF_KEY]['pdf'].startswith(b'%PDF'))
        self.click(app, 'Done')
        self.assertEqual(app.session_state[MODE_KEY], 'viewer')
        self.assertIn('Download Template PDF', self.downloads(app))
        self.assertEqual(app.session_state[DRAFT_KEY], draft)

    def test_editing_invalidates_the_previous_download(self):
        draft = small_sides()
        app = self.start(draft)
        self.click(app, 'Create Template PDF')
        face = draft.template.facings[0]
        app.session_state[DRAFT_KEY] = edits.update_facing(draft, face.id, name='Side Z',
                                                         width=face.width, height=face.height, order=face.order)
        app.run()
        self.assertNotIn('Download Template PDF', self.downloads(app))
        self.assertNotIn(PDF_KEY, app.session_state)
        self.click(app, 'Create Template PDF')
        document = fitz.open(stream=app.session_state[PDF_KEY]['pdf'], filetype='pdf')
        self.addCleanup(document.close)
        self.assertIn('SIDE Z', ''.join(page.get_text() for page in document))

    def test_export_can_be_created_from_the_completed_draft_screen(self):
        app = self.start(small_sides())
        self.click(app, 'Done')
        self.click(app, 'Create Template PDF')
        self.assertIn('Download Template PDF', self.downloads(app))

    def test_empty_draft_disables_export(self):
        app = self.start(edits.new_draft('Empty', 'Custom', '1', 'normalized'))
        self.assertTrue(next(b for b in app.button if b.label == 'Create Template PDF').disabled)
        self.assertEqual(self.downloads(app), [])

    def test_old_pdf_layout_is_invalidated_without_changing_the_draft(self):
        draft = small_sides()
        app = self.start(draft)
        app.session_state[PDF_KEY] = dict(signature=(draft.template.id, draft.revision, PDF_LAYOUT_VERSION - 1), pdf=b'%PDF-old')
        app.run()
        self.assertNotIn(PDF_KEY, app.session_state)
        self.assertEqual(self.downloads(app), [])
        self.assertEqual(app.session_state[DRAFT_KEY], draft)


if __name__ == '__main__':
    unittest.main()
