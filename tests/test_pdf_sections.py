"""Section headings fit around the printed layout; new Sides use letters."""
from pathlib import Path
import unittest

import fitz
from streamlit.testing.v1 import AppTest

from app.pages import _direct_builder as direct
from app.pages._canvas_events import ACTIVE_SIDE_KEY
from app.pages.template_builder import (DRAFT_KEY, MODE_KEY, PDF_READY_KEY, _next_side_name,
                                         _side_letter)
from app.templates import draft as edits
from app.templates.print_style import (HEADER_COLOR_KEY, section_header_color,
                                       section_preserves_spacing)
from app.templates.render_pdf import PDF_PAGE_SIZE, _grow_row_space, render_template_pdf


ROOT = Path(__file__).resolve().parents[1]


def section_layout():
    draft = edits.new_draft('Holiday Display', 'Custom', '1', 'normalized')
    draft = edits.add_facing(draft, 'Side A', 1, 1)
    fid = draft.template.facings[0].id
    for name in ('Greeting Cards', 'Gift Bags'):
        draft = direct.add_section(draft, fid, name)
        rid = draft.template.facings[0].regions[-1].id
        draft = direct.create_grid(draft, fid, rid, rows=3, columns=4)
    return draft


class SideLetterTests(unittest.TestCase):
    def test_letters_continue_past_z(self):
        for number, label in ((1, 'A'), (26, 'Z'), (27, 'AA'), (52, 'AZ'), (53, 'BA')):
            self.assertEqual(_side_letter(number), label)

    def test_next_name_skips_used_letters_and_ignores_copy_suffix(self):
        draft = edits.new_draft('Names', 'Custom', '1', 'normalized')
        self.assertEqual(_next_side_name(draft), 'Side A')
        for name in (' side a ', 'Side A Copy', 'SIDE B'):
            draft = edits.add_facing(draft, name, 1, 1)
        self.assertEqual(_next_side_name(draft), 'Side C')

    def test_new_sides_default_to_a_then_b_and_open_for_editing(self):
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = edits.new_draft('Names', 'Custom', '1', 'normalized')
        app.run()
        for name in ('Side A', 'Side B'):
            field = next(w for w in app.text_input if w.label == 'New side name')
            self.assertEqual(field.value, name)
            next(w for w in app.button if w.label == 'Add Side').click().run()
            self.assertFalse(app.exception)
            side = app.session_state[DRAFT_KEY].template.facings[-1]
            self.assertEqual(side.name, name)
            self.assertEqual(app.session_state[ACTIVE_SIDE_KEY], side.id)

    def test_color_picker_updates_only_the_active_section_and_requires_done_again(self):
        draft = section_layout()
        first, second = draft.template.facings[0].regions
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = draft
        app.session_state[PDF_READY_KEY] = (draft.template.id, draft.revision)
        app.run()
        picker = next(w for w in app.color_picker if first.id in (w.key or ''))
        picker.set_value('#FFC800')
        next(w for w in app.button if w.label == 'Apply section changes').click().run()
        self.assertFalse(app.exception)
        updated = app.session_state[DRAFT_KEY]
        regions = updated.template.facings[0].regions
        self.assertEqual(section_header_color(regions[0]), '#FFC800')
        self.assertEqual(regions[1], second)
        self.assertEqual(updated.revision, draft.revision + 1)
        self.assertNotEqual(regions[0], first)
        self.assertNotEqual(app.session_state[PDF_READY_KEY], (updated.template.id, updated.revision))

    def test_color_validation_and_copy_preserve_section_styles(self):
        draft = section_layout()
        fid = draft.template.facings[0].id
        rid = draft.template.facings[0].regions[0].id
        updated = edits.set_region_header_color(draft, fid, rid, '#ffcc00')
        copied = direct.duplicate_section(updated, fid, rid, 'Cards Copy')
        copied_region = next(r for r in copied.template.facings[0].regions if r.name == 'Cards Copy')
        self.assertEqual(copied_region.metadata[HEADER_COLOR_KEY], '#FFCC00')
        with self.assertRaises(ValueError):
            edits.set_region_header_color(draft, fid, rid, 'invalid')
        self.assertNotIn(HEADER_COLOR_KEY, draft.template.facings[0].regions[0].metadata)

    def test_section_spacing_can_be_chosen_during_creation_and_editing(self):
        draft = edits.new_draft('Spacing Options', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side A', 1, 1)
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = draft
        app.run()
        create = next(w for w in app.checkbox if ':add_region:' in (w.key or '') and w.label == 'Keep vertical spacing in PDF')
        self.assertTrue(create.value)
        create.uncheck()
        next(w for w in app.button if w.label == 'Add Section').click().run()
        self.assertFalse(app.exception)
        draft = app.session_state[DRAFT_KEY]
        region = draft.template.facings[0].regions[0]
        self.assertFalse(section_preserves_spacing(region))
        copied = direct.duplicate_section(draft, draft.template.facings[0].id, region.id, 'Copy')
        self.assertFalse(section_preserves_spacing(copied.template.facings[0].regions[-1]))
        edit = next(w for w in app.checkbox if region.id in (w.key or '') and w.label == 'Keep vertical spacing in PDF')
        edit.check()
        next(w for w in app.button if w.label == 'Apply section changes').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(section_preserves_spacing(app.session_state[DRAFT_KEY].template.facings[0].regions[0]))


class PdfSectionTests(unittest.TestCase):
    def test_full_width_header_bands_use_section_colors_and_contrasting_text(self):
        draft = section_layout()
        side = draft.template.facings[0]
        for region, color in zip(side.regions, ('#123456', '#FFDF00')):
            draft = edits.set_region_header_color(draft, side.id, region.id, color)
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            page = document[0]
            bands = [d for d in page.get_drawings() if d['fill'] is not None
                     and any(item[0] == 're' for item in d['items'])]
            self.assertEqual(len(bands), 2)
            spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                     for line in b['lines'] for s in line['spans']]
            for band, name, color, text_color in zip(
                bands, ('Greeting Cards', 'Gift Bags'), ('123456', 'FFDF00'), (0xFFFFFF, 0),
            ):
                self.assertAlmostEqual(band['rect'].x0, 6, delta=.01)
                self.assertAlmostEqual(band['rect'].x1, PDF_PAGE_SIZE[0] - 6, delta=.01)
                expected = tuple(int(color[i:i+2], 16) / 255 for i in (0, 2, 4))
                for channel, value in zip(band['fill'], expected):
                    self.assertAlmostEqual(channel, value, places=4)
                title = next(s for s in spans if s['text'] == name)
                self.assertEqual(title['color'], text_color)
                self.assertTrue(band['rect'].contains(fitz.Rect(title['bbox'])))
            self.assert_titles_clear_of_boxes(page, ('Greeting Cards', 'Gift Bags'))

    def test_each_side_keeps_the_configured_page_size_with_headers_and_footers_aligned(self):
        draft = section_layout()
        draft = edits.duplicate_facing(draft, draft.template.facings[0].id, 'Side A Copy')
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            self.assertEqual(len(document), 2)
            for index, page in enumerate(document):
                self.assertAlmostEqual(page.rect.width, PDF_PAGE_SIZE[0])
                self.assertAlmostEqual(page.rect.height, PDF_PAGE_SIZE[1])
                spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                         for line in b['lines'] for s in line['spans']]
                footer = next(s for s in spans if s['text'] == 'Generated By Kendal King')
                self.assertAlmostEqual((footer['bbox'][0] + footer['bbox'][2]) / 2, page.rect.width / 2, delta=1)
                self.assertGreater(footer['bbox'][1], page.rect.height - 44)
                label = next(s for s in spans if s['text'] == ('SIDE A COPY' if index else 'SIDE A'))
                self.assertAlmostEqual(label['bbox'][2], page.rect.width - 6, delta=1)
                self.assertLess(label['bbox'][1], 60)
                bands = [d['rect'] for d in page.get_drawings() if d['fill'] is not None]
                self.assertLess(label['bbox'][3], min(b.y0 for b in bands))
                boxes = [d['rect'] for d in page.get_drawings() if d['fill'] is None
                         and any(item[0] == 're' for item in d['items'])]
                self.assertGreater(footer['bbox'][1], max(b.y1 for b in boxes))

    def assert_titles_clear_of_boxes(self, page, names):
        outer = [d['rect'] for d in page.get_drawings()
                 if d['fill'] is None and any(item[0] == 're' for item in d['items'])][::2]
        spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                 for line in block['lines'] for span in line['spans']]
        for name in names:
            matches = [span for span in spans if span['text'] == name]
            self.assertEqual(len(matches), 1, name)
            title = fitz.Rect(matches[0]['bbox'])
            self.assertTrue(page.rect.contains(title))
            self.assertFalse(any(title.intersects(box) for box in outer), name)
        return outer

    def test_titles_appear_once_above_sections_and_keep_placeholders(self):
        draft = section_layout()
        before = draft.template
        with fitz.open(stream=render_template_pdf(before), filetype='pdf') as document:
            self.assertEqual(len(document), 1)
            page = document[0]
            outer = self.assert_titles_clear_of_boxes(page, ('Greeting Cards', 'Gift Bags'))
            self.assertEqual(len(outer), 24)
            self.assertEqual(page.get_text().splitlines().count('UPC:'), 24)
            self.assertIn('SIDE A', page.get_text())
        self.assertEqual(draft.template, before)

    def test_titles_also_print_for_empty_sections(self):
        draft = section_layout()
        draft = direct.add_section(draft, draft.template.facings[0].id, 'Coming Soon')
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            self.assert_titles_clear_of_boxes(document[0],
                                              ('Greeting Cards', 'Gift Bags', 'Coming Soon'))

    def test_each_grid_fills_the_width_with_larger_boxes_in_narrower_sections(self):
        draft = edits.new_draft('Centered Sections', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side A', 1, 1)
        fid = draft.template.facings[0].id
        for name, columns in (('Wide Cards', 5), ('Narrow Cards', 2)):
            draft = direct.add_section(draft, fid, name)
            rid = draft.template.facings[0].regions[-1].id
            draft = direct.create_grid(draft, fid, rid, rows=2, columns=columns)
        before = draft.template
        with fitz.open(stream=render_template_pdf(before), filetype='pdf') as document:
            page = document[0]
            boxes = self.assert_titles_clear_of_boxes(page, ('Wide Cards', 'Narrow Cards'))
            spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                     for line in b['lines'] for s in line['spans']]
            for name, section_boxes in (('Wide Cards', boxes[:10]), ('Narrow Cards', boxes[10:])):
                self.assertAlmostEqual(min(b.x0 for b in section_boxes), 6, delta=.01)
                self.assertAlmostEqual(max(b.x1 for b in section_boxes), page.rect.width - 6, delta=.01)
                center = (min(b.x0 for b in section_boxes) + max(b.x1 for b in section_boxes)) / 2
                self.assertAlmostEqual(center, page.rect.width / 2, delta=.01)
                title = next(s for s in spans if s['text'] == name)
                self.assertAlmostEqual((title['bbox'][0] + title['bbox'][2]) / 2, center, delta=1)
            self.assertGreater(boxes[10].width, boxes[0].width * 2)
        self.assertEqual(draft.template, before)

    def test_boxes_grow_into_row_gaps_without_changing_page_size_or_draft(self):
        draft = edits.new_draft('Spaced Rows', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side A', 10, 10)
        fid = draft.template.facings[0].id
        draft = edits.add_region(draft, fid, 'Cards', preserve_spacing=False)
        rid = draft.template.facings[0].regions[0].id
        for y in (1, 2.3, 5.3):
            draft = edits.add_slot(draft, fid, rid, x=1, y=y, width=2, height=1)
        before = draft.template
        with fitz.open(stream=render_template_pdf(before), filetype='pdf') as document:
            boxes = self.assert_titles_clear_of_boxes(document[0], ('Cards',))
            self.assertEqual(len(boxes), 3)
            small_gap = boxes[1].y0 - boxes[0].y1
            large_gap = boxes[2].y0 - boxes[1].y1
            self.assertAlmostEqual(small_gap, 6, places=3)
            self.assertAlmostEqual(large_gap, 6, places=3)
            original_height = (PDF_PAGE_SIZE[1] - 80 - 22) / 5.3
            self.assertGreater(boxes[0].height, original_height)
            self.assertAlmostEqual(boxes[0].height, boxes[1].height, places=3)
            self.assertAlmostEqual(boxes[1].height, boxes[2].height, places=3)
            self.assertEqual((document[0].rect.width, document[0].rect.height), PDF_PAGE_SIZE)
        self.assertEqual(draft.template, before)

    def test_legacy_and_preserved_sections_print_the_users_row_gap_proportions(self):
        for setting in (None, True):
            with self.subTest(setting=setting):
                draft = edits.new_draft('Keep My Gaps', 'Custom', '1', 'normalized')
                draft = edits.add_facing(draft, 'Side A', 10, 10)
                fid = draft.template.facings[0].id
                draft = edits.add_region(draft, fid, 'Cards', preserve_spacing=setting)
                rid = draft.template.facings[0].regions[0].id
                for y in (1, 2.3, 5.3):
                    draft = edits.add_slot(draft, fid, rid, x=1, y=y, width=2, height=1)
                with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
                    boxes = self.assert_titles_clear_of_boxes(document[0], ('Cards',))
                    self.assertAlmostEqual((boxes[1].y0 - boxes[0].y1) / boxes[0].height, .3, places=4)
                    self.assertAlmostEqual((boxes[2].y0 - boxes[1].y1) / boxes[0].height, 2, places=4)

    def test_preserved_and_expanding_sections_can_share_one_side(self):
        draft = edits.new_draft('Mixed Spacing', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side A', 12, 12)
        fid = draft.template.facings[0].id
        for name, ys, preserve in (('Keep Space', (1, 2.6), True), ('Fill Space', (6, 8.8), False)):
            draft = edits.add_region(draft, fid, name, preserve_spacing=preserve)
            rid = draft.template.facings[0].regions[-1].id
            for y in ys:
                draft = edits.add_slot(draft, fid, rid, x=1, y=y, width=2, height=1)
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            boxes = self.assert_titles_clear_of_boxes(document[0], ('Keep Space', 'Fill Space'))
            self.assertAlmostEqual((boxes[1].y0 - boxes[0].y1) / boxes[0].height, .6, places=4)
            self.assertAlmostEqual(boxes[3].y0 - boxes[2].y1, 6, places=3)
            self.assertGreater(boxes[2].height, boxes[0].height)
            self.assertFalse(any(box.intersects(other) for i, box in enumerate(boxes)
                                 for other in boxes[i+1:]))

    def test_growth_stops_at_headers_and_keeps_short_existing_gaps(self):
        rectangles = [(0, 80, 20, 20), (25, 80, 20, 20), (0, 40, 20, 10)]
        grown = _grow_row_space(rectangles, [120, 25])
        self.assertEqual(grown, [(0, 56, 20, 44), (25, 56, 20, 44), (0, 31, 20, 19)])
        self.assertEqual(_grow_row_space([(0, 25, 20, 20), (0, 0, 20, 20)], [90])[0],
                         (0, 25, 20, 20))

    def test_print_uses_the_full_frame_without_overlapping_boxes(self):
        draft = section_layout()
        before = draft.template
        side = before.facings[0]
        raw_height = max(s.y + s.height for s in side.slots) - min(s.y for s in side.slots)
        raw_width = max(s.x + s.width for s in side.slots) - min(s.x for s in side.slots)
        old_scale = min(528 / raw_width, (648 - 36) / raw_height)
        with fitz.open(stream=render_template_pdf(before), filetype='pdf') as document:
            page = document[0]
            boxes = self.assert_titles_clear_of_boxes(page, ('Greeting Cards', 'Gift Bags'))
            self.assertAlmostEqual(min(b.x0 for b in boxes), 6, places=2)
            self.assertAlmostEqual(max(b.x1 for b in boxes), page.rect.width - 6, places=2)
            self.assertAlmostEqual(max(b.y1 for b in boxes), page.rect.height - 30, places=2)
            self.assertGreater(boxes[0].width, side.slots[0].width * old_scale)
            self.assertGreater(boxes[0].height, side.slots[0].height * old_scale)
            drawings = [d['rect'] for d in page.get_drawings()
                        if d['fill'] is None and any(item[0] == 're' for item in d['items'])]
            self.assertEqual(len(drawings), len(side.slots) * 2)
            for box, square in zip(drawings[::2], drawings[1::2]):
                self.assertTrue(box.contains(square))
                self.assertAlmostEqual(square.width, square.height, places=3)
            for i, box in enumerate(boxes):
                self.assertFalse(any(box.intersects(other) for other in boxes[i+1:]))
        self.assertEqual(draft.template, before)

    def test_side_by_side_and_staggered_sections_do_not_print_titles_over_boxes(self):
        draft = edits.new_draft('Staggered', 'Custom', '1', 'normalized')
        draft = edits.add_facing(draft, 'Side B', 10, 10)
        fid = draft.template.facings[0].id
        for name, x, y, width, height in (
            ('Tall Cards', 0, 0, 3, 8), ('Small Cards', 5, 2, 3, 2),
            ('Gift Tags', 5, 6, 3, 2),
        ):
            draft = edits.add_region(draft, fid, name)
            rid = draft.template.facings[0].regions[-1].id
            draft = edits.add_slot(draft, fid, rid, x=x, y=y, width=width, height=height)
        with fitz.open(stream=render_template_pdf(draft.template), filetype='pdf') as document:
            boxes = self.assert_titles_clear_of_boxes(document[0], ('Tall Cards', 'Small Cards', 'Gift Tags'))
            for i, box in enumerate(boxes):
                self.assertGreaterEqual(box.x0, 6 - .01)
                self.assertLessEqual(box.x1, document[0].rect.width - 6 + .01)
                self.assertFalse(any(box.intersects(other) for other in boxes[i+1:]))


if __name__ == '__main__':
    unittest.main()
