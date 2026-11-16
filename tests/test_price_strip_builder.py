"""Multiple manually entered strip rows retain their products and PDF order."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.sams_club.holiday_price_strips import (holiday_geometry_for_row,
                                               holiday_geometry_for_size)
from app.sams_club.render_price_strips_html import compute_strip_canvas


ROOT = Path(__file__).resolve().parents[1]
RENDERER = 'app.sams_club.render_price_strips_html'


class PriceStripRowsTests(unittest.TestCase):
    def test_only_drawing_sizes_are_accepted_and_work_on_every_side(self):
        for size, width, count in (('thin', 30.25, 6), ('wide', 38.4375, 8)):
            geometry = holiday_geometry_for_size(size)
            self.assertEqual((geometry.width_in, geometry.height_in), (width, 3.4375))
            for side in (1, 2, 3, 4):
                strip = builder._build_strip('Sizes', side, 1,
                    builder._default_tickets(side, size), size)
                self.assertEqual(len(strip.segments), count)
                self.assertEqual(holiday_geometry_for_row(strip), geometry)
                self.assertEqual(compute_strip_canvas(strip, [], "Sam's Holiday")[:2],
                                 (width * 72, 3.4375 * 72))
                self.assertIn(f'Side: {side}', strip.footer_text)
        with self.assertRaises(ValueError):
            holiday_geometry_for_size('custom')

    def test_pdf_receives_selected_size_instead_of_the_sides_default(self):
        rows = [builder._default_tickets(1, 'wide')]
        renderer = Mock(return_value=SimpleNamespace(pdf_bytes=b'pdf', warnings=[]))
        with patch.dict('sys.modules', {RENDERER: SimpleNamespace(render_sams_price_strips_pdf=renderer)}):
            builder._create_pdf('Sizes', 1, 1, rows, strip_size='wide')
        strip = renderer.call_args.args[0][0]
        self.assertEqual(strip.side, 1)
        self.assertEqual(strip.strip_size, 'wide')
        self.assertEqual(len(strip.segments), 8)

    def test_pdf_contains_each_rows_own_products_and_footer_in_order(self):
        rows = [builder._default_tickets(2) for _ in range(3)]
        for index, tickets in enumerate(rows):
            tickets[0].update({'Brand': f'Product {index}', 'Price': f'{index + 1}.25'})
        renderer = Mock(return_value=SimpleNamespace(pdf_bytes=b'pdf', warnings=[]))
        with patch.dict('sys.modules', {RENDERER: SimpleNamespace(render_sams_price_strips_pdf=renderer)}):
            self.assertEqual(builder._create_pdf('Holiday', 2, 4, rows), b'pdf')
        strips = renderer.call_args.args[0]
        self.assertEqual([row.row for row in strips], [4, 5, 6])
        self.assertEqual([len(row.segments) for row in strips], [8, 8, 8])
        for index, row in enumerate(strips):
            self.assertEqual(row.segments[0].brand, f'Product {index}')
            self.assertEqual(row.segments[0].retail, f'{index + 1}.25')
            self.assertTrue(all(segment.row == row.row for segment in row.segments))
            self.assertIn(f'Side: 2, Row: {row.row}', row.footer_text)

    def test_bad_price_identifies_the_row_and_position_before_rendering(self):
        rows = [builder._default_tickets(1) for _ in range(2)]
        rows[1][2]['Price'] = 'invalid'
        renderer = Mock()
        with patch.dict('sys.modules', {RENDERER: SimpleNamespace(render_sams_price_strips_pdf=renderer)}):
            with self.assertRaisesRegex(ValueError, 'Row 5, position 3'):
                builder._create_pdf('Holiday', 1, 4, rows)
        renderer.assert_not_called()

    def test_more_than_fifty_rows_are_independent_and_sides_remain_separate(self):
        with patch.object(builder.st, 'session_state', {}):
            for _ in range(60):
                builder._add_row(1)
            rows = builder._side_rows(1)['rows']
            self.assertEqual(len(rows), 61)
            rows[0]['tickets'][0]['Brand'] = 'First only'
            self.assertEqual(rows[1]['tickets'][0]['Brand'], 'PRODUCT NAME')
            self.assertEqual(len(builder._side_rows(2)['rows']), 1)
            self.assertEqual(len(builder._side_rows(2)['rows'][0]['tickets']), 8)

    def test_previous_single_row_entries_are_migrated(self):
        state = {'holiday_builder:row': 4, 'holiday_builder:tickets:1': {
            'edited_rows': {0: {'Brand': 'Keep this', 'Price': '12.34'}}}}
        with patch.object(builder.st, 'session_state', state):
            draft = builder._side_rows(1)
            self.assertEqual(draft['start'], 4)
            self.assertEqual(draft['rows'][0]['tickets'][0]['Brand'], 'Keep this')


class PriceStripRowsAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        self.app.session_state['active_workflow'] = 'Templates'
        self.app.session_state['templates:selected_type'] = '2. Price Strips'
        self.app.run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def edit_first_product(self, brand):
        self.app.session_state['holiday_builder:row_tickets:1:0:thin'] = {
            'edited_rows': {0: {'Brand': brand}}, 'added_rows': [], 'deleted_rows': []}
        self.app.run()

    def test_edits_survive_added_rows_side_changes_and_template_changes(self):
        self.edit_first_product('Kept brand')
        self.button('Add Row').click().run()
        self.button('Add Row').click().run()
        next(w for w in self.app.number_input if w.label == 'Starting row').set_value(4).run()
        next(w for w in self.app.selectbox if w.label == 'Side').select(2).run()
        next(w for w in self.app.selectbox if w.label == 'Side').select(1).run()
        next(w for w in self.app.radio if w.label == 'Template type').set_value('1. POG Template').run()
        next(w for w in self.app.radio if w.label == 'Template type').set_value('2. Price Strips').run()
        self.assertFalse(self.app.exception)
        draft = self.app.session_state[builder.ROWS_KEY][1]
        self.assertEqual(len(draft['rows']), 3)
        self.assertEqual(draft['start'], 4)
        self.assertEqual(draft['rows'][0]['tickets'][0]['Brand'], 'Kept brand')
        self.assertEqual([w.label for w in self.app.expander if w.label.startswith('Row ')], ['Row 4', 'Row 5', 'Row 6'])

    def test_download_is_invalidated_by_edits_and_new_rows(self):
        self.button('Add Row').click().run()
        with patch.object(builder, '_create_pdf', return_value=b'pdf') as renderer:
            self.button('Create Price Strip PDF').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(renderer.call_args.args[3]), 2)
        self.assertEqual(self.app.session_state[builder.PDF_KEY]['row_count'], 2)
        self.edit_first_product('Changed')
        self.assertNotIn(builder.PDF_KEY, self.app.session_state)
        with patch.object(builder, '_create_pdf', return_value=b'pdf'):
            self.button('Create Price Strip PDF').click().run()
        self.button('Add Row').click().run()
        self.assertNotIn(builder.PDF_KEY, self.app.session_state)
        self.assertFalse(self.app.exception)

    def test_sizes_can_change_without_losing_hidden_products_or_rows(self):
        picker = next(w for w in self.app.selectbox if w.label == 'Strip dimensions')
        self.assertEqual(picker.options, list(builder.STRIP_SIZES.values()))
        picker.select('wide').run()
        self.app.session_state['holiday_builder:row_tickets:1:0:wide'] = {
            'edited_rows': {0: {'Brand': 'First'}, 7: {'Brand': 'Eighth'}},
            'added_rows': [], 'deleted_rows': []}
        self.app.run()
        self.button('Add Row').click().run()
        next(w for w in self.app.selectbox if w.label == 'Strip dimensions').select('thin').run()
        self.assertEqual(self.app.session_state[builder.ROWS_KEY][1]['rows'][0]['tickets'][7]['Brand'], 'Eighth')
        self.edit_first_product('First edited while narrow')
        next(w for w in self.app.selectbox if w.label == 'Strip dimensions').select('wide').run()
        draft = self.app.session_state[builder.ROWS_KEY][1]
        self.assertEqual(len(draft['rows']), 2)
        self.assertEqual(draft['rows'][0]['tickets'][0]['Brand'], 'First edited while narrow')
        self.assertEqual(draft['rows'][0]['tickets'][7]['Brand'], 'Eighth')
        next(w for w in self.app.selectbox if w.label == 'Side').select(2).run()
        next(w for w in self.app.selectbox if w.label == 'Strip dimensions').select('thin').run()
        with patch.object(builder, '_create_pdf', return_value=b'pdf') as renderer:
            self.button('Create Price Strip PDF').click().run()
        self.assertEqual(renderer.call_args.kwargs['strip_size'], 'thin')
        self.assertEqual(len(renderer.call_args.args[3][0]), 6)
        self.assertFalse(self.app.exception)
