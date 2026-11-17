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
    def start(self, imported=False):
        from tests.test_price_strip_sheet import upload_state
        app = AppTest.from_file(str(ROOT / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state['templates:selected_type'] = '2. Price Strips'
        if imported:
            for key, value in upload_state().items():
                app.session_state[key] = value
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_upload_is_required_and_manual_controls_are_absent(self):
        app = self.start()
        self.assertFalse(any(w.label == 'Create Price Strip PDF' for w in app.button))
        self.assertFalse(any(w.label == 'Add Row' for w in app.button))
        self.assertFalse(any(w.label in ('Side', 'Strip dimensions') for w in app.selectbox))
        self.assertFalse(any(w.label == 'Starting row' for w in app.number_input))
        self.assertTrue(any('Upload a valid spreadsheet' in w.value for w in app.info))

    def test_imported_products_are_read_only_and_old_edits_cannot_change_export(self):
        app = self.start(imported=True)
        app.session_state[builder.ROWS_KEY][1]['rows'][0]['tickets'][0]['Brand'] = 'OLD EDIT'
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.get('data_editor'))
        self.assertFalse(any(w.label in ('Template name', 'Row footer') for w in app.text_input))
        with patch.object(builder, '_create_all_sides_pdf', return_value=b'pdf') as export:
            next(w for w in app.button if w.label == 'Create Price Strip PDF').click().run()
        source = export.call_args.args[1]
        self.assertEqual(set(source), {1, 2})
        self.assertEqual(source[1]['rows'][0]['tickets'][0]['Brand'], 'POG A Product 1')
        self.assertEqual([r['number'] for r in source[1]['rows']], [2, 4])
        self.assertTrue(app.get('download_button'))
        self.assertFalse(app.exception)

    def test_failed_upload_hides_preview_and_stale_download(self):
        app = self.start(imported=True)
        with patch.object(builder, '_create_all_sides_pdf', return_value=b'pdf'):
            next(w for w in app.button if w.label == 'Create Price Strip PDF').click().run()
        app.session_state[builder.SHEET_ERROR_KEY] = 'Invalid workbook'
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn(builder.PDF_KEY, app.session_state)
        self.assertFalse(app.get('download_button'))
        self.assertFalse(any(w.label == 'Create Price Strip PDF' for w in app.button))
