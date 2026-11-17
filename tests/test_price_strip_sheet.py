"""Workbook products populate editable POGs without mixing positions or drafts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.pages import price_strip_builder as builder
from app.pages._price_strip_sheet import import_price_strip_sheet
from app.sams_club.price_strip_models import (SamsPriceStripBuildResult, SamsPriceStripRow,
                                             SamsPriceStripSegment)


ROOT = Path(__file__).resolve().parents[1]


def sample_build():
    rows = []
    for pog, side, number in [('POG A', 1, 2), ('POG A', 1, 4), ('POG A', 2, 2), ('POG B', 1, 1)]:
        segments = [SamsPriceStripSegment(pog=pog, side=side, row=number, column=column,
                    brand=f'{pog} Product {column}', desc_1='1 X $25', desc_2='Gift Card',
                    retail='0' if column == 1 else '22.50', item_number='00123456789')
                    for column in (1, 3)]
        rows.append(SamsPriceStripRow(pog=pog, side=side, row=number, segments=segments,
                                      footer_text=f'Custom footer {pog} {side} {number}'))
    return SamsPriceStripBuildResult(strip_rows=rows, included_segment_count=8)


def imported():
    with patch('app.pages._price_strip_sheet.build_sams_price_strip_rows', return_value=sample_build()):
        return import_price_strip_sheet(b'xlsx')


def upload_state():
    state = {builder.SHEET_UPLOAD_KEY: SimpleNamespace(name='source.xlsx', getvalue=lambda:b'xlsx')}
    with patch.object(builder.st, 'session_state', state), \
            patch('app.pages._price_strip_sheet.import_price_strip_sheet', return_value=imported()):
        builder._load_sheet_upload()
    state.pop(builder.SHEET_UPLOAD_KEY)
    return state


class SheetImportTests(unittest.TestCase):
    def test_import_preserves_pogs_sides_positions_gaps_identifiers_and_zero_prices(self):
        data = imported()
        self.assertEqual(set(data['pogs']), {'POG A', 'POG B'})
        first = data['pogs']['POG A'][1]
        self.assertEqual(builder._row_numbers(first), [2, 4])
        self.assertEqual(len(first['rows'][0]['tickets']), 6)
        self.assertEqual(len(data['pogs']['POG A'][2]['rows'][0]['tickets']), 8)
        tickets = first['rows'][0]['tickets']
        self.assertEqual(tickets[0]['Item #'], '00123456789')
        self.assertEqual(tickets[2]['Brand'], 'POG A Product 3')
        strip = builder._build_strip('POG A', 1, 2, tickets, 'thin')
        self.assertEqual(strip.segments[0].retail, '0.00')
        self.assertTrue(strip.segments[1].is_empty)

    def test_duplicate_positions_and_out_of_range_columns_are_rejected(self):
        for column, message in [(1, 'duplicate column'), (7, 'exceeds 6')]:
            build = sample_build()
            build.strip_rows[0].segments[1].column = column
            with patch('app.pages._price_strip_sheet.build_sams_price_strip_rows', return_value=build):
                with self.assertRaisesRegex(ValueError, message):
                    import_price_strip_sheet(b'xlsx')

    def test_invalid_file_does_not_replace_existing_rows(self):
        state = {builder.ROWS_KEY:{'keep':'existing'}, builder.SHEET_UPLOAD_KEY:
                 SimpleNamespace(name='bad.xlsx', getvalue=lambda:b'not a workbook')}
        with patch.object(builder.st, 'session_state', state):
            builder._load_sheet_upload()
        self.assertEqual(state[builder.ROWS_KEY], {'keep':'existing'})
        self.assertIn(builder.SHEET_ERROR_KEY, state)

    def test_all_side_export_keeps_source_row_numbers_and_footers(self):
        drafts = imported()['pogs']['POG A']
        with patch.object(builder, '_render_strips', return_value=b'pdf') as render:
            self.assertEqual(builder._create_all_sides_pdf('POG A', drafts), b'pdf')
        rows = render.call_args.args[0]
        self.assertEqual([(r.side,r.row) for r in rows], [(1,2),(1,4),(2,2)])
        self.assertEqual(rows[1].footer_text, 'Custom footer POG A 1 4')

    def test_append_after_changed_start_keeps_source_row_spacing(self):
        draft = deepcopy(imported()['pogs']['POG A'][1])
        draft['start'] = 8
        with patch.object(builder.st, 'session_state', {builder.ROWS_KEY:{1:draft}}):
            builder._add_row(1)
        self.assertEqual(builder._row_numbers(draft), [8,10,11])


class SheetImportAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT/'home.py'), default_timeout=15)
        self.app.session_state['active_workflow'] = 'Templates'
        self.app.session_state['templates:selected_type'] = '2. Price Strips'
        for key, value in upload_state().items():
            self.app.session_state[key] = value
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_imported_pogs_are_read_only_and_source_survives_navigation(self):
        app = self.app
        self.assertFalse(app.get('data_editor'))
        self.assertFalse(any(w.label == 'Template name' for w in app.text_input))
        selector = next(w for w in app.selectbox if w.label == 'Preview strip')
        self.assertEqual(selector.options, ['Side 1, Row 2', 'Side 1, Row 4', 'Side 2, Row 2'])
        next(w for w in app.selectbox if w.label == 'POG from sheet').select('POG B').run()
        self.assertEqual(next(w for w in app.selectbox if w.label == 'Preview strip').options, ['Side 1, Row 1'])
        next(w for w in app.selectbox if w.label == 'POG from sheet').select('POG A').run()
        next(w for w in app.radio if w.label == 'Template type').set_value('1. POG Template').run()
        next(w for w in app.radio if w.label == 'Template type').set_value('2. Price Strips').run()
        self.assertEqual(app.session_state[builder.SHEET_KEY]['current'], 'POG A')
        self.assertFalse(app.exception)

    def test_preview_navigation_does_not_change_all_sides_export_and_pog_change_invalidates_it(self):
        app = self.app
        next(w for w in app.selectbox if w.label == 'Preview strip').select(2).run()
        with patch.object(builder, '_create_all_sides_pdf', return_value=b'pdf') as export:
            next(w for w in app.button if w.label == 'Create Price Strip PDF').click().run()
        self.assertEqual(set(export.call_args.args[1]), {1, 2})
        self.assertEqual(len(export.call_args.args[1][1]['rows']), 2)
        self.assertTrue(app.get('download_button'))
        next(w for w in app.selectbox if w.label == 'POG from sheet').select('POG B').run()
        self.assertNotIn(builder.PDF_KEY, app.session_state)
        self.assertFalse(app.get('download_button'))
        self.assertFalse(app.exception)
