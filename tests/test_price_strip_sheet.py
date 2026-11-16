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

    def test_imported_rows_are_editable_and_pog_switches_retain_edits(self):
        app = self.app
        self.assertEqual(next(w for w in app.text_input if w.label=='Template name').value, 'POG A')
        self.assertEqual([w.label for w in app.expander if w.label.startswith('Row ')], ['Row 2','Row 4'])
        self.assertTrue(next(w for w in app.checkbox if w.label=='Include all sides in PDF').value)
        app.session_state['holiday_builder:row_tickets:1:0:thin'] = {
            'edited_rows':{0:{'Brand':'Edited product'}}, 'added_rows':[], 'deleted_rows':[]}
        app.run()
        next(w for w in app.selectbox if w.label=='POG from sheet').select('POG B').run()
        self.assertEqual(app.session_state[builder.ROWS_KEY][1]['rows'][0]['tickets'][0]['Brand'], 'POG B Product 1')
        next(w for w in app.selectbox if w.label=='POG from sheet').select('POG A').run()
        self.assertEqual(app.session_state[builder.ROWS_KEY][1]['rows'][0]['tickets'][0]['Brand'], 'Edited product')
        self.assertEqual(len(app.session_state[builder.ROWS_KEY][1]['rows']), 2)
        next(w for w in app.radio if w.label=='Template type').set_value('1. POG Template').run()
        next(w for w in app.radio if w.label=='Template type').set_value('2. Price Strips').run()
        self.assertEqual(next(w for w in app.text_input if w.label=='Template name').value, 'POG A')
        self.assertFalse(app.exception)

    def test_all_side_and_selected_side_exports_use_edited_rows_and_invalidate_download(self):
        app = self.app
        with patch.object(builder,'_create_all_sides_pdf',return_value=b'pdf') as export:
            next(b for b in app.button if b.label=='Create Price Strip PDF').click().run()
        self.assertEqual(set(export.call_args.args[1]), {1,2})
        self.assertEqual(app.session_state[builder.PDF_KEY]['pages'], 3)
        next(w for w in app.checkbox if w.label=='Include all sides in PDF').uncheck().run()
        self.assertNotIn(builder.PDF_KEY, app.session_state)
        with patch.object(builder,'_create_pdf',return_value=b'pdf') as export:
            next(b for b in app.button if b.label=='Create Price Strip PDF').click().run()
        self.assertEqual(export.call_args.kwargs['row_numbers'], [2,4])
        self.assertEqual(export.call_args.kwargs['footers'][1], 'Custom footer POG A 1 4')
        self.assertFalse(app.exception)
