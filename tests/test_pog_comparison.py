"""UPC coverage scores must distinguish missing images from failed checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from app.templates.image_compare import (DEFAULT_IMAGE_URL, compare_products, comparison_upc,
                                         normalize_upc, read_comparison_workbook, validate_image_url)
from app.pages import _pog_comparison as page


ROOT = DEFAULT_IMAGE_URL
PRODUCT = ROOT + 'Product/'
PACK = ROOT + 'Pack/'
UPC = '196742040135'
FOLDER = PRODUCT + UPC + '/'
IMAGE = FOLDER + UPC + '_front.jpg'


def product(upc='19674204013'):
    return {'Workbook row':10, 'UPC':upc, 'Product':'Example', 'Item #':'00123'}


class Directory:
    def __init__(self, broken=False):
        self.calls = []
        self.broken = broken

    def links(self, url):
        self.calls.append(url)
        if url == ROOT:
            return [PRODUCT, PACK]
        if url == PRODUCT:
            return [FOLDER, PRODUCT+'1196742040135/']
        if url == PACK:
            return []
        if url == FOLDER:
            if self.broken:
                raise ValueError('Directory unavailable')
            return [IMAGE]
        return []

    def image_exists(self, url):
        return url == IMAGE


class ComparisonTests(unittest.TestCase):
    def test_bos_check_digit_and_full_gtin_leading_zeros(self):
        self.assertEqual(comparison_upc('19674204013', True), UPC)
        self.assertEqual(comparison_upc('19674204013', False), '19674204013')
        self.assertEqual(normalize_upc('00'+UPC), normalize_upc(UPC))
        for value in ('bad', '123.5', '', None, True, '00000000000'):
            self.assertIsNone(normalize_upc(value))

    def test_exact_matches_duplicate_rows_and_invalid_upcs_have_explicit_denominator(self):
        directory = Directory()
        rows = [product(), product(), product('123456789012'), product('')]
        result = compare_products(rows, ROOT, directory)
        self.assertEqual((result['matched'], result['missing'], result['total']), (2,2,4))
        self.assertEqual(result['accuracy'], 50)
        self.assertEqual(directory.calls.count(FOLDER), 1)
        self.assertEqual(result['rows'][0]['Comparison UPC'], UPC)
        self.assertEqual(result['rows'][0]['Image'], IMAGE)
        self.assertEqual(result['rows'][0]['UPC'], '19674204013')
        self.assertEqual(result['rows'][-1]['Status'], 'Missing / invalid UPC')

    def test_connection_failure_is_unverified_not_zero_accuracy(self):
        result = compare_products([product()], ROOT, Directory(broken=True))
        self.assertEqual(result['unverified'], 1)
        self.assertEqual(result['missing'], 0)
        self.assertIsNone(result['accuracy'])

    def test_full_upc_option_does_not_append_another_check_digit(self):
        result = compare_products([product(UPC)], PRODUCT, Directory(), bos_upc11=False)
        self.assertEqual(result['accuracy'], 100)

    def test_empty_or_failed_library_cannot_produce_a_score(self):
        with self.assertRaises(ValueError):
            compare_products([], ROOT, Directory())
        with self.assertRaisesRegex(ValueError, 'No score'):
            compare_products([product()], ROOT, SimpleNamespace(links=lambda url:[]))

    def test_only_public_incomm_image_paths_are_accepted(self):
        self.assertEqual(validate_image_url(PRODUCT), PRODUCT)
        for url in ('http://assets.incomm.com/Images/', 'https://localhost/Images/',
                    'https://assets.incomm.com.evil.test/Images/',
                    'https://assets.incomm.com/Images/../../private/',
                    'https://user@assets.incomm.com/Images/',
                    'https://assets.incomm.com/Images/?token=secret'):
            with self.assertRaises(ValueError):
                validate_image_url(url)

    def test_workbook_tables_preserve_rows_and_do_not_require_images_or_facings(self):
        rows = [(None, None, None), ('UPC','Name','ID'),
                (19674204013,'Example','00123'), (None,'Missing UPC','00999')]
        sheet = SimpleNamespace(title='Products',iter_rows=lambda **kwargs:iter(rows))
        workbook = Mock()
        workbook.__iter__ = Mock(return_value=iter([sheet]))
        with patch('app.templates.image_compare.openpyxl.load_workbook',return_value=workbook):
            result = read_comparison_workbook(b'xlsx')
        self.assertEqual(len(result['Products']), 2)
        self.assertEqual(result['Products'][0]['UPC'], '19674204013')
        self.assertEqual(result['Products'][0]['Item #'], '00123')
        self.assertEqual(result['Products'][1]['Workbook row'], 4)
        workbook.close.assert_called_once()


class ComparisonAppTests(unittest.TestCase):
    def test_compare_shows_score_and_input_changes_remove_stale_results(self):
        app = AppTest.from_string('from app.pages._pog_comparison import render_comparison\nrender_comparison()')
        app.session_state['inputs:pog_compare:workbook'] = SimpleNamespace(name='data.xlsx',getvalue=lambda:b'xlsx')
        result = compare_products([product(),product('')], ROOT, Directory())
        with patch.object(page, '_read_sheet',return_value={'Products':[product(),product('')]}), \
                patch.object(page, 'compare_products',return_value=result) as compare:
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.metric)
            next(b for b in app.button if b.label=='Compare').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, '50.0%')
            self.assertTrue(compare.call_args.kwargs['bos_upc11'])
            self.assertTrue(any(d.proto.label=='Download comparison CSV' for d in app.get('download_button')))
            next(w for w in app.selectbox if w.label=='Spreadsheet UPC format').select('Full UPC / GTIN').run()
            self.assertNotIn(page.RESULT_KEY, app.session_state)
            self.assertFalse(app.metric)

    def test_compare_is_disabled_until_a_sheet_is_uploaded(self):
        app = AppTest.from_string('from app.pages._pog_comparison import render_comparison\nrender_comparison()').run()
        self.assertFalse(app.exception)
        self.assertTrue(next(b for b in app.button if b.label=='Compare').disabled)
