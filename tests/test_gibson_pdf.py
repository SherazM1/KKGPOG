"""PDF creation must never silently substitute another font for Gibson."""
import unittest
from unittest.mock import patch

from app.sams_club import render_price_strips_html as renderer
from app.shared.async_runtime import run_browser_job


class GibsonPdfTests(unittest.TestCase):
    def test_both_required_font_files_are_embedded(self):
        css = renderer.sams_font_face_css([])
        self.assertEqual(renderer.SAMS_FONT_STACK, '"Gibson"')
        self.assertEqual(css.count('base64,'), 2)
        self.assertIn('font-weight: 400', css)
        self.assertIn('font-weight: 600', css)

    def test_missing_font_stops_pdf_creation(self):
        with patch.object(renderer, '_font_file_to_data_uri', side_effect=OSError('missing')):
            with self.assertRaisesRegex(ValueError, 'Gibson is required for PDF output'):
                renderer.sams_font_face_css([])

    def test_browser_rejects_output_without_loaded_gibson(self):
        async def render_without_fonts():
            async with renderer.async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                try:
                    return await renderer._render_page_to_pdf(browser, '<html>Test</html>', 100, 100)
                finally:
                    await browser.close()
        with self.assertRaisesRegex(Exception, 'Gibson Regular and SemiBold are required'):
            run_browser_job(render_without_fonts)
