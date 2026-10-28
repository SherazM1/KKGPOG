import asyncio
import sys
import unittest
from unittest.mock import patch

import fitz

from app.shared.async_runtime import run_browser_job
from app.sams_club.price_pocket import pocket_records, render_price_pocket_pdf
from app.sams_club.price_strip_models import SamsPriceStripRow, SamsPriceStripSegment
from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf


@unittest.skipUnless(sys.platform == "win32", "Windows Streamlit event-loop regression")
class BrowserRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.previous_policy = asyncio.get_event_loop_policy()
        self.selector_policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(self.selector_policy)

    def tearDown(self):
        asyncio.set_event_loop_policy(self.previous_policy)

    def test_subprocess_under_selector_policy_and_running_loop(self):
        async def subprocess_job():
            self.assertIsInstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop)
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "print('browser-worker-ok')", stdout=asyncio.subprocess.PIPE)
            output, _ = await process.communicate()
            self.assertEqual(process.returncode, 0)
            return output.strip()

        async def host():
            return run_browser_job(subprocess_job)

        self.assertEqual(asyncio.run(host()), b"browser-worker-ok")
        self.assertIs(asyncio.get_event_loop_policy(), self.selector_policy)

    def test_renderers_generate_real_pdfs_under_streamlit_policy(self):
        records = pocket_records([{"name": "Sam's test", "price": "60", "item": "001", "upc": "0002"}])
        pocket = render_price_pocket_pdf(records)
        with fitz.open(stream=pocket.pdf_bytes, filetype="pdf") as document:
            self.assertEqual(document.page_count, 1)
            self.assertIn("Sam's test", document[0].get_text())
        row = SamsPriceStripRow(pog="TEST", side=1, row=1, segments=[
            SamsPriceStripSegment(pog="TEST", side=1, row=1, column=1,
                                  brand="TEST", desc_1="Gift card", retail="60", item_number="001")])
        for mode in ("Standard", "Sam's Holiday"):
            # Chromium was already exercised by Pocket; skip the legacy network installer.
            with patch("app.sams_club.render_price_strips_html._ensure_playwright_chromium_installed", return_value=None):
                result = render_sams_price_strips_pdf([row], template_name=mode)
            self.assertTrue(result.pdf_bytes, result.warnings)
            with fitz.open(stream=result.pdf_bytes, filetype="pdf") as document:
                self.assertGreater(document.page_count, 0)
                self.assertGreater(document[0].rect.width, 0)
        self.assertIs(asyncio.get_event_loop_policy(), self.selector_policy)


if __name__ == "__main__":
    unittest.main()
