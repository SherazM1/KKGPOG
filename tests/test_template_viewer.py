"""Read-only Templates workflow checks with no external files or services."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class TemplateViewerTests(unittest.TestCase):
    def start(self):
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select("Templates").run()
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([error.message for error in app.exception], [])

    def test_qp_all_facings_have_read_only_geometry_preview(self):
        app = self.start()
        self.assertEqual([metric.value for metric in app.metric], ["15", "4", "60"])
        self.assertTrue(any("not calibrated" in notice.value for notice in app.info))
        self.assertEqual(len(app.dataframe[0].value), 15)
        for facing in ("left", "front", "right", "back"):
            app.selectbox(key="template_viewer_facing:holiday_qp_2026").select(facing).run()
            self.assert_clean(app)
            self.assertEqual(len(app.get("imgs")), 1)
            self.assertTrue(all(f".{facing}." in value for value in app.dataframe[0].value["Slot ID"]))
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertEqual(app.expander[0].label, "Advanced / Geometry details")

    def test_nre_viewer_uses_generic_sections_and_preserves_other_workflow_state(self):
        app = self.start()
        app.session_state["sams_pdf_result"] = b"existing PDF"
        app.selectbox(key="template_viewer_template").select("holiday_nre").run()
        self.assert_clean(app)
        self.assertEqual([metric.value for metric in app.metric], ["40", "1", "40"])
        self.assertEqual(app.selectbox(key="template_viewer_facing:holiday_nre").value, "main")
        table = app.dataframe[0].value
        self.assertEqual(len(table), 40)
        self.assertEqual(len(table["Section"].unique()), 10)
        self.assertTrue(any("locked" in caption.value for caption in app.caption))
        self.assertEqual(app.session_state["sams_pdf_result"], b"existing PDF")
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.get("imgs")), 1)

    def test_existing_routes_remain_and_viewer_selection_survives_navigation(self):
        from home import WORKFLOWS
        self.assertEqual(list(WORKFLOWS), ["Standard Display", "Full Pallet", "Holiday Quarter Pallet",
                                          "Holiday NRE", "Sam's Planogram", "Sam's Price Strips",
                                          "Image Review / Mapping", "Templates"])
        app = self.start()
        app.selectbox(key="template_viewer_template").select("holiday_nre").run()
        app.selectbox(key="active_workflow").select("Standard Display").run()
        self.assert_clean(app)
        app.selectbox(key="active_workflow").select("Templates").run()
        self.assert_clean(app)
        self.assertEqual(app.selectbox(key="template_viewer_template").value, "holiday_nre")
