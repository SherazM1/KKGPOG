"""Read-only region inspection and contained image layout."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class RegionViewerTests(unittest.TestCase):
    def test_side_a_viewer_shows_regions_and_hint_summary_without_edit_controls(self):
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select("Templates").run()
        app.selectbox(key="template_viewer_template").select("holiday_2026_side_a").run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([metric.value for metric in app.metric], ["53", "1", "53"])
        self.assertEqual(len(app.get("imgs")), 1)
        self.assertEqual([button.label for button in app.button], ["Create New Template"])
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len(app.dataframe), 2)
        slots, regions = app.dataframe[0].value, app.dataframe[1].value
        self.assertEqual(len(slots), 53)
        self.assertEqual(list(regions["Order"]), [1, 2, 3, 4])
        self.assertTrue(all(slots["Regions"].str.startswith("holiday_2026_side_a.")))
        self.assertEqual(list(regions["Presentation hint"]), ["custom_rows", "single_row", "grid", "section"])
        self.assertEqual(app.expander[0].label, "Advanced / Geometry details")
        self.assertTrue(any("not approved" in item.value for item in app.caption))
        self.assertTrue(any("does not move" in item.value for item in app.caption))

    def test_image_uses_content_width_and_existing_templates_remain_reachable(self):
        from unittest.mock import patch
        import app.pages.templates as controller
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select("Templates").run()
        actual_image = controller.st.image
        with patch.object(controller.st, "image", wraps=actual_image) as image:
            for template_id, count in (("holiday_2026_side_a", 53), ("holiday_qp_2026", 15), ("holiday_nre", 40)):
                app.selectbox(key="template_viewer_template").select(template_id).run()
                self.assertEqual([error.message for error in app.exception], [])
                self.assertEqual(app.metric[0].value, str(count))
                self.assertEqual(image.call_args.kwargs["width"], "content")
