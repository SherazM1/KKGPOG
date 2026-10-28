"""Exercise routing and result lifecycles without changing production mappings."""
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from app.sams_club.models import SamsPlanogram, SamsSidePage
from app.sams_club.service import SamsBuildResult
from app.sams_club.price_strip_models import SamsPriceStripBuildResult, SamsPriceStripPdfResult
from app.shared.upload_utils import NamedImageIndex
from home import WORKFLOWS
from tests.test_workflow_state import upload


ROOT = Path(__file__).resolve().parents[1]


def rendered():
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, format="PNG")
    return SimpleNamespace(pdf_bytes=b"%PDF-test", preview_png_bytes=image.getvalue(),
                           panels_rendered=1, slots_rendered=0, image_slots=0,
                           placeholder_slots=0, skipped_slots=0, missing_image_rows=[],
                           warnings=[])


class WorkflowControllerTests(unittest.TestCase):
    def start(self, workflow):
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select(workflow).run()
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([error.message for error in app.exception], [])

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assert_clean(app)

    def downloads(self, app):
        return [item.proto.label for item in app.get("download_button")]

    def holiday_job(self, app, name):
        app.session_state[f"inputs:{name}_bos"] = upload()
        app.session_state[f"inputs:{name}_assortment"] = upload(b"assortment")
        app.run()
        self.assert_clean(app)

    def test_all_workflows_and_strip_modes_are_reachable_without_duplicate_controls(self):
        app = self.start("Standard Display")
        expected_uploads = [3, 6, 2, 2, 3, 1, 0]
        for (name, _), count in zip(WORKFLOWS.items(), expected_uploads):
            with self.subTest(workflow=name):
                app.selectbox(key="active_workflow").select(name).run()
                self.assert_clean(app)
                self.assertEqual(len(app.get("file_uploader")), count)
                labels = [button.label for button in app.button]
                self.assertEqual(len(labels), len(set(labels)))
        app.selectbox(key="active_workflow").select("Sam's Price Strips").run()
        for mode in ("Standard", "Sam's Holiday", "Price Pocket"):
            app.selectbox(key="sams_price_strip_template").select(mode).run()
            self.assert_clean(app)

    def test_qp_source_change_removes_current_download_and_preserves_stable_result(self):
        job = SimpleNamespace(errors=[], warnings=[], summary={}, placements=[],
                              reference_image=None, geometry=None,
                              placement_qa_rows=[], product_qa_rows=[])
        with patch("app.holiday_planograms.service.build_holiday_qp_qa", return_value=job), \
             patch("app.holiday_planograms.render_qp.render_holiday_qp", return_value=rendered()):
            app = self.start("Holiday Quarter Pallet")
            self.holiday_job(app, "holiday_qp")
            self.click(app, "Build Holiday QP QA")
            self.click(app, "Render Holiday QP")
            self.assertIn("Download Holiday QP PDF", self.downloads(app))
            app.run()
            self.assertIn("Download Holiday QP PDF", self.downloads(app))
            app.session_state["inputs:holiday_qp_bos"] = upload(b"changed")
            app.run()
            self.assert_clean(app)
            self.assertIsNone(app.session_state["holiday_qp_result"])
            self.assertNotIn("Download Holiday QP PDF", self.downloads(app))
            self.assertTrue(any("Inputs changed" in item.value for item in app.info))

    def test_nre_configuration_and_sources_remove_downloads(self):
        job = SimpleNamespace(errors=[], warnings=[], summary={}, placements=[],
                              reference=None, geometry=None, configuration="4x78",
                              placement_qa_rows=[], product_qa_rows=[])
        with patch("app.holiday_planograms.service.build_holiday_nre_qa", return_value=job), \
             patch("app.holiday_planograms.render_nre.render_holiday_nre", return_value=rendered()):
            app = self.start("Holiday NRE")
            self.holiday_job(app, "holiday_nre")
            self.click(app, "Build Holiday NRE QA")
            self.click(app, "Render Holiday NRE")
            self.assertIn("Download Holiday NRE PDF", self.downloads(app))
            app.selectbox(key="holiday_nre_configuration").select("3x60").run()
            self.assert_clean(app)
            self.assertNotIn("Download Holiday NRE PDF", self.downloads(app))
            app.session_state["inputs:holiday_nre_assortment"] = upload(b"changed")
            app.run()
            self.assertIsNone(app.session_state["holiday_nre_result_4x78"])
            self.assertIsNone(app.session_state["holiday_nre_render_result_4x78"])
            self.assertTrue(any("Inputs changed" in item.value for item in app.info))

    def sams_job(self, app):
        app.session_state["inputs:sams_main_source_file"] = upload(b"records", "source.csv")
        app.run()
        self.click(app, "Build Sam's Planogram Structure")
        self.click(app, "Generate Completed Sam's Planogram PDF")
        self.assertIn("Download Completed Sam's Planogram PDF", self.downloads(app))

    def sams_result(self, candidates=None):
        return SamsBuildResult(
            planogram=SamsPlanogram("TEST", [SamsSidePage("TEST", 1, 8)]),
            extracted_record_count=1, normalized_record_count=1,
            detected_pogs=["TEST"], selected_pog="TEST",
            debug={"image_resolution": {"ocr_candidates": candidates or []}},
        )

    def test_sams_title_and_image_dependency_invalidate_correct_stages(self):
        with patch("app.pages.sams_planogram.detect_sams_pogs", return_value=(["TEST"], [])), \
             patch("app.pages.sams_planogram.build_sams_planogram_structure", return_value=self.sams_result()), \
             patch("app.pages.sams_planogram.render_sams_planogram_pdf", return_value=SimpleNamespace(
                 pdf_bytes=b"%PDF-test", rendered_slots=0, warnings=[], missing_image_slots=0)):
            app = self.start("Sam's Planogram")
            self.sams_job(app)
            app.text_input(key="sams_pdf_title").set_value("New title").run()
            self.assert_clean(app)
            self.assertIsNotNone(app.session_state["sams_build_result"])
            self.assertNotIn("Download Completed Sam's Planogram PDF", self.downloads(app))
            self.click(app, "Generate Completed Sam's Planogram PDF")
            app.text_input(key="sams_local_image_root").set_value("other-images").run()
            self.assert_clean(app)
            self.assertIsNone(app.session_state["sams_build_result"])
            self.assertIsNone(app.session_state["sams_pdf_result"])

    def test_mapping_review_retains_job_and_invalidates_output_without_writing_storage(self):
        candidate = dict(unresolved_upc="001", item_number="002", candidate_rank=1,
                         candidate_file_path="verified.jpg")
        with patch("app.pages.sams_planogram.detect_sams_pogs", return_value=(["TEST"], [])), \
             patch("app.pages.sams_planogram.build_sams_planogram_structure", return_value=self.sams_result([candidate])), \
             patch("app.pages.sams_planogram.render_sams_planogram_pdf", return_value=SimpleNamespace(
                 pdf_bytes=b"%PDF-test", rendered_slots=0, warnings=[], missing_image_slots=0)), \
             patch("app.pages.image_review.preview_image_status", return_value=(False, "No preview")), \
             patch("app.pages.image_review.append_manual_image_mapping") as save:
            app = self.start("Sam's Planogram")
            self.sams_job(app)
            app.selectbox(key="active_workflow").select("Image Review / Mapping").run()
            self.assert_clean(app)
            app.button(key="sams_ocr_accept_1_1").click().run()
            self.assert_clean(app)
            save.assert_called_once()
            self.assertIsNone(app.session_state["sams_pdf_result"])
            self.assertIsNone(app.session_state["sams_build_result"])
            self.assertIsNotNone(app.session_state["sams_review_result"])
            self.click(app, "Rebuild Sam's Planogram With Saved Mappings")
            app.selectbox(key="active_workflow").select("Sam's Planogram").run()
            self.assert_clean(app)
            self.assertIsNotNone(app.session_state["sams_build_result"])
            self.click(app, "Generate Completed Sam's Planogram PDF")

    def test_standard_generate_still_calls_existing_service_and_exposes_download(self):
        prepared = (b"matrix", b"labels", b"images", NamedImageIndex(), False, {})
        rows = [{"Side": "A", "Row": 1, "Col": 1}]
        with patch("app.pages.standard_display.prepare_images", return_value=prepared), \
             patch("app.standard_display.service.prepare_standard_display", return_value=(["page"], {}, rows)), \
             patch("app.standard_display.service.render_standard_display_pdf", return_value=b"%PDF") as render:
            app = self.start("Standard Display")
            app.session_state["inputs:standard_display_matrix"] = upload()
            app.session_state["inputs:standard_display_labels"] = upload(b"labels", "labels.pdf")
            app.run()
            self.click(app, "Generate POG PDF")
            render.assert_called_once_with(["page"], b"images", {}, "POG")
            self.assertIn("Download Planogram PDF", self.downloads(app))

    def test_full_pallet_generation_and_transfer_remain_reachable(self):
        from contextlib import ExitStack
        prepared = (b"matrix", b"labels", b"images", NamedImageIndex(), False, {})
        page = SimpleNamespace(side_letter="A")
        rows = [{"Side": "A", "Row": 1, "Col": 1}]
        with ExitStack() as stack:
            stack.enter_context(patch("app.pages.full_pallet.prepare_images", return_value=prepared))
            for name, value in {
                "parse_ppt_cards": {}, "validate_ppt_cards": [], "parse_gift_holders": {},
                "load_full_pallet_matrix": {}, "parse_full_pallet_pages": [page],
                "build_full_pallet_label_audit": [{"Status": "RESOLVED", "Side": "A"}],
                "build_full_pallet_rows": rows,
            }.items():
                stack.enter_context(patch(f"app.full_pallet.service.{name}", return_value=value))
            render = stack.enter_context(patch("app.full_pallet.service.render_full_pallet_display_pdf", return_value=b"%PDF"))
            transfer = stack.enter_context(patch("app.full_pallet.image_transfer.add_images_to_existing_label_pog",
                                                return_value=SimpleNamespace(pdf_bytes=b"%PDF", audit_rows=[{"Status": "INSERTED"}])))
            app = self.start("Full Pallet")
            for key, name in (("full_pallet_matrix", "matrix.xlsx"), ("full_pallet_labels", "labels.pdf"),
                              ("full_pallet_pptx_file", "top.pptx"), ("full_pallet_gift_file", "holders.xlsx")):
                app.session_state[f"inputs:{key}"] = upload(name=name)
            app.run()
            self.click(app, "Generate POG PDF")
            render.assert_called_once_with([page], b"images", b"labels", {}, "POG", {}, {}, 0, False, False, None, None)
            self.assertIn("Download Planogram PDF", self.downloads(app))
            app.session_state["inputs:image_transfer_label_pdf"] = upload(b"target", "labels.pdf")
            app.session_state["inputs:image_transfer_source_pdf"] = upload(b"source", "images.pdf")
            self.click(app, "Add Images to Existing Label POG")
            transfer.assert_called_once_with(b"target", b"source", include_middle=True, include_bonus=True)
            self.assertIn("Download POG With Images", self.downloads(app))
            self.assertIn("Download Image Transfer Audit CSV", self.downloads(app))

    def test_price_strips_are_independent_and_mode_changes_remove_download(self):
        from unittest.mock import Mock
        result = SamsPriceStripBuildResult()
        renderer = Mock(return_value=SamsPriceStripPdfResult(b"%PDF", 1, 1))
        with patch("app.pages.sams_price_strips._build_sams_price_strip_rows_compat", return_value=result), \
             patch("app.pages.sams_price_strips._load_sams_price_strip_renderer", return_value=renderer):
            app = self.start("Sam's Price Strips")
            app.session_state["inputs:sams_excel_file"] = upload()
            app.session_state["sams_build_result"] = "unrelated planogram"
            app.session_state["sams_pdf_result"] = b"unrelated PDF"
            app.run()
            self.click(app, "Generate Sam's Price Strips")
            self.assertIn("Download Sam's Price Strips PDF", self.downloads(app))
            app.selectbox(key="sams_price_strip_template").select("Sam's Holiday").run()
            self.assert_clean(app)
            self.assertNotIn("Download Sam's Price Strips PDF", self.downloads(app))
            self.assertEqual(app.session_state["sams_pdf_result"], b"unrelated PDF")
