import io
import unittest

from app.shared.workflow_state import (
    fingerprint, holiday_fingerprint, sams_fingerprint, sync_inputs, mark_current,
)


def upload(content=b"source", name="source.xlsx"):
    stream = io.BytesIO(content)
    stream.name = name
    return stream


class WorkflowStateTests(unittest.TestCase):
    def assert_invalidates(self, before, after):
        state = {}
        sync_inputs(state, "job", before, ["build", "pdf"])
        state.update(build={"reviewed": True}, pdf=b"previous output")
        self.assertTrue(sync_inputs(state, "job", after, ["build", "pdf"]))
        self.assertIsNone(state["build"])
        self.assertIsNone(state["pdf"])
        self.assertTrue(sync_inputs(state, "job", after, ["build", "pdf"]))
        mark_current(state, "job")
        self.assertFalse(sync_inputs(state, "job", after, ["build", "pdf"]))

    def test_qp_changed_source_invalidates_build_and_download(self):
        before = holiday_fingerprint(upload(), upload(b"assortment"), "images")
        after = holiday_fingerprint(upload(b"changed"), upload(b"assortment"), "images")
        self.assert_invalidates(before, after)

    def test_nre_configuration_and_source_dependencies(self):
        before = holiday_fingerprint(upload(), upload(), "images", "4x78")
        for after in (
            holiday_fingerprint(upload(), upload(), "images", "3x60"),
            holiday_fingerprint(upload(), upload(b"new"), "images", "4x78"),
            holiday_fingerprint(upload(), upload(), "other-images", "4x78"),
        ):
            with self.subTest(after=after):
                self.assert_invalidates(before, after)

    def test_sams_all_resolution_dependencies_invalidate(self):
        inputs = dict(source=upload(), selected_pog="A", image_root="images",
                      image_zip=upload(b"zip", "images.zip"),
                      catalog=upload(b"catalog", "catalog.csv"), mappings=b"approved")
        before = sams_fingerprint(**inputs)
        for field, value in (
            ("source", upload(b"new")), ("selected_pog", "B"),
            ("image_root", "other"), ("image_zip", upload(b"new zip")),
            ("catalog", upload(b"new catalog")), ("mappings", b"revised approval"),
        ):
            with self.subTest(field=field):
                self.assert_invalidates(before, sams_fingerprint(**{**inputs, field: value}))

    def test_stable_upload_content_ignores_object_identity_and_stream_position(self):
        first, second = upload(), upload()
        second.seek(3)
        current = fingerprint(source=first, settings={"b": 2, "a": 1})
        same = fingerprint(settings={"a": 1, "b": 2}, source=second)
        self.assertEqual(current, same)
        self.assertEqual(second.tell(), 3)
        state = {}
        self.assertFalse(sync_inputs(state, "job", current, ["build", "pdf"]))
        result = object()
        state.update(build=result, pdf=b"valid")
        self.assertFalse(sync_inputs(state, "job", same, ["build", "pdf"]))
        self.assertIs(state["build"], result)
        self.assertEqual(state["pdf"], b"valid")

    def test_title_invalidates_render_only_and_workflows_are_isolated(self):
        state = {"build": "reviewed", "other_pdf": b"unrelated"}
        sync_inputs(state, "render", fingerprint(title="A"), ["pdf"])
        state["pdf"] = b"A output"
        sync_inputs(state, "render", fingerprint(title="B"), ["pdf"])
        self.assertEqual(state["build"], "reviewed")
        self.assertEqual(state["other_pdf"], b"unrelated")
        self.assertIsNone(state["pdf"])

    def test_filename_mode_calibration_and_layout_revision(self):
        base = dict(source=upload(), mode="Standard", calibration=False, layout=b"layout")
        before = fingerprint(**base)
        for field, value in (("source", upload(name="source.pdf")), ("mode", "Sam's Holiday"),
                             ("calibration", True), ("layout", b"new layout")):
            with self.subTest(field=field):
                self.assertNotEqual(before, fingerprint(**{**base, field: value}))
