"""Page-height canvas and atomic downward sketch growth."""
from dataclasses import replace
import unittest
from xml.etree import ElementTree as ET

from app.pages import _direct_builder as direct
from app.pages._canvas_events import CanvasTransform, apply_event
from app.pages._direct_canvas import canvas_svg
from app.templates import draft as edits


class GrowingCanvasTests(unittest.TestCase):
    def layout(self, unit='normalized', width=1, height=1):
        draft = edits.new_draft('Growing layout', 'Custom', '1', unit)
        draft = edits.add_facing(draft, 'Side A', width, height)
        fid = draft.template.facings[0].id
        draft = direct.add_section(draft, fid, 'Cards')
        rid = draft.template.facings[0].regions[0].id
        return direct.create_grid(draft, fid, rid, rows=1, columns=2), fid, rid

    def move(self, draft, fid, *, y, group=True):
        face = edits.get_facing(draft, fid)
        transform = CanvasTransform.from_svg(canvas_svg(draft, face), face)
        event = dict(event_id='grow:1', template_id=draft.template.id,
                     facing_id=fid, revision=draft.revision)
        if group:
            event.update(kind='move_many', slot_ids=[s.id for s in face.slots],
                         delta={'x': 0, 'y': y * transform.scale})
        else:
            slot = face.slots[0]
            event.update(kind='drag', slot_id=slot.id,
                         rect=transform.to_canvas(replace(slot, y=slot.y + y)))
        return apply_event(draft, fid, event, transform, allow_resize=False)

    def test_initial_workspace_has_extra_depth_without_relabeling_dimensions(self):
        draft, fid, _ = self.layout()
        face = edits.get_facing(draft, fid)
        root = ET.fromstring(canvas_svg(draft, face))
        boundary = next(n for n in root if n.get('data-kind') == 'facing')
        workspace = next(n for n in root if n.get('data-kind') == 'workspace')
        self.assertGreaterEqual(float(workspace.get('height')), float(workspace.get('width')) * 1.6)
        self.assertGreater(float(root.get('height')), 1200)
        self.assertEqual(root.get('data-grow-down'), 'true')
        self.assertEqual(face.height, 1)
        transform = CanvasTransform.from_svg(ET.tostring(root), face)
        self.assertAlmostEqual(float(boundary.get('height')), face.height * transform.scale)
        slot = face.slots[0]
        for name, value in transform.to_template(transform.to_canvas(slot)).items():
            self.assertAlmostEqual(value, getattr(slot, name))

    def test_adding_rows_increases_the_full_page_canvas_height(self):
        draft, fid, rid = self.layout()
        before = float(ET.fromstring(canvas_svg(draft, edits.get_facing(draft, fid))).get('height'))
        draft = direct.create_grid(draft, fid, rid, rows=20, columns=2)
        face = edits.get_facing(draft, fid)
        root = ET.fromstring(canvas_svg(draft, face))
        self.assertGreater(float(root.get('height')), before)
        self.assertTrue(edits.validate_draft(draft).valid)

    def test_group_move_grows_side_and_section_in_one_revision(self):
        draft, fid, rid = self.layout()
        draft = edits.add_facing(draft, 'Side B', 1, 1)
        source, other = draft.template.facings
        result = self.move(draft, fid, y=2)
        self.assertTrue(result.accepted)
        self.assertEqual(result.draft.revision, draft.revision + 1)
        self.assertEqual(result.draft.next_id, draft.next_id)
        face = edits.get_facing(result.draft, fid)
        self.assertGreater(face.height, max(s.y + s.height for s in face.slots))
        self.assertEqual(face.width, source.width)
        self.assertEqual(face.regions[0].slot_ids, source.regions[0].slot_ids)
        for old, new in zip(source.slots, face.slots):
            self.assertEqual(replace(new, y=old.y), old)
            self.assertAlmostEqual(new.y - old.y, 2)
        self.assertEqual(result.draft.template.facings[1], other)
        self.assertTrue(edits.validate_draft(result.draft).valid)

    def test_single_box_drag_grows_without_moving_other_boxes(self):
        draft, fid, _ = self.layout()
        source = edits.get_facing(draft, fid)
        result = self.move(draft, fid, y=3, group=False)
        self.assertTrue(result.accepted)
        face = edits.get_facing(result.draft, fid)
        self.assertGreater(face.height, 3)
        self.assertEqual(face.slots[1:], source.slots[1:])
        self.assertTrue(edits.validate_draft(result.draft).valid)

    def test_measured_dimensions_are_preserved(self):
        draft, fid, _ = self.layout('in', 12, 24)
        face = edits.get_facing(draft, fid)
        root = ET.fromstring(canvas_svg(draft, face))
        self.assertEqual(root.get('data-grow-down'), 'false')
        result = self.move(draft, fid, y=30)
        self.assertEqual(edits.get_facing(result.draft, fid).height, 24)
        self.assertEqual(edits.get_facing(result.draft, fid).regions, face.regions)
        self.assertIn('slot_bounds', {i.code for i in edits.validate_draft(result.draft).errors})

    def test_moving_back_up_keeps_the_extra_workspace(self):
        draft, fid, _ = self.layout()
        grown = self.move(draft, fid, y=2).draft
        restored = self.move(grown, fid, y=-2).draft
        self.assertEqual(edits.get_facing(restored, fid).height, edits.get_facing(grown, fid).height)
        for old, new in zip(edits.get_facing(draft, fid).slots, edits.get_facing(restored, fid).slots):
            self.assertAlmostEqual(new.y, old.y)
            self.assertEqual(replace(new, y=old.y), old)


if __name__ == '__main__':
    unittest.main()
