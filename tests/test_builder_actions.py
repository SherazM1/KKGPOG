"""Operator defaults use the existing draft/grid API without changing its contract."""
import unittest

from app.pages import _builder_actions as actions
from app.templates import draft as edits
from app.templates.models import Bounds, SlotRole
from dataclasses import replace


def setup(unit="normalized", width=1, height=1):
    draft = edits.new_draft("Operator", "Custom", "1.0", unit)
    draft = edits.add_facing(draft, "Side 1", width, height)
    face = draft.template.facings[0]
    draft = edits.add_region(draft, face.id, "Cards")
    return draft, face.id, draft.template.facings[0].regions[0].id


class BuilderActionsTests(unittest.TestCase):
    def test_automatic_box_and_grid_membership_with_safe_spacing(self):
        draft, face_id, region_id = setup()
        draft = actions.add_box(draft, face_id, region_id)
        face = draft.template.facings[0]
        settings = actions.grid_defaults(face, region_id, 3, 4)
        updated = edits.create_grid(draft, face_id, region_id, rows=3, columns=4, **settings)
        face = updated.template.facings[0]
        self.assertEqual(len(face.slots), 13)
        self.assertEqual(len(face.slots_for_region(region_id)), 13)
        self.assertTrue(edits.validate_draft(updated).valid)
        self.assertTrue(all(slot.role == SlotRole.PRODUCT for slot in face.slots))
        self.assertIsNone(face.regions[0].presentation_hint)

    def test_grid_spacing_scales_with_row_column_count(self):
        draft, _, region_id = setup()
        face = draft.template.facings[0]
        for rows, columns in ((1, 1), (3, 4), (50, 50)):
            settings = actions.grid_defaults(face, region_id, rows, columns)
            self.assertGreater(settings["width"] - (columns - 1) * settings["horizontal_gap"], 0)
            self.assertGreater(settings["height"] - (rows - 1) * settings["vertical_gap"], 0)

    def test_duplicates_are_independent_preserve_source_and_append_creation_order(self):
        draft, face_id, region_id = setup()
        draft = actions.add_box(draft, face_id, region_id)
        source = draft.template.facings[0].slots[0]
        updated = actions.duplicate_box(draft, face_id, region_id, source.id)
        face = updated.template.facings[0]
        copy = face.slots[1]
        self.assertIs(face.slots[0], source)
        self.assertNotEqual(copy.id, source.id)
        self.assertEqual((copy.width, copy.height, copy.role), (source.width, source.height, source.role))
        self.assertEqual(copy.order, source.order + 1)
        self.assertEqual(face.regions_for_slot(copy.id)[0].id, region_id)
        self.assertTrue(edits.validate_draft(updated).valid)

    def test_movement_and_resize_preserve_identity_order_and_other_boxes(self):
        draft, face_id, region_id = setup()
        draft = actions.add_box(draft, face_id, region_id)
        original = draft.template.facings[0].slots[0]
        for action in ("left", "right", "up", "down", "wider", "narrower", "taller", "shorter"):
            changed = actions.adjust_box(draft, face_id, region_id, original.id, action)
            slot = changed.template.facings[0].slots[0]
            self.assertEqual((slot.id, slot.order), (original.id, original.order))
            self.assertTrue(edits.validate_draft(changed).valid)
            self.assertNotEqual((slot.x, slot.y, slot.width, slot.height),
                                (original.x, original.y, original.width, original.height))
        with self.assertRaises(ValueError):
            actions.adjust_box(draft, face_id, region_id, original.id, "drag")

    def test_no_room_does_not_overwrite_or_overlap_existing_boxes(self):
        draft, face_id, region_id = setup()
        draft = edits.add_slot(draft, face_id, region_id, x=0, y=0, width=1, height=1)
        for action in (lambda: actions.add_box(draft, face_id, region_id),
                       lambda: actions.duplicate_box(draft, face_id, region_id, draft.template.facings[0].slots[0].id),
                       lambda: actions.grid_defaults(draft.template.facings[0], region_id, 3, 4)):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(len(draft.template.facings[0].slots), 1)

    def test_explicit_section_bounds_and_inches_are_respected(self):
        draft, face_id, region_id = setup("in", 12, 24)
        face = draft.template.facings[0]
        bounded = replace(face.regions[0], bounds=Bounds(3, 6, 6, 12))
        draft = replace(draft, template=replace(draft.template, facings=(replace(face, regions=(bounded,)),)))
        draft = actions.add_box(draft, face_id, region_id)
        box = draft.template.facings[0].slots[0]
        self.assertGreaterEqual(box.x, 3)
        self.assertGreaterEqual(box.y, 6)
        self.assertLessEqual(box.x + box.width, 9)
        self.assertLessEqual(box.y + box.height, 18)
        self.assertEqual(draft.template.unit, "in")
        self.assertTrue(edits.validate_draft(draft).valid)
