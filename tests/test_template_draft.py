"""Pure draft operations; independent of Streamlit, personal files and renderers."""
import unittest
from dataclasses import replace

from app.templates import draft as edit
from app.templates.models import GeometryUnit, PresentationHint, SlotRole
from app.templates.registry import get_templates


def blank(unit="normalized"):
    return edit.new_draft("Manual Display", "Custom", "1.0", unit)


def region_draft():
    draft = edit.add_facing(blank(), "Front", 1, 1)
    draft = edit.add_region(draft, draft.template.facings[0].id, "Main")
    facing = draft.template.facings[0]
    return draft, facing.id, facing.regions[0].id


class TemplateDraftTests(unittest.TestCase):
    def test_blank_draft_basics_and_explicit_units(self):
        draft = blank()
        self.assertFalse(draft.template.locked)
        self.assertEqual(draft.template.facings, ())
        self.assertEqual(draft.template.unit, GeometryUnit.NORMALIZED)
        self.assertIn("Uncalibrated", draft.template.metadata["calibration"])
        self.assertEqual(blank("in").template.unit, GeometryUnit.INCHES)
        self.assertNotEqual(blank().template.id, draft.template.id)
        for args in (("", "Custom", "1", "in"), ("Name", "", "1", "in"),
                     ("Name", "Custom", "", "in"), ("Name", "Custom", "1", "px")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                edit.new_draft(*args)

    def test_add_multiple_facings_and_require_physical_dimensions(self):
        original = blank("in")
        with self.assertRaises(ValueError):
            edit.add_facing(original, "Front", 0, 0)
        draft = edit.add_facing(original, "Front", 12, 24)
        draft = edit.add_facing(draft, "Back", 10, 18)
        self.assertEqual(len(original.template.facings), 0)
        self.assertEqual([facing.order for facing in draft.template.ordered_facings()], [1, 2])
        self.assertEqual([facing.width for facing in draft.template.facings], [12, 10])
        self.assertEqual(len({facing.id for facing in draft.template.facings}), 2)
        self.assertTrue(edit.validate_draft(draft).valid)

    def test_rename_resize_reorder_facing_without_changing_identity(self):
        draft, facing_id, _ = region_draft()
        updated = edit.update_facing(draft, facing_id, name="Left", width=2, height=3, order=5)
        facing = updated.template.facings[0]
        self.assertEqual((facing.id, facing.name, facing.width, facing.height, facing.order),
                         (facing_id, "Left", 2, 3, 5))
        self.assertEqual(draft.template.facings[0].width, 1)

    def test_region_creation_editing_and_order(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_region(draft, facing_id, "Signs", SlotRole.SIGNAGE, PresentationHint.SECTION)
        draft = edit.update_region(draft, facing_id, region_id, name="Cards", role=SlotRole.PRODUCT,
                                   hint=PresentationHint.GRID, order=3)
        facing = draft.template.facings[0]
        self.assertEqual([region.name for region in facing.ordered_regions()], ["Signs", "Cards"])
        self.assertEqual(facing.regions[0].id, region_id)
        self.assertEqual(facing.regions[0].presentation_hint, PresentationHint.GRID)
        self.assertTrue(edit.validate_draft(draft).valid)

    def test_single_slot_default_geometry_membership_and_fixed_ids(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        first = draft.template.facings[0].slots[0]
        draft = edit.add_slot(draft, facing_id, region_id)
        facing = draft.template.facings[0]
        self.assertEqual([slot.order for slot in facing.slots], [1, 2])
        self.assertIs(facing.slots[0], first)
        self.assertNotEqual(first.id, facing.slots[1].id)
        self.assertEqual(len(facing.slots_for_region(region_id)), 2)
        self.assertTrue(edit.validate_draft(draft).valid)

    def test_grid_has_independent_slots_correct_gaps_and_row_major_order(self):
        draft, facing_id, region_id = region_draft()
        generated = edit.create_grid(draft, facing_id, region_id, rows=2, columns=3,
                                     x=.05, y=.05, width=.9, height=.4,
                                     horizontal_gap=.03, vertical_gap=.02)
        facing = generated.template.facings[0]
        self.assertEqual(len(facing.slots), 6)
        self.assertEqual(len({id(slot) for slot in facing.slots}), 6)
        self.assertEqual(len({slot.id for slot in facing.slots}), 6)
        for index, slot in enumerate(facing.ordered_slots()):
            row, col = divmod(index, 3)
            self.assertEqual((slot.row, slot.column, slot.order), (row + 1, col + 1, index + 1))
            self.assertAlmostEqual(slot.width, .28)
            self.assertAlmostEqual(slot.height, .19)
            self.assertAlmostEqual(slot.x, .05 + col * .31)
            self.assertAlmostEqual(slot.y, .05 + row * .21)
        self.assertTrue(edit.validate_draft(generated).valid)
        selected = facing.slots[0]
        edited = edit.update_slot(generated, facing_id, selected.id, x=.05, y=.65, width=.2, height=.1,
                                  order=selected.order, region_id=region_id)
        self.assertEqual(edited.template.facings[0].slots[1:], facing.slots[1:])
        self.assertEqual(edited.template.facings[0].slots[0].id, selected.id)
        self.assertEqual(len(draft.template.facings[0].slots), 0)

    def test_invalid_grid_is_rejected_atomically(self):
        draft, facing_id, region_id = region_draft()
        defaults = dict(rows=2, columns=3, x=0, y=0, width=.9, height=.9)
        for change in ({"rows": 0}, {"columns": 1.5}, {"rows": True}, {"width": 0},
                       {"x": -.1}, {"x": .8}, {"horizontal_gap": .8}, {"vertical_gap": -.1},
                       {"height": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                edit.create_grid(draft, facing_id, region_id, **{**defaults, **change})
            self.assertEqual(draft.template.facings[0].slots, ())

    def test_slot_move_resize_reassignment_and_order(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        draft = edit.add_region(draft, facing_id, "Bonus", hint=PresentationHint.SECTION)
        facing = draft.template.facings[0]
        slot, target = facing.slots[0], facing.regions[-1]
        changed = edit.update_slot(draft, facing_id, slot.id, x=.5, y=.6, width=.3, height=.2,
                                   order=7, region_id=target.id)
        facing = changed.template.facings[0]
        self.assertEqual(facing.slots_for_region(region_id), ())
        self.assertEqual(facing.slots_for_region(target.id)[0].id, slot.id)
        self.assertEqual((facing.slots[0].x, facing.slots[0].y, facing.slots[0].width, facing.slots[0].height,
                          facing.slots[0].order), (.5, .6, .3, .2, 7))
        self.assertEqual(len(facing.regions_for_slot(slot.id)), 1)
        self.assertTrue(edit.validate_draft(changed).valid)

    def test_validation_reports_bounds_sizes_overlaps_and_orders_without_clamping(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id, x=0, y=0, width=.4, height=.4)
        slot = draft.template.facings[0].slots[0]
        for changes, code in (({"x": .9}, "slot_bounds"), ({"width": 0}, "slot_dimensions"),
                              ({"order": 0}, "slot_order")):
            params = dict(x=0, y=0, width=.4, height=.4, order=1, region_id=region_id)
            invalid = edit.update_slot(draft, facing_id, slot.id, **{**params, **changes})
            self.assertIn(code, {issue.code for issue in edit.validate_draft(invalid).errors})
        overlap = edit.add_slot(draft, facing_id, region_id, x=.2, y=.2, width=.4, height=.4, order=1)
        self.assertTrue({"product_overlap", "duplicate_slot_order"} <= {
            issue.code for issue in edit.validate_draft(overlap).errors})
        invalid_face = edit.update_facing(draft, facing_id, name="Front", width=0, height=1, order=1)
        self.assertIn("facing_dimensions", {issue.code for issue in edit.validate_draft(invalid_face).errors})

    def test_invalid_identity_membership_and_hint_still_use_core_validation(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        facing = draft.template.facings[0]
        bad_region = replace(facing.regions[0], presentation_hint="not-a-hint", slot_ids=("missing",))
        invalid = replace(draft, template=replace(draft.template, facings=(replace(facing, regions=(bad_region,)),)))
        self.assertTrue({"presentation_hint", "region_slot_reference", "unassigned_region_slot"} <= {
            issue.code for issue in edit.validate_draft(invalid).errors})
        duplicated = replace(draft, template=replace(draft.template, facings=(facing, facing)))
        self.assertTrue({"duplicate_facing_id", "duplicate_region_id", "duplicate_slot_id"} <= {
            issue.code for issue in edit.validate_draft(duplicated).errors})

    def test_removals_clean_membership_and_cascade_within_draft(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        slot_id = draft.template.facings[0].slots[0].id
        deleted = edit.remove_slot(draft, facing_id, slot_id)
        self.assertEqual(deleted.template.facings[0].regions[0].slot_ids, ())
        self.assertEqual(deleted.template.facings[0].slots, ())
        removed = edit.remove_region(draft, facing_id, region_id)
        self.assertEqual(removed.template.facings[0].regions, ())
        self.assertEqual(removed.template.facings[0].slots, ())
        self.assertEqual(edit.remove_facing(draft, facing_id).template.facings, ())
        with self.assertRaises(KeyError):
            edit.remove_slot(draft, facing_id, "unknown")

    def test_reset_keeps_basics_and_never_recycles_ids(self):
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        reset = edit.reset_draft(draft)
        self.assertEqual(reset.template.facings, ())
        self.assertEqual((reset.template.id, reset.template.name, reset.template.unit, reset.template.version),
                         (draft.template.id, draft.template.name, draft.template.unit, draft.template.version))
        reset = edit.add_facing(reset, "New", 1, 1)
        self.assertNotEqual(reset.template.facings[0].id, facing_id)

    def test_builtins_stay_unchanged_and_locked_templates_are_not_drafts(self):
        before = get_templates()
        draft, facing_id, region_id = region_draft()
        draft = edit.add_slot(draft, facing_id, region_id)
        edit.reset_draft(draft)
        self.assertEqual(get_templates(), before)
        for template in before:
            if template.locked:
                with self.assertRaises(ValueError):
                    edit.TemplateDraft(template)
