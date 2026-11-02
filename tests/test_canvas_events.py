"""Deterministic boundary coverage independent of browser pointer support in AppTest."""
from dataclasses import replace
import unittest

from app.pages import _builder_actions as actions
from app.pages._canvas_events import CanvasTransform, apply_event, consume_event, SELECTION_EPOCH_KEY
from app.templates import draft as edits
from app.templates.preview import geometry_svg
from app.templates.registry import get_templates


class CanvasEventTests(unittest.TestCase):
    def setUp(self):
        d = edits.new_draft('Canvas', 'Custom', '1', 'normalized')
        d = edits.add_facing(d, 'Side A', 1, 1)
        self.fid = d.template.facings[0].id
        d = edits.add_region(d, self.fid, 'Main Cards')
        self.rid = d.template.facings[0].regions[0].id
        self.draft = edits.create_grid(d, self.fid, self.rid, rows=3, columns=4,
            x=.05, y=.05, width=.8, height=.6, horizontal_gap=.02, vertical_gap=.02)
        self.face = self.draft.template.facings[0]
        self.slot = self.face.slots[0]
        self.transform = CanvasTransform.from_svg(geometry_svg(self.draft.template, self.fid), self.face)

    def event(self, kind='select', **changes):
        return dict(event_id='gesture:1', kind=kind, template_id=self.draft.template.id,
                    facing_id=self.fid, revision=self.draft.revision, slot_id=self.slot.id, **changes)

    def apply(self, kind, **geometry):
        rect = self.transform.to_canvas(replace(self.slot, **geometry))
        return apply_event(self.draft, self.fid, self.event(kind, rect=rect), self.transform)

    def test_select_changes_no_geometry(self):
        result = apply_event(self.draft, self.fid, self.event(), self.transform)
        self.assertTrue(result.accepted)
        self.assertFalse(result.changed)
        self.assertIs(result.draft, self.draft)
        self.assertEqual((result.slot_id, result.region_id), (self.slot.id, self.rid))

    def test_drag_preserves_id_membership_metadata_size_and_grid_independence(self):
        result = self.apply('drag', x=.07, y=.08, width=.9)
        self.assertTrue(result.changed)
        face = result.draft.template.facings[0]
        changed = face.slots[0]
        self.assertAlmostEqual(changed.x, .07)
        self.assertAlmostEqual(changed.y, .08)
        self.assertEqual(replace(changed, x=self.slot.x, y=self.slot.y), self.slot)
        self.assertEqual(face.slots[1:], self.face.slots[1:])
        self.assertEqual(face.regions, self.face.regions)
        self.assertEqual(result.draft.next_id, self.draft.next_id)
        self.assertEqual(result.draft.revision, self.draft.revision+1)

    def test_resize_changes_only_target_geometry(self):
        result = self.apply('resize', width=.11, height=.12)
        changed = result.draft.template.facings[0].slots[0]
        self.assertAlmostEqual(changed.width, .11)
        self.assertAlmostEqual(changed.height, .12)
        self.assertEqual(replace(changed, width=self.slot.width, height=self.slot.height), self.slot)
        self.assertEqual(result.draft.template.facings[0].regions, self.face.regions)

    def test_overlaps_and_bounds_remain_editable_and_validate(self):
        for changes, code in ((dict(x=.3), 'product_overlap'), (dict(x=-.1), 'slot_bounds'),
                              (dict(width=2), 'slot_bounds')):
            with self.subTest(changes=changes):
                result = self.apply('resize', **changes)
                self.assertTrue(result.accepted)
                self.assertIn(code, {i.code for i in edits.validate_draft(result.draft).errors})
                with self.assertRaises(ValueError):
                    geometry_svg(result.draft.template, self.fid)
                self.assertIn('data-slot-id', geometry_svg(result.draft.template, self.fid, allow_layout_errors=True))

    def test_invalid_resize_or_drag_is_rejected_without_losing_draft(self):
        for kind in ('drag', 'resize'):
            for value in (0, -1, float('nan'), float('inf'), 'bad', True):
                rectangle = self.transform.to_canvas(self.slot)
                rectangle['width'] = value
                result = apply_event(self.draft, self.fid, self.event(kind, rect=rectangle), self.transform)
                self.assertFalse(result.accepted)
                self.assertIs(result.draft, self.draft)
                self.assertTrue(result.error)

    def test_stale_foreign_and_malformed_events_are_rejected(self):
        for event in (None, {}, {**self.event(), 'revision':-1}, {**self.event(), 'revision':True},
                      {**self.event(), 'facing_id':'other'}, {**self.event(), 'template_id':'other'},
                      {**self.event(), 'slot_id':'deleted'}, {**self.event(), 'kind':'replace_scene'}):
            result = apply_event(self.draft, self.fid, event, self.transform)
            self.assertFalse(result.accepted)
            self.assertIs(result.draft, self.draft)

    def test_same_draft_state_and_selection_are_updated_exactly_once(self):
        state = {'draft':self.draft, 'canvas':self.event('drag', rect=self.transform.to_canvas(replace(self.slot, x=.08)))}
        def consume():
            consume_event(state, component_key='canvas', draft_key='draft', selection_key='selection',
                          facing_id=self.fid, transform=self.transform)
        consume()
        updated = state['draft']
        self.assertEqual(state['selection'], dict(facing=self.fid, region=self.rid, slot=self.slot.id))
        self.assertEqual(state[SELECTION_EPOCH_KEY], 1)
        consume()
        self.assertIs(state['draft'], updated)
        self.assertEqual(state['canvas:ack'], 'gesture:1')

    def test_selection_can_switch_sections_but_geometry_cannot_reassign(self):
        d = edits.add_region(self.draft, self.fid, 'Bonus')
        other = d.template.facings[0].regions[-1].id
        event = {**self.event(), 'revision':d.revision}
        result = apply_event(d, self.fid, event, self.transform, other)
        self.assertEqual(result.region_id, self.rid)
        self.assertEqual(result.draft.template.facings[0].regions, d.template.facings[0].regions)

    def test_explicit_shared_memberships_are_preserved(self):
        d = edits.add_region(self.draft, self.fid, 'Shared')
        face = d.template.facings[0]
        regions = tuple(replace(r, allow_shared_slots=True, slot_ids=(self.slot.id,) if r.id != self.rid else r.slot_ids) for r in face.regions)
        d = replace(d, template=replace(d.template, facings=(replace(face, regions=regions),)))
        event = {**self.event('resize', rect=self.transform.to_canvas(replace(self.slot, width=.1))), 'revision':d.revision}
        result = apply_event(d, self.fid, event, self.transform)
        self.assertTrue(result.accepted)
        self.assertEqual(result.draft.template.facings[0].regions, regions)

    def test_inches_and_normalized_projection_round_trip(self):
        for unit, width, height in (('in',12.35,27.87), ('normalized',1,1)):
            d = edits.new_draft('Units', 'Custom', '1', unit)
            d = edits.add_facing(d, 'Side', width, height)
            face = d.template.facings[0]
            transform = CanvasTransform.from_svg(geometry_svg(d.template, face.id), face)
            values = transform.to_template(transform.to_canvas(self.slot))
            for field, value in values.items():
                self.assertAlmostEqual(value, getattr(self.slot, field))
            self.assertEqual(d.template.unit, unit)

    def test_duplicate_delete_and_fallback_after_canvas_edit(self):
        d = self.apply('resize', width=.1, height=.1).draft
        d = actions.adjust_box(d, self.fid, self.rid, self.slot.id, 'right')
        d = actions.duplicate_box(d, self.fid, self.rid, self.slot.id)
        copy = d.template.facings[0].slots[-1]
        self.assertNotEqual(copy.id, self.slot.id)
        d = edits.remove_slot(d, self.fid, copy.id)
        self.assertEqual(len(d.template.facings[0].slots),12)
        self.assertEqual(d.template.facings[0].slots[1:],self.face.slots[1:])

    def test_builtins_cannot_be_editable_drafts(self):
        before = get_templates()
        for template in before:
            if template.locked:
                with self.assertRaises(ValueError):
                    edits.TemplateDraft(template)
        self.apply('drag', x=.08)
        self.assertEqual(get_templates(), before)
