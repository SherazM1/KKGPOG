"""AppTest covers selection synchronization and the existing fallback controls."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.pages._canvas_events import consume_event, CanvasTransform
from app.pages.template_builder import DRAFT_KEY, MODE_KEY, side_selection_key
from app.templates import draft as edits
from app.templates.preview import geometry_svg


class CanvasAppTests(unittest.TestCase):
    def test_canvas_selection_drives_controls_and_fallback_actions(self):
        d = edits.new_draft('Selection', 'Custom', '1', 'normalized')
        d = edits.add_facing(d, 'Side', 1, 1)
        fid = d.template.facings[0].id
        for name, x in (('Cards', .05), ('Bonus', .5)):
            d = edits.add_region(d, fid, name)
            rid = d.template.facings[0].regions[-1].id
            d = edits.add_slot(d, fid, rid, x=x, y=.05, width=.1, height=.1)
        face = d.template.facings[0]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'home.py'), default_timeout=15)
        app.session_state['active_workflow'] = 'Templates'
        app.session_state[MODE_KEY] = 'builder'
        app.session_state[DRAFT_KEY] = d
        selection_key = side_selection_key(fid)
        app.session_state[selection_key] = dict(facing=fid, region=face.regions[0].id, slot=face.slots[0].id)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('component_instance')), 1)
        key = f'template_canvas:{d.template.id}:{fid}'
        app.session_state[key] = dict(event_id='select:2', kind='select', template_id=d.template.id,
            facing_id=fid, revision=d.revision, slot_id=face.slots[1].id)
        state = app.session_state.filtered_state
        consume_event(state, component_key=key, draft_key=DRAFT_KEY, selection_key=selection_key,
                      facing_id=fid, transform=CanvasTransform.from_svg(geometry_svg(d.template, fid), face))
        for name, value in state.items():
            if name.startswith(('template_builder:', 'template_canvas:')):
                app.session_state[name] = value
        app.run()
        self.assertEqual(app.session_state[selection_key]['region'], face.regions[1].id)
        self.assertEqual(next(w for w in app.button if w.label == 'Bonus').proto.type, 'primary')
        self.assertEqual(next(w for w in app.selectbox if w.label == 'Box').value, face.slots[1].id)
        for label in ('Move down', 'Duplicate Box', 'Delete Box'):
            next(b for b in app.button if b.label == label).click().run()
            self.assertFalse(app.exception)
        updated = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual(updated.slots[0], face.slots[0])
        self.assertEqual(len(updated.slots),2)
        self.assertGreater(updated.slots[1].y, face.slots[1].y)

    def test_catalog_has_no_editable_canvas(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'home.py'), default_timeout=15).run()
        app.selectbox(key='active_workflow').select('Templates').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('component_instance')),0)
        self.assertFalse(any(b.label in ('Duplicate Box', 'Delete Box') for b in app.button))
