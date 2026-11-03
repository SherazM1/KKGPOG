"""Side hierarchy copies and operator workflows remain independent."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from app.pages.template_builder import DRAFT_KEY, MODE_KEY, side_selection_key
from app.pages._canvas_events import CanvasTransform, consume_event
from app.templates import draft as edits
from app.templates.models import Bounds, PresentationHint
from app.templates.preview import geometry_svg


def layout():
    d = edits.new_draft('Workspace', 'Custom', '1', 'normalized')
    for name in ('Side A', 'Side B'):
        d = edits.add_facing(d, name, 1, 1)
        face = d.template.facings[-1]
        d = edits.add_region(d, face.id, 'Cards')
        rid = d.template.facings[-1].regions[0].id
        d = edits.create_grid(d, face.id, rid, rows=1, columns=2, x=.05, y=.05,
                              width=.6, height=.2, horizontal_gap=.05)
        d = edits.add_region(d, face.id, 'Bonus')
    return d


class DuplicateSideTests(unittest.TestCase):
    def test_complete_hierarchy_remap_geometry_and_settings(self):
        d = layout()
        source = d.template.facings[0]
        source = replace(source, metadata={'notes': {'labels': ['original']}}, regions=(
            replace(source.regions[0], bounds=Bounds(0,0,1,1), presentation_hint=PresentationHint.SINGLE_ROW,
                    allow_shared_slots=True, metadata={'hint':'keep'}),
            replace(source.regions[1], slot_ids=(source.slots[0].id,), allow_shared_slots=True)),
            slots=tuple(replace(s, section_id=source.regions[0].id, metadata={'description':'keep'}) for s in source.slots))
        d = replace(d, template=replace(d.template, facings=(source, *d.template.facings[1:])))
        result = edits.duplicate_facing(d, source.id, 'Side A Copy')
        copy = result.template.facings[-1]
        old_ids = {i.id for f in d.template.facings for i in (f,*f.regions,*f.slots)}
        new_ids = {i.id for i in (copy,*copy.regions,*copy.slots)}
        self.assertFalse(old_ids & new_ids)
        self.assertEqual(len(new_ids), 1+len(source.regions)+len(source.slots))
        self.assertEqual((copy.width,copy.height,copy.metadata), (source.width,source.height,source.metadata))
        self.assertEqual(copy.order,3)
        for old,new in zip(source.slots,copy.slots):
            self.assertEqual(replace(new,id=old.id,section_id=old.section_id),old)
            self.assertEqual(new.section_id,copy.regions[0].id)
            self.assertIsNot(new,old)
        ids = dict(zip((s.id for s in source.slots),(s.id for s in copy.slots)))
        for old,new in zip(source.regions,copy.regions):
            self.assertEqual(new.slot_ids,tuple(ids[sid] for sid in old.slot_ids))
            self.assertEqual(replace(new,id=old.id,slot_ids=old.slot_ids),old)
        copy.metadata['notes']['labels'].append('copy only')
        self.assertEqual(source.metadata['notes']['labels'],['original'])
        self.assertTrue(edits.validate_draft(result).valid)
        self.assertEqual(result.revision,d.revision+1)
        self.assertGreater(result.next_id,d.next_id)

    def test_edit_rename_delete_copy_preserves_original(self):
        d = layout(); source=d.template.facings[0]
        d = edits.duplicate_facing(d,source.id,'Copy')
        copy=d.template.facings[-1]; slot=copy.slots[0]
        d=edits.update_slot(d,copy.id,slot.id,x=.1,y=.5,width=.1,height=.1,order=slot.order,region_id=copy.regions[0].id)
        d=edits.update_facing(d,copy.id,name='Renamed',width=2,height=2,order=copy.order)
        d=edits.remove_slot(d,copy.id,slot.id)
        self.assertEqual(d.template.facings[0],source)
        self.assertEqual(d.template.facings[-1].name,'Renamed')
        self.assertEqual(len(d.template.facings[-1].slots),len(source.slots)-1)

    def test_repeated_copy_and_subsequent_additions_have_unique_ids(self):
        d=layout()
        for _ in range(3):
            d=edits.duplicate_facing(d,d.template.facings[-1].id,'Another copy')
        face=d.template.facings[-1]
        d=edits.add_slot(d,face.id,face.regions[0].id,x=.8,y=.8,width=.1,height=.1)
        ids=[item.id for f in d.template.facings for item in (f,*f.regions,*f.slots)]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(edits.validate_draft(d).valid)

    def test_empty_side_and_bad_request_do_not_mutate_source(self):
        d=edits.add_facing(layout(),'Empty',2,3)
        result=edits.duplicate_facing(d,d.template.facings[-1].id,'Empty Copy')
        self.assertEqual((result.template.facings[-1].slots,result.template.facings[-1].regions),((),()))
        for fid,name in ((d.template.facings[0].id,' '),('missing','Copy')):
            with self.assertRaises((ValueError,KeyError)):
                edits.duplicate_facing(d,fid,name)
        face=d.template.facings[0]
        invalid=replace(d,template=replace(d.template,facings=(replace(face,regions=(
            replace(face.regions[0],slot_ids=('missing',)),)),*d.template.facings[1:])))
        with self.assertRaises(ValueError):
            edits.duplicate_facing(invalid,face.id,'Bad copy')
        self.assertEqual(len(d.template.facings),3)

    def test_legacy_group_ids_are_remapped_without_losing_grouping(self):
        d=layout();face=d.template.facings[0]
        face=replace(face,slots=tuple(replace(s,section_id='legacy-group') for s in face.slots))
        d=replace(d,template=replace(d.template,facings=(face,*d.template.facings[1:])))
        copy=edits.duplicate_facing(d,face.id,'Copy').template.facings[-1]
        self.assertEqual(len({s.section_id for s in copy.slots}),1)
        self.assertNotEqual(copy.slots[0].section_id,'legacy-group')


class SideWorkspaceTests(unittest.TestCase):
    def start(self):
        d=layout()
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow']='Templates'
        app.session_state[MODE_KEY]='builder'
        app.session_state[DRAFT_KEY]=d
        app.run();self.assertFalse(app.exception)
        return app,d

    def local(self,app,kind,label,fid):
        return next(w for w in getattr(app,kind) if w.label==label and
                    fid in ((w.key or '')+getattr(w.proto,'form_id','')))

    def click(self,app,label,fid):
        self.local(app,'button',label,fid).click().run()
        self.assertFalse(app.exception)

    def test_cards_canvases_and_section_selection_are_local(self):
        app,d=self.start()
        self.assertEqual([h.value for h in app.subheader if h.value.startswith('Side ')],['Side A','Side B'])
        self.assertFalse(any(w.label in ('Side','Section') for w in app.selectbox))
        scenes=[json.loads(c.proto.json_args) for c in app.get('component_instance')]
        self.assertEqual([s['facing_id'] for s in scenes],[f.id for f in d.template.facings])
        for scene,face in zip(scenes,d.template.facings):
            self.assertEqual(set(scene['boxes']),{s.id for s in face.slots})
        first,second=d.template.facings
        self.click(app,'Bonus',first.id)
        self.assertEqual(app.session_state[side_selection_key(first.id)]['region'],first.regions[1].id)
        self.assertEqual(app.session_state[side_selection_key(second.id)]['region'],second.regions[0].id)
        self.assertEqual(self.local(app,'button','Bonus',first.id).proto.type,'primary')
        self.click(app,'Cards',second.id)
        self.assertEqual(app.session_state[side_selection_key(first.id)]['region'],first.regions[1].id)

    def test_add_section_box_and_grid_only_affect_their_side(self):
        app,d=self.start();first,second=d.template.facings
        self.local(app,'text_input','New section name',second.id).set_value('Holders')
        self.click(app,'Add Section',second.id)
        scene=json.loads(app.get('component_instance')[1].proto.json_args)
        self.assertIsNone(scene['selected_box_id'])
        self.click(app,'Add Box',second.id)
        updated=app.session_state[DRAFT_KEY]
        self.assertEqual(updated.template.facings[0],first)
        target=updated.template.facings[1]
        self.assertEqual([len(r.slot_ids) for r in target.regions],[2,0,1])
        self.click(app,'Bonus',first.id)
        self.local(app,'number_input','Rows',first.id).set_value(3)
        self.local(app,'number_input','Columns',first.id).set_value(4)
        self.click(app,'Create Grid',first.id)
        updated=app.session_state[DRAFT_KEY]
        self.assertEqual(updated.template.facings[1],target)
        self.assertEqual([len(r.slot_ids) for r in updated.template.facings[0].regions],[2,12])

    def test_duplicate_side_ui_rename_and_delete_independently(self):
        app,d=self.start();source=d.template.facings[0]
        self.local(app,'text_input','Copy side name',source.id).set_value('New Side')
        self.click(app,'Duplicate Side',source.id)
        copy=app.session_state[DRAFT_KEY].template.facings[-1]
        self.assertEqual(copy.name,'New Side')
        self.assertEqual(len(app.get('component_instance')),3)
        self.local(app,'text_input','Side name',copy.id).set_value('Renamed Copy')
        self.click(app,'Apply side changes',copy.id)
        self.click(app,'Delete Box',copy.id)
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0],source)
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[-1].name,'Renamed Copy')
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[-1].slots),1)
        self.click(app,'Delete Side',copy.id)
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings,d.template.facings)

    def test_canvas_callback_targets_side_after_editing_another_card(self):
        app,d=self.start();first,second=d.template.facings
        self.click(app,'Bonus',first.id)
        key=f'template_canvas:{d.template.id}:{second.id}'
        transform=CanvasTransform.from_svg(geometry_svg(d.template,second.id),second)
        state=app.session_state.filtered_state
        state[key]=dict(event_id='side-b:1',kind='drag',template_id=d.template.id,
            facing_id=second.id,revision=d.revision,slot_id=second.slots[0].id,
            rect=transform.to_canvas(replace(second.slots[0],y=.4)))
        consume_event(state,component_key=key,draft_key=DRAFT_KEY,selection_key=side_selection_key(second.id),
                      facing_id=second.id,transform=transform)
        for name,value in state.items():
            if name.startswith(('template_builder:','template_canvas:')):
                app.session_state[name]=value
        app.run();self.assertFalse(app.exception)
        updated=app.session_state[DRAFT_KEY]
        self.assertEqual(updated.template.facings[0],first)
        self.assertAlmostEqual(updated.template.facings[1].slots[0].y,.4)
        self.assertEqual(app.session_state[side_selection_key(first.id)]['region'],first.regions[1].id)

    def test_invalid_side_does_not_hide_other_canvas_or_leak_errors(self):
        app,d=self.start();first,second=d.template.facings
        self.local(app,'number_input','Side width',first.id).set_value(0)
        self.click(app,'Apply side changes',first.id)
        scenes=[json.loads(c.proto.json_args) for c in app.get('component_instance')]
        self.assertEqual([s['facing_id'] for s in scenes],[second.id])
        self.assertTrue(any('positive' in e.value for e in app.error))
        self.assertTrue(next(b for b in app.button if b.label=='Done').disabled)
