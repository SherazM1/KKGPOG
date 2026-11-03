"""Fixed-size placement, stacked Sections, atomic selection and complex sides."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from app.pages import _direct_builder as direct
from app.pages._canvas_events import CanvasTransform, apply_event, consume_event
from app.pages._direct_canvas import canvas_svg
from app.pages.template_builder import DRAFT_KEY, MODE_KEY, side_selection_key
from app.templates import draft as edits
from app.templates.models import GeometryUnit


def start(unit='normalized',width=1,height=1):
    d=edits.new_draft('Complex layout','Custom','1',unit)
    d=edits.add_facing(d,'Side A',width,height)
    fid=d.template.facings[0].id
    d=direct.add_section(d,fid,'Main Cards')
    return d,fid,d.template.facings[0].regions[0].id


class FixedBuilderTests(unittest.TestCase):
    def test_add_box_and_different_grids_have_identical_sizes(self):
        d,fid,rid=start()
        d=direct.add_box(d,fid,rid)
        d=direct.create_grid(d,fid,rid,rows=2,columns=3)
        d=direct.create_grid(d,fid,rid,rows=5,columns=8)
        sizes={(round(s.width,10),round(s.height,10)) for s in d.template.facings[0].slots}
        self.assertEqual(sizes,{(.1,.12)})
        self.assertEqual(len(d.template.facings[0].slots),47)
        self.assertTrue(edits.validate_draft(d).valid)

    def test_standard_is_stable_across_side_growth_and_other_sides(self):
        d,fid,rid=start()
        d=direct.create_grid(d,fid,rid,rows=3,columns=15)
        size=direct.standard_size(d)
        self.assertGreater(d.template.facings[0].width,1)
        d=edits.add_facing(d,'Side B',2,2)
        second=d.template.facings[-1].id
        d=direct.add_section(d,second,'Bonus')
        d=direct.add_box(d,second,d.template.facings[-1].regions[-1].id)
        box=d.template.facings[-1].slots[-1]
        self.assertEqual((box.width,box.height),size)
        self.assertTrue(edits.validate_draft(d).valid)

    def test_sections_stack_and_earlier_growth_translates_later_boxes(self):
        d,fid,rid=start()
        d=direct.add_section(d,fid,'Holders')
        second=d.template.facings[0].regions[-1].id
        d=direct.add_box(d,fid,second)
        before=d.template.facings[0]
        d=direct.create_grid(d,fid,rid,rows=5,columns=8)
        face=d.template.facings[0]
        self.assertGreater(face.regions[1].bounds.y,before.regions[1].bounds.y)
        moved=next(s for s in face.slots if s.id==before.slots[0].id)
        self.assertEqual(replace(moved,y=before.slots[0].y),before.slots[0])
        self.assertGreater(face.regions[1].bounds.y,face.regions[0].bounds.y+face.regions[0].bounds.height)
        self.assertTrue(edits.validate_draft(d).valid)

    def test_adding_boxes_to_full_earlier_section_does_not_skip_later_band(self):
        d,fid,rid=start()
        for _ in range(8):d=direct.add_box(d,fid,rid)
        d=direct.add_section(d,fid,'Bonus')
        second=d.template.facings[0].regions[-1].id
        d=direct.add_box(d,fid,second)
        for _ in range(10):d=direct.add_box(d,fid,rid)
        self.assertTrue(edits.validate_draft(d).valid)
        self.assertEqual(len(d.template.facings[0].slots_for_region(rid)),18)

    def test_duplicate_section_preserves_arrangement_and_is_independent(self):
        d,fid,rid=start()
        d=direct.create_grid(d,fid,rid,rows=2,columns=3)
        face=d.template.facings[0]
        first=face.slots[0]
        d=edits.update_slot(d,fid,first.id,x=first.x+.01,y=first.y+.01,width=first.width,
                            height=first.height,order=first.order,region_id=rid)
        source=edits.get_facing(d,fid)
        d=direct.add_section(d,fid,'Below')
        d=direct.duplicate_section(d,fid,rid,'Main Cards Copy')
        face=d.template.facings[0];copy=face.ordered_regions()[1]
        self.assertNotEqual(copy.id,rid)
        originals=source.slots_for_region(rid);copies=face.slots_for_region(copy.id)
        offsets={(round(b.x-a.x,9),round(b.y-a.y,9)) for a,b in zip(originals,copies)}
        self.assertEqual(len(offsets),1)
        self.assertGreater(copies[0].y,originals[0].y)
        self.assertFalse({s.id for s in originals}&{s.id for s in copies})
        self.assertEqual([(s.width,s.height,s.role,s.row,s.column) for s in originals],
                         [(s.width,s.height,s.role,s.row,s.column) for s in copies])
        d=edits.remove_slots(d,fid,[copies[0].id])
        d=edits.remove_region(d,fid,copy.id)
        self.assertEqual(d.template.facings[0].slots_for_region(rid),originals)
        self.assertTrue(edits.validate_draft(d).valid)

    def test_measured_side_keeps_units_and_dimensions_or_rejects_full_grid(self):
        d,fid,rid=start('in',12,24)
        d=direct.create_grid(d,fid,rid,rows=3,columns=4)
        face=d.template.facings[0]
        self.assertEqual((face.width,face.height,d.template.unit),(12,24,GeometryUnit.INCHES))
        with self.assertRaises(ValueError):direct.create_grid(d,fid,rid,rows=50,columns=50)
        self.assertEqual(d.template.facings[0],face)

    def test_bad_grid_is_atomic(self):
        d,fid,rid=start();before=d
        for rows,columns in ((0,2),(2,0),(True,2),(2,51),('2',2)):
            with self.assertRaises(ValueError):direct.create_grid(d,fid,rid,rows=rows,columns=columns)
            self.assertEqual(d,before)

    def test_holiday_like_fifty_plus_boxes_and_side_duplication(self):
        d,fid,rid=start()
        d=direct.create_grid(d,fid,rid,rows=2,columns=5)
        for name,rows,columns in (('Main',4,8),('Holders',1,5),('Bonus',1,3)):
            d=direct.add_section(d,fid,name);rid=d.template.facings[0].regions[-1].id
            d=direct.create_grid(d,fid,rid,rows=rows,columns=columns)
        d=direct.duplicate_section(d,fid,rid,'Bonus Copy')
        self.assertEqual(len(d.template.facings[0].slots),53)
        self.assertTrue(edits.validate_draft(d).valid)
        original=d.template.facings[0]
        d=edits.duplicate_facing(d,fid,'Side B')
        copy=d.template.facings[-1]
        d=edits.remove_slots(d,copy.id,[s.id for s in copy.slots[:20]])
        self.assertEqual(d.template.facings[0],original)
        self.assertEqual(len(d.template.facings[-1].slots),33)


class MultiSelectTests(unittest.TestCase):
    def setUp(self):
        self.d,self.fid,self.rid=start()
        self.d=direct.create_grid(self.d,self.fid,self.rid,rows=5,columns=8)
        self.face=self.d.template.facings[0]
        self.transform=CanvasTransform.from_svg(canvas_svg(self.d,self.face),self.face)

    def event(self,kind,**extra):
        return dict(event_id='many:1',kind=kind,template_id=self.d.template.id,facing_id=self.fid,
                    revision=self.d.revision,**extra)

    def test_select_many_preserves_draft_and_writes_local_selection(self):
        ids=[s.id for s in self.face.slots[:12]]
        state={'draft':self.d,'canvas':self.event('select_many',slot_ids=ids)}
        consume_event(state,component_key='canvas',draft_key='draft',selection_key='local',
                      facing_id=self.fid,transform=self.transform,allow_resize=False)
        self.assertIs(state['draft'],self.d)
        self.assertEqual(state['local']['slots'],tuple(ids))
        self.assertEqual(state['canvas:ack'],'many:1')

    def test_move_many_changes_only_positions_once_and_preserves_membership(self):
        ids=[s.id for s in self.face.slots[:4]]
        result=apply_event(self.d,self.fid,self.event('move_many',slot_ids=ids,
            delta={'x':self.transform.scale*.005,'y':self.transform.scale*.005}),self.transform,allow_resize=False)
        self.assertTrue(result.accepted)
        self.assertEqual(result.draft.revision,self.d.revision+1)
        face=result.draft.template.facings[0]
        self.assertEqual(face.regions,self.face.regions)
        self.assertEqual(face.slots[4:],self.face.slots[4:])
        for old,new in zip(self.face.slots,face.slots[:4]):
            self.assertEqual(replace(new,x=old.x,y=old.y),old)
            self.assertAlmostEqual(new.x-old.x,.005)
            self.assertAlmostEqual(new.y-old.y,.005)

    def test_resize_is_rejected_by_normal_canvas_boundary(self):
        s=self.face.slots[0]
        result=apply_event(self.d,self.fid,self.event('resize',slot_id=s.id,
            rect=self.transform.to_canvas(replace(s,width=1))),self.transform,allow_resize=False)
        self.assertFalse(result.accepted)
        self.assertIs(result.draft,self.d)
        self.assertIn('fixed',result.error)

    def test_invalid_and_foreign_selection_or_move_retains_draft(self):
        for ids,delta in ((['missing'],{'x':1,'y':1}),([self.face.slots[0].id]*2,{'x':1,'y':1}),
                          ([self.face.slots[0].id],{'x':float('inf'),'y':1}),([],{'x':1,'y':1})):
            result=apply_event(self.d,self.fid,self.event('move_many',slot_ids=ids,delta=delta),self.transform,allow_resize=False)
            self.assertFalse(result.accepted)
            self.assertIs(result.draft,self.d)

    def test_bulk_delete_updates_all_references_once_and_preserves_others(self):
        ids=[s.id for s in self.face.slots[:18]]
        result=edits.remove_slots(self.d,self.fid,ids)
        self.assertEqual(result.revision,self.d.revision+1)
        self.assertEqual(result.template.facings[0].slots,self.face.slots[18:])
        self.assertFalse(set(ids)&set(result.template.facings[0].regions[0].slot_ids))
        self.assertTrue(edits.validate_draft(result).valid)
        with self.assertRaises(ValueError):edits.remove_slots(self.d,self.fid,[*ids,'missing'])

    def test_out_of_bounds_group_move_retains_editable_validation(self):
        result=apply_event(self.d,self.fid,self.event('move_many',slot_ids=[self.face.slots[0].id],
            delta={'x':-100,'y':-100}),self.transform,allow_resize=False)
        self.assertTrue(result.accepted)
        self.assertIn('slot_bounds',{i.code for i in edits.validate_draft(result.draft).errors})
        self.assertIn('data-slot-id',canvas_svg(result.draft,result.draft.template.facings[0]))


class DirectBuilderAppTests(unittest.TestCase):
    def start(self):
        d,fid,rid=start()
        d=direct.create_grid(d,fid,rid,rows=5,columns=8)
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'home.py'),default_timeout=15)
        app.session_state['active_workflow']='Templates';app.session_state[MODE_KEY]='builder'
        app.session_state[DRAFT_KEY]=d;app.run();self.assertFalse(app.exception)
        return app,d,fid,rid

    def click(self,app,label):
        next(b for b in app.button if b.label==label).click().run()
        self.assertFalse(app.exception)

    def test_no_normal_size_controls_or_resize_buttons(self):
        app,_,_,_=self.start()
        self.assertFalse(any(b.label in ('Wider','Narrower','Taller','Shorter') for b in app.button))
        def walk(node,advanced=False):
            if isinstance(node,Block):
                advanced=advanced or (node.type=='expander' and node.label.startswith('Advanced'))
                for child in node.children.values():yield from walk(child,advanced)
            else:yield node,advanced
        normal_inputs=[node.label for node,advanced in walk(app._tree) if not advanced and getattr(node,'type','')=='number_input']
        self.assertEqual(normal_inputs,['Rows','Columns'])
        self.assertIn('Technical compatibility', ' '.join(c.value for c in app.caption))

    def test_delete_selected_and_section_confirmation(self):
        app,d,fid,rid=self.start();ids=tuple(s.id for s in d.template.facings[0].slots[:10])
        app.session_state[side_selection_key(fid)]=dict(facing=fid,region=rid,slot=ids[0],slots=ids)
        app.run();self.click(app,'Delete Selected')
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].slots),30)
        self.assertTrue(next(b for b in app.button if b.label=='Delete Selected').disabled)
        self.assertTrue(any(b.label=='Confirm Delete Section' for b in app.button))
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].regions),1)
        self.click(app,'Confirm Delete Section')
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0].slots,())
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0].regions,())

    def test_duplicate_section_ui_has_new_ids_and_becomes_selected(self):
        app,d,fid,rid=self.start()
        next(w for w in app.text_input if w.label=='Copy section name').set_value('Repeat')
        self.click(app,'Create Section Copy')
        face=app.session_state[DRAFT_KEY].template.facings[0]
        copied=face.regions[-1]
        self.assertEqual(copied.name,'Repeat')
        self.assertEqual(len(face.slots),80)
        self.assertFalse(set(copied.slot_ids)&set(face.regions[0].slot_ids))
        self.assertEqual(app.session_state[side_selection_key(fid)]['region'],copied.id)
        scene=json.loads(app.get('component_instance')[0].proto.json_args)
        self.assertEqual(scene['active_section_id'],copied.id)
        self.assertEqual(len(scene['boxes']),80)
