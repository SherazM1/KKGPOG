"""Builder operator flows, including recovery from invalid drafts."""
import unittest
from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.pages.template_builder import DRAFT_KEY
from app.templates.models import GeometryUnit
from app.templates.registry import get_templates


ROOT = Path(__file__).resolve().parents[1]


class TemplateBuilderTests(unittest.TestCase):
    def clean(self, app):
        self.assertEqual([error.message for error in app.exception], [])

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.clean(app)

    def field(self, app, kind, label):
        return next(item for item in getattr(app, kind) if item.label == label)

    def start(self, unit=GeometryUnit.NORMALIZED):
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select("Templates").run()
        self.click(app, "Create New Template")
        self.field(app, "text_input", "Template name").set_value("Operator draft")
        self.field(app, "selectbox", "Geometry unit").select(unit)
        self.click(app, "Start blank draft")
        return app

    def layout(self):
        app = self.start()
        self.click(app, "Add facing")
        self.click(app, "Add region")
        return app

    def test_blank_draft_multiple_facings_and_basics(self):
        app = self.start()
        self.assertEqual(app.session_state[DRAFT_KEY].template.name, "Operator draft")
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings, ())
        self.assertTrue(any("Uncalibrated" in notice.value for notice in app.info))
        self.click(app, "Add facing")
        first = app.session_state[DRAFT_KEY].template.facings[0]
        self.click(app, "Add facing")
        draft = app.session_state[DRAFT_KEY]
        self.assertEqual(len(draft.template.facings), 2)
        self.assertNotEqual(first.id, draft.template.facings[1].id)
        self.field(app, "text_input", "Facing name").set_value("Back side")
        self.field(app, "number_input", "Facing width").set_value(2.0)
        self.field(app, "number_input", "Facing order").set_value(7)
        self.click(app, "Apply facing changes")
        edited = app.session_state[DRAFT_KEY].template.facings[1]
        self.assertEqual((edited.name, edited.width, edited.order), ("Back side", 2.0, 7))

    def test_inches_require_dimensions_and_do_not_convert_units(self):
        app = self.start(GeometryUnit.INCHES)
        self.click(app, "Add facing")
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings, ())
        self.assertTrue(any("greater than zero" in error.value for error in app.error))
        self.field(app, "number_input", "New facing width").set_value(12.0)
        self.field(app, "number_input", "New facing height").set_value(24.0)
        self.click(app, "Add facing")
        draft = app.session_state[DRAFT_KEY]
        self.assertEqual(draft.template.unit, GeometryUnit.INCHES)
        self.assertEqual((draft.template.facings[0].width, draft.template.facings[0].height), (12, 24))

    def test_single_slot_grid_and_individual_edit_reassignment(self):
        app = self.layout()
        self.click(app, "Add position")
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].slots), 1)
        self.click(app, "Remove position")
        self.click(app, "Create grid")
        before = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual(len(before.slots), 6)
        self.assertEqual(len(app.get("imgs")), 1)
        self.field(app, "text_input", "New region name").set_value("Bonus area")
        self.click(app, "Add region")
        facing = app.session_state[DRAFT_KEY].template.facings[0]
        target = facing.regions[-1]
        next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:x")).set_value(.1)
        next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:y")).set_value(.7)
        next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:width")).set_value(.1)
        next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:height")).set_value(.1)
        self.field(app, "number_input", "Position order").set_value(9)
        self.field(app, "selectbox", "Belongs to region").select(target.id)
        self.click(app, "Apply position changes")
        facing = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual(facing.slots[:-1], before.slots[:-1])
        selected = facing.slots[-1]
        self.assertEqual((selected.id, selected.x, selected.y, selected.width, selected.height, selected.order),
                         (before.slots[-1].id, .1, .7, .1, .1, 9))
        self.assertEqual(facing.regions_for_slot(selected.id)[0].id, target.id)
        self.assertEqual(len(app.error), 0)

    def test_invalid_geometry_is_reported_then_can_be_fixed(self):
        app = self.layout()
        self.click(app, "Add position")
        x = next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:x"))
        x.set_value(2.0)
        self.click(app, "Apply position changes")
        self.assertTrue(any("beyond" in error.value for error in app.error))
        self.assertEqual(len(app.get("imgs")), 0)
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0].slots[0].x, 2)
        next(item for item in app.number_input if item.key and item.key.endswith(":edit_slot:x")).set_value(.1)
        self.click(app, "Apply position changes")
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.get("imgs")), 1)
        self.field(app, "number_input", "Facing width").set_value(0.0)
        self.click(app, "Apply facing changes")
        self.assertTrue(any("positive" in error.value for error in app.error))
        self.field(app, "number_input", "Facing width").set_value(1.0)
        self.click(app, "Apply facing changes")
        self.assertEqual(len(app.error), 0)

    def test_grid_errors_do_not_crash_or_partially_update_the_draft(self):
        app = self.layout()
        self.field(app, "number_input", "Space between columns").set_value(1.0)
        self.click(app, "Create grid")
        self.assertTrue(any("gaps" in error.value for error in app.error))
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0].slots, ())

    def test_selection_loads_each_slots_geometry_and_each_regions_hint(self):
        app = self.layout()
        self.click(app, "Create grid")
        facing = app.session_state[DRAFT_KEY].template.facings[0]
        for slot in (facing.slots[0], facing.slots[-1], facing.slots[1]):
            self.field(app, "selectbox", "Position to edit").select(slot.id).run()
            self.clean(app)
            for coordinate in ("x", "y", "width", "height"):
                actual = next(item.value for item in app.number_input if item.key and item.key.endswith(f":edit_slot:{coordinate}"))
                self.assertEqual(actual, getattr(slot, coordinate))
        self.field(app, "text_input", "New region name").set_value("Bonus")
        addition_hint = next(item for item in app.selectbox if item.key and item.key.endswith(":add_region:hint"))
        from app.templates.models import PresentationHint
        addition_hint.select(PresentationHint.SECTION)
        self.click(app, "Add region")
        facing = app.session_state[DRAFT_KEY].template.facings[0]
        for region in facing.regions:
            self.field(app, "selectbox", "Region to work on").select(region.id).run()
            self.clean(app)
            selected = next(item.value for item in app.selectbox if item.key and item.key.endswith(":edit_region:hint"))
            self.assertEqual(selected, region.presentation_hint)

    def test_overlap_validation_is_visible_without_crashing(self):
        app = self.layout()
        self.click(app, "Create grid")
        self.click(app, "Create grid")
        self.assertTrue(any("overlap" in error.value for error in app.error))
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].slots), 12)

    def test_invalid_membership_and_hint_are_reported_without_crashing(self):
        app = self.layout()
        self.click(app, "Add position")
        draft = app.session_state[DRAFT_KEY]
        facing = draft.template.facings[0]
        bad = replace(facing.regions[0], slot_ids=("missing",), presentation_hint="invalid")
        app.session_state[DRAFT_KEY] = replace(draft, template=replace(
            draft.template, facings=(replace(facing, regions=(bad,)),)), revision=draft.revision + 1)
        app.run()
        self.clean(app)
        messages = " ".join(error.value for error in app.error)
        self.assertIn("unsupported presentation hint", messages)
        self.assertIn("unknown facing slot", messages)

    def test_session_resume_reset_clear_and_builtins_are_safe(self):
        before = get_templates()
        app = self.layout()
        self.click(app, "Create grid")
        draft = app.session_state[DRAFT_KEY]
        app.run()
        self.assertEqual(app.session_state[DRAFT_KEY], draft)
        self.click(app, "Back to templates")
        self.assertEqual(len(app.dataframe[0].value), 15)
        self.assertEqual(app.session_state[DRAFT_KEY], draft)
        self.click(app, "Resume draft")
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].slots), 6)
        app.selectbox(key="active_workflow").select("Standard Display").run()
        app.selectbox(key="active_workflow").select("Templates").run()
        self.clean(app)
        self.assertEqual(app.session_state[DRAFT_KEY], draft)
        self.click(app, "Reset draft")
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings, ())
        self.assertEqual(app.session_state[DRAFT_KEY].template.name, draft.template.name)
        self.click(app, "Clear draft")
        self.assertNotIn(DRAFT_KEY, app.session_state)
        self.assertTrue(any(button.label == "Start blank draft" for button in app.button))
        self.assertEqual(get_templates(), before)

    def test_remove_region_and_facing_are_draft_only(self):
        app = self.layout()
        self.click(app, "Create grid")
        self.click(app, "Remove region")
        facing = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual(facing.regions, ())
        self.assertEqual(facing.slots, ())
        self.click(app, "Remove facing")
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings, ())
