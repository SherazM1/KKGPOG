"""Prove the coordinator flow never requires an Advanced control."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from app.pages.template_builder import DRAFT_KEY
from app.templates.models import SlotRole, GeometryUnit
from app.templates.registry import get_templates


ROOT = Path(__file__).resolve().parents[1]


def widgets_with_ancestors(node, ancestors=()):
    if isinstance(node, Block):
        if node.type == "expander":
            ancestors = (*ancestors, node.label)
        for child in node.children.values():
            yield from widgets_with_ancestors(child, ancestors)
    else:
        yield node, ancestors


class OperatorUXTests(unittest.TestCase):
    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertEqual([error.message for error in app.exception], [])

    def start(self):
        app = AppTest.from_file(str(ROOT / "home.py"), default_timeout=15).run()
        app.selectbox(key="active_workflow").select("Templates").run()
        self.click(app, "Create New Template")
        next(item for item in app.text_input if item.label == "Template name").set_value("Coordinator layout")
        self.click(app, "Start blank draft")
        self.click(app, "Add Side")
        next(item for item in app.text_input if item.label == "New section name").set_value("Main Cards")
        self.click(app, "Add Section")
        return app

    def test_name_side_section_three_by_four_grid_without_advanced(self):
        before = get_templates()
        app = self.start()
        for item in app.expander:
            if item.label.startswith("Advanced"):
                self.assertFalse(item.proto.expanded)
        next(item for item in app.number_input if item.label == "Rows").set_value(3)
        next(item for item in app.number_input if item.label == "Columns").set_value(4)
        self.click(app, "Create Grid")
        draft = app.session_state[DRAFT_KEY]
        face = draft.template.facings[0]
        self.assertEqual(draft.template.unit, GeometryUnit.NORMALIZED)
        self.assertEqual(face.name, "Side 1")
        self.assertEqual(face.regions[0].name, "Main Cards")
        self.assertEqual(face.regions[0].role, SlotRole.PRODUCT)
        self.assertIsNone(face.regions[0].presentation_hint)
        self.assertEqual(len(face.slots), 12)
        self.assertEqual(face.regions[0].slot_ids, tuple(box.id for box in face.slots))
        self.assertEqual([box.order for box in face.slots], list(range(1, 13)))
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.get("imgs")), 1)
        self.assertTrue(any("Sections: 1 | Boxes: 12" in item.value for item in app.caption))
        self.click(app, "Done")
        self.assertEqual(app.session_state[DRAFT_KEY], draft)
        self.assertTrue(any(button.label == "Resume draft" for button in app.button))
        self.assertEqual(get_templates(), before)

    def test_technical_controls_are_advanced_only(self):
        app = self.start()
        self.click(app, "Add Box")
        required = {"Section type", "Planogram grouping preference", "Side width", "Side height", "Side order",
                    "Box width", "Box height", "X (from left)", "Y (from top)", "Box order", "Box section",
                    "Starting X", "Starting Y", "Total grid width", "Total grid height", "Horizontal gap", "Vertical gap"}
        found = set()
        for widget, ancestors in widgets_with_ancestors(app._tree):
            label = getattr(widget, "label", None)
            if label in required:
                found.add(label)
                self.assertTrue(any(name.startswith("Advanced") for name in ancestors), (label, ancestors))
            if getattr(widget, "type", None) in ("button", "selectbox", "text_input", "number_input") and not any(
                    name.startswith("Advanced") for name in ancestors):
                self.assertFalse(any(term in (label or "").lower() for term in ("facing", "region", "position", "geometry", "order", "hint")), label)
        self.assertEqual(found, required)

    def test_add_duplicate_and_movement_need_no_geometry_entry(self):
        app = self.start()
        self.click(app, "Add Box")
        first = app.session_state[DRAFT_KEY].template.facings[0].slots[0]
        self.click(app, "Duplicate Box")
        face = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual(len(face.slots), 2)
        selected = face.slots[-1]
        self.click(app, "Move down")
        self.click(app, "Wider")
        changed = app.session_state[DRAFT_KEY].template.facings[0].slots[-1]
        self.assertEqual((changed.id, changed.order), (selected.id, selected.order))
        self.assertGreater(changed.y, selected.y)
        self.assertGreater(changed.width, selected.width)
        self.assertEqual(app.session_state[DRAFT_KEY].template.facings[0].slots[0], first)
        self.click(app, "Delete Box")
        self.assertEqual(len(app.session_state[DRAFT_KEY].template.facings[0].slots), 1)

    def test_new_boxes_belong_to_the_selected_section_and_editing_is_scoped(self):
        app = self.start()
        self.click(app, "Add Box")
        first_section = app.session_state[DRAFT_KEY].template.facings[0].regions[0]
        next(item for item in app.text_input if item.label == "New section name").set_value("Bonus")
        self.click(app, "Add Section")
        self.assertFalse(any(item.label == "Box" for item in app.selectbox))
        self.click(app, "Add Box")
        face = app.session_state[DRAFT_KEY].template.facings[0]
        self.assertEqual([len(region.slot_ids) for region in face.regions], [1, 1])
        box_options = next(item for item in app.selectbox if item.label == "Box").options
        self.assertEqual(box_options, ["Box 2"])
        self.click(app, first_section.name)
        self.assertEqual(next(item for item in app.selectbox if item.label == "Box").options, ["Box 1"])
