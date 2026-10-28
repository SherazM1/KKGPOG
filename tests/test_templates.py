"""Portable geometry-contract tests; no production inputs or personal files."""
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from xml.etree import ElementTree as ET

from app.templates.models import DisplayTemplate, Facing, GeometryUnit, Slot, SlotRole
from app.templates.preview import geometry_svg
from app.templates.registry import get_facings, get_template, get_templates
from app.templates.validation import Severity, validate_template


def product(slot_id="position01", **changes):
    return replace(Slot(slot_id, SlotRole.PRODUCT, 0, 0, 1, 1, 1), **changes)


def fixture(*slots, **changes):
    template = DisplayTemplate("test", "Fixture", "1.0", "test", GeometryUnit.INCHES,
                               False, (Facing("front", "Front", 4, 6, 1, slots),))
    return replace(template, **changes)


class TemplateContractTests(unittest.TestCase):
    def codes(self, template):
        return {issue.code for issue in validate_template(template).errors}

    def test_physical_model_creation_requires_explicit_unit(self):
        template = fixture(product())
        self.assertEqual(template.unit, GeometryUnit.INCHES)
        self.assertTrue(validate_template(template).valid)
        with self.assertRaises(TypeError):
            DisplayTemplate(id="missing", name="Missing unit", version="1", template_family="test",
                            locked=False, facings=())

    def test_definitions_and_metadata_are_read_only(self):
        notes = {"source": "measured"}
        template = fixture(product(metadata=notes))
        notes["source"] = "changed externally"
        self.assertEqual(template.facings[0].slots[0].metadata["source"], "measured")
        with self.assertRaises(FrozenInstanceError):
            template.name = "changed"
        with self.assertRaises(TypeError):
            template.facings[0].slots[0].metadata["source"] = "changed"
        self.assertIsInstance(template.facings, tuple)
        self.assertIsInstance(template.facings[0].slots, tuple)

    def test_pixel_pdf_and_implicit_units_are_rejected(self):
        for unit in ("px", "pt", "inch", "", None):
            with self.subTest(unit=unit):
                self.assertIn("unit", self.codes(fixture(product(), unit=unit)))
        normalized = fixture(product(), unit=GeometryUnit.NORMALIZED)
        self.assertEqual(normalized.facings[0].width, 4)
        self.assertTrue(validate_template(normalized).valid)

    def test_invalid_facing_sizes(self):
        for value in (0, -1, float("nan"), float("inf"), True, "10"):
            for dimension in ("width", "height"):
                with self.subTest(value=value, dimension=dimension):
                    facing = replace(fixture().facings[0], **{dimension: value})
                    self.assertIn("facing_dimensions", self.codes(fixture(facings=(facing,))))

    def test_invalid_slot_dimensions_and_coordinates(self):
        for field in ("width", "height"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    self.assertIn("slot_dimensions", self.codes(fixture(product(**{field: value}))))
        for field in ("x", "y", "width", "height"):
            for value in (float("nan"), float("inf"), True, "1"):
                with self.subTest(field=field, value=value):
                    self.assertIn("slot_coordinates", self.codes(fixture(product(**{field: value}))))

    def test_top_left_bounds_and_edge_contact(self):
        for values in ({"x": -0.1}, {"y": -0.1}, {"x": 3.1}, {"y": 5.1}):
            with self.subTest(values=values):
                self.assertIn("slot_bounds", self.codes(fixture(product(**values))))
        # Exact bottom-right contact is allowed; Y is measured down from the top.
        self.assertTrue(validate_template(fixture(product(x=3, y=5))).valid)

    def test_facing_and_slot_ids_unique_across_template(self):
        facing = fixture(product()).facings[0]
        repeated = replace(facing, order=2)
        self.assertIn("duplicate_facing_id", self.codes(fixture(facings=(facing, repeated))))
        other = replace(facing, id="back", order=2)
        self.assertIn("duplicate_slot_id", self.codes(fixture(facings=(facing, other))))
        unique = replace(other, slots=(product("back.position01"),))
        self.assertTrue(validate_template(fixture(facings=(facing, unique))).valid)

    def test_missing_identity_and_metadata_indices(self):
        for field in ("id", "name", "version", "template_family"):
            self.assertIn("template_field", self.codes(fixture(**{field: ""})))
        self.assertIn("facings", self.codes(fixture(facings=())))
        self.assertIn("locked", self.codes(fixture(locked="yes")))
        self.assertIn("slot_id", self.codes(fixture(product(slot_id=""))))
        for field in ("row", "column"):
            self.assertIn("grid_index", self.codes(fixture(product(**{field: 0}))))
        self.assertIn("section_id", self.codes(fixture(product(section_id=""))))

    def test_product_overlap_is_structured_error_and_edge_touch_is_not_overlap(self):
        first, second = product(), product("position02", x=0.5, order=2)
        result = validate_template(fixture(first, second))
        issue = next(issue for issue in result.issues if issue.code == "product_overlap")
        self.assertEqual(issue.severity, Severity.ERROR)
        self.assertEqual(issue.facing_id, "front")
        self.assertEqual(issue.slot_ids, ("position01", "position02"))
        self.assertFalse(result.valid)
        self.assertTrue(validate_template(fixture(first, replace(second, x=1))).valid)

    def test_cross_role_overlap_warns_and_other_roles_are_supported(self):
        for role in SlotRole:
            second = product("other", role=role, order=2)
            result = validate_template(fixture(product(), second))
            if role != SlotRole.PRODUCT:
                self.assertTrue(result.valid)
                self.assertEqual(result.warnings[0].code, "slot_overlap")
        self.assertIn("role", self.codes(fixture(product(role="BRAND"))))

    def test_orders_are_explicit_unique_and_scope_is_facing(self):
        first = product()
        self.assertIn("duplicate_slot_order", self.codes(fixture(first, product("second", x=2))))
        for order in (0, -1, 1.5, True, "1"):
            self.assertIn("slot_order", self.codes(fixture(product(order=order))))
        facing = fixture(first).facings[0]
        other = replace(facing, id="other", slots=(product("other.slot"),))
        self.assertIn("duplicate_facing_order", self.codes(fixture(facings=(facing, other))))
        self.assertTrue(validate_template(fixture(facings=(facing, replace(other, order=2)))).valid)

    def test_storage_order_and_names_do_not_define_slot_identity(self):
        first, second = product(), product("second", x=2, order=7)
        facing = fixture(first, second).facings[0]
        shuffled = replace(facing, name="Renamed", slots=(second, first))
        self.assertEqual(facing.ordered_slots(), shuffled.ordered_slots())
        self.assertTrue(validate_template(fixture(facings=(shuffled,))).valid)
        result = validate_template(fixture(first, replace(second, x=0.5)))
        reverse = validate_template(fixture(replace(second, x=0.5), first))
        self.assertEqual(result, reverse)


class DefaultTemplateTests(unittest.TestCase):
    def test_builtins_are_uncalibrated_valid_and_have_explicit_lock_policy(self):
        qp, nre = get_templates()
        for template in (qp, nre):
            with self.subTest(template=template.id):
                self.assertEqual(template.unit, GeometryUnit.NORMALIZED)
                self.assertEqual(validate_template(template).issues, ())
                self.assertIn("Unmeasured", template.metadata["calibration"])
                self.assertTrue(all(facing.width == facing.height == 1 for facing in template.facings))
        self.assertFalse(qp.locked)
        self.assertTrue(nre.locked)

    def test_qp_four_facings_sixty_positions_and_row_major_identity(self):
        template = get_template("holiday_qp_2026")
        facings = template.ordered_facings()
        self.assertEqual([facing.id for facing in facings], ["left", "front", "right", "back"])
        self.assertEqual([facing.order for facing in facings], [1, 2, 3, 4])
        self.assertEqual(sum(len(facing.slots) for facing in facings), 60)
        for facing in facings:
            self.assertEqual(len(facing.slots), 15)
            for index, slot in enumerate(facing.ordered_slots()):
                row, col = divmod(index, 3)
                self.assertEqual((slot.row, slot.column, slot.order), (row + 1, col + 1, index + 1))
                self.assertEqual(slot.id, f"holiday_qp_2026.{facing.id}.r{row+1:02d}.c{col+1:02d}")
                self.assertEqual(slot.role, SlotRole.PRODUCT)
                self.assertAlmostEqual(slot.x, col / 3)
                self.assertAlmostEqual(slot.y, row / 5)

    def test_nre_forty_positions_ten_generic_sections_four_each(self):
        template = get_template("holiday_nre")
        self.assertEqual(len(template.facings), 1)
        slots = template.facings[0].ordered_slots()
        self.assertEqual(len(slots), 40)
        self.assertEqual(Counter(slot.section_id for slot in slots),
                         {f"section{index:02d}": 4 for index in range(1, 11)})
        self.assertEqual([slot.order for slot in slots], list(range(1, 41)))
        self.assertTrue(all(slot.role == SlotRole.PRODUCT for slot in slots))
        for product_group in ("ZIFT", "NINTENDO", "PLAYSTATION", "XBOX", "ROBLOX", "FORTNITE",
                              "META", "RAZER GOLD", "POKEMON GO", "GOOGLE PLAY"):
            # Inspect values rather than the word 'metadata' in dataclass field names.
            values = " ".join([*template.metadata.values(), *[slot.id for slot in slots],
                               *[slot.section_id for slot in slots]]).upper()
            self.assertNotIn(product_group, values)

    def test_nre_structural_counts_agree_with_existing_presentation_without_using_its_dimensions(self):
        from app.holiday_planograms.render_nre import _template_for_configuration
        for config in ("4x78", "3x78", "3x60", "4x60"):
            presentation = _template_for_configuration(config)
            positions = [slot for slot in presentation.slots if slot.slot_type == "merchandise"]
            self.assertEqual(len(positions), 40)
            self.assertEqual(sorted(Counter(slot.group_name for slot in positions).values()), [4] * 10)
        self.assertEqual(get_template("holiday_nre").unit, GeometryUnit.NORMALIZED)

    def test_registry_lookup_is_explicit_and_cannot_mutate_catalog(self):
        self.assertEqual([template.id for template in get_templates()], ["holiday_qp_2026", "holiday_nre"])
        self.assertEqual(len(get_facings("holiday_qp_2026")), 4)
        first, second = get_template("holiday_nre"), get_template("holiday_nre")
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        with self.assertRaises(KeyError):
            get_template("unknown")
        with self.assertRaises(KeyError):
            get_facings("unknown")


class GeometryPreviewTests(unittest.TestCase):
    def test_svg_contains_blank_slots_and_boundaries_for_every_facing(self):
        namespace = {"s": "http://www.w3.org/2000/svg"}
        for template in get_templates():
            for facing in template.ordered_facings():
                root = ET.fromstring(geometry_svg(template, facing.id))
                rectangles = root.findall("s:rect", namespace)
                self.assertEqual(sum("data-slot-id" in item.attrib for item in rectangles), len(facing.slots))
                self.assertEqual(sum(item.get("data-kind") == "facing" for item in rectangles), 1)
                self.assertEqual([item.get("data-slot-id") for item in rectangles if "data-slot-id" in item.attrib],
                                 [slot.id for slot in facing.ordered_slots()])
                self.assertIn("Uncalibrated", "".join(root.itertext()))
                self.assertEqual(len(root.findall(".//s:image", namespace)), 0)
        root = ET.fromstring(geometry_svg(get_template("holiday_nre"), "main"))
        self.assertEqual(sum("data-section-id" in element.attrib for element in root), 10)

    def test_preview_escapes_labels_and_does_not_invent_physical_units(self):
        template = fixture(product(), name="<script>unsafe & title</script>")
        svg = geometry_svg(template, "front")
        self.assertIn("&lt;script&gt;", svg)
        self.assertNotIn("<script>", svg)
        self.assertIn("Inches", svg)
        normalized = geometry_svg(get_template("holiday_qp_2026"), "left")
        self.assertNotIn("Inches", normalized)

    def test_preview_rejects_invalid_template_and_unknown_facing(self):
        with self.assertRaises(ValueError):
            geometry_svg(fixture(product(width=-1)), "front")
        with self.assertRaises(KeyError):
            geometry_svg(fixture(product()), "unknown")
