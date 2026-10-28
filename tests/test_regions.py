"""Region membership is explicit; grouping hints never transform fixture slots."""
import unittest
from dataclasses import FrozenInstanceError, replace
from xml.etree import ElementTree as ET

from app.templates.models import Bounds, PresentationHint, Region, SlotRole
from app.templates.preview import MAX_PREVIEW_HEIGHT, MAX_PREVIEW_WIDTH, geometry_svg
from app.templates.registry import get_template, get_templates
from app.templates.validation import validate_template
from tests.test_templates import fixture, product


def grouped(*regions, slots=None):
    template = fixture(*(slots if slots is not None else (product(),)))
    return replace(template, facings=(replace(template.facings[0], regions=regions),))


def region(region_id="area", **changes):
    return replace(Region(region_id, "Cards", SlotRole.PRODUCT, 1, ("position01",)), **changes)


class RegionContractTests(unittest.TestCase):
    def codes(self, template):
        return {issue.code for issue in validate_template(template).errors}

    def test_region_creation_and_read_only_membership(self):
        members, metadata = ["position01"], {"note": "fixture"}
        item = Region("cards", "Cards", SlotRole.PRODUCT, 1, members,
                      Bounds(0, 0, 1, 1), PresentationHint.GRID, metadata)
        members.append("other")
        metadata["note"] = "changed"
        self.assertEqual(item.slot_ids, ("position01",))
        self.assertEqual(item.metadata["note"], "fixture")
        with self.assertRaises(FrozenInstanceError):
            item.name = "changed"
        with self.assertRaises(TypeError):
            item.metadata["note"] = "changed"
        self.assertTrue(validate_template(grouped(item)).valid)

    def test_region_ids_names_roles_and_sharing_flag(self):
        self.assertIn("region_id", self.codes(grouped(region(""))))
        self.assertIn("region_name", self.codes(grouped(region(name=""))))
        self.assertIn("region_role", self.codes(grouped(region(role="CUSTOM_BRAND"))))
        self.assertIn("shared_membership_flag", self.codes(grouped(region(allow_shared_slots="yes"))))
        self.assertIn("duplicate_region_id", self.codes(grouped(region(), region(order=2))))

    def test_region_ids_unique_across_facings(self):
        first = grouped(region()).facings[0]
        second = replace(first, id="back", order=2, slots=(product("back.slot"),),
                         regions=(region(slot_ids=("back.slot",)),))
        self.assertIn("duplicate_region_id", self.codes(fixture(facings=(first, second))))

    def test_region_order_is_positive_unique_and_explicit(self):
        for value in (0, -1, 1.5, True, "1"):
            self.assertIn("region_order", self.codes(grouped(region(order=value))))
        self.assertIn("duplicate_region_order", self.codes(grouped(region(), region("other"))))

    def test_membership_and_slot_order_do_not_depend_on_storage_order(self):
        first, second = product(), product("second", x=2, order=9)
        template = grouped(region(slot_ids=("second", "position01")), slots=(second, first))
        facing = template.facings[0]
        self.assertEqual([slot.id for slot in facing.slots_for_region("area")], ["position01", "second"])
        self.assertIs(facing.slots_for_region("area")[0], first)
        self.assertEqual(facing.regions_for_slot("second")[0].id, "area")
        self.assertEqual([slot.order for slot in facing.slots_for_region("area")], [1, 9])
        with self.assertRaises(KeyError):
            facing.slots_for_region("unknown")
        with self.assertRaises(KeyError):
            facing.regions_for_slot("unknown")

    def test_invalid_slot_reference_is_structured_and_cannot_cross_facings(self):
        template = grouped(region(slot_ids=("missing",)))
        issues = validate_template(template).errors
        issue = next(issue for issue in issues if issue.code == "region_slot_reference")
        self.assertEqual(issue.region_id, "area")
        self.assertEqual(issue.facing_id, "front")
        self.assertIn("unassigned_region_slot", {issue.code for issue in issues})
        other = replace(fixture(product("missing")).facings[0], id="back", order=2)
        self.assertIn("region_slot_reference", self.codes(replace(template, facings=(*template.facings, other))))

    def test_duplicate_membership_requires_all_regions_to_explicitly_allow_it(self):
        first, second = region(), region("other", order=2)
        self.assertIn("multiple_region_membership", self.codes(grouped(first, second)))
        self.assertIn("multiple_region_membership", self.codes(grouped(replace(first, allow_shared_slots=True), second)))
        allowed = grouped(replace(first, allow_shared_slots=True), replace(second, allow_shared_slots=True))
        self.assertTrue(validate_template(allowed).valid)
        self.assertEqual(len(allowed.facings[0].regions_for_slot("position01")), 2)
        self.assertIn("duplicate_region_slot", self.codes(grouped(region(
            slot_ids=("position01", "position01"), allow_shared_slots=True))))

    def test_presentation_vocabulary_is_separate_from_geometry(self):
        original = grouped(region(presentation_hint=PresentationHint.PRESERVE_RELATIVE))
        for hint in (None, *PresentationHint):
            changed = replace(original.facings[0], regions=(region(presentation_hint=hint),))
            template = replace(original, facings=(changed,))
            self.assertTrue(validate_template(template).valid)
            self.assertEqual(changed.slots, original.facings[0].slots)
            self.assertEqual(geometry_svg(template, "front"), geometry_svg(original, "front"))
        self.assertIn("presentation_hint", self.codes(grouped(region(presentation_hint="render_pdf"))))

    def test_region_and_member_bounds(self):
        for bounds in (Bounds(0, 0, 0, 1), Bounds(0, 0, -1, 1), Bounds(float("nan"), 0, 1, 1)):
            self.assertIn("region_dimensions", self.codes(grouped(region(bounds=bounds))))
        for bounds in (Bounds(-1, 0, 2, 2), Bounds(0, 0, 5, 2), Bounds(0, 5, 1, 2)):
            self.assertIn("region_bounds", self.codes(grouped(region(bounds=bounds))))
        self.assertIn("slot_region_bounds", self.codes(grouped(region(bounds=Bounds(.1, .1, .5, .5)))))
        self.assertTrue(validate_template(grouped(region(bounds=Bounds(0, 0, 1, 1)))).valid)
        self.assertTrue(validate_template(grouped(region(bounds=None))).valid)

    def test_duplicate_slot_order_is_still_invalid_within_a_region(self):
        template = grouped(region(slot_ids=("position01", "second")),
                           slots=(product(), product("second", x=2)))
        self.assertIn("duplicate_slot_order", self.codes(template))

    def test_region_and_member_shuffle_preserve_preview_and_order(self):
        template = get_template("holiday_2026_side_a")
        facing = template.facings[0]
        shuffled = replace(facing, slots=tuple(reversed(facing.slots)), regions=tuple(
            replace(item, slot_ids=tuple(reversed(item.slot_ids))) for item in reversed(facing.regions)))
        changed = replace(template, facings=(shuffled,))
        self.assertTrue(validate_template(changed).valid)
        self.assertEqual(geometry_svg(template, "side_a"), geometry_svg(changed, "side_a"))


class RegionProofTests(unittest.TestCase):
    def test_holiday_side_a_has_four_independent_regions_and_assumed_counts(self):
        template = get_template("holiday_2026_side_a")
        self.assertEqual(template.unit, "normalized")
        facing = template.facings[0]
        self.assertEqual((facing.width, facing.height), (1, 1.5))
        self.assertEqual([item.name for item in facing.ordered_regions()],
                         ["Top Cards", "Gift Card Holders", "Main Cards", "Bonus"])
        self.assertEqual([item.order for item in facing.ordered_regions()], [1, 2, 3, 4])
        self.assertEqual([len(item.slot_ids) for item in facing.ordered_regions()], [14, 5, 24, 10])
        self.assertEqual([item.presentation_hint for item in facing.ordered_regions()],
                         [PresentationHint.CUSTOM_ROWS, PresentationHint.SINGLE_ROW,
                          PresentationHint.GRID, PresentationHint.SECTION])
        self.assertEqual(len(facing.slots), 53)
        holders = facing.slots_for_region("holiday_2026_side_a.gift_card_holders")
        self.assertEqual([slot.order for slot in holders], list(range(15, 20)))
        self.assertEqual(len({slot.y for slot in holders}), 1)
        for index, slot in enumerate(holders):
            self.assertAlmostEqual(slot.x, .10 + index * .17125)
        self.assertAlmostEqual(holders[-1].x + holders[-1].width, .90)
        self.assertTrue(all(len(facing.regions_for_slot(slot.id)) == 1 for slot in facing.slots))
        self.assertEqual(validate_template(template).issues, ())
        self.assertIn("not approved", template.metadata["count_assumptions"])

    def test_bonus_wrap_remains_physical_despite_grouped_section_hint(self):
        facing = get_template("holiday_2026_side_a").facings[0]
        slots = facing.slots_for_region("holiday_2026_side_a.bonus")
        self.assertEqual([slot.order for slot in slots], list(range(45, 55)))
        self.assertEqual([sum(f".{part}." in slot.id for slot in slots) for part in ("left", "bottom", "right")], [3, 4, 3])
        self.assertEqual(len({slot.width for slot in slots}), 2)
        main = facing.slots_for_region("holiday_2026_side_a.main_cards")
        self.assertLess(min(slot.x for slot in slots), min(slot.x for slot in main))
        self.assertGreater(max(slot.x for slot in slots), max(slot.x for slot in main))

    def test_compact_svg_shows_region_envelopes_names_colors_and_numbered_slots(self):
        namespace = {"s": "http://www.w3.org/2000/svg"}
        for template in get_templates():
            for facing in template.facings:
                root = ET.fromstring(geometry_svg(template, facing.id))
                self.assertLessEqual(float(root.get("width")), MAX_PREVIEW_WIDTH)
                self.assertLessEqual(float(root.get("height")), MAX_PREVIEW_HEIGHT)
                self.assertIn("max-width:100%", root.get("style"))
                slots = [rect for rect in root.findall("s:rect", namespace) if rect.get("data-slot-id")]
                self.assertEqual(len(slots), len(facing.slots))
                regions = [rect for rect in root.findall("s:rect", namespace) if rect.get("data-region-id")]
                self.assertEqual(len(regions), len(facing.regions))
                labels = [label.text for label in root.findall("s:text", namespace) if label.get("data-region-name")]
                self.assertEqual(labels, [item.name for item in facing.ordered_regions()])
                if facing.regions:
                    self.assertEqual(len({rect.get("fill") for rect in regions}), 4)
                    self.assertTrue(all(rect.get("data-region-ids") for rect in slots))

    def test_existing_defaults_do_not_require_region_migration(self):
        for template_id, total in (("holiday_qp_2026", 60), ("holiday_nre", 40)):
            template = get_template(template_id)
            self.assertEqual(sum(len(facing.slots) for facing in template.facings), total)
            self.assertTrue(all(not facing.regions for facing in template.facings))
            self.assertEqual(validate_template(template).issues, ())
