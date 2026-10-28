"""Illustrative region proof, not an approved Side A fixture specification."""
from app.templates.models import (
    Bounds, DisplayTemplate, Facing, GeometryUnit, PresentationHint, Region, Slot, SlotRole,
)


def holiday_2026_side_a() -> DisplayTemplate:
    template_id = "holiday_2026_side_a"
    slots, regions = [], []

    def add(region_key, position_id, x, y, width, height, order, row, column):
        slot = Slot(f"{template_id}.side_a.{region_key}.{position_id}", SlotRole.PRODUCT,
                    x, y, width, height, order, row=row, column=column)
        slots.append(slot)
        return slot.id

    members = []
    for row, columns, y, left, offset in ((1, 8, .08, .08, 0), (2, 6, .18, .186, 8)):
        for column in range(1, columns + 1):
            members.append(add("top_cards", f"r{row:02d}.c{column:02d}",
                               left + (column - 1) * .106, y, .09, .075,
                               offset + column, row, column))
    regions.append(Region(
        f"{template_id}.top_cards", "Top Cards", SlotRole.PRODUCT, 1, tuple(members),
        Bounds(.04, .02, .92, .30), PresentationHint.CUSTOM_ROWS,
        {"arrangement": "Two rows: 8 and 6 positions; illustrative."},
    ))

    members = tuple(add("gift_card_holders", f"p{position:02d}",
                        .10 + (position - 1) * .17125, .40, .115, .09,
                        14 + position, 1, position) for position in range(1, 6))
    regions.append(Region(
        f"{template_id}.gift_card_holders", "Gift Card Holders", SlotRole.PRODUCT, 2,
        members, Bounds(.04, .34, .92, .18), PresentationHint.SINGLE_ROW,
        {"arrangement": "Five positions in one row; count confirmed by the user from the Side A reference. Geometry remains illustrative."},
    ))

    members = tuple(
        add("main_cards", f"r{row:02d}.c{column:02d}",
            .13 + (column - 1) * .095, .62 + (row - 1) * .15, .075, .12,
            20 + (row - 1) * 8 + column, row, column)
        for row in range(1, 4) for column in range(1, 9)
    )
    regions.append(Region(
        f"{template_id}.main_cards", "Main Cards", SlotRole.PRODUCT, 3, members,
        Bounds(.10, .55, .80, .52), PresentationHint.GRID,
        {"arrangement": "Three rows of eight positions; illustrative main-card region."},
    ))

    # Physical wrap: descend left, cross the base, then ascend right. The hint
    # suggests grouping in a future planogram, without changing any rectangle.
    members = []
    for position in range(1, 4):
        members.append(add("bonus", f"left.p{position:02d}", .025,
                           .96 + (position - 1) * .16, .075, .12,
                           44 + position, position, 1))
    for position in range(1, 5):
        members.append(add("bonus", f"bottom.p{position:02d}",
                           .20 + (position - 1) * .16, 1.30, .12, .12,
                           47 + position, 4, position + 1))
    for position in range(1, 4):
        members.append(add("bonus", f"right.p{position:02d}", .90,
                           1.28 - (position - 1) * .16, .075, .12,
                           51 + position, 4 - position, 6))
    regions.append(Region(
        f"{template_id}.bonus", "Bonus", SlotRole.PRODUCT, 4, tuple(members),
        Bounds(.02, .94, .96, .51), PresentationHint.SECTION,
        {"arrangement": "Ten illustrative positions wrapping left/base/right, not a filled rectangular grid.",
         "presentation_intent": "A future planogram may group these as one clean section; physical coordinates stay unchanged."},
    ))

    return DisplayTemplate(
        id=template_id, name="Holiday 2026 Side A — Region Proof", version="1.0",
        template_family="holiday_region_proof", unit=GeometryUnit.NORMALIZED, locked=True,
        facings=(Facing("side_a", "Side A", 1.0, 1.5, 1, tuple(slots),
                        {"structure": "4 independent regions · 53 illustrative positions"}, tuple(regions)),),
        metadata={
            "calibration": "Unmeasured: normalized geometry and proportions are illustrative, not fixture truth.",
            "scope": "Read-only structure/presentation-hint proof; no product assignment or PDF mapping is performed.",
            "count_assumptions": "14 top cards, 5 holders, 24 main cards, 10 bonus. Holder count confirmed by the user from the Side A reference; this structural proof is not approved as a production specification. Bonus count remains provisional.",
            "source": "Piece 3 region brief; top 8+6 and main 24 are corroborated by existing model concepts, not a verified fixture drawing.",
        },
    )
