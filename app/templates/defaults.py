"""Uncalibrated structural baselines, never inferred physical measurements.

The QP facing sequence mirrors existing reference detection. NRE retains only
the five-by-two section structure, not product labels or PDF layout constants.
"""
from app.templates.models import DisplayTemplate, Facing, GeometryUnit, Slot, SlotRole


def holiday_qp_2026() -> DisplayTemplate:
    template_id = "holiday_qp_2026"
    facings = []
    for facing_order, (facing_id, name) in enumerate(
        (("left", "Left Side"), ("front", "Front"), ("right", "Right Side"), ("back", "Back")), 1
    ):
        slots = tuple(
            Slot(
                id=f"{template_id}.{facing_id}.r{row:02d}.c{column:02d}",
                role=SlotRole.PRODUCT,
                x=(column - 1) / 3, y=(row - 1) / 5,
                width=1 / 3, height=1 / 5,
                order=(row - 1) * 3 + column, row=row, column=column,
            )
            for row in range(1, 6) for column in range(1, 4)
        )
        facings.append(Facing(facing_id, name, 1.0, 1.0, facing_order, slots,
                              {"structure": "5 rows × 3 columns"}))
    return DisplayTemplate(
        id=template_id, name="Holiday Quarter Pallet 2026", version="1.0",
        template_family="holiday_qp", unit=GeometryUnit.NORMALIZED, locked=False,
        facings=tuple(facings), metadata={
            "calibration": "Unmeasured: structure only, not a physical specification.",
            "layout_basis": "Four facings, each 5 × 3, with explicit facing and row-major slot order.",
            "spacing": "Equal cells are schematic. Physical slot sizes, gaps and aspect ratios are unknown.",
            "source": "Piece 2 baseline; facing/order convention corroborated by Holiday reference detection.",
        },
    )


def holiday_nre() -> DisplayTemplate:
    slots = []
    # Generic section identity survives changes to job/product-group labels.
    for section_row in range(5):
        for section_column in range(2):
            section_number = section_row * 2 + section_column + 1
            section_id = f"section{section_number:02d}"
            for position in range(1, 5):
                slots.append(Slot(
                    id=f"holiday_nre.main.{section_id}.p{position:02d}",
                    role=SlotRole.PRODUCT,
                    x=section_column * 0.52 + (position - 1) * 0.12,
                    y=section_row * 0.21, width=0.12, height=0.16,
                    order=(section_number - 1) * 4 + position,
                    row=section_row + 1, column=section_column * 4 + position,
                    section_id=section_id,
                ))
    return DisplayTemplate(
        id="holiday_nre", name="Holiday NRE", version="1.0",
        template_family="holiday_nre", unit=GeometryUnit.NORMALIZED, locked=True,
        facings=(Facing("main", "Main Display", 1.0, 1.0, 1, tuple(slots),
                        {"structure": "10 sections · 4 product positions per section"}),),
        metadata={
            "calibration": "Unmeasured: structure only, not a physical specification.",
            "layout_basis": "Five rows of two logical sections, four positions each; no product-group names.",
            "spacing": "Section gaps and slot proportions are illustrative, not measured fixture geometry.",
            "scope": "Product region only. Configuration-specific fillers and fixture footprint are not calibrated.",
            "source": "Piece 2 baseline; section structure corroborated by NRE presentation, not its PDF dimensions.",
        },
    )
