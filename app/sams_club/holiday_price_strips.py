from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from reportlab.lib.units import inch

from app.sams_club.price_strip_models import SamsPriceStripRow, SamsPriceStripSegment

SAMS_HOLIDAY_TEMPLATE_NAME = "Sam's Holiday"
SAMS_HOLIDAY_2026_TEMPLATE_NAME = "Sam's Holiday 2026"


@dataclass(frozen=True)
class SamsHolidaySideGeometry:
    designation: str
    width_in: float
    height_in: float
    slot_count: int
    expected_priced_rows: int
    applicable_sides: tuple[int, ...]
    excluded_cr80_rows: int = 0

    @property
    def width_pt(self) -> float:
        return self.width_in * inch

    @property
    def height_pt(self) -> float:
        return self.height_in * inch


@dataclass(frozen=True)
class SamsHolidayTemplate:
    name: str
    thin_side: SamsHolidaySideGeometry
    wide_side: SamsHolidaySideGeometry
    brand_top_pt: float
    description_top_pt: float
    pack_range_top_pt: float
    description_inset_pt: float
    description_font_size_pt: float
    single_digit_description_shift_pt: float
    maximum_description_shift_pt: float
    minimum_description_to_price_gap_pt: float
    description_box_width_pt: float
    long_description_left_shift_pt: float
    long_description_up_shift_pt: float
    long_description_extra_width_pt: float
    description_min_font_size_pt: float
    description_max_lines: int
    price_dollar_sign_font_size_pt: float
    price_dollars_font_size_pt: float
    price_cents_font_size_pt: float
    price_cents_translate_y_pt: float
    footer_font_size_pt: float
    sku_font_size_pt: float
    footer_inset_pt: float
    sku_inset_pt: float
    center_price_amount: bool = True


SAMS_HOLIDAY_TEMPLATE = SamsHolidayTemplate(
    name=SAMS_HOLIDAY_TEMPLATE_NAME,
    thin_side=SamsHolidaySideGeometry(
        designation="thin",
        width_in=30.25,
        height_in=3.4375,
        slot_count=6,
        expected_priced_rows=5,
        applicable_sides=(1, 3),
    ),
    wide_side=SamsHolidaySideGeometry(
        designation="wide",
        width_in=38.4375,
        height_in=3.4375,
        slot_count=8,
        expected_priced_rows=4,
        applicable_sides=(2, 4),
        excluded_cr80_rows=1,
    ),
    brand_top_pt=45.5,
    description_top_pt=60.9,
    pack_range_top_pt=76.3,
    description_inset_pt=36.0,
    description_font_size_pt=18.0,
    single_digit_description_shift_pt=10.0,
    maximum_description_shift_pt=12.0,
    minimum_description_to_price_gap_pt=18.0,
    description_box_width_pt=170.0,
    long_description_left_shift_pt=12.0,
    long_description_up_shift_pt=6.0,
    long_description_extra_width_pt=24.0,
    description_min_font_size_pt=13.0,
    description_max_lines=3,
    price_dollar_sign_font_size_pt=38.0,
    price_dollars_font_size_pt=110.0,
    price_cents_font_size_pt=44.0,
    price_cents_translate_y_pt=-8.0,
    footer_font_size_pt=8.0,
    sku_font_size_pt=9.0,
    footer_inset_pt=36.0,
    sku_inset_pt=24.0,
)


def is_sams_holiday_template(template_name: str | None) -> bool:
    normalized = str(template_name or "").strip().lower()
    return normalized in {
        SAMS_HOLIDAY_TEMPLATE_NAME.lower(),
        SAMS_HOLIDAY_2026_TEMPLATE_NAME.lower(),
    }


def holiday_geometry_for_side(side: int) -> SamsHolidaySideGeometry:
    if side in SAMS_HOLIDAY_TEMPLATE.thin_side.applicable_sides:
        return SAMS_HOLIDAY_TEMPLATE.thin_side
    if side in SAMS_HOLIDAY_TEMPLATE.wide_side.applicable_sides:
        return SAMS_HOLIDAY_TEMPLATE.wide_side
    raise ValueError(
        f"Sam's Holiday side must be Side 1, 2, 3, or 4; received Side {side}."
    )


def holiday_slot_width_pt(geometry: SamsHolidaySideGeometry) -> float:
    return geometry.width_pt / geometry.slot_count


def holiday_slot_center_pt(
    geometry: SamsHolidaySideGeometry,
    slot_index: int,
    strip_left_x: float = 0.0,
) -> float:
    return strip_left_x + ((slot_index + 0.5) * geometry.width_pt / geometry.slot_count)


def holiday_slot_centers_pt(geometry: SamsHolidaySideGeometry) -> list[float]:
    return [
        holiday_slot_center_pt(geometry, slot_index)
        for slot_index in range(geometry.slot_count)
    ]


def holiday_placeholder_segment(
    row_data: SamsPriceStripRow,
    column: int,
) -> SamsPriceStripSegment:
    return SamsPriceStripSegment(
        pog=row_data.pog,
        side=row_data.side,
        row=row_data.row,
        column=column,
        length=f"{holiday_geometry_for_side(row_data.side).width_in}x{holiday_geometry_for_side(row_data.side).height_in}",
        data_on_bottom_left="",
        is_empty=True,
    )


def expand_holiday_row_to_slots(row_data: SamsPriceStripRow) -> SamsPriceStripRow:
    geometry = holiday_geometry_for_side(row_data.side)
    by_column = {
        segment.column: segment
        for segment in row_data.segments
        if 1 <= segment.column <= geometry.slot_count
    }
    expanded_segments = [
        by_column.get(column) or holiday_placeholder_segment(row_data, column)
        for column in range(1, geometry.slot_count + 1)
    ]
    return SamsPriceStripRow(
        pog=row_data.pog,
        side=row_data.side,
        row=row_data.row,
        segments=expanded_segments,
        footer_text=row_data.footer_text,
        warnings=row_data.warnings.copy(),
    )


def map_holiday_rows_to_strips(
    rows: Iterable[SamsPriceStripRow],
) -> tuple[list[SamsPriceStripRow], list[str]]:
    warnings: list[str] = []
    mapped_rows: list[SamsPriceStripRow] = []

    rows_by_side: dict[int, list[SamsPriceStripRow]] = {}
    for row in rows:
        rows_by_side.setdefault(row.side, []).append(row)

    for side in sorted(rows_by_side):
        geometry = holiday_geometry_for_side(side)
        side_rows = sorted(rows_by_side[side], key=lambda row: row.row)

        priced_rows = side_rows
        if geometry.excluded_cr80_rows:
            excluded_rows = side_rows[: geometry.excluded_cr80_rows]
            priced_rows = side_rows[geometry.excluded_cr80_rows :]
            if not excluded_rows:
                warnings.append(
                    f"Sam's Holiday Side {side} expected one CR80 row to exclude before priced rows, but no rows were available."
                )
            elif len(excluded_rows[0].segments) != 9:
                warnings.append(
                    f"Sam's Holiday Side {side} expected CR80 row with 9 cards before generated strips, "
                    f"but row {excluded_rows[0].row} has {len(excluded_rows[0].segments)} populated segment(s)."
                )

        if len(priced_rows) != geometry.expected_priced_rows:
            warnings.append(
                f"Sam's Holiday Side {side} expected {geometry.expected_priced_rows} priced rows of "
                f"{geometry.slot_count} positions after excluding the CR80 row, but received {len(priced_rows)} priced rows."
            )

        for row in priced_rows:
            max_column = max((segment.column for segment in row.segments), default=0)
            if max_column > geometry.slot_count:
                warnings.append(
                    f"Sam's Holiday Side {side} Row {row.row} supports {geometry.slot_count} positions, "
                    f"but received column {max_column}."
                )
            mapped_rows.append(expand_holiday_row_to_slots(row))

    return mapped_rows, warnings


def validate_holiday_rows(rows: Iterable[SamsPriceStripRow]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        geometry = holiday_geometry_for_side(row.side)
        if len(row.segments) != geometry.slot_count:
            warnings.append(
                f"Sam's Holiday Side {row.side} Row {row.row} expected {geometry.slot_count} slots, "
                f"but received {len(row.segments)}."
            )
    return warnings
