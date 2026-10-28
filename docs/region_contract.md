# Regions and presentation intent — Piece 3

## Ownership and compatibility

The logical hierarchy is `DisplayTemplate → Facing → Region → Slot`.
The existing `Facing.slots` tuple remains the canonical store of physical
rectangles. A region contains explicit `slot_ids`, not copies of slot geometry.
This avoids two competing coordinate stores and keeps existing QP/NRE definitions
unchanged. `Facing.slots_for_region(id)` resolves these references in slot order;
`regions_for_slot(id)` resolves the reverse membership in region order.
Unknown lookup IDs raise `KeyError`; validation reports invalid stored references
as structured issues before a viewer renders them.

Region fields: stable ID, display name, role (`SlotRole` vocabulary), positive
order, member slot IDs, optional `Bounds`, optional `PresentationHint`, read-only
metadata, and an explicit `allow_shared_slots` flag. No product/category naming
convention is required. Slots keep their original IDs, roles, order, coordinates,
row/column values and optional section IDs.

Region IDs are unique across a template; region orders are unique within a facing.
Slot order remains unique within the facing and therefore within any region.
Region/member tuple order is irrelevant. Grid row/column metadata may describe
a region-local arrangement; it does not impose one grid on the whole facing.

## Physical coordinates and bounds

All region bounds and slot coordinates are **facing-local**, in the template's
declared unit, with the existing top-left / X-right / Y-down convention.
Adding a region does not add an origin transform or change a slot rectangle.
Bounds are optional rectangular envelopes. When supplied, they must be positive,
finite, inside the facing, and contain their member slots. When omitted, the
preview derives an envelope from member rectangles without changing the model.

Envelopes may overlap: a wraparound region's bounding rectangle can contain empty
space occupied by another region. An envelope is not a filled rectangular grid
or occupied fixture surface. Product-slot overlap rules still apply across all
slots, including slots in different regions.

## Membership validation

- Every member ID must reference a slot on the same facing.
- References cannot repeat inside one region.
- On a facing with regions, every slot must have explicit membership.
- Sharing one slot between regions is rejected unless **every** participating
  region explicitly sets `allow_shared_slots=True`. It remains one physical slot,
  not several overlapping copies. Repeating it inside a region is still invalid.
- Facings with no regions retain the previous behavior; QP and NRE need no migration.
- Region IDs/names/roles/orders, sharing flags, optional bounds and hints are
  validated. Issues include `region_id` as well as facing/slot context.
- Deterministic diagnostic ordering and explicit slot ordering remain in force.

## Presentation hints are not presentation geometry

The small optional vocabulary is `grid`, `single_row`, `custom_rows`, `section`,
and `preserve_relative`. A hint states a possible future planogram grouping.
It contains no PDF points, fonts, page sizes, margins or target coordinates.
Changing a hint does not move slots, reorder their identities, or change the
physical preview. No production renderer reads these hints.

In particular, the Bonus region can occupy left/base/right physical positions
while suggesting a clean `section` in a future planogram. This pass shows that
intent in a read-only summary; it does not implement a mapping engine or populate
a PDF.

## Holiday 2026 Side A proof

ID: `holiday_2026_side_a`, version `1.0`, locked, one facing `side_a`.
Facing size is **1 × 1.5 normalized units**, not inches. All positions, gaps and
aspect ratios are illustrative and uncalibrated.

| Region | Order | Positions | Physical example | Suggested presentation |
|---|---:|---:|---|---|
| Top Cards | 1 | 14 | Rows of 8 and 6 | Custom rows |
| Gift Card Holders | 2 | 5 | One row of larger positions | Single row |
| Main Cards | 3 | 24 | Three rows of eight | Grid |
| Bonus | 4 | 10 | Three left, four along base, three right | Grouped section |

Total: 53 positions. Slot order is explicit: top 1–14, holders 15–19, main
21–44, bonus 45–54. Bonus traversal descends left, crosses the base, then ascends
right. Order 20 is unused following the holder correction, preserving other slot
IDs and order values. IDs include the region and logical position, independent of tuple order.

**This is a structural proof, not an approved production specification.** The user
confirmed five holders from the actual Side A reference. The 8+6 top-card and
24-main concepts are corroborated by existing source models; the ten bonus
positions remain provisional because the current Side A reference was not
available for independent inspection. All normalized dimensions, spacing and
proportions remain illustrative, not approved physical measurements.

## Read-only preview

SVG intrinsic dimensions are capped at **560 × 520 pixels**, with a single scale
for both axes and responsive shrinking in narrow containers. Streamlit uses
content width instead of stretching the image to page width. Color-coded region
envelopes and a matching name legend distinguish regions; slot numbers retain
their explicit order. The small summary table shows counts and suggested grouping.
Raw coordinates, region IDs and bounds remain under Advanced / Geometry details.

There are no edit, create, drag, resize, delete, save, clone, reference upload,
overlay, detection or population actions. Existing production outputs are isolated.

## Next step

A first editor should select a region, place/move/resize rectangles in physical
facing coordinates, assign explicit region membership, and validate before
accepting a draft. Keep presentation hints as a separate region setting. Begin
with this small rectangle/region interaction; persistence, reference overlays,
production mapping and renderer migration should remain separate passes.
