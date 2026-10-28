# Fixture geometry contract — Piece 2

Piece 3 adds optional ID-based regions without migrating the original defaults.
See [region_contract.md](region_contract.md) for membership, envelopes, presentation
hints, the Side A proof, and the compact read-only viewer.

## Three separate coordinate domains

`app.templates` describes where content may go on a fixture. It contains no
products, images, UPCs, placement decisions, fonts, PDF headers or footers.
Production workflows do not consume this model in Piece 2.

1. **Physical fixture:** `GeometryUnit.INCHES` (`in`) is the canonical physical
   unit. Every coordinate and size in every facing uses that template's unit.
2. **Uncalibrated template space:** `GeometryUnit.NORMALIZED` (`normalized`) is
   dimensionless. Both built-ins use a 1 × 1 facing. Equal cells and illustrative
   gaps encode structure only; they are not measured spacing or aspect ratios.
   Normalized templates are not physical fixture specifications. No implicit
   conversion to inches exists. Changing the unit label is not calibration.
3. **Reference and presentation:** existing Holiday `ReferenceImage`,
   `GeometryResult`, `LayoutPanel` and `LayoutSlot` retain their source/pixel
   coordinates. NRE `_NreTemplate`/`_TemplateSlot` and other renderer layouts
   retain their output/PDF coordinates. Neither is imported into the built-in
   definitions or automatically treated as physical truth.

A future import/overlay adapter must carry source units and an explicit,
verified source-to-fixture transformation. Production renderers will need a
separate fixture-to-presentation transformation. Neither adapter is implemented.
Sam's strip profiles demonstrate explicit physical dimensions, but their PDF
field positions and unrelated strip sizes cannot calibrate these fixtures.

## Coordinate convention

Origin is the **top-left of each facing**. Positive X goes right; positive Y
goes down. Slot `(x, y)` is its top-left corner. Width extends along X; height
extends along Y. Facing width/height define the available rectangular region.
All slot bounds are local to that facing; slots on different facings cannot
overlap each other. No rotation or fold transformation is assumed.

This convention is natural for a future rectangular editor and image overlay,
without inheriting a PDF renderer's bottom-left convention. Viewport SVG pixels
are a display transform only: a single scale preserves the declared aspect
ratio, and those pixels never feed back into template dimensions.

## Identity, order and roles

- Template `id` identifies a definition/lineage; `version` is explicit text.
  `locked` marks system versus future editable baselines. The viewer is read-only
  for both. These are data contracts, not persistence or editing permissions.
- Facing IDs are unique within the template. Facing `order` is a positive,
  unique integer. Facings have no implied identity from list position.
- Slot IDs are unique across the entire template. IDs are independent of
  presentation styling, product names and tuple positions. Renaming a template
  or changing geometry does not automatically regenerate IDs.
- Slot `order` is positive and unique **within its facing**. Gaps in order values
  are allowed. `ordered_facings()` and `ordered_slots()` explicitly sort by
  order, with ID as a deterministic tie-break; duplicate orders remain invalid.
- Optional row/column are one-based. `section_id` is an optional generic grouping,
  not a required product category or brand. No group-name vocabulary is enforced.
- Roles: `PRODUCT`, `FILLER`, `HEADER`, `SIGNAGE`, `PRICE_POCKET`, `RESERVED`.
  Only `PRODUCT` is used by the initial defaults. Strip/pocket rectangles can
  use the same facing-local coordinates later.
- Definitions are frozen dataclasses; nested slots/facings are tuples. Metadata
  is a read-only string mapping for provenance/notes, not an alternate geometry
  channel. Registry calls return fresh definitions.

## Initial parallel definitions

**Holiday Quarter Pallet 2026**, ID `holiday_qp_2026`, version `1.0`, unlocked:
four ordered facings: Left, Front, Right, Back. Each is 1 × 1 normalized space
with five rows and three columns; each cell is 1/3 × 1/5. There are 15 product
slots per facing, 60 total. Order is top-to-bottom, left-to-right, matching the
existing detector's logical facing/row/column ordering. Example identity:
`holiday_qp_2026.left.r01.c01`. Existing placement logic is not changed.

**Holiday NRE**, ID `holiday_nre`, version `1.0`, locked:
one main logical product region in 1 × 1 normalized space. Five section rows,
two sections per row, four positions per section: 40 product slots. Sections
`section01` through `section10` are geometric identifiers. No job/group names
are stored. Positions are 0.12 wide × 0.16 high; section starts are X=0 and
X=0.52, row pitch is 0.21. These proportions/gaps are **illustrative** and do not
come from physical measurements or PDF point-to-inch conversion. Order runs
through each section's four positions, then the next section in row-major order.
Example identity: `holiday_nre.main.section01.p01`.

NRE's physical facing count, surrounding fixture footprint and configuration-
specific filler regions are not established by this 40-position logical baseline.
The production NRE configurations and all filler behavior remain separate.

Still needed: authoritative facing/merchandise-region dimensions, card/slot
sizes, offsets, row/column pitch, margins, section gaps, fillers/reserved regions,
and any associated price-pocket/signage dimensions. Do not use output PDF
dimensions, NRE presentation card sizes, or reference-image resolution as substitutes.

## Validation

`validate_template()` returns `ValidationResult` containing structured issues:
severity, code, message, facing ID and related slot IDs. It performs no UI/I/O.

Checks include template identity/version/family, explicit supported unit,
boolean lock status, at least one facing, valid unique IDs, positive finite
dimensions, finite nonnegative origins, in-bounds rectangles, valid roles,
positive integer orders without duplicates in their scope, and valid optional
row/column/section values. Blank facings are allowed for future blank templates.

Product/product positive-area overlap is an **error**: two independently
populated product positions occupying the same space are unsafe to approve
silently. Other role combinations produce warnings because future signage,
header or reserved overlays may be intentional. Edge contact is not overlap.
A tolerance of 1e-9 declared units handles floating-point boundary noise.

Validation confirms structure, not real-world measurement accuracy or business
placement correctness. Both built-ins pass with no overlaps or duplicate IDs.

## Viewer and registry

`get_templates()`, `get_template(id)` and `get_facings(id)` expose the two built-ins;
unknown IDs raise `KeyError`. The UI uses this catalog, not renderer constants.
The Templates workflow shows template/version/status, facing selection, counts,
uncalibrated-status notice, and blank numbered SVG rectangles. Coordinates and
provenance are behind Advanced / Geometry details. No product content, editing,
save/clone/version actions, reference overlays or production PDF generation exist.

Next: obtain/verify measurements and add a read-only calibrated reference overlay
before allowing operators to edit or save a fixture as physically authoritative.
