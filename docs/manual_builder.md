# Manual template builder — Piece 4

## Operator flow

Open **Templates → Create New Template**. Set a name, family/display type label,
version and unit, then **Start blank draft**. Built-in definitions remain in the
read-only catalog. Starting a new draft replaces only the current session draft;
returning to the catalog without starting it leaves the existing draft intact.

- **Facings:** add one or many, rename, resize, reorder or remove them. Inches
  start with zero dimensions and require positive measured values before adding
  the facing. Normalized facings start at 1 × 1 and are explicitly uncalibrated.
- **Regions:** add, rename, choose a role and the existing presentation hint,
  reorder or remove. Region bounds are derived from member slots for the preview;
  no new local coordinate transform or presentation layout is introduced.
- **Positions:** add a small suggested rectangle, create a grid, or select an
  existing position to edit its width, height, placement, order and region.
  Distances are measured from the facing's top-left in the chosen unit.
  Placement coordinates are under expanders; IDs appear only in Advanced details.
  Each form has an explicit Apply/Add/Create action that updates the preview.

The unit is fixed at creation. Choosing another unit requires a new draft; there
is no relabeling or silent conversion of existing measurements. A physical draft
contains operator-entered measurements, not automatically verified fixture truth.

## Session isolation

`TemplateDraft` wraps a new, unlocked `DisplayTemplate`; the UI never wraps a
built-in template or offers cloning. The draft, current selection and builder
mode are ordinary namespaced Streamlit session state. The immutable core models
are updated by pure operations in `app/templates/draft.py`, which has no file,
database, registry or production-rendering dependency.

IDs use a draft UUID plus a monotonic counter. Names, geometry, order and region
assignments can change without changing identity. IDs cannot be edited in this
pass, and removed IDs are not reused, including after Reset draft.

**Back to templates / Resume draft** preserves the draft. Navigation to another
workflow and normal reruns also preserve it within the session.
**Reset draft** removes geometry while retaining name/family/version/unit and
template identity. **Clear draft** discards the session draft and returns to
template basics. Removing a facing removes its regions and slots. Removing a
region removes its exclusive slots while preserving any explicitly shared slot
still referenced by another region. Removing a position removes its references.

Nothing is written to disk or a database. Browser/session loss can lose the draft.

## Grid creation

Rows, columns, start distances, overall width/height and horizontal/vertical gaps
define the initial cell rectangles:

`cell_width = (overall_width - (columns - 1) × horizontal_gap) / columns`

`cell_height = (overall_height - (rows - 1) × vertical_gap) / rows`

Creation produces independent `Slot` objects, with distinct immutable IDs,
row/column metadata, explicit facing-level order and region membership. There is
no stored grid definition controlling them afterward. Moving or removing one
does not move the others. The UI supports 1–50 rows and columns per grid action.
Invalid grid dimensions, excessive gaps and grids outside the facing are rejected
atomically with a plain-language message, leaving the draft unchanged.

## Validation and preview

The existing geometry validator checks every draft after each applied operation.
It reports identity/order/membership/hint issues, invalid sizes/bounds and
product overlaps. Geometry edits are not silently clamped into validity. A draft
may temporarily be invalid, including overlapping positions or a zero-width
facing, and can be corrected through the same forms. Required text/unsupported
enum values and malformed grid inputs are rejected before an operation commits.

Messages use facing/region names and position numbers instead of long IDs.
The selected facing uses the existing compact, region-aware SVG preview when the
whole draft passes validation; otherwise issues and a recovery message replace
the preview. Empty regions gain envelopes as positions are added. Advanced
details expose stable IDs and coordinates. No production output is generated.

Deletion actions use callbacks so the draft changes before dependent widgets
render. Per-selection form keys ensure one slot or region cannot inherit another
selection's geometry or hint values.

## Intentional boundaries and next step

No drag/drop, mouse resize, snapping, alignment tools, undo/redo, reference upload,
PDF overlay, detection, cloning, persistence, product assignment, assortment
population, image matching, price-strip generation or production PDF generation.
The existing production renderers and built-in geometry definitions are isolated.

The next small pass should add explicit draft serialization plus versioned
save/load with a round-trip validation check. Preserve units, IDs, membership and
order, keep built-ins protected, and leave renderer integration for a later pass.
