# Manual template builder — Piece 4

## Operator flow

Open **Templates → Create New Template**. Enter a name, then **Start blank draft**.
Family, version and units are optional under **Advanced template settings**.
The default is an unmeasured layout sketch. Built-in definitions remain in the
read-only catalog. Starting a new draft replaces only the current session draft;
returning to the catalog without starting it leaves the existing draft intact.

- **Sides:** add by name. Each Side stays visible in its own bordered workspace,
  with its name, compact live canvas and local controls. There is no global Side
  selector. Sketch sides default to 1 × 1 normalized units internally.
  Rename/delete are ordinary Side options; dimensions/order live under Advanced
  side settings. Inches still require positive measured dimensions there.
- **Sections:** use **Add a section** inside the intended Side card. Select a
  Section using its visible button; the selected button is highlighted. Each Side
  remembers its own Section/Box selection. Defaults are Product role and no presentation
  hint. Role, hint and manual order remain under Advanced section settings.
- **Boxes:** Add Box places a box automatically in the selected section. Create
  Grid normally asks only for rows/columns. Boxes are selected within the current
  section. Duplicate/Delete and Move left/right/up/down or Wider/Narrower/Taller/
  Shorter buttons handle basic arrangement without coordinate entry.
- **Duplicate Side:** expand this action in the Side card, confirm or change the
  proposed name, then create an independent copy with fresh hierarchy IDs.
- **Review / Done:** section colors, box numbers and validation are visible inside
  each workspace. **Review All Sides** is a collapsed static review lower on the
  page. Done returns to the catalog and retains the session draft; it
  does not permanently save or publish anything.

The normal name → side → section → 3 × 4 grid flow needs no Advanced controls.
Operator labels are Side/Section/Box; backend Facing/Region/Slot classes are unchanged.

## Advanced controls

Advanced template settings contain family/version/units. Advanced side settings
contain physical dimensions and ordering. Advanced section settings contain roles,
presentation hints and ordering. Advanced grid settings contain X/Y, total size
and gaps; check **Use custom grid settings** to apply them instead of automatic
sizing. Advanced box geometry contains X/Y, width/height, order and manual section
assignment. IDs and reset/clear actions are under Advanced draft details.

New boxes always join the selected section. Creation order is preserved through
moves/resizes to avoid silently renumbering boxes. Manual ordering stays Advanced.

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

Rows and columns are the only normal inputs. UI adapters choose an open rectangle
inside the selected section's explicit bounds, or the side if it has no bounds.
Spacing scales with row/column count. Other boxes are avoided; lack of free room
produces a message instead of silently adding overlaps. Advanced overrides retain
the existing backend grid behavior and validation.

Start distances, overall width/height and horizontal/vertical gaps define cells:

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

Messages use side/section names and box numbers instead of long IDs.
Each Side uses a compact interactive SVG canvas. Click a box to select,
drag to move, or drag its handles to resize. Selection synchronizes the Section
and Box controls; Duplicate/Delete act on the selected box. Spatial validation
errors remain visible and editable. Structural errors require fallback controls.
Empty sections gain envelopes as boxes are added. Advanced details expose stable
IDs and coordinates, and a collapsed static preview remains available for valid
drafts under Review All Sides. See [interactive_canvas.md](interactive_canvas.md) for event handling,
coordinate conversion and limitations. No production output is generated.

Deletion actions use callbacks so the draft changes before dependent widgets
render. Per-selection form keys ensure one slot or region cannot inherit another
selection's geometry or hint values.

## Intentional boundaries and next step

No snapping, alignment tools, undo/redo, reference upload,
PDF overlay, detection, cloning, persistence, product assignment, assortment
population, image matching, price-strip generation or production PDF generation.
The existing production renderers and built-in geometry definitions are isolated.

The next small interaction pass can add undo/redo for manual geometry edits.
Production integration and durable persistence remain separate work.
