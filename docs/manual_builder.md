# Manual template builder — Piece 4

## Operator flow

Open **Templates → Create New Template**. Enter a name, then **Start blank draft**.
Family, version and units are optional under **Advanced template settings**.
The default is an unmeasured layout sketch. Built-in definitions remain in the
read-only catalog. Starting a new draft replaces only the current session draft;
returning to the catalog without starting it leaves the existing draft intact.

- **Sides:** add by name. Click a small Side preview above the main canvas to
  select it. One Side is shown large for editing, with its controls on the left;
  each Side remembers its own Section/Box selection. New and duplicated Sides
  become active immediately. Sketch sides default to 1 × 1 normalized units internally.
  Rename/delete are ordinary Side options; dimensions/order live under Advanced
  side settings. Inches still require positive measured dimensions there.
- **Sections:** use **Add Section** while the intended Side is active. Select a
  Section using its visible button; the selected button is highlighted. Each Side
  remembers its own Section/Box selection. Defaults are Product role and no presentation
  hint. Role, hint and manual order remain under Advanced section settings.
- **Boxes:** Add Box creates one fixed-size box automatically. Create Grid asks
  only for Rows and Columns and uses the same fixed size and spacing. Drag boxes
  directly; use Select boxes or drag empty space for rectangle selection. Drag a
  selected box to move the group, or use Delete Selected to remove the group.
  There are no normal resize handles or resize buttons.
- **Section workspaces:** new Sections appear below earlier ones, with their names
  and boundaries on the canvas. Workspace growth shifts later Sections downward
  without resizing boxes. Duplicate Section copies a whole arrangement below
  its source; Delete Section requires confirmation.
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
sizing. Advanced box controls contain technical compatibility geometry, order and manual
section assignment. These fields are for legacy repair, not normal box sizing. IDs and reset/clear actions are under Advanced draft details.

New boxes always join the selected section. Creation order is preserved through
moves to avoid silently renumbering boxes. Manual ordering stays Advanced.

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
The active Side uses an interactive SVG canvas that grows with its layout;
click a minimized Side preview to switch. Scroll the website to see the full
height. Click a box to select,
drag to move, or drag a selection rectangle around many boxes. Selection synchronizes the Section
and Box controls; Duplicate/Delete act on the selected box. Spatial validation
errors remain visible and editable. Structural errors require fallback controls.
Sections show named workspace bands even when empty. Advanced details expose stable
IDs and coordinates, and a collapsed static preview remains available for valid
drafts under Review All Sides. See [interactive_canvas.md](interactive_canvas.md) for event handling,
coordinate conversion and limitations. No production output is generated.

Deletion actions use callbacks so the draft changes before dependent widgets
render. Per-selection form keys ensure one slot or region cannot inherit another
selection's geometry or hint values.

## Printing the template

Use **Create Template PDF** at the bottom of the builder, then **Download Template
PDF**. After **Done**, the same actions are available in the **Your template**
card on the Templates screen. The export uses the current session draft, not
the separately selected built-in catalog template.

Sides use separate portrait pages at the configured size (currently **14 x 22 inches**)
in display order. Printed cards fill the available width. **Keep vertical spacing
in PDF**, beneath the section title/color fields, is checked by default: row and
section gaps remain proportional to the arranged layout. Save the setting with
**Add Section** or **Apply section changes**; copies retain it.
Uncheck it to use that section's spare vertical space for taller boxes. Regular
grids share the recovered space evenly, with gaps up to 6 points; irregular layouts
grow only where no row or header would be crossed. Header positions and the editable
draft remain unchanged. Horizontal gaps
are tightened for wider cards. Their outer proportions can change for printing; image placeholders
stay square. The editable draft and its geometry stay unchanged.
For the larger printed boxes, select paper matching the PDF dimensions and print at Actual Size.
The drawing uses 6-point side margins with no extra frame padding. The header
and footer sit outside the drawing area, which extends to 30 points from the
page bottom. Only sections with spacing preservation unchecked use their blank
gaps as additional height inside PDF boxes.
Each page has the template name at top left and its Side number/letter at top right.
The footer shows the PDF creation date at bottom left and **Generated By Kendal
King** in the center. Dates use America/Chicago time. Explicit names such as
**Side B** or **Side #12** retain their letter/number; other names use the Side's
display order. Copy suffixes stay visible, for example **SIDE A COPY** or
**SIDE 1 COPY 2**. Cards have plain, light-gray outlines, box numbers,
Description/CPP/UPC writing lines by default, and square image placeholders;
outer Side borders, colored Section backgrounds, setup notes, counts, and legends
are omitted. Each separate section grid stretches across the printable width,
so sections with fewer columns get wider boxes instead of empty side margins.
Use **Section header color** directly below **New section name** or **Section name**
to choose its PDF band color. Save it with **Add Section** or **Apply section changes**;
copies retain the color. Each title is
centered in a full-width band, with black or white lettering selected for contrast.
Sections sharing vertical space scale together to preserve their side-by-side
arrangement. Titles have reserved space so they
do not overlap the writing lines or image placeholders. New Sides default to
**Side A**, **Side B**, and onward; existing names are preserved.
This is a blank layout printout, not a product-populated or actual-size fixture
specification. Edits invalidate the previous PDF so it must be regenerated.
Export is enabled only when the template passes layout validation.
Beside the box controls, below **Add Box**, **Third PDF text line** chooses between
**UPC** (the default), **ID#**, and **Item #**. Description and CPP are always
printed. Preferences last for the draft's session, and changing the third line
clears the previous PDF download so it must be regenerated.
Generated downloads include a timestamp and show their preparation time, making
fresh exports easy to distinguish from older files. The Templates workflow
refreshes changed UI/renderer source on rerun without resetting the session draft.

## Intentional boundaries and next step

No snapping, alignment tools, undo/redo, reference upload,
PDF overlay, detection, cloning, persistence, product assignment, assortment
population, image matching, price-strip generation or production PDF generation.
The existing production renderers and built-in geometry definitions are isolated.

The next small interaction pass can add undo/redo for manual geometry edits.
Production integration and durable persistence remain separate work.

See [direct_canvas_builder.md](direct_canvas_builder.md) for fixed sizing, automatic
placement, bulk actions, measured-side limits and the complex-side browser check.
