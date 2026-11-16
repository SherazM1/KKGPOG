# Direct canvas builder for complex Sides (Piece 4F)

Normal boxes have one fixed size across the draft. Add Box uses automatic
placement; Create Grid asks only for Rows and Columns and materializes independent
fixed-size boxes with consistent gaps. Normal resizing is removed from both the
JavaScript interface and the server callback. Technical geometry and custom grids
remain collapsed under Advanced for legacy repair and internal compatibility.
Existing legacy/custom geometry is retained; it is not silently resized.

The standard is initialized from the first Side's width: one tenth for box width,
0.12 times for height. The default unmeasured Side therefore uses 0.10 × 0.12
template units. This is an illustrative layout default, not an approved physical
product measurement or a production specification. The values are stored in the
existing template metadata and do not change as the workspace grows or another
Side is added. Units never change. No new backend model is introduced.

## Section workspace

New Sections appear below the existing Sections, including empty named bands.
Adding rows expands that band's workspace and shifts subsequent Sections and
their boxes together. Unmeasured Sides grow to contain the layout; existing box
sizes and relative arrangements remain unchanged. Measured inch Sides keep their
entered dimensions: a layout that cannot fit is rejected atomically with a
readable request to check Advanced side dimensions. Shared legacy memberships
that cannot be shifted together safely require technical geometry repair.

Duplicate Section copies the complete arrangement below its source and makes
space by shifting later bands. It generates a new Region ID and new Slot IDs,
deep-copies metadata, remaps membership and legacy grouping IDs, and preserves
roles/hints, sizes and relative arrangement. New Box orders append in source order
to keep orders unique; later Section orders shift to insert the copy below its
source. Delete Section requires explicit confirmation. For legacy shared boxes,
the existing removal contract retains boxes still owned by another Section.
Duplicate Side retains its existing deep-copy/remap behavior.

## Selection and movement

Drag empty canvas space to draw a selection rectangle, or enable **Select boxes**
to start a rectangle anywhere (including on a box). Fully enclosed boxes become
selected. **Move boxes** exits selection mode. Shift-drag adds to a selection;
Ctrl/Command-click toggles an individual box. Click a selected box and drag to
move the group. Click an unselected box to select/move it alone. Escape or pointer
cancellation cancels an in-progress gesture. Keyboard Enter/Space selects a
focused box; movement buttons remain under Advanced box controls.

Delete Selected removes the selected Slots and all Region references in one draft
operation. Selection may span Sections on the same Side; it never crosses Sides.
No membership is inferred from drop position. Group movement preserves IDs,
order, dimensions, metadata, membership and pairwise offsets. Bounds/overlap
validation continues to report temporary invalid layouts without discarding them.

The component emits IDs and a pixel delta, never a replacement geometry model.
The isolated event adapter converts the delta through the current SVG scale and
updates the same session TemplateDraft atomically. Stale, foreign, malformed and
normal resize events are rejected without overwriting the draft. Each Side has
its own selection and component acknowledgment. A gesture commits on release;
the next gesture waits for the server acknowledgment.

## Page-height canvas and verification

The editing SVG draws Section names, backgrounds and active boundaries. It fits
the available column width and shows its full height; users scroll the website,
not an inner canvas. Unmeasured Sides start with a taller editing workspace and
expand downward during a drag and after adding rows. A committed downward move
grows the sketch Side and affected Section bounds without resizing boxes or
shifting other boxes. Measured Side dimensions remain fixed. The component frame
tracks its content height, including browser-width changes. Catalog previews
and production output retain their existing sizing.

Run unit/AppTest checks:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q
```

For the live browser check, start Streamlit on port 8514, then run:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run home.py --server.headless true --server.address 127.0.0.1 --server.port 8514
# In another terminal:
.\.venv\Scripts\python.exe -B tests/browser_side_workspaces.py
```

The browser creates 50 boxes in four uneven Sections, manually offsets a Bonus
box, duplicates/confirms deletion of a Section, selects and moves 32 boxes,
bulk-deletes a smaller selection, and verifies an independent Side copy. It checks
no resize UI, fixed dimensions and page-scrolling responsive sizing. Unit tests also
exercise a 53-box/five-Section layout and an 80-box Section-copy AppTest case.

The builder and completed-draft screen offer a blank template PDF with one Side
per portrait page, template-name/Side-identifier headers, and a creation-date/Kendal King
footer. Product mapping, persistence, reference
overlays and production renderer integration remain separate. After operator testing, the next small pass is session
undo/redo, particularly for bulk deletion and Section actions. Not implemented.
