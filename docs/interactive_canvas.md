# Manual builder interactive canvas (Piece 4F)

Each Side has a contained live canvas with named Section boundaries. Click boxes
or drag a selection rectangle. Selected boxes highlight together; drag a selected
box to move the group. Normal box size is fixed. No handles or resize buttons are
rendered. Delete Selected acts on the current Side's selection.

The local Streamlit v1 component uses plain JavaScript and SVG, with no framework,
build step, CDN or added package. The editing viewport scrolls internally at a
maximum height of 640 pixels, keeping large Sections readable. The separate
catalog/static SVG preview retains its original size limits and remains read-only.

## State and conversion

The component temporarily adjusts SVG attributes during one pointer gesture.
On release it emits an event ID, action, template/Side/box IDs, draft revision,
and selected box IDs. Group moves emit only a movement delta in SVG coordinates.
It never sends a replacement
template, grid, Section assignment or pixel-based geometry model.

`_template_canvas.py` registers a callback before the next controls render.
`_canvas_events.py` checks event identity and revision, translates coordinates,
and calls the existing immutable draft update API. It writes the result to the
same `template_builder:draft` session key. Selection is namespaced per Side at
`template_builder:selection:<facing_id>`; a widget epoch lets a canvas click
override an older local Box selectbox value. No global Side selector is required.
Events are consumed once and acknowledged; stale/foreign/malformed events retain
the current draft and show a warning. The server's regenerated SVG is authoritative.

Pointer CSS coordinates use the inverse SVG screen matrix, so resizing the browser
does not change units. The server reads the rendered Side rectangle's origin and
uniform scale: `template_x = (svg_x - origin_x) / scale`, `template_width =
svg_width / scale` (likewise for Y/height). Inches and normalized units remain as
defined by the draft. The inverse conversion is isolated and tested too.

Only selected Slot positions change in the normal canvas; width and height stay
fixed. Stable IDs, order, role, row/column,
legacy section ID, metadata and all Region memberships (including shared ones)
are retained. Dropping over another Section never reassigns membership.

## Validation and limits

Positive finite dimensions are required. Dragging is constrained to Side bounds;
group movement preserves relative arrangement and fixed dimensions. Existing
validation reports bounds and overlaps after each edit, and the draft is retained.
The editing projection permits spatial errors to remain visually editable;
the catalog preview's default validation behavior stays strict. Structural
errors or nonpositive dimensions require the retained fallback controls.

Pointer events support mouse, touch and pen in modern browsers. Keyboard users
can select a focused box with Enter/Space and use the existing buttons and
Advanced fields for movement. Technical geometry remains available for legacy
repair. Pointer feedback commits on release; there is no undo history or autosave.
Normal resize events are rejected at the server boundary. A gesture waits for server acknowledgment
before the next gesture; connection latency is visible as “Updating…”.

AppTest cannot drive JavaScript pointer events. Deterministic tests cover the
translation/state boundary and AppTest covers synchronized controls and the
read-only viewer. A live Chromium check covers actual component selection,
rectangle selection, group dragging, bulk deletion and contained sizing.

This remains an uncalibrated structural builder, not a production specification.
No production rendering, persistence, product mapping, PDF output or catalog
editing is connected. The next small pass is undo/redo for manual geometry edits.

Piece 4F details and current browser stress coverage: [direct_canvas_builder.md](direct_canvas_builder.md).
