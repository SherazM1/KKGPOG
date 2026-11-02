# Manual builder interactive canvas (Piece 4D)

The builder canvas selects any box on the active Side. Selection highlights the
box, shows its number and Section, and synchronizes the existing Section/Box
controls. Duplicate Box and Delete Box act on that selection. Drag a box to move
it inside the Side; drag any of eight handles to resize it. Grid cells remain
independent. Move/resize buttons and Advanced geometry remain available.

`app/components/template_canvas/` is a local Streamlit v1 component using plain
JavaScript and SVG, with no frontend framework, build step, CDN or added package.
The shared SVG keeps its 560 × 520 maximum size. The static preview is collapsed
under Advanced. The catalog viewer remains read-only.

## State and conversion

The component temporarily adjusts SVG attributes during one pointer gesture.
On release it emits an event ID, action, template/Side/box IDs, draft revision,
and SVG rectangle (selection omits the rectangle). It never sends a replacement
template, grid, Section assignment or pixel-based geometry model.

`_template_canvas.py` registers a callback before the next controls render.
`_canvas_events.py` checks event identity and revision, translates coordinates,
and calls the existing immutable draft update API. It writes the result to the
same `template_builder:draft` session key. Selection uses the existing selection
key; a widget epoch lets a canvas click override an older selectbox value.
Events are consumed once and acknowledged; stale/foreign/malformed events retain
the current draft and show a warning. The server's regenerated SVG is authoritative.

Pointer CSS coordinates use the inverse SVG screen matrix, so resizing the browser
does not change units. The server reads the rendered Side rectangle's origin and
uniform scale: `template_x = (svg_x - origin_x) / scale`, `template_width =
svg_width / scale` (likewise for Y/height). Inches and normalized units remain as
defined by the draft. The inverse conversion is isolated and tested too.

Only the selected Slot geometry changes. Stable IDs, order, role, row/column,
legacy section ID, metadata and all Region memberships (including shared ones)
are retained. Dropping over another Section never reassigns membership.

## Validation and limits

Positive finite dimensions are required. Dragging is constrained to Side bounds;
resizing may extend outside the Side. Existing validation reports bounds and
overlaps after each edit, and the draft is retained. An explicit builder-only
`allow_layout_errors` SVG option permits spatial errors to remain visually
editable; the renderer's default validation behavior stays strict. Structural
errors or nonpositive dimensions require the retained fallback controls.

Pointer events support mouse, touch and pen in modern browsers. Keyboard users
can select a focused box with Enter/Space and use the existing buttons and
Advanced fields for movement/resizing. Pointer feedback commits on release;
there is no undo history or autosave. A gesture waits for server acknowledgment
before the next gesture; connection latency is visible as “Updating…”.

AppTest cannot drive JavaScript pointer events. Deterministic tests cover the
translation/state boundary and AppTest covers synchronized controls and the
read-only viewer. A live Chromium check covers actual component selection,
dragging, resizing and compact sizing.

This remains an uncalibrated structural builder, not a production specification.
No production rendering, persistence, product mapping, PDF output or catalog
editing is connected. The next small pass is undo/redo for manual geometry edits.
