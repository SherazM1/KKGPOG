# Side workspaces and Duplicate Side (Piece 4E)

Every Side stays visible as a bordered workspace with a heading, its own compact
interactive canvas, local Section buttons and Box controls. **Add a section**
opens a small form inside that Side's context. The selected Section is highlighted
and named. **Add Box**, **Create Grid**, **Duplicate Box**, **Delete Box**, fallback
movement buttons and technical Advanced geometry all target that Side's selection.
There are no global Side or Section selectors. Side options contain rename,
dimensions, order and deletion. Role, hints, membership and raw geometry remain
under Advanced. IDs remain inspectable under Advanced draft details.

Each Side uses a separate selection key and a stable component key. Canvas edits
use the same Piece 4D event boundary and the same session TemplateDraft. All
canvases re-render from the authoritative draft, including after changes to
another Side; stale revision checks still apply. A structural error on one Side
does not hide the other Side's canvas. Side form keys depend on that Side's
contents, so editing another card does not discard its pending form values.

Side cards summarize overlaps and bounds errors, with technical details collapsed.
The old separate Review pane is removed. Optional **Review All Sides** below the
cards shows static valid layouts, followed by the overall Done action. Validation
still checks the complete draft before Done is enabled.

## Duplication

Expand **Duplicate Side**, accept or change `<Side name> Copy`, then submit.
`draft.duplicate_facing` returns a new draft and appends a complete independent
hierarchy. It preserves dimensions, Slot geometry/order/role/grid indexes,
Region role/order/hints/bounds/shared membership and metadata. The new Side
receives the next display order to avoid colliding with the source's order.

The existing monotonic draft allocator issues new Side, Region and Slot IDs.
Region member references are remapped; legacy Slot section/group IDs are also
remapped consistently. Metadata is recursively copied, including nested mappings
and lists. Metadata text remains opaque: arbitrary text is not rewritten as an
ID reference. Explicit hierarchy references are remapped. Invalid missing-box
references reject duplication without changing the original draft. Positive
geometry and layout validation retain their existing behavior.

The original immutable hierarchy remains intact when the copy is moved,
renamed, or deleted. There is no persistence, production rendering integration or
catalog-template cloning. Built-in templates remain read-only.

## Verification

Run the full unit/AppTest suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q
```

For real pointer interaction, start the app in one terminal:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run home.py --server.headless true --server.address 127.0.0.1 --server.port 8514
```

Then use the existing Playwright/Chromium installation:

```powershell
.\.venv\Scripts\python.exe -B tests/browser_side_workspaces.py
```

This creates a fresh browser session, builds a 3 × 4 grid, duplicates the Side,
checks fixed-size geometry, rectangle selection, group dragging and bulk deletion
on a 50-box irregular Side. It also verifies Section copying/confirmed deletion,
Side-copy independence and contained sizing. See [direct_canvas_builder.md](direct_canvas_builder.md). It never invokes production output or saves a draft.

After operator testing, the next smallest UI pass is undo/redo for manual builder
actions. It is not part of Piece 4E.
