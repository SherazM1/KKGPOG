# Side workspaces and Duplicate Side (Piece 4E)

The builder shows one large editable Side at a time. Small clickable previews
for every Side appear at the top of the main canvas; the active preview is
highlighted. Click a preview to make that Side large and show its Section and
Box controls on the left. Adding or duplicating a Side opens the new Side.
Under **Side options**, enter a **New side name** and click **Add Side** to
start a blank Side with no Sections or Boxes. The existing copy controls follow
it for reusing a completed layout. Measured drafts also need the new Side's
width and height in **Advanced side settings**.
Default names use the first available letter: **Side A**, **Side B**, through
**Side Z**, then **Side AA**. Copies keep their source name with a Copy suffix.
**Add Box**, **Create Grid**, **Duplicate Box**, **Delete Box**, fallback
movement buttons and technical Advanced geometry all target the active Side.
Side options contain rename,
dimensions, order and deletion. Role, hints, membership and raw geometry remain
under Advanced. IDs remain inspectable under Advanced draft details.

Each Side uses a separate selection key and a stable component key. Canvas edits
use the same Piece 4D event boundary and the same session TemplateDraft. The
active canvas and thumbnails are regenerated from the authoritative draft;
stale revision checks still apply. Switching previews preserves each Side's
Section/Box selection and never changes geometry. Late events from a minimized
Side are rejected. Invalid dimensions do not block switching to other Sides.
Apply form changes before switching; unsubmitted form inputs are not saved.

The active Side summarizes overlaps and bounds errors, with technical details collapsed.
The old separate Review pane is removed. Optional **Review All Sides** below the
editor shows static valid layouts, followed by the overall Done action. Validation
still checks the complete draft before Done is enabled.

## Duplication

Under **Duplicate Side**, accept or change `<Side name> Copy`, then submit.
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
