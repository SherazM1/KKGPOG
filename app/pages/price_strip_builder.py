"""Create price-strip templates using the Holiday ones layout"""
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import streamlit as st

from app.sams_club.holiday_price_strips import holiday_geometry_for_side
from app.sams_club.generic_price_strips import geometry_for_size as holiday_geometry_for_size
from app.sams_club.price_strip_models import (
    SamsPriceStripRow,
    SamsPriceStripSegment,
)


PDF_KEY = "blank_price_strips:holiday_pdf"
ROWS_KEY = "blank_price_strips:rows"
EPS_KEY = "blank_price_strips:eps"
EPS_UPLOAD_KEY = "holiday_builder:eps_upload"
CARD_SPACING_KEY = "blank_price_strips:card_spacing"
CARD_SPACING_UPLOAD_KEY = "holiday_builder:card_spacing_upload"
CARD_SPACING_ERROR_KEY = "blank_price_strips:card_spacing_error"
BASELINE_EPS_PATH = (Path(__file__).resolve().parents[2] / 'assets' / 'templates' /
                     'source' / 'sams_holiday_price_strips_baseline.eps')
SHEET_KEY = "blank_price_strips:sheet"
SHEET_UPLOAD_KEY = "holiday_builder:sheet_upload"
SHEET_POG_KEY = "holiday_builder:sheet_pog"
SHEET_ERROR_KEY = "blank_price_strips:sheet_error"
FORMAT_KEY = 'blank_price_strips:format'
FORMAT_WIDGET_KEY = 'price_strips:format'
FORMAT_DRAFTS_KEY = 'blank_price_strips:format_drafts'
GENERIC_EPS_PATH = BASELINE_EPS_PATH.with_name('generic_price_strips_baseline.eps')
KK250133_EPS_PATH = BASELINE_EPS_PATH.with_name(
    'sams_KK250133-11_1.9375_Tall_Sams_Price_Strip.eps'
)
TWO_CARD_EPS_PATH = BASELINE_EPS_PATH.with_name('sams_two_card_baseline.eps')
STRIP_SIZES = {
    "thin": "30 1/4 × 3 7/16 inches — 6 positions",
    "wide": "38 7/16 × 3 7/16 inches — 8 positions",
}
GENERIC_STRIP_SIZES = {
    'generic_11': '11 × 1 15/16 inches — 3 positions',
    'generic_15_25': '15 1/4 × 1 15/16 inches — 4 positions',
    'generic_19_25': '19 1/4 × 1 15/16 inches — 5 positions',
    'generic_23_5': '23 1/2 × 1 15/16 inches — 6 positions',
}


def _switch_format(target=None):
    target = target or st.session_state[FORMAT_WIDGET_KEY]
    previous = st.session_state.get(FORMAT_KEY, 'holiday')
    if previous != target:
        keys = (ROWS_KEY, SHEET_KEY, EPS_KEY, CARD_SPACING_KEY, SHEET_POG_KEY, 'holiday_builder:name',
                'holiday_builder:side', 'holiday_builder:all_sides')
        saved = st.session_state.setdefault(FORMAT_DRAFTS_KEY, {})
        saved[previous] = {key:deepcopy(st.session_state[key]) for key in keys if key in st.session_state}
        for key in keys:
            st.session_state.pop(key, None)
        st.session_state.update(deepcopy(saved.get(target, {})))
        for key in list(st.session_state):
            if key.startswith(('holiday_builder:row_tickets:', 'holiday_builder:start:',
                               'holiday_builder:strip_size:', 'holiday_builder:preview_row:',
                               'holiday_builder:footer:')) or key in (EPS_UPLOAD_KEY, SHEET_UPLOAD_KEY, CARD_SPACING_UPLOAD_KEY):
                st.session_state.pop(key, None)
        st.session_state.pop(PDF_KEY, None)
        st.session_state.pop(SHEET_ERROR_KEY, None)
        st.session_state.pop(CARD_SPACING_ERROR_KEY, None)
    st.session_state[FORMAT_KEY] = target
    st.session_state[FORMAT_WIDGET_KEY] = target


def _default_tickets(side, strip_size=None):
    geometry = (holiday_geometry_for_side(side) if strip_size is None
                else holiday_geometry_for_size(strip_size))
    return [
        {"Brand": "PRODUCT NAME", "Pack / value": "PACK / VALUE",
         "Type": "GIFT CARD", "Price": "0.00", "Item #": ""}
        for _ in range(geometry.slot_count)
    ]


def _side_rows(side):
    drafts = st.session_state.setdefault(ROWS_KEY, {})
    default_size = ('generic_11' if st.session_state.get(FORMAT_KEY) == 'generic'
                    else holiday_geometry_for_side(side).designation)
    if side not in drafts:
        tickets = _default_tickets(side, default_size)
        # Keep entries from the former single-row editor in open sessions.
        previous = st.session_state.get(f"holiday_builder:tickets:{side}", {})
        for index, changes in previous.get("edited_rows", {}).items():
            if 0 <= int(index) < len(tickets):
                tickets[int(index)].update(changes)
        drafts[side] = {
            "start": st.session_state.get("holiday_builder:row", 1),
            "rows": [{"tickets": tickets}],
        }
    drafts[side].setdefault("strip_size", default_size)
    first = drafts[side]['rows'][0].get('number', drafts[side]['start'])
    for index, row in enumerate(drafts[side]['rows']):
        row.setdefault('number', first + index)
    return drafts[side]


def _add_row(side):
    draft = _side_rows(side)
    number = draft['rows'][0]['number'] + _row_numbers(draft)[-1] - draft['start'] + 1
    draft["rows"].append({"tickets": _default_tickets(side, draft["strip_size"]), "number":number})
    st.session_state.pop(PDF_KEY, None)


def _row_numbers(draft):
    first = draft['rows'][0].get('number', draft['start'])
    return [draft['start'] + row.get('number', first + index) - first
            for index, row in enumerate(draft['rows'])]


def _build_strip(name, side, row_number, tickets, strip_size=None, footer_text=None):
    geometry = (holiday_geometry_for_side(side) if strip_size is None
                else holiday_geometry_for_size(strip_size))
    if len(tickets) != geometry.slot_count:
        raise ValueError(f"Row {row_number}: this size requires {geometry.slot_count} positions.")
    segments = []
    
    for column, ticket in enumerate(tickets, start=1):
        if not any(value is not None and str(value).strip() for value in ticket.values()):
            segments.append(SamsPriceStripSegment(pog=name, side=side, row=row_number,
                                                  column=column, is_empty=True))
            continue
        raw_price = str(ticket.get("Price") if ticket.get("Price") is not None else "").strip()
        raw_price = raw_price.replace("$", "").replace(",", "")
        
        try:
            amount = Decimal(raw_price)
        except InvalidOperation:
            raise ValueError(
                f"Row {row_number}, position {column}: enter a price such as 22.50."
            ) from None
            
        if not amount.is_finite() or amount < 0:
            raise ValueError(
                f"Row {row_number}, position {column}: enter a valid, nonnegative price."
            )
        
        segments.append(
            SamsPriceStripSegment(
                pog=name,
                side=side,
                row=row_number,
                column=column,
                brand=str(ticket.get("Brand") or "").strip(),
                desc_1=str(ticket.get("Pack / value") or "").strip(),
                desc_2=str(ticket.get("Type") or "").strip(),
                retail=f"{amount:.2f}",
                item_number=str(ticket.get("Item #") or "").strip(),
                length=f"{geometry.width_in}x{geometry.height_in}",
                is_empty=False,
            )
        )
    return SamsPriceStripRow(
        pog=name,
        side=side,
        row=row_number,
        segments=segments,
        strip_size=strip_size,
        footer_text=footer_text if footer_text is not None else (
            f"Side: {side}, Row: {row_number} - POG: {name}"
        ),
    )
    

def _create_pdf(name, side, row_number, ticket_rows, *, strip_size=None, calibration=False,
                row_numbers=None, footers=None, eps_data=None):
    numbers = row_numbers if row_numbers is not None else range(row_number, row_number + len(ticket_rows))
    strips = [_build_strip(name, side, numbers[index], tickets, strip_size,
                           footers[index] if footers is not None else None)
              for index, tickets in enumerate(ticket_rows)]
    return _render_strips(strips, calibration, eps_data=eps_data)


def _render_strips(strips, calibration=False, *, eps_data=None):
    if eps_data is not None:
        from app.templates.eps_preview import parse_card_spacing, strip_crosshair_layout, widest_eps_card_width
        try:
            drawing = parse_card_spacing(eps_data)
            strips = [replace(row, eps_card_centers=strip_crosshair_layout(drawing,
                              row.strip_size or holiday_geometry_for_side(row.side).designation, row.row))
                      for row in strips]
            width,_ = widest_eps_card_width(drawing)
        except ValueError as exc:
            raise ValueError(f'Card Spacing: {exc}') from exc
        strips = [replace(row,eps_card_width_pt=width) for row in strips]
    if strips and all(str(row.strip_size).startswith('generic_') for row in strips):
        from app.sams_club.render_generic_price_strips import render_generic_price_strips_pdf
        result = render_generic_price_strips_pdf(strips, calibration=calibration)
    else:
        from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf
        result = render_sams_price_strips_pdf(
            strips,
            template_name="Sam's Holiday",
            calibration=calibration,
        )
    
    if not result.pdf_bytes:
        message = (
            result.warnings[-1]
            if result.warnings
            else "The price-strip renderer produced no PDF."
        )
        raise ValueError(message)
    
    return result.pdf_bytes


def _create_all_sides_pdf(name, drafts, *, eps_data=None):
    strips = []
    for side, draft in sorted(drafts.items()):
        for number, row in zip(_row_numbers(draft), draft['rows']):
            size = row.get('strip_size',draft['strip_size'])
            strips.append(_build_strip(name, side, number, row['tickets'], size,
                                       row.get('footer_text')))
    return _render_strips(strips, eps_data=eps_data)


def _spacing_compatible_drafts(drafts, data):
    """Separate unsupported sizes from malformed alignment for a supported size."""
    from app.templates.eps_preview import parse_card_spacing, matching_strip_rectangle, strip_crosshair_layout
    drawing = parse_card_spacing(data)
    matching = {}
    for side,draft in drafts.items():
        rows = []
        for number,row in zip(_row_numbers(draft),draft['rows']):
            size = row.get('strip_size',draft['strip_size'])
            try:
                matching_strip_rectangle(drawing,size)
            except ValueError:
                continue  # No outline of this size; do not invent its card spacing.
            strip_crosshair_layout(drawing,size,number)
            rows.append(dict(row,number=number))
        if rows:
            matching[side] = dict(draft,start=rows[0]['number'],rows=rows)
    if not matching:
        raise ValueError('Card Spacing has no strip size matching the imported spreadsheet. '
                         'Use a six-card drawing for six-position strips or an eight-card drawing for eight-position strips.')
    return matching


def _activate_sheet_pog():
    sheet = st.session_state[SHEET_KEY]
    previous = sheet.get('current')
    if previous is not None:
        sheet['saved'][previous] = {
            'drafts':deepcopy(st.session_state[ROWS_KEY]),
            'name':st.session_state.get('holiday_builder:name', previous),
        }
    pog = st.session_state[SHEET_POG_KEY]
    saved = sheet['saved'].get(pog)
    st.session_state[ROWS_KEY] = deepcopy(saved['drafts'] if saved else sheet['pogs'][pog])
    st.session_state['holiday_builder:name'] = saved['name'] if saved else pog
    sheet['active_name'] = st.session_state['holiday_builder:name']
    for key in list(st.session_state):
        if key.startswith(('holiday_builder:row_tickets:', 'holiday_builder:start:',
                           'holiday_builder:strip_size:', 'holiday_builder:preview_row:',
                           'holiday_builder:footer:')):
            st.session_state.pop(key, None)
    sides = st.session_state[ROWS_KEY]
    current_side = st.session_state.get('holiday_builder:side')
    st.session_state['holiday_builder:side'] = current_side if current_side in sides else min(sides)
    st.session_state.pop(PDF_KEY, None)
    sheet['current'] = pog


def _load_sheet_upload():
    from app.pages._price_strip_sheet import import_price_strip_sheet
    uploaded = st.session_state.get(SHEET_UPLOAD_KEY)
    st.session_state.pop(SHEET_ERROR_KEY, None)
    if uploaded is None:
        st.session_state.pop(SHEET_KEY, None)
        return
    try:
        selected_format = st.session_state.get(FORMAT_KEY, 'holiday')
        
        sheet = import_price_strip_sheet(
            uploaded.getvalue(),
            template_format=selected_format,
        )
    except ValueError as exc:
        st.session_state[SHEET_ERROR_KEY] = str(exc)
        return
    # Users can upload artwork before the workbook detects its format.
    uploaded_eps = st.session_state.get(EPS_KEY)
    uploaded_spacing = st.session_state.get(CARD_SPACING_KEY)
    spacing_error = st.session_state.get(CARD_SPACING_ERROR_KEY)
    _switch_format(sheet.get('format', 'holiday'))
    if uploaded_eps is not None:
        st.session_state[EPS_KEY] = uploaded_eps
    if uploaded_spacing is not None:
        st.session_state[CARD_SPACING_KEY] = uploaded_spacing
    if spacing_error is not None:
        st.session_state[CARD_SPACING_ERROR_KEY] = spacing_error
    sheet.update(name=uploaded.name, current=None, saved={})
    st.session_state[SHEET_KEY] = sheet
    st.session_state[SHEET_POG_KEY] = next(iter(sheet['pogs']))
    _activate_sheet_pog()
    st.session_state['holiday_builder:all_sides'] = True


def _sheet_import_controls():
    st.file_uploader('Upload price-strip data (.xlsx)', type=['xlsx'], key=SHEET_UPLOAD_KEY,
                     on_change=_load_sheet_upload,
                     help='Reads Holiday data or Generic merge data and fills products by side, row and position.')
    st.file_uploader('Card Spacing', type=['eps','pdf'], key=CARD_SPACING_UPLOAD_KEY,
                     accept_multiple_files=True,
                     on_change=_save_card_spacing_upload,
                     help='Select the six-card and eight-card EPS/PDF files together to generate all sides in one PDF. Use one spacing file per strip size, at actual print dimensions, with green card-center crosshairs.')
    if st.session_state.get(CARD_SPACING_ERROR_KEY):
        st.error(st.session_state[CARD_SPACING_ERROR_KEY])
    spacing = st.session_state.get(CARD_SPACING_KEY)
    if spacing:
        st.caption(f"Card Spacing: {spacing['name']}. Green crosshairs control horizontal price alignment.")
    elif not st.session_state.get(CARD_SPACING_ERROR_KEY):
        st.caption('Upload Card Spacing to center prices on the green crosshairs. Otherwise, the template spacing is used.')
    if st.session_state.get(SHEET_ERROR_KEY):
        st.error(st.session_state[SHEET_ERROR_KEY])
        st.caption('Import failed. Your existing rows are unchanged.')
    sheet = st.session_state.get(SHEET_KEY)
    if sheet:
        st.selectbox('POG from sheet', list(sheet['pogs']), key=SHEET_POG_KEY,
                      index=list(sheet['pogs']).index(sheet['current']), on_change=_activate_sheet_pog)
        st.caption(f"Loaded {sheet['name']}. Imported data is read-only.")
        for warning in sheet['warnings']:
            st.warning(warning)


def _save_eps_upload():
    uploaded = st.session_state.get(EPS_UPLOAD_KEY)
    if uploaded is None:
        st.session_state.pop(EPS_KEY, None)
    else:
        st.session_state[EPS_KEY] = {"name": uploaded.name, "bytes": uploaded.getvalue()}
    st.session_state.pop(PDF_KEY, None)


def _save_card_spacing_upload():
    uploaded = st.session_state.get(CARD_SPACING_UPLOAD_KEY)
    st.session_state.pop(CARD_SPACING_ERROR_KEY, None)
    if not uploaded:
        st.session_state.pop(CARD_SPACING_KEY, None)
    else:
        files = uploaded if isinstance(uploaded,(list,tuple)) else [uploaded]
        sources = [{'name':file.name,'bytes':file.getvalue()} for file in files]
        try:
            if len(sources)==1:
                spacing = sources[0]
            else:
                from app.templates.eps_preview import combine_card_spacing
                spacing = {'name':', '.join(source['name'] for source in sources),
                           'bytes':combine_card_spacing(sources)}
        except ValueError as exc:
            st.session_state.pop(CARD_SPACING_KEY, None)
            st.session_state[CARD_SPACING_ERROR_KEY] = f'Card Spacing: {exc}'
        else:
            st.session_state[CARD_SPACING_KEY] = spacing
    st.session_state.pop(PDF_KEY, None)
    st.session_state.pop('holiday_builder:pdf_scope', None)


@st.cache_data(show_spinner=False, max_entries=12)
def _eps_preview(data, name, side, row_number, tickets, strip_size, *, footer_text=None):
    from app.templates.eps_preview import parse_eps, render_eps_preview
    # Validate the uploaded drawing before starting the print renderer.
    parse_eps(data)
    pdf = _create_pdf(name, side, row_number, [tickets], strip_size=strip_size,
                      calibration=True, footers=[footer_text], eps_data=data)
    return render_eps_preview(data, strip_size, pdf, guides_in_pdf=True, row_number=row_number)


def _active_eps():
    custom = st.session_state.get(EPS_KEY)
    if custom is not None:
        return custom
    try:
        if st.session_state.get(FORMAT_KEY) == 'two_card':
            return {'name':"Sam's Two Cards baseline",'bytes':TWO_CARD_EPS_PATH.read_bytes(),
                    'use_card_centers':True}
        if st.session_state.get(FORMAT_KEY) == 'kk250133':
                return {
                    'name': "Sam's KK250133-11 baseline",
                    'bytes': KK250133_EPS_PATH.read_bytes(),
                    'use_card_centers': False,
                }
        if st.session_state.get(FORMAT_KEY) == 'generic':
            return {'name':'Generic baseline template (KK240100-13)', 'bytes':GENERIC_EPS_PATH.read_bytes(),
                    'use_card_centers':False}
        return {"name": "Sam's Holiday baseline template", "bytes": BASELINE_EPS_PATH.read_bytes(),
                'use_card_centers':False}
    except OSError as exc:
        raise ValueError('The baseline EPS is missing. Restore it or upload an EPS template.') from exc
       

@st.cache_data(show_spinner=False, max_entries=12)
def _live_strip_preview(
    name, side, row_number, tickets, strip_size, footer_text=None, eps_data=None
):
    import pymupdf
    
    pdf = _create_pdf(
        name, 
        side,
        row_number,
        [tickets],
        strip_size=strip_size,
        calibration=True,
        footers=[footer_text],
        eps_data=eps_data,
    )
    
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        page = document[0]
        scale = min(2, 1800 / page.rect.width)
        
        image = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale)
        ).tobytes("png")
        
    return image, pdf
def render():
    st.subheader("Price Strips")
    st.caption("Upload a spreadsheet to preview and generate price strips.")
    formats = {'holiday':"Sam's Holiday", 'generic':'Generic (KK240100-13)',
               'kk250133': "Sam's 1 15/16 inch (KK250133-11)",
               'two_card': "Sam's Two Cards (17 x 4 inches)", 'unique':'Unique'}
    st.selectbox('Template format', list(formats),
                 index=list(formats).index(st.session_state.get(FORMAT_KEY, 'holiday')),
                 format_func=formats.get,
                 key=FORMAT_WIDGET_KEY, on_change=_switch_format)
    
    st.file_uploader(
        "Upload a different EPS (optional)", type=["eps"], key=EPS_UPLOAD_KEY,
        on_change=_save_eps_upload,
        help="Upload an ArtiosCAD / Illustrator 8 EPS for the strip outline. Use Card Spacing below the data upload to control price centering.",
    )
    _sheet_import_controls()
    
    sheet = st.session_state.get(SHEET_KEY)

    if not sheet or st.session_state.get(SHEET_ERROR_KEY) or st.session_state.get(CARD_SPACING_ERROR_KEY):
        st.session_state.pop(PDF_KEY, None)
        if not st.session_state.get(CARD_SPACING_ERROR_KEY):
            st.info("Upload a valid spreadsheet to preview and generate price strips.")
        return

    name = sheet["current"]

    # Use the original import, not the former editable draft.
    source = sheet["pogs"][name]

    entries = [
        (side, number, row.get('strip_size',draft["strip_size"]), row)
        for side, draft in sorted(source.items())
        for number, row in zip(_row_numbers(draft), draft["rows"])
    ]

    if not entries:
        st.warning("No strip rows were found.")
        return
    source_count = len(entries)

    try:
        eps = _active_eps()
    except ValueError as exc:
        st.session_state.pop(PDF_KEY, None)
        st.error(str(exc))
        return
    card_spacing = st.session_state.get(CARD_SPACING_KEY)
    if card_spacing is None and st.session_state.get(FORMAT_KEY)=='two_card':
        card_spacing = eps  # This template's supplied marks are its baseline spacing.
    alignment_eps = card_spacing['bytes'] if card_spacing is not None else None
    compatible, output_source, output_scope = source, source, 'all sides'
    if alignment_eps is not None:
        try:
            compatible = _spacing_compatible_drafts(source,alignment_eps)
        except ValueError as exc:
            st.session_state.pop(PDF_KEY,None)
            st.error(f'Card Spacing: {exc}')
            return
        compatible_rows = {(side,number) for side,draft in compatible.items() for number in _row_numbers(draft)}
        if len(compatible_rows) != source_count:
            missing = ', '.join(f'Side {side}, Row {number}' for side,number,_,_ in entries
                                if (side,number) not in compatible_rows)
            st.info(f'Card Spacing matches {len(compatible_rows)} of {source_count} strip rows. '
                    f'These rows need a different strip size: {missing}. '
                    'Choose matching strips to test this file, or add the other spacing file in Card Spacing for all sides.')
            output_scope = st.radio('PDF output', ['matching strips only','all sides'],
                                    horizontal=True,key='holiday_builder:pdf_scope')
            output_source = compatible if output_scope=='matching strips only' else source
            # Keep previews on supported sizes even when all-side output is
            # unavailable, so the user can still inspect their uploaded file.
            entries = [entry for entry in entries if (entry[0],entry[1]) in compatible_rows]
    output_count = sum(len(draft['rows']) for draft in output_source.values())
    all_sizes_available = sum(len(draft['rows']) for draft in compatible.values())==source_count
    output_allowed = output_scope!='all sides' or all_sizes_available
    signature = ('card-spacing-upload-v7', name, repr(source), output_scope, sha256(eps['bytes']).hexdigest(),
                 sha256(alignment_eps).hexdigest() if alignment_eps is not None else None)
    previous = st.session_state.get(PDF_KEY)
    if previous and previous.get('signature') != signature:
        st.session_state.pop(PDF_KEY, None)
    st.caption(f"Strip outline: {eps['name']}.")

    st.caption(
        f"{len(source)} sides / {source_count} strip rows in the spreadsheet. "
        "Data is read-only. Correct the spreadsheet and upload it again "
        "to make changes."
    )
    if not output_allowed:
        st.warning('All-side export needs spacing for every strip size. Select matching strips only to generate this test.')
    else:
        st.caption(f'PDF output: {output_scope} - {output_count} pages.')

    selected = st.selectbox(
        "Preview strip",
        range(len(entries)),
        format_func=lambda index: (
            f"Side {entries[index][0]}, Row {entries[index][1]}"
        ),
    )

    side, number, size, row = entries[selected]

    with st.expander("View imported products"):
        st.dataframe(
            row["tickets"],
            hide_index=True,
            width="stretch",
        )
        if row.get("footer_text"):
            st.caption(row["footer_text"])

    preview_valid = False
    comparison_overview = None
    try:
        from app.templates.eps_preview import parse_eps, matching_strip_rectangle
        matching_strip_rectangle(parse_eps(eps['bytes']),size,number)
        preview_image, preview_pdf = _live_strip_preview(
            name,
            side,
            number,
            row["tickets"],
            size,
            footer_text=row.get("footer_text"),
            eps_data=alignment_eps,
        )
        if alignment_eps is not None:
            from app.templates.eps_preview import render_eps_preview
            comparison_overview,preview_image = render_eps_preview(
                alignment_eps,size,preview_pdf,guides_in_pdf=True,row_number=number,price_above=True)
    except ValueError as exc:
        st.error(str(exc))
    else:
        preview_valid = True
        st.image(
            preview_image,
            width="stretch",
            caption=f"Side {side}, Row {number} - " + (
                'price strip above card positions' if alignment_eps is not None else 'print-position preview'),
        )
        if alignment_eps is not None:
            st.caption('Top: printable price strip. Bottom: Card Spacing drawing. Red centerlines align with the green crosshairs.')
            from app.templates.eps_preview import parse_card_spacing, card_spacing_overflow
            overhang = card_spacing_overflow(parse_card_spacing(alignment_eps),size,number)
            if max(overhang) > .05:
                st.warning(f'The card layout extends beyond the strip by {overhang[0]/72:.4f} in on the left '
                           f'and {overhang[1]/72:.4f} in on the right. Price centers are preserved; the cards do not fit within the strip.')
            with st.expander('Full Card Spacing drawing'):
                st.image(comparison_overview, caption=card_spacing['name'], width='stretch')
        st.caption("Guide lines are excluded from the finished PDF.")

        with st.expander("Full EPS drawing"):
            from app.templates.eps_preview import render_eps_preview

            try:
                overview, _ = render_eps_preview(
                    eps['bytes'],size,preview_pdf,guides_in_pdf=True,row_number=number)
                st.image(
                    overview,
                    caption=eps["name"],
                    width="stretch",
                )
            except ValueError as exc:
                st.warning(str(exc))

    if st.button(
        "Create Price Strip PDF",
        type="primary",
        key="holiday_builder:create_pdf",
        disabled=not preview_valid or not output_allowed,
    ):
        try:
            with st.spinner("Creating price strips..."):
                pdf = _create_all_sides_pdf(name, output_source, eps_data=alignment_eps)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[PDF_KEY] = {
                "signature": signature,
                "bytes": pdf,
                "scope":output_scope,
                "pages":output_count,
            }

    result = st.session_state.get(PDF_KEY)

    if result:
        st.success(
            f"Ready: {name} - {result['scope']} ({result['pages']} pages)"
        )
        st.download_button(
            "Download Price Strip PDF",
            data=result["bytes"],
            file_name=('price_strips_all_sides.pdf' if result['scope']=='all sides' else 'price_strips_matching.pdf'),
            mime="application/pdf",
            on_click="ignore",
        )
