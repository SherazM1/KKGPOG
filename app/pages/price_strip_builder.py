"""Create price-strip templates using the Holiday ones layout"""
from decimal import Decimal, InvalidOperation
from copy import deepcopy
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
        keys = (ROWS_KEY, SHEET_KEY, EPS_KEY, SHEET_POG_KEY, 'holiday_builder:name',
                'holiday_builder:side', 'holiday_builder:all_sides')
        saved = st.session_state.setdefault(FORMAT_DRAFTS_KEY, {})
        saved[previous] = {key:deepcopy(st.session_state[key]) for key in keys if key in st.session_state}
        for key in keys:
            st.session_state.pop(key, None)
        st.session_state.update(deepcopy(saved.get(target, {})))
        for key in list(st.session_state):
            if key.startswith(('holiday_builder:row_tickets:', 'holiday_builder:start:',
                               'holiday_builder:strip_size:', 'holiday_builder:preview_row:',
                               'holiday_builder:footer:')) or key in (EPS_UPLOAD_KEY, SHEET_UPLOAD_KEY):
                st.session_state.pop(key, None)
        st.session_state.pop(PDF_KEY, None)
        st.session_state.pop(SHEET_ERROR_KEY, None)
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
                row_numbers=None, footers=None):
    numbers = row_numbers if row_numbers is not None else range(row_number, row_number + len(ticket_rows))
    strips = [_build_strip(name, side, numbers[index], tickets, strip_size,
                           footers[index] if footers is not None else None)
              for index, tickets in enumerate(ticket_rows)]
    return _render_strips(strips, calibration)


def _render_strips(strips, calibration=False):
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


def _create_all_sides_pdf(name, drafts):
    strips = []
    for side, draft in sorted(drafts.items()):
        size = draft['strip_size']
        count = holiday_geometry_for_size(size).slot_count
        for number, row in zip(_row_numbers(draft), draft['rows']):
            strips.append(_build_strip(name, side, number, row['tickets'][:count], size,
                                       row.get('footer_text')))
    return _render_strips(strips)


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
        sheet = import_price_strip_sheet(uploaded.getvalue())
    except ValueError as exc:
        st.session_state[SHEET_ERROR_KEY] = str(exc)
        return
    _switch_format(sheet.get('format', 'holiday'))
    sheet.update(name=uploaded.name, current=None, saved={})
    st.session_state[SHEET_KEY] = sheet
    st.session_state[SHEET_POG_KEY] = next(iter(sheet['pogs']))
    _activate_sheet_pog()
    st.session_state['holiday_builder:all_sides'] = True


def _sheet_import_controls():
    st.file_uploader('Upload price-strip data (.xlsx)', type=['xlsx'], key=SHEET_UPLOAD_KEY,
                     on_change=_load_sheet_upload,
                     help='Reads Holiday data or Generic merge data and fills products by side, row and position.')
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


@st.cache_data(show_spinner=False, max_entries=12)
def _eps_preview(data, name, side, row_number, tickets, strip_size, *, footer_text=None):
    from app.templates.eps_preview import parse_eps, render_eps_preview
    # Validate the uploaded drawing before starting the print renderer.
    parse_eps(data)
    pdf = _create_pdf(name, side, row_number, [tickets], strip_size=strip_size,
                      calibration=True, footers=[footer_text])
    return render_eps_preview(data, strip_size, pdf, guides_in_pdf=True, row_number=row_number)


def _active_eps():
    custom = st.session_state.get(EPS_KEY)
    if custom is not None:
        return custom
    try:
        if st.session_state.get(FORMAT_KEY) == 'generic':
            return {'name':'Generic baseline template (KK240100-13)', 'bytes':GENERIC_EPS_PATH.read_bytes()}
        return {"name": "Sam's Holiday baseline template", "bytes": BASELINE_EPS_PATH.read_bytes()}
    except OSError as exc:
        raise ValueError('The baseline EPS is missing. Restore it or upload an EPS template.') from exc


@st.cache_data(show_spinner=False, max_entries=12)
def _live_strip_preview(
    name, side, row_number, tickets, strip_size, footer_text=None
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
    formats = {'holiday':"Sam's Holiday", 'generic':'Generic (KK240100-13)', 'unique':'Unique'}
    st.selectbox('Template format', list(formats),
                 index=list(formats).index(st.session_state.get(FORMAT_KEY, 'holiday')),
                 format_func=formats.get,
                 key=FORMAT_WIDGET_KEY, on_change=_switch_format)
    
    st.file_uploader(
        "Upload a different EPS (optional)", type=["eps"], key=EPS_UPLOAD_KEY,
        on_change=_save_eps_upload,
        help="The selected format's baseline EPS is used automatically. Upload an ArtiosCAD / Illustrator 8 EPS to override it.",
    )
    _sheet_import_controls()
    
    sheet = st.session_state.get(SHEET_KEY)

    if not sheet or st.session_state.get(SHEET_ERROR_KEY):
        st.session_state.pop(PDF_KEY, None)
        st.info("Upload a valid spreadsheet to preview and generate price strips.")
        return

    name = sheet["current"]

    # Use the original import, not the former editable draft.
    source = sheet["pogs"][name]

    entries = [
        (side, number, draft["strip_size"], row)
        for side, draft in sorted(source.items())
        for number, row in zip(_row_numbers(draft), draft["rows"])
    ]

    if not entries:
        st.warning("No strip rows were found.")
        return

    st.caption(
        f"{len(source)} sides / {len(entries)} strip rows. "
        "Data is read-only. Correct the spreadsheet and upload it again "
        "to make changes."
    )

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

    try:
        preview_image, preview_pdf = _live_strip_preview(
            name,
            side,
            number,
            row["tickets"],
            size,
            footer_text=row.get("footer_text"),
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.image(
            preview_image,
            width="stretch",
            caption=f"Side {side}, Row {number} - print-position preview",
        )
        st.caption("Guide lines are excluded from the finished PDF.")

        with st.expander("Full EPS drawing"):
            from app.templates.eps_preview import render_eps_preview

            try:
                eps = _active_eps()
                overview, _ = render_eps_preview(
                    eps["bytes"],
                    size,
                    preview_pdf,
                    guides_in_pdf=True,
                    row_number=number,
                )
                st.image(
                    overview,
                    caption=eps["name"],
                    width="stretch",
                )
            except ValueError as exc:
                st.warning(str(exc))

    signature = ("import-only-v1", name, repr(source))
    previous = st.session_state.get(PDF_KEY)

    if previous and previous.get("signature") != signature:
        st.session_state.pop(PDF_KEY, None)

    if st.button(
        "Create Price Strip PDF",
        type="primary",
        key="holiday_builder:create_pdf",
    ):
        try:
            with st.spinner("Creating price strips..."):
                pdf = _create_all_sides_pdf(name, source)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[PDF_KEY] = {
                "signature": signature,
                "bytes": pdf,
            }

    result = st.session_state.get(PDF_KEY)

    if result:
        st.success(
            f"Ready: {name} - all sides ({len(entries)} pages)"
        )
        st.download_button(
            "Download Price Strip PDF",
            data=result["bytes"],
            file_name="price_strips_all_sides.pdf",
            mime="application/pdf",
            on_click="ignore",
        )
