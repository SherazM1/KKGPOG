"""Create price-strip templates using the Holiday ones layout"""
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from pathlib import Path

import streamlit as st

from app.sams_club.holiday_price_strips import (holiday_geometry_for_side,
                                               holiday_geometry_for_size)
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
STRIP_SIZES = {
    "thin": "30 1/4 × 3 7/16 inches — 6 positions",
    "wide": "38 7/16 × 3 7/16 inches — 8 positions",
}


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
    if side not in drafts:
        tickets = _default_tickets(side)
        # Keep entries from the former single-row editor in open sessions.
        previous = st.session_state.get(f"holiday_builder:tickets:{side}", {})
        for index, changes in previous.get("edited_rows", {}).items():
            if 0 <= int(index) < len(tickets):
                tickets[int(index)].update(changes)
        drafts[side] = {
            "start": st.session_state.get("holiday_builder:row", 1),
            "rows": [{"tickets": tickets}],
        }
    drafts[side].setdefault("strip_size", holiday_geometry_for_side(side).designation)
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
    sheet.update(name=uploaded.name, current=None, saved={})
    st.session_state[SHEET_KEY] = sheet
    st.session_state[SHEET_POG_KEY] = next(iter(sheet['pogs']))
    _activate_sheet_pog()
    st.session_state['holiday_builder:all_sides'] = True


def _sheet_import_controls():
    st.file_uploader('Upload price-strip data (.xlsx)', type=['xlsx'], key=SHEET_UPLOAD_KEY,
                     on_change=_load_sheet_upload,
                     help='Reads the Price Strip Data sheet and fills products by POG, side, row and column.')
    if st.session_state.get(SHEET_ERROR_KEY):
        st.error(st.session_state[SHEET_ERROR_KEY])
        st.caption('Import failed. Your existing rows are unchanged.')
    sheet = st.session_state.get(SHEET_KEY)
    if sheet:
        st.selectbox('POG from sheet', list(sheet['pogs']), key=SHEET_POG_KEY,
                      index=list(sheet['pogs']).index(sheet['current']), on_change=_activate_sheet_pog)
        st.caption(f"Loaded {sheet['name']}. Choose a side below to review or edit its products.")
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
    return render_eps_preview(data, strip_size, pdf, guides_in_pdf=True)


def _active_eps():
    custom = st.session_state.get(EPS_KEY)
    if custom is not None:
        return custom
    try:
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
    st.caption(
        "Add as many rows as you need. Each row prints as one full-width PDF page."
    )
    _sheet_import_controls()
    
    name = st.text_input(
        "Template name",
        value=None if 'holiday_builder:name' in st.session_state else
            st.session_state.get(SHEET_KEY, {}).get('active_name', "Holiday Price Strips"),
        key="holiday_builder:name",
    )
    if st.session_state.get(SHEET_KEY):
        st.session_state[SHEET_KEY]['active_name'] = name
    
    left, right = st.columns(2)
    
    side = left.selectbox(
        "Side",
        [1, 2, 3, 4],
        key="holiday_builder:side",
    )
    
    draft = _side_rows(side)
    row_number = right.number_input(
        "Starting row",
        min_value=1,
        value=draft["start"],
        step=1,
        key=f"holiday_builder:start:{side}",
    )
    draft["start"] = row_number
    numbers = _row_numbers(draft)
    
    strip_size = st.selectbox(
        "Strip dimensions", list(STRIP_SIZES),
        index=list(STRIP_SIZES).index(draft["strip_size"]),
        format_func=STRIP_SIZES.get,
        key=f"holiday_builder:strip_size:{side}",
    )
    draft["strip_size"] = strip_size
    geometry = holiday_geometry_for_size(strip_size)

    st.file_uploader(
        "Upload a different EPS (optional)", type=["eps"], key=EPS_UPLOAD_KEY,
        on_change=_save_eps_upload,
        help="The Sam's Holiday EPS is used automatically. Upload an ArtiosCAD / Illustrator 8 EPS to override it.",
    )
    
    st.caption(
        f"{geometry.slot_count} positions across a "
        f"{geometry.width_in:g} x {geometry.height_in:g} inch strip."
    )
    
    st.button("Add Row", key=f"holiday_builder:add_row:{side}",
              on_click=_add_row, args=(side,))
    st.caption(f"{len(draft['rows'])} row(s) on Side {side}")

    for index, row in enumerate(draft["rows"]):
        if len(row["tickets"]) < geometry.slot_count:
            row["tickets"].extend(_default_tickets(side, strip_size)[len(row["tickets"]):])
        editor_key = f"holiday_builder:row_tickets:{side}:{index}:{strip_size}"
        if editor_key not in st.session_state:
            row["editor_base"] = deepcopy(row["tickets"][:geometry.slot_count])
        with st.expander(f"Row {numbers[index]}", expanded=True):
            # A stable editor base prevents edits from being applied twice.
            # Persist the returned values outside widget state so switching
            # sides or template types does not erase the entered products.
            edited = st.data_editor(
                row["editor_base"], hide_index=True, num_rows="fixed",
                width="stretch", key=editor_key,
            )
            row["tickets"] = edited + row["tickets"][geometry.slot_count:]
            if 'footer_text' in row:
                row['footer_text'] = st.text_input('Row footer', value=row['footer_text'],
                    key=f'holiday_builder:footer:{side}:{index}')

    ticket_rows = [row["tickets"][:geometry.slot_count] for row in draft["rows"]]

    st.subheader("Price strip preview")
    
    selected_row = st.selectbox(
        "Preview row",
        range(len(ticket_rows)),
        format_func=lambda index: f"Row {numbers[index]}",
        key=f"holiday_builder:preview_row:{side}",
    )
    
    try:
        with st.spinner("Updating preview..."):
            preview_image, preview_pdf = _live_strip_preview(
                name.strip(),
                side,
                numbers[selected_row],
                ticket_rows[selected_row],
                strip_size,
                footer_text=draft["rows"][selected_row].get(
                    "footer_text"
                ),
            )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.caption(
            "Blue: product dividers. "
            "Red: strip outline and dashed centers. "
            "Green: alignment marks. "
        )
        
        st.image(
            preview_image,
            caption=(
                f"Side {side}, Row {numbers[selected_row]} "
                "- print-position preview"
            ),
            width="stretch",
        )
        
        st.caption(
            "Preview updates when you change the products. "
            "Guide lines are excluded from the finished PDF. "
        )
        
        try:
            uploaded_eps = _active_eps()
        except ValueError as exc:
            st.error(str(exc))
            uploaded_eps = None
        
        if uploaded_eps:
            st.caption(f"Using {uploaded_eps['name']}.")
            with st.expander("Full EPS drawing"):
                from app.templates.eps_preview import (
                    render_eps_preview,
                )
                
                try:
                    overview, _ = render_eps_preview(
                        uploaded_eps["bytes"],
                        strip_size,
                        preview_pdf,
                        guides_in_pdf=True,
                    )
                    
                except ValueError as exc:
                    st.error(str(exc))
                
                else:
                    st.image(
                        overview,
                        caption=uploaded_eps["name"],
                        width="stretch",
                    )
    sheet = st.session_state.get(SHEET_KEY)
    all_sides = st.checkbox('Include all sides in PDF',
                            value=False if 'holiday_builder:all_sides' in st.session_state else
                                (sheet.get('all_sides', False) if sheet else False),
                            key='holiday_builder:all_sides')
    if sheet:
        sheet['all_sides'] = all_sides
    signature = (name.strip(), side, row_number, strip_size, repr(ticket_rows),
                 tuple(numbers), tuple(row.get('footer_text') for row in draft['rows']), all_sides,
                 repr(st.session_state[ROWS_KEY]) if all_sides else None)
    previous = st.session_state.get(PDF_KEY)
    if previous and previous.get("signature") != signature:
        st.session_state.pop(PDF_KEY, None)

    submitted = st.button("Create Price Strip PDF", type="primary",
                          key="holiday_builder:create_pdf")
    
    if submitted:
        if not name.strip():
            st.error("Enter a template name.")
            return
        
        try:
            with st.spinner("Creating price strips..."):
                if all_sides:
                    pdf = _create_all_sides_pdf(name.strip(), st.session_state[ROWS_KEY])
                else:
                    pdf = _create_pdf(name.strip(), side, row_number, ticket_rows, strip_size=strip_size,
                        row_numbers=numbers, footers=[row.get('footer_text') for row in draft['rows']])
        except ValueError as exc:
            st.error(str(exc))
            return
        
        st.session_state[PDF_KEY] = {
            "name": name.strip(),
            "side": side,
            "row": row_number,
            "strip_size": strip_size,
            "last_row": numbers[-1],
            "row_count": len(ticket_rows),
            "signature": signature,
            "bytes": pdf,
            "all_sides": all_sides,
            "pages": sum(len(d['rows']) for d in st.session_state[ROWS_KEY].values()) if all_sides else len(ticket_rows),
        }        
        
    result = st.session_state.get(PDF_KEY)
    
    if result:
        if result.get('all_sides'):
            st.success(f"Ready: {result['name']} - all sides ({result['pages']} pages)")
        else:
            st.success(
                f"Ready: {result['name']} - "
                f"Side {result['side']}, Rows {result['row']}-{result['last_row']} "
                f"({result['row_count']} pages)"
            )
        
        st.download_button(
            "Download Price Strip PDF",
            data=result["bytes"],
            file_name='price_strips_all_sides.pdf' if result.get('all_sides') else (
                f"price_strip_side_{result['side']}"
                f"_rows_{result['row']}-{result['last_row']}.pdf"
            ),
            mime="application/pdf",
            on_click="ignore",
        )
