"""Product UPC image coverage beside the POG template controls."""
import csv
import io

import streamlit as st

from app.pages._inputs import text_input, upload_file, selectbox
from app.shared.workflow_state import fingerprint
from app.templates.image_compare import (DEFAULT_IMAGE_URL, compare_products,
                                         read_comparison_workbook)


RESULT_KEY = 'pog_compare:result'


@st.cache_data(show_spinner=False, max_entries=4)
def _read_sheet(data):
    return read_comparison_workbook(data)


def render_comparison():
    with st.container(border=True):
        st.subheader('Compare spreadsheet with image library')
        st.caption('Accuracy means the percentage of spreadsheet product rows with an exact UPC image match. '
                   'This does not verify names, CPP, prices or POG placement.')
        url = text_input('Image library link', value=DEFAULT_IMAGE_URL, key='pog_compare:url')
        workbook = upload_file('Excel data sheet (.xlsx)', type=['xlsx'], key='pog_compare:workbook')
        upc_format = selectbox('Spreadsheet UPC format',
            ['BOS UPC without check digit', 'Full UPC / GTIN'], key='pog_compare:upc_format')
        sheets, selected, payload = {}, None, None
        if workbook:
            payload = workbook.getvalue()
            try:
                sheets = _read_sheet(payload)
            except ValueError as exc:
                st.error(str(exc))
            if sheets:
                selected = st.selectbox('Worksheet to compare', list(sheets), key='pog_compare:sheet')
        signature = fingerprint(url=url.strip(), workbook=payload, sheet=selected, upc_format=upc_format, 
                                match_rule="ignore_leading_zeros_first11_v2",)
        previous = st.session_state.get(RESULT_KEY)
        if previous and previous['signature'] != signature:
            st.session_state.pop(RESULT_KEY, None)
        if st.button('Compare', key='pog_compare:compare', disabled=not sheets):
            st.session_state.pop(RESULT_KEY, None)
            try:
                with st.spinner('Comparing product UPCs with the image library...'):
                    result = compare_products(sheets[selected], url,
                        bos_upc11=upc_format == 'BOS UPC without check digit')
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[RESULT_KEY] = {'signature':signature, 'result':result}
        saved = st.session_state.get(RESULT_KEY)
        if saved:
            result = saved['result']
            score, matched, missing = st.columns(3)
            score.metric('Accuracy (UPC image match)',
                         f"{result['accuracy']:.1f}%" if result['accuracy'] is not None else 'Unavailable')
            matched.metric('Matched product rows', f"{result['matched']} / {result['total']}")
            missing.metric('Missing images / invalid UPCs', result['missing'])
            if result['unverified']:
                st.warning(f"{result['unverified']} product rows could not be checked because of connection or access errors. "
                           'Retry Compare to calculate an accuracy percentage.')
            else:
                st.caption(f"{result['matched']} of {result['total']} product rows matched. "
                           'BOS mode calculates the missing UPC check digit. Leading zeros are normalized. '
                           'Duplicate product rows are counted separately.')
            st.dataframe(result['rows'], hide_index=True, width='stretch',
                          column_config={'Image':st.column_config.LinkColumn('Matched image')})
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(result['rows'][0]))
            writer.writeheader()
            writer.writerows(result['rows'])
            st.download_button('Download comparison CSV', output.getvalue().encode('utf-8-sig'),
                               file_name='pog_image_comparison.csv', mime='text/csv',
                               key='pog_compare:download', on_click='ignore')
