"""Translate production price-strip workbooks into editable builder rows."""
import io
from decimal import Decimal, InvalidOperation

from openpyxl import load_workbook

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import holiday_geometry_for_side
from app.sams_club.generic_price_strips import GENERIC_SIZES
from app.sams_club.generic_price_strips import geometry_for_size

def _generic_sheet(data):
    """Recognize the Generic merge schema; leave production workbooks to their reader."""
    try:
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception:
        return None
    try:
        for sheet in book:
            records = sheet.iter_rows(values_only=True)
            headers = [str(v or '').strip().upper() for v in next(records, ())]
            required = {'UPC12', 'BRAND', 'SIGNING DESC 1', 'SIGNING DESC 2', 'SIDE', 'ROW', 'POSITION'}
            if not required.issubset(headers):
                continue
            grouped, occupied = {}, set()
            products = 0
            for source_row, values in enumerate(records, 2):
                if not any(v is not None and str(v).strip() for v in values):
                    continue
                record = dict(zip(headers, values))
                try:
                    numbers = [Decimal(str(record[key])) for key in ('SIDE', 'ROW', 'POSITION')]
                    if any(not n.is_finite() or n != int(n) for n in numbers):
                        raise ValueError
                    side, row, position = map(int, numbers)
                    if side not in (1, 2, 3, 4) or row < 1 or not 1 <= position <= 6:
                        raise ValueError
                except (ValueError, TypeError, InvalidOperation):
                    raise ValueError(f'Excel row {source_row}: use Side 1–4, a positive Row and Position 1–6.') from None
                key = side, row, position
                if key in occupied:
                    raise ValueError(f'Excel row {source_row}: duplicate position {position} on Side {side}, Row {row}.')
                occupied.add(key)
                raw = record.get('RETAIL PRICE')
                try:
                    amount = (Decimal(str(raw).replace('$', '').replace(',', ''))
                              if raw is not None and str(raw).strip() else
                              Decimal(str(record['DOLLAR'])) + Decimal(str(record['CENTS'])) / 100)
                    if not amount.is_finite() or amount < 0:
                        raise ValueError
                except (ValueError, TypeError, KeyError, InvalidOperation):
                    raise ValueError(f'Excel row {source_row}: enter a valid nonnegative Retail Price.') from None
                target = grouped.setdefault(side, {}).setdefault(row, {'tickets':{}, 'footer_text':''})
                footer = str(record.get('MESSAGE ON BOTTOM LEFT OF STRIP') or '').strip()
                if footer:
                    if target['footer_text'] and target['footer_text'] != footer:
                        raise ValueError(f'Side {side}, Row {row}: conflicting strip footer messages.')
                    target['footer_text'] = footer
                target['tickets'][position] = {
                    'Brand':str(record.get('BRAND') or '').strip(),
                    'Pack / value':str(record.get('SIGNING DESC 1') or '').strip(),
                    'Type':str(record.get('SIGNING DESC 2') or '').strip(),
                    'Price':f'{amount:.2f}', 'Item #':str(record.get('UPC12') or '').strip(),
                }
                products += 1
            if not grouped:
                raise ValueError('No Generic price-strip products were found in the workbook.')
            sides = {}
            for side, rows in sorted(grouped.items()):
                maximum = max(max(row['tickets']) for row in rows.values())
                size = next(key for key, (_, centers) in GENERIC_SIZES.items() if len(centers) >= maximum)
                count = len(GENERIC_SIZES[size][1])
                sides[side] = {'start':min(rows), 'strip_size':size, 'rows':[
                    {'number':number, 'footer_text':row['footer_text'], 'tickets':[
                        row['tickets'].get(position, {key:'' for key in ('Brand', 'Pack / value', 'Type', 'Price', 'Item #')})
                        for position in range(1, count + 1)]}
                    for number, row in sorted(rows.items())]}
            return {'format':'generic', 'pogs':{'Generic Price Strips':sides}, 'warnings':[],
                    'products':products, 'rows':sum(len(s['rows']) for s in sides.values())}
        return None
    finally:
        book.close()


def import_price_strip_sheet(data, template_format=None):
    if template_format not in ('kk250133','two_card'):
        
        generic = _generic_sheet(data)
        if generic is not None:
            return generic
    # This builder derives dimensions from side geometry and generates a
    # footer when the workbook does not supply either layout-only column.
    build = build_sams_price_strip_rows(data, require_layout_columns=False)
    if build.errors:
        raise ValueError('\n'.join(build.errors))
    if build.skipped_segment_count:
        raise ValueError('Some records have invalid POG, Side, Row or Column values. '
                         'Correct the sheet before importing.\n' + '\n'.join(build.warnings[:5]))
    if not build.strip_rows:
        raise ValueError('No price-strip rows were found in the workbook.')
    strip_rows = build.strip_rows
    warnings = list(build.warnings)
    if template_format == 'kk250133':
        incompatible = {}
        for strip in strip_rows:
            count = len(strip.segments)
            if count not in (8,10):
                incompatible.setdefault(strip.pog,
                    f'Side {strip.side}, Row {strip.row}: this template requires exactly 8 or 10 prices, '
                    f'but {count} product positions were supplied.')
        strip_rows = [strip for strip in strip_rows if strip.pog not in incompatible]
        if not strip_rows:
            raise ValueError('\n'.join(f'{pog}: {reason}' for pog,reason in incompatible.items()))
        warnings.extend(f'Skipped POG "{pog}" for the 1 15/16-inch template. {reason}'
                        for pog,reason in incompatible.items())
    pogs = {}
    for strip in strip_rows:
        if template_format == 'two_card':
            geometry = geometry_for_size('generic_two_card')
        elif template_format == 'kk250133':
            size = (
                'generic_kk250133_short'
                if len(strip.segments) == 8
                else 'generic_kk250133_long'
            )
            geometry = geometry_for_size(size)
        else:
            geometry = holiday_geometry_for_side(strip.side)
        sides = pogs.setdefault(strip.pog, {})
        draft = sides.setdefault(strip.side, {'start': strip.row,
                                             'strip_size': geometry.designation, 'rows': []})
        tickets = [{'Brand':'', 'Pack / value':'', 'Type':'', 'Price':'', 'Item #':''}
                   for _ in range(geometry.slot_count)]
        occupied = set()
        for segment in strip.segments:
            if not 1 <= segment.column <= geometry.slot_count:
                raise ValueError(f'{strip.pog}, Side {strip.side}, Row {strip.row}: '
                                 f'column {segment.column} exceeds {geometry.slot_count} positions.')
            if segment.column in occupied:
                raise ValueError(f'{strip.pog}, Side {strip.side}, Row {strip.row}: '
                                 f'duplicate column {segment.column}.')
            occupied.add(segment.column)
            tickets[segment.column - 1] = {
                'Brand': segment.brand, 'Pack / value': segment.desc_1,
                'Type': segment.desc_2, 'Price': segment.retail, 'Item #': segment.item_number,
            }
        if template_format in ('kk250133','two_card'):
            required = geometry.slot_count
            if len(occupied) != required:
                raise ValueError(
                    f"{strip.pog}, Side {strip.side}, Row {strip.row}: "
                    f"this template requires exactly {required} prices, "
                    f"but {len(occupied)} product positions were supplied."
                )
            for segment in strip.segments:
                try:
                    price = Decimal(str(segment.retail).replace('$','').replace(',','').strip())
                except (InvalidOperation, ValueError):
                    raise ValueError(
                        f"{strip.pog}, Side {strip.side}, Row {strip.row}, "
                        f"Column {segment.column}: enter a valid price."
                    ) from None
                if not price.is_finite() or price < 0:
                    raise ValueError(
                        f"{strip.pog}, Side {strip.side}, Row {strip.row}, "
                        f"Column {segment.column}: price must be a finite, nonnegative number."
                    )
        draft['rows'].append({'number':strip.row, 'footer_text':strip.footer_text,
                              'tickets':tickets, 'strip_size':geometry.designation})
    return {
        'format':template_format if template_format in ('kk250133','two_card') else 'holiday',
        'pogs':pogs,
        'warnings':warnings,
        'products':sum(len(strip.segments) for strip in strip_rows),
        'rows':len(strip_rows)}
