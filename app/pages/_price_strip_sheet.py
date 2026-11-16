"""Translate production price-strip workbooks into editable builder rows."""
import io
from decimal import Decimal, InvalidOperation

from openpyxl import load_workbook

from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import holiday_geometry_for_side
from app.sams_club.generic_price_strips import GENERIC_SIZES


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


def import_price_strip_sheet(data):
    generic = _generic_sheet(data)
    if generic is not None:
        return generic
    build = build_sams_price_strip_rows(data)
    if build.errors:
        raise ValueError('\n'.join(build.errors))
    if build.skipped_segment_count:
        raise ValueError('Some records have invalid POG, Side, Row or Column values. '
                         'Correct the sheet before importing.\n' + '\n'.join(build.warnings[:5]))
    if not build.strip_rows:
        raise ValueError('No price-strip rows were found in the workbook.')
    pogs = {}
    for strip in build.strip_rows:
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
        draft['rows'].append({'number':strip.row, 'footer_text':strip.footer_text, 'tickets':tickets})
    return {'pogs':pogs, 'warnings':build.warnings,
            'products':build.included_segment_count, 'rows':len(build.strip_rows)}
