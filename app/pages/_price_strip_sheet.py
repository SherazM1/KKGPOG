"""Translate production price-strip workbooks into editable builder rows."""
from app.sams_club.extract_price_strips import build_sams_price_strip_rows
from app.sams_club.holiday_price_strips import holiday_geometry_for_side


def import_price_strip_sheet(data):
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
