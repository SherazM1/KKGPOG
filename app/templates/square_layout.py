"""Center square print layouts without repacking away canvas edits."""
from statistics import median

from app.templates.print_style import section_header_color


def square_page_layout(facing, width, height):
    spans, assigned = [], set()
    bottom = max((slot.y + slot.height for slot in facing.slots), default=0)
    for region in facing.ordered_regions():
        members = facing.slots_for_region(region.id)
        assigned.update(slot.id for slot in members)
        start = min((slot.y for slot in members), default=bottom)
        stop = max((slot.y + slot.height for slot in members), default=start)
        spans.append((start, stop, members, [(region.name, section_header_color(region))]))
    unassigned = tuple(slot for slot in facing.ordered_slots() if slot.id not in assigned)
    if unassigned:
        spans.append((min(slot.y for slot in unassigned),
                      max(slot.y + slot.height for slot in unassigned), unassigned, []))

    # Sections that overlap vertically share one coordinate transform, keeping
    # side-by-side and staggered arrangements intact. Shared slots print once.
    groups = []
    for start, stop, members, titles in sorted(spans, key=lambda span: span[0]):
        if groups and start < groups[-1]['stop'] - 1e-9:
            group = groups[-1]
            group['stop'] = max(stop, group['stop'])
            group['slots'].update((slot.id, slot) for slot in members)
            group['titles'].extend(titles)
        else:
            groups.append({'stop':stop, 'slots':{slot.id:slot for slot in members}, 'titles':titles})

    for group in groups:
        slots = tuple(group['slots'].values())
        if not slots:
            group.update(cells=[], width=0, height=0, limit=0)
            continue
        base_w = median(slot.width for slot in slots)
        base_h = median(slot.height for slot in slots)
        left, top = min(slot.x for slot in slots), min(slot.y for slot in slots)
        cells = [(slot, (slot.x-left)/base_w, (slot.y-top)/base_h,
                  min(slot.width/base_w, slot.height/base_h)) for slot in slots]
        content_w = max(x + side for _, x, _, side in cells)
        content_h = max(y + side for _, _, y, side in cells)
        group.update(cells=cells, width=content_w, height=content_h, limit=width/content_w)

    title_count = sum(len(group['titles']) for group in groups)
    band_height = min(22, height/(3*title_count)) if title_count else 0
    gutter = min(4, height/(3*max(len(groups), 1)))
    budget = max(0, height-title_count*band_height-max(0, len(groups)-1)*gutter)
    low, high = 0.0, max((group['limit'] for group in groups), default=0)
    for _ in range(64):
        middle = (low+high)/2
        required = sum(group['height']*min(group['limit'], middle) for group in groups)
        if required <= budget:
            low = middle
        else:
            high = middle

    cards, bands, cursor = [], [], height
    for index, group in enumerate(groups):
        if index:
            cursor -= gutter
        for name, color in group['titles']:
            bands.append((name, color, cursor, band_height))
            cursor -= band_height
        scale = min(group['limit'], low)
        left = (width-group['width']*scale)/2
        # Coordinates, rather than row indices or survivor counts, determine
        # placement: deleted holes and manually moved rows remain visible.
        for slot, x, y, side in sorted(group['cells'], key=lambda cell: cell[0].order):
            size = side*scale
            cards.append((slot, (left+x*scale, cursor-(y+side)*scale, size, size)))
        cursor -= group['height']*scale
    return tuple(cards), tuple(bands)
