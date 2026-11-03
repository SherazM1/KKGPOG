"""Fixed-size manual layout policy. No production or catalog dependencies."""
from dataclasses import replace
from math import floor

from app.templates import draft as edits
from app.templates.models import Bounds, GeometryUnit, SlotRole
from app.templates.validation import EPSILON


def standard_size(draft):
    metadata = draft.template.metadata
    base = draft.template.facings[0].width if draft.template.facings else 1.0
    return (float(metadata.get('manual_box_width', base*.10)),
            float(metadata.get('manual_box_height', base*.12)))


def _prepare(draft):
    width, height = standard_size(draft)
    if width <= 0 or height <= 0:
        raise ValueError('Set a positive Side width in Advanced side settings first.')
    return replace(draft, template=replace(draft.template, metadata={**draft.template.metadata,
        'manual_box_width': str(width), 'manual_box_height': str(height)}))


def _region(face, rid):
    return next(region for region in face.regions if region.id == rid)


def envelope(face, region, size):
    if region.bounds:
        return region.bounds
    members = face.slots_for_region(region.id)
    w, h = size
    y = max(0, min((s.y for s in members), default=h*.25) - h*.75)
    bottom = max((s.y+s.height for s in members), default=y+h*1.5) + h*.25
    return Bounds(0, y, max(face.width, max((s.x+s.width for s in members), default=0)), bottom-y)


def _store(draft, face):
    return replace(draft, template=replace(draft.template, facings=tuple(
        face if item.id == face.id else item for item in draft.template.facings)))


def _fit(draft, fid, rid, width, height):
    """Grow one workspace; translate later Sections together without resizing boxes."""
    face = edits.get_facing(draft, fid)
    size = standard_size(draft)
    region = _region(face, rid)
    old = envelope(face, region, size)
    delta = max(0, height-old.height)
    later = {r.id for r in face.regions if r.order > region.order}
    shifted_slots = {sid for r in face.regions if r.id in later for sid in r.slot_ids}
    if delta and any(sid in shifted_slots for r in face.regions if r.id not in later for sid in r.slot_ids):
        raise ValueError('This shared-box layout needs technical geometry edits before adding workspace.')
    regions = []
    for item in face.regions:
        bounds = envelope(face, item, size)
        if item.id == rid:
            bounds = replace(bounds, width=max(width, bounds.width), height=max(height, bounds.height))
        elif item.id in later:
            bounds = replace(bounds, y=bounds.y+delta)
        regions.append(replace(item, bounds=bounds))
    slots = tuple(replace(slot, y=slot.y+delta) if delta and slot.id in shifted_slots else slot for slot in face.slots)
    required_width = max(face.width, max((r.bounds.x+r.bounds.width for r in regions), default=0))
    required_height = max(face.height, max((r.bounds.y+r.bounds.height for r in regions), default=0)+size[1]*.25)
    if draft.template.unit == GeometryUnit.INCHES and (required_width > face.width+EPSILON or required_height > face.height+EPSILON):
        raise ValueError('Not enough room on this measured Side. Check its dimensions in Advanced side settings.')
    return _store(draft, replace(face, regions=tuple(regions), slots=slots,
                                width=required_width, height=required_height))


def add_section(draft, fid, name, role=SlotRole.PRODUCT, hint=None):
    draft = _prepare(draft)
    face = edits.get_facing(draft, fid)
    w, h = standard_size(draft)
    top = max((envelope(face,r,(w,h)).y+envelope(face,r,(w,h)).height for r in face.regions), default=0)+h*.5
    updated = edits.add_region(draft, fid, name, role, hint)
    face = edits.get_facing(updated, fid)
    region = replace(face.regions[-1], bounds=Bounds(0,top,face.width,h*2))
    updated = _store(updated, replace(face, regions=(*face.regions[:-1],region)))
    return _fit(updated,fid,region.id,face.width,h*2)


def create_grid(draft, fid, rid, *, rows, columns):
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 or v > 50 for v in (rows,columns)):
        raise ValueError('Rows and Columns must be whole numbers from 1 to 50.')
    draft = _prepare(draft)
    face = edits.get_facing(draft,fid)
    w,h = standard_size(draft)
    bounds = envelope(face,_region(face,rid),(w,h))
    members = face.slots_for_region(rid)
    x = bounds.x+w*.25
    y = max((s.y+s.height for s in members),default=bounds.y+h*.5)+h*.25
    width, height = columns*w+(columns-1)*w*.2, rows*h+(rows-1)*h*.2
    draft = _fit(draft,fid,rid,max(bounds.width,width+w*.5),max(bounds.height,y+height+h*.25-bounds.y))
    return edits.create_grid(draft,fid,rid,rows=rows,columns=columns,x=x,y=y,width=width,height=height,
                             horizontal_gap=w*.2,vertical_gap=h*.2)


def add_box(draft, fid, rid):
    draft = _prepare(draft)
    face = edits.get_facing(draft,fid)
    w,h = standard_size(draft)
    bounds = envelope(face,_region(face,rid),(w,h))
    columns = max(1,floor((bounds.width-w*.5+w*.2+EPSILON)/(w*1.2)))
    later = {r.id for r in face.regions if r.order > _region(face,rid).order}
    later_slots = {sid for r in face.regions if r.id in later for sid in r.slot_ids}
    obstacles = tuple(s for s in face.slots if s.id not in later_slots)
    # Later bands move down if this band grows, so they do not block its next row.
    for row in range(len(face.slots)+1):
        y = bounds.y+h*.75+row*h*1.2
        for column in range(columns):
            x = bounds.x+w*.25+column*w*1.2
            if not any(min(x+w,s.x+s.width)-max(x,s.x)>EPSILON and
                       min(y+h,s.y+s.height)-max(y,s.y)>EPSILON for s in obstacles):
                draft = _fit(draft,fid,rid,max(bounds.width,w*1.5),max(bounds.height,y+h+h*.25-bounds.y))
                return edits.add_slot(draft,fid,rid,x=x,y=y,width=w,height=h)
    raise ValueError('Make room in this Section before adding another box.')


def duplicate_box(draft, fid, rid, slot_id):
    source = next(s for s in edits.get_facing(draft,fid).slots_for_region(rid) if s.id == slot_id)
    result = add_box(draft,fid,rid)
    face = edits.get_facing(result,fid)
    copied = replace(face.slots[-1], role=source.role, row=source.row, column=source.column,
                     metadata=dict(source.metadata))
    return _store(result,replace(face,slots=(*face.slots[:-1],copied)))


def duplicate_section(draft, fid, rid, name):
    draft = _prepare(draft)
    face = edits.get_facing(draft,fid)
    w,h = standard_size(draft)
    source = _region(face,rid)
    bounds = envelope(face,source,(w,h))
    distance = bounds.height+h*.5
    # Reserve a blank band immediately below the source and shift later bands.
    expanded = _fit(draft,fid,rid,bounds.width,bounds.height+distance)
    face = edits.get_facing(expanded,fid)
    expanded = _store(expanded,replace(face,regions=tuple(
        replace(r,bounds=bounds) if r.id==rid else r for r in face.regions)))
    return edits.duplicate_region(expanded,fid,rid,name,dy=distance)
