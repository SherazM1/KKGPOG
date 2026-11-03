"""Larger scrollable editing projection; catalog/production previews stay unchanged."""
from xml.etree import ElementTree as ET

from app.pages._direct_builder import envelope, standard_size
from app.templates.validation import validate_template


COLORS = (('#eaf0fa','#647fa8'),('#f7f0df','#a28b55'),('#e9f3ec','#648d72'),('#f4eaf2','#9a6c93'))


def canvas_svg(draft, facing):
    validation = validate_template(draft.template)
    if any(i.code not in {'product_overlap','slot_bounds','slot_region_bounds'} for i in validation.errors):
        raise ValueError('Repair structural layout errors before using this canvas.')
    w,h = standard_size(draft)
    scale = max(760/facing.width,40/w)
    left,top = 16,24
    width,height = facing.width*scale,facing.height*scale
    root = ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg',
        'width':f'{width+32:.9f}','height':f'{height+48:.9f}',
        'viewBox':f'0 0 {width+32:.9f} {height+48:.9f}',
        'role':'group','aria-label':f'{facing.name} editing canvas','font-family':'Arial, sans-serif'})
    def add(tag, **attrs):
        return ET.SubElement(root,tag,{k.replace('_','-'):str(v) for k,v in attrs.items()})
    add('rect',x=left,y=top,width=width,height=height,fill='#fff',stroke='#64748b',data_kind='facing')
    colors = {}
    for region in facing.ordered_regions():
        fill,stroke = COLORS[(region.order-1)%len(COLORS)]
        bounds = envelope(facing,region,(w,h))
        add('rect',x=left+bounds.x*scale,y=top+bounds.y*scale,width=bounds.width*scale,
            height=bounds.height*scale,fill=fill,fill_opacity='.5',stroke=stroke,
            data_region_id=region.id,data_kind='region')
        add('text',x=left+bounds.x*scale+6,y=top+bounds.y*scale+18,
            fill=stroke,font_size=14,data_region_label=region.id).text=region.name
        for sid in region.slot_ids:
            colors.setdefault(sid,(fill,stroke))
    for slot in facing.ordered_slots():
        fill,stroke=colors.get(slot.id,('#f1f5f9','#526577'))
        x,y=left+slot.x*scale,top+slot.y*scale
        add('rect',x=x,y=y,width=slot.width*scale,height=slot.height*scale,
            fill=fill,stroke=stroke,data_slot_id=slot.id)
        add('text',x=x+slot.width*scale/2,y=y+slot.height*scale/2,fill='#202020',
            font_size=16,text_anchor='middle',dominant_baseline='middle').text=str(slot.order)
    return ET.tostring(root,encoding='unicode')
