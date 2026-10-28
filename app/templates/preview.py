"""Blank SVG geometry QA; SVG display coordinates never become fixture units."""
from collections import defaultdict
from xml.etree import ElementTree as ET

from app.templates.models import Bounds, DisplayTemplate, SlotRole
from app.templates.validation import validate_template


MAX_PREVIEW_WIDTH = 560
MAX_PREVIEW_HEIGHT = 520
_REGION_COLORS = (
    ("#eaf0fa", "#647fa8"), ("#f7f0df", "#a28b55"),
    ("#e9f3ec", "#648d72"), ("#f4eaf2", "#9a6c93"),
)


def _envelope(facing, region):
    if region.bounds is not None:
        return region.bounds
    members = facing.slots_for_region(region.id)
    if not members:
        return None
    x, y = min(slot.x for slot in members), min(slot.y for slot in members)
    return Bounds(x, y, max(slot.x + slot.width for slot in members) - x,
                  max(slot.y + slot.height for slot in members) - y)


def geometry_svg(template: DisplayTemplate, facing_id: str) -> str:
    validation = validate_template(template)
    if not validation.valid:
        raise ValueError("Cannot preview invalid geometry: " + "; ".join(issue.message for issue in validation.errors))
    facing = next((facing for facing in template.facings if facing.id == facing_id), None)
    if facing is None:
        raise KeyError(f"Unknown facing: {facing_id}")

    # Cap intrinsic size as well as viewport size. No width="stretch" in the UI.
    # A single scale preserves geometry; presentation hints never affect placement.
    left = 24
    top = 72 + ((len(facing.regions) + 1) // 2) * 16 if facing.regions else 64
    scale = min((MAX_PREVIEW_WIDTH - 48) / facing.width,
                (MAX_PREVIEW_HEIGHT - top - 20) / facing.height)
    width, height = facing.width * scale, facing.height * scale
    total_width, total_height = width + 48, height + top + 20
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": f"{total_width:g}", "height": f"{total_height:g}",
        "viewBox": f"0 0 {total_width:g} {total_height:g}",
        "style": "max-width:100%;height:auto", "font-family": "Arial, sans-serif",
        "role": "img", "aria-label": f"{template.name}: {facing.name}, blank geometry preview",
    })

    def add(tag, **attributes):
        return ET.SubElement(root, tag, {key.replace("_", "-"): str(value) for key, value in attributes.items()})

    add("title").text = f"{template.name} — {facing.name}"
    add("text", x=left, y=19, font_size=min(15, width / max(1, len(template.name) * .57)),
        fill="#202020").text = template.name
    add("text", x=left, y=36, font_size=12, fill="#333333").text = facing.name
    add("text", x=left, y=52, font_size=10.5, fill="#555555").text = (
        "Uncalibrated structure — not to physical scale"
        if template.unit == "normalized" else "Inches — scaled to fit this preview"
    )
    add("rect", x=left, y=top, width=width, height=height,
        fill="#ffffff", stroke="#222222", stroke_width=1.5, data_kind="facing")

    slot_colors = {}
    for index, region in enumerate(facing.ordered_regions()):
        fill, stroke = _REGION_COLORS[(region.order - 1) % len(_REGION_COLORS)]
        legend_x, legend_y = left + (index % 2) * width / 2, 70 + (index // 2) * 16
        add("rect", x=legend_x, y=legend_y - 8, width=8, height=8, fill=fill, stroke=stroke)
        add("text", x=legend_x + 13, y=legend_y, font_size=10.5, fill="#333333",
            data_region_name=region.id).text = region.name
        bounds = _envelope(facing, region)
        if bounds:
            rect = add("rect", x=f"{left + bounds.x * scale:g}", y=f"{top + bounds.y * scale:g}",
                       width=f"{bounds.width * scale:g}", height=f"{bounds.height * scale:g}",
                       fill=fill, fill_opacity=.5, stroke=stroke, stroke_width=1.5,
                       stroke_dasharray="4 3", data_region_id=region.id, data_kind="region")
            ET.SubElement(rect, "title").text = region.name + " — envelope, not a filled grid"
        for slot_id in region.slot_ids:
            slot_colors.setdefault(slot_id, (fill, stroke))

    sections = defaultdict(list)
    for slot in facing.ordered_slots():
        if slot.section_id:
            sections[slot.section_id].append(slot)
        x, y = left + slot.x * scale, top + slot.y * scale
        slot_width, slot_height = slot.width * scale, slot.height * scale
        fill, stroke = slot_colors.get(slot.id, ("#f1f5f9" if slot.role == SlotRole.PRODUCT else "#fafafa", "#526577"))
        rect = add("rect", x=f"{x:g}", y=f"{y:g}", width=f"{slot_width:g}", height=f"{slot_height:g}",
                   fill=fill,
                   stroke=stroke, stroke_width=1, data_slot_id=slot.id, data_role=SlotRole(slot.role).value,
                   data_region_ids=" ".join(region.id for region in facing.regions_for_slot(slot.id)))
        ET.SubElement(rect, "title").text = slot.id
        label = add("text", x=f"{x + slot_width / 2:g}", y=f"{y + slot_height / 2:g}",
                    text_anchor="middle", dominant_baseline="middle", fill="#202020",
                    font_size=f"{min(16, max(10, min(slot_width * .55, slot_height * .45))):g}")
        label.text = str(slot.order)

    for section_id, slots in sorted(sections.items()):
        x0, y0 = min(slot.x for slot in slots), min(slot.y for slot in slots)
        x1 = max(slot.x + slot.width for slot in slots)
        y1 = max(slot.y + slot.height for slot in slots)
        add("rect", x=f"{left + x0 * scale:g}", y=f"{top + y0 * scale:g}",
            width=f"{(x1 - x0) * scale:g}", height=f"{(y1 - y0) * scale:g}",
            fill="none", stroke="#34495e", stroke_width=2, stroke_dasharray="5 3",
            data_section_id=section_id)
    return ET.tostring(root, encoding="unicode")
