"""Blank SVG geometry QA; SVG display coordinates never become fixture units."""
from collections import defaultdict
from xml.etree import ElementTree as ET

from app.templates.models import DisplayTemplate, SlotRole
from app.templates.validation import validate_template


def geometry_svg(template: DisplayTemplate, facing_id: str) -> str:
    validation = validate_template(template)
    if not validation.valid:
        raise ValueError("Cannot preview invalid geometry: " + "; ".join(issue.message for issue in validation.errors))
    facing = next((facing for facing in template.facings if facing.id == facing_id), None)
    if facing is None:
        raise KeyError(f"Unknown facing: {facing_id}")

    # One scale for both axes: preserve the declared geometry, never stretch it.
    scale = min(800 / facing.width, 600 / facing.height)
    width, height = facing.width * scale, facing.height * scale
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 {width + 80:g} {height + 110:g}",
        "role": "img", "aria-label": f"{template.name}: {facing.name}, blank geometry preview",
    })

    def add(tag, **attributes):
        return ET.SubElement(root, tag, {key.replace("_", "-"): str(value) for key, value in attributes.items()})

    add("title").text = f"{template.name} — {facing.name}"
    add("text", x=40, y=26, font_size=17, fill="#202020").text = facing.name
    add("text", x=40, y=46, font_size=12, fill="#555555").text = (
        "Uncalibrated structure — not to physical scale"
        if template.unit == "normalized" else "Inches — scaled to fit this preview"
    )
    add("rect", x=40, y=70, width=width, height=height,
        fill="#ffffff", stroke="#222222", stroke_width=2, data_kind="facing")

    sections = defaultdict(list)
    for slot in facing.ordered_slots():
        if slot.section_id:
            sections[slot.section_id].append(slot)
        x, y = 40 + slot.x * scale, 70 + slot.y * scale
        slot_width, slot_height = slot.width * scale, slot.height * scale
        rect = add("rect", x=f"{x:g}", y=f"{y:g}", width=f"{slot_width:g}", height=f"{slot_height:g}",
                   fill="#f1f5f9" if slot.role == SlotRole.PRODUCT else "#fafafa",
                   stroke="#526577", stroke_width=1, data_slot_id=slot.id, data_role=SlotRole(slot.role).value)
        ET.SubElement(rect, "title").text = slot.id
        label = add("text", x=f"{x + slot_width / 2:g}", y=f"{y + slot_height / 2:g}",
                    text_anchor="middle", dominant_baseline="middle", fill="#202020",
                    font_size=f"{min(18, slot_width * 0.20, slot_height * 0.30):g}")
        label.text = str(slot.order)

    for section_id, slots in sorted(sections.items()):
        x0, y0 = min(slot.x for slot in slots), min(slot.y for slot in slots)
        x1 = max(slot.x + slot.width for slot in slots)
        y1 = max(slot.y + slot.height for slot in slots)
        add("rect", x=f"{40 + x0 * scale:g}", y=f"{70 + y0 * scale:g}",
            width=f"{(x1 - x0) * scale:g}", height=f"{(y1 - y0) * scale:g}",
            fill="none", stroke="#34495e", stroke_width=2, stroke_dasharray="5 3",
            data_section_id=section_id)
    return ET.tostring(root, encoding="unicode")
