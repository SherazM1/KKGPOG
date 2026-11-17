"""Centered print layouts retain the holes and offsets edited on the canvas."""
from dataclasses import replace
import unittest

import pymupdf

from app.templates import draft as edits
from app.templates.render_pdf import render_template_pdf


def edited_grid():
    draft = edits.new_draft('Canvas positions', 'Custom', '1', 'normalized')
    draft = edits.add_facing(draft, 'Side A', 12, 12)
    fid = draft.template.facings[0].id
    draft = edits.add_region(draft, fid, 'Cards')
    rid = draft.template.facings[0].regions[0].id
    for row in range(4):
        for column in range(8):
            draft = edits.add_slot(draft, fid, rid, x=1+column*1.2, y=1.5+row*1.2,
                                   width=1, height=1)
    face = draft.template.facings[0]
    removed = [slot.id for slot in face.slots if slot.order in (*range(17, 22), *range(25, 30))]
    return edits.remove_slots(draft, fid, removed)


def printed_boxes(page):
    rectangles = [d['rect'] for d in page.get_drawings() if d['fill'] is None
                  and any(item[0] == 're' for item in d['items'])][::2]
    boxes = {}
    for rectangle in rectangles:
        number = next(int(value) for value in page.get_text(clip=rectangle).splitlines() if value.isdigit())
        boxes[number] = rectangle
    return boxes


class PdfCanvasPositionTests(unittest.TestCase):
    def test_deleted_bottom_left_ten_leave_the_survivors_in_the_right_columns(self):
        draft = edited_grid()
        before = draft.template
        with pymupdf.open(stream=render_template_pdf(before)) as document:
            page = document[0]
            boxes = printed_boxes(page)
            self.assertEqual(len(boxes), 22)
            self.assertNotIn(17, boxes)
            self.assertNotIn(25, boxes)
            for lower, upper in ((22, 6), (23, 7), (24, 8), (30, 6), (31, 7), (32, 8)):
                self.assertAlmostEqual(boxes[lower].x0, boxes[upper].x0, delta=.01)
                self.assertAlmostEqual(boxes[lower].width, boxes[upper].width, delta=.01)
            self.assertAlmostEqual((boxes[1].x0+boxes[8].x1)/2, page.rect.width/2, delta=.01)
            self.assertTrue(all(abs(box.width-box.height) < .01 for box in boxes.values()))
        self.assertEqual(draft.template, before)

    def test_moving_top_row_up_keeps_the_extra_gap_without_shifting_its_columns(self):
        draft = edited_grid()
        face = draft.template.facings[0]
        face = replace(face, slots=tuple(replace(slot, y=slot.y-1) if slot.order <= 8 else slot
                                         for slot in face.slots))
        template = replace(draft.template, facings=(face,))
        with pymupdf.open(stream=render_template_pdf(template)) as document:
            boxes = printed_boxes(document[0])
            size = boxes[1].width
            first_gap = boxes[9].y0-boxes[1].y1
            ordinary_gap = boxes[22].y0-boxes[14].y1
            self.assertAlmostEqual(first_gap/size, 1.2, places=3)
            self.assertAlmostEqual(ordinary_gap/size, .2, places=3)
            self.assertAlmostEqual(boxes[1].x0, boxes[9].x0, delta=.01)
            self.assertAlmostEqual(boxes[6].x0, boxes[22].x0, delta=.01)
            rectangles = list(boxes.values())
            self.assertFalse(any(a.intersects(b) for i, a in enumerate(rectangles) for b in rectangles[i+1:]))
