"""Printable blank template layouts; independent of product/PDF workflows."""
from __future__ import annotations

import io
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame,
                               NextPageTemplate, PageBreak, PageTemplate)

from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT
from app.templates.models import DisplayTemplate
from app.templates.print_style import DEFAULT_TEXT_FIELDS, normalize_text_fields
from app.templates.square_layout import square_page_layout
from app.templates.validation import validate_template


_INK = colors.HexColor('#243247')
PDF_PAGE_SIZE = (14 * inch, 22 * inch)


def _side_identifier(facing):
    """Keep Side letters/numbers and any Copy suffix visible in print."""
    copy = re.search(r'\bcopy\b.*$', facing.name, re.IGNORECASE)
    name = facing.name[:copy.start()].rstrip(' -:') if copy else facing.name
    match = re.match(r'^\s*side\s*#?\s*([a-z]{1,2}|\d+)(?=\s|[-:]|$)', name, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r'\s*([a-z]{1,2}|\d+)\s*', name, re.IGNORECASE)
    label = 'SIDE ' + (match.group(1).upper() if match else str(facing.order))
    return label + (' ' + ' '.join(copy.group().upper().split()) if copy else '')


def _fit_header(text, font, size, width, min_size=9):
    while size > min_size and pdfmetrics.stringWidth(text, font, size) > width:
        size = max(min_size, size - .5)
    if pdfmetrics.stringWidth(text, font, size) > width:
        if pdfmetrics.stringWidth('...', font, size) > width:
            return '', size
        while text and pdfmetrics.stringWidth(text + '...', font, size) > width:
            text = text[:-1]
        text += '...'
    return text, size


def _wrap_description(text, size, first_width, full_width, max_lines=3):
    """Start beside the label, then wrap beneath it at the full box width."""
    remaining = ' '.join(text.split())
    lines = []
    for index in range(max_lines):
        width = max(0, first_width if index == 0 else full_width)
        if index == max_lines - 1:
            lines.append(_fit_header(remaining, BODY_FONT, size, width, min_size=size)[0])
            break
        if pdfmetrics.stringWidth(remaining, BODY_FONT, size) <= width:
            lines.append(remaining)
            break
        low, high = 0, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if pdfmetrics.stringWidth(remaining[:middle], BODY_FONT, size) <= width:
                low = middle
            else:
                high = middle - 1
        # Prefer whole words; split an oversized word without losing its tail.
        boundary = remaining.rfind(' ', 0, low + 1)
        cut = boundary if boundary > 0 else low
        lines.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return lines


def _band_text_color(fill):
    channels = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
                for value in (fill.red, fill.green, fill.blue)]
    luminance = sum(value * weight for value, weight in zip(channels, (.2126, .7152, .0722)))
    return colors.black if luminance > .179 else colors.white


class _SideDrawing(Flowable):
    def __init__(self, facing, width, max_height, text_fields=DEFAULT_TEXT_FIELDS):
        super().__init__()
        self.facing = facing
        self.text_fields = text_fields
        self.width = width
        self.height = max_height
        self.cards, self.bands = square_page_layout(facing, width, max_height)

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        for name, color, band_top, band_height in self.bands:
            size = min(10, band_height * .6)
            label, size = _fit_header(name, BODY_BOLD_FONT, size,
                                      self.width - 12, min_size=min(5, size))
            fill = colors.HexColor(color)
            inset = min(2, band_height * .1)
            canvas.setFillColor(fill)
            canvas.rect(0, band_top - band_height + inset,
                        self.width, band_height - 2 * inset, fill=1, stroke=0)
            canvas.setFillColor(_band_text_color(fill))
            canvas.setFont(BODY_BOLD_FONT, size)
            canvas.drawCentredString(self.width / 2, band_top - .75 * band_height, label)
        for slot, (x, y, width, height) in self.cards:
            # Plain Box outline
            canvas.setStrokeColor(colors.HexColor('#c8cdd4'))
            canvas.setLineWidth(.6)
            canvas.rect(x, y, width, height, fill=0, stroke=1)

            # Padding scaled down for smaller boxes
            padding = min(6, width * .06, height * .06)
            inner_width = width - 2 * padding
            inner_height = height - 2 * padding

            # Compact text area at the top of each box
            top = y + height - padding
            number = str(slot.order)

            number_size = min(
                18,
                inner_height * .10,
                inner_width * .18
                / pdfmetrics.stringWidth(number, BODY_BOLD_FONT, 1),
            )

            number_width = pdfmetrics.stringWidth(
                number, BODY_BOLD_FONT, number_size
            )
            gap = min(4, inner_width * .03)
            text_width = inner_width - number_width - gap

            labels = tuple(f'{field}:' for field in self.text_fields)
            widest_label = max((pdfmetrics.stringWidth(label, BODY_FONT, 1) for label in labels),
                               default=pdfmetrics.stringWidth('Description:', BODY_FONT, 1))
            field_size = min(
                14,
                inner_height * .075,
                text_width * .55
                / widest_label
            )

            # Smaller multiplier = tighter spacing between text lines.
            row_spacing = field_size * 1.35
            first_baseline = top - max(number_size, field_size)

            canvas.setFillColor(_INK)

            # Box number in the upper-right corner.
            canvas.setFont(BODY_BOLD_FONT, number_size)
            canvas.drawRightString(
                x + width - padding,
                first_baseline,
                number,
            )


            # Only the selected writing lines take up space above the image.
            canvas.setFont(BODY_FONT, field_size)
            canvas.setLineWidth(min(.4, field_size * .08))

            baseline = first_baseline
            last_baseline = first_baseline

            for field, label in zip(self.text_fields, labels):
                line_size = field_size * (
                    1.25 if field in ("CPP", "UPC", "ID#", "Item #") else 1
                )
                baseline -= max(0, line_size - field_size)
                text_x = x + padding
                canvas.setFont(BODY_FONT, line_size)
                canvas.drawString(text_x, baseline, label)

                label_width = pdfmetrics.stringWidth(
                    label, BODY_FONT, line_size
                )
                line_start = text_x + label_width + line_size * .4
                line_end = text_x + text_width
                available_width = max(0, line_end - line_start)
                value = slot.metadata.get(f"pdf_text:{field}", "")

                if field == "Description" and value and available_width > 0:
                    lines = _wrap_description(
                        value, line_size, available_width, inner_width
                    )
                else:
                    lines = [value]
                
                for index, line in enumerate(lines):
                    if field == "Description" and index > 0:
                        line_start = text_x
                        line_end = text_x + inner_width
                        available_width = inner_width
                    printed, size = _fit_header(
                        line,
                        BODY_FONT,
                        line_size,
                        available_width,
                        min_size=line_size,
                    )

                    canvas.setFont(BODY_FONT, size)
                    canvas.drawString(line_start, baseline, printed)
                    canvas.line(
                        line_start,
                        baseline - field_size * .12,
                        line_end,
                        baseline - field_size * .12,
                    )

                    last_baseline = baseline
                    baseline -= row_spacing

                #Square image placeholder below the text


            image_gap = min(6, inner_height * .04)

            # Leave clearance below the last selected line, or the box number.
            image_top = last_baseline - (line_size * .25 if labels else 0) - image_gap
            image_bottom = y + padding
            available_height = max(0, image_top - image_bottom)
            square_size = min(inner_width, available_height)

            if square_size > 0:
                image_x = x + (width - square_size) / 2
                image_y = image_bottom + (available_height - square_size) / 2

                canvas.setStrokeColor(colors.HexColor("#c8cdd4"))
                canvas.setLineWidth(.6)
                canvas.rect(
                    image_x,
                    image_y,
                    square_size,
                    square_size,
                    fill=0,
                    stroke=1,
                )

        canvas.restoreState()


def render_template_pdf(template: DisplayTemplate, *, created_on: date | None = None,
                        text_fields=None, use_facing_names=False, layout_note=None) -> bytes:
    """Print each Side with square boxes fitted to its existing canvas rows."""
    text_fields = normalize_text_fields(text_fields)
    validation = validate_template(template)
    if not validation.valid:
        raise ValueError('Resolve template layout checks before creating the PDF.')

    output = io.BytesIO()
    doc = BaseDocTemplate(output, pagesize=PDF_PAGE_SIZE, leftMargin=6, rightMargin=6,
                          topMargin=64 if layout_note else 44, bottomMargin=28, title=template.name,
                          author='Kendal King')
    created_on = created_on or datetime.now(ZoneInfo('America/Chicago')).date()

    def page_header(canvas, document, facing):
        canvas.saveState()
        page_width, page_height = document.pagesize
        identifier, right_size = _fit_header(facing.name if use_facing_names else _side_identifier(facing),
                                            BODY_BOLD_FONT, 16, doc.width * .45)
        right_width = pdfmetrics.stringWidth(identifier, BODY_BOLD_FONT, right_size)
        label, size = _fit_header(template.name, BODY_BOLD_FONT, 16, doc.width - right_width - 24)
        canvas.setFillColor(_INK)
        canvas.setFont(BODY_BOLD_FONT, size)
        canvas.drawString(doc.leftMargin, page_height - 28, label)
        canvas.setFont(BODY_BOLD_FONT, right_size)
        canvas.drawRightString(page_width - doc.rightMargin, page_height - 28, identifier)
        canvas.setStrokeColor(colors.HexColor('#d7dee8'))
        canvas.line(doc.leftMargin, page_height - 40, page_width - doc.rightMargin, page_height - 40)
        if layout_note:
            note, note_size = _fit_header(layout_note, BODY_FONT, 8, doc.width, min_size=6)
            canvas.setFont(BODY_FONT, note_size)
            canvas.drawString(doc.leftMargin, page_height - 53, note)
        canvas.line(doc.leftMargin, 24, page_width - doc.rightMargin, 24)
        canvas.setFont(BODY_FONT, 8)
        canvas.drawString(doc.leftMargin, 12, f'Date: {created_on.isoformat()}')
        canvas.drawCentredString(page_width / 2, 12, 'Generated By Kendal King')
        canvas.drawRightString(page_width - doc.rightMargin, 12, f'Page {document.page}')
        canvas.restoreState()

    story = []
    for index, facing in enumerate(template.ordered_facings()):
        page_id = f'side_page_{index}'
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id=page_id, frames=[frame],
                            onPage=lambda canvas, document, side=facing: page_header(canvas, document, side)))
        if index:
            story.extend([NextPageTemplate(page_id), PageBreak()])
        story.append(_SideDrawing(facing, doc.width, doc.height, text_fields))
    doc.build(story)
    return output.getvalue()
