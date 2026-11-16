"""Preview ArtiosCAD/Illustrator 8 vector EPS drawings without executing PostScript."""
from dataclasses import dataclass
import io
import math
import re

from reportlab.pdfgen import canvas

from app.sams_club.holiday_price_strips import holiday_geometry_for_size


NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)'
RED = (0.9, 0.1, 0.15)
BLUE = (0.1, 0.35, 0.9)


@dataclass
class EpsDrawing:
    bounds: tuple
    paths: list
    texts: list
    cuts: list


def _ps_text(value):
    def replace(match):
        token = match.group(1)
        if token[0] in '01234567':
            return chr(int(token, 8))
        return {'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'f': '\f'}.get(token, token)
    return re.sub(r'\\([0-7]{1,3}|.)', replace, value)


def parse_eps(data):
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Use an EPS drawing smaller than 10 MB.')
    text = data.decode('latin-1')
    if not text.startswith('%!PS-Adobe-') or 'Adobe_Illustrator' not in text:
        raise ValueError('Upload an ArtiosCAD / Illustrator 8 vector EPS, like the sample template.')
    bbox = re.search(r'^%%BoundingBox:\s*(' + NUMBER + r')\s+(' + NUMBER +
                     r')\s+(' + NUMBER + r')\s+(' + NUMBER + r')', text, re.M)
    if not bbox or '%%EndSetup' not in text or '%AI5_EndLayer' not in text:
        raise ValueError('The EPS is missing its drawing bounds or vector layer.')
    bounds = tuple(map(float, bbox.groups()))
    if (not all(math.isfinite(v) for v in bounds) or
            not 0 < bounds[2] - bounds[0] <= 14400 or
            not 0 < bounds[3] - bounds[1] <= 14400):
        raise ValueError('The EPS drawing bounds are invalid or too large.')
    body = text.split('%%EndSetup', 1)[1].split('%AI5_EndLayer', 1)[0]
    paths, texts, cuts, points = [], [], [], []
    stroke, fill, width, cut = BLUE, BLUE, 1, False
    matrix, font, size = (1, 0, 0, 1, 0, 0), 'Helvetica', 12
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line in ('S', 's'):
            if points:
                paths.append((list(points), stroke, width, cut))
                coords = [(args[0], args[1]) for op, args in points if op in ('m', 'L')]
                if cut and len(points) == 5 and len(coords) == 5 and coords[0] == coords[-1]:
                    xs, ys = sorted({p[0] for p in coords}), sorted({p[1] for p in coords})
                    if len(xs) == len(ys) == 2 and all(
                            a[0] == b[0] or a[1] == b[1] for a, b in zip(coords, coords[1:])):
                        cuts.append((xs[0], ys[0], xs[1], ys[1]))
            points = []
            continue
        match = re.fullmatch(r'\((.*)\) (Tx|TX)', line)
        if match:
            value = _ps_text(match[1])
            if match[2] == 'Tx' and value.strip():
                texts.append((value, matrix, font, size, fill))
            continue
        match = re.fullmatch(r'/_(Arial(?:-Bold)?MT) (' + NUMBER + r') Tf', line)
        if match:
            font = 'Helvetica-Bold' if 'Bold' in match[1] else 'Helvetica'
            size = float(match[2])
            if not 0 < size <= 1000:
                raise ValueError('The EPS contains an unsupported text size.')
            continue
        color = re.fullmatch(r'(' + NUMBER + r')\s+(' + NUMBER + r')\s+(' + NUMBER +
                            r')\s+(' + NUMBER + r')(?:\s+\(([^)]*)\)\s+0)?\s+(X|k)', line)
        if color:
            c, m, y, k = map(float, color.groups()[:4])
            rgb = tuple(max(0, min(1, (1 - v) * (1 - k))) for v in (c, m, y))
            if color[6] == 'k':
                fill = rgb
            else:
                cut = color[5] == 'Artios_Cut'
                stroke = RED if cut else rgb
            continue
        tokens = line.split()
        op = tokens[-1]
        try:
            args = tuple(float(t) for t in tokens[:-1])
        except ValueError:
            args = ()
        if op in ('m', 'L', 'c') and len(args) == (6 if op == 'c' else 2):
            if not all(math.isfinite(v) and abs(v) <= 100000 for v in args):
                raise ValueError('The EPS contains invalid vector coordinates.')
            points.append((op, args))
        elif op == 'w' and len(args) == 1:
            width = max(.1, min(20, args[0]))
        elif op == 'Tp' and len(args) == 7:
            if not all(math.isfinite(v) and abs(v) <= 100000 for v in args):
                raise ValueError('The EPS contains an invalid text transform.')
            matrix = args[:6]
        elif (line in ('TP', 'TO', 'LB', '0 To', '0 Tr', '0 Ta', '100.000 Tz', '0.000 Tt')
              or re.fullmatch(r'[012] [Jj]', line) or line == '[]0 d'
              or op == 'Lb' or re.fullmatch(r'\([^)]*\) Ln', line)):
            continue
        else:
            raise ValueError('This EPS uses unsupported artwork. Export it as an ArtiosCAD / Illustrator 8 vector EPS.')
    if not cuts:
        raise ValueError('No rectangular Artios_Cut strip outlines were found in this EPS.')
    return EpsDrawing(bounds, paths, texts, cuts)


def render_eps_preview(data, strip_size, print_pdf, *, guides_in_pdf=False):
    import pymupdf

    drawing = parse_eps(data)
    geometry = holiday_geometry_for_size(strip_size)
    matches = [r for r in drawing.cuts if abs(r[2] - r[0] - geometry.width_pt) < .05
               and abs(r[3] - r[1] - geometry.height_pt) < .05]
    if not matches:
        raise ValueError('This EPS has no strip outline matching the selected dimensions. Choose the other size or upload its EPS.')
    left, bottom, right, top = drawing.bounds
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(right - left, top - bottom))
    pdf.translate(-left, -bottom)
    for commands, color, width, _ in drawing.paths:
        path = pdf.beginPath()
        for op, args in commands:
            {'m': path.moveTo, 'L': path.lineTo, 'c': path.curveTo}[op](*args)
        pdf.setStrokeColorRGB(*color)
        pdf.setLineWidth(width)
        pdf.drawPath(path, stroke=1, fill=0)
    for value, matrix, font, size, color in drawing.texts:
        pdf.saveState()
        pdf.transform(*matrix)
        pdf.setFillColorRGB(*color)
        pdf.setFont(font, size)
        pdf.drawString(0, 0, value)
        pdf.restoreState()
    pdf.showPage()
    pdf.save()
    with pymupdf.open(stream=output.getvalue(), filetype='pdf') as doc, \
            pymupdf.open(stream=print_pdf, filetype='pdf') as source:
        page = doc[0]
        # The print renderer adds 1/4-inch crop margins around the finished strip.
        clip = pymupdf.Rect(18, 18, 18 + geometry.width_pt, 18 + geometry.height_pt)
        if len(source) != 1 or not source[0].rect.contains(clip):
            raise ValueError('The print preview does not match the selected strip size.')
        zoom = None
        for x0, y0, x1, y1 in matches:
            target = pymupdf.Rect(x0 - left, top - y1, x1 - left, top - y0)
            page.show_pdf_page(target, source, 0, clip=clip, keep_proportion=False)
            if not guides_in_pdf:
                page.draw_rect(target, color=RED, width=2)
                for column in range(1, geometry.slot_count):
                    x = target.x0 + target.width * column / geometry.slot_count
                    page.draw_line((x, target.y0), (x, target.y1), color=BLUE, width=1)
            zoom = target
        scale = min(2, 1800 / page.rect.width)
        overview = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes('png')
        # Use the renderer's full strip page for the reference guide style,
        # including crop marks, numbered positions, dashed centers and crosshairs.
        if guides_in_pdf:
            scale = min(2, 1800 / source[0].rect.width)
            detail = source[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes('png')
        else:
            scale = min(2, 1800 / zoom.width)
            detail = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=zoom).tobytes('png')
        return overview, detail
