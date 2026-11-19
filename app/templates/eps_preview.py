"""Read vector EPS/PDF card drawings and preview alignment without executing PostScript."""
from dataclasses import dataclass
import io
import math
import re

from reportlab.pdfgen import canvas

from app.sams_club.generic_price_strips import geometry_for_size as holiday_geometry_for_size
from app.sams_club.generic_price_strips import GENERIC_SIZES, generic_boundaries


NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)'
RED = (0.9, 0.1, 0.15)
BLUE = (0.1, 0.35, 0.9)


@dataclass
class EpsDrawing:
    bounds: tuple
    paths: list
    texts: list
    cuts: list
    preview_pdf: bytes | None = None
    card_outlines: tuple = ()


def parse_card_spacing(data):
    """Read card-center vectors from either supported upload format."""
    if data.lstrip().startswith(b'%PDF-'):
        return _parse_pdf_spacing(data)
    return parse_eps(data)


def _parse_pdf_spacing(data):
    import pymupdf

    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Use a Card Spacing PDF smaller than 10 MB.')
    try:
        document = pymupdf.open(stream=data, filetype='pdf')
    except Exception as exc:
        raise ValueError('The Card Spacing PDF could not be opened. Export a new vector PDF.') from exc
    with document, pymupdf.open() as combined:
        if document.needs_pass:
            raise ValueError('Use a Card Spacing PDF without password protection.')
        if not 1 <= len(document) <= 50:
            raise ValueError('Use a Card Spacing PDF with 1 to 50 pages.')
        width = max(page.rect.width for page in document)
        height = sum(page.rect.height for page in document) + 36 * (len(document)-1)
        if not 0 < width <= 14400 or not 0 < height <= 14400:
            raise ValueError('The Card Spacing PDF drawing is too large. Upload fewer pages.')
        preview = combined.new_page(width=width, height=height)
        paths, card_outlines, offset = [], [], 0
        for page in document:
            # Use a shared, bottom-left coordinate system, including rotated
            # pages, so EPS geometry and crosshair validation can be reused.
            def point(value):
                value = value * page.rotation_matrix
                return value.x, height-offset-value.y
            for item in page.get_drawings():
                color = item.get('color')
                if color is None or item.get('stroke_opacity',1) <= 0:
                    continue
                commands, subpath_start, endpoint = [], None, None
                for segment in item['items']:
                    op = segment[0]
                    if op == 'l':
                        start,end = point(segment[1]),point(segment[2])
                        if start != endpoint:
                            subpath_start = start
                        commands.extend([('m',start),('L',end)])
                        endpoint = end
                    elif op == 're':
                        rect = segment[1]
                        corners = [rect.tl,rect.tr,rect.br,rect.bl,rect.tl]
                        commands.extend(('m' if i==0 else 'L',point(p)) for i,p in enumerate(corners))
                        subpath_start = endpoint = None
                    elif op == 'qu':
                        quad = segment[1]
                        corners = [quad.ul,quad.ur,quad.lr,quad.ll,quad.ul]
                        commands.extend(('m' if i==0 else 'L',point(p)) for i,p in enumerate(corners))
                        subpath_start = endpoint = None
                    elif op == 'c':
                        if point(segment[1]) != endpoint:
                            subpath_start = point(segment[1])
                        commands.extend([('m',point(segment[1])),
                                         ('c',tuple(v for p in segment[2:] for v in point(p)))])
                        endpoint = point(segment[-1])
                if item.get('closePath') and subpath_start is not None and endpoint != subpath_start:
                    commands.extend([('m',endpoint),('L',subpath_start)])
                if commands:
                    paths.append((commands,tuple(color),item.get('width') or 1,False))
                    # Rounded cards use curves rather than four straight
                    # edges. Their vector bounds still give the true width.
                    first = commands[0][1]
                    closed = endpoint is not None and all(abs(a-b)<.01 for a,b in zip(first,endpoint))
                    if any(op=='c' for op,_ in commands) and (closed or item.get('closePath')):
                        corners = [point(p) for p in (item['rect'].tl,item['rect'].tr,item['rect'].br,item['rect'].bl)]
                        card_outlines.append((min(x for x,y in corners),min(y for x,y in corners),
                                              max(x for x,y in corners),max(y for x,y in corners)))
            if page.get_contents():
                preview.show_pdf_page(pymupdf.Rect(0,offset,page.rect.width,offset+page.rect.height),document,page.number)
            offset += page.rect.height+36
        cuts = _axis_rectangles(paths)
        # A printed border may be inset slightly from the physical trim edge.
        # Recover that edge without scaling or moving any crosshairs.
        for rectangle in tuple(cuts):
            left,bottom,right,top = rectangle
            for size in ('thin','wide',*GENERIC_SIZES):
                geometry = holiday_geometry_for_size(size)
                dw,dh = geometry.width_pt-(right-left),geometry.height_pt-(top-bottom)
                if -.05 <= min(dw,dh) and .05 < max(dw,dh) <= 3:
                    edge = (left-dw/2,bottom-dh/2,right+dw/2,top+dh/2)
                    if edge[0]>=-.05 and edge[1]>=-.05 and edge[2]<=width+.05 and edge[3]<=height+.05:
                        cuts.append(tuple(round(v,3) for v in edge))
        drawing = EpsDrawing((0,0,width,height),paths,[],cuts,combined.tobytes(),tuple(card_outlines))
    if not green_crosshairs(drawing):
        raise ValueError('The Card Spacing PDF needs green vector crosshairs. Scanned or image-only PDFs cannot provide card centers.')
    if not drawing.cuts:
        raise ValueError('The Card Spacing PDF needs rectangular strip outlines at the actual print dimensions.')
    return drawing


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
    if text.startswith('%!PS-Adobe-') and 'EPSF-' in text.splitlines()[0] and 'Adobe_Illustrator' not in text:
        return _parse_simple_eps(text)
    if not text.startswith('%!PS-Adobe-') or 'Adobe_Illustrator' not in text:
        raise ValueError('Upload an ArtiosCAD / Illustrator 8 vector EPS, like the sample template.')
    bbox = re.search(r'^%%BoundingBox:\s*(' + NUMBER + r')\s+(' + NUMBER +
                     r')\s+(' + NUMBER + r')\s+(' + NUMBER + r')', text, re.M)
    if not bbox or '%%EndSetup' not in text or not any(
            marker in text for marker in ('%%PageTrailer', '%AI5_EndLayer')):
        raise ValueError('The EPS is missing its drawing bounds or vector layer.')
    bounds = tuple(map(float, bbox.groups()))
    if (not all(math.isfinite(v) for v in bounds) or
            not 0 < bounds[2] - bounds[0] <= 14400 or
            not 0 < bounds[3] - bounds[1] <= 14400):
        raise ValueError('The EPS drawing bounds are invalid or too large.')
    body = text.split('%%EndSetup', 1)[1].split('%%PageTrailer', 1)[0].split('%AI5_EndLayer', 1)[0]
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
    # ArtiosCAD can share edges rather than closing each rectangle separately.
    cuts.extend(_axis_rectangles(paths, only_cuts=True))
    cuts = list(dict.fromkeys(tuple(round(v, 3) for v in r) for r in cuts))
    if not cuts:
        raise ValueError('No rectangular Artios_Cut strip outlines were found in this EPS.')
    return EpsDrawing(bounds, paths, texts, cuts)


def _parse_simple_eps(text):
    """Read a small declarative EPS subset; never evaluate PostScript code."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    bbox = re.search(r'^%%BoundingBox:\s*(' + NUMBER + r')\s+(' + NUMBER +
                     r')\s+(' + NUMBER + r')\s+(' + NUMBER + r')',text,re.M)
    if not bbox or '%%EndComments' not in text:
        raise ValueError('The vector EPS is missing its drawing bounds.')
    bounds = tuple(map(float,bbox.groups()))
    if (not all(math.isfinite(v) for v in bounds) or
            not 0 < bounds[2]-bounds[0] <= 14400 or not 0 < bounds[3]-bounds[1] <= 14400):
        raise ValueError('The EPS drawing bounds are invalid or too large.')
    paths,texts,color,width = [],[],(0,0,0),1
    for line in text.split('%%EndComments',1)[1].splitlines():
        line = line.strip()
        if not line or line.startswith('%') or line in ('gsave','grestore','showpage'):
            continue
        match = re.fullmatch(r'('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r') setrgbcolor',line)
        if match:
            color = tuple(map(float,match.groups()))
            if any(not 0 <= v <= 1 for v in color):
                raise ValueError('The EPS contains an invalid color.')
            continue
        match = re.fullmatch(r'('+NUMBER+r') setlinewidth',line)
        if match:
            width = max(.1,min(20,float(match[1])))
            continue
        match = re.fullmatch(
            r'/(Helvetica(?:-Bold)?) findfont ('+NUMBER+r') scalefont setfont '
            r'('+NUMBER+r') ('+NUMBER+r') moveto \((.*)\) '
            r'dup stringwidth pop 2 div neg 0 rmoveto show',line)
        if match:
            font,size,x,y,value = match.groups()
            size,x,y = map(float,(size,x,y))
            if not 0 < size <= 1000 or not all(math.isfinite(v) and abs(v)<=100000 for v in (x,y)):
                raise ValueError('The EPS contains invalid text coordinates or size.')
            value = _ps_text(value)
            texts.append((value,(1,0,0,1,x-stringWidth(value,font,size)/2,y),font,size,color))
            continue
        if line.startswith('newpath ') and line.split()[-1] in ('stroke','fill'):
            commands,stack,start = [],[],None
            tokens = line.split()[1:]
            for token in tokens[:-1]:
                if re.fullmatch(NUMBER,token):
                    value = float(token)
                    if not math.isfinite(value) or abs(value)>100000:
                        raise ValueError('The EPS contains invalid vector coordinates.')
                    stack.append(value)
                elif token in ('moveto','lineto') and len(stack)==2:
                    point = tuple(stack)
                    stack.clear()
                    if token=='moveto':
                        start = point
                    commands.append(('m' if token=='moveto' else 'L',point))
                elif token=='closepath' and start is not None and not stack:
                    commands.append(('L',start))
                else:
                    raise ValueError('This EPS uses unsupported vector commands.')
            if stack or not commands or commands[0][0]!='m':
                raise ValueError('The EPS contains an incomplete vector path.')
            if tokens[-1]=='stroke':
                paths.append((commands,color,width,False))
            continue
        raise ValueError('This EPS uses unsupported artwork. Use a vector PDF or an ArtiosCAD / Illustrator 8 EPS.')
    cuts = _axis_rectangles(paths)
    if not cuts:
        raise ValueError('No rectangular strip outlines were found in this EPS.')
    return EpsDrawing(bounds,paths,texts,cuts)


def _axis_rectangles(paths, only_cuts=False):
    horizontal, vertical = {}, {}
    for commands, _, _, is_cut in paths:
        if only_cuts and not is_cut:
            continue
        previous = None
        for op, args in commands:
            point = tuple(round(v, 3) for v in args) if op in ('m', 'L') else None
            if op == 'L' and previous is not None:
                x0, y0 = previous
                x1, y1 = point
                if abs(y1 - y0) < .01 and x0 != x1:
                    horizontal.setdefault(tuple(sorted((x0, x1))), set()).add(y0)
                if abs(x1 - x0) < .01 and y0 != y1:
                    vertical.setdefault(x0, []).append(tuple(sorted((y0, y1))))
            previous = point
    def covered(x, bottom, top):
        end = bottom
        intervals = [
            interval
            for edge_x, ranges in vertical.items()
            if abs(edge_x - x) <= .01
            for interval in ranges
        ]
        
        for low, high in sorted(intervals):
            if low > end + .01:
                continue
            if high > end:
                end = high
            if end >= top - .01:
                return True
        return False
    rectangles = []
    for (left, right), levels in horizontal.items():
        levels = sorted(levels)
        for bottom, top in zip(levels, levels[1:]):
            if covered(left, bottom, top) and covered(right, bottom, top):
                rectangles.append((left, bottom, right, top))
    return rectangles


def green_crosshairs(drawing):
    """Recognize green plus marks by intersecting their two short strokes."""
    horizontal, vertical = [], []
    for commands, (red, green, blue), _, cut in drawing.paths:
        if cut or green < .15 or green < max(red, blue) + .1:
            continue
        previous = None
        for op, point in commands:
            if op == 'L' and previous is not None:
                x0, y0 = previous
                x1, y1 = point
                if abs(y1-y0) < .05 and .5 <= abs(x1-x0) <= 72:
                    horizontal.append((min(x0,x1), max(x0,x1), (y0+y1)/2))
                elif abs(x1-x0) < .05 and .5 <= abs(y1-y0) <= 72:
                    vertical.append(((x0+x1)/2, min(y0,y1), max(y0,y1)))
            previous = point if op in ('m', 'L') else None
    centers = set()
    for left, right, y in horizontal:
        for x, bottom, top in vertical:
            if (left+.05 < x < right-.05 and bottom+.05 < y < top-.05
                    and abs(x-(left+right)/2) < .25 
                    and abs(y-(bottom+top)/2) < .25):
                centers.add((round(x,3), round(y,3)))
    return tuple(sorted(centers))


def _relative_crosshairs(drawing, rectangle):
    left, bottom, right, top = rectangle
    return tuple((x-left, top-y) for x,y in green_crosshairs(drawing)
                 if left < x < right and bottom < y < top)


def matching_strip_rectangle(drawing, strip_size, row_number=1):
    geometry = holiday_geometry_for_size(strip_size)
    matches = sorted((r for r in drawing.cuts
                      if abs(r[2]-r[0]-geometry.width_pt) < .05
                      and abs(r[3]-r[1]-geometry.height_pt) < .05), key=lambda r:(-r[3],r[0]))
    if not matches:
        raise ValueError(f'This drawing has no strip outline matching the selected dimensions '
                         f'({geometry.width_in:g} x {geometry.height_in:g} inches, {geometry.slot_count} positions). '
                         'Upload an EPS or PDF at the actual print dimensions for this strip size.')
    if len(matches) == 1:
        return matches[0]
    if 1 <= row_number <= len(matches):
        return matches[row_number-1]
    # Repeated engineering outlines with identical marks are reusable for
    # additional spreadsheet rows; differing layouts require an explicit row.
    first = _relative_crosshairs(drawing, matches[0])
    if first and all(len(other) == len(first) and all(
            abs(a-c) < .05 and abs(b-d) < .05 for (a,b),(c,d) in zip(first,other))
            for other in (_relative_crosshairs(drawing,r) for r in matches[1:])):
        return matches[0]
    raise ValueError(f'The drawing has {len(matches)} different strip rows of this size; '
                     f'cannot locate the alignment for spreadsheet Row {row_number}.')


def strip_crosshair_layout(drawing, strip_size, row_number=1):
    """Return card-center coordinates relative to the strip's top-left corner."""
    rectangle = matching_strip_rectangle(drawing, strip_size, row_number)
    centers = _relative_crosshairs(drawing, rectangle)
    expected = holiday_geometry_for_size(strip_size).slot_count
    if len(centers) != expected:
        raise ValueError(f'This strip needs {expected} green crosshairs, one per product '
                         f'position, but {len(centers)} were found. Check the uploaded drawing.')
    if any(right[0]-left[0] < .1 for left,right in zip(centers,centers[1:])):
        raise ValueError('The drawing has multiple card crosshairs in the same horizontal position; '
                         'product order cannot be determined.')
    return centers


def widest_eps_card_width(drawing):
    """Use the widest marked card outline, or the widest pitch if none exist."""
    centers = green_crosshairs(drawing)
    widths = []
    for left,bottom,right,top in (*_axis_rectangles(drawing.paths),*drawing.card_outlines):
        if min(right-left,top-bottom) < 36:
            continue
        contained = [(x,y) for x,y in centers if left-.25 <= x <= right+.25 and bottom-.25 <= y <= top+.25]
        if len(contained) == 1:
            x,y = contained[0]
            if abs(x-(left+right)/2) < .25 and min(abs(y-mid) for mid in (bottom,(bottom+top)/2,top)) < .25:
                widths.append(right-left)
    if widths:
        return max(widths), 'card outlines'
    frames = [(r,sorted({x for x,y in centers if r[0] < x < r[2] and r[1] < y < r[3]}))
              for r in drawing.cuts]
    frames = [(r,xs) for r,xs in frames if len(xs)>1]
    pitches = []
    for r,xs in frames:
        if any(other != r and r[0] <= other[0] and r[1] <= other[1]
               and r[2] >= other[2] and r[3] >= other[3] for other,_ in frames):
            continue
        pitches.extend(b-a for a,b in zip(xs,xs[1:]))
    if not pitches:
        raise ValueError('The drawing needs marked card outlines or multiple crosshairs to determine card width.')
    return max(pitches), 'crosshair spacing'


def card_spacing_overflow(drawing, strip_size, row_number=1):
    """Return physical card overhang at each end; pitch alone is not card width."""
    width,source = widest_eps_card_width(drawing)
    if source != 'card outlines':
        return 0,0
    centers = strip_crosshair_layout(drawing,strip_size,row_number)
    strip_width = holiday_geometry_for_size(strip_size).width_pt
    return max(0,width/2-centers[0][0]),max(0,centers[-1][0]+width/2-strip_width)


def _drawing_pdf_bytes(drawing):
    if drawing.preview_pdf is not None:
        return drawing.preview_pdf
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
    return output.getvalue()


def combine_card_spacing(uploads):
    """Combine complementary EPS/PDF layouts without changing their scale."""
    import pymupdf

    if sum(len(upload['bytes']) for upload in uploads) > 10*1024*1024:
        raise ValueError('Use Card Spacing files totaling less than 10 MB.')
    owners = {}
    with pymupdf.open() as combined:
        for upload in uploads:
            try:
                drawing = parse_card_spacing(upload['bytes'])
                sizes = []
                for size in ('thin','wide',*GENERIC_SIZES):
                    try:
                        matching_strip_rectangle(drawing,size)
                    except ValueError:
                        continue
                    strip_crosshair_layout(drawing,size)
                    sizes.append(size)
                if not sizes:
                    raise ValueError('No supported strip size was found.')
            except ValueError as exc:
                raise ValueError(f"{upload['name']}: {exc}") from exc
            for size in sizes:
                if size in owners:
                    count = holiday_geometry_for_size(size).slot_count
                    raise ValueError(f"{owners[size]} and {upload['name']} both define the {count}-position {size} strip. "
                                     'Choose one spacing file for each strip size.')
                owners[size] = upload['name']
            with pymupdf.open(stream=_drawing_pdf_bytes(drawing),filetype='pdf') as source:
                combined.insert_pdf(source)
        result = combined.tobytes(garbage=3,deflate=True)
    parse_card_spacing(result)  # Apply the same page and drawing-size limits.
    return result


def render_eps_preview(data, strip_size, print_pdf, *, guides_in_pdf=False, row_number=1,
                       price_above=False):
    import pymupdf

    drawing = parse_card_spacing(data)
    geometry = holiday_geometry_for_size(strip_size)
    matches = [matching_strip_rectangle(drawing, strip_size, row_number)]
    left,bottom,right,top = drawing.bounds
    with pymupdf.open(stream=_drawing_pdf_bytes(drawing), filetype='pdf') as doc, \
            pymupdf.open(stream=print_pdf, filetype='pdf') as source:
        page = doc[0]
        # The print renderer adds 1/4-inch crop margins around the finished strip.
        clip = pymupdf.Rect(18, 18, 18 + geometry.width_pt, 18 + geometry.height_pt)
        if len(source) != 1 or not source[0].rect.contains(clip):
            raise ValueError('The print preview does not match the selected strip size.')
        if price_above:
            x0,y0,x1,y1 = matches[0]
            eps_clip = pymupdf.Rect(x0-left,top-y1,x1-left,top-y0)
            # Keep the engineering drawing intact. The close-up presents the
            # printable price strip ABOVE the EPS/card row at the same scale.
            scale = min(2,1800/max(page.rect.width,page.rect.height))
            overview = page.get_pixmap(matrix=pymupdf.Matrix(scale,scale)).tobytes('png')
            gap,margin = 12,18
            with pymupdf.open() as comparison:
                pair = comparison.new_page(width=geometry.width_pt+2*margin,
                    height=2*geometry.height_pt+gap+2*margin)
                price_target = pymupdf.Rect(margin,margin,margin+geometry.width_pt,margin+geometry.height_pt)
                eps_target = price_target + (0,geometry.height_pt+gap,0,geometry.height_pt+gap)
                pair.show_pdf_page(price_target,source,0,clip=clip,keep_proportion=False)
                pair.show_pdf_page(eps_target,doc,0,clip=eps_clip,keep_proportion=False)
                for x,y in strip_crosshair_layout(drawing,strip_size,row_number):
                    center = margin+x
                    pair.draw_line((center,price_target.y0),(center,eps_target.y1),
                                   color=RED,width=.6,dashes='[4 3]')
                    cy = eps_target.y0+y
                    pair.draw_line((center-9,cy),(center+9,cy),color=(0,.8,0),width=1)
                    pair.draw_line((center,cy-9),(center,cy+9),color=(0,.8,0),width=1)
                scale = min(2,1800/pair.rect.width)
                detail = pair.get_pixmap(matrix=pymupdf.Matrix(scale,scale)).tobytes('png')
            return overview,detail
        zoom = None
        for x0, y0, x1, y1 in matches:
            target = pymupdf.Rect(x0 - left, top - y1, x1 - left, top - y0)
            page.show_pdf_page(target, source, 0, clip=clip, keep_proportion=False)
            if not guides_in_pdf:
                page.draw_rect(target, color=RED, width=2)
                boundaries = (generic_boundaries(strip_size)[1:-1] if strip_size in GENERIC_SIZES else
                              [geometry.width_pt * c / geometry.slot_count for c in range(1, geometry.slot_count)])
                for boundary in boundaries:
                    x = target.x0 + boundary
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
