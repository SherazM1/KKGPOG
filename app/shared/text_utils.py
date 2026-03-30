from __future__ import annotations



import math
import re
from datetime import date

from typing import List, Optional, Tuple



import pandas as pd
from PIL import Image

from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics

from reportlab.pdfgen import canvas



from app.shared.constants import DIGITS_RE, NAVY_RGB

from app.shared.fonts import BODY_BOLD_FONT, BODY_FONT, TITLE_FONT

_STOPWORDS = {"A", "AN", "AND", "FOR", "IN", "OF", "ON", "OR", "THE", "TO", "WITH"}



def _norm_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()



def _coerce_upc12(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"\.0$", "", str(v).strip())
    s = DIGITS_RE.sub("", s)
    return s.zfill(12) if s else None



def _coerce_int(v: object) -> Optional[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"[^\d\-]", "", re.sub(r"\.0$", "", str(v).strip()))
    try:
        return int(s) if s else None
    except ValueError:
        return None



def _to_last5(v: object) -> str:
    s = DIGITS_RE.sub("", str(v or "").strip())
    if not s:
        return ""
    if len(s) >= 5:
        return s[-5:]
    return s.zfill(5)



def _norm_header(v: object) -> str:
    s = re.sub(r"[^A-Z0-9]+", "_", str(v).strip().upper()).strip("_")
    return s or "COL"



def _find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(len(df), 50)):
        row = df.iloc[i].astype(str).str.upper().tolist()
        if any("UPC" in c for c in row) and any(
            tok in " ".join(row) for tok in ("NAME", "DESCRIPTION", "CPP")
        ):
            return i
    return 0



def _pick_col(cols: List[str], tokens: List[str], fallback: int) -> str:
    uc = [c.upper() for c in cols]
    for t in tokens:
        for i, c in enumerate(uc):
            if t in c:
                return cols[i]
    return cols[min(fallback, len(cols) - 1)]



def _pick_col_optional(cols: List[str], tokens: List[str]) -> Optional[str]:
    uc = [c.upper() for c in cols]
    for t in tokens:
        for i, c in enumerate(uc):
            if t in c:
                return cols[i]
    return None



def wrap_text(text: str, max_w: float, font: str, size: float) -> List[str]:
    parts = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []
    for p in parts:
        trial = " ".join(cur + [p])
        if not cur or pdfmetrics.stringWidth(trial, font, size) <= max_w:
            cur.append(p)
        else:
            lines.append(" ".join(cur))
            cur = [p]
    if cur:
        lines.append(" ".join(cur))
    return lines



def _fit_font(
    text: str,
    font: str,
    max_w: float,
    max_h: float,
    lo: float,
    hi: float,
    step: float = 0.5,
) -> float:
    t = (text or "").strip()
    if not t:
        return lo
    s = hi
    while s >= lo:
        if s <= max_h and pdfmetrics.stringWidth(t, font, s) <= max_w:
            return s
        s -= step
    return lo



def _ellipsis(text: str, font: str, size: float, max_w: float) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if pdfmetrics.stringWidth(t, font, size) <= max_w:
        return t
    ell = "…"
    lo, hi = 0, len(t)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = t[:mid].rstrip() + ell
        if pdfmetrics.stringWidth(cand, font, size) <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best



def _draw_gradient(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
    steps: int = 80,
) -> None:
    for i in range(max(8, steps)):
        t = i / (steps - 1)
        c.setFillColorRGB(
            left[0] * (1 - t) + right[0] * t,
            left[1] * (1 - t) + right[1] * t,
            left[2] * (1 - t) + right[2] * t,
        )
        xi = x + w * i / steps
        c.rect(xi, y, w / steps + 0.5, h, stroke=0, fill=1)



def _draw_header(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    top_bar_h: float,
    title_text: str,
    right_label: str,
    logo_img: Optional[Image.Image],
    grad_l: Tuple[float, float, float],
    grad_r: Tuple[float, float, float],
) -> None:
    hy = page_h - top_bar_h
    _draw_gradient(c, 0, hy, page_w, top_bar_h, grad_l, grad_r, steps=120)

    tx = 14.0
    if logo_img:
        lw, lh = logo_img.size
        th = top_bar_h * 0.62
        r = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(
            ImageReader(logo_img),
            14,
            hy + (top_bar_h - dh) / 2,
            dw,
            dh,
            mask="auto",
        )
        tx = 14 + dw + 10

    c.setFillColorRGB(1, 1, 1)
    fs = _fit_font(title_text, TITLE_FONT, page_w - tx - 130, top_bar_h * 0.72, 18, 36)
    c.setFont(TITLE_FONT, fs)
    c.drawString(tx, hy + (top_bar_h - fs) / 2 + 2, title_text)

    if right_label:
        rfs = _fit_font(right_label, TITLE_FONT, 120, top_bar_h * 0.68, 14, 28)
        rw = pdfmetrics.stringWidth(right_label, TITLE_FONT, rfs)
        c.setFont(TITLE_FONT, rfs)
        c.drawString(page_w - rw - 14, hy + (top_bar_h - rfs) / 2 + 2, right_label)

    c.setLineWidth(0.8)
    c.setStrokeColorRGB(0.88, 0.88, 0.92)
    c.line(0, hy, page_w, hy)



def _draw_cell_text_block(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    name: str,
    upc12: Optional[str],
    last5: str,
    qty: Optional[int],
) -> None:
    px, py = 5, 4
    mw = max(12.0, w - px * 2)
    avail = max(10.0, h - py * 2)

    upc_str = upc12 if upc12 else (f"???????{last5}" if last5 else "")
    ns = 13.0
    while ns >= 5.0:
        ms = max(5.0, ns * 0.86)
        lhn = ns * 1.18
        lhm = ms * 1.18

        nls = wrap_text(name or "", mw, BODY_BOLD_FONT, ns)
        ul = f"UPC: {upc_str}" if upc_str else ""
        ql = f"Qty: {qty}" if qty is not None else ""

        need = (
            len(nls) * lhn
            + (2.5 if nls and (ul or ql) else 0)
            + (lhm if ul else 0)
            + (lhm if ql else 0)
        )
        if need <= avail:
            ty = y + h - py - ns
            c.setFillColorRGB(*NAVY_RGB)
            c.setFont(BODY_BOLD_FONT, ns)
            for ln in nls:
                c.drawString(x + px, max(y + py, ty), ln)
                ty -= lhn

            if nls and (ul or ql):
                ty -= 2.5
            c.setFont(BODY_FONT, ms)
            if ul:
                c.drawString(x + px, max(y + py, ty), ul)
                ty -= lhm
            if ql:
                c.drawString(x + px, max(y + py, ty), ql)
            return
        ns -= 0.5

    fs = 5.0
    ty = y + h - py - fs
    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, fs)
    for ln in wrap_text(name or "", mw, BODY_BOLD_FONT, fs):
        c.drawString(x + px, max(y + py, ty), ln)
        ty -= fs * 1.18
    c.setFont(BODY_FONT, fs)
    if upc_str:
        c.drawString(x + px, max(y + py, ty), f"UPC: {upc_str}")
        ty -= fs * 1.18
    if qty is not None:
        c.drawString(x + px, max(y + py, ty), f"Qty: {qty}")



def _is_important_token(tok: str) -> bool:
    t = tok.strip().upper()
    if not t:
        return False
    if "$" in t and re.search(r"\d", t):
        return True
    if re.fullmatch(r"\d{1,3}", t):
        return True
    if re.search(r"\d", t) and any(u in t for u in ("PK", "CT", "OZ", "LB", "MG", "ML")):
        return True
    if re.fullmatch(r"[A-Z]\d{1,3}", t):
        return True
    return False



def _compact_one_line_name(name: str) -> str:
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    tokens = raw.split()
    imp = [t for t in tokens if _is_important_token(t)]
    core = [t for t in tokens if t.upper() not in _STOPWORDS]
    use = core if core else tokens
    for t in imp:
        if t not in use:
            use.append(t)
    return " ".join(use)



def _fit_name_preserve_qualifiers(name: str, font: str, size: float, max_w: float) -> str:
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    tokens = raw.split()

    def is_important(t: str) -> bool:
        tt = t.upper()
        if "$" in tt and re.search(r"\d", tt):
            return True
        if re.search(r"\d", tt) and any(x in tt for x in ("PK", "CT", "OZ", "LB", "MG", "ML")):
            return True
        if re.fullmatch(r"[A-Z]\d{1,3}", tt):
            return True
        return False

    keep = tokens[:]
    while keep and pdfmetrics.stringWidth(" ".join(keep), font, size) > max_w:
        removable = [i for i, t in enumerate(keep) if 0 < i < len(keep) - 1 and not is_important(t)]
        if not removable:
            break
        mid = removable[len(removable) // 2]
        keep.pop(mid)

    candidate = " ".join(keep)
    if pdfmetrics.stringWidth(candidate, font, size) <= max_w:
        return candidate
    return _ellipsis(candidate, font, size, max_w)



def _draw_full_pallet_header(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    header_h: float,
    title_text: str,
    right_label: str,
    logo_img: Optional[Image.Image],
) -> None:
    y0 = page_h - header_h
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, y0, page_w, header_h, stroke=0, fill=1)

    tx = 18.0
    if logo_img:
        lw, lh = logo_img.size
        th = header_h * 0.62
        r = th / max(1, lh)
        dw, dh = lw * r, lh * r
        c.drawImage(
            ImageReader(logo_img),
            18,
            y0 + (header_h - dh) / 2,
            dw,
            dh,
            mask="auto",
        )
        tx = 18 + dw + 12

    c.setFillColorRGB(*NAVY_RGB)
    fs = _fit_font(title_text, BODY_BOLD_FONT, page_w - tx - 160, header_h * 0.70, 14, 26)
    c.setFont(BODY_BOLD_FONT, fs)
    c.drawString(tx, y0 + (header_h - fs) / 2 + 1, title_text)

    if right_label:
        rfs = _fit_font(right_label, BODY_BOLD_FONT, 150, header_h * 0.68, 12, 24)
        rw = pdfmetrics.stringWidth(right_label, BODY_BOLD_FONT, rfs)
        c.setFont(BODY_BOLD_FONT, rfs)
        c.drawString(page_w - rw - 18, y0 + (header_h - rfs) / 2 + 1, right_label)

    c.setLineWidth(0.8)
    c.setStrokeColorRGB(0.86, 0.86, 0.88)
    c.line(0, y0, page_w, y0)



def _draw_footer(c: canvas.Canvas, page_w: float, outer_margin: float, footer_h: float) -> None:
    fy = outer_margin + footer_h
    c.setLineWidth(0.7)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.line(outer_margin, fy, page_w - outer_margin, fy)

    c.setFillColorRGB(*NAVY_RGB)
    c.setFont(BODY_BOLD_FONT, 11)
    left = f"Date: {date.today().isoformat()}"
    mid = "Generated by Kendal King"
    c.drawString(outer_margin, outer_margin + 11, left)
    mw = pdfmetrics.stringWidth(mid, BODY_BOLD_FONT, 11)
    c.drawString((page_w - mw) / 2, outer_margin + 11, mid)
