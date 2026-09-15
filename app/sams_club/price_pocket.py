"""Vector Price Pocket tickets, two true-size 8.5 x 4 inch strips per page."""
from dataclasses import dataclass
from decimal import Decimal
import asyncio
import html
import re

from app.sams_club import render_price_strips_html as strip_html

from app.sams_club.price_strip_inputs import normalize_records, validate_price
from app.sams_club.price_strip_models import SamsPriceStripPdfResult
from app.sams_club.render_price_strips_html import _normalize_price_parts


@dataclass(frozen=True)
class PricePocketRecord:
    name: str
    price: Decimal
    item: str
    upc: str
    quantity: int = 1
    type: str = "GIFT CARD"


def pocket_records(records):
    result = []
    for i, record in enumerate(normalize_records(records), 1):
        missing = [f for f in ("name", "retail", "item_number", "upc") if not record.get(f)]
        if missing:
            raise ValueError(f"Record {i}: missing required fields: {', '.join(missing)}")
        quantity = record.get("quantity", "1")
        if not re.fullmatch(r"[1-9]\d*", quantity):
            raise ValueError(f"Record {i}: quantity must be a positive integer.")
        for field in ("item_number", "upc"):
            if not re.fullmatch(r"\d+", record[field]):
                raise ValueError(f"Record {i}: {field} must contain identifier digits as text.")
        if "type" in record and not record["type"]:
            raise ValueError(f"Record {i}: type cannot be blank.")
        result.append(PricePocketRecord(record["name"], validate_price(record["retail"]), record["item_number"], record["upc"], int(quantity), record.get("type", "GIFT CARD")))
    return result


def _pocket_html(records, warnings):
    font_css = strip_html.sams_font_face_css(warnings)
    cards = []
    for record in records:
        title_size = 32.0
        while title_size > 8 and strip_html._estimate_text_width(record.name, title_size, "semibold") > 576:
            title_size -= .25
        dollars, cents = _normalize_price_parts(str(record.price))
        amount = format(record.price, "f")
        if "." in amount:
            amount = amount.rstrip("0").rstrip(".")
        def text(value):
            return html.escape(str(value))
        cards.append(f'''<section class="card">
            <div class="title fit" style="font-size: {title_size}pt">{text(record.name)}</div>
            <div class="quantity fit">({record.quantity} X ${text(amount)})</div>
            <div class="type fit">{text(record.type)}</div>
            <div class="price"><span class="sign">$</span><span class="dollars">{text(dollars)}</span><span class="cents">{text(cents)}</span></div>
            <div class="item fit">{text(record.item)}</div>
            <div class="upc fit">({text(record.upc)})</div>
        </section>''')
    return f'''<!DOCTYPE html><html><head><style>
        {font_css}
        @page {{ size: 612pt 576pt; margin: 0; }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: {strip_html.SAMS_FONT_STACK}; color: black; }}
        .card {{ position: relative; width: 612pt; height: 288pt; box-shadow: inset 0 0 0 .4pt #bfbfbf; }}
        .card:nth-child(2n) {{ break-after: page; }}
        .fit {{ position: absolute; left: 18pt; width: 576pt; white-space: nowrap; line-height: 1; }}
        .title {{ top: 18pt; font-size: 32pt; font-weight: 600; }}
        .quantity {{ top: 55pt; font-size: 22pt; font-weight: 400; }}
        .type {{ top: 79pt; font-size: 22pt; font-weight: 400; }}
        .price {{ position: absolute; top: 113pt; left: 32pt; width: 548pt; display: flex; justify-content: center; align-items: flex-start; gap: 5pt; font-weight: 600; font-size: 144pt; line-height: .82; }}
        .sign, .cents {{ font-size: .5em; }}
        .sign {{ padding-top: .12em; }}
        .item {{ top: 238pt; font-size: 18pt; font-weight: 400; }}
        .upc {{ top: 259pt; font-size: 18pt; font-weight: 400; }}
    </style></head><body>{''.join(cards)}</body></html>'''


async def _render_pocket_async(content):
    async with strip_html.async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.set_content(content)
            await page.evaluate("() => document.fonts.ready")
            # Measure the actual loaded Gibson face; keep complete text and reject overflow.
            await page.evaluate("""() => {
                for (const el of document.querySelectorAll('.fit, .price')) {
                    let size = parseFloat(getComputedStyle(el).fontSize);
                    const min = el.classList.contains('price') ? 32 : 8;
                    while (el.scrollWidth > el.clientWidth + 1 && size > min) {
                        size -= .25;
                        el.style.fontSize = size + 'px';
                    }
                    if (el.scrollWidth > el.clientWidth + 1) {
                        throw new Error('Price Pocket text cannot fit without clipping: ' + el.textContent);
                    }
                }
            }""")
            return await page.pdf(width="8.5in", height="8in", print_background=True,
                                  prefer_css_page_size=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        finally:
            await browser.close()


def render_price_pocket_pdf(records):
    records = pocket_records([{"name": r.name, "price": r.price, "item": r.item, "upc": r.upc, "quantity": r.quantity, "type": r.type} for r in records])
    if not strip_html.PLAYWRIGHT_AVAILABLE:
        raise ValueError("Price Pocket requires the same Playwright/Chromium renderer as the existing Sam's layouts.")
    warnings = []
    content = _pocket_html(records, warnings)
    pdf_bytes = asyncio.run(_render_pocket_async(content))
    return SamsPriceStripPdfResult(pdf_bytes, (len(records)+1)//2, len(records), warnings)
