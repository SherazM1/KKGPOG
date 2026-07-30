from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sams_club.holiday_price_strips import (
    SAMS_HOLIDAY_TEMPLATE_NAME,
    holiday_geometry_for_side,
)
from app.sams_club.price_strip_models import SamsPriceStripRow, SamsPriceStripSegment
from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf

BLEED_PT = 0.25 * inch


def _write_fallback_calibration_pdf(side: int, output_path: Path) -> None:
    geometry = holiday_geometry_for_side(side)
    page_w = geometry.width_pt + (BLEED_PT * 2)
    page_h = geometry.height_pt + (BLEED_PT * 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=(page_w, page_h))
    c.translate(BLEED_PT, BLEED_PT)
    c.setStrokeColorRGB(0.88, 0.12, 0.28)
    c.rect(0, 0, geometry.width_pt, geometry.height_pt, stroke=1, fill=0)
    slot_width = geometry.width_pt / geometry.slot_count
    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(0.88, 0.12, 0.28)
    c.drawString(
        6,
        geometry.height_pt - 14,
        f"Sam's Holiday {geometry.designation} Side {side} {geometry.width_pt:g}pt x {geometry.height_pt:g}pt",
    )
    for idx in range(geometry.slot_count + 1):
        x = idx * slot_width
        c.setStrokeColorRGB(0.04, 0.45, 0.75)
        c.line(x, 0, x, geometry.height_pt)
    for idx in range(geometry.slot_count):
        center_x = (idx + 0.5) * slot_width
        c.setStrokeColorRGB(0.86, 0.15, 0.15)
        c.line(center_x, 0, center_x, geometry.height_pt)
        c.setStrokeColorRGB(0.08, 0.64, 0.24)
        center_y = geometry.height_pt * 0.50
        c.line(center_x - 5, center_y, center_x + 5, center_y)
        c.line(center_x, center_y - 5, center_x, center_y + 5)
        c.setFillColorRGB(0.04, 0.42, 0.66)
        c.drawString(center_x + 3, 8, str(idx + 1))
    c.showPage()
    c.save()


def _sample_row(side: int) -> SamsPriceStripRow:
    geometry = holiday_geometry_for_side(side)
    return SamsPriceStripRow(
        pog=f"Sam's Holiday Side {side} Calibration",
        side=side,
        row=1,
        footer_text=f"Sam's Holiday Side {side} Calibration",
        segments=[
            SamsPriceStripSegment(
                pog=f"Sam's Holiday Side {side} Calibration",
                side=side,
                row=1,
                column=column,
                item_number=f"SLOT {column}",
                brand="CALIBRATION",
                desc_1=f"{geometry.designation.upper()} SLOT {column}",
                desc_2="CENTER CHECK",
                retail=str(10 * column),
                length=f"{geometry.width_in}x{geometry.height_in}",
            )
            for column in range(1, geometry.slot_count + 1)
        ],
    )


def write_calibration_pdf(side: int, output_path: Path, use_html_renderer: bool = False) -> None:
    if not use_html_renderer:
        _write_fallback_calibration_pdf(side, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = render_sams_price_strips_pdf(
        [_sample_row(side)],
        generated_by="Kendal King",
        template_name=SAMS_HOLIDAY_TEMPLATE_NAME,
        calibration=True,
    )
    if not result.pdf_bytes:
        _write_fallback_calibration_pdf(side, output_path)
        print(
            "HTML calibration renderer did not produce bytes; wrote ReportLab guide-only fallback. "
            + "; ".join(result.warnings)
        )
        return
    output_path.write_bytes(result.pdf_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Sam's Holiday calibration price-strip PDFs.")
    parser.add_argument("--side", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--output", required=True)
    parser.add_argument("--html-renderer", action="store_true")
    args = parser.parse_args()

    write_calibration_pdf(args.side, Path(args.output), use_html_renderer=args.html_renderer)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
