from __future__ import annotations



import os

from pathlib import Path



from reportlab.pdfbase import pdfmetrics

from reportlab.pdfbase.ttfonts import TTFont



def _safe_register_font(name: str, rel_path: str) -> None:
    try:
        full = str(Path(__file__).resolve().parents[2] / "assets" / rel_path)
        if os.path.isfile(full):
            pdfmetrics.registerFont(TTFont(name, full))
    except Exception:
        pass



def _font(preferred: str, fallback: str) -> str:
    try:
        pdfmetrics.getFont(preferred)
        return preferred
    except Exception:
        return fallback



_safe_register_font("Raleway", "Raleway-Regular.ttf")

_safe_register_font("Raleway-Bold", "Raleway-Bold.ttf")



TITLE_FONT = _font("Raleway-Bold", "Helvetica-Bold")

BODY_FONT = _font("Raleway", "Helvetica")

BODY_BOLD_FONT = _font("Raleway-Bold", "Helvetica-Bold")
