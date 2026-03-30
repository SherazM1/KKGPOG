from __future__ import annotations



import io

from pathlib import Path

from typing import Optional, Tuple



import fitz

from PIL import Image



from app.shared.constants import NAVY_RGB



def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = (h or "").lstrip("#").strip()
    if len(h) != 6:
        return NAVY_RGB
    return int(h[:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:], 16) / 255



def _try_load_logo() -> Optional[Image.Image]:
    for p in [
        Path.cwd() / "assets" / "KKG-Logo-02.png",
        Path(__file__).resolve().parents[2] / "assets" / "KKG-Logo-02.png",
    ]:
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                pass
    return None



def image_from_bytes(img_bytes: Optional[bytes]) -> Optional[Image.Image]:
    """Convert image bytes to PIL Image, or return None if bytes are None/empty."""
    if not img_bytes:
        return None
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception:
        return None



def crop_image_cell(
    images_doc: fitz.Document,
    page_index: int,
    bbox: Tuple[float, float, float, float],
    zoom: float = 2.6,
    inset: float = 0.045,
) -> Image.Image:
    page = images_doc.load_page(page_index)
    x0, top, x1, bottom = bbox
    w, h = x1 - x0, bottom - top
    rect = fitz.Rect(x0 + w * inset, top + h * inset, x1 - w * inset, bottom - h * inset)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
