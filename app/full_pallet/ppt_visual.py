from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from PIL import Image

from app.shared.models import PptCard, PptSideCards
from app.shared.upload_utils import SUPPORTED_UPLOAD_IMAGE_EXTENSIONS


def _uploaded_name(uploaded: Any, fallback: str = "") -> str:
    return str(getattr(uploaded, "name", fallback) or fallback)


def _uploaded_bytes(uploaded: Any) -> bytes:
    if isinstance(uploaded, bytes):
        return uploaded
    if isinstance(uploaded, bytearray):
        return bytes(uploaded)
    if hasattr(uploaded, "getvalue"):
        return bytes(uploaded.getvalue())
    raise TypeError("Unsupported uploaded file object.")


def _image_to_pdf_page(doc: fitz.Document, payload: bytes) -> None:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="PNG")
        image_bytes = out.getvalue()

    page = doc.new_page(width=float(width), height=float(height))
    page.insert_image(page.rect, stream=image_bytes, keep_proportion=False)


def _visual_upload_to_pdf(uploaded: Any) -> bytes:
    name = _uploaded_name(uploaded, "top_cards")
    ext = Path(name).suffix.lower()
    payload = _uploaded_bytes(uploaded)

    if ext == ".pdf":
        return payload

    doc = fitz.open()
    try:
        if ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename.lower()):
                    if info.is_dir():
                        continue
                    if Path(info.filename).suffix.lower() not in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
                        continue
                    _image_to_pdf_page(doc, archive.read(info))
        elif ext in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
            _image_to_pdf_page(doc, payload)
        else:
            raise ValueError("Upload a PPTX, PDF, image file, or ZIP containing images.")

        if doc.page_count == 0:
            raise ValueError("No supported image files were found in the Top Cards upload.")
        return doc.tobytes()
    finally:
        doc.close()


def _line_items(page: fitz.Page) -> List[dict]:
    items: List[dict] = []
    data = page.get_text("dict") or {}
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(str(span.get("text", "")).strip() for span in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))
            items.append({"text": text, "bbox": (float(x0), float(y0), float(x1), float(y1))})
    return items


def _detect_side(page: fitz.Page, fallback: Optional[str]) -> Optional[str]:
    text = page.get_text("text") or ""
    match = re.search(r"\bSIDE\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return fallback


def _detect_labels(page: fitz.Page) -> List[dict]:
    labels: List[dict] = []
    id_re = re.compile(r"\bID\s*#?\s*[:\-]?\s*(\d{1,8})\b", re.IGNORECASE)
    lines = _line_items(page)
    for idx, item in enumerate(lines):
        match = id_re.search(item["text"])
        if not match:
            continue
        title_parts: List[str] = []
        x0, y0, x1, y1 = item["bbox"]
        for prev in reversed(lines[max(0, idx - 3):idx]):
            px0, py0, px1, py1 = prev["bbox"]
            if abs(((px0 + px1) / 2) - ((x0 + x1) / 2)) <= max(48.0, (x1 - x0) * 1.1):
                if py1 <= y0 + 2:
                    title_parts.insert(0, prev["text"])
        labels.append(
            {
                "card_id": match.group(1),
                "title": " ".join(title_parts).strip(),
                "cx": (x0 + x1) / 2,
                "top": y0,
            }
        )
    return labels


def _crop_page_png(page: fitz.Page, rect: fitz.Rect) -> bytes:
    clipped = rect & page.rect
    if clipped.is_empty or clipped.width <= 1 or clipped.height <= 1:
        clipped = page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clipped, alpha=False)
    return pix.tobytes("png")


def _tighten_card_crop(rect: fitz.Rect, *, is_side: bool) -> fitz.Rect:
    """Trim visual-export crops toward the card art to avoid adjacent-slot bleed."""
    if rect.width <= 4 or rect.height <= 4:
        return rect

    x_inset = rect.width * (0.125 if is_side else 0.115)
    top_inset = rect.height * 0.025
    bottom_inset = rect.height * (0.045 if is_side else 0.035)
    tightened = fitz.Rect(
        rect.x0 + x_inset,
        rect.y0 + top_inset,
        rect.x1 - x_inset,
        rect.y1 - bottom_inset,
    )
    if tightened.width < rect.width * 0.60 or tightened.height < rect.height * 0.70:
        return rect
    return tightened


def _cards_from_labels(page: fitz.Page, labels: List[dict]) -> Tuple[List[PptCard], List[PptCard]]:
    width = float(page.rect.width)
    height = float(page.rect.height)
    top_labels = sorted([label for label in labels if label["top"] < height * 0.42], key=lambda item: item["cx"])
    side_labels = sorted(
        [label for label in labels if label["top"] >= height * 0.42],
        key=lambda item: (round(item["top"] / max(1.0, height * 0.12)), item["cx"]),
    )

    def make_card(label: dict, idx: int, is_side: bool) -> PptCard:
        crop_w = width * (0.13 if not is_side else 0.15)
        crop_h = height * (0.19 if not is_side else 0.16)
        x0 = label["cx"] - crop_w / 2
        x1 = label["cx"] + crop_w / 2
        y1 = max(1.0, label["top"] - height * 0.012)
        y0 = max(0.0, y1 - crop_h)
        crop_rect = _tighten_card_crop(fitz.Rect(x0, y0, x1, y1), is_side=is_side)
        return PptCard(
            card_id=str(label.get("card_id") or idx + 1),
            title=str(label.get("title") or ""),
            image_bytes=_crop_page_png(page, crop_rect),
            image_ext="png",
        )

    top8 = [make_card(label, idx, False) for idx, label in enumerate(top_labels[:8])]
    side6 = [make_card(label, idx, True) for idx, label in enumerate(side_labels[:6])]
    return top8, side6


def _cards_from_fixed_regions(page: fitz.Page) -> Tuple[List[PptCard], List[PptCard]]:
    width = float(page.rect.width)
    height = float(page.rect.height)
    top8: List[PptCard] = []
    side6: List[PptCard] = []

    top_left = width * 0.035
    top_right = width * 0.965
    top_gap = width * 0.008
    top_w = (top_right - top_left - top_gap * 7) / 8
    for idx in range(8):
        x0 = top_left + idx * (top_w + top_gap)
        top8.append(
            PptCard(
                card_id=str(idx + 1),
                title="",
                image_bytes=_crop_page_png(
                    page,
                    _tighten_card_crop(
                        fitz.Rect(x0, height * 0.06, x0 + top_w, height * 0.36),
                        is_side=False,
                    ),
                ),
                image_ext="png",
            )
        )

    side_left = width * 0.55
    side_right = width * 0.965
    side_top = height * 0.46
    side_bottom = height * 0.90
    side_gap_x = width * 0.012
    side_gap_y = height * 0.018
    side_w = (side_right - side_left - side_gap_x * 2) / 3
    side_h = (side_bottom - side_top - side_gap_y) / 2
    for row in range(2):
        for col in range(3):
            idx = row * 3 + col
            x0 = side_left + col * (side_w + side_gap_x)
            y0 = side_top + row * (side_h + side_gap_y)
            side6.append(
                PptCard(
                    card_id=str(idx + 1),
                    title="",
                    image_bytes=_crop_page_png(
                        page,
                        _tighten_card_crop(fitz.Rect(x0, y0, x0 + side_w, y0 + side_h), is_side=True),
                    ),
                    image_ext="png",
                )
            )
    return top8, side6


def load_visual_ppt_cards(uploaded: Any) -> Dict[str, PptSideCards]:
    """Build top-card slots from a PDF/image export without changing PPTX parsing."""
    pdf_bytes = _visual_upload_to_pdf(uploaded)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parsed: Dict[str, PptSideCards] = {side: PptSideCards(side=side, top8=[], side6=[]) for side in "ABCD"}
        side_iter = iter("ABCD")
        for page in doc:
            fallback = next(side_iter, None)
            side = _detect_side(page, fallback)
            if side not in parsed:
                continue
            labels = _detect_labels(page)
            top8, side6 = _cards_from_labels(page, labels) if labels else _cards_from_fixed_regions(page)
            parsed[side] = PptSideCards(side=side, top8=top8[:8], side6=side6[:6])
        return parsed
    finally:
        doc.close()
