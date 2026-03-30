from __future__ import annotations



import re

from typing import List



import pdfplumber



from app.shared.models import AnnotationBox, CellData, FullPalletPage

from app.shared.pdf_utils import _group_nearby, _union

from app.standard_display.extract import parse_label_cell_text



def extract_full_pallet_pages(labels_pdf_bytes: bytes) -> List[FullPalletPage]:
    pages: List[FullPalletPage] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [
                w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))
            ]
            if not five:
                continue

            pw, ph = float(page.width), float(page.height)

            # ── FIX B: exclude tokens in the gift-card-holder zone ───────────
            # The top ~31 % of each labels page contains the "WM GIFTCARD IN NEW
            # PKG" holder labels plus GCI packaging product codes (08470, 08478,
            # 08481, 08705, 08706, …).  These are physical display fixtures — not
            # the gift cards that belong in the main / bonus product grid.
            # Filtering them out prevents them from becoming phantom grid cells.
            holder_zone_bottom = ph * 0.31
            five = [w for w in five if float(w.get("top", 0)) >= holder_zone_bottom]
            if not five:
                continue

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            # IMPORTANT: reduce over-splitting of columns
            x_centers = cluster_positions(xs, tol=max(10, pw * 0.025))
            y_centers = cluster_positions(ys, tol=max(7, ph * 0.012))
            if len(x_centers) == 0 or len(y_centers) == 0:
                continue

            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)
            if len(x_bounds) < 2 or len(y_bounds) < 2:
                continue

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col = int(np.argmin(np.abs(x_centers - xc)))
                row = int(np.argmin(np.abs(y_centers - yc)))
                dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                key = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_, token_w) in sorted(cell_map.items()):
                bbox = (
                    float(x_bounds[col]),
                    float(y_bounds[row]),
                    float(x_bounds[col + 1]),
                    float(y_bounds[row + 1]),
                )
                txt = (page.crop(bbox).extract_text() or "").strip()
                parsed_name, parsed_last5, qty = parse_label_cell_text(txt)
                token_last5 = str(token_w.get("text", "")).strip()
                raw_last5 = token_last5 if re.fullmatch(r"\d{5}", token_last5) else parsed_last5
                last5 = _to_last5(raw_last5)

                cells.append(
                    CellData(
                        row=row,
                        col=col,
                        bbox=bbox,
                        name=parsed_name,
                        last5=last5 or "",
                        qty=qty,
                        upc12=None,
                    )
                )

            annotations: List[AnnotationBox] = []
            wt = lambda w: str(w.get("text", "")).strip().upper()

            if xs:
                cx0_content = min(xs) - 15
                cx1_content = max(xs) + 15
            else:
                cx0_content, cx1_content = 150.0, 470.0

            wm_grp = [
                w
                for w in words
                if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "IN", "NEW", "PKG", "D"}
                and float(w["top"]) < ph * 0.30
            ]
            if wm_grp:
                by0 = min(w["top"] for w in wm_grp) - 8
                annotations.append(
                    AnnotationBox(
                        kind="gift_card_holders",
                        label="GIFT CARD HOLDERS",
                        bbox=(cx0_content, by0 - 22, cx1_content, by0 - 1),
                    )
                )

                sub_groups = _group_nearby(
                    [w for w in wm_grp if wt(w) in {"WM", "GIFTCAR", "GIFTCARD", "D", "IN", "NEW", "PKG"}],
                    x_tol=14,
                    y_tol=22,
                )
                for sg in sub_groups:
                    if len(sg) >= 3:
                        annotations.append(
                            AnnotationBox(
                                kind="wm_new_pkg",
                                label="WM GIFTCARD\nIN NEW PKG",
                                bbox=_union(sg, px=4, py=3),
                            )
                        )

            bonus_words = [w for w in words if wt(w) == "BONUS"]
            if bonus_words:
                bw = min(bonus_words, key=lambda w: float(w["top"]))
                bcy = (float(bw["top"]) + float(bw["bottom"])) / 2
                annotations.append(
                    AnnotationBox(
                        kind="bonus_strip",
                        label="BONUS",
                        bbox=(cx0_content, bcy - 12, cx1_content, bcy + 12),
                    )
                )

            mkt_words = [w for w in words if wt(w) in {"MARKETING", "MESSAGE", "PANEL"}]
            for grp in _group_nearby(mkt_words, x_tol=40, y_tol=20):
                if {wt(w) for w in grp} & {"MARKETING"}:
                    annotations.append(
                        AnnotationBox(
                            kind="marketing_signage",
                            label="MARKETING\nMESSAGE PANEL",
                            bbox=_union(grp, px=6, py=4),
                        )
                    )

            fraud_words = [w for w in words if wt(w) in {"FRAUD", "SIGNAGE"}]
            for grp in _group_nearby(fraud_words, x_tol=30, y_tol=16):
                if {wt(w) for w in grp} & {"FRAUD"}:
                    annotations.append(
                        AnnotationBox(
                            kind="fraud_signage",
                            label="FRAUD\nSIGNAGE",
                            bbox=_union(grp, px=6, py=4),
                        )
                    )

            pages.append(
                FullPalletPage(
                    page_index=pidx,
                    side_letter=chr(ord("A") + min(pidx, 3)),
                    cells=cells,
                    annotations=annotations,
                )
            )
    return pages
