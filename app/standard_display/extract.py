from __future__ import annotations



from typing import List, Optional, Tuple



import pdfplumber



from app.shared.clustering import boundaries_from_centers, cluster_positions

from app.shared.constants import LAST5_RE

from app.shared.models import CellData, PageData



def parse_label_cell_text(text: str) -> Tuple[str, str, Optional[int]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    joined = " ".join(lines)

    m = LAST5_RE.search(joined)
    last5 = m.group(1) if m else ""

    nums = re.findall(r"\b(\d{1,3})\b", joined)
    qty = int(nums[-1]) if nums else None

    name = " ".join(
        ln
        for ln in lines
        if not (last5 and last5 in ln)
        and not (qty is not None and re.fullmatch(str(qty), ln))
    ).strip()

    return name, last5, qty



def extract_pages_from_labels(labels_pdf_bytes: bytes, n_cols: int) -> List[PageData]:
    pages: List[PageData] = []
    with pdfplumber.open(io.BytesIO(labels_pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            five = [
                w for w in words if re.fullmatch(r"\d{5}", str(w.get("text", "")))
            ]
            if not five:
                continue

            xs = [(w["x0"] + w["x1"]) / 2 for w in five]
            ys = [(w["top"] + w["bottom"]) / 2 for w in five]

            x_centers = kmeans_1d(xs, n_cols)
            y_centers = kmeans_1d(ys, max(1, round(len(five) / max(1, n_cols))))

            x_bounds = boundaries_from_centers(x_centers)
            y_bounds = boundaries_from_centers(y_centers)

            cell_map: Dict[Tuple[int, int], Tuple[float, dict]] = {}
            for w, xc, yc in zip(five, xs, ys):
                col = int(np.argmin(np.abs(x_centers - xc)))
                row = int(np.argmin(np.abs(y_centers - yc)))
                dist = abs(x_centers[col] - xc) + abs(y_centers[row] - yc)
                key = (row, col)
                if key not in cell_map or dist < cell_map[key][0]:
                    cell_map[key] = (dist, w)

            cells: List[CellData] = []
            for (row, col), (_, _w) in sorted(cell_map.items()):
                bbox = (
                    float(x_bounds[col]),
                    float(y_bounds[row]),
                    float(x_bounds[col + 1]),
                    float(y_bounds[row + 1]),
                )
                txt = (page.crop(bbox).extract_text() or "").strip()
                name, last5, qty = parse_label_cell_text(txt)
                cells.append(
                    CellData(
                        row=row,
                        col=col,
                        bbox=bbox,
                        name=name,
                        last5=last5,
                        qty=qty,
                        upc12=None,
                    )
                )

            pages.append(
                PageData(page_index=pidx, x_bounds=x_bounds, y_bounds=y_bounds, cells=cells)
            )
    return pages
