from __future__ import annotations



from typing import List, Tuple



def _wc(w: dict) -> Tuple[float, float]:
    return (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2



def _union(words: List[dict], px: float = 0, py: float = 0) -> Tuple[float, float, float, float]:
    return (
        min(w["x0"] for w in words) - px,
        min(w["top"] for w in words) - py,
        max(w["x1"] for w in words) + px,
        max(w["bottom"] for w in words) + py,
    )



def _group_nearby(words: List[dict], x_tol: float, y_tol: float) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for w in sorted(words, key=lambda ww: (_wc(ww)[1], _wc(ww)[0])):
        cx, cy = _wc(w)
        placed = False
        for g in groups:
            bx0, bt, bx1, bb = _union(g)
            gcx, gcy = (bx0 + bx1) / 2, (bt + bb) / 2
            if abs(cx - gcx) <= max(x_tol, (bx1 - bx0) / 2 + x_tol * 0.4) and abs(
                cy - gcy
            ) <= max(y_tol, (bb - bt) / 2 + y_tol * 0.4):
                g.append(w)
                placed = True
                break
        if not placed:
            groups.append([w])

    changed = True
    while changed:
        changed = False
        merged: List[List[dict]] = []
        while groups:
            base = groups.pop(0)
            bx0, bt, bx1, bb = _union(base)
            i = 0
            while i < len(groups):
                gx0, gt, gx1, gb = _union(groups[i])
                if not (
                    bx1 + x_tol < gx0
                    or gx1 + x_tol < bx0
                    or bb + y_tol < gt
                    or gb + y_tol < bt
                ):
                    base.extend(groups.pop(i))
                    bx0, bt, bx1, bb = _union(base)
                    changed = True
                else:
                    i += 1
            merged.append(base)
        groups = merged

    return groups
