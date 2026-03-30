from __future__ import annotations



from typing import List



import numpy as np



def kmeans_1d(values: List[float], k: int, iters: int = 40) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.array([], dtype=float)
    k = max(1, min(k, len(v)))
    qs = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    centers = np.quantile(v, qs)

    for _ in range(iters):
        d = np.abs(v[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        nc = centers.copy()
        for i in range(k):
            pts = v[labels == i]
            if len(pts):
                nc[i] = float(pts.mean())
        if np.allclose(nc, centers):
            break
        centers = nc

    return np.sort(centers)



def cluster_positions(values: List[float], tol: float) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    vals = sorted(float(v) for v in values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - float(np.mean(groups[-1]))) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([float(np.mean(g)) for g in groups])



def boundaries_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.sort(np.asarray(centers, dtype=float))
    if len(c) == 0:
        return c
    if len(c) == 1:
        return np.array([c[0] - 50, c[0] + 50])
    mids = (c[:-1] + c[1:]) / 2
    left = c[0] - (mids[0] - c[0])
    right = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[left], mids, [right]])
