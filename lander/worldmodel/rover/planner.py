"""Physics route planner for the rover (the backup; also gives training labels and reward shaping). No AI, no torch.

cost-to-go field = Dijkstra distance from a goal over an 8-connected grid; cell cost = 1 + 10 * P(unsafe) + 3 * (slope / rover limit)^2, cells steeper than the
rover's slope limit are impassable. Distances are in metres-equivalent (cost-weighted).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

W_HAZARD, W_SLOPE = 10.0, 3.0
_NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def cost_grid(slope: np.ndarray, hazard: np.ndarray, slope_limit_deg: float) -> np.ndarray:
    c = 1.0 + W_HAZARD * np.clip(hazard, 0, 1) + W_SLOPE * np.clip(slope / slope_limit_deg, 0, 1) ** 2
    c = c.astype(np.float64)
    c[slope > slope_limit_deg] = np.inf
    return c


def build_graph(cost: np.ndarray, mpp: float):
    h, w = cost.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dr, dc in _NEIGH:
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        a, b = idx[r0:r1, c0:c1], idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ca, cb = cost[r0:r1, c0:c1], cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = np.isfinite(ca) & np.isfinite(cb)
        rows.append(a[ok])
        cols.append(b[ok])
        vals.append(mpp * np.hypot(dr, dc) * 0.5 * (ca[ok] + cb[ok]))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)).tocsr()


def cost_to_go(graph, shape: tuple[int, int], goal_rc: tuple[float, float]) -> np.ndarray:
    """[H,W] cost-weighted distance from every cell to the goal (inf where unreachable)."""
    r, c = round(goal_rc[0]), round(goal_rc[1])
    d = dijkstra(graph, directed=False, indices=r * shape[1] + c)
    return d.reshape(shape)


def fields_for(slope, hazard, slope_limit_deg, mpp, goals_rc: np.ndarray) -> np.ndarray:
    """[G,H,W] float32 cost-to-go to each goal."""
    cost = cost_grid(slope, hazard, slope_limit_deg)
    g = build_graph(cost, mpp)
    return np.stack([cost_to_go(g, slope.shape, tuple(p)) for p in goals_rc]).astype(np.float32)
