"""Physics backup for the terrain generator (no AI, no torch): gap-fill the laser pixels and add noise from LOLA's own error map.

Each 'version' = gaussian gap-fill of the measured heights + N(0, sigma) noise with sigma = the tile's `uncertainty` (LOLA toterr).
It is honest about what is known (measured pixels stay put, up to their error) and ignores terrain texture between them.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gap_fill(height: np.ndarray, measured: np.ndarray, sigmas: list[float], min_weight: float = 0.05) -> np.ndarray:
    h = height.astype(np.float64)
    if measured.all():
        return h
    ref = float(h[measured].mean())
    hm = np.where(measured, h - ref, 0.0)
    m = measured.astype(np.float64)
    out = np.full(h.shape, np.nan)
    for s in sigmas:
        den = gaussian_filter(m, s, mode="nearest")
        num = gaussian_filter(hm, s, mode="nearest")
        ok = np.isnan(out) & (den >= min_weight)
        out[ok] = num[ok] / den[ok]
    out[np.isnan(out)] = 0.0
    return np.where(measured, hm, out) + ref


def backup_versions(height: np.ndarray, measured: np.ndarray, sigma_m: np.ndarray | float, sigmas: list[float], n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """[n, H, W] plausible versions: gap-filled terrain + LOLA-error noise."""
    base = gap_fill(height, measured, sigmas)
    return base[None] + rng.normal(0.0, 1.0, (n, *base.shape)) * np.asarray(sigma_m)[None]
