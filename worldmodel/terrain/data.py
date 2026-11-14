"""Patches of real LOLA terrain with their laser-measured pixels. Splits are whole tiles (test = Site23, never loaded in training)."""

from __future__ import annotations

import numpy as np

from contracts.paths import tile_path
from contracts.tiles import Tile, load_tile

TRAIN_SITES = ["Site01", "Site04", "Site06", "Site07", "Site20"]
VAL_SITES = ["Site11"]
TEST_SITES = ["Site23"]
PATCH = 64            # 64 x 64 px = 320 m at 5 m/px
MIN_MEASURED = 40     # a patch needs at least this many laser pixels
SD_FLOOR_M = 1.0      # scaling floor so flat patches are not blown up


def region_name(site: str) -> str:
    return f"lola5m_{site.lower()}"


def load_site(site: str) -> Tile:
    t = load_tile(tile_path(region_name(site)))
    if t.real_mask is None:
        raise ValueError(f"{t.region} has no real_mask: cannot tell measured pixels from interpolated ones")
    return t


def sample_patch_origins(mask: np.ndarray, n: int, rng: np.random.Generator, patch: int = PATCH, min_measured: int = MIN_MEASURED) -> np.ndarray:
    """n random (row, col) top-left corners of patches that contain at least `min_measured` laser pixels."""
    H, W = mask.shape
    ii = np.cumsum(np.cumsum(mask.astype(np.int32), 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    out = []
    while len(out) < n:
        r = rng.integers(0, H - patch + 1, 4 * n)
        c = rng.integers(0, W - patch + 1, 4 * n)
        cnt = ii[r + patch, c + patch] - ii[r, c + patch] - ii[r + patch, c] + ii[r, c]
        out += [(a, b) for a, b, k in zip(r, c, cnt) if k >= min_measured]
    return np.array(out[:n])


def normalise(height: np.ndarray, known: np.ndarray, sd_floor: float = SD_FLOOR_M) -> tuple[float, float]:
    """Patch mean and scale from the KNOWN (conditioning) pixels only, so inference can recompute them exactly."""
    v = height[known]
    return float(v.mean()), float(max(v.std(), sd_floor))
