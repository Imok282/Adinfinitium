"""Patches of real LOLA terrain with their laser-measured pixels. Splits are whole tiles (test = Site23, never loaded in training)."""

from __future__ import annotations

import numpy as np

from contracts.paths import tile_path
from contracts.tiles import Tile, load_tile
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES, regions_for_site
from worldmodel.hazard.lola import region_name as _region_name

TEST_SITES = HOLDOUT_SITES      # one source of truth for the splits (worldmodel/hazard/lola.py)
__all__ = ["PATCH", "TEST_SITES", "TRAIN_SITES", "VAL_SITES", "load_region", "load_site", "normalise", "region_name", "regions_for_site", "sample_patch_origins"]
PATCH = 64            # 64 x 64 px = 320 m at 5 m/px
MIN_MEASURED = 40     # a patch needs at least this many laser pixels
SD_FLOOR_M = 1.0      # scaling floor so flat patches are not blown up


def region_name(site: str) -> str:
    return _region_name(site)


def load_site(site: str) -> Tile:
    """The centre window of a site (see regions_for_site + load_region for every window)."""
    return load_region(region_name(site))


def load_region(region: str) -> Tile:
    t = load_tile(tile_path(region))
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
