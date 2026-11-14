"""Terrain lookups on a tile, JAX-friendly (jit/vmap safe)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from contracts.tiles import Tile


class TerrainJ(NamedTuple):
    height: jnp.ndarray   # [H, W] m
    slope: jnp.ndarray    # [H, W] deg
    hazard: jnp.ndarray   # [H, W] 0..1
    mpp: float
    half_w_m: float       # x extent from centre
    half_h_m: float       # y extent from centre


def terrain_from_tile(tile: Tile) -> TerrainJ:
    h, w = tile.shape
    return TerrainJ(
        height=jnp.asarray(tile.height, jnp.float32),
        slope=jnp.asarray(tile.slope, jnp.float32),
        hazard=jnp.asarray(tile.hazard, jnp.float32),
        mpp=float(tile.meters_per_pixel),
        half_w_m=(w - 1) / 2 * float(tile.meters_per_pixel),
        half_h_m=(h - 1) / 2 * float(tile.meters_per_pixel),
    )


def flat_terrain(size: int = 64, mpp: float = 50.0, height: float = 0.0) -> TerrainJ:
    """Perfectly flat, hazard-free terrain — for unit tests and the guidance sanity check."""
    z = jnp.full((size, size), height, jnp.float32)
    zero = jnp.zeros((size, size), jnp.float32)
    half = (size - 1) / 2 * mpp
    return TerrainJ(z, zero, zero, mpp, half, half)


def _to_rc(t: TerrainJ, x, y):
    h, w = t.height.shape
    col = x / t.mpp + (w - 1) / 2.0
    row = (h - 1) / 2.0 - y / t.mpp
    return jnp.clip(row, 0.0, h - 1.0), jnp.clip(col, 0.0, w - 1.0)


def in_bounds(t: TerrainJ, x, y):
    return (jnp.abs(x) <= t.half_w_m) & (jnp.abs(y) <= t.half_h_m)


def bilinear(t: TerrainJ, grid: jnp.ndarray, x, y):
    h, w = grid.shape
    r, c = _to_rc(t, x, y)
    r0 = jnp.floor(r).astype(jnp.int32)
    c0 = jnp.floor(c).astype(jnp.int32)
    r1 = jnp.minimum(r0 + 1, h - 1)
    c1 = jnp.minimum(c0 + 1, w - 1)
    fr = r - r0
    fc = c - c0
    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
    return top * (1 - fr) + bot * fr


def nearest(t: TerrainJ, grid: jnp.ndarray, x, y):
    r, c = _to_rc(t, x, y)
    return grid[jnp.round(r).astype(jnp.int32), jnp.round(c).astype(jnp.int32)]


def height_at(t: TerrainJ, x, y):
    return bilinear(t, t.height, x, y)


def slope_at(t: TerrainJ, x, y):
    return bilinear(t, t.slope, x, y)


def hazard_at(t: TerrainJ, x, y):
    return nearest(t, t.hazard, x, y)


def safe_target_mask(tile: Tile, margin_px: int = 3) -> np.ndarray:
    """Cells where the backup hazard map is clear in a (2*margin+1)^2 window (numpy, for picking targets)."""
    haz = tile.hazard >= 0.5
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * margin_px + 1
    padded = np.pad(haz, margin_px, constant_values=True)
    win = sliding_window_view(padded, (k, k))
    return ~win.any(axis=(-1, -2))
