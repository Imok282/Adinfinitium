"""Turn a tile into what the hazard detector (and its physics backup) are allowed to see.

Real LOLA tiles are ~90% interpolated: only pixels with a laser return (tile.real_mask) are measured.
We HIDE the interpolated heights, refill the gaps ourselves with a fixed gaussian gap-fill (physics,
no AI), and build the features from that. The slope-rule backup uses exactly the same gap-filled
surface, so the U-Net / backup comparison is fair. The label is the OFFICIAL full-resolution slope
(tile.slope >= limit), which neither of them sees.

Normalisation constants below are model scalings, not mission numbers; mission numbers (slope limit,
gap-fill widths) come from config/mission.yaml.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from contracts.tiles import Tile, slope_from_height

CHANNELS = ("slope_filled_over_limit", "highpass_height", "laser_mask", "uncertainty", "distance_to_laser")
N_CHANNELS = len(CHANNELS)
HP_SIGMA_PX = 10.0   # high-pass: filled height minus its 10 px blur ...
HP_SCALE_M = 10.0    # ... divided by 10 m
UNC_SCALE_M = 1.0
DIST_SCALE_PX = 20.0
MIN_WEIGHT = 0.05    # a gap-fill level is trusted where the local laser weight is at least this


def measured_mask(tile: Tile) -> np.ndarray:
    """Pixels that hold a real measurement. Tiles without a mask (e.g. the demo tile) count as all measured."""
    if tile.real_mask is None:
        return np.ones(tile.shape, dtype=bool)
    return tile.real_mask.astype(bool)


def gap_fill(height: np.ndarray, measured: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Fill unmeasured pixels by normalised gaussian convolution, finest width first.

    Measured pixels keep their value exactly. Pixels farther than the coarsest width from any laser
    return get the mean of the measured heights."""
    h = height.astype(np.float64)
    if measured.all():
        return h
    if not measured.any():
        raise ValueError("tile has no measured (laser) pixels")
    ref = float(h[measured].mean())
    hm = np.where(measured, h - ref, 0.0)
    m = measured.astype(np.float64)
    out = np.full(h.shape, np.nan)
    for s in sigmas:
        den = gaussian_filter(m, s, mode="nearest")
        num = gaussian_filter(hm, s, mode="nearest")
        ok = np.isnan(out) & (den >= MIN_WEIGHT)
        out[ok] = num[ok] / den[ok]
    out[np.isnan(out)] = 0.0
    return np.where(measured, hm, out) + ref


def build_features(tile: Tile, limit_deg: float, sigmas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [C,H,W] float32, gap-filled slope [H,W] deg). The slope is the backup's input."""
    meas = measured_mask(tile)
    filled = gap_fill(tile.height, meas, sigmas)
    slope = slope_from_height(filled, tile.meters_per_pixel)
    hp = (filled - gaussian_filter(filled, HP_SIGMA_PX, mode="nearest")) / HP_SCALE_M
    dist = np.zeros(tile.shape) if meas.all() else distance_transform_edt(~meas)
    feats = np.stack([
        np.clip(slope / limit_deg, 0.0, 4.0),
        np.clip(hp, -4.0, 4.0),
        meas.astype(np.float64),
        np.clip(tile.uncertainty / UNC_SCALE_M, 0.0, 8.0),
        np.clip(dist / DIST_SCALE_PX, 0.0, 4.0),
    ]).astype(np.float32)
    return feats, slope
