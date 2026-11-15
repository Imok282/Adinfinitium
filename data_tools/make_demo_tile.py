"""Synthetic crater-field tile in the official tile format.

Purpose: everyone can build and test against data/tiles/demo_craters.npz BEFORE the real
LOLA tiles from KAN-21 exist. The file is labelled GENERATED — it is never the real Moon.

    python -m data_tools.make_demo_tile                  # -> data/tiles/demo_craters.npz
    python -m data_tools.make_demo_tile --size 512 --mpp 5 --seed 7 --region demo_small
"""

from __future__ import annotations

import argparse

import numpy as np

from contracts.paths import tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, hazard_from_slope, save_tile, slope_from_height


def crater_profile(r: np.ndarray, radius: float, depth: float) -> np.ndarray:
    """Simple bowl + raised rim. r and radius in metres. Returns height delta in metres."""
    x = r / radius
    bowl = np.where(x < 1.0, depth * (x**2 - 1.0), 0.0)
    rim_h = 0.15 * depth
    rim = rim_h * np.exp(-((x - 1.0) / 0.25) ** 2)
    return bowl + rim


def make_demo_tile(size: int = 400, mpp: float = 5.0, seed: int = 0, n_craters: int = 60,
                   region: str = "demo_craters", slope_limit_deg: float = 12.0) -> Tile:
    rng = np.random.default_rng(seed)
    half = (size - 1) / 2 * mpp
    xs = np.linspace(-half, half, size)
    X, Y = np.meshgrid(xs, xs[::-1])  # row 0 = north (+y)

    # Gentle regional tilt + long-wavelength undulation.
    height = 0.02 * X - 0.01 * Y + 8.0 * np.sin(X / 400.0) * np.cos(Y / 550.0)

    # Power-law crater sizes (many small, few large). Keep the centre 150 m mostly clear
    # so there is always at least one plausible landing zone.
    radii = np.clip(rng.pareto(1.8, n_craters) * 15.0 + 8.0, 8.0, size * mpp / 6)
    for rad in radii:
        cx, cy = rng.uniform(-half, half, 2)
        if np.hypot(cx, cy) < 150.0 + rad:
            continue
        depth = 0.2 * rad
        height += crater_profile(np.hypot(X - cx, Y - cy), rad, depth)

    # Small-scale roughness.
    height += rng.normal(0.0, 0.15, height.shape)

    slope = slope_from_height(height, mpp)
    hazard = hazard_from_slope(slope, slope_limit_deg)
    uncertainty = np.full_like(height, 1.0, dtype=np.float32)
    psr = np.zeros(height.shape, dtype=bool)

    meta = Meta(provenance=Provenance.GENERATED,
                source=f"data_tools.make_demo_tile seed={seed} (synthetic crater field)",
                frame="local tile ENU (synthetic)", created_by="data_tools.make_demo_tile")
    return Tile(region=region, meters_per_pixel=mpp, height=height.astype(np.float32),
                slope=slope, hazard=hazard, uncertainty=uncertainty, psr_mask=psr, meta=meta,
                center_lat_deg=-89.5, center_lon_deg=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--mpp", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--craters", type=int, default=60)
    ap.add_argument("--region", default="demo_craters")
    a = ap.parse_args()
    tile = make_demo_tile(a.size, a.mpp, a.seed, a.craters, a.region)
    p = save_tile(tile, tile_path(a.region))
    print(f"wrote {p}  shape={tile.shape}  mpp={tile.meters_per_pixel}  "
          f"hazard={tile.hazard.mean():.1%}  [{tile.meta.provenance.banner}]")


if __name__ == "__main__":
    main()
