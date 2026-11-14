"""STARTER (Arnav): lunar terrain helper for Basilisk scenarios + OBJ export for Vizard.

    from basilisk_mods.lunar_terrain import Terrain
    terr = Terrain.from_tile("data/tiles/demo_craters.npz")
    terr.ground_height(x_m, y_m)       # metres, tile grid (x east-ish, y north-ish, origin = tile centre)
    terr.slope(x_m, y_m)               # degrees
    python -m basilisk_mods.lunar_terrain data/tiles/demo_craters.npz --step 2   # -> vizard/terrain_<region>.obj

Projection: south-polar stereographic on a 1737.4 km sphere (the LOLA polar products use this).
    x = 2R tan(pi/4 + lat/2) sin(lon),   y = 2R tan(pi/4 + lat/2) cos(lon)      (lat < 0, radians)
!! VERIFY the sign/axis convention against the tile's PDS label (.lbl) and Imok's 3 test points
   before trusting absolute positions (KAN-81). Round-trip consistency is tested in tests/.
Tile grids are centred on the tile: tile_x = ps_x - centre_ps_x (see contracts/tiles.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.paths import repo_root
from contracts.tiles import Tile, load_tile

R_MOON_M = 1737.4e3


# ---------------------------------------------------------------- projection
def ps_from_latlon(lat_deg, lon_deg, radius_m: float = R_MOON_M):
    """South-polar stereographic (metres) from latitude/longitude (degrees, lat < 0)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    k = 2.0 * radius_m * np.tan(np.pi / 4.0 + lat / 2.0)
    return k * np.sin(lon), k * np.cos(lon)


def latlon_from_ps(x_m, y_m, radius_m: float = R_MOON_M):
    rho = np.hypot(x_m, y_m)
    lat = 2.0 * np.arctan(rho / (2.0 * radius_m)) - np.pi / 2.0
    lon = np.arctan2(x_m, y_m)
    return np.degrees(lat), np.degrees(lon) % 360.0


def moon_fixed_from_latlon(lat_deg, lon_deg, height_m=0.0, radius_m: float = R_MOON_M):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r = radius_m + height_m
    return np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)], axis=-1)


def ps_from_moon_fixed(r_m, radius_m: float = R_MOON_M):
    """Moon-fixed position (metres, x/y/z) -> (ps_x, ps_y, altitude above the sphere)."""
    r_m = np.asarray(r_m, float)
    rn = np.linalg.norm(r_m, axis=-1)
    lat = np.degrees(np.arcsin(r_m[..., 2] / rn))
    lon = np.degrees(np.arctan2(r_m[..., 1], r_m[..., 0]))
    x, y = ps_from_latlon(lat, lon, radius_m)
    return x, y, rn - radius_m


# ------------------------------------------------------------------- lookup
@dataclass
class Terrain:
    tile: Tile
    center_ps_x: float
    center_ps_y: float

    @classmethod
    def from_tile(cls, path_or_tile) -> Terrain:
        tile = path_or_tile if isinstance(path_or_tile, Tile) else load_tile(path_or_tile)
        cx, cy = ps_from_latlon(tile.center_lat_deg, tile.center_lon_deg)
        return cls(tile, float(cx), float(cy))

    def _rc(self, x_m, y_m):
        h, w = self.tile.shape
        mpp = self.tile.meters_per_pixel
        col = np.asarray(x_m) / mpp + (w - 1) / 2.0
        row = (h - 1) / 2.0 - np.asarray(y_m) / mpp
        return np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)

    def _bilinear(self, grid, x_m, y_m):
        h, w = grid.shape
        r, c = self._rc(x_m, y_m)
        r0 = np.floor(r).astype(int)
        c0 = np.floor(c).astype(int)
        r1 = np.minimum(r0 + 1, h - 1)
        c1 = np.minimum(c0 + 1, w - 1)
        fr, fc = r - r0, c - c0
        top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
        bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
        return top * (1 - fr) + bot * fr

    def ground_height(self, x_m, y_m):
        """Ground height [m] at tile coordinates [m]."""
        return self._bilinear(self.tile.height, x_m, y_m)

    def slope(self, x_m, y_m):
        """Local slope [deg] at tile coordinates [m]."""
        return self._bilinear(self.tile.slope, x_m, y_m)

    def tile_xy_from_moon_fixed(self, r_m):
        """Moon-fixed position [m] -> tile x, y [m] and altitude above the sphere [m]."""
        x, y, alt = ps_from_moon_fixed(r_m)
        return x - self.center_ps_x, y - self.center_ps_y, alt

    def altitude_above_ground(self, r_m):
        """Moon-fixed position [m] -> height above the terrain surface [m]."""
        x, y, alt = self.tile_xy_from_moon_fixed(r_m)
        return alt - self.ground_height(x, y)


# ---------------------------------------------------------------- OBJ export
def tile_to_obj(tile: Tile, path: str | Path, step: int = 2, exaggeration: float = 1.0) -> Path:
    """Write the tile as an OBJ mesh (metres, origin = tile centre, +Y up for Vizard/Unity).

    step: keep every Nth pixel (smaller files). exaggeration != 1 is written into the file header —
    if you use it, the video MUST show "vertical exaggeration xN".
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hgt = tile.height[::step, ::step].astype(np.float64)
    h, w = hgt.shape
    mpp = tile.meters_per_pixel * step
    xs = (np.arange(w) - (w - 1) / 2.0) * mpp
    ys = ((h - 1) / 2.0 - np.arange(h)) * mpp
    zc = float(np.median(hgt))
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Adinfinitium terrain {tile.region} [{tile.meta.provenance.banner}] source: {tile.meta.source}\n")
        f.write(f"# metres, origin tile centre, step {step}, vertical exaggeration x{exaggeration}\n")
        for i in range(h):
            f.writelines(f"v {xs[j]:.2f} {(hgt[i, j] - zc) * exaggeration:.2f} {ys[i]:.2f}\n" for j in range(w))
        for i in range(h - 1):
            for j in range(w - 1):
                a = i * w + j + 1
                b, c, d = a + 1, a + w, a + w + 1
                f.write(f"f {a} {c} {b}\nf {b} {c} {d}\n")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", type=Path)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--exaggeration", type=float, default=1.0)
    a = ap.parse_args()
    tile = load_tile(a.tile)
    out = tile_to_obj(tile, repo_root() / "vizard" / f"terrain_{tile.region}.obj", a.step, a.exaggeration)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)  [{tile.meta.provenance.banner}]")


if __name__ == "__main__":
    main()
