#
#  Copyright (c) 2026, Adinfinitium team
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
"""Nested real-terrain levels of the lunar south pole for the Wall view (all OBSERVED, NASA LRO LOLA).

All levels use the south polar stereographic grid of the tiles and of the LROC WAC mosaic:
    x = 2R tan(pi/4 + lat/2) sin(lon),  y = 2R tan(pi/4 + lat/2) cos(lon),  R = 1737.4 km
so they nest without reprojection. Heights are metres above the 1737.4 km sphere.

  cap     75 S -> pole, from LDEM_75S_240M (PDS, LOLA GDR polar), block-averaged to ~1.2 km
  region  +-REGION_HALF_M around the site at the native 240 m
  local   the PGDA LOLA 5 m site DEM (<Site>_final_adj_5mpp_surf.tif) averaged to 20 m (~16 km)
The 5 m tile itself (with its data layers) is served by /api/layers.

Files (big, never in git) live in <data>/raw/lola_polar/ and <data>/raw/lola5m/; missing files are skipped.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
R_MOON_M = 1737.4e3
CAP_FILE = RAW / "lola_polar" / "LDEM_75S_240M.IMG"
CAP_N, CAP_SCALE_M, CAP_OFFSET_PX = 3812, 240.0, 1905.5  # [px], [m/px], [px] from the PDS label
CAP_BLOCK = 5  # [-] 240 m -> 1.2 km for the cap mesh
REGION_HALF_M = 50.0e3  # [m]


def ps_from_latlon(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    k = 2.0 * R_MOON_M * np.tan(np.pi / 4.0 + np.radians(lat_deg) / 2.0)
    return float(k * np.sin(np.radians(lon_deg))), float(k * np.cos(np.radians(lon_deg)))


@functools.lru_cache(maxsize=1)
def _cap_heights() -> np.ndarray | None:
    """[3812, 3812] heights [m]; row 0 = +y edge (PDS line 1), column 0 = -x edge."""
    if not CAP_FILE.exists() or CAP_FILE.stat().st_size != CAP_N * CAP_N * 2:
        return None
    dn = np.fromfile(CAP_FILE, dtype="<i2").reshape(CAP_N, CAP_N)
    return dn.astype(np.float32) * 0.5


def _cap_xy(row: float, col: float) -> tuple[float, float]:
    """PDS3 polar stereographic: x = (sample - S0 - 1) * scale, y = (1 + L0 - line) * scale (1-based line/sample)."""
    return (col + 1 - CAP_OFFSET_PX - 1) * CAP_SCALE_M, (1 + CAP_OFFSET_PX - (row + 1)) * CAP_SCALE_M


def _b64_i16(a: np.ndarray, scale: float) -> str:
    return base64.b64encode(np.clip(np.rint(a / scale), -32767, 32767).astype("<i2").tobytes()).decode()


def _level(name: str, h: np.ndarray, x0: float, y0: float, step: float, provenance: str) -> dict:
    """Grid level: h[rows, cols], (x0, y0) = polar-stereographic coordinates of h[0, 0] (north-west corner), step [m]."""
    return {"name": name, "rows": int(h.shape[0]), "cols": int(h.shape[1]), "x0_m": float(x0), "y0_m": float(y0), "step_m": float(step),
            "height_scale_m": 0.5, "height_i16": _b64_i16(h, 0.5), "provenance": provenance}


@functools.lru_cache(maxsize=4)
def polar_message(site_lat_deg: float, site_lon_deg: float, site_key: str = "Site01") -> dict:
    sx, sy = ps_from_latlon(site_lat_deg, site_lon_deg)
    levels = []
    H = _cap_heights()
    if H is not None:
        b = CAP_BLOCK
        n = (CAP_N // b) * b
        cap = H[:n, :n].reshape(n // b, b, n // b, b).mean(axis=(1, 3))
        x0, y0 = _cap_xy((b - 1) / 2, (b - 1) / 2)
        levels.append(_level("cap", cap, x0, y0, CAP_SCALE_M * b, "OBSERVED · NASA LRO LOLA GDR LDEM_75S_240M (block mean 1.2 km)"))
        c0 = int(round((sx - REGION_HALF_M) / CAP_SCALE_M + CAP_OFFSET_PX))
        r0 = int(round(CAP_OFFSET_PX - (sy + REGION_HALF_M) / CAP_SCALE_M))
        k = int(2 * REGION_HALF_M / CAP_SCALE_M)
        reg = H[max(r0, 0):r0 + k, max(c0, 0):c0 + k]
        x0, y0 = _cap_xy(max(r0, 0), max(c0, 0))
        levels.append(_level("region", reg, x0, y0, CAP_SCALE_M, "OBSERVED · NASA LRO LOLA GDR LDEM_75S_240M (240 m)"))
    tif = RAW / "lola5m" / f"{site_key}_final_adj_5mpp_surf.tif"
    if tif.exists():
        import tifffile

        with tifffile.TiffFile(tif) as t:
            p = t.pages[0]
            a = p.asarray().astype(np.float32)
            sc = p.tags["ModelPixelScaleTag"].value
            tp = p.tags["ModelTiepointTag"].value
        f = 4  # 5 m -> 20 m
        n0, n1 = (a.shape[0] // f) * f, (a.shape[1] // f) * f
        loc = np.nanmean(a[:n0, :n1].reshape(n0 // f, f, n1 // f, f), axis=(1, 3))
        step = float(sc[0]) * f
        levels.append(_level("local", loc, float(tp[3]) + step / 2 - float(sc[0]) / 2, float(tp[4]) - step / 2 + float(sc[1]) / 2, step,
                             f"OBSERVED · NASA LRO LOLA 5 m {site_key} DEM, PGDA (Barker et al. 2021), averaged to 20 m"))
    return {"type": "polar", "site_ps_m": [sx, sy], "levels": levels, "R_m": R_MOON_M}


def check_alignment(site_lat_deg: float, site_lon_deg: float, site_height_m: float) -> dict:
    """Height of the cap DEM at the site vs the 5 m tile's height there (the two must agree within tens of metres)."""
    H = _cap_heights()
    if H is None:
        return {"ok": None}
    sx, sy = ps_from_latlon(site_lat_deg, site_lon_deg)
    c = sx / CAP_SCALE_M + CAP_OFFSET_PX
    r = CAP_OFFSET_PX - sy / CAP_SCALE_M
    h = float(H[int(round(r)), int(round(c))])
    return {"cap_height_m": h, "tile_height_m": site_height_m, "difference_m": h - site_height_m}
