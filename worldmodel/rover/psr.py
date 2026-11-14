"""NASA permanent-shadow regions (PSR) for our tiles: LOLA LPSR 20 m, from PGDA. Fills tile.psr_mask (was all False / UNKNOWN).

    python -I -m worldmodel.rover.psr                       # every centre-window tile lola5m_site*, rewrites the tile in place

LPSR_80S_20MPP_ADJ.TIF (NASA PGDA, ~148 MB, 30400 x 30400 px, 20 m, same south-polar-stereographic frame as the 5 m products) is read as a small
WINDOW over HTTP, never downloaded whole. 1 = permanently shadowed. Upsampled 4x (nearest) to the 5 m tile grid.
Honest: PSR means DARK AND COLD (never sunlit in the model), not confirmed ice. Registration may be off by up to half a 20 m pixel (10 m).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from contracts.paths import data_root, tile_path
from contracts.tiles import Tile, load_tile, save_tile
from worldmodel.hazard.lola import raw_window_path

PSR_URL = "/vsicurl/https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LPSR_80S_20MPP_ADJ.TIF"
PSR_MPP = 20.0
SOURCE = "NASA LOLA LPSR 20 m permanent shadow (PGDA), upsampled 4x"


def tile_centre_xy(tile: Tile) -> tuple[float, float]:
    """South-polar-stereographic (x, y) in metres of the tile centre, from the raw window file saved by worldmodel.hazard.lola."""
    site = tile.extra.get("site")
    origin = tile.extra.get("window_origin")
    origin = None if origin in (None, "centre") else (int(origin[0]), int(origin[1]))
    p = raw_window_path(site, origin)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: need the raw LOLA window to know where {tile.region} is (python -I -m worldmodel.hazard.lola fetch)")
    cx, cy = np.load(p)["centre_xy_m"]
    return float(cx), float(cy)


def read_psr_window(cx: float, cy: float, half_m: float, retries: int = 5) -> np.ndarray:
    """PSR window [n, n] (n = 2*half_m/20) around (cx, cy), uint8 0/1."""
    import rasterio
    from rasterio.windows import from_bounds

    last = None
    for attempt in range(retries):
        try:
            env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif", GDAL_HTTP_TIMEOUT="120")
            with env, rasterio.open(PSR_URL) as r:
                w = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, r.transform)
                a = r.read(1, window=w, boundless=False)
            return (np.nan_to_num(a) > 0.5).astype(np.uint8)
        except Exception as e:  # noqa: BLE001 — network flakiness: retry
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not read the PSR window: {last}")


def attach_psr(tile: Tile) -> Tile:
    """Return the tile with psr_mask filled from NASA's PSR map."""
    cx, cy = tile_centre_xy(tile)
    h, w = tile.shape
    half = max(h, w) * tile.meters_per_pixel / 2.0
    small = read_psr_window(cx, cy, half)
    f = round(PSR_MPP / tile.meters_per_pixel)
    big = np.kron(small, np.ones((f, f), dtype=np.uint8)).astype(bool)
    out = np.zeros(tile.shape, dtype=bool)
    hh, ww = min(h, big.shape[0]), min(w, big.shape[1])
    r0, c0 = (h - hh) // 2, (w - ww) // 2
    b0, b1 = (big.shape[0] - hh) // 2, (big.shape[1] - ww) // 2
    out[r0:r0 + hh, c0:c0 + ww] = big[b0:b0 + hh, b1:b1 + ww]
    tile.psr_mask = out
    tile.extra = {**tile.extra, "psr_mask": SOURCE, "psr_fraction": float(out.mean()),
                  "psr_note": "PSR = permanently shadowed (dark and cold), NOT confirmed ice"}
    return tile


def default_regions() -> list[str]:
    d = data_root() / "tiles"
    return sorted(p.stem for p in d.glob("lola5m_*.npz") if "_gen" not in p.stem) if d.exists() else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=None)
    a = ap.parse_args(argv)
    for region in a.regions or default_regions():
        t = attach_psr(load_tile(tile_path(region)))
        save_tile(t, tile_path(region))
        print(f"{region}: PSR {t.psr_mask.mean():.2%} of the tile ({int(t.psr_mask.sum())} px) [{SOURCE}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
