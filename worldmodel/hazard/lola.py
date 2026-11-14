"""Fetch real LOLA 5 m south-pole data (NASA PGDA) and convert it into the official tile format.

Source: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/  (Barker et al. 2021). Each site has 5 m/pixel
south-polar-stereographic GeoTIFFs. We read only a WINDOW of each file over HTTP (the files are
70 MB each and the download is slow), save the raw window under data/raw/lola5mpp/<Site>/, then
write data/tiles/lola5m_<site>.npz:

    height       <- *_surf.tif    surface height (m); empty pixels were interpolated by LOLA
    slope        <- *_slp.tif     OFFICIAL LOLA slope (deg)      (the label source for training)
    uncertainty  <- *_toterr.tif  total Z uncertainty (m, RMS)
    real_mask    <- *_ldec.tif > 0   a LOLA laser return exists in the pixel (not interpolated)
    hazard       <- backup: official slope >= max_slope_deg (modelled)
    psr_mask     <- all False (PSR atlas not loaded yet; UNKNOWN, see header extra)

    python -I -m worldmodel.hazard.lola fetch --sites Site23 Site01 --size 1024
    python -I -m worldmodel.hazard.lola build --sites Site23 Site01
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from contracts.mission import load_mission
from contracts.paths import data_root, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, hazard_from_slope, save_tile

BASE_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp"
LAYERS = {"surf": "surf", "slp": "slp", "ldec": "ldec", "toterr": "toterr"}
MPP = 5.0
MOON_RADIUS_M = None  # taken from config at use time

# Training sites and the hold-out. Site20v2 is an extension of Site20 and is NOT used (leakage).
TRAIN_SITES = ["Site01", "Site04", "Site06", "Site07", "Site11", "Site20"]
HOLDOUT_SITES = ["Site23"]  # Malapert massif: never trained on, never tuned on
SITE_NAMES = {"Site01": "Connecting ridge", "Site04": "Shackleton rim", "Site06": "Nobile rim 1",
              "Site07": "Peak near Shackleton", "Site11": "de Gerlache rim",
              "Site20": "Leibnitz beta plateau", "Site23": "Malapert massif"}


def region_name(site: str) -> str:
    return f"lola5m_{site.lower()}"


def raw_dir(site: str) -> Path:
    return data_root() / "raw" / "lola5mpp" / site


def raw_window_path(site: str) -> Path:
    return raw_dir(site) / "window.npz"


def layer_url(site: str, layer: str) -> str:
    return f"/vsicurl/{BASE_URL}/{site}/{site}_final_adj_5mpp_{LAYERS[layer]}.tif"


def _read_window(site: str, layer: str, size: int, retries: int = 4):
    import rasterio
    from rasterio.windows import Window

    last = None
    for attempt in range(retries):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                              GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="3"):
                with rasterio.open(layer_url(site, layer)) as r:
                    size_c = min(size, r.width, r.height)
                    c0, r0 = (r.width - size_c) // 2, (r.height - size_c) // 2
                    arr = r.read(1, window=Window(c0, r0, size_c, size_c)).astype(np.float32)
                    # Pixel-centre coordinates of the window's centre (GMT pixel-registered, metres).
                    cx = r.bounds.left + (c0 + size_c / 2) * r.res[0]
                    cy = r.bounds.top - (r0 + size_c / 2) * r.res[1]
                    return layer, arr, (cx, cy), r.res[0]
        except Exception as e:  # noqa: BLE001 — network flakiness: retry, then raise
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not read {site}/{layer}: {last}")


def fetch_site(site: str, size: int = 1024, workers: int = 4) -> Path:
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda k: _read_window(site, k, size), LAYERS))
    arrays = {k: a for k, a, _, _ in results}
    (cx, cy), res = results[0][2], results[0][3]
    if abs(res - MPP) > 1e-6:
        raise ValueError(f"{site}: expected {MPP} m/pixel, got {res}")
    p = raw_window_path(site)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, centre_xy_m=np.array([cx, cy]), **arrays)
    print(f"fetched {site} {arrays['surf'].shape} in {time.time() - t0:.0f}s -> {p}", flush=True)
    return p


def xy_to_latlon(x_m: float, y_m: float, radius_m: float) -> tuple[float, float]:
    """South-polar stereographic (true scale at the pole) -> (lat_deg, lon_deg east, 0..360).
    Convention x = r sin(lon), y = r cos(lon) is the LOLA one but NOT yet verified (Imok KAN-21)."""
    r = math.hypot(x_m, y_m)
    colat = 2.0 * math.atan(r / (2.0 * radius_m))
    return -90.0 + math.degrees(colat), math.degrees(math.atan2(x_m, y_m)) % 360.0


def build_tile(site: str) -> Tile:
    cfg = load_mission()
    z = np.load(raw_window_path(site), allow_pickle=False)
    surf, slp, ldec, err = (z[k].astype(np.float32) for k in ("surf", "slp", "ldec", "toterr"))
    bad = ~(np.isfinite(surf) & np.isfinite(slp) & np.isfinite(err))
    if bad.any():
        raise ValueError(f"{site}: {bad.mean():.2%} non-finite pixels in the window; pick another window")
    cx, cy = (float(v) for v in z["centre_xy_m"])
    lat, lon = xy_to_latlon(cx, cy, cfg.moon.radius_m)
    limit = cfg.lander.touchdown_limits.max_slope_deg
    meta = Meta(
        provenance=Provenance.OBSERVED,
        source=(f"LOLA PGDA 5 m/pixel {site} ({SITE_NAMES.get(site, site)}), Barker et al. 2021; "
                f"centre window {surf.shape[1]}x{surf.shape[0]}; height partly interpolated by LOLA"),
        frame="MOON_ME_DE421 south polar stereographic, metres (tile centre = window centre)",
        created_by="worldmodel.hazard.lola",
        units={"height": "m", "slope": "deg (official LOLA slp)", "uncertainty": "m (LOLA toterr, RMS)"},
    )
    tile = Tile(
        region=region_name(site), meters_per_pixel=MPP, height=surf, slope=np.clip(slp, 0, 90),
        hazard=hazard_from_slope(np.clip(slp, 0, 90), limit), uncertainty=err,
        psr_mask=np.zeros(surf.shape, dtype=bool), meta=meta, center_lat_deg=lat, center_lon_deg=lon,
        real_mask=ldec > 0,
        extra={"site": site, "hazard_source": f"backup: official slope >= {limit} deg (modelled)",
               "psr_mask": "UNKNOWN: LROC/LOLA PSR atlas not loaded, all False",
               "laser_coverage": float((ldec > 0).mean()),
               "projection_check": "lat/lon from stereographic formula; convention unverified (KAN-21)"},
    )
    tile.validate()
    return tile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fetch", "build", "all"])
    ap.add_argument("--sites", nargs="+", default=HOLDOUT_SITES + TRAIN_SITES)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args(argv)
    if a.action in ("fetch", "all"):
        for s in a.sites:
            fetch_site(s, a.size)
    if a.action in ("build", "all"):
        for s in a.sites:
            t = build_tile(s)
            p = save_tile(t, tile_path(t.region))
            print(f"wrote {p.name}: {t.shape} lat {t.center_lat_deg:.2f} lon {t.center_lon_deg:.1f} "
                  f"laser {t.real_mask.mean():.1%} slope>= limit {t.hazard.mean():.1%} [{t.meta.provenance.banner}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
