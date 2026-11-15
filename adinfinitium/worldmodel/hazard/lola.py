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
    psr_mask     <- all False (UNKNOWN, see header extra) unless worldmodel.rover.psr already attached one: a rebuild keeps it

    python -I -m worldmodel.hazard.lola fetch --sites Site23 Site01 --size 1024
    python -I -m worldmodel.hazard.lola build --sites Site23 Site01
    python -I -m worldmodel.hazard.lola all --grid --sites Site01       # + every non-overlapping 1024 px window of the site
    python -I -m worldmodel.hazard.lola all --grid --workers 3          # ONE process, all sites, at most 3 NASA connections
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
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
TRAIN_SITES = ["Site01", "Site04", "Site06", "Site07", "Site20", "Haworth", "Shoemaker", "DM2"]
VAL_SITES = ["Site11"]
HOLDOUT_SITES = ["Site23", "Site42"]  # never trained on, never tuned on (Malapert massif; de Gerlache-Kocher massif)
SITE_NAMES = {"Site01": "Connecting ridge", "Site04": "Shackleton rim", "Site06": "Nobile rim 1",
              "Site07": "Peak near Shackleton", "Site11": "de Gerlache rim", "Site20": "Leibnitz beta plateau",
              "Site23": "Malapert massif", "Site42": "de Gerlache-Kocher massif", "Haworth": "Haworth crater",
              "Shoemaker": "Shoemaker crater", "DM2": "Nobile rim 2"}
ALL_SITES = HOLDOUT_SITES + VAL_SITES + TRAIN_SITES
# Download order: hold-out and validation first, then the sites missing from the seed data, so a timeout leaves none empty.
SITE_PRIORITY = ["Site42", "Site23", "Site11", "Haworth", "Shoemaker", "DM2"]
WINDOW = 1024
BAND_ROWS = 128


def region_name(site: str, origin: tuple[int, int] | None = None) -> str:
    """Centre window: lola5m_<site>. Grid window at pixel origin (row, col): lola5m_<site>_r<row/1024>c<col/1024>."""
    base = f"lola5m_{site.lower()}"
    return base if origin is None else f"{base}_r{origin[0] // WINDOW}c{origin[1] // WINDOW}"


def regions_for_site(site: str) -> list[str]:
    """Every built tile of a site (centre window + grid windows), sorted; excludes generated versions (*_gen*)."""
    d = data_root() / "tiles"
    base = f"lola5m_{site.lower()}"
    names = [base] if (d / f"{base}.npz").exists() else []
    names += sorted(p.stem for p in d.glob(f"{base}_r*c*.npz")) if d.exists() else []
    if site in TRAIN_SITES or site in VAL_SITES:
        # Lazy import: ops.spatial_split imports this module. Test sites are never filtered.
        from ops.spatial_split import active_exclusions

        skip = active_exclusions()
        names = [n for n in names if n not in skip]
    return names


def raw_dir(site: str) -> Path:
    return data_root() / "raw" / "lola5mpp" / site


def raw_window_path(site: str, origin: tuple[int, int] | None = None) -> Path:
    return raw_dir(site) / ("window.npz" if origin is None else f"window_r{origin[0] // WINDOW}c{origin[1] // WINDOW}.npz")


def layer_url(site: str, layer: str) -> str:
    return f"/vsicurl/{BASE_URL}/{site}/{site}_final_adj_5mpp_{LAYERS[layer]}.tif"


def _read_window(site: str, layer: str, size: int, retries: int = 6, origin: tuple[int, int] | None = None):
    import rasterio
    from rasterio.windows import Window

    last = None
    for attempt in range(retries):
        try:
            env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                               GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="3")
            with env, rasterio.open(layer_url(site, layer)) as r:
                size_c = min(size, r.width, r.height)
                if origin is None:
                    c0, r0 = (r.width - size_c) // 2, (r.height - size_c) // 2
                else:
                    r0, c0 = origin
                    if r0 + size_c > r.height or c0 + size_c > r.width:
                        raise ValueError(f"window {origin} outside {site} ({r.height}x{r.width})")
                # row bands keep each HTTP range request small: one 1024-row read of the 5960 px wide Haworth file failed 6 times in a row
                arr = np.concatenate([r.read(1, window=Window(c0, r0 + k, size_c, min(BAND_ROWS, size_c - k)))
                                      for k in range(0, size_c, BAND_ROWS)]).astype(np.float32)
                # Pixel-centre coordinates of the window's centre (GMT pixel-registered, metres).
                cx = r.bounds.left + (c0 + size_c / 2) * r.res[0]
                cy = r.bounds.top - (r0 + size_c / 2) * r.res[1]
                return layer, arr, (cx, cy), r.res[0]
        except Exception as e:  # noqa: BLE001 — network flakiness: retry, then raise
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not read {site}/{layer}: {last}")


def site_dims(site: str, retries: int = 6) -> tuple[int, int]:
    """(height, width) in pixels of a site's 5 m GeoTIFFs."""
    import rasterio

    last = None
    for attempt in range(retries):
        try:
            env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="120")
            with env, rasterio.open(layer_url(site, "surf")) as r:
                return r.height, r.width
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not open {site}: {last}")


def grid_origins(site: str, size: int = WINDOW) -> list[tuple[int, int]]:
    """Non-overlapping grid windows (multiples of `size`) that do NOT overlap the centre window, so no pixel is used twice."""
    h, w = site_dims(site)
    rc, cc = (h - size) // 2, (w - size) // 2
    out = []
    for r in range(0, h - size + 1, size):
        for c in range(0, w - size + 1, size):
            if abs(r - rc) < size and abs(c - cc) < size:
                continue
            out.append((r, c))
    return out


def fetch_site(site: str, size: int = 1024, workers: int = 3, origin: tuple[int, int] | None = None) -> Path:
    """One window, 4 layers; `workers` threads = the most NASA connections open at once."""
    t0 = time.time()
    with ThreadPoolExecutor(max(1, min(workers, len(LAYERS)))) as ex:
        results = list(ex.map(lambda k: _read_window(site, k, size, origin=origin), LAYERS))
    arrays = {k: a for k, a, _, _ in results}
    (cx, cy), res = results[0][2], results[0][3]
    if abs(res - MPP) > 1e-6:
        raise ValueError(f"{site}: expected {MPP} m/pixel, got {res}")
    p = raw_window_path(site, origin)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename: a kill mid-write must not leave a truncated window that later counts as "have".
    tmp = p.with_name(p.stem + ".part.npz")
    np.savez_compressed(tmp, centre_xy_m=np.array([cx, cy]), **arrays)
    os.replace(tmp, p)
    print(f"fetched {site} {origin or 'centre'} {arrays['surf'].shape} in {time.time() - t0:.0f}s -> {p.name}", flush=True)
    return p


def xy_to_latlon(x_m: float, y_m: float, radius_m: float) -> tuple[float, float]:
    """South-polar stereographic (true scale at the pole) -> (lat_deg, lon_deg east, 0..360).
    Convention x = r sin(lon), y = r cos(lon) is the LOLA one but NOT yet verified (Imok KAN-21)."""
    r = math.hypot(x_m, y_m)
    colat = 2.0 * math.atan(r / (2.0 * radius_m))
    return -90.0 + math.degrees(colat), math.degrees(math.atan2(x_m, y_m)) % 360.0


def _existing_psr(region: str, shape: tuple[int, ...]) -> tuple[np.ndarray, dict] | None:
    """(psr_mask, psr_* header keys) of the tile already on disk when worldmodel.rover.psr attached a real mask, else None."""
    p = tile_path(region)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            extra = json.loads(str(z["header"])).get("extra", {})
            mask = z["psr_mask"].astype(bool)
    except Exception:  # noqa: BLE001 — an unreadable old tile has nothing to keep; the rebuild replaces it
        return None
    if "psr_fraction" not in extra and str(extra.get("psr_mask", "UNKNOWN")).startswith("UNKNOWN"):
        return None
    if mask.shape != tuple(shape):
        return None
    return mask, {k: v for k, v in extra.items() if k.startswith("psr_")}


def build_tile(site: str, origin: tuple[int, int] | None = None) -> Tile:
    cfg = load_mission()
    z = np.load(raw_window_path(site, origin), allow_pickle=False)
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
                f"{'centre' if origin is None else 'grid'} window {surf.shape[1]}x{surf.shape[0]}; height partly interpolated by LOLA"),
        frame="MOON_ME_DE421 south polar stereographic, metres (tile centre = window centre)",
        created_by="worldmodel.hazard.lola",
        units={"height": "m", "slope": "deg (official LOLA slp)", "uncertainty": "m (LOLA toterr, RMS)"},
    )
    tile = Tile(
        region=region_name(site, origin), meters_per_pixel=MPP, height=surf, slope=np.clip(slp, 0, 90),
        hazard=hazard_from_slope(np.clip(slp, 0, 90), limit), uncertainty=err,
        psr_mask=np.zeros(surf.shape, dtype=bool), meta=meta, center_lat_deg=lat, center_lon_deg=lon,
        real_mask=ldec > 0,
        extra={"site": site, "window_origin": list(origin) if origin else "centre", "hazard_source": f"backup: official slope >= {limit} deg (modelled)",
               "psr_mask": "UNKNOWN: LROC/LOLA PSR atlas not loaded, all False",
               "laser_coverage": float((ldec > 0).mean()), "centre_xy_m": [cx, cy],
               "projection_check": "lat/lon from stereographic formula; convention unverified (KAN-21)"},
    )
    # Same window, same ground: a rebuild must not wipe the PSR mask the psr stage attached.
    kept = _existing_psr(tile.region, surf.shape)
    if kept is not None:
        tile.psr_mask, tile.extra = kept[0], {**tile.extra, **kept[1]}
    tile.validate()
    return tile


def _ordered_sites(sites: list[str]) -> list[str]:
    """Sites in download priority order (SITE_PRIORITY, then ALL_SITES order, then unknown names), whatever order --sites gave."""
    order = SITE_PRIORITY + [s for s in ALL_SITES if s not in SITE_PRIORITY]
    return sorted(dict.fromkeys(sites), key=lambda s: order.index(s) if s in order else len(order))


def _grid_targets(sites: list[str], limit: int) -> tuple[list[tuple[str, tuple[int, int]]], list[str]]:
    """Grid windows round-robin across sites, plus the sites whose size could not be read (they keep only their centre window)."""
    per_site, unresolved = [], []
    for s in sites:
        try:
            origins = grid_origins(s)
        except Exception as e:  # noqa: BLE001 — one unreachable site must not stop the others
            print(f"FAILED site {s}: {e}", flush=True)
            unresolved.append(s)
            continue
        per_site.append([(s, o) for o in (origins[:limit] if limit else origins)])
    return [t for row in itertools.zip_longest(*per_site) for t in row if t is not None], unresolved


def _process(s: str, o: tuple[int, int] | None, a: argparse.Namespace, st: dict[str, int]) -> None:
    """One window: fetch if missing, then build and save its tile straight away, so a stage timeout keeps every finished tile."""
    st["wanted"] += 1
    raw, fetched = raw_window_path(s, o), False
    if a.action != "build":
        if raw.exists() and not a.overwrite:
            print(f"have {s} {o or 'centre'}", flush=True)
        else:
            try:
                fetch_site(s, a.size, workers=a.workers, origin=o)
            except Exception as e:  # noqa: BLE001 — one bad window must not stop the rest
                print(f"FAILED {s} {o}: {e}", flush=True)
                st["failed"] += 1
                return
            fetched = True
            st["fetched"] += 1
    if a.action == "fetch" or not raw.exists():
        return
    if a.action == "all" and not fetched and not a.overwrite and tile_path(region_name(s, o)).exists():
        st["kept"] += 1
        return
    try:
        t = build_tile(s, o)
        p = save_tile(t, tile_path(t.region))
    except Exception as e:  # noqa: BLE001 — a bad or unreadable window must not stop the rest
        print(f"{'skipped' if isinstance(e, ValueError) else 'FAILED'} {s} {o}: {e}", flush=True)
        st["failed"] += 1
        return
    st["built"] += 1
    print(f"wrote {p.name}: {t.shape} lat {t.center_lat_deg:.2f} lon {t.center_lon_deg:.1f} "
          f"laser {t.real_mask.mean():.1%} slope>= limit {t.hazard.mean():.1%} [{t.meta.provenance.banner}]", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fetch", "build", "all"])
    ap.add_argument("--sites", nargs="+", default=ALL_SITES)
    ap.add_argument("--size", type=int, default=WINDOW)
    ap.add_argument("--grid", action="store_true", help="also every non-overlapping grid window of each site (many more tiles)")
    ap.add_argument("--no-centre", action="store_true", help="skip the centre window (grid windows only)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many grid windows per site (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="refetch windows that are already on disk (and rebuild their tiles)")
    ap.add_argument("--workers", type=int, default=3, help="most NASA connections open at once in this process (run ONE process)")
    a = ap.parse_args(argv)
    sites = _ordered_sites(a.sites)
    stats = {s: dict.fromkeys(("wanted", "fetched", "built", "kept", "failed"), 0) for s in sites}
    done: list[tuple[str, tuple[int, int] | None]] = []
    # Centre windows of every site first, and before any site size is read: an unreachable site costs only its own windows.
    for s, o in [(s, None) for s in sites if not a.no_centre]:
        _process(s, o, a, stats[s])
        done.append((s, o))
    grid, unresolved = _grid_targets(sites, a.limit) if a.grid else ([], [])
    streak = dict.fromkeys(sites, 0)     # each failed window costs ~7 min of retries: stop a site after 2 in a row
    for s, o in grid:
        if streak[s] >= 2:
            if streak[s] == 2:
                print(f"FAILED site {s}: 2 windows in a row failed; its remaining grid windows are skipped in this run", flush=True)
                streak[s] = 3
            continue
        before = stats[s]["failed"]
        _process(s, o, a, stats[s])
        streak[s] = streak[s] + 1 if stats[s]["failed"] > before else 0
        done.append((s, o))
    for s in sites:
        st = stats[s]
        print(f"summary {s}: wanted {st['wanted']} fetched {st['fetched']} built {st['built']} failed {st['failed']} "
              f"(already built, left alone {st['kept']}){' GRID UNRESOLVED: centre window only' if s in unresolved else ''}", flush=True)
    have = raw_window_path if a.action == "fetch" else (lambda s, o: tile_path(region_name(s, o)))
    return 0 if any(have(s, o).exists() for s, o in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
