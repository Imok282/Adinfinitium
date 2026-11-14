"""REAL light point series from NASA data: SPICE Sun/Earth directions + the exact skyline of real LOLA terrain. Retires the analytic stand-in.

    python -I -m worldmodel.light.make_points                       # every centre-window tile lola5m_site*, 2 dates each
    python -I -m worldmodel.light.make_points --regions lola5m_site23 --dates 2027-03-01 2027-06-01

Writes data/light/<region>_<date>_points.npz in the light contract v2 format (contracts/light.py), provenance MODELLED:
real NASA ephemeris (SPICE DE440, checked against JPL Horizons) + exact geometry on real LOLA heights.
Honest limits: the skyline is ray-cast only out to `light_model.max_ray_range_m` inside the ~5 km LOLA window, so far mountains are missing and
the skyline is biased low; terrain heights are mostly interpolated by LOLA; grid azimuth uses the unverified x = r sin(lon) convention (KAN-21).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import numpy as np

from contracts.light import CADENCE_S, SYNODIC_SAMPLES, PointSeries, save_points
from contracts.mission import load_mission
from contracts.paths import data_root, points_path, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, load_tile
from worldmodel.hazard.lola import ALL_SITES, region_name
from worldmodel.light.ephemeris import grid_azimuth, sun_earth_directions
from worldmodel.light.physics import light_from_skyline
from worldmodel.light.teacher import raycast_skyline_numpy

RING_STEP_PX = 20
DATES = ("2027-03-01", "2027-06-01")


def time_axis(date: str, n: int = SYNODIC_SAMPLES) -> list[str]:
    t0 = datetime.fromisoformat(date).replace(tzinfo=UTC)
    return [(t0 + timedelta(seconds=CADENCE_S * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]


def ring_cells(tile: Tile) -> np.ndarray:
    """Candidate cell (tile centre) then its 8 neighbours at +-RING_STEP_PX."""
    h, w = tile.shape
    r0, c0 = (h - 1) // 2, (w - 1) // 2
    return np.array([(r0, c0)] + [(r0 + dr * RING_STEP_PX, c0 + dc * RING_STEP_PX) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])


def make_points(tile: Tile, date: str) -> PointSeries:
    cfg = load_mission()
    lm = cfg.light_model
    times = time_axis(date)
    cells = ring_cells(tile)
    skyline = raycast_skyline_numpy(tile.height, tile.meters_per_pixel, cells.astype(float), cfg.moon.radius_m, lm.max_ray_range_m, lm.n_azimuth)
    lon_pm = (tile.center_lon_deg + 180.0) % 360.0 - 180.0
    d = sun_earth_directions(tile.center_lat_deg, lon_pm, times)
    sun_az, earth_az = grid_azimuth(d["sun_az"], lon_pm), grid_azimuth(d["earth_az"], lon_pm)
    light = light_from_skyline(skyline, sun_az, d["sun_el"], earth_az, d["earth_el"], lm.sun_angular_radius_deg)
    meta = Meta(provenance=Provenance.MODELLED,
                source=(f"NASA SPICE DE440 (NAIF) Sun/Earth directions, topocentric; exact skyline ray-cast of {tile.meta.source} "
                        f"out to {lm.max_ray_range_m:g} m (window-limited)"),
                frame="local tile ENU (south polar stereographic, row 0 = +y); grid azimuth = true azimuth + east longitude (unverified)",
                created_by="worldmodel.light.make_points",
                units={"sun_elev_deg": "deg relative to the terrain horizon (>0: centre above skyline)", "earth_elev_deg": "same"})
    ps = PointSeries(region=tile.region, date=date, cells=cells, times_utc=times, et_s=d["et_s"],
                     sun_fraction=light["sun_fraction"].astype(np.float32), sun_elev_deg=light["sun_elev_rel_deg"].astype(np.float32),
                     earth_elev_deg=light["earth_elev_rel_deg"].astype(np.float32), sun_az_deg=sun_az.astype(np.float32),
                     earth_az_deg=earth_az.astype(np.float32), horizon_elev_deg=skyline.astype(np.float32), meta=meta)
    ps.validate()
    return ps


def default_regions() -> list[str]:
    """Centre-window tiles of every site that is on this machine (grid windows have no ring cells worth a Navigator file)."""
    d = data_root() / "tiles"
    return [region_name(s) for s in ALL_SITES if (d / f"{region_name(s)}.npz").exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--dates", nargs="+", default=list(DATES))
    a = ap.parse_args(argv)
    for region in a.regions or default_regions():
        tile = load_tile(tile_path(region))
        for date in a.dates:
            ps = make_points(tile, date)
            p = save_points(ps, points_path(region, date))
            print(f"wrote {p.name}: K={len(ps.cells)} T={len(ps.times_utc)} sun up {float((ps.sun_fraction[:, 0] > 0).mean()):.0%} "
                  f"Earth up {float(ps.earth_visible()[:, 0].mean()):.0%} [{ps.meta.provenance.banner}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
