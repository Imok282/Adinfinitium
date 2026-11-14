"""The rover's world: real LOLA terrain, NASA PSR ice targets, real sunlight (SPICE + exact skyline) for every coarse cell over a lunar month.

build_world(region, date) -> RoverWorld, cached on disk. The world is the central `rover_sim.world_px` square of a 5 m tile (rays stay inside the tile).
Sunlight/Earth fields are on a coarse grid (`light_stride_px` px). Provenance: terrain OBSERVED (LOLA, mostly interpolated), light MODELLED
(NASA SPICE DE440 + exact skyline out to ~1.3 km), ice targets = NASA PSR edges (PSR = dark and cold, NOT confirmed ice). No made-up terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.mission import MissionConfig, load_mission
from contracts.paths import data_root, model_path, tile_path
from contracts.tiles import Tile, load_tile

N_HOURS = 709


@dataclass
class RoverWorld:
    region: str
    date: str
    mpp: float
    size: int
    origin_rc: tuple[int, int]       # world pixel (0,0) in the full tile
    height: np.ndarray               # [S,S] m
    slope: np.ndarray                # [S,S] deg (official LOLA)
    hazard: np.ndarray               # [S,S] P(unsafe): U-Net if a checkpoint exists, else the slope-rule backup
    hazard_source: str
    psr: np.ndarray                  # [S,S] bool
    sun_frac: np.ndarray             # [T,Sc,Sc] float16, visible fraction of the solar disk
    earth_up: np.ndarray             # [T,Sc,Sc] uint8, Earth centre above the skyline
    times_utc: list[str]
    lander_rc: tuple[float, float]   # world pixels
    targets_rc: np.ndarray           # [K,2] ice (PSR-edge) targets, world pixels
    ice_source: str


def hazard_probability(tile: Tile, cfg: MissionConfig) -> tuple[np.ndarray, str]:
    """What the rover believes is unsafe: the hazard U-Net (CPU, cached) if trained, else the slope-rule backup on sparse input."""
    from worldmodel.hazard.backup import backup_hazard

    ckpt = model_path("hazard_unet")
    if not ckpt.exists():
        return backup_hazard(tile, cfg).astype(np.float32), "slope-rule backup"
    cache = data_root() / "rover_cache" / f"{tile.region}_hazard_{ckpt.stat().st_mtime_ns}.npy"
    if cache.exists():
        return np.load(cache), f"hazard U-Net {ckpt.name}"
    import torch

    from worldmodel.hazard.backup import slope_limit_deg
    from worldmodel.hazard.features import build_features
    from worldmodel.hazard.model import load_checkpoint, predict_probability

    model, _ = load_checkpoint(ckpt, torch.device("cpu"))
    feats, _ = build_features(tile, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
    p = predict_probability(model, feats, torch.device("cpu")).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, p)
    return p, f"hazard U-Net {ckpt.name}"


def choose_targets(psr: np.ndarray, slope: np.ndarray, lander_rc: tuple[float, float], mpp: float, rover_slope_limit: float, n: int,
                   rng: np.random.Generator, min_dist_m: float = 150.0) -> tuple[np.ndarray, str]:
    """Ice targets = PSR cells on the shadow EDGE (the rover dips into the dark edge), reachable slope, at least min_dist from the lander."""
    from scipy.ndimage import binary_dilation, binary_erosion

    s = psr.shape[0]
    edge = psr & ~binary_erosion(psr, iterations=3)
    near_dark = binary_dilation(psr, iterations=3) & ~psr
    cand = (edge | near_dark) & (slope <= rover_slope_limit)
    rr, cc = np.nonzero(cand)
    margin = 12
    ok = (rr > margin) & (rr < s - margin) & (cc > margin) & (cc < s - margin)
    rr, cc = rr[ok], cc[ok]
    d = np.hypot(rr - lander_rc[0], cc - lander_rc[1]) * mpp
    keep = d >= min_dist_m
    rr, cc = rr[keep], cc[keep]
    if len(rr) == 0:
        # No reachable PSR edge in this world: UNKNOWN science targets (never reported as ice).
        ang = rng.uniform(0, 2 * np.pi, n)
        dist = rng.uniform(300, 900, n) / mpp
        t = np.stack([lander_rc[0] - np.cos(ang) * dist, lander_rc[1] + np.sin(ang) * dist], axis=1)
        return np.clip(t, margin, s - margin - 1), "UNKNOWN (no reachable PSR edge in this world: plain science targets)"
    pick = rng.choice(len(rr), size=min(n, len(rr)), replace=False)
    return np.stack([rr[pick], cc[pick]], axis=1).astype(np.float64), "NASA LOLA PSR edge (dark and cold; NOT confirmed ice)"


def build_light(tile: Tile, date: str, cfg: MissionConfig, origin_rc: tuple[int, int], size: int, device=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from worldmodel.light.ephemeris import grid_azimuth, sun_earth_directions
    from worldmodel.light.make_points import time_axis
    from worldmodel.light.physics import light_from_skyline
    from worldmodel.light.teacher import raycast_skyline_torch

    rs, lm = cfg.rover_sim, cfg.light_model
    stride = rs.light_stride_px
    n = size // stride
    rows = origin_rc[0] + stride * np.arange(n) + (stride - 1) / 2.0
    cols = origin_rc[1] + stride * np.arange(n) + (stride - 1) / 2.0
    cells = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(-1, 2)
    sky = raycast_skyline_torch(tile.height, tile.meters_per_pixel, cells, cfg.moon.radius_m, lm.max_ray_range_m, lm.n_azimuth, device=device)
    times = time_axis(date)
    lon_pm = (tile.center_lon_deg + 180.0) % 360.0 - 180.0
    d = sun_earth_directions(tile.center_lat_deg, lon_pm, times)
    out = light_from_skyline(sky, grid_azimuth(d["sun_az"], lon_pm), d["sun_el"], grid_azimuth(d["earth_az"], lon_pm), d["earth_el"], lm.sun_angular_radius_deg)
    sun = out["sun_fraction"].reshape(len(times), n, n).astype(np.float16)
    earth = out["earth_visible"].reshape(len(times), n, n).astype(np.uint8)
    return sun, earth, times


def build_world(region: str, date: str, cfg: MissionConfig | None = None, seed: int = 0, device=None, cache: bool = True) -> RoverWorld:
    cfg = cfg or load_mission()
    rs = cfg.rover_sim
    tile = load_tile(tile_path(region))
    s = rs.world_px
    r0, c0 = (tile.shape[0] - s) // 2, (tile.shape[1] - s) // 2
    sl = (slice(r0, r0 + s), slice(c0, c0 + s))
    haz, haz_src = hazard_probability(tile, cfg)
    cache_p = data_root() / "rover_cache" / f"{region}_{date}_light_s{rs.light_stride_px}_w{s}.npz"
    if cache and cache_p.exists():
        z = np.load(cache_p)
        sun, earth, times = z["sun"], z["earth"], list(z["times"])
    else:
        sun, earth, times = build_light(tile, date, cfg, (r0, c0), s, device)
        if cache:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_p, sun=sun, earth=earth, times=np.array(times))
    lander = ((s - 1) / 2.0, (s - 1) / 2.0)
    targets, ice_src = choose_targets(tile.psr_mask[sl], tile.slope[sl], lander, tile.meters_per_pixel, cfg.rover.max_slope_deg,
                                      rs.n_ice_targets, np.random.default_rng(seed))
    return RoverWorld(region, date, tile.meters_per_pixel, s, (r0, c0), tile.height[sl].astype(np.float32), tile.slope[sl].astype(np.float32),
                      haz[sl].astype(np.float32), haz_src, tile.psr_mask[sl].copy(), sun, earth, times, lander, targets, ice_src)
