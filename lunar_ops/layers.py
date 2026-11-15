#
#  Copyright (c) 2026, Adinfinitium team
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
"""Data layers for the Wall terrain view: what the terrain is, what the AI models say about it, and the light over the day.

Layers (all on the tile grid, row 0 = north edge):
  height, slope          NASA LRO LOLA (OBSERVED height; slope derived)
  hazard                 the hazard U-Net's probability when its checkpoint is present, else the slope-rule physics
                         backup (worldmodel.rover.world.hazard_probability); ``hazard_source`` says which one
  safe                   cells where a lander may aim: no hazard within 3 cells and gentle slope at footpad scale
                         (same rule as basilisk_mods.mission.pick_target)
  psr, ice               permanently shadowed regions and the ice-EVIDENCE score (MODELLED; never "ice is here")
  sun[t], earth[t]       Sun above the local horizon / Earth above the local horizon, every ``STEP_MIN`` minutes over
                         ``HOURS`` hours from the epoch: SPICE DE440 directions (worldmodel.light.ephemeris) against the
                         per-pixel terrain horizon of this tile (lunar_ops.world). The light model replaces this when loaded.
Cached in <data>/layer_cache/<region>_<epoch>.npz.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STEP_MIN = 30  # [min] light layer time step
HOURS = 48  # [h] light layer span from the epoch


def _cache_path(region: str, epoch_utc: str) -> Path:
    return REPO / "data" / "layer_cache" / f"{region}_{epoch_utc.replace(':', '').replace('-', '')}.npz"


def build_layers(region: str, epoch_utc: str = "2027-03-02T00:00:00") -> dict:
    """Compute (or load) every layer for ``region``. Returns numpy arrays and labels."""
    p = _cache_path(region, epoch_utc)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    from contracts.mission import load_mission
    from contracts.paths import tile_path
    from contracts.tiles import load_tile
    from lunar_ops.config import MissionConfig as OpsConfig
    from lunar_ops.world import World
    from worldmodel.light.ephemeris import grid_azimuth, sun_earth_directions
    from worldmodel.rover.world import hazard_probability

    cfg = load_mission()
    tile = load_tile(tile_path(region))
    world = World(OpsConfig(region=region))
    hazard, hazard_src = hazard_probability(tile, cfg)
    safe = safe_cells(tile, hazard, cfg.lander.touchdown_limits.max_slope_deg)

    t0 = datetime.fromisoformat(epoch_utc).replace(tzinfo=timezone.utc)
    times = [(t0 + timedelta(minutes=STEP_MIN * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(HOURS * 60 // STEP_MIN + 1)]
    lon_pm = (tile.center_lon_deg + 180.0) % 360.0 - 180.0
    d = sun_earth_directions(tile.center_lat_deg, lon_pm, times)
    sun_gaz, earth_gaz = grid_azimuth(np.asarray(d["sun_az"]), lon_pm), grid_azimuth(np.asarray(d["earth_az"]), lon_pm)
    sun = np.stack([world._horizon_field(float(a)) < float(e) for a, e in zip(sun_gaz, d["sun_el"])])
    earth = np.stack([world._horizon_field(float(a)) < float(e) for a, e in zip(earth_gaz, d["earth_el"])])
    out = {"region": region, "epoch_utc": epoch_utc, "times": np.array(times), "mpp": float(tile.meters_per_pixel),
           "center_lat_deg": float(tile.center_lat_deg), "center_lon_deg": float(tile.center_lon_deg),
           "height": tile.height.astype(np.float32), "slope": tile.slope.astype(np.float32), "hazard": hazard.astype(np.float32),
           "hazard_source": hazard_src, "safe": safe, "psr": world.psr.astype(bool), "ice": world.ice_score.astype(np.float32),
           "sun": sun, "earth": earth, "sun_el": np.asarray(d["sun_el"], np.float32), "earth_el": np.asarray(d["earth_el"], np.float32),
           "sun_az_grid": np.asarray(sun_gaz, np.float32), "earth_az_grid": np.asarray(earth_gaz, np.float32)}
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    return out


def safe_cells(tile, hazard: np.ndarray, max_slope_deg: float, margin_px: int = 3) -> np.ndarray:
    """No hazard (>= 0.5) within ``margin_px`` and every slope within 2 cells under half the touchdown slope limit."""
    from numpy.lib.stride_tricks import sliding_window_view

    k = 2 * margin_px + 1
    clear = ~sliding_window_view(np.pad(hazard >= 0.5, margin_px, constant_values=True), (k, k)).any(axis=(-1, -2))
    flat = sliding_window_view(np.pad(tile.slope, 2, mode="edge"), (5, 5)).max(axis=(-1, -2)) < 0.5 * max_slope_deg
    return clear & flat


def layers_message(region: str, epoch_utc: str = "2027-03-02T00:00:00") -> dict:
    """JSON-ready layers (uint8 / packed bits, base64)."""
    L = build_layers(region, epoch_utc)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    h = L["height"]
    return {"type": "layers", "region": str(L["region"]), "epoch_utc": str(L["epoch_utc"]), "rows": int(h.shape[0]), "cols": int(h.shape[1]),
            "mpp": float(L["mpp"]), "center_lat_deg": float(L["center_lat_deg"]), "center_lon_deg": float(L["center_lon_deg"]),
            "height_f32": b64(h.astype("<f4")), "slope_u8": b64(np.clip(np.rint(L["slope"] * 4), 0, 255).astype(np.uint8)),  # quarter degrees
            "hazard_u8": b64(np.clip(np.rint(L["hazard"] * 255), 0, 255).astype(np.uint8)), "hazard_source": str(L["hazard_source"]),
            "safe_bits": b64(np.packbits(L["safe"])), "psr_bits": b64(np.packbits(L["psr"])),
            "ice_u8": b64(np.clip(np.rint(L["ice"] * 255), 0, 255).astype(np.uint8)),
            "times": [str(t) for t in L["times"]], "sun_bits": b64(np.packbits(L["sun"].reshape(L["sun"].shape[0], -1), axis=1)),
            "earth_bits": b64(np.packbits(L["earth"].reshape(L["earth"].shape[0], -1), axis=1)),
            "sun_el": [round(float(x), 3) for x in L["sun_el"]], "earth_el": [round(float(x), 3) for x in L["earth_el"]],
            "provenance": {"height": "OBSERVED (NASA LRO LOLA)", "hazard": f"MODELLED ({L['hazard_source']})",
                           "light": "MODELLED (SPICE DE440 directions + LOLA terrain horizon)", "ice": "MODELLED ice EVIDENCE, not confirmed ice"}}


if __name__ == "__main__":
    import sys

    m = layers_message(sys.argv[1] if len(sys.argv) > 1 else "lola5m_site01")
    print(json.dumps({k: (v if not isinstance(v, str) or len(v) < 80 else f"<{len(v)} chars>") for k, v in m.items() if k not in ("times", "sun_el", "earth_el")}, indent=1))
