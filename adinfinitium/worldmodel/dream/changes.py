"""The "something changed after training" tests. Each change is applied IN MEMORY to a real world or to the rover's physics; no data file is edited.

apply_change(world, cfg, name) -> (changed world, changed mission config, description). The original world and config are left untouched.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from contracts.mission import MissionConfig
from worldmodel.rover.world import RoverWorld

PANEL_FACTOR = 0.6
TIGHT_SLOPE_DEG = 15.0
RING_INNER_M, RING_OUTER_M = 90.0, 135.0      # wider than one hourly move (36 m), so it can't be stepped over; starts are within 60 m of the lander
RING_GAPS, RING_GAP_DEG = 3, 40.0
RING_SLOPE_DEG = 35.0
LATE_LIGHT_H = 12                             # true sunlight = forecast delayed by this many hours (the weak-flags definition)
EARLY_LIGHT_H = 8                             # rover v3.1: true sunlight = forecast advanced by this many hours
WEAK_BATTERY_FACTOR = 0.7                     # rover v3.1: battery capacity x 0.7
CHANGES = ("none", "dusty_panels", "tight_slope", "blocked_route", "late_light", "early_light", "weak_battery")


def blocked_ring(size: int, lander_rc: tuple[float, float], mpp: float, seed: int) -> np.ndarray:
    """[S,S] bool: a ring around the lander with RING_GAPS open gaps at seeded angles."""
    rr, cc = np.mgrid[0:size, 0:size]
    dr, dc = rr - lander_rc[0], cc - lander_rc[1]
    dist = np.hypot(dr, dc) * mpp
    ang = np.degrees(np.arctan2(dc, -dr)) % 360.0
    start = np.random.default_rng(seed).uniform(0, 360.0 / RING_GAPS)
    gap = np.zeros((size, size), bool)
    for k in range(RING_GAPS):
        centre = (start + k * 360.0 / RING_GAPS) % 360.0
        gap |= np.abs((ang - centre + 180.0) % 360.0 - 180.0) <= RING_GAP_DEG / 2
    return (dist >= RING_INNER_M) & (dist <= RING_OUTER_M) & ~gap


def apply_change(world: RoverWorld, cfg: MissionConfig, name: str, seed: int = 0) -> tuple[RoverWorld, MissionConfig, dict]:
    if name == "none":
        return world, cfg, {"name": name, "synthetic": False, "what": "no change: the world as built from NASA data", "visible_to_rover": "n/a"}
    if name == "dusty_panels":
        new = cfg.rover.solar_panel_w * PANEL_FACTOR
        cfg2 = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"solar_panel_w": new})})
        return world, cfg2, {"name": name, "synthetic": True, "what": f"solar panel output x {PANEL_FACTOR} ({cfg.rover.solar_panel_w:g} W -> {new:g} W); terrain unchanged",
                             "visible_to_rover": "no: nothing in the observation changes, only how the battery behaves"}
    if name == "tight_slope":
        cfg2 = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"max_slope_deg": TIGHT_SLOPE_DEG})})
        return world, cfg2, {"name": name, "synthetic": True, "what": f"rover slope limit {cfg.rover.max_slope_deg:g} -> {TIGHT_SLOPE_DEG:g} deg; terrain unchanged",
                             "visible_to_rover": "yes: the slope channel is slope / limit"}
    if name == "blocked_route":
        ring = blocked_ring(world.size, world.lander_rc, world.mpp, seed)
        slope = world.slope.copy()
        slope[ring] = np.maximum(slope[ring], RING_SLOPE_DEG)
        w2 = dataclasses.replace(world, slope=slope)
        return w2, cfg, {"name": name, "synthetic": True,
                         "what": f"SYNTHETIC obstacle, not real terrain: a ring of {RING_SLOPE_DEG:g} deg ground {RING_INNER_M:g}-{RING_OUTER_M:g} m from the lander with "
                                 f"{RING_GAPS} gaps of {RING_GAP_DEG:g} deg ({int(ring.sum())} cells); hazard map left as it was",
                         "visible_to_rover": "yes in the slope channel; no in the hazard channel"}
    if name == "late_light":
        k = LATE_LIGHT_H
        sun = np.concatenate([np.repeat(world.sun_frac[:1], k, axis=0), world.sun_frac[:-k]], axis=0)        # true[t] = forecast[t - k]; first k hours = forecast[0]
        assert sun.shape == world.sun_frac.shape and sun.dtype == world.sun_frac.dtype
        return dataclasses.replace(world, sun_frac=sun), cfg, {
            "name": name, "synthetic": True,
            "what": f"SYNTHETIC forecast error: the true sunlight at every cell is the forecast delayed by {k} h; terrain, hazard and physics unchanged",
            "visible_to_rover": "no: the planner and executor read the forecast; only the battery behaves differently"}
    if name == "early_light":
        k = EARLY_LIGHT_H
        sun = np.concatenate([world.sun_frac[k:], np.repeat(world.sun_frac[-1:], k, axis=0)], axis=0)        # true[t] = forecast[t + k]; last k hours = last frame
        assert sun.shape == world.sun_frac.shape and sun.dtype == world.sun_frac.dtype
        return dataclasses.replace(world, sun_frac=sun), cfg, {
            "name": name, "synthetic": True,
            "what": f"SYNTHETIC forecast error: the true sunlight at every cell is the forecast advanced by {k} h; terrain, hazard and physics unchanged",
            "visible_to_rover": "no: the planner and executor read the forecast; only the battery behaves differently"}
    if name == "weak_battery":
        new = cfg.rover.battery_wh * WEAK_BATTERY_FACTOR
        cfg2 = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"battery_wh": new})})
        return world, cfg2, {"name": name, "synthetic": True, "what": f"battery capacity x {WEAK_BATTERY_FACTOR} ({cfg.rover.battery_wh:g} Wh -> {new:g} Wh); terrain unchanged",
                             "visible_to_rover": "only through its own battery reading (the planner still believes the old capacity)"}
    raise ValueError(f"unknown change {name!r}; choose one of {CHANGES}")
