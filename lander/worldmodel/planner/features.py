"""Candidate features for the mission planner: Navigator-style metrics for (landing cell, landing hour) in a rover world, computed from the real
SPICE + skyline light fields, the hazard map and the NASA PSR ice-target edges. Everything here is observed or modelled data, no learning."""

from __future__ import annotations

import numpy as np

from worldmodel.rover.world import RoverWorld

HORIZON_H = 300                       # the rover mission horizon in hours (config rover_sim.max_steps)
NAMES = ("sun_mean", "sun_first_day", "longest_dark_h", "hours_to_first_night", "earth_mean", "longest_blackout_h", "hazard_150m", "slope_here",
         "clear_radius_m", "nearest_target_km", "targets_within_1km", "psr_frac_1km", "start_hour_frac")
NF = len(NAMES)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def candidate_features(w: RoverWorld, cell_rc: tuple[float, float], hour: int, stride: int, horizon: int = HORIZON_H) -> np.ndarray:
    r, c = cell_rc
    ci, cj = int(np.clip(r // stride, 0, w.sun_frac.shape[1] - 1)), int(np.clip(c // stride, 0, w.sun_frac.shape[2] - 1))
    sl = slice(hour, min(hour + horizon, w.sun_frac.shape[0]))
    sun = w.sun_frac[sl, ci, cj].astype(np.float32)
    earth = w.earth_up[sl, ci, cj].astype(np.float32)
    dark = sun < 0.05
    first_night = int(np.argmax(dark)) if dark.any() else len(sun)
    ri, cc = int(np.clip(r, 0, w.size - 1)), int(np.clip(c, 0, w.size - 1))
    y, x = np.ogrid[:w.size, :w.size]
    d = np.hypot(y - r, x - c) * w.mpp
    haz150 = float(w.hazard[d <= 150.0].mean()) if (d <= 150.0).any() else 1.0
    unsafe = w.hazard >= 0.5
    clear = float(min(d[unsafe].min(), 400.0)) if unsafe.any() else 400.0
    td = np.hypot(w.targets_rc[:, 0] - r, w.targets_rc[:, 1] - c) * w.mpp if len(w.targets_rc) else np.array([5000.0])
    return np.array([sun.mean(), sun[:24].mean(), _longest_run(dark) / horizon, first_night / horizon, earth.mean(), _longest_run(earth < 0.5) / horizon, haz150,
                     float(w.slope[ri, cc]) / 20.0, clear / 400.0, float(td.min()) / 1000.0, float((td <= 1000.0).sum()) / 8.0, float(w.psr[d <= 1000.0].mean()),
                     hour / max(w.sun_frac.shape[0] - horizon, 1)], dtype=np.float32)


def navigator_score(f: np.ndarray) -> float:
    """The exact, non-learned ranking (the planner's physics backup): reward sunlight, Earth link and clear ground, penalise long darkness,
    blackout, hazard and a far ice target. Equal, hand-set weights; no tuning on any outcome."""
    g = dict(zip(NAMES, f, strict=True))
    return float(2.0 * g["sun_mean"] + 1.0 * g["earth_mean"] + 1.0 * g["clear_radius_m"] + 1.0 * g["hours_to_first_night"] - 2.0 * g["longest_dark_h"]
                 - 1.0 * g["longest_blackout_h"] - 2.0 * g["hazard_150m"] - 1.0 * g["nearest_target_km"] + 0.5 * min(g["targets_within_1km"], 1.0))
