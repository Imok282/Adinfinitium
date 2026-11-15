"""Real skyline-model outputs for 4 cells of the held-out Site23 tile -> website/assets/examples/light/light_examples.json.

    ~/venvs/adi/bin/python -I website/tools/examples/light_examples.py

Read-only use of data/models/light_skyline.pt. For 300 random cells (fixed seed) computes: truth = exact ray-cast of the
5 m grid, backup = exact ray-cast of the 4x coarse grid, model = backup + learned correction; then picks 4 cells
(low / median / high skyline, plus the cell at the 95th percentile of model error as an honest hard case).
Sun/Earth paths: real NASA SPICE (DE440) series from data/light/lola5m_site23_2027-09-01_points.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from contracts.light import load_points  # noqa: E402
from contracts.mission import load_mission  # noqa: E402
from contracts.paths import data_root, model_path, tile_path  # noqa: E402
from contracts.tiles import load_tile  # noqa: E402
from worldmodel.light.data import sample_cells  # noqa: E402
from worldmodel.light.model import SkylineStudent, load_checkpoint, pick_device  # noqa: E402
from worldmodel.light.physics import light_from_skyline  # noqa: E402
from worldmodel.light.teacher import n_steps, raycast_skyline_torch  # noqa: E402

OUT = ROOT / "website/assets/examples/light"
REGION, DATE = "lola5m_site23", "2027-09-01"


def r2(a, d=2):
    return [round(float(v), d) for v in np.asarray(a).ravel()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_mission()
    lm = cfg.light_model
    device = pick_device(None)
    model, info = load_checkpoint(model_path("light_skyline"), device)
    student = SkylineStudent(model, cfg, device, info)
    tile = load_tile(tile_path(REGION))
    mpp = tile.meters_per_pixel
    cells = sample_cells(tile.shape, n_steps(lm.max_ray_range_m, mpp) + 1, 300, np.random.default_rng(2026))
    truth = raycast_skyline_torch(tile.height, mpp, cells, cfg.moon.radius_m, lm.max_ray_range_m, lm.n_azimuth, device=device)
    base = student.skyline(tile, cells, use_model=False)
    pred = student.skyline(tile, cells, use_model=True)
    e_m = np.abs(pred - truth).mean(1)
    e_b = np.abs(base - truth).mean(1)
    mean_sky = truth.mean(1)
    order = np.argsort(mean_sky)
    picks = {"open_rim_low_skyline": int(order[int(0.1 * len(order))]),
             "typical_median_skyline": int(order[len(order) // 2]),
             "enclosed_high_skyline": int(order[int(0.9 * len(order))])}
    used = set(picks.values())
    hard = int(np.argsort(e_m)[int(0.95 * len(e_m))])
    if hard in used:
        hard = int(np.argsort(e_m)[int(0.95 * len(e_m)) - 1])
    picks["hard_case_p95_model_error"] = hard

    ps = load_points(data_root() / "light" / f"{REGION}_{DATE}_points.npz")
    sun_abs = ps.absolute_elev_deg("sun")[:, 0]
    earth_abs = ps.absolute_elev_deg("earth")[:, 0]
    dt_h = float(np.median(np.diff(ps.et_s)) / 3600.0)
    sr = lm.sun_angular_radius_deg

    def light(sky):
        return light_from_skyline(sky[None], ps.sun_az_deg, sun_abs, ps.earth_az_deg, earth_abs, sr)

    out_cells = []
    for kind, i in picks.items():
        lt, lb, lmod = light(truth[i]), light(base[i]), light(pred[i])
        sun_h = {k: float((v["sun_fraction"][:, 0] > 0).sum() * dt_h) for k, v in (("truth", lt), ("backup", lb), ("model", lmod))}
        earth_h = {k: float(v["earth_visible"][:, 0].sum() * dt_h) for k, v in (("truth", lt), ("backup", lb), ("model", lmod))}
        out_cells.append({
            "kind": kind, "cell_rc": r2(cells[i], 1),
            "x_m": round(float(cells[i][1] * mpp), 1), "y_m": round(float(cells[i][0] * mpp), 1),
            "skyline_deg": {"truth_raycast_5m": r2(truth[i]), "model": r2(pred[i]), "backup_coarse_raycast_20m": r2(base[i])},
            "error_deg": {"model_mae": round(float(e_m[i]), 3), "backup_mae": round(float(e_b[i]), 3),
                          "model_max": round(float(np.abs(pred[i] - truth[i]).max()), 3),
                          "backup_max": round(float(np.abs(base[i] - truth[i]).max()), 3)},
            "sunlit_hours": {k: round(v, 1) for k, v in sun_h.items()},
            "earth_visible_hours": {k: round(v, 1) for k, v in earth_h.items()},
            "sun_fraction_mae": {"model": round(float(np.abs(lmod["sun_fraction"] - lt["sun_fraction"]).mean()), 4),
                                 "backup": round(float(np.abs(lb["sun_fraction"] - lt["sun_fraction"]).mean()), 4)},
        })
        print(kind, cells[i].round(0), "MAE model", round(e_m[i], 3), "backup", round(e_b[i], 3), "sun h", sun_h)

    step = 2  # every 2nd hour for the sky path
    doc = {
        "model": "neural light field / skyline model", "checkpoint": "data/models/light_skyline.pt",
        "checkpoint_id": info["checkpoint_id"], "trained_on": info["trained_on"], "validated_on": info["val_regions"],
        "region": REGION, "site": "Site23 (Malapert massif)", "split": "TEST (held out: never trained or tuned on)",
        "input": "LOLA 5 m heights block-averaged 4x (20 m grid); model = coarse ray-cast + learned correction",
        "truth": f"exact ray-cast of the 5 m LOLA grid out to {lm.max_ray_range_m:g} m (window-limited; far mountains missing)",
        "azimuth": "grid azimuth, index = degree 0..359 (0 = up on the tile map, clockwise)",
        "units": "skyline in degrees above the horizontal plane",
        "sample": "300 random cells (seed 2026); 4 picked by truth mean skyline percentile 10/50/90 and model-error p95",
        "summary_300_cells": {"model_mae_deg": round(float(e_m.mean()), 3), "backup_mae_deg": round(float(e_b.mean()), 3),
                              "model_p95_deg": round(float(np.percentile(np.abs(pred - truth), 95)), 3),
                              "backup_p95_deg": round(float(np.percentile(np.abs(base - truth), 95)), 3),
                              "cells_model_better_pct": round(float(100 * (e_m < e_b).mean()), 1)},
        "sun_earth_path": {
            "source": "NASA/JPL SPICE DE440, topocentric apparent, MOON_ME (MODELLED); grid azimuth = true az + east lon (unverified, KAN-21)",
            "date_start_utc": ps.times_utc[0], "date_end_utc": ps.times_utc[-1], "step_hours": dt_h * step,
            "times_utc": ps.times_utc[::step],
            "sun_az_deg": r2(ps.sun_az_deg[::step]), "sun_elev_deg": r2(sun_abs[::step]),
            "earth_az_deg": r2(ps.earth_az_deg[::step]), "earth_elev_deg": r2(earth_abs[::step]),
            "window_hours": round(len(ps.times_utc) * dt_h, 1),
        },
        "cells": out_cells,
        "provenance": "observed LOLA heights; skylines and light MODELLED",
    }
    p = OUT / "light_examples.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote", p, p.stat().st_size, "bytes; summary", doc["summary_300_cells"])


if __name__ == "__main__":
    main()
