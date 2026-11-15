"""Score the skyline model against the exact teacher on the held-out TEST tile (Site23), plus the downstream light numbers.

    python -I -m worldmodel.light.evaluate

Compared on the same 5 m truth (exact ray-cast of the fine grid): (1) the physics backup = exact ray-cast of the coarse grid,
(2) the model = backup + learned correction. Directions of Sun and Earth come from a light point-series file in data/light/
(GENERATED stand-in until Imok's SPICE files exist); the geometry that turns skyline + direction into Sun fraction and Earth
visibility is exact (physics.py), so all error comes from the skyline.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from contracts.light import PointSeries, load_points
from contracts.mission import load_mission
from contracts.paths import data_root, model_path, run_dir, tile_path
from contracts.tiles import load_tile
from worldmodel.light.data import TEST_SITES, VAL_SITES, build_region, regions_for_site
from worldmodel.light.physics import light_from_skyline


def find_directions(region: str | None = None, date: str = "2027-09-01") -> PointSeries | None:
    """Sun/Earth direction series for a region: its own file for `date` if present (real SPICE), else any region's first file."""
    d = data_root() / "light"
    if not d.exists():
        return None
    own = d / f"{region}_{date}_points.npz" if region else None
    if own is not None and own.exists():
        return load_points(own)
    files = sorted(d.glob("*_points.npz"))
    return load_points(files[0]) if files else None


def skyline_stats(pred: np.ndarray, truth: np.ndarray) -> dict:
    e = np.abs(pred - truth)
    return {"mae_deg": float(e.mean()), "p50_deg": float(np.percentile(e, 50)), "p95_deg": float(np.percentile(e, 95)), "max_deg": float(e.max()),
            "bias_deg": float((pred - truth).mean())}


def light_stats(sky: np.ndarray, truth_sky: np.ndarray, ps: PointSeries, sun_radius: float) -> dict:
    sun_abs, earth_abs = ps.absolute_elev_deg("sun")[:, 0], ps.absolute_elev_deg("earth")[:, 0]
    args = (ps.sun_az_deg, sun_abs, ps.earth_az_deg, earth_abs, sun_radius)
    a, t = light_from_skyline(sky, *args), light_from_skyline(truth_sky, *args)
    dt_h = float(np.median(np.diff(ps.et_s)) / 3600.0)
    sun_h_err = np.abs((a["sun_fraction"] > 0).sum(0) - (t["sun_fraction"] > 0).sum(0)) * dt_h
    earth_h_err = np.abs(a["earth_visible"].sum(0) - t["earth_visible"].sum(0)) * dt_h
    return {"sun_fraction_mae": float(np.abs(a["sun_fraction"] - t["sun_fraction"]).mean()),
            "earth_visible_wrong_pct": float(100 * (a["earth_visible"] != t["earth_visible"]).mean()),
            "sun_up_wrong_pct": float(100 * ((a["sun_fraction"] > 0) != (t["sun_fraction"] > 0)).mean()),
            "sunlit_hours_error_mean": float(sun_h_err.mean()), "sunlit_hours_error_p95": float(np.percentile(sun_h_err, 95)),
            "earth_hours_error_mean": float(earth_h_err.mean()), "earth_hours_error_p95": float(np.percentile(earth_h_err, 95)),
            "truth_sunlit_hours_mean": float((t["sun_fraction"] > 0).sum(0).mean() * dt_h)}


def main(argv: list[str] | None = None) -> int:
    from worldmodel.light.model import SkylineStudent, load_checkpoint, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="light_skyline")
    ap.add_argument("--cells", type=int, default=1000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_mission()
    device = pick_device(a.device)
    model, info = load_checkpoint(model_path(a.name), device)
    student = SkylineStudent(model, cfg, device, info)
    sun_r = cfg.light_model.sun_angular_radius_deg
    report = {"checkpoint_name": a.name, "checkpoint_id": info["checkpoint_id"], "trained_on": info["trained_on"], "val_regions": info["val_regions"],
              "coarse_factor": cfg.light_model.coarse_factor, "max_ray_range_m": cfg.light_model.max_ray_range_m,
              "truth": "exact ray-cast of the fine 5 m LOLA grid", "regions": {}}
    print(f"checkpoint {info['checkpoint_id']}  trained on {info['trained_on']}")
    jobs = [(site, region) for site in TEST_SITES + VAL_SITES for region in regions_for_site(site)]     # test sites first
    for site in VAL_SITES + TEST_SITES:
        if not regions_for_site(site):
            print(f"note: no tiles for {site} on this machine; skipped", flush=True)
    for site, region in jobs:
        role = "TEST" if site in TEST_SITES else "val"
        s = build_region(region, cfg, a.cells, 2000, device, cache=False)
        tile = load_tile(tile_path(s.region))
        ps = find_directions(s.region)
        res_dir = f"{ps.region} {ps.date} [{ps.meta.provenance.banner}]" if ps else None
        t0 = time.time(); base = student.skyline(tile, s.cells_fine, use_model=False); t_base = time.time() - t0
        t0 = time.time(); pred = student.skyline(tile, s.cells_fine, use_model=True); t_net = time.time() - t0
        res = {"role": role, "n_cells": a.cells, "directions": res_dir, "skyline_baseline": skyline_stats(base, s.truth), "skyline_model": skyline_stats(pred, s.truth),
               "seconds_baseline": t_base, "seconds_model": t_net}
        if ps is not None:
            res["light_baseline"] = light_stats(base, s.truth, ps, sun_r)
            res["light_model"] = light_stats(pred, s.truth, ps, sun_r)
        report["regions"][s.region] = res
        b, m = res["skyline_baseline"], res["skyline_model"]
        print(f"\n{s.region} [{role}]  skyline error vs exact 5 m ray-cast (deg):   MAE    p95    max")
        print(f"  physics backup (coarse ray-cast)                     {b['mae_deg']:6.3f} {b['p95_deg']:6.3f} {b['max_deg']:6.2f}")
        print(f"  model (backup + learned correction)                  {m['mae_deg']:6.3f} {m['p95_deg']:6.3f} {m['max_deg']:6.2f}")
        if ps is not None:
            lb, lm_ = res["light_baseline"], res["light_model"]
            print(f"  downstream over {len(ps.times_utc)} h, Sun/Earth from {res_dir}:   backup  ->  model")
            for k, lab in (("sun_fraction_mae", "Sun fraction MAE"), ("sun_up_wrong_pct", "Sun up/down wrong %"),
                           ("earth_visible_wrong_pct", "Earth visible wrong %"), ("sunlit_hours_error_mean", "sunlit hours error (mean h)"),
                           ("sunlit_hours_error_p95", "sunlit hours error (p95 h)"), ("earth_hours_error_mean", "Earth-link hours error (mean h)")):
                print(f"    {lab:34s} {lb[k]:8.3f} -> {lm_[k]:8.3f}")
        _write_partial(run_dir("light") / "eval.json", report, len(jobs))
    out = run_dir("light") / "eval.json"
    _write_partial(out, report, len(jobs))
    print(f"\nwrote {out}")
    return 0


def _write_partial(out, report: dict, planned: int) -> None:
    """eval.json is rewritten after every window, so a stage killed at its budget still leaves the test-site numbers."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**report, "windows_planned": planned, "complete": len(report["regions"]) >= planned}, indent=2), encoding="utf-8")
    tmp.replace(out)


if __name__ == "__main__":
    raise SystemExit(main())
