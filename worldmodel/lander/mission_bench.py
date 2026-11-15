"""Two lander benchmarks beyond the standard map list, flown by classic ZEM/ZEV guidance and the learned pilot on the SAME missions (paired).

1. rehearsal  - the plan's own landing: every mission targets the cell chosen in a plan file (default results_pc/.../runs/demo/plan_001.json,
                Site23). This is a MISSION REHEARSAL of one planned landing, not the Site23 final exam (one cell, not random cells).
2. procedural - procedural crater maps proc_00..proc_NN built in memory (nothing written to data/tiles/). GENERATED - NOT OBSERVED terrain:
                no safety claim rests on these maps alone. Generator = data_tools.make_demo_tile (same crater_profile and random sequence),
                made closer to real LOLA tiles: craters evaluated only near themselves (same output, ~60x faster), 500-2500 craters per map,
                a random regional tilt of 0-7 deg, and a real LOLA laser-coverage mask (rotated, from the pilot's training tiles) so the
                backup hazard belief is gap-filled from ~10 % of pixels and is imperfect, as on real maps. Touchdown truth = slope >= limit.

    python -m worldmodel.lander.mission_bench time                      # time one batch per policy first
    python -m worldmodel.lander.mission_bench rehearsal --n 1024
    python -m worldmodel.lander.mission_bench procedural --maps 20 --n 256

Safety guards (shared Mac): before every batch the free space of the home volume is checked; below --min-free-gb the run stops and the
partial results are saved. Same if the process RSS passes --max-rss-gb.
Outputs: <runs>/lander/mission_rehearsal.{json,md}, <runs>/lander/procedural_benchmark.{json,md}, website/assets/data/mission_bench.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import model_path, repo_root, run_dir, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, hazard_from_slope, load_tile, slope_from_height
from data_tools.make_demo_tile import crater_profile, make_demo_tile
from worldmodel.hazard.backup import backup_hazard
from worldmodel.lander.benchmark import paired_diff_ci, sign_test, wilson
from worldmodel.lander.env import make_params
from worldmodel.lander.fly import fly, summarize
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.pilot import load_checkpoint, pick_device
from worldmodel.lander.scenes import Scene, scene_for_region
from worldmodel.lander.terrain import safe_target_cells, terrain_from_tile

DEFAULT_PLAN = repo_root() / "results_pc" / "extracted" / "runs" / "demo" / "plan_001.json"
WEB_JSON = repo_root() / "website" / "assets" / "data" / "mission_bench.json"
CONDITIONS = (("nominal", 1.0), ("harsh", 3.0))     # harsh = engine-out, nav-drift, blind-camera rates x3 (same as worldmodel.lander.benchmark)
GENERATED = Provenance.GENERATED.banner


class Stop(RuntimeError):
    """A safety guard tripped: save what we have and stop."""


# ------------------------------------------------------------------------------------------------------------------ guards
def rss_gb() -> float:
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()) / 1e6
    except Exception:  # noqa: BLE001
        return float("nan")


class Guard:
    def __init__(self, min_free_gb: float, max_rss_gb: float):
        self.min_free_gb, self.max_rss_gb, self.peak_rss, self.min_free = min_free_gb, max_rss_gb, 0.0, float("inf")

    def check(self, where: str) -> None:
        free = shutil.disk_usage(Path.home()).free / 1e9
        rss = rss_gb()
        self.min_free, self.peak_rss = min(self.min_free, free), max(self.peak_rss, rss if rss == rss else 0.0)
        if free < self.min_free_gb:
            raise Stop(f"free disk {free:.2f} GB < {self.min_free_gb} GB before {where}")
        if rss > self.max_rss_gb:
            raise Stop(f"process RSS {rss:.2f} GB > {self.max_rss_gb} GB before {where}")


# ------------------------------------------------------------------------------------------------------------------ flying
def condition_params(cfg, mult: float):
    p = make_params(cfg)
    p.p_engine_out, p.p_nav_drift, p.p_camera_blind = (min(1.0, v * mult) for v in (p.p_engine_out, p.p_nav_drift, p.p_camera_blind))
    gp = guidance_params(p, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    return p, gp


def fly_paired(scene: Scene, cfg, model, n: int, batch: int, seed0: int, device, guard: Guard, targets_xy: np.ndarray | None = None) -> dict:
    """For each condition: the same n missions (seeded batches) flown by classic and by the pilot. Returns merged metrics per condition/policy."""
    out = {}
    for cond, mult in CONDITIONS:
        params, gp = condition_params(cfg, mult)
        res = {"classic": [], "pilot": []}
        for b in range(math.ceil(n / batch)):
            k = min(batch, n - b * batch)
            seed = seed0 + 1009 * b + (0 if cond == "nominal" else 500_000)
            tg = None if targets_xy is None else torch.as_tensor(targets_xy, dtype=torch.float32, device=device)[None].repeat(k, 1)
            for pol, mdl in (("classic", None), ("pilot", model)):
                guard.check(f"{scene.region} {cond} batch {b} {pol}")
                m = fly(scene, params, gp, mdl, k, seed, 5, device=device, targets=tg)
                res[pol].append({key: v.cpu() for key, v in m.items()})
            if device.type == "mps":
                torch.mps.empty_cache()
        out[cond] = {pol: {key: torch.cat([m[key] for m in ms]) for key in ms[0]} for pol, ms in res.items()}
    return out


def compare(m: dict) -> dict:
    """Paired statistics of classic vs pilot on one set of missions."""
    okc, okp = m["classic"]["success"].numpy().astype(bool), m["pilot"]["success"].numpy().astype(bool)
    n = okc.size
    b, c = int((okp & ~okc).sum()), int((~okp & okc).sum())
    d, lo, hi = paired_diff_ci(b, c, n)
    pol = {}
    for name, ok, mm in (("classic", okc, m["classic"]), ("pilot", okp, m["pilot"])):
        s = summarize(mm)
        dist = mm["target_dist_m"].numpy()
        s.update({"ci95": [round(v, 4) for v in wilson(int(ok.sum()), n)],
                  "miss_p95_m_success": float(np.percentile(dist[ok], 95)) if ok.any() else None,
                  "td_vz_mean_success": float(mm["td_vz"].numpy()[ok].mean()) if ok.any() else None,
                  "td_vh_mean_success": float(mm["td_vh"].numpy()[ok].mean()) if ok.any() else None})
        pol[name] = s
    return {"n": n, **pol, "only_pilot_landed": b, "only_classic_landed": c, "sign_test_p": sign_test(b, c), "diff": d, "diff_ci95": [lo, hi]}


# ------------------------------------------------------------------------------------------------------------------ job 1
def rehearsal(a, cfg, device, model, info, guard: Guard) -> dict:
    plan = json.loads(Path(a.plan).read_text())
    region = plan["region"]
    scene = scene_for_region(region, cfg, device, "backup")
    if scene is None:
        raise SystemExit(f"{region}: no usable scene")
    h, w = scene.terrain.shape
    s = cfg.rover_sim.world_px
    origin = ((h - s) // 2, (w - s) // 2)                                    # same window as worldmodel.rover.world.build_world
    rc_world = np.array(plan["target_cell_rc"], dtype=float)
    rc_tile = rc_world + np.array(origin)
    xy = scene.terrain.cell_to_xy(rc_tile[None])[0]                          # same mapping as worldmodel.planner.plan
    r, c = int(round(rc_tile[0])), int(round(rc_tile[1]))
    T = scene.terrain
    win = lambda g, k: g[r - k:r + k + 1, c - k:c + k + 1].cpu().numpy()   # noqa: E731
    target = {"region": region, "cell_rc_rover_window": rc_world.tolist(), "window_origin_rc": list(origin), "cell_rc_tile": rc_tile.tolist(),
              "xy_m": [float(v) for v in xy], "truth_slope_deg": float(T.slope[r, c]), "truth_hazard": float(T.hazard[r, c]),
              "belief_hazard": float(T.belief[r, c]), "truth_hazard_frac_within_15m": float(win(T.hazard, 3).mean()),
              "belief_hazard_frac_within_15m": float(win(T.belief, 3).mean()), "truth_hazard_frac_within_50m": float(win(T.hazard, 10).mean()),
              "in_safe_target_cells": bool(((scene.safe_rc == [r, c]).all(axis=1)).any())}
    print(f"rehearsal target {region} window rc {rc_world.tolist()} -> tile rc {rc_tile.tolist()} -> xy {xy.round(1).tolist()} m; "
          f"slope {target['truth_slope_deg']:.1f} deg, truth hazard {target['truth_hazard']:.0f}, safe cell {target['in_safe_target_cells']}", flush=True)
    t0 = time.time()
    rows, partial, stop = {}, False, None
    try:
        merged = fly_paired(scene, cfg, model, a.n, a.batch, 77_000, device, guard, xy)
        rows = {cond: compare(m) for cond, m in merged.items()}
    except Stop as e:
        partial, stop = True, str(e)
    for cond, r_ in rows.items():
        print(f"  {cond:7s} classic {r_['classic']['success_rate']:.4f}  pilot {r_['pilot']['success_rate']:.4f}  "
              f"(pilot-only {r_['only_pilot_landed']}, classic-only {r_['only_classic_landed']}, p={r_['sign_test_p']:.3f})", flush=True)
    meta = Meta(provenance=Provenance.MODELLED, source=f"plan {Path(a.plan).name} ({plan['meta']['source'][:120]}); terrain NASA LOLA 5 m {region}; "
                "lander physics worldmodel.lander.env", frame="tile ENU m (centre origin)", units={"miss": "m", "td_vz": "m/s", "success": "fraction"},
                created_by="worldmodel.lander.mission_bench rehearsal")
    return {"label": "MISSION REHEARSAL (one planned cell; not the random-cell Site23 final exam)", "meta": json.loads(meta.model_dump_json()),
            "plan_file": str(Path(a.plan).relative_to(repo_root())) if Path(a.plan).is_relative_to(repo_root()) else str(a.plan),
            "plan": {k: plan[k] for k in ("decision", "region", "landing_time_utc", "target_cell_rc", "scatter_radius_m", "backup_used")},
            "target": target, "pilot": a.name, "checkpoint_id": info.get("checkpoint_id"), "parameters": info.get("parameters"),
            "trained_on": info.get("trained_on"), "missions_per_condition": a.n, "conditions": dict(CONDITIONS), "results": rows,
            "partial": partial, "stopped_because": stop, "wall_s": round(time.time() - t0, 1),
            "notes": ["Every mission targets the plan's cell; start offset, velocity, failures and sensor noise are random (seeded) and identical for both pilots.",
                      "Touchdown truth = official LOLA slope rule; the pilot's camera sees the slope-rule backup belief from the sparse laser pixels.",
                      "Lighting/time of landing does not enter the lander simulation; the plan's landing time is carried for reference only.",
                      "Site23 is this pilot's final-exam site: this rehearsal flies ONE cell there and does not replace the random-cell final exam."]}


# ------------------------------------------------------------------------------------------------------------------ job 2
def local_crater_field(size: int, mpp: float, seed: int, n_craters: int) -> tuple[np.ndarray, int]:
    """make_demo_tile's height field with each crater evaluated only within 2.5 radii of its centre (the rim term is < 1e-6 of the depth beyond).
    Same random sequence: radii, then one centre per crater, then the roughness. Returns (height, craters placed)."""
    rng = np.random.default_rng(seed)
    half = (size - 1) / 2 * mpp
    xs = np.linspace(-half, half, size)
    X, Y = np.meshgrid(xs, xs[::-1])
    height = 0.02 * X - 0.01 * Y + 8.0 * np.sin(X / 400.0) * np.cos(Y / 550.0)
    radii = np.clip(rng.pareto(1.8, n_craters) * 15.0 + 8.0, 8.0, size * mpp / 6)
    placed = 0
    for rad in radii:
        cx, cy = rng.uniform(-half, half, 2)
        if np.hypot(cx, cy) < 150.0 + rad:
            continue
        col, row = (cx + half) / mpp, (half - cy) / mpp
        k = int(math.ceil(2.5 * rad / mpp)) + 1
        r0, r1 = max(0, int(row) - k), min(size, int(row) + k + 2)
        c0, c1 = max(0, int(col) - k), min(size, int(col) + k + 2)
        if r0 >= r1 or c0 >= c1:
            continue
        height[r0:r1, c0:c1] += crater_profile(np.hypot(X[r0:r1, c0:c1] - cx, Y[r0:r1, c0:c1] - cy), rad, 0.2 * rad)
        placed += 1
    height += rng.normal(0.0, 0.15, height.shape)
    return height, placed


def generator_check() -> float:
    """Max |height difference| between the local evaluation and make_demo_tile itself (size 400, 300 craters, seed 3)."""
    ref = make_demo_tile(400, 5.0, 3, 300).height.astype(np.float64)
    mine, _ = local_crater_field(400, 5.0, 3, 300)
    return float(np.abs(mine.astype(np.float32) - ref).max())


def borrowed_masks(regions: list[str]) -> list[tuple[str, np.ndarray]]:
    out = []
    for reg in regions:
        with np.load(tile_path(reg), allow_pickle=False) as z:
            if "real_mask" in z.files:
                out.append((reg, z["real_mask"].astype(bool)))
    return out


def procedural_tile(i: int, size: int, mpp: float, masks: list, limit: float, seed_base: int) -> tuple[Tile, dict]:
    pr = np.random.default_rng([seed_base, i])                               # per-map knobs, separate from the crater sequence
    n_craters = int(pr.integers(500, 2501))
    tilt_deg, tilt_az = float(pr.uniform(0.0, 7.0)), float(pr.uniform(0.0, 2 * np.pi))
    mask_src, mask = masks[i % len(masks)]
    rot = (i // len(masks)) % 4
    height, placed = local_crater_field(size, mpp, seed_base + i, n_craters)
    half = (size - 1) / 2 * mpp
    xs = np.linspace(-half, half, size)
    X, Y = np.meshgrid(xs, xs[::-1])
    height += math.tan(math.radians(tilt_deg)) * (X * math.cos(tilt_az) + Y * math.sin(tilt_az))
    slope = slope_from_height(height, mpp)
    real = np.rot90(mask, rot).copy()
    meta = Meta(provenance=Provenance.GENERATED, source=f"worldmodel.lander.mission_bench procedural map {i} (make_demo_tile crater field, seed {seed_base + i}); "
                f"laser mask borrowed from {mask_src} rotated {90 * rot} deg", frame="local tile ENU (synthetic)", created_by="worldmodel.lander.mission_bench")
    tile = Tile(region=f"proc_{i:02d}", meters_per_pixel=mpp, height=height.astype(np.float32), slope=slope.astype(np.float32),
                hazard=hazard_from_slope(slope, limit), uncertainty=np.where(real, 0.5, 2.0).astype(np.float32), psr_mask=np.zeros(height.shape, bool),
                meta=meta, real_mask=real)
    knobs = {"region": tile.region, "seed": seed_base + i, "craters_drawn": n_craters, "craters_placed": placed, "tilt_deg": round(tilt_deg, 2),
             "tilt_azimuth_deg": round(math.degrees(tilt_az), 1), "laser_mask_from": mask_src, "laser_mask_rot_deg": 90 * rot,
             "laser_coverage": float(real.mean()), "slope_median_deg": float(np.median(slope)), "slope_p90_deg": float(np.percentile(slope, 90)),
             "truth_hazard_frac": float(tile.hazard.mean())}
    return tile, knobs


def real_slope_stats(regions: list[str]) -> dict:
    out = {}
    for reg in regions:
        if tile_path(reg).exists():
            with np.load(tile_path(reg), allow_pickle=False) as z:
                s = z["slope"]
                out[reg] = {"median": float(np.median(s)), "p90": float(np.percentile(s, 90)), "hazard_frac": float((s >= 12.0).mean())}
    return out


def procedural(a, cfg, device, model, info, guard: Guard, write) -> dict:
    limit = cfg.lander.touchdown_limits.max_slope_deg
    trained = info.get("trained_on") or ["lola5m_site01", "lola5m_site04", "lola5m_site06", "lola5m_site07", "lola5m_site20"]
    masks = borrowed_masks(trained)
    check = generator_check()
    print(f"generator check vs make_demo_tile: max |dh| = {check:.2e} m", flush=True)
    real = real_slope_stats([f"lola5m_{s}" for s in ("site01", "site04", "site06", "site07", "site20", "site11", "site23", "site42", "haworth", "shoemaker", "dm2")])
    res = {"label": f"PROCEDURAL MAPS - {GENERATED}", "meta": json.loads(Meta(provenance=Provenance.GENERATED,
           source="worldmodel.lander.mission_bench procedural crater maps (data_tools.make_demo_tile generator, made realistic as described); "
           "laser masks borrowed from NASA LOLA tiles; lander physics worldmodel.lander.env", frame="local tile ENU (synthetic)",
           units={"slope": "deg", "miss": "m", "success": "fraction"}, created_by="worldmodel.lander.mission_bench procedural").model_dump_json()),
           "pilot": a.name, "checkpoint_id": info.get("checkpoint_id"), "trained_on": info.get("trained_on"), "missions_per_map_per_condition": a.n,
           "conditions": dict(CONDITIONS), "generator_max_abs_height_diff_vs_make_demo_tile_m": check, "real_lola_slope_stats": real,
           "maps": [], "skipped": [], "partial": False, "stopped_because": None,
           "notes": [f"{GENERATED}: synthetic terrain; never used alone for a safety claim.",
                     "Craters evaluated only within 2.5 radii of their centre (same output as make_demo_tile, checked above).",
                     "Per map: 500-2500 craters, regional tilt 0-7 deg (random azimuth), real LOLA laser-coverage mask (pilot training tiles, rotated) so the "
                     "camera's backup belief is gap-filled from ~10 % of pixels; truth = gap-free slope >= limit. Targets = random truth-safe cells.",
                     "Nothing is written to data/tiles/; the maps exist only in memory and are reproducible from the seeds."]}
    t0, oks = time.time(), {c: {"classic": [], "pilot": []} for c, _ in CONDITIONS}
    try:
        for i in (a.map_ids if a.map_ids else range(a.maps)):          # --map-ids: a disjoint subset for parallel workers (same seeds per map)
            guard.check(f"building proc_{i:02d}")
            tg = time.time()
            tile, knobs = procedural_tile(i, a.size, 5.0, masks, limit, a.seed_base)
            belief = backup_hazard(tile, cfg)
            safe = safe_target_cells(tile)
            knobs.update({"belief_hazard_frac": float(belief.mean()), "belief_truth_disagree_frac": float(((belief >= 0.5) != (tile.hazard >= 0.5)).mean()),
                          "safe_target_cells": int(len(safe)), "build_s": round(time.time() - tg, 1)})
            if len(safe) < 50:
                res["skipped"].append({**knobs, "why": "fewer than 50 safe target cells"})
                print(f"proc_{i:02d}: skipped ({len(safe)} safe cells; tilt {knobs['tilt_deg']} deg)", flush=True)
                continue
            scene = Scene(terrain_from_tile(tile, device, belief), safe, tile.region)
            del tile, belief
            merged = fly_paired(scene, cfg, model, a.n, a.batch, 90_000 + 131 * i, device, guard)
            row = {**knobs, "results": {cond: compare(m) for cond, m in merged.items()}}
            for cond, m in merged.items():
                for pol in ("classic", "pilot"):
                    oks[cond][pol].append(m[pol]["success"].numpy().astype(bool))
            res["maps"].append(row)
            del scene, merged
            r_ = row["results"]
            print(f"proc_{i:02d} craters {knobs['craters_drawn']:4d} tilt {knobs['tilt_deg']:4.1f} slope med {knobs['slope_median_deg']:4.1f} p90 "
                  f"{knobs['slope_p90_deg']:4.1f} | nominal C {r_['nominal']['classic']['success_rate']:.3f} P {r_['nominal']['pilot']['success_rate']:.3f} | "
                  f"harsh C {r_['harsh']['classic']['success_rate']:.3f} P {r_['harsh']['pilot']['success_rate']:.3f} | {time.time() - t0:.0f}s rss {rss_gb():.2f} GB", flush=True)
            res["pooled"] = pooled(oks)
            write(res, final=False)
    except Stop as e:
        res["partial"], res["stopped_because"] = True, str(e)
        print(f"STOP: {e}", flush=True)
    res["pooled"] = pooled(oks)
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def merge_parts(paths: list[str]) -> dict:
    """Join procedural part files (parallel workers on disjoint --map-ids) into one result. pooled() only uses per-map counts (successes, the
    two discordant counts, map means), so each map's paired outcome table is rebuilt from its saved counts: the pooled numbers equal a serial run's."""
    parts = [json.loads(Path(p).read_text()) for p in paths]
    maps = {m["region"]: m for p in parts for m in p["maps"]}
    skipped = {s["region"]: s for p in parts for s in p["skipped"] if s["region"] not in maps}
    rows = [maps[k] for k in sorted(maps)]
    oks = {c: {"classic": [], "pilot": []} for c, _ in CONDITIONS}
    for m in rows:
        for cond in oks:
            x = m["results"][cond]
            n, b, c = x["n"], x["only_pilot_landed"], x["only_classic_landed"]
            both = round(x["classic"]["success_rate"] * n) - c
            neither = n - both - b - c
            oks[cond]["classic"].append(np.r_[np.ones(both + c, bool), np.zeros(b + neither, bool)])
            oks[cond]["pilot"].append(np.r_[np.ones(both, bool), np.zeros(c, bool), np.ones(b, bool), np.zeros(neither, bool)])
    stops = [p["stopped_because"] for p in parts if p.get("stopped_because")]
    res = {**parts[0], "maps": rows, "skipped": [skipped[k] for k in sorted(skipped)], "partial": any(p["partial"] for p in parts),
           "stopped_because": "; ".join(stops) or None, "pooled": pooled(oks), "wall_s": None,
           "merged_from": [{"file": Path(pp).name, "maps": [m["region"] for m in p["maps"]], "wall_s": p.get("wall_s"), "device": (p.get("guard") or {}).get("device"),
                            "partial": p["partial"]} for pp, p in zip(paths, parts)]}
    res["notes"] = list(parts[0]["notes"]) + [f"Merged from {len(parts)} runs on disjoint map ids (same seeds per map as a serial run); pooled statistics "
                                              "recomputed from the per-map paired counts."]
    return res


def pooled(oks: dict) -> dict:
    out = {}
    for cond, d in oks.items():
        if not d["classic"]:
            continue
        okc, okp = np.concatenate(d["classic"]), np.concatenate(d["pilot"])
        b, c = int((okp & ~okc).sum()), int((~okp & okc).sum())
        dd, lo, hi = paired_diff_ci(b, c, okc.size)
        # map bootstrap: resample whole maps, so map-to-map variation is in the interval
        rng = np.random.default_rng(0)
        mc, mp = np.array([x.mean() for x in d["classic"]]), np.array([x.mean() for x in d["pilot"]])
        k = len(mc)
        idx = rng.integers(k, size=(4000, k))
        boot = lambda v: [float(np.percentile(v[idx].mean(1), 2.5)), float(np.percentile(v[idx].mean(1), 97.5))]  # noqa: E731
        out[cond] = {"maps": k, "missions": int(okc.size), "classic": float(okc.mean()), "pilot": float(okp.mean()),
                     "classic_ci95_wilson": list(wilson(int(okc.sum()), okc.size)), "pilot_ci95_wilson": list(wilson(int(okp.sum()), okp.size)),
                     "classic_ci95_map_bootstrap": boot(mc), "pilot_ci95_map_bootstrap": boot(mp), "diff": dd, "diff_ci95": [lo, hi],
                     "diff_ci95_map_bootstrap": boot(mp - mc), "only_pilot_landed": b, "only_classic_landed": c, "sign_test_p": sign_test(b, c),
                     "maps_pilot_better": int((mp > mc).sum()), "maps_classic_better": int((mp < mc).sum())}
    return out


# ------------------------------------------------------------------------------------------------------------------ outputs
ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"  # noqa: E731


def md_rehearsal(r: dict) -> str:
    t, p = r["target"], r["plan"]
    lines = [f"# Mission rehearsal: {p['region']} plan cell {tuple(p['target_cell_rc'])} ({r['label']})", "",
             f"Plan `{r['plan_file']}`: decision {p['decision']}, landing {p['landing_time_utc']}, backup ranking used: {p['backup_used']}. "
             f"Cell rc {t['cell_rc_rover_window']} in the {512} px rover window = tile rc {t['cell_rc_tile']} = xy {[round(v, 1) for v in t['xy_m']]} m. "
             f"Truth slope {t['truth_slope_deg']:.1f} deg, truth hazard {t['truth_hazard']:.0f}, in safe target cells: {t['in_safe_target_cells']}; "
             f"truth hazard within 50 m: {t['truth_hazard_frac_within_50m']:.1%}.", "",
             f"Pilot `{r['pilot']}` {r['checkpoint_id']} ({r['parameters']:,} params). {r['missions_per_condition']} missions per condition, same missions for both.", "",
             "| condition | classic (95% CI) | pilot (95% CI) | pilot - classic (95% CI) | pilot-only / classic-only | exact p | miss p95 C / P (m) |",
             "|---|---|---|---|---|---|---|"]
    for cond, x in r["results"].items():
        lines.append(f"| {cond} | {x['classic']['success_rate']:.4f} {ci(x['classic']['ci95'])} | {x['pilot']['success_rate']:.4f} {ci(x['pilot']['ci95'])} | "
                     f"{x['diff']:+.4f} {ci(x['diff_ci95'])} | {x['only_pilot_landed']} / {x['only_classic_landed']} | {x['sign_test_p']:.3f} | "
                     f"{x['classic']['miss_p95_m_success'] or float('nan'):.1f} / {x['pilot']['miss_p95_m_success'] or float('nan'):.1f} |")
    lines += ["", "Failures: " + "; ".join(f"{cond} classic {x['classic']['failures']}, pilot {x['pilot']['failures']}" for cond, x in r["results"].items())]
    if r["partial"]:
        lines += ["", f"**PARTIAL**: {r['stopped_because']}"]
    lines += [""] + [f"- {n}" for n in r["notes"]]
    return "\n".join(lines) + "\n"


def md_procedural(r: dict) -> str:
    lines = [f"# Procedural map benchmark ({GENERATED})", "",
             f"Pilot `{r['pilot']}` {r['checkpoint_id']}; {r['missions_per_map_per_condition']} missions per map per condition, paired. "
             f"Generator check vs make_demo_tile: max |dh| {r['generator_max_abs_height_diff_vs_make_demo_tile_m']:.1e} m.", "",
             "| map | craters | tilt deg | slope median / p90 | truth hazard | belief wrong | nominal C / P | harsh C / P | harsh P - C (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in r["maps"]:
        n_, h_ = m["results"]["nominal"], m["results"]["harsh"]
        lines.append(f"| {m['region']} | {m['craters_drawn']} | {m['tilt_deg']:.1f} | {m['slope_median_deg']:.1f} / {m['slope_p90_deg']:.1f} | "
                     f"{m['truth_hazard_frac']:.1%} | {m['belief_truth_disagree_frac']:.1%} | {n_['classic']['success_rate']:.3f} / {n_['pilot']['success_rate']:.3f} | "
                     f"{h_['classic']['success_rate']:.3f} / {h_['pilot']['success_rate']:.3f} | {h_['diff']:+.3f} {ci(h_['diff_ci95'])} |")
    for cond, s in r.get("pooled", {}).items():
        lines.append(f"| **POOLED {s['maps']} maps** | | | | | | {cond}: C {s['classic']:.4f} {ci(s['classic_ci95_map_bootstrap'])} / P {s['pilot']:.4f} "
                     f"{ci(s['pilot_ci95_map_bootstrap'])} | | {s['diff']:+.4f} {ci(s['diff_ci95'])}; map bootstrap {ci(s['diff_ci95_map_bootstrap'])}; "
                     f"{s['only_pilot_landed']} / {s['only_classic_landed']}, p={s['sign_test_p']:.4f} |")
    for s in r["skipped"]:
        lines.append(f"| {s['region']} | {s['craters_drawn']} | {s['tilt_deg']:.1f} | {s['slope_median_deg']:.1f} / {s['slope_p90_deg']:.1f} | "
                     f"{s['truth_hazard_frac']:.1%} | | skipped: {s['why']} | | |")
    rl = r["real_lola_slope_stats"]
    lines += ["", "Real LOLA tiles for comparison (official slope): " + ", ".join(f"{k[7:]} {v['median']:.1f}/{v['p90']:.1f}" for k, v in rl.items()) + " (median/p90 deg)."]
    if r["partial"]:
        lines += ["", f"**PARTIAL**: {r['stopped_because']}"]
    lines += [""] + [f"- {n}" for n in r["notes"]]
    return "\n".join(lines) + "\n"


def write_web(rehearsal_res: dict | None, proc_res: dict | None) -> None:
    cur = json.loads(WEB_JSON.read_text()) if WEB_JSON.exists() else {}
    if rehearsal_res is not None:
        cur["rehearsal"] = rehearsal_res
    if proc_res is not None:
        cur["procedural"] = proc_res
    cur["meta"] = {"provenance": "modelled (rehearsal: NASA LOLA terrain) / generated (procedural maps: GENERATED — NOT OBSERVED)",
                   "source": "worldmodel.lander.mission_bench", "frame": "tile ENU m", "units": "success fraction, m", "built": datetime.now(UTC).isoformat()}
    WEB_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEB_JSON.write_text(json.dumps(cur, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["time", "rehearsal", "procedural", "merge"])
    ap.add_argument("--map-ids", type=int, nargs="*", default=None, help="procedural: only these map indices (parallel workers)")
    ap.add_argument("--stem", default="procedural_benchmark", help="procedural output file stem (parallel workers: procedural_part_<k>)")
    ap.add_argument("--parts", nargs="*", default=[], help="merge: procedural part JSONs to join into <runs>/lander/procedural_benchmark.*")
    ap.add_argument("--name", default="lander_pilot")
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--n", type=int, default=1024, help="missions per condition (rehearsal) or per map per condition (procedural)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-free-gb", type=float, default=1.5)
    ap.add_argument("--max-rss-gb", type=float, default=2.5)
    ap.add_argument("--no-web", action="store_true")
    a = ap.parse_args(argv)
    if a.job == "merge":                                                     # no model, no flights
        res = merge_parts(a.parts)
        out = run_dir("lander")
        out.mkdir(parents=True, exist_ok=True)
        (out / "procedural_benchmark.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        (out / "procedural_benchmark.md").write_text(md_procedural(res), encoding="utf-8")
        if not a.no_web:
            write_web(None, res)
        print(md_procedural(res))
        return 0
    cfg = load_mission()
    device = pick_device(a.device)
    model, info = load_checkpoint(model_path(a.name), device)
    guard = Guard(a.min_free_gb, a.max_rss_gb)
    out = run_dir("lander")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, pilot {info.get('checkpoint_id')} ({info.get('parameters'):,} params), rss {rss_gb():.2f} GB", flush=True)

    if a.job == "time":
        scene = scene_for_region("lola5m_site23", cfg, device, "backup")
        params, gp = condition_params(cfg, 1.0)
        for pol, mdl in (("classic", None), ("pilot", model)):
            guard.check(f"timing {pol}")
            t = time.time()
            m = fly(scene, params, gp, mdl, a.batch, 123, 5, device=device)
            float(m["success"].float().mean())
            print(f"{pol}: {a.batch} landers in {time.time() - t:.1f} s, success {m['success'].float().mean():.3f}, rss {rss_gb():.2f} GB"
                  + (f", mps {torch.mps.driver_allocated_memory() / 1e9:.2f} GB" if device.type == "mps" else ""), flush=True)
        t = time.time()
        masks = borrowed_masks(info.get("trained_on", [])[:1])
        tile, knobs = procedural_tile(0, a.size, 5.0, masks, cfg.lander.touchdown_limits.max_slope_deg, a.seed_base)
        backup_hazard(tile, cfg)
        print(f"procedural map build + belief: {time.time() - t:.1f} s ({knobs['craters_drawn']} craters); rss {rss_gb():.2f} GB", flush=True)
        return 0

    def write(res: dict, final: bool = True, stem: str = a.stem, md=md_procedural) -> None:
        res = {**res, "guard": {"min_free_gb_seen": round(guard.min_free, 2), "peak_rss_gb": round(guard.peak_rss, 2), "device": str(device)}}
        (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        (out / f"{stem}.md").write_text(md(res), encoding="utf-8")
        if not a.no_web:
            write_web(res if stem == "mission_rehearsal" else None, res if stem == "procedural_benchmark" else None)

    if a.job == "rehearsal":
        res = rehearsal(a, cfg, device, model, info, guard)
        write(res, stem="mission_rehearsal", md=md_rehearsal)
        print(md_rehearsal(res))
    else:
        res = procedural(a, cfg, device, model, info, guard, write)
        write(res)
        print(md_procedural(res))
    print(f"peak rss {guard.peak_rss:.2f} GB, min free disk {guard.min_free:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
