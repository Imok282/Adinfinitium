"""Simulated mission outcomes = the planner's training labels (no human labels, no made-up numbers).

For each landing CELL of a (region, month) world:
  * LANDER: n_land landers are flown to that exact cell by classic ZEM/ZEV guidance (or a learned pilot) in the PyTorch lander twin -> p_land and the
    touchdown miss distances (the scatter radius).
  * ROVER: n_rover rovers start around the cell at each chosen landing hour and are driven by the physics planner baseline (Dijkstra + turn-back rule)
    in the rover environment -> p_ice, p_return.
Mission success = landed AND rover reached ice AND returned; p_mission = p_land x p_return-with-ice (independent draws). One .npz per (region, month),
resumable. Cost is dominated by the rover Dijkstra fields (~seconds per cell), so `--max-minutes` bounds a run.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from contracts.mission import MissionConfig
from contracts.paths import data_root
from worldmodel.lander.env import make_params
from worldmodel.lander.fly import fly
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.scenes import scene_for_region
from worldmodel.planner.features import candidate_features, navigator_score
from worldmodel.rover.env import RoverEnv
from worldmodel.rover.world import RoverWorld, build_world, choose_targets

N_HOURS = 709


def label_path(region: str, date: str) -> Path:
    return data_root() / "planner" / f"{region}_{date}.npz"


def pick_cells(w: RoverWorld, n: int, rng: np.random.Generator, margin: int = 60) -> np.ndarray:
    """Landing-cell candidates inside the rover world, on ground the hazard map calls safe over a 7 x 7 neighbourhood."""
    from numpy.lib.stride_tricks import sliding_window_view

    unsafe = w.hazard >= 0.5
    ok = ~sliding_window_view(np.pad(unsafe, 3, constant_values=True), (7, 7)).any(axis=(-1, -2))
    ok[:margin], ok[-margin:], ok[:, :margin], ok[:, -margin:] = False, False, False, False
    cand = np.argwhere(ok)
    if len(cand) == 0:
        return np.zeros((0, 2))
    return cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)].astype(np.float64)


def rover_outcomes(w: RoverWorld, cell: np.ndarray, hours: list[int], n: int, cfg: MissionConfig, device, seed: int) -> list[tuple[float, float]]:
    """(p_ice, p_ice_and_return) per landing hour for rovers driven by the physics planner, landing at `cell`."""
    targets, _ = choose_targets(w.psr, w.slope, tuple(cell), w.mpp, cfg.rover.max_slope_deg, cfg.rover_sim.n_ice_targets, np.random.default_rng(seed))
    w2 = dataclasses.replace(w, lander_rc=(float(cell[0]), float(cell[1])), targets_rc=targets)
    out = []
    for h in hours:
        env = RoverEnv(w2, n, device, cfg, seed=seed)
        env.reset()
        env.t[:] = h
        ice = torch.zeros(n, dtype=torch.bool, device=device)
        ret = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(env.rs.max_steps):
            _, _, done, info = env.step(env.planner_action(turn_back=True))
            ice |= info["reached_ice"]
            ret |= info["returned"] & ice
            if bool(done.all()):
                break
        out.append((float(ice.float().mean()), float(ret.float().mean())))
    return out


def label_world(region: str, date: str, cfg: MissionConfig, device, cells: int = 12, hours: tuple[int, ...] = (24, 96, 168, 240, 312, 384), n_land: int = 48,
                n_rover: int = 32, seed: int = 0, belief: str = "backup", force: bool = False) -> Path:
    path = label_path(region, date)
    if path.exists() and not force:
        return path
    w = build_world(region, date, cfg, seed=seed, device=device)
    rng = np.random.default_rng(seed)
    cs = pick_cells(w, cells, rng)
    scene = scene_for_region(region, cfg, device, belief)
    assert scene is not None
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    r0, c0 = w.origin_rc
    tile_rc = cs + np.array([r0, c0])
    xy = torch.as_tensor(scene.terrain.cell_to_xy(tile_rc), dtype=torch.float32, device=device)
    m = fly(scene, params, gp, None, len(cs) * n_land, seed + 5, 5, device=device, targets=xy.repeat_interleave(n_land, dim=0))
    ok = m["success"].cpu().numpy().reshape(len(cs), n_land)
    dist = m["target_dist_m"].cpu().numpy().reshape(len(cs), n_land)
    p_land = ok.mean(1)
    scatter = np.array([np.percentile(dist[i][ok[i]], 95) if ok[i].any() else 300.0 for i in range(len(cs))])
    stride = cfg.rover_sim.light_stride_px
    feats, y_ice, y_ret, p_mission, cell_idx, hr = [], [], [], [], [], []
    for i, cell in enumerate(cs):
        outs = rover_outcomes(w, cell, list(hours), n_rover, cfg, device, seed + i)
        for h, (pi, pr) in zip(hours, outs, strict=True):
            f = candidate_features(w, tuple(cell), h, stride)
            feats.append(f)
            y_ice.append(pi); y_ret.append(pr); p_mission.append(p_land[i] * pr); cell_idx.append(i); hr.append(h)
    feats = np.array(feats, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=feats, p_ice=np.array(y_ice, np.float32), p_return=np.array(y_ret, np.float32), p_mission=np.array(p_mission, np.float32),
                        p_land=p_land[np.array(cell_idx)].astype(np.float32), scatter_m=scatter[np.array(cell_idx)].astype(np.float32), cell_rc=cs[np.array(cell_idx)],
                        hour=np.array(hr), n_land=n_land, n_rover=n_rover, backup=np.array([navigator_score(f) for f in feats], np.float32),
                        region=region, date=date, lander_pilot="classic ZEM/ZEV (physics)", provenance="modelled: simulated outcomes on real LOLA terrain and SPICE light")
    return path


def generate(worlds: list[tuple[str, str]], cfg: MissionConfig, device, max_minutes: float = 0.0, **kw) -> list[Path]:
    t0, out = time.time(), []
    for region, date in worlds:
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print(f"time budget reached after {len(out)} worlds", flush=True)
            break
        t1 = time.time()
        try:
            out.append(label_world(region, date, cfg, device, **kw))
            print(f"labelled {region}@{date} in {time.time() - t1:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad world must not stop the run; recorded
            print(f"FAILED {region}@{date}: {e!r}", flush=True)
    return out
