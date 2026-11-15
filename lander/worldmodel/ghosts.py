"""Ghost-trajectory export: REAL simulated runs (successes and failures) for the Basilisk/Vizard, web UI and Blender "ghost" shots.

    python -m worldmodel.ghosts --kind rover  --region lola5m_site23 --date 2027-03-01 --n 24
    python -m worldmodel.ghosts --kind lander --region lola5m_site23 --n 48

Writes runs/ghosts/rover_<region>_<date>.json and runs/ghosts/lander_<region>.json. Frame: tile ENU metres, origin = tile centre, x east,
y north, z = LOLA height. Every run is replayed exactly as simulated; nothing is invented or tuned. "best" marks ONE run: the fastest rover that
reached the ice edge and returned (else the one that got furthest), the lander that landed closest to its target with the least fuel.
Provenance: MODELLED (simulation on NASA LOLA terrain and SPICE light). Ice targets = PSR edges (dark and cold, not confirmed ice).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import run_dir, tile_path
from contracts.runs import FAILURE_REASONS
from contracts.tiles import load_tile


def _tile_xyz(tile, rows, cols):
    h, w = tile.shape
    mpp = tile.meters_per_pixel
    x = (np.asarray(cols) - (w - 1) / 2.0) * mpp
    y = ((h - 1) / 2.0 - np.asarray(rows)) * mpp
    r0 = np.clip(np.floor(rows).astype(int), 0, h - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, w - 2)
    fr, fc = np.asarray(rows) - r0, np.asarray(cols) - c0
    H = tile.height
    z = H[r0, c0] * (1 - fr) * (1 - fc) + H[r0, c0 + 1] * (1 - fr) * fc + H[r0 + 1, c0] * fr * (1 - fc) + H[r0 + 1, c0 + 1] * fr * fc
    return np.stack([x, y, z], -1)


def export_rover(region: str, date: str, n: int, seed: int = 7, learned: str | None = None) -> dict:
    from worldmodel.rover.env import RoverEnv
    from worldmodel.rover.world import build_world

    cfg = load_mission()
    dev = torch.device("cpu")
    w = build_world(region, date, cfg, device=dev)
    tile = load_tile(tile_path(region))
    r0, c0 = w.origin_rc
    policies = {"naive": lambda e: e.planner_action(turn_back=False), "cautious": lambda e: e.planner_action(turn_back=True),
                "wait_light": lambda e: e.planner_action(turn_back=True, wait_for_light=True)}
    if learned:
        from contracts.paths import model_path
        from worldmodel.rover.policy import load_checkpoint

        ck = model_path(learned)
        if ck.exists():
            model, _ = load_checkpoint(ck, dev)
            policies["learned"] = lambda e, m=model: m(e.observe())[0].argmax(1)
    runs = []
    for pname, fn in policies.items():
        env = RoverEnv(w, n, dev, cfg, seed=seed)                                # same starts for every policy
        env.reset()
        path = [env.pos.clone()]
        ice = torch.zeros(n, dtype=torch.bool)
        ret = torch.zeros(n, dtype=torch.bool)
        dead = torch.zeros(n, dtype=torch.bool)
        tout = torch.zeros(n, dtype=torch.bool)
        end_step = torch.full((n,), -1, dtype=torch.long)
        for step in range(env.rs.max_steps):
            with torch.no_grad():
                a = fn(env)
            _, _, done, info = env.step(a)
            path.append(env.pos.clone())
            ice |= info["reached_ice"]
            ret |= info["returned"] & ice
            dead |= info["dead"]
            tout |= info["timeout"]
            end_step = torch.where(info["newly_done"] & (end_step < 0), torch.full_like(end_step, step + 1), end_step)
            if bool(done.all()):
                break
        P = torch.stack(path).numpy()                                            # [T+1, n, 2] world px (row, col)
        for i in range(n):
            T = int(end_step[i]) if end_step[i] >= 0 else P.shape[0] - 1
            rows, cols = P[: T + 1, i, 0] + r0, P[: T + 1, i, 1] + c0
            xyz = _tile_xyz(tile, rows, cols)
            d = np.diff(xyz[:, :2], axis=0)
            heading = np.degrees(np.arctan2(d[:, 1], d[:, 0])) if len(d) else np.zeros(0)
            heading = np.concatenate([heading[:1] if len(heading) else [0.0], heading])
            outcome = ("returned" if ret[i] else "died_battery" if dead[i] else "reached_ice_only" if ice[i] else "timeout")
            runs.append({"id": f"{pname}_{i:02d}", "policy": pname, "outcome": outcome, "best": False,
                         "fail_t_s": None if outcome == "returned" else float(T * env.rs.step_s),
                         "t_s": [float(k * env.rs.step_s) for k in range(T + 1)], "xyz_m": np.round(xyz, 2).tolist(),
                         "heading_deg": np.round(heading, 1).tolist(), "reached_ice": bool(ice[i])})
    ok = [r for r in runs if r["outcome"] == "returned"]
    if ok:
        min(ok, key=lambda r: r["t_s"][-1])["best"] = True
    else:
        lx, ly = _tile_xyz(tile, [w.lander_rc[0] + r0], [w.lander_rc[1] + c0])[0][:2]
        max(runs, key=lambda r: max(np.hypot(np.array(r["xyz_m"])[:, 0] - lx, np.array(r["xyz_m"])[:, 1] - ly)))["best"] = True
    lander_xyz = _tile_xyz(tile, [w.lander_rc[0] + r0], [w.lander_rc[1] + c0])[0]
    targets = _tile_xyz(tile, w.targets_rc[:, 0] + r0, w.targets_rc[:, 1] + c0)
    return {"kind": "rover", "region": region, "date": date, "frame": "tile ENU metres, origin = tile centre, x east, y north, z = LOLA height",
            "dt_s": float(cfg.rover_sim.step_s), "provenance": "MODELLED: rover environment on NASA LOLA terrain + SPICE/skyline light; ice targets = PSR edges",
            "lander_xyz_m": np.round(lander_xyz, 2).tolist(), "ice_targets_xyz_m": np.round(targets, 2).tolist(), "ice_source": w.ice_source,
            "models": {"rover": "vizard/assets/blender/rover_B.obj", "rover_glb": "vizard/assets/ui_models/rover.glb"}, "runs": runs}


def export_lander(region: str, n: int, seed: int = 11, pilot: str | None = None) -> dict:
    from worldmodel.lander.env import LanderBatch, make_params
    from worldmodel.lander.guidance import classic_action, guidance_params
    from worldmodel.lander.pilot import authority, pilot_features, residual_action
    from worldmodel.lander.scenes import scene_for_region

    cfg = load_mission()
    dev = torch.device("cpu")
    scene = scene_for_region(region, cfg, dev)
    params = make_params(cfg)
    params.p_engine_out, params.p_nav_drift, params.p_camera_blind = 0.3, 0.3, 0.15  # more failures than the config, so the failure modes are visible
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    rng = np.random.default_rng(seed)
    target = scene.sample_targets(1, rng, dev).repeat(n, 1)                      # one landing cell, many attempts
    flyers = {"classic": None, "open_loop": "open_loop"}                         # open_loop: fixed thrust, no sensors (shows why guidance matters)
    if pilot:
        from contracts.paths import model_path
        from worldmodel.lander.pilot import load_checkpoint

        ck = model_path(pilot)
        if ck.exists():
            flyers["pilot"] = load_checkpoint(ck, dev)[0]
    runs = []
    for name, model in flyers.items():
        env = LanderBatch(params, scene.terrain, n, dev, seed=seed, targets=target)
        obs = env.observe()
        path, r = [env.pos.clone()], torch.zeros(n, 3)
        for i in range(params.n_steps):
            classic = classic_action(gp, obs)
            if isinstance(model, str):                                           # open loop: thrust fixed in advance (~0.5 m/s2 braking), no feedback at all
                act = torch.zeros(n, 3)
                act[:, 2] = params.start_mass * (params.g + 0.5) / params.max_total_thrust
            elif model is not None:
                if i % 5 == 0:
                    with torch.no_grad():
                        r = model(pilot_features(obs, classic), env.crop(32))[0]
                act = residual_action(classic, r, authority(obs, getattr(model, "rule", None)))
            else:
                act = classic
            obs, _, _ = env.step(act)
            if i % 10 == 9:
                path.append(env.pos.clone())
            if bool(env.done.all()):
                path.append(env.pos.clone())
                break
        P = torch.stack(path).numpy()
        m = env.metrics()
        for k in range(n):
            T = int(np.ceil(float(m["time_s"][k]))) + 1
            xyz = P[: min(T, P.shape[0]), k]
            oc = int(m["outcome"][k])
            outcome = FAILURE_REASONS[oc] if oc >= 0 else "timeout"
            runs.append({"id": f"{name}_{k:02d}", "policy": name, "outcome": outcome, "best": False,
                         "fail_t_s": None if outcome == "ok" else float(m["time_s"][k]), "t_s": [float(j) for j in range(len(xyz))],
                         "xyz_m": np.round(xyz, 2).tolist(), "target_dist_m": round(float(m["target_dist_m"][k]), 2),
                         "fuel_used_kg": round(float(m["fuel_used_kg"][k]), 1), "touchdown_vz_m_s": round(float(m["td_vz"][k]), 2),
                         "engine_failed": bool(m["engine_failed"][k]), "nav_drift": bool(m["nav_drift"][k]), "camera_blind": bool(m["camera_blind"][k])})
    ok = [r for r in runs if r["outcome"] == "ok" and r["policy"] != "open_loop"]
    if ok:
        min(ok, key=lambda r: r["target_dist_m"] + 0.05 * r["fuel_used_kg"])["best"] = True
    return {"kind": "lander", "region": region, "frame": "tile ENU metres, origin = tile centre, x east, y north, z = LOLA height", "dt_s": 1.0,
            "target_xyz_m": [round(float(target[0, 0]), 2), round(float(target[0, 1]), 2), round(float(scene.terrain.height_at(target[:1, 0], target[:1, 1])[0]), 2)],
            "provenance": "MODELLED: PyTorch lander twin (parity-tested vs sim_jax) on NASA LOLA terrain; failures injected at raised rates for the visual",
            "failure_rates": {"engine_out": params.p_engine_out, "nav_drift": params.p_nav_drift, "camera_blind": params.p_camera_blind},
            "models": {"lander": "vizard/assets/blender/lander_B.obj", "lander_glb": "vizard/assets/ui_models/lander.glb"}, "runs": runs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["rover", "lander", "both"], default="both")
    ap.add_argument("--region", default="lola5m_site06")
    ap.add_argument("--date", default="2027-03-01")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--learned", default=None, help="rover checkpoint name to add (e.g. rover_driver_mini)")
    ap.add_argument("--pilot", default=None, help="lander pilot checkpoint name to add (e.g. lander_pilot)")
    a = ap.parse_args(argv)
    out = run_dir("ghosts")
    out.mkdir(parents=True, exist_ok=True)
    for kind in (("rover", "lander") if a.kind == "both" else (a.kind,)):
        d = export_rover(a.region, a.date, a.n, learned=a.learned) if kind == "rover" else export_lander(a.region, a.n * 2, pilot=a.pilot)
        p = out / (f"rover_{a.region}_{a.date}.json" if kind == "rover" else f"lander_{a.region}.json")
        p.write_text(json.dumps(d), encoding="utf-8")
        counts = {}
        for r in d["runs"]:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        best = next((r["id"] for r in d["runs"] if r["best"]), None)
        print(f"{kind}: {len(d['runs'])} runs {counts} best={best} -> {p} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
