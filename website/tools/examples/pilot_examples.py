"""Recorded descents of the learned lander pilot vs classic ZEM/ZEV guidance on Site11 -> website/assets/examples/pilot/.

    ~/venvs/adi/bin/python -I website/tools/examples/pilot_examples.py

Read-only use of data/models/lander_pilot.pt (no training). Site11 is the pilot's VALIDATION map (never a test map).
1) fly 128 missions with classic guidance only (mission rates from config/mission.yaml failures) to draw scenarios;
2) pick the first nominal mission, the first engine-out-only mission and the first nav-drift-only mission (camera working);
3) replay those 3 identical missions (same initial state, same failure times, same sensor-noise seed) with classic alone and
   with the pilot, recording every 1 s: altitude, velocity, classic command, pilot residual, final command, fuel; plus the
   32x32 camera crop the pilot saw at 4 moments.
Runs on CPU so it does not compete with GPU jobs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from contracts.mission import load_mission  # noqa: E402
from contracts.paths import model_path  # noqa: E402
from contracts.runs import FAILURE_REASONS  # noqa: E402
from worldmodel.lander.env import LanderBatch, make_params  # noqa: E402
from worldmodel.lander.guidance import classic_action, guidance_params  # noqa: E402
from worldmodel.lander.pilot import RESIDUAL_SCALE, authority, load_checkpoint, pilot_features, residual_action  # noqa: E402
from worldmodel.lander.scenes import load_scenes  # noqa: E402

OUT = ROOT / "website/assets/examples/pilot"
SITE, SEED, N_DRAW, REPEAT, CROP = "Site11", 70_311, 128, 5, 32
REC_EVERY = 10  # steps (dt 0.1 s -> 1 s)


def r(a, d=2):
    return [round(float(v), d) for v in np.asarray(a).ravel()]


@torch.no_grad()
def replay(scene, params, gp, model, sc, seed, device):
    n = len(sc["pos"])
    env = LanderBatch(params, scene.terrain, n, device, seed=seed, targets=torch.as_tensor(sc["target"], device=device))
    env.load_scenarios(torch.arange(n, device=device), sc)
    obs = env.observe()
    res = torch.zeros(n, 3, device=device)
    rec = [dict(t=[], alt_m=[], alt_meas_m=[], vx=[], vy=[], vz=[], dx_m=[], dy_m=[], classic=[], residual_raw=[], residual_applied=[],
                final=[], fuel_kg=[], engines_ok=[], thrust_n=[]) for _ in range(n)]
    crops = [[] for _ in range(n)]
    for i in range(params.n_steps):
        classic = classic_action(gp, obs)
        if model is not None:
            if i % REPEAT == 0:
                img = env.crop(CROP)
                res = model(pilot_features(obs, classic), img)[0].float()
                for k in range(n):
                    if not bool(env.done[k]):
                        alt = float(env.pos[k, 2] - scene.terrain.height_at(env.pos[k:k + 1, 0], env.pos[k:k + 1, 1])[0])
                        crops[k].append((round(float(env.t[k]), 1), alt, img[k].cpu().numpy()))
            act = residual_action(classic, res, authority(obs, getattr(model, "rule", None)))
        else:
            act = classic
        if i % REC_EVERY == 0:
            ground = scene.terrain.height_at(env.pos[:, 0], env.pos[:, 1])
            for k in range(n):
                if bool(env.done[k]):
                    continue
                R = rec[k]
                R["t"].append(round(float(env.t[k]), 1))
                R["alt_m"].append(round(float(env.pos[k, 2] - ground[k]), 1))
                R["alt_meas_m"].append(round(float(obs[k, 2] * 1000), 1))
                R["vx"].append(round(float(env.vel[k, 0]), 2)); R["vy"].append(round(float(env.vel[k, 1]), 2)); R["vz"].append(round(float(env.vel[k, 2]), 2))
                R["dx_m"].append(round(float(env.pos[k, 0] - env.target[k, 0]), 1)); R["dy_m"].append(round(float(env.pos[k, 1] - env.target[k, 1]), 1))
                R["classic"].append(r(classic[k].clamp(-1, 1), 4))
                R["residual_raw"].append(r(res[k], 4) if model is not None else [0.0, 0.0, 0.0])
                R["residual_applied"].append(r(act[k] - classic[k].clamp(-1, 1), 4))
                R["final"].append(r(act[k], 4))
                R["fuel_kg"].append(round(float(env.mass[k] - params.dry_mass), 2))
                R["engines_ok"].append(int(env.engines_ok[k]))
        obs, _, _ = env.step(act)
        if i % REC_EVERY == 0:
            for k in range(n):
                if len(rec[k]["thrust_n"]) < len(rec[k]["t"]):
                    rec[k]["thrust_n"].append(r(env.thrust[k], 0))
        if bool(env.done.all()):
            break
    m = env.metrics()
    outs = []
    for k in range(n):
        oc = int(m["outcome"][k])
        outs.append({"outcome": FAILURE_REASONS[oc] if oc >= 0 else "still_flying", "success": bool(m["success"][k]),
                     "touchdown_time_s": round(float(m["time_s"][k]), 1), "miss_distance_m": round(float(m["target_dist_m"][k]), 2),
                     "fuel_used_kg": round(float(m["fuel_used_kg"][k]), 2), "td_vertical_speed_m_s": round(float(m["td_vz"][k]), 3),
                     "td_horizontal_speed_m_s": round(float(m["td_vh"][k]), 3), "td_slope_deg": round(float(m["td_slope"][k]), 2)})
    return rec, crops, outs


def crop_image(img, path):
    import matplotlib
    up = 4
    panels = []
    for ch, cmap, lo, hi in ((0, "magma", 0.0, 2.0), (1, "RdYlGn_r", 0.0, 1.0), (2, "cividis", None, None)):
        a = img[ch]
        if lo is None:
            lo, hi = float(a.min()), float(a.max()) + 1e-6
        rgb = (matplotlib.colormaps[cmap](np.clip((a - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
        panels.append(np.kron(rgb, np.ones((up, up, 1), np.uint8)))
    gap = np.full((CROP * up, 4, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)).save(path, "WEBP", quality=88, method=6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    cfg = load_mission()
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    scene = load_scenes([SITE], cfg, device, "backup", 1)[0]
    model, info = load_checkpoint(model_path("lander_pilot"), device)

    # 1) draw scenarios (classic only, cheap)
    rng = np.random.default_rng(SEED)
    env = LanderBatch(params, scene.terrain, N_DRAW, device, seed=SEED, targets=scene.sample_targets(N_DRAW, rng, device))
    init = env.scenarios()
    eo = np.isfinite(init["engine_out_t"]); dr = np.linalg.norm(init["nav_bias"], axis=1) > 0; cam = init["camera_ok"]
    pick = {"nominal": int(np.nonzero(~eo & ~dr & cam)[0][0]),
            "engine_out": int(np.nonzero(eo & ~dr & cam)[0][0]),
            "nav_drift": int(np.nonzero(~eo & dr & cam)[0][0])}
    idx = list(pick.values())
    sc = {k: v[idx] for k, v in init.items()}
    print("picked", pick, "engine_out_t", sc["engine_out_t"], "nav_bias", sc["nav_bias"])

    # 2) same 3 missions, classic alone and with the pilot (same replay seed => same sensor noise draws)
    rc, _, oc = replay(scene, params, gp, None, sc, SEED + 1, device)
    rp, crops, op = replay(scene, params, gp, model, sc, SEED + 1, device)

    missions = []
    for j, (kind, mi) in enumerate(pick.items()):
        cr = crops[j]
        alts = np.array([c[1] for c in cr])
        # 4 moments: start, ~500 m, ~50 m, last frame before touchdown
        sel = sorted({0, int(np.argmin(np.abs(alts - 500))), int(np.argmin(np.abs(alts - 50))), len(cr) - 1})
        imgs = []
        for s in sel:
            t, alt, im = cr[s]
            name = f"{kind}_crop_t{int(round(t)):03d}.webp"
            crop_image(im, OUT / name)
            imgs.append({"file": name, "t_s": t, "altitude_m": round(alt, 1),
                         "ground_span_m": round(float(np.clip(max(alt, 0) + 60.0, 60.0, 640.0)), 1)})
        missions.append({
            "kind": kind, "mission_index_in_draw": mi,
            "injected": {"engine_out_at_s": None if not np.isfinite(sc["engine_out_t"][j]) else round(float(sc["engine_out_t"][j]), 1),
                         "nav_drift_bias_m_s": r(sc["nav_bias"][j], 3), "camera_ok": bool(sc["camera_ok"][j])},
            "start": {"altitude_m": rp[j]["alt_m"][0], "offset_m": [rp[j]["dx_m"][0], rp[j]["dy_m"][0]],
                      "velocity_m_s": [rp[j]["vx"][0], rp[j]["vy"][0], rp[j]["vz"][0]]},
            "outcome": {"pilot": op[j], "classic": oc[j]},
            "pilot_series": rp[j], "classic_series": rc[j], "camera_crops": imgs,
        })
        print(kind, "pilot", op[j]["outcome"], op[j]["miss_distance_m"], "m |", "classic", oc[j]["outcome"], oc[j]["miss_distance_m"], "m")

    doc = {
        "model": "lander pilot (PilotFormer, residual on classic ZEM/ZEV guidance)", "checkpoint": "data/models/lander_pilot.pt",
        "checkpoint_id": info["checkpoint_id"], "parameters": info.get("parameters"), "trained_on": info["trained_on"],
        "validated_on": info["val_regions"],
        "map": "lola5m_site11 (Site11, the pilot's VALIDATION map: used to pick the checkpoint, not a test map)",
        "split": "validation", "hazard_belief": "slope-rule backup map from sparse laser pixels",
        "selection": f"first nominal / engine-out-only / nav-drift-only mission among {N_DRAW} drawn with seed {SEED} at config failure rates "
                     "(not chosen by outcome)",
        "pairing": "classic and pilot fly the identical mission: same initial state, failure time, drift vector and sensor-noise seed",
        "command": "thrust command in [-1,1] per axis (x east, y north, z up) as a fraction of total max thrust; "
                   f"final = clip(classic + {RESIDUAL_SCALE} * authority * clip(residual_raw)); authority = 1 (no rule in this checkpoint); "
                   f"pilot re-plans every {REPEAT} steps (0.5 s)",
        "series_fields": {"t": "s", "alt_m": "true altitude above ground", "alt_meas_m": "altimeter reading (noisy)",
                          "vx,vy,vz": "true velocity m/s", "dx_m,dy_m": "true offset from target", "classic": "classic command [3]",
                          "residual_raw": "pilot network output [3]", "residual_applied": "final - classic [3]", "final": "command sent [3]",
                          "fuel_kg": "propellant left", "engines_ok": "working engines", "thrust_n": "thrust after engine limits [3] N"},
        "series_step_s": REC_EVERY * params.dt,
        "camera": {"size_px": CROP, "upscale": 4, "panels": ["slope / 12 deg limit (magma, 0..2)", "hazard belief (green safe .. red unsafe)",
                                                             "height relative to lander nadir (cividis, per-image range)"],
                   "footprint": "north-up, centred under the lander, side = clip(altitude + 60 m, 60, 640) m"},
        "provenance": "real LOLA terrain (observed); trajectories SIMULATED (modelled)",
        "missions": missions,
    }
    p = OUT / "pilot_examples.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
