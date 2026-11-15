"""3D figure data for research/pilot-benchmark.html: PAIRED descents (same missions, classic guidance vs the learned pilot) on Site42.

Run from the repo root (needs torch; ~3-5 min on a busy 8-core CPU for n = 48):
    PYTHONPATH=. ~/venvs/adi/bin/python website/tools/export_fig3d_pilot.py [n_pairs, default 48]
(optional FIG3D_CACHE=<dir>: keep each verified flight there, so an interrupted run resumes)

What it flies: n paired missions on Site42 drawn exactly as the lander benchmark draws them (worldmodel/lander/benchmark.py:
fly(scene, params, gp, model, n, seed) with its seed-0 value 50_000 = 50_000 + 97*0 + 7*ALL.index("Site42"), nominal and harsh x3
failure rates, slope-rule belief map, pilot = data/models/lander_pilot.pt). Because missions depend on n, n = 48 gives 48 NEW
missions from the same generator, not the benchmark's first 48 (the benchmark flew n = 256 x 3 seeds = 768 per condition). The flight
loop is worldmodel/lander/fly.py's, unchanged, with the lander position recorded every 1 s simulated. Outcomes are checked against
fly() itself (same seed, same n, same code) before anything is written. An illustrative subset, not the benchmark.

Writes website/assets/fig3d/pilot/:
  site42_height.bin   256 x 256 uint16 heights (models3d encoding: row 0 = north, h = min + v/65535*(max-min))
  site42_tex.jpg      grey hillshade of the observed DEM, slope-rule hazard (>= 12 deg) tinted
  tracks.bin          uint16 x, y, z, t per point (decode: offset + q*scale), tracks back to back
  pilot3d.json        meta, per-track columns, per-condition counts, provenance
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import model_path
from contracts.runs import FAILURE_REASONS
from worldmodel.lander.env import LanderBatch, make_params
from worldmodel.lander.fly import fly
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.pilot import authority, load_checkpoint, pilot_features, residual_action
from worldmodel.lander.scenes import load_scenes

SITE, SEED, HARSH = "Site42", 50_000, 3.0              # benchmark seed for s = 0 on Site42: 50_000 + 97*0 + 7*ALL.index("Site42") (= 0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
MAXPTS = 48
OUTCOMES = list(dict.fromkeys(["ok", *FAILURE_REASONS, "timeout"]))
OUT = Path("website/assets/fig3d/pilot")
torch.set_num_threads(6)


@torch.no_grad()
def fly_recorded(scene, params, gp, model, n, seed, repeat=5, crop=32, device=torch.device("cpu")):
    """worldmodel.lander.fly.fly, unchanged, plus the position of every lander every 10 steps (1 s at dt 0.1 s)."""
    rng = np.random.default_rng(seed)
    env = LanderBatch(params, scene.terrain, n, device, seed=seed, targets=scene.sample_targets(n, rng, device))
    obs = env.observe()
    r = torch.zeros(n, 3, device=device)
    path = [env.pos.clone()]
    for i in range(params.n_steps):
        classic = classic_action(gp, obs)
        if model is not None:
            if i % repeat == 0:
                mu, _ = model(pilot_features(obs, classic), env.crop(crop))
                r = mu
            act = residual_action(classic, r, authority(obs, getattr(model, "rule", None)))
        else:
            act = classic
        obs, _, _ = env.step(act)
        if i % 10 == 9:
            path.append(env.pos.clone())
        if bool(env.done.all()):
            path.append(env.pos.clone())
            break
    return env.metrics(), torch.stack(path).cpu().numpy(), env.target.cpu().numpy()


def hillshade(h, mpp):
    gy, gx = np.gradient(h, mpp)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    az, alt = np.radians(315), np.radians(35)
    nx, ny, nz = -gx, gy, np.ones_like(h)          # row 0 = north: +y is up the rows
    nrm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    sh = (nx * np.sin(az) * np.cos(alt) + ny * np.cos(az) * np.cos(alt) + nz * np.sin(alt)) / nrm
    return np.clip(sh, 0, 1), slope


T0 = time.time()


def main() -> int:
    cfg = load_mission()
    dev = torch.device("cpu")
    ck = model_path("lander_pilot")
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    model, info = load_checkpoint(ck, dev)
    sc = load_scenes([SITE], cfg, dev, "backup", 1)[0]
    T = sc.terrain
    tracks, counts = [], {}
    for cond, mult in (("nominal", 1.0), ("harsh", HARSH)):
        params = make_params(cfg)
        params.p_engine_out = min(1.0, params.p_engine_out * mult)
        params.p_nav_drift = min(1.0, params.p_nav_drift * mult)
        params.p_camera_blind = min(1.0, params.p_camera_blind * mult)
        gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
        res = {}
        for pol, mdl in (("classic", None), ("pilot", model)):
            cache = Path(os.environ["FIG3D_CACHE"]) / f"pilot3d_{cond}_{pol}_{N}_{SEED}_{sha[:10]}.npz" if os.environ.get("FIG3D_CACHE") else None
            if cache and cache.exists():                                               # resume after an interruption (verified flights only)
                z = np.load(cache)
                m = {k: torch.as_tensor(z[k]) for k in ("success", "outcome", "time_s", "engine_failed", "nav_drift", "camera_blind")}
                P, tgt = z["P"], z["tgt"]
            else:
                m, P, tgt = fly_recorded(sc, params, gp, mdl, N, SEED)
                ref = fly(sc, params, gp, mdl, N, SEED, 5, device=dev)                # the benchmark's own call
                assert torch.equal(m["success"].cpu(), ref["success"].cpu()), f"{cond}/{pol}: recorded flight differs from fly()"
                if cache:
                    np.savez(cache, P=P, tgt=tgt, **{k: m[k].cpu().numpy() for k in ("success", "outcome", "time_s", "engine_failed", "nav_drift", "camera_blind")})
            res[pol] = (m, P, tgt)
            print(f"{cond:7s} {pol:7s} landed {int(m['success'].sum())}/{N}  {time.time() - T0:.0f}s", flush=True)
        okc = res["classic"][0]["success"].cpu().numpy().astype(bool)
        okp = res["pilot"][0]["success"].cpu().numpy().astype(bool)
        counts[cond] = {"n_pairs": N, "classic_landed": int(okc.sum()), "pilot_landed": int(okp.sum()),
                        "only_pilot": int((okp & ~okc).sum()), "only_classic": int((~okp & okc).sum()), "failure_multiplier": mult}
        for pol in ("classic", "pilot"):
            m, P, tgt = res[pol]
            ok = okc if pol == "classic" else okp
            for k in range(N):
                Tn = int(np.ceil(float(m["time_s"][k]))) + 1
                xyz = P[: min(Tn, P.shape[0]), k].astype(np.float64)
                t = np.arange(len(xyz), dtype=np.float64)
                if len(xyz) > MAXPTS:                         # keep the end exactly; dense near the ground (sqrt spacing in time)
                    s = np.unique(np.round((1 - (1 - np.linspace(0, 1, MAXPTS)) ** 1.6) * (len(xyz) - 1)).astype(int))
                    xyz, t = xyz[s], t[s]
                oc = int(m["outcome"][k])
                tracks.append({"cond": cond, "policy": pol, "pair": k, "ok": bool(ok[k]),
                               "outcome": FAILURE_REASONS[oc] if oc >= 0 else "timeout",
                               "differs": bool(okc[k] != okp[k]),
                               "engine_failed": bool(m["engine_failed"][k]), "nav_drift": bool(m["nav_drift"][k]), "camera_blind": bool(m["camera_blind"][k]),
                               "target": [float(tgt[k, 0]), float(tgt[k, 1])], "xyz": xyz, "t": t})

    # ---- terrain: the whole tile resampled to 256 x 256
    H = T.height.cpu().numpy().astype(np.float64)
    hh, ww = H.shape
    n = 256
    ri = np.linspace(0, hh - 1, n); ci = np.linspace(0, ww - 1, n)
    r0 = np.floor(ri).astype(int).clip(0, hh - 2); c0 = np.floor(ci).astype(int).clip(0, ww - 2)
    fr = (ri - r0)[:, None]; fc = (ci - c0)[None, :]
    Hs = (H[np.ix_(r0, c0)] * (1 - fr) * (1 - fc) + H[np.ix_(r0 + 1, c0)] * fr * (1 - fc)
          + H[np.ix_(r0, c0 + 1)] * (1 - fr) * fc + H[np.ix_(r0 + 1, c0 + 1)] * fr * fc)
    size_m = (ww - 1) * T.mpp
    mpp_s = size_m / (n - 1)
    lo, hi = float(Hs.min()), float(Hs.max())
    if os.statvfs(str(Path.home())).f_bavail * os.statvfs(str(Path.home())).f_frsize < 1.5e9:
        print("disk below 1.5 GB: nothing written"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    q = np.round((Hs - lo) / (hi - lo) * 65535).astype("<u2")
    (OUT / "site42_height.bin").write_bytes(q.tobytes())

    # texture: hillshade from the full-resolution DEM (1024 px), slope-rule hazard (slope >= limit) tinted
    from PIL import Image
    shade, slope = hillshade(H, T.mpp)
    lim = float(cfg.lander.touchdown_limits.max_slope_deg)
    g = 0.18 + 0.72 * shade
    rgb = np.stack([g * 0.95, g * 0.93, g * 0.88], -1)
    unsafe = slope >= lim
    rgb[unsafe] = rgb[unsafe] * np.array([1.0, 0.62, 0.55]) + np.array([0.10, 0.0, 0.0])
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
    img.save(OUT / "site42_tex.jpg", quality=82)

    # ---- tracks: uint16 quantised x, y, z, t
    allp = np.concatenate([np.column_stack([tr["xyz"], tr["t"]]) for tr in tracks])
    off = allp.min(0); scale = np.maximum((allp.max(0) - off) / 65535, 1e-9)
    buf, cols, first = [], [], 0
    for tr in tracks:
        a = np.column_stack([tr["xyz"], tr["t"]])
        buf.append(np.round((a - off) / scale).astype("<u2"))
        cols.append([["nominal", "harsh"].index(tr["cond"]), ["classic", "pilot"].index(tr["policy"]), tr["pair"], int(tr["ok"]),
                     int(tr["differs"]), first, len(a), int(tr["engine_failed"]) | int(tr["nav_drift"]) << 1 | int(tr["camera_blind"]) << 2,
                     OUTCOMES.index(tr["outcome"]) if tr["outcome"] in OUTCOMES else -1])
        first += len(a)
    (OUT / "tracks.bin").write_bytes(np.concatenate(buf).tobytes())
    outcome_names = OUTCOMES
    targets = {}
    for tr in tracks:
        targets.setdefault(tr["pair"], [round(tr["target"][0], 1), round(tr["target"][1], 1)])
    meta = {
        "meta": {"provenance": "modelled",
                 "source": f"worldmodel/lander/fly.py loop (positions recorded every 1 s) on {sc.terrain.region}; missions drawn by the benchmark generator (seed {SEED}, n {N})",
                 "created_by": "website/tools/export_fig3d_pilot.py",
                 "command": "PYTHONPATH=. ~/venvs/adi/bin/python website/tools/export_fig3d_pilot.py",
                 "frame": "tile ENU metres, origin = tile centre, x east, y north, z = LOLA height",
                 "units": "m (x, y, z), s (t)",
                 "terrain_source": f"data/tiles/{sc.terrain.region}.npz (NASA LRO LOLA PGDA 5 m/px, observed)",
                 "note": f"illustrative subset: {N} paired missions per condition on Site42 from the benchmark's mission generator (seed {SEED}); not the benchmark's own missions (768 per condition)"},
        "site": SITE, "region": sc.terrain.region, "seed": SEED, "n_pairs": N, "harsh_multiplier": HARSH,
        "checkpoint_id": info.get("checkpoint_id"), "checkpoint_sha256": sha, "checkpoint": "data/models/lander_pilot.pt",
        "counts": counts, "slope_limit_deg": lim,
        "terrain": {"n": n, "m_per_px": mpp_s, "size_m": size_m, "min_m": lo, "max_m": hi, "file": "site42_height.bin", "texture": "site42_tex.jpg",
                    "encoding": "uint16 little-endian, row 0 = north, h = min + v/65535*(max-min)",
                    "texture_note": "grey hillshade of the observed DEM; slope >= limit tinted red (slope rule)"},
        "tracks": {"file": "tracks.bin", "dtype": "uint16 little-endian", "fields": ["x", "y", "z", "t"], "offset": off.tolist(), "scale": scale.tolist(),
                   "columns": ["cond_idx", "policy_idx", "pair", "ok", "differs", "first_point", "n_points", "flags", "outcome_idx"],
                   "conds": ["nominal", "harsh"], "policies": ["classic", "pilot"], "outcomes": outcome_names,
                   "flags": {"1": "engine_failed", "2": "nav_drift", "4": "camera_blind"}, "rows": cols},
        "targets_xy_m": [targets[k] for k in range(N)],
    }
    (OUT / "pilot3d.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    for c, v in counts.items():
        print(c, v)
    print("wrote", sorted((p.name, p.stat().st_size) for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
