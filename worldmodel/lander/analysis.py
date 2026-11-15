"""Paired failure analysis and failure mining for the lander pilot (classic ZEM/ZEV vs learned residual pilot, SAME missions).

    python -m worldmodel.lander.analysis report --sites Site11 --n 512 --seeds 2       # runs/lander/failure_analysis.md (+ .json)
    python -m worldmodel.lander.analysis mine --sites Site01 Site04 Site06 Site07 Site20 --n 256 --seeds 2 --out pool.npz

For every mission the pilot crashed (and, separately, every mission classic landed but the pilot crashed) the report logs: failure type,
injected failures, slope and roughness (std of height in a 15 x 15 m window) at the touchdown point, altitude above ground when the engine
failed, fuel left, touchdown vertical / horizontal speed, thrust tilt at touchdown, and the first time the pilot's trajectory diverged
from classic guidance by more than 1 m. `mine` saves the initial conditions of failed and near-miss missions as a scenario pool for
targeted Dream-RSI (training maps only: never validation or test maps).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import data_root, model_path, run_dir
from contracts.runs import FAILURE_REASONS
from worldmodel.hazard.lola import HOLDOUT_SITES
from worldmodel.lander.env import LanderBatch, make_params
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.pilot import load_checkpoint, pick_device, authority, pilot_features, residual_action
from worldmodel.lander.scenes import load_scenes

CONDITIONS = {"normal": 1.0, "harsh": 3.0}


def condition_params(cfg, mult: float):
    p = make_params(cfg)
    p.p_engine_out = min(1.0, p.p_engine_out * mult)
    p.p_nav_drift = min(1.0, p.p_nav_drift * mult)
    p.p_camera_blind = min(1.0, p.p_camera_blind * mult)
    return p


@torch.no_grad()
def fly_recorded(scene, params, gp, model, n: int, seed: int, repeat: int = 5, device=None, scenarios: dict | None = None,
                 record_every: int = 10) -> dict:
    """Fly n landers to the end; return final metrics, initial scenarios, the sampled path and per-lander event data."""
    device = device or scene.terrain.device
    rng = np.random.default_rng(seed)
    env = LanderBatch(params, scene.terrain, n, device, seed=seed, targets=scene.sample_targets(n, rng, device))
    if scenarios is not None:
        env.load_scenarios(torch.arange(n, device=device), scenarios)
    init = env.scenarios()
    obs = env.observe()
    r = torch.zeros(n, 3, device=device)
    path = [env.pos.clone()]
    alt_at_eo = torch.full((n,), float("nan"), device=device)
    for i in range(params.n_steps):
        classic = classic_action(gp, obs)
        if model is not None:
            if i % repeat == 0:
                r = model(pilot_features(obs, classic), env.crop(32))[0].float()
            act = residual_action(classic, r, authority(obs, getattr(model, "rule", None)))
        else:
            act = classic
        crossing = (env.t < env.engine_out_t) & (env.t + params.dt >= env.engine_out_t) & ~env.done
        if bool(crossing.any()):
            ground = env.terrain.height_at(env.pos[:, 0], env.pos[:, 1])
            alt_at_eo = torch.where(crossing, env.pos[:, 2] - ground, alt_at_eo)
        obs, _, _ = env.step(act)
        if i % record_every == record_every - 1:
            path.append(env.pos.clone())
        if bool(env.done.all()):
            break
    path.append(env.pos.clone())
    m = env.metrics()
    th = env.thrust
    tilt = torch.rad2deg(torch.atan2(th[:, :2].norm(dim=1), th[:, 2].clamp(min=1e-6)))
    x, y = env.pos[:, 0], env.pos[:, 1]
    rough = torch.stack([env.terrain.height_at(x + dx, y + dy) for dx in (-7.5, 0.0, 7.5) for dy in (-7.5, 0.0, 7.5)]).std(0)
    return {"m": {k: v.cpu().numpy() for k, v in m.items()}, "init": init, "path": torch.stack(path).cpu().numpy(),
            "alt_at_engine_out": alt_at_eo.cpu().numpy(), "fuel_left": (env.mass - params.dry_mass).cpu().numpy(),
            "tilt_deg": tilt.cpu().numpy(), "slope_td": env.terrain.slope_at(x, y).cpu().numpy(), "rough_td_m": rough.cpu().numpy(),
            "record_dt_s": params.dt * record_every}


def first_divergence_s(a: np.ndarray, b: np.ndarray, dt: float, tol: float = 1.0) -> np.ndarray:
    T = min(len(a), len(b))
    d = np.linalg.norm(a[:T] - b[:T], axis=-1)                                         # [T, n]
    over = d > tol
    first = np.where(over.any(0), over.argmax(0), -1)
    return np.where(first >= 0, first * dt, np.nan)


def paired_runs(sites, n: int, seeds: int, model, device, conditions=CONDITIONS, seed_base: int = 70_000) -> list[dict]:
    cfg = load_mission()
    out = []
    for site in sites:
        scenes = load_scenes([site], cfg, device, "backup", 1)
        if not scenes:
            print(f"{site}: no usable map; skipped", flush=True)
            continue
        sc = scenes[0]
        for cond, mult in conditions.items():
            params = condition_params(cfg, mult)
            gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
            for s in range(seeds):
                seed = seed_base + 101 * s + 13 * sum(map(ord, site)) + (0 if cond == "normal" else 5)
                rc = fly_recorded(sc, params, gp, None, n, seed, device=device)
                rp = fly_recorded(sc, params, gp, model, n, seed, device=device, scenarios=rc["init"])
                out.append({"site": site, "condition": cond, "seed": seed, "classic": rc, "pilot": rp, "region": sc.region})
                print(f"  {site} {cond} seed {seed}: classic {rc['m']['success'].mean():.3f} pilot {rp['m']['success'].mean():.3f}", flush=True)
    return out


def failure_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for R in runs:
        c, p = R["classic"], R["pilot"]
        okc, okp = c["m"]["success"].astype(bool), p["m"]["success"].astype(bool)
        div = first_divergence_s(c["path"], p["path"], p["record_dt_s"])
        for i in np.nonzero(~okp)[0]:
            oc = int(p["m"]["outcome"][i])
            rows.append({"site": R["site"], "condition": R["condition"], "seed": R["seed"], "mission": int(i),
                         "classic_landed": bool(okc[i]), "pilot_failure": FAILURE_REASONS[oc] if oc >= 0 else "still_flying",
                         "classic_outcome": FAILURE_REASONS[int(c["m"]["outcome"][i])] if c["m"]["outcome"][i] >= 0 else "still_flying",
                         "engine_out": bool(p["m"]["engine_failed"][i]), "nav_drift": bool(p["m"]["nav_drift"][i]), "camera_blind": bool(p["m"]["camera_blind"][i]),
                         "slope_td_deg": round(float(p["slope_td"][i]), 2), "roughness_td_m": round(float(p["rough_td_m"][i]), 2),
                         "alt_at_engine_out_m": None if np.isnan(p["alt_at_engine_out"][i]) else round(float(p["alt_at_engine_out"][i]), 1),
                         "fuel_left_kg": round(float(p["fuel_left"][i]), 1), "td_vz_m_s": round(float(p["m"]["td_vz"][i]), 2),
                         "td_vh_m_s": round(float(p["m"]["td_vh"][i]), 2), "tilt_td_deg": round(float(p["tilt_deg"][i]), 1),
                         "miss_m": round(float(p["m"]["target_dist_m"][i]), 1),
                         "first_divergence_s": None if np.isnan(div[i]) else round(float(div[i]), 1)})
    return rows


def _cause(r: dict) -> str:
    inj = "+".join(k for k in ("engine_out", "nav_drift", "camera_blind") if r[k]) or "no injected failure"
    return f"{r['pilot_failure']} with {inj}"


def report(sites, n, seeds, name, device) -> str:
    model = load_checkpoint(model_path(name), device)[0]
    runs = paired_runs(sites, n, seeds, model, device)
    rows = failure_rows(runs)
    lost = [r for r in rows if r["classic_landed"]]
    lines = [f"# Lander pilot failure analysis ({', '.join(sites)}; validation maps only, never Site23)", "",
             f"Paired: the same {n * seeds} missions per map and condition flown by classic ZEM/ZEV and by `{name}`. Conditions: "
             + ", ".join(f"{k} (failure rates x{v})" for k, v in CONDITIONS.items()) + ".", ""]
    agg = {}
    for R in runs:
        k = (R["site"], R["condition"])
        a = agg.setdefault(k, {"n": 0, "c": 0, "p": 0, "b": 0, "cc": 0})
        okc, okp = R["classic"]["m"]["success"].astype(bool), R["pilot"]["m"]["success"].astype(bool)
        a["n"] += okc.size; a["c"] += int(okc.sum()); a["p"] += int(okp.sum())
        a["b"] += int((okp & ~okc).sum()); a["cc"] += int((~okp & okc).sum())
    lines += ["| map | condition | missions | classic | pilot | pilot-only landed | classic-only landed |", "|---|---|---|---|---|---|---|"]
    for (site, cond), a in agg.items():
        lines.append(f"| {site} | {cond} | {a['n']} | {a['c'] / a['n']:.3f} | {a['p'] / a['n']:.3f} | {a['b']} | {a['cc']} |")
    lines += ["", f"## Every mission the pilot crashed ({len(rows)}; {len(lost)} of them classic landed)", "",
              "| map | cond | mission | pilot failure | classic | injected | slope deg | rough m | alt at engine-out m | fuel kg | td vz | td vh | tilt deg | first divergence s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        inj = ",".join(k for k in ("engine_out", "nav_drift", "camera_blind") if r[k]) or "-"
        lines.append(f"| {r['site']} | {r['condition']} | {r['mission']} | {r['pilot_failure']} | {r['classic_outcome']} | {inj} | {r['slope_td_deg']} | "
                     f"{r['roughness_td_m']} | {r['alt_at_engine_out_m'] if r['alt_at_engine_out_m'] is not None else '-'} | {r['fuel_left_kg']} | "
                     f"{r['td_vz_m_s']} | {r['td_vh_m_s']} | {r['tilt_td_deg']} | {r['first_divergence_s'] if r['first_divergence_s'] is not None else '-'} |")
    top = Counter(_cause(r) for r in rows).most_common(3)
    top_lost = Counter(_cause(r) for r in lost).most_common(3)
    lines += ["", "## Top 3 failure causes (all pilot crashes)", ""] + [f"{i + 1}. **{c}**: {k} missions" for i, (c, k) in enumerate(top)]
    lines += ["", "## Top causes where classic landed but the pilot crashed (regressions)", ""] + \
             ([f"{i + 1}. **{c}**: {k} missions" for i, (c, k) in enumerate(top_lost)] or ["none"])
    p = run_dir("lander") / "failure_analysis.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps({"summary": {f"{k[0]}|{k[1]}": v for k, v in agg.items()}, "rows": rows}, indent=1), encoding="utf-8")
    print("\n".join(lines[-12:]), flush=True)
    return str(p)


def mine(sites, n, seeds, name, device, out: str) -> str:
    """Initial conditions of failed and near-miss missions on TRAINING maps -> scenario pool for targeted Dream-RSI."""
    assert not set(sites) & set(HOLDOUT_SITES), "never mine test maps"
    model = load_checkpoint(model_path(name), device)[0]
    cfg = load_mission()
    lim = cfg.lander.touchdown_limits
    runs = paired_runs(sites, n, seeds, model, device, seed_base=80_000)
    pool = {k: [] for k in ("pos", "vel", "target", "engine_out_t", "nav_bias", "camera_ok")}
    regions, why = [], []
    for R in runs:
        p, c = R["pilot"], R["classic"]
        okp, okc = p["m"]["success"].astype(bool), c["m"]["success"].astype(bool)
        near = okp & ((p["m"]["td_vz"] > 0.8 * lim.max_vertical_speed_m_s) | (p["m"]["td_vh"] > 0.8 * lim.max_horizontal_speed_m_s)
                      | (p["slope_td"] > 0.8 * lim.max_slope_deg))
        pick = (~okp) | (~okc) | near
        for k in pool:
            pool[k].append(p["init"][k][pick])
        regions += [R["region"]] * int(pick.sum())
        why += ["pilot_fail" if not a else ("classic_fail" if not b else "near_miss") for a, b in zip(okp[pick], okc[pick], strict=True)]
    arr = {k: np.concatenate(v) for k, v in pool.items()}
    path = data_root() / "pilot_dream" / out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, region=np.array(regions), why=np.array(why), **arr)
    print(f"scenario pool: {len(regions)} missions ({Counter(why)}) -> {path}", flush=True)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["report", "mine"])
    ap.add_argument("--sites", nargs="+", default=["Site11"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--name", default="lander_pilot")
    ap.add_argument("--out", default="failure_pool.npz")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    assert "Site23" not in a.sites, "Site23 is the final exam: never analysed or mined here"
    dev = pick_device(a.device)
    if a.action == "report":
        report(a.sites, a.n, a.seeds, a.name, dev)
    else:
        mine(a.sites, a.n, a.seeds, a.name, dev, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
