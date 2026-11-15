"""Lander pilot benchmark: every terrain map available, random landing targets, the SAME missions flown by classic ZEM/ZEV guidance and by the
learned pilot, under nominal and harsh failure rates. Paired statistics, so luck in the missions cancels out.

    python -m worldmodel.lander.benchmark --n 256 --seeds 2           # writes runs/lander/benchmark.json and benchmark.md

Per map and condition: success rate with a 95 % Wilson interval for each pilot; paired counts (missions only the pilot landed / only classic
landed) with an exact two-sided sign test p-value; success when the engine failed / nav drifted / camera was blind; mean miss distance, fuel
and touchdown speed of successful landings. Map roles: train (the pilot practised there), val (used to pick the checkpoint), TEST (never seen).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from worldmodel.hazard.lola import VAL_SITES
from worldmodel.lander.env import make_params
from worldmodel.lander.fly import fly
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.pilot import load_checkpoint, pick_device
from worldmodel.lander.scenes import load_scenes

ALL = ["Site42", "Haworth", "Shoemaker", "DM2"]          # default: the maps this pilot never saw (Site23 only with --final)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(b: int, c: int) -> float:
    """Exact two-sided binomial test on discordant pairs (b: only pilot landed, c: only classic landed)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


TRAINED_ON: set[str] = set()


def role(site: str) -> str:
    """Role FOR THIS PILOT (from its checkpoint's training record), not the planner's site split."""
    if site == "Site23":
        return "FINAL EXAM"
    if f"lola5m_{site.lower()}" in TRAINED_ON:
        return "seen (trained on)"
    if site in VAL_SITES:
        return "validation (checkpoint selection / gate)"
    return "UNSEEN"


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Pilot minus classic success, paired (Wald interval for a paired difference of proportions)."""
    d = (b - c) / n
    var = max((b + c) / n**2 - (b - c) ** 2 / n**3, 0.0)
    return d, d - z * math.sqrt(var), d + z * math.sqrt(var)


def _stats(m: dict) -> dict:
    ok = m["success"].cpu().numpy().astype(bool)
    out = {"n": int(ok.size), "success": float(ok.mean()), "ci95": [round(v, 4) for v in wilson(int(ok.sum()), ok.size)]}
    for key in ("engine_failed", "nav_drift", "camera_blind"):
        f = m[key].cpu().numpy().astype(bool)
        out[f"success_{key}"] = float(ok[f].mean()) if f.any() else None
        out[f"n_{key}"] = int(f.sum())
    if ok.any():
        out["miss_m"] = float(m["target_dist_m"].cpu().numpy()[ok].mean())
        out["fuel_kg"] = float(m["fuel_used_kg"].cpu().numpy()[ok].mean())
        out["td_vz_m_s"] = float(m["td_vz"].cpu().numpy()[ok].mean())
    return out, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="lander_pilot")
    ap.add_argument("--sites", nargs="+", default=ALL)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--harsh", type=float, default=3.0, help="failure-rate multiplier for the harsh condition (0 = skip)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--final", action="store_true", help="the ONE final Site23 exam (only Site23 is flown); otherwise Site23 is refused")
    ap.add_argument("--out", default="benchmark")
    a = ap.parse_args(argv)
    if a.final:
        a.sites = ["Site23"]
    elif "Site23" in a.sites:
        raise SystemExit("Site23 is the final exam: run it once with --final after training is frozen")
    cfg = load_mission()
    device = pick_device(a.device)
    model, info = load_checkpoint(model_path(a.name), device)
    TRAINED_ON.update(info.get("trained_on", []))
    t0 = time.time()
    conditions = [("nominal", 1.0)] + ([("harsh", a.harsh)] if a.harsh else [])
    rows, totals = [], {c: {"classic": [], "pilot": []} for c, _ in conditions}
    for site in a.sites:
        scenes = load_scenes([site], cfg, device, "backup", 1)
        if not scenes:
            print(f"{site}: no usable map (no safe landing cells or tile missing); skipped", flush=True)
            continue
        sc = scenes[0]
        for cond, mult in conditions:
            params = make_params(cfg)
            params.p_engine_out = min(1.0, params.p_engine_out * mult)
            params.p_nav_drift = min(1.0, params.p_nav_drift * mult)
            params.p_camera_blind = min(1.0, params.p_camera_blind * mult)
            gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
            oks = {"classic": [], "pilot": []}
            res = {}
            for pol, mdl in (("classic", None), ("pilot", model)):
                ms = [fly(sc, params, gp, mdl, a.n, 50_000 + 97 * s + 7 * (ALL.index(site) if site in ALL else len(ALL)), 5, device=device) for s in range(a.seeds)]
                merged = {k: torch.cat([m[k] for m in ms]) for k in ms[0]}
                st, ok = _stats(merged)
                res[pol] = st
                oks[pol] = ok
                totals[cond][pol].append(ok)
            b = int((oks["pilot"] & ~oks["classic"]).sum())
            c = int((~oks["pilot"] & oks["classic"]).sum())
            d, lo, hi = paired_diff_ci(b, c, a.n * a.seeds)
            row = {"site": site, "role": role(site), "condition": cond, "failure_multiplier": mult, **{f"{p}": res[p] for p in res},
                   "only_pilot_landed": b, "only_classic_landed": c, "sign_test_p": sign_test(b, c), "diff": d, "diff_ci95": [lo, hi]}
            rows.append(row)
            print(f"{site:7s} {role(site):5s} {cond:7s} classic {res['classic']['success']:.3f}  pilot {res['pilot']['success']:.3f}  "
                  f"(pilot-only {b}, classic-only {c}, p={row['sign_test_p']:.3f})  {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for cond, d in totals.items():
        if d["classic"]:
            okc, okp = np.concatenate(d["classic"]), np.concatenate(d["pilot"])
            b, c = int((okp & ~okc).sum()), int((~okp & okc).sum())
            dd, lo, hi = paired_diff_ci(b, c, okc.size)
            summary[cond] = {"diff": dd, "diff_ci95": [lo, hi], "maps_roles": sorted({role(r["site"]) for r in rows}),"maps": len(d["classic"]), "missions": int(okc.size), "classic": float(okc.mean()), "pilot": float(okp.mean()),
                             "classic_ci95": wilson(int(okc.sum()), okc.size), "pilot_ci95": wilson(int(okp.sum()), okp.size),
                             "only_pilot_landed": b, "only_classic_landed": c, "sign_test_p": sign_test(b, c),
                             "failure_reduction": (1 - (1 - okp.mean()) / max(1 - okc.mean(), 1e-9))}
    out = {"pilot": a.name, "checkpoint_id": info.get("checkpoint_id"), "parameters": info.get("parameters"), "trained_on": info.get("trained_on"),
           "missions_per_map_per_condition": a.n * a.seeds, "rows": rows, "summary": summary,
           "notes": "same missions for both pilots (paired); harsh = engine-out, nav-drift and blind-camera rates x multiplier; Site42 tiles not available"}
    p = run_dir("lander") / f"{a.out}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
    lines = ["| map | role for this pilot | condition | missions | classic (95% CI) | pilot (95% CI) | pilot - classic (95% CI) | pilot-only / classic-only | McNemar exact p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['site']} | {r['role']} | {r['condition']} | {a.n * a.seeds} | {r['classic']['success']:.4f} {ci(r['classic']['ci95'])} | "
                     f"{r['pilot']['success']:.4f} {ci(r['pilot']['ci95'])} | {r['diff']:+.4f} {ci(r['diff_ci95'])} | "
                     f"{r['only_pilot_landed']} / {r['only_classic_landed']} | {r['sign_test_p']:.3f} |")
    for cond, s in summary.items():
        lines.append(f"| **POOLED ({s['maps']} maps)** | | {cond} | {s['missions']} | {s['classic']:.4f} {ci(s['classic_ci95'])} | {s['pilot']:.4f} "
                     f"{ci(s['pilot_ci95'])} | {s['diff']:+.4f} {ci(s['diff_ci95'])} | {s['only_pilot_landed']} / {s['only_classic_landed']} | {s['sign_test_p']:.4f} |")
    (p.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
