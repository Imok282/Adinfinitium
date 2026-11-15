"""Targeted, gated Dream-RSI for the lander pilot (plan of 15 Nov):

  train   : the 5 training maps only (Site01 04 06 07 20), with mined failure / near-miss scenarios oversampled and randomised
            (worldmodel.lander.analysis mine), warm start from the CURRENT production pilot, residual network only, small learning step.
  gate    : validation map Site11 (separate from training), FIXED missions (seeds), normal and harsh failure rates, paired.
            ACCEPT only if (a) the candidate loses NO mission that the current pilot or classic guidance lands, and (b) its total success
            is strictly higher than the current pilot's. Otherwise reject, record why, and try the next (gentler) adjustment. Max 5 rounds.
  never   : Site23 (final exam) and the unseen benchmark maps are not touched here.

    python -m worldmodel.lander.rsi_targeted --pool data/pilot_dream/failure_pool.npz --rounds 5 --minutes 25

Writes runs/lander/rsi_changelog.md (+ .jsonl). The production checkpoint lander_pilot.pt changes only on ACCEPT (old one kept as
lander_pilot_before_rsi_r<k>.pt). The residual structure is unchanged: an untrained residual still equals classic guidance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import data_root, model_path, run_dir
from worldmodel.lander.analysis import CONDITIONS, condition_params, fly_recorded
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.pilot import load_checkpoint, pick_device
from worldmodel.lander.scenes import load_scenes

TRAIN_MAPS = ["Site01", "Site04", "Site06", "Site07", "Site20"]
GATE_MAP = "Site11"
PROD, CAND = "lander_pilot", "lander_pilot_rsi_cand"
SCHEDULE = [   # each round a smaller / more targeted step than the last
    {"lr": 5e-5, "clip": 0.10, "pool_frac": 0.5, "td_penalty": 0.0},
    {"lr": 3e-5, "clip": 0.10, "pool_frac": 0.5, "td_penalty": 0.3},
    {"lr": 2e-5, "clip": 0.08, "pool_frac": 0.7, "td_penalty": 0.5},
    {"lr": 1e-5, "clip": 0.05, "pool_frac": 0.8, "td_penalty": 0.5},
    {"lr": 1e-5, "clip": 0.05, "pool_frac": 0.6, "td_penalty": 1.0},
]


def gate_missions(n: int, seeds: int, device) -> list[dict]:
    """Fixed validation missions on Site11 (normal + harsh). Classic is flown once and cached with the exact initial conditions."""
    cache = data_root() / "pilot_dream" / f"gate_{GATE_MAP}_n{n}_s{seeds}.npz"
    cfg = load_mission()
    scene = load_scenes([GATE_MAP], cfg, device, "backup", 1)[0]
    out = []
    z = np.load(cache, allow_pickle=False) if cache.exists() else None
    store = {}
    for cond, mult in CONDITIONS.items():
        params = condition_params(cfg, mult)
        gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
        for s in range(seeds):
            seed = 90_000 + 101 * s + (0 if cond == "normal" else 5)
            key = f"{cond}_{seed}"
            if z is not None and f"{key}_ok" in z:
                init = {k: z[f"{key}_{k}"] for k in ("pos", "vel", "target", "engine_out_t", "nav_bias", "camera_ok")}
                ok = z[f"{key}_ok"]
            else:
                rc = fly_recorded(scene, params, gp, None, n, seed, device=device)
                init, ok = rc["init"], rc["m"]["success"].astype(bool)
            store.update({f"{key}_ok": ok, **{f"{key}_{k}": v for k, v in init.items()}})
            out.append({"cond": cond, "seed": seed, "params": params, "gp": gp, "scene": scene, "init": init, "classic_ok": ok})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **store)
    return out


def fly_set(missions: list[dict], name: str, device) -> list[np.ndarray]:
    model = load_checkpoint(model_path(name), device)[0]
    res = []
    for M in missions:
        r = fly_recorded(M["scene"], M["params"], M["gp"], model, len(M["classic_ok"]), M["seed"], device=device, scenarios=M["init"])
        res.append(r["m"]["success"].astype(bool))
    return res


def gate(missions, cur_ok, cand_ok) -> dict:
    reg = {"vs_classic": 0, "vs_current": 0}
    tot = {"classic": 0, "current": 0, "candidate": 0, "n": 0}
    per = []
    for M, cu, ca in zip(missions, cur_ok, cand_ok, strict=True):
        cl = M["classic_ok"]
        reg["vs_classic"] += int((cl & ~ca).sum())
        reg["vs_current"] += int((cu & ~ca).sum())
        lost_any = int(((cl | cu) & ~ca).sum())
        tot["classic"] += int(cl.sum()); tot["current"] += int(cu.sum()); tot["candidate"] += int(ca.sum()); tot["n"] += cl.size
        per.append({"cond": M["cond"], "seed": M["seed"], "classic": float(cl.mean()), "current": float(cu.mean()), "candidate": float(ca.mean()),
                    "lost_missions": lost_any, "new_wins": int((ca & ~(cl | cu)).sum())})
    regressions = sum(p["lost_missions"] for p in per)
    improved = tot["candidate"] > tot["current"]
    accept = regressions == 0 and improved
    why = "accepted: no regressions and higher success" if accept else \
        ("; ".join(x for x in ((f"{regressions} regressions (missions landed by current or classic, lost by candidate)" if regressions else ""),
                               ("no improvement in total success" if not improved else "")) if x))
    return {"accept": accept, "reason": why, "regressions": regressions, "regressions_detail": reg, "totals": tot, "per_condition": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--gate-n", type=int, default=512)
    ap.add_argument("--gate-seeds", type=int, default=2)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--mb-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from worldmodel.lander import train as tr

    device = pick_device(a.device)
    t0 = time.time()
    missions = gate_missions(a.gate_n, a.gate_seeds, device)
    cur_ok = fly_set(missions, PROD, device)
    log_md = run_dir("lander") / "rsi_changelog.md"
    log_jl = run_dir("lander") / "rsi_changelog.jsonl"
    log_md.parent.mkdir(parents=True, exist_ok=True)
    tot_c = sum(int(m["classic_ok"].sum()) for m in missions)
    tot_p = sum(int(x.sum()) for x in cur_ok)
    n_all = sum(m["classic_ok"].size for m in missions)
    lines = ["# Targeted Dream-RSI changelog (lander pilot)", "",
             f"Gate: {GATE_MAP}, {n_all} fixed paired missions (normal + harsh). Before: classic {tot_c / n_all:.4f}, current pilot {tot_p / n_all:.4f}.",
             "Rule: accept only with zero lost missions (vs current pilot and vs classic) and strictly higher total success. Training maps: "
             + ", ".join(TRAIN_MAPS) + ". Site23 and the unseen benchmark maps are never used here.", "",
             "| round | lr | clip | pool share | td penalty | updates | candidate | current | classic | lost missions | new wins | decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    with open(log_md, "a", encoding="utf-8") as fh:                     # append: earlier rounds (and worldmodel.lander.solve rows) are history, never erased
        fh.write(("\n" if log_md.stat().st_size else "") + "\n".join(lines) + "\n")
    for r in range(1, a.rounds + 1):
        cfg = SCHEDULE[min(r - 1, len(SCHEDULE) - 1)]
        st = torch.load(model_path(PROD), map_location="cpu", weights_only=False)
        arch = dict(st["arch"])
        for f in model_path(CAND).parent.glob(CAND + "*.pt"):
            f.unlink()
        torch.save({"state_dict": st["state_dict"], "arch": st["arch"], "info": {"update": 0, "best_val_success": -1.0}}, model_path(CAND + "_last"))
        args = ["--name", CAND, "--resume", "--max-minutes", str(a.minutes), "--updates", "100000", "--eval-every", "100000", "--val-n", "16",
                "--val-sites", "Site01", "--sites", *TRAIN_MAPS, "--envs", str(a.envs), "--horizon", "64", "--mb-size", str(a.mb_size), "--epochs", "2",
                "--lr", str(cfg["lr"]), "--clip", str(cfg["clip"]), "--scenario-pool", a.pool, "--pool-frac", str(cfg["pool_frac"]),
                "--td-penalty", str(cfg["td_penalty"]), "--device", str(device),
                "--arch", "former", "--d", str(arch["d"]), "--layers", str(arch["layers"]), "--heads", str(arch["heads"])]
        rule = st["info"].get("residual_rule")              # authority rule learned by worldmodel.lander.solve: kept in training and in the gate
        if rule:
            args += ["--residual-rule", json.dumps(rule)]
        tr.main(args)
        last = torch.load(model_path(CAND + "_last"), map_location="cpu", weights_only=False)
        updates = int(last["info"].get("update", 0))
        last["info"] = {**last["info"], "residual_rule": rule}
        last.pop("optimizer", None)
        torch.save(last, model_path(CAND))
        cand_ok = fly_set(missions, CAND, device)
        g = gate(missions, cur_ok, cand_ok)
        t = g["totals"]
        row = {"round": r, "time_utc": datetime.now(UTC).isoformat(), **cfg, "updates": updates, "gate": g}
        if g["accept"]:
            shutil.copy2(model_path(PROD), model_path(f"{PROD}_before_rsi_r{r}"))
            cand = torch.load(model_path(CAND), map_location="cpu", weights_only=False)
            cand["info"] = {**st["info"], "rsi_round": r, "rsi_gate": {k: v for k, v in g.items() if k != "per_condition"},
                            "checkpoint_id": cand["info"].get("checkpoint_id", st["info"].get("checkpoint_id"))}   # the id of the NEW weights
            torch.save(cand, model_path(PROD))
            cur_ok = cand_ok
        with open(log_jl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        with open(log_md, "a", encoding="utf-8") as fh:
            fh.write(f"| {r} | {cfg['lr']} | {cfg['clip']} | {cfg['pool_frac']} | {cfg['td_penalty']} | {updates} | {t['candidate'] / t['n']:.4f} | "
                     f"{t['current'] / t['n']:.4f} | {t['classic'] / t['n']:.4f} | {g['regressions']} | "
                     f"{sum(p['new_wins'] for p in g['per_condition'])} | {'ACCEPTED' if g['accept'] else 'rejected'}: {g['reason']} |\n")
        print(f"RSI round {r}: candidate {t['candidate'] / t['n']:.4f} vs current {t['current'] / t['n']:.4f} (classic {t['classic'] / t['n']:.4f}); "
              f"lost {g['regressions']}; {'ACCEPTED' if g['accept'] else 'rejected'} ({g['reason']}) | {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
