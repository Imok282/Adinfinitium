"""Solve-then-learn loop for the lander pilot (the lander's Dream-RSI, no gradient training needed, runs on a laptop):

  1. EXECUTE   the current pilot flies missions in the physics twin; the failed and near-miss ones are the scenario pool
               (worldmodel.lander.analysis mine: training maps only);
  2. RETRY     every pooled mission is replayed, same hidden failures, under a ladder of alternative strategies until one lands it:
               classic guidance alone, the pilot with less authority, the pilot handing over to classic guidance below some altitude;
               the table "which strategy solves which failure" is the solution library (runs/lander/solve_report.md, solutions.npz);
  3. LEARN     the strategy that solves the most pooled missions becomes a candidate RULE for the production pilot (stored in the
               checkpoint as info["residual_rule"], applied by worldmodel.lander.pilot.authority). It only uses what the lander can
               measure (its altitude), never the hidden failure;
  4. GATE      same gate as targeted Dream-RSI: Site11 fixed paired missions, normal + harsh. Accept only with ZERO lost missions
               (vs the current pilot and vs classic) and strictly higher success. Otherwise the next-best strategy is tried.
  never        Site23 (final exam) and the unseen benchmark maps.

    python -m worldmodel.lander.solve --pool data/pilot_dream/failure_pool.npz --try 2

The network weights are not changed, so an untrained residual still equals classic guidance, and a rule can be removed by deleting one key.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import data_root, model_path, run_dir
from worldmodel.lander.env import LanderBatch, make_params
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.pilot import RESIDUAL_SCALE, load_checkpoint, pick_device, pilot_features
from worldmodel.lander.rsi_targeted import GATE_MAP, PROD, TRAIN_MAPS, fly_set, gate, gate_missions
from worldmodel.lander.scenes import load_scenes

CAND = "lander_pilot_solve_cand"
# name -> rule (None = current pilot, "classic" = residual off). Deployable rules use only the measured altitude.
LADDER: dict[str, dict | str | None] = {
    "current pilot": None,
    "classic guidance only": "classic",
    "pilot, hand over to classic below 5-15 m": {"fade_lo": 5.0, "fade_hi": 15.0},
    "pilot, hand over to classic below 20-40 m": {"fade_lo": 20.0, "fade_hi": 40.0},
    "pilot, hand over to classic below 40-80 m": {"fade_lo": 40.0, "fade_hi": 80.0},
    "pilot, hand over to classic below 80-160 m": {"fade_lo": 80.0, "fade_hi": 160.0},
    "pilot at 60 % authority": {"scale": 0.6},
    "pilot at 60 % authority, hand over below 20-40 m": {"fade_lo": 20.0, "fade_hi": 40.0, "scale": 0.6},
}


def _weights(rule, alt: torch.Tensor) -> torch.Tensor:
    if rule == "classic":
        return torch.zeros_like(alt)
    if not rule:
        return torch.ones_like(alt)
    lo, hi = rule.get("fade_lo", -1.0), rule.get("fade_hi", 0.0)
    return rule.get("scale", 1.0) * ((alt - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


@torch.no_grad()
def fly_ladder(scene, params, gp, model, init: dict, ladder: dict, seed: int, device, repeat: int = 5) -> dict:
    """Replay the m missions in `init` once per strategy (all in one batch). Returns success [K, m] and touchdown numbers."""
    m, K = len(init["pos"]), len(ladder)
    n = m * K
    env = LanderBatch(params, scene.terrain, n, device, seed=seed, targets=torch.zeros(n, 2, device=device))
    env.load_scenarios(torch.arange(n, device=device), {k: np.concatenate([np.asarray(v)] * K) for k, v in init.items()})
    base = getattr(model, "rule", None)                     # "current pilot" = the production pilot WITH the rule it already has (as fly_recorded flies it)
    rules = [base if v is None else v for v in ladder.values()]
    obs = env.observe()
    r = torch.zeros(n, 3, device=device)
    for i in range(params.n_steps):
        classic = classic_action(gp, obs)
        if i % repeat == 0:
            r = model(pilot_features(obs, classic), env.crop(32))[0].float()
        alt = obs[:, 2:3] * 1000.0
        w = torch.cat([_weights(rules[k], alt[k * m:(k + 1) * m]) for k in range(K)])
        obs, _, _ = env.step((classic + RESIDUAL_SCALE * w * r.clamp(-1.0, 1.0)).clamp(-1.0, 1.0))
        if bool(env.done.all()):
            break
    mt = env.metrics()
    sh = lambda k: mt[k].float().cpu().numpy().reshape(K, m)  # noqa: E731
    return {"ok": sh("success").astype(bool), "td_vh": sh("td_vh"), "td_vz": sh("td_vz"), "miss": sh("target_dist_m")}


def retry(pool_path: str, device, ladder: dict = LADDER) -> dict:
    """Step 2: replay every pooled failure under every strategy."""
    cfg = load_mission()
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    model = load_checkpoint(model_path(PROD), device)[0]
    z = np.load(pool_path, allow_pickle=False)
    regions = z["region"].astype(str)
    assert not any(GATE_MAP.lower() in r or "site23" in r for r in regions), "the pool must come from training maps only"
    scenes = {s.region: s for s in load_scenes(TRAIN_MAPS, cfg, device, "backup", 1)}
    ok = np.zeros((len(ladder), len(regions)), dtype=bool)
    vh = np.zeros(ok.shape, dtype=np.float32)
    flown = np.zeros(len(regions), dtype=bool)
    for reg in np.unique(regions):
        if reg not in scenes:
            print(f"  {reg}: not a loaded training map; its {int((regions == reg).sum())} pooled missions are left out", flush=True)
            continue
        sel = np.nonzero(regions == reg)[0]
        init = {k: z[k][sel] for k in LanderBatch.SCENARIO_KEYS}
        res = fly_ladder(scenes[reg], params, gp, model, init, ladder, 4242, device)
        ok[:, sel], vh[:, sel] = res["ok"], res["td_vh"]
        flown[sel] = True
        print(f"  retried {len(sel)} missions on {reg}: " + ", ".join(f"{int(o.sum())}" for o in res["ok"]), flush=True)
    return {"ok": ok[:, flown], "td_vh": vh[:, flown], "why": z["why"].astype(str)[flown], "regions": regions[flown]}   # never count unflown missions as crashes


def write_report(R: dict, ladder: dict, order: list[str]) -> None:
    names = list(ladder)
    ok, why = R["ok"], R["why"]
    n = ok.shape[1]
    cur = ok[names.index("current pilot")]
    first = np.full(n, -1)
    for k in range(len(names)):
        first = np.where((first < 0) & ok[k], k, first)
    lines = ["# Lander solve-then-learn: retries of failed and near-miss missions", "",
             f"{n} pooled missions from the training maps ({dict(Counter(why))}), each replayed with the same hidden failures under every strategy.",
             f"On replay the current pilot lands {int(cur.sum())}; {int((first >= 0).sum())} are landed by at least one strategy; "
             f"{int((first < 0).sum())} are landed by none (no strategy in the ladder survives them).", "",
             "| strategy | pooled missions landed | of the current pilot's crashes, solved | of the current pilot's landings, lost | mean touchdown sideways speed m/s |",
             "|---|---|---|---|---|"]
    for k, nm in enumerate(names):
        lines.append(f"| {nm} | {int(ok[k].sum())} / {n} | {int((ok[k] & ~cur).sum())} / {int((~cur).sum())} | {int((~ok[k] & cur).sum())} | {R['td_vh'][k].mean():.3f} |")
    lines += ["", "Order in which deployable strategies go to the Site11 gate: " + "; ".join(order) + "."]
    p = run_dir("lander") / "solve_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_root() / "pilot_dream").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_root() / "pilot_dream" / "solutions.npz", ok=ok, strategies=np.array(names), first_solver=first, why=why, region=R["regions"])
    print("\n".join(lines), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--try", dest="tries", type=int, default=2, help="how many of the best strategies may go to the gate")
    ap.add_argument("--gate-n", type=int, default=256)
    ap.add_argument("--gate-seeds", type=int, default=2)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    device = pick_device(a.device)
    t0 = time.time()
    R = retry(a.pool, device)
    names = list(LADDER)
    cur_score = int(R["ok"][0].sum())
    deploy = [(int(R["ok"][k].sum()), -k, nm) for k, nm in enumerate(names) if isinstance(LADDER[nm], dict)]
    order = [nm for sc, _, nm in sorted(deploy, reverse=True) if sc > cur_score][: a.tries]
    write_report(R, LADDER, order)
    if not order:
        print("no strategy beats the current pilot on the pooled missions: nothing goes to the gate", flush=True)
        return 0
    missions = gate_missions(a.gate_n, a.gate_seeds, device)
    cur_ok = fly_set(missions, PROD, device)
    log_md, log_jl = run_dir("lander") / "rsi_changelog.md", run_dir("lander") / "rsi_changelog.jsonl"
    with open(log_md, "a", encoding="utf-8") as fh:
        fh.write("\n## Solve-then-learn rounds (worldmodel.lander.solve): authority rules, network weights unchanged\n\n"
                 "| round | rule | pooled missions landed (training maps) | candidate | current | classic | lost missions | new wins | decision |\n|---|---|---|---|---|---|---|---|---|\n")
    for r, nm in enumerate(order, 1):
        st = torch.load(model_path(PROD), map_location="cpu", weights_only=False)
        st["info"] = {**st["info"], "residual_rule": LADDER[nm]}
        torch.save(st, model_path(CAND))
        cand_ok = fly_set(missions, CAND, device)
        g = gate(missions, cur_ok, cand_ok)
        t = g["totals"]
        if g["accept"]:
            shutil.copy2(model_path(PROD), model_path(f"{PROD}_before_solve_r{r}"))
            st["info"]["solve_gate"] = {k: v for k, v in g.items() if k != "per_condition"}
            torch.save(st, model_path(PROD))
            cur_ok = cand_ok
        pooled = int(R["ok"][names.index(nm)].sum())
        with open(log_jl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "solve", "round": r, "time_utc": datetime.now(UTC).isoformat(), "rule": LADDER[nm], "name": nm,
                                 "pooled_landed": pooled, "pooled_n": int(R["ok"].shape[1]), "gate": g}, default=float) + "\n")
        with open(log_md, "a", encoding="utf-8") as fh:
            fh.write(f"| {r} | {nm} | {pooled} / {R['ok'].shape[1]} (current {cur_score}) | {t['candidate'] / t['n']:.4f} | {t['current'] / t['n']:.4f} | "
                     f"{t['classic'] / t['n']:.4f} | {g['regressions']} | {sum(p['new_wins'] for p in g['per_condition'])} | "
                     f"{'ACCEPTED' if g['accept'] else 'rejected'}: {g['reason']} |\n")
        print(f"solve round {r} [{nm}]: candidate {t['candidate'] / t['n']:.4f} vs current {t['current'] / t['n']:.4f} (classic {t['classic'] / t['n']:.4f}); "
              f"lost {g['regressions']}; {'ACCEPTED' if g['accept'] else 'rejected'} | {time.time() - t0:.0f}s", flush=True)
        if g["accept"]:
            break
    model_path(CAND).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
