"""Dream-RSI: the mission planner keeps improving itself OFFLINE (e.g. on a laptop after training was cut off), safely.

One generation ("dream"):
  1. IMAGINE  pick a training world and propose hundreds of candidate missions (landing cell x landing hour) the planner has never seen;
  2. CHOOSE   keep the few it is least sure about: where its two brains (set transformer vs logistic) disagree most, plus a little exploitation
              of the most promising ones (active learning: spend simulation time where learning is largest);
  3. LIVE IT  play those missions out in the physics simulators that ship with it (lander twin + rover environment on NASA LOLA terrain and
              SPICE sunlight): no internet, no human labels, real simulated outcomes;
  4. LEARN    warm-start retrain on everything it has lived so far;
  5. GATE     keep the new version ONLY if it is better calibrated (lower Brier score) on a FROZEN validation set that dreams never touch.
              Otherwise roll back. It can not get worse, and the test sites (Site23, Site42) are never dreamt about.
Then repeat: each accepted version chooses the next dreams (recursive self-improvement). Everything is resumable (state in
<ADI_DATA>/planner_dream/) and logged (history.jsonl): which world, how many missions, old vs new validation score, accepted or not.

    python -m worldmodel.planner.dream --generations 3                 # a few rounds
    python -m worldmodel.planner.dream --forever --max-minutes 0       # keep improving whenever the laptop is on (Ctrl-C safe)

Honest framing: this is model-based active self-training with a physics simulator as the dream engine (in the spirit of Dreamer's
"learn in imagination"), with an acceptance gate. It improves the planner's calibration on the kinds of missions it was unsure about;
it does not invent new physics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import data_root, model_path
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, regions_for_site
from worldmodel.planner.data import Group, load_groups, role
from worldmodel.planner.features import NF, candidate_features, navigator_score
from worldmodel.planner.metrics import brier

DREAM_DATES = ("2027-01-01", "2027-03-01", "2027-04-15", "2027-06-01", "2027-07-15", "2027-09-01", "2027-12-01")
HOURS = (24, 96, 168, 240, 312, 384)


def state_dir() -> Path:
    return data_root() / "planner_dream"


def load_state() -> dict:
    p = state_dir() / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"generation": 0, "accepted": 0, "rejected": 0}


def save_state(st: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / "state.json").write_text(json.dumps(st, indent=2))


def log_history(row: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    with open(state_dir() / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def choose(p_set: np.ndarray, p_log: np.ndarray, k: int, explore: float = 0.3) -> np.ndarray:
    """Indices of the k candidates worth simulating: disagreement between the two predictors (uncertainty) + a little predicted value."""
    score = np.abs(p_set - p_log) + explore * p_set
    return np.argsort(-score)[:k]


def accept(old_brier: float, new_brier: float, tol: float = 1e-4) -> bool:
    """The gate: a new version must be strictly better on the frozen validation set."""
    return bool(np.isfinite(new_brier) and new_brier < old_brier - tol)


def split_val(groups: list[Group]) -> tuple[list[Group], list[Group]]:
    """Frozen validation = the validation sites (Site11). If none are on this machine, the first training world is set aside (never dreamt)."""
    train = [g for g in groups if role(g) == "train"]
    val = [g for g in groups if role(g) == "val"]
    if not val and len(train) > 1:
        first = min(train, key=lambda g: (g.region, g.date))
        val = [first]
        train = [g for g in train if g is not first]
    return train, val


def _predict(model, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32)[None]))[0].numpy()


def dream_once(gen: int, cfg, device, rng: np.random.Generator, n_cells: int, pool_cells: int, n_land: int, n_rover: int, quiet: bool = False) -> dict:
    """One full generation. Returns the history row."""
    from worldmodel.lander.env import make_params
    from worldmodel.lander.fly import fly
    from worldmodel.lander.guidance import guidance_params
    from worldmodel.lander.scenes import scene_for_region
    from worldmodel.planner.labels import pick_cells, rover_outcomes
    from worldmodel.planner.model import load_checkpoint, save_checkpoint
    from worldmodel.planner.train import fit
    from worldmodel.rover.world import build_world

    t0 = time.time()
    p_set, p_log = model_path("planner_set"), model_path("planner_logistic")
    if not (p_set.exists() and p_log.exists()):
        raise SystemExit("no trained planner yet (python -m worldmodel.planner.train)")
    m_set, info_set = load_checkpoint(p_set)
    m_log, _ = load_checkpoint(p_log)
    # 1. imagine: a training world (never a test site), many unseen candidates
    worlds = [(r, d) for s in TRAIN_SITES for r in regions_for_site(s)[:1] for d in DREAM_DATES]
    region, date = worlds[int(rng.integers(len(worlds)))]
    assert region.split("lola5m_")[1].replace("site", "Site") not in HOLDOUT_SITES
    w = build_world(region, date, cfg, device=device)
    cells = pick_cells(w, pool_cells, rng)
    if len(cells) == 0:
        return {"generation": gen, "world": f"{region}@{date}", "skipped": "no safe landing cells in this world"}
    stride = cfg.rover_sim.light_stride_px
    cand = [(c, h) for c in cells for h in HOURS]
    x = np.stack([candidate_features(w, tuple(c), h, stride) for c, h in cand]).astype(np.float32)
    # 2. choose where the planner is least sure (per cell: mean disagreement over its hours)
    ps, pl = _predict(m_set, x), _predict(m_log, x)
    nh = len(HOURS)
    dis = (np.abs(ps - pl) + 0.3 * ps).reshape(len(cells), nh)                     # uncertainty + a little predicted value, per (cell, hour)
    per_cell = dis.mean(1)
    pick = np.argsort(-per_cell)[:n_cells]
    chosen = cells[pick]
    # 3. live it: physics simulators score the chosen missions
    scene = scene_for_region(region, cfg, device)
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    xy = torch.as_tensor(scene.terrain.cell_to_xy(chosen + np.array(w.origin_rc)), dtype=torch.float32, device=device)
    m = fly(scene, params, gp, None, len(chosen) * n_land, int(rng.integers(1 << 30)), 5, device=device, targets=xy.repeat_interleave(n_land, dim=0))
    ok = m["success"].cpu().numpy().reshape(len(chosen), n_land)
    dist = m["target_dist_m"].cpu().numpy().reshape(len(chosen), n_land)
    p_land = ok.mean(1)
    scatter = np.array([np.percentile(dist[i][ok[i]], 95) if ok[i].any() else 300.0 for i in range(len(chosen))])
    feats, y_ice, y_ret, y_mis, idx, hr = [], [], [], [], [], []
    for i, cell in enumerate(chosen):
        outs = rover_outcomes(w, cell, list(HOURS), n_rover, cfg, device, int(rng.integers(1 << 30)))
        for h, (pi, pr) in zip(HOURS, outs, strict=True):
            feats.append(candidate_features(w, tuple(cell), h, stride))
            y_ice.append(pi); y_ret.append(pr); y_mis.append(p_land[i] * pr); idx.append(i); hr.append(h)
    feats = np.array(feats, np.float32)
    out = data_root() / "planner" / f"dream_g{gen:04d}_{region}_{date}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, p_ice=np.array(y_ice, np.float32), p_return=np.array(y_ret, np.float32), p_mission=np.array(y_mis, np.float32),
                        p_land=p_land[np.array(idx)].astype(np.float32), scatter_m=scatter[np.array(idx)].astype(np.float32), cell_rc=chosen[np.array(idx)],
                        hour=np.array(hr), n_land=n_land, n_rover=n_rover, backup=np.array([navigator_score(f) for f in feats], np.float32),
                        region=region, date=date, lander_pilot="classic ZEM/ZEV (physics)", provenance=f"modelled: Dream-RSI generation {gen}, simulated outcomes")
    # 4. learn (warm start) on everything lived so far, 5. gate on the frozen validation set
    train, val = split_val(load_groups())
    old_b = float(np.mean([brier(_predict(m_set, g.x), g.y) for g in val]))
    new_set, _ = fit("set", train, val, 60, 5e-4, gen, init_state=m_set.state_dict(), **{k: v for k, v in m_set.arch().items() if k != "kind"})
    new_b = float(np.mean([brier(_predict(new_set, g.x), g.y) for g in val]))
    ok_gate = accept(old_b, new_b)
    if ok_gate:
        shutil.copy2(p_set, p_set.with_name("planner_set_prev.pt"))
        info = {**{k: v for k, v in info_set.items() if k != "checkpoint_id"}, "dream_generation": gen, "best_val_brier": new_b,
                "trained_on": sorted({g.region for g in train}), "val_regions": sorted({g.region for g in val})}
        save_checkpoint(p_set, new_set, info)
        new_log, _ = fit("logistic", train, val, 120, 3e-3, gen, init_state=m_log.state_dict(), nf=NF)
        save_checkpoint(p_log, new_log, {"dream_generation": gen, "trained_on": sorted({g.region for g in train})})
    row = {"generation": gen, "time_utc": datetime.now(UTC).isoformat(), "world": f"{region}@{date}", "candidates_imagined": len(cand),
           "missions_simulated": int(len(chosen) * (n_land + n_rover * len(HOURS))), "cells_chosen": len(chosen),
           "mean_disagreement_chosen": round(float(per_cell[pick].mean()), 4), "simulated_p_mission_mean": round(float(np.mean(y_mis)), 4),
           "val_brier_before": round(old_b, 5), "val_brier_after": round(new_b, 5), "accepted": ok_gate, "train_worlds": len(train),
           "val_worlds": [f"{g.region}@{g.date}" for g in val][:3], "seconds": round(time.time() - t0, 1), "file": out.name}
    if not quiet:
        print(f"dream {gen}: {row['world']} imagined {len(cand)}, simulated {len(chosen)} cells | val Brier {old_b:.4f} -> {new_b:.4f} "
              f"{'ACCEPTED' if ok_gate else 'rejected (kept the previous version)'} | {row['seconds']:.0f}s", flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--forever", action="store_true", help="keep dreaming until stopped (Ctrl-C safe; resumes next time)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop starting new generations after this many minutes (0 = no limit)")
    ap.add_argument("--cells", type=int, default=6, help="missions (landing cells x 6 hours) simulated per generation")
    ap.add_argument("--pool", type=int, default=60, help="candidate landing cells imagined per generation")
    ap.add_argument("--n-land", type=int, default=32)
    ap.add_argument("--n-rover", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = load_mission()
    st = load_state()
    rng = np.random.default_rng(1000 + st["generation"])
    t0 = time.time()
    done = 0
    try:
        while a.forever or done < a.generations:
            if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
                break
            gen = st["generation"] + 1
            try:
                row = dream_once(gen, cfg, torch.device(a.device), rng, a.cells, a.pool, a.n_land, a.n_rover)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001 - a bad dream is logged, never fatal
                row = {"generation": gen, "error": repr(e)[:300]}
                print(f"dream {gen}: FAILED {e!r}", flush=True)
            log_history(row)
            st["generation"] = gen
            st["accepted"] += int(bool(row.get("accepted")))
            st["rejected"] += int("accepted" in row and not row["accepted"])
            save_state(st)
            done += 1
    except KeyboardInterrupt:
        print("stopped; state saved, the next run resumes", flush=True)
    print(f"Dream-RSI: {st['generation']} generations so far, {st['accepted']} accepted, {st['rejected']} rejected (state in {state_dir()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
