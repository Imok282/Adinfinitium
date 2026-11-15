"""Held-out exam for the mission planner (TEST Site23 + Site42 worlds, never used in training): calibration (Brier, ECE) and decisions (rank correlation,
chosen-candidate success, regret vs the best candidate) for the set-transformer planner, the logistic baseline, the exact Navigator ranking (backup) and a
random pick. Writes <ADI_RUNS>/planner/eval.json. Missing models are skipped with a note.

    python -m worldmodel.planner.evaluate
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from contracts.paths import model_path, run_dir
from worldmodel.planner.data import load_groups, role
from worldmodel.planner.metrics import brier, decision, ece, summarize
from worldmodel.planner.model import load_checkpoint
from worldmodel.planner.train import predict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="planner")
    ap.add_argument("--roles", nargs="+", default=["test", "val"])
    a = ap.parse_args(argv)
    groups = load_groups()
    models = {}
    for kind in ("set", "logistic"):
        p = model_path(f"{a.name}_{kind}")
        if p.exists():
            models[kind] = load_checkpoint(p)
        else:
            print(f"note: no {p.name}", flush=True)
    res = {"label": "simulated p(mission) per candidate", "models": {k: {"checkpoint_id": i.get("checkpoint_id"), "parameters": i.get("parameters"),
                                                                          "trained_on": i.get("trained_on")} for k, (_, i) in models.items()}}
    for r in a.roles:
        gs = [g for g in groups if role(g) == r]
        if not gs:
            res[r] = {"note": "no label files for these sites"}
            continue
        for (_, info) in models.values():
            assert not {g.region for g in gs if r == "test"} & set(info.get("trained_on", [])), "a test world was used for training"
        rng = np.random.default_rng(0)
        out = {"worlds": [f"{g.region}@{g.date}" for g in gs], "candidates": int(sum(len(g.y) for g in gs)),
               "simulated_p_mission_mean": float(np.mean(np.concatenate([g.y for g in gs])))}
        scorers = {"navigator_backup": lambda g: g.backup, "random": lambda g, rng=rng: rng.random(len(g.y))}
        for k, (m, _) in models.items():
            scorers[k] = lambda g, m=m: predict(m, g)
        for name, fn in scorers.items():
            per, ps, ys = [], [], []
            for g in gs:
                s = fn(g)
                per.append(decision(s, g.y))
                ps.append(s)
                ys.append(g.y)
            entry = summarize(per)
            if name in models:
                p, y = np.concatenate(ps), np.concatenate(ys)
                entry.update({"brier": brier(p, y), "ece": ece(p, y), "brier_constant_mean": brier(np.full_like(y, y.mean()), y)})
            out[name] = entry
        res[r] = out
        print(f"[{r}] {len(gs)} worlds, {out['candidates']} candidates, simulated mean p_mission {out['simulated_p_mission_mean']:.3f}", flush=True)
        for name in scorers:
            e = out[name]
            extra = f" brier {e['brier']:.4f} (constant {e['brier_constant_mean']:.4f}) ece {e['ece']:.3f}" if "brier" in e else ""
            rc = f"{e['mean_rank_corr']:+.2f}" if e["mean_rank_corr"] is not None else "  n/a"
            print(f"   {name:17s} chosen p {e['mean_chosen_p']:.3f} (best {e['mean_best_p']:.3f}, regret {e['mean_regret']:.3f}) rank corr {rc}{extra}", flush=True)
    out_p = run_dir("planner") / "eval.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
