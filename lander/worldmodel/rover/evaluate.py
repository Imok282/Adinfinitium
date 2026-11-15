"""Held-out evaluation of the rover driver against the three physics baselines, on the SAME starts, in worlds the model never trained on.

    python -I -m worldmodel.rover.evaluate --name rover_driver_big --n 1024
    python -I -m worldmodel.rover.evaluate --sites Site23 Site42 --dates 2027-03-01 2027-06-01 2027-09-01 2027-12-01

Worlds = (test site window, month of 2027): real LOLA terrain, real SPICE sunlight, NASA PSR ice targets. Reports reached-ice / returned-home / died
rates and the mean episode return for: naive planner, cautious planner (turn back on battery), cautious + wait-for-light, and the learned driver (greedy).
Honest: simulation only (kinematic, hourly steps); PSR = dark and cold, not confirmed ice; a model that does not beat the baselines is reported as such.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from worldmodel.hazard.lola import HOLDOUT_SITES, regions_for_site

DATES = ("2027-03-01", "2027-06-01", "2027-09-01", "2027-12-01")
METHODS = ("naive", "cautious", "cautious_wait_light", "model")


def summarise(per_world: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for m in METHODS:
        rows = [w[m] for w in per_world.values() if m in w]
        if rows:
            out[m] = {k: float(np.mean([r[k] for r in rows])) for k in ("reached_ice", "returned", "died", "mean_return")}
            out[m]["worlds"] = len(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    import torch

    from worldmodel.rover.policy import load_checkpoint, pick_device
    from worldmodel.rover.train import evaluate_policy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="rover_driver")
    ap.add_argument("--sites", nargs="+", default=HOLDOUT_SITES)
    ap.add_argument("--dates", nargs="+", default=list(DATES))
    ap.add_argument("--n", type=int, default=1024, help="rovers per world")
    ap.add_argument("--windows-per-site", type=int, default=1)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_mission()
    device = pick_device(a.device)
    ckpt = model_path(a.name)
    model = None
    info = {}
    if ckpt.exists():
        model, info = load_checkpoint(ckpt, device)
        model.eval()
    else:
        print(f"note: no checkpoint {ckpt.name}: evaluating the physics baselines only", flush=True)
    per_world: dict[str, dict] = {}
    for site in a.sites:
        regions = regions_for_site(site)[: a.windows_per_site]
        if not regions:
            print(f"note: no tiles for {site} on this machine; skipped", flush=True)
        for region in regions:
            for date in a.dates:
                with torch.no_grad():
                    per_world[f"{region}@{date}"] = evaluate_policy(model, region, date, cfg, device, n=a.n)
    summary = summarise(per_world)
    report = {"checkpoint": a.name, "checkpoint_id": info.get("checkpoint_id"), "parameters": sum(p.numel() for p in model.parameters()) if model is not None else None,
              "test_sites": a.sites, "rovers_per_world": a.n, "worlds": per_world, "summary": summary,
              "note": "held-out test sites, never trained on; simulation only; PSR = dark and cold, not confirmed ice"}
    out = run_dir("rover") / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'method':22s} {'ice':>6s} {'home':>6s} {'died':>6s} {'return':>8s}   ({len(per_world)} held-out worlds x {a.n} rovers)")
    for m in METHODS:
        if m in summary:
            s = summary[m]
            print(f"{m:22s} {s['reached_ice']:6.1%} {s['returned']:6.1%} {s['died']:6.1%} {s['mean_return']:8.1f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
