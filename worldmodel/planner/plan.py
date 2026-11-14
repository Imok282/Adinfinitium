"""Make a mission plan for one world: score candidate (cell, landing hour) pairs with the planner (or the exact Navigator ranking if no trained planner,
or if --backup), fly the lander to the chosen cell to measure its touchdown scatter, and write runs/<run>/plan_XXX.json (contracts/plan.py)
plus the orchestrator's decision log.

    python -m worldmodel.planner.plan --region lola5m_site23 --date 2027-03-01 --run demo
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from contracts.plan import Factor, Plan, RankedCandidate, write_plan
from contracts.provenance import Meta, Provenance
from worldmodel.lander.env import make_params
from worldmodel.lander.fly import fly
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.scenes import scene_for_region
from worldmodel.planner.features import NAMES, candidate_features, navigator_score
from worldmodel.planner.labels import pick_cells
from worldmodel.planner.orchestrator import decision_log, write_decisions
from worldmodel.rover.world import build_world, choose_targets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--date", default="2027-03-01")
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--hours", nargs="+", type=int, default=[24, 96, 168, 240, 312, 384])
    ap.add_argument("--go-threshold", type=float, default=0.2, help="assumption: minimum predicted success to commit to the earliest good slot")
    ap.add_argument("--backup", action="store_true", help="force the exact Navigator ranking")
    ap.add_argument("--name", default="planner_set")
    ap.add_argument("--run", default="plan")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_mission()
    dev = torch.device("cpu")
    w = build_world(a.region, a.date, cfg, device=dev)
    cells = pick_cells(w, a.cells, np.random.default_rng(a.seed))
    stride = cfg.rover_sim.light_stride_px
    cand = [(c, h) for c in cells for h in a.hours]
    x = np.stack([candidate_features(w, tuple(c), h, stride) for c, h in cand])
    backup = np.array([navigator_score(f) for f in x])
    ck = model_path(a.name)
    use_backup = a.backup or not ck.exists()
    if use_backup:
        p = None
        order = np.argsort(-backup)
        why = "exact Navigator ranking (" + ("forced" if a.backup else f"no trained planner {ck.name}") + ")"
    else:
        from worldmodel.planner.model import load_checkpoint

        model, _ = load_checkpoint(ck)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.as_tensor(x)[None]))[0].numpy()
        order = np.argsort(-p)
        why = f"learned planner {ck.name}"
    # earliest landing hour whose best candidate clears the bar => go; otherwise wait for the best later slot
    best = int(order[0])
    decision = "go"
    if p is not None:
        early = [i for i in order if p[i] >= a.go_threshold]
        if early:
            best = min(early, key=lambda i: (cand[i][1], -p[i]))
        else:
            decision = "wait"
    cell, hour = cand[best]
    scene = scene_for_region(a.region, cfg, dev)
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    xy = torch.as_tensor(scene.terrain.cell_to_xy((cell + np.array(w.origin_rc))[None]), dtype=torch.float32).repeat(128, 1)
    m = fly(scene, params, gp, None, 128, a.seed + 9, device=dev, targets=xy)
    ok = m["success"].numpy()
    scatter = float(np.percentile(m["target_dist_m"].numpy()[ok], 95)) if ok.any() else 999.0
    goals, ice_src = choose_targets(w.psr, w.slope, tuple(cell), w.mpp, cfg.rover.max_slope_deg, 4, np.random.default_rng(a.seed))
    t_land = datetime.fromisoformat(a.date + "T00:00:00+00:00") + timedelta(hours=int(hour))
    f = dict(zip(NAMES, x[best], strict=True))
    first_night_h = float(f["hours_to_first_night"]) * 300.0
    deadline = t_land + timedelta(hours=max(min(first_night_h - 24.0, 300.0), 12.0))
    z = (x[best] - x.mean(0)) / (x.std(0) + 1e-6)
    top = np.argsort(-np.abs(z))[:4]
    plan = Plan(decision=decision, region=a.region, landing_time_utc=t_land.strftime("%Y-%m-%dT%H:%M:%SZ"), target_cell_rc=(float(cell[0]), float(cell[1])),
                scatter_radius_m=scatter, rover_goals_rc=[(float(r), float(c)) for r, c in goals], return_deadline_utc=deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                p_success=float(p[best]) if p is not None else 0.0, p_success_backup_rank1=None, backup_used=use_backup,
                rationale=[Factor(name=NAMES[i], value=float(x[best][i]), note=f"{z[i]:+.1f} sd vs the other candidates") for i in top],
                ranking=[RankedCandidate(region=a.region, date=a.date, landing_hour=int(cand[i][1]), cell_rc=(float(cand[i][0][0]), float(cand[i][0][1])),
                                         p_success=float(p[i]) if p is not None else 0.0, backup_score=float(backup[i])) for i in order[:10]],
                meta=Meta(provenance=Provenance.MODELLED, source=f"{why}; light NASA SPICE DE440 + exact skyline; terrain NASA LOLA; ice goals: {ice_src}",
                          created_by="worldmodel.planner.plan", units={"scatter_radius_m": "m", "target_cell_rc": "px (5 m)"}))
    out = write_plan(run_dir(a.run), a.idx, plan)
    rows = decision_log(plan, w, stride)
    log = write_decisions(rows, run_dir(a.run) / f"decisions_{a.idx:03d}.csv")
    print(f"{decision.upper()}: {a.region} at {plan.landing_time_utc}, cell {plan.target_cell_rc}, scatter {scatter:.0f} m, p {plan.p_success:.2f} ({why})")
    print(f"wrote {out} and {log} ({len(rows)} decisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
