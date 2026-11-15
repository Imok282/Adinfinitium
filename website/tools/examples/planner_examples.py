"""Offline ranking example for the mission planner on one held-out Site23 world -> website/assets/examples/planner/planner_examples.json.

    ~/venvs/adi/bin/python -I website/tools/examples/planner_examples.py

Read-only: loads the existing simulated-label files data/planner/lola5m_site23_<date>.npz (already used by worldmodel.planner.evaluate)
and the trained data/models/planner_set.pt + planner_logistic.pt. No flight, no simulation. The world shown is the Site23 date
whose best candidate has the highest simulated p(mission) (stated in the JSON); all 7 Site23 worlds are summarised too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from contracts.paths import model_path  # noqa: E402
from worldmodel.planner.data import load_groups, role  # noqa: E402
from worldmodel.planner.features import NAMES  # noqa: E402
from worldmodel.planner.metrics import decision, spearman  # noqa: E402
from worldmodel.planner.model import load_checkpoint  # noqa: E402
from worldmodel.planner.train import predict  # noqa: E402

OUT = ROOT / "website/assets/examples/planner"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = [g for g in load_groups() if g.region == "lola5m_site23"]
    assert groups and all(role(g) == "test" for g in groups)
    models = {k: load_checkpoint(model_path(f"planner_{k}")) for k in ("set", "logistic")}
    for _, info in models.values():
        assert "lola5m_site23" not in info.get("trained_on", []), "Site23 used in training"

    summary = []
    for g in groups:
        ps = {k: predict(m, g) for k, (m, _) in models.items()}
        row = {"date": g.date, "candidates": len(g.y), "best_simulated_p": round(float(g.y.max()), 4),
               "mean_simulated_p": round(float(g.y.mean()), 4)}
        for name, s in (("navigator", g.backup), ("set", ps["set"]), ("logistic", ps["logistic"])):
            d = decision(s, g.y)
            row[name] = {"chosen_simulated_p": round(float(d["chosen_p"]), 4), "regret": round(float(d["regret"]), 4),
                         "rank_corr": None if d.get("rank_corr") is None else round(float(d["rank_corr"]), 3)}
        summary.append(row)
    g = max(groups, key=lambda g: (g.y.max(), g.y.mean()))
    ps = {k: predict(m, g) for k, (m, _) in models.items()}
    f = dict(zip(NAMES, g.x.T, strict=True))
    ex = g.extra

    def cand(i):
        return {"id": int(i), "cell_rc": [int(v) for v in ex["cell_rc"][i]], "hour": int(ex["hour"][i]),
                "navigator_score": round(float(g.backup[i]), 3),
                "p_set": round(float(ps["set"][i]), 4), "p_logistic": round(float(ps["logistic"][i]), 4),
                "simulated": {"p_mission": round(float(g.y[i]), 4), "p_land": round(float(ex["p_land"][i]), 4),
                              "p_ice": round(float(ex["p_ice"][i]), 4), "p_return": round(float(ex["p_return"][i]), 4),
                              "touchdown_scatter_m": round(float(ex["scatter_m"][i]), 2)},
                "factors": {"slope_deg": round(float(f["slope_here"][i] * 20.0), 2),
                            "sunlit_mean_frac": round(float(f["sun_mean"][i]), 3),
                            "sunlit_first_day_frac": round(float(f["sun_first_day"][i]), 3),
                            "earth_visible_frac": round(float(f["earth_mean"][i]), 3),
                            "longest_dark_h": round(float(f["longest_dark_h"][i] * 300), 1),
                            "longest_blackout_h": round(float(f["longest_blackout_h"][i] * 300), 1),
                            "hazard_frac_150m": round(float(f["hazard_150m"][i]), 3),
                            "clear_radius_m": round(float(f["clear_radius_m"][i] * 400), 1),
                            "ice_target_distance_km": round(float(f["nearest_target_km"][i]), 3),
                            "ice_targets_within_1km": int(round(float(f["targets_within_1km"][i] * 8))),
                            "psr_frac_1km": round(float(f["psr_frac_1km"][i]), 3)}}

    rankings = {}
    for name, s in (("navigator", g.backup), ("set", ps["set"]), ("logistic", ps["logistic"])):
        order = np.argsort(-s, kind="stable")
        rankings[name] = [{"rank": r + 1, **cand(i)} for r, i in enumerate(order[:10])]
    truth_order = np.argsort(-g.y, kind="stable")
    doc = {
        "model": "mission planner: PlannerSetNet (listwise set transformer) + LogisticPlanner, vs the Navigator ranking (non-learned backup)",
        "checkpoints": {k: {"file": f"data/models/planner_{k}.pt", "checkpoint_id": i.get("checkpoint_id"), "parameters": i.get("parameters"),
                            "trained_on": i.get("trained_on")} for k, (_, i) in models.items()},
        "world": {"region": g.region, "site": "Site23 (Malapert massif)", "date": g.date, "candidates": len(g.y),
                  "split": "TEST (held out: never trained or tuned on)",
                  "chosen_because": "Site23 world whose best candidate has the highest simulated p(mission)"},
        "what_it_is": "offline ranking of candidate (landing cell, landing hour) pairs; no flight in this example. The simulated label "
                      "comes from the existing label file: the lander twin flies to the cell (p_land) and the rover environment drives "
                      "from it (p_ice, p_return); p_mission = p_land x p_return-with-ice",
        "label_provenance": data_note(g),
        "factor_units": {"slope_deg": "deg at the cell", "sunlit_mean_frac": "mean Sun fraction over the 300 h horizon",
                         "earth_visible_frac": "fraction of hours with Earth above the skyline", "ice_target_distance_km":
                         "distance to the nearest NASA PSR ice-edge target (PSR = dark and cold, not confirmed ice)",
                         "hazard_frac_150m": "share of unsafe cells within 150 m", "clear_radius_m": "distance to nearest unsafe cell (cap 400)"},
        "top10": rankings,
        "simulated_best10_ids": [int(i) for i in truth_order[:10]],
        "this_world_metrics": next(s for s in summary if s["date"] == g.date),
        "all_site23_worlds": summary,
        "rank_corr_note": "Spearman rank correlation between each ranking and the simulated p(mission) over all candidates of the world",
    }
    p = OUT / "planner_examples.json"
    p.write_text(json.dumps(doc, indent=1))
    print("world", g.region, g.date, "best y", g.y.max())
    for s in summary:
        print(s)
    print("wrote", p, p.stat().st_size)


def data_note(g):
    import numpy as np  # noqa: F401
    from contracts.paths import data_root
    z = np.load(data_root() / "planner" / f"{g.region}_{g.date}.npz", allow_pickle=False)
    return {"provenance": str(z["provenance"]), "lander_pilot_used": str(z["lander_pilot"]),
            "n_lander_sims_per_candidate": int(z["n_land"]), "n_rover_sims_per_candidate": int(z["n_rover"])}


if __name__ == "__main__":
    main()
