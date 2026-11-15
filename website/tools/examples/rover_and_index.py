"""Rover v3.1 trace excerpt (from an existing file, nothing re-run) + website/assets/examples/index.json.

    ~/venvs/adi/bin/python -I website/tools/examples/rover_and_index.py

No v3.1 per-hour action/reason decision log exists in the repo or hand-off; only the route trace
website/assets/data/rover_route_dm1_v31.json (position, distance, battery, Sun per hour). This writes an excerpt of that trace
around the key events and says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
EX = ROOT / "website/assets/examples"


def rover():
    src = ROOT / "website/assets/data/rover_route_dm1_v31.json"
    d = json.loads(src.read_text())
    n = len(d["t_s"])
    ice = d["ice_reach_index"]
    keep = sorted(set(list(range(0, 6)) + list(range(max(ice - 3, 0), min(ice + 4, n))) + list(range(n - 4, n))))
    rows = []
    for i in keep:
        x, y, z = d["points"][i]
        rows.append({"hour": i, "t_s": d["t_s"][i], "x_east_m": x, "z_south_m": z, "height_rel_m": y,
                     "dist_travelled_m": d["dist_m"][i], "battery_wh": d["battery_wh"][i], "sun_frac": d["sun_frac"][i],
                     "event": ("start at lander" if i == 0 else "reached ice target (PSR edge)" if i == ice else "end of mission" if i == n - 1 else None)})
    out = {"model": "rover brain v3.1 (scripted, 0 learned parameters; uses hazard U-Net map + light model)",
           "is_decision_log": False,
           "note": "No per-hour action/reason decision log for v3.1 exists in the repo or the Mac mini hand-off; this is an excerpt of the "
                   "recorded route trace (state per hour) of rover 0, copied unchanged from the source file. Nothing was re-run.",
           "source": "website/assets/data/rover_route_dm1_v31.json <- " + d["meta"]["source"],
           "site": d["site"], "date": d["date"], "split": "DM1: not in train, validation or frozen held-out site lists (end-to-end test site)",
           "outcome": d["outcome"], "total_dist_m": d["total_dist_m"], "duration_h": d["duration_h"], "ice_reach_hour": ice,
           "target_distance_m": d["target_distance_m"], "min_distance_to_target_m": d["min_distance_to_target_m"],
           "end_distance_from_lander_m": d["end_distance_from_lander_m"], "battery_cap_wh": d["battery_cap_wh"],
           "fleet_512": d["fleet"], "ice_source": d["ice_source"], "provenance": "modelled (simulation)", "frame": d["meta"]["frame"],
           "rows": rows}
    p = EX / "rover/rover_v31_trace_excerpt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return p


def index():
    def files(sub):
        return sorted(str(p.relative_to(EX)) for p in (EX / sub).glob("*") if p.is_file())

    hz = json.loads((EX / "hazard/hazard_examples.json").read_text())
    lt = json.loads((EX / "light/light_examples.json").read_text())
    pl = json.loads((EX / "pilot/pilot_examples.json").read_text())
    pn = json.loads((EX / "planner/planner_examples.json").read_text())
    doc = {"generated_by": "website/tools/examples/*.py (read-only use of trained checkpoints; no training, no tuning)",
           "examples": [
               {"id": "hazard", "model": "hazard U-Net", "checkpoint": hz["checkpoint"], "checkpoint_id": hz["checkpoint_id"],
                "data": "hazard/hazard_examples.json", "files": files("hazard"),
                "input": hz["input"], "output": "per-pixel unsafe probability (threshold 0.5) for a 256 x 256 px (1.28 km) window",
                "fields": ["windows[].images.{input_hillshade,input_laser_pixels,unet_probability,unet_mask,backup_slope_rule,truth,unet_error,backup_error}",
                           "windows[].unet / backup_slope_rule: precision, recall, f1, missed_hazard_rate, tp/fp/fn/tn"],
                "truth": hz["truth"], "split": "Site23 (steep, moderate, flat) and Site42 (hard_case): TEST, never trained or tuned on",
                "trained_on": hz["trained_on"]},
               {"id": "light", "model": "skyline model (neural light field)", "checkpoint": lt["checkpoint"], "checkpoint_id": lt["checkpoint_id"],
                "data": "light/light_examples.json", "input": lt["input"],
                "output": "360-value skyline (deg) per cell; Sun fraction / Earth visibility via exact geometry",
                "fields": ["cells[].skyline_deg.{truth_raycast_5m,model,backup_coarse_raycast_20m}[360]", "cells[].error_deg",
                           "cells[].sunlit_hours / earth_visible_hours {truth,model,backup}",
                           "sun_earth_path.{times_utc,sun_az_deg,sun_elev_deg,earth_az_deg,earth_elev_deg}"],
                "truth": lt["truth"], "split": lt["split"] + " (Site23)", "trained_on": lt["trained_on"]},
               {"id": "pilot", "model": pl["model"], "checkpoint": pl["checkpoint"], "checkpoint_id": pl["checkpoint_id"],
                "data": "pilot/pilot_examples.json", "files": files("pilot"),
                "input": "state vector (21: nav estimate, altimeter, velocity, fuel, engines, hazard ring) + classic command + 32x32x3 camera crop",
                "output": "3-axis residual on the classic ZEM/ZEV thrust command",
                "fields": ["missions[].pilot_series / classic_series: t, alt_m, vx, vy, vz, classic[3], residual_raw[3], residual_applied[3], final[3], fuel_kg, engines_ok",
                           "missions[].outcome.{pilot,classic}", "missions[].camera_crops[]"],
                "map": pl["map"], "split": "validation map (Site11); not a test map", "trained_on": pl["trained_on"]},
               {"id": "planner", "model": pn["model"], "checkpoints": pn["checkpoints"], "data": "planner/planner_examples.json",
                "input": "13 features per candidate (cell, hour): sunlight, darkness, Earth link, hazard, slope, clear radius, ice-target distance",
                "output": "predicted p(mission) per candidate (learned) / Navigator score (backup); top-10 per ranker",
                "fields": ["top10.{navigator,set,logistic}[]: rank, cell_rc, hour, navigator_score, p_set, p_logistic, simulated, factors",
                           "all_site23_worlds[]", "site23_means_7_worlds"],
                "world": pn["world"], "split": "TEST (Site23), offline ranking of existing simulated labels, no flight"},
               {"id": "rover", "model": "rover brain v3.1", "data": "rover/rover_v31_trace_excerpt.json",
                "input": "hazard U-Net map + light model + battery state", "output": "hourly route (excerpt of the route trace; NOT an action/reason log)",
                "fields": ["rows[]: hour, x_east_m, z_south_m, dist_travelled_m, battery_wh, sun_frac, event"],
                "split": "DM1 end-to-end test site (2027-06-01)", "source": "existing file; nothing re-run"},
           ],
           "provenance_banner": "Terrain is OBSERVED NASA LOLA 5 m data; all model outputs, trajectories and labels are MODELLED / simulated. "
                                "Nothing here is GENERATED terrain. Not a safety claim on its own."}
    (EX / "index.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    print(rover())
    index()
    print("wrote index.json")
