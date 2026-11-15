"""Rover statistics for the website (website/assets/data/rover_stats.json). READ-ONLY on run files; nothing simulated here.

    ~/venvs/adi/bin/python website/tools/export_rover_stats.py

Sources (each block names its own):
  runs/ghosts_web/rover_lola5m_site23_2027-03-01.json   600 runs (200 per rule driver), held-out Site 23
  runs/ghosts/rover_lola5m_site23_2027-09-01.json       48 runs, Site 23, another date
  runs/ghosts/rover_lola5m_site06_2027-03-01.json       48 runs, Site 06 (training site)
  website/assets/data/rover_route_site23.json           the shown run cautious_15 (battery + sunlit fraction per hour)
  runs/rover/eval.json, runs/rover/train_log.jsonl      rule drivers vs the learned driver (not trained: smoke runs only)
Not in any file (left out): hazards avoided, comms / Earth windows, battery for runs other than cautious_15.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website/assets/data/rover_stats.json"
POL = ["naive", "cautious", "wait_light"]
POL_NAME = {"naive": "naive", "cautious": "cautious", "wait_light": "cautious + wait for light"}


def load(p):
    return json.loads((REPO / p).read_text())


def path_len(xyz):
    a = np.asarray(xyz, float)
    return float(np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1).sum()) if len(a) > 1 else 0.0


def by_policy(d, src):
    out = {}
    for p in POL:
        rs = [r for r in d["runs"] if r["policy"] == p]
        if not rs: continue
        c = Counter(r["outcome"] for r in rs)
        out[p] = {"n": len(rs), "returned": c.get("returned", 0), "died_battery": c.get("died_battery", 0), "timeout": c.get("timeout", 0),
                  "reached_ice": sum(1 for r in rs if r.get("reached_ice") in (True, "True")),
                  "ice_and_returned": sum(1 for r in rs if r.get("reached_ice") in (True, "True") and r["outcome"] == "returned")}
    return {"site": d["region"], "date": d.get("date"), "runs": len(d["runs"]), "by_policy": out, "source": src}


big_src = "runs/ghosts_web/rover_lola5m_site23_2027-03-01.json"
big = load(big_src)
worlds = [by_policy(big, big_src)]
for f in ["runs/ghosts/rover_lola5m_site23_2027-09-01.json", "runs/ghosts/rover_lola5m_site06_2027-03-01.json"]:
    if (REPO / f).exists(): worlds.append(by_policy(load(f), f))

# distance driven (path length over the ground) per run, 600-run set, histogram per policy
dist = {p: [path_len(r["xyz_m"]) for r in big["runs"] if r["policy"] == p] for p in POL}
mx = max(max(v) for v in dist.values())
edges = np.linspace(0, np.ceil(mx / 500) * 500, 13)
hist = {p: np.histogram(v, edges)[0].tolist() for p, v in dist.items()}
dist_block = {"edges_m": [round(float(e)) for e in edges], "hist": hist,
              "median_m": {p: round(float(np.median(v))) for p, v in dist.items()},
              "max_m": {p: round(float(np.max(v))) for p, v in dist.items()},
              "source": big_src + " (sum of hourly ground-track segments)"}

# range from the lander over time (a fan of runs; battery is only recorded for cautious_15)
lx, ly = big["lander_xyz_m"][0], big["lander_xyz_m"][1]
fan = []
rng = np.random.default_rng(0)
for p in POL:
    rs = [r for r in big["runs"] if r["policy"] == p]
    for r in [rs[i] for i in sorted(rng.choice(len(rs), size=min(40, len(rs)), replace=False))]:
        a = np.asarray(r["xyz_m"], float)
        fan.append({"policy": p, "outcome": r["outcome"],
                    "h": [round(t / 3600, 1) for t in r["t_s"][:len(a)]],
                    "range_m": [round(float(v)) for v in np.hypot(a[:, 0] - lx, a[:, 1] - ly)]})
fan_block = {"runs": fan, "note": "40 runs per driver, sampled with a fixed seed", "source": big_src}

# the shown run
rt = load("website/assets/data/rover_route_site23.json")
sun = np.asarray(rt["sun_frac"], float)
shown = {"run_id": rt["run_id"], "policy": rt["policy"], "outcome": rt["outcome"], "date": rt["date"],
         "h": [round(t / 3600, 1) for t in rt["t_s"]], "battery_wh": rt["battery_wh"], "battery_cap_wh": rt["battery_cap_wh"],
         "sun_frac": rt["sun_frac"], "total_dist_m": rt["total_dist_m"], "duration_h": rt["duration_h"], "max_range_m": rt["max_range_m"],
         "ice_reach_h": round(rt["ice_reach_t_s"] / 3600, 1) if rt.get("ice_reach_t_s") is not None else None,
         "hours_dark": int((sun <= 0.0).sum()), "hours_mostly_dark": int((sun < 0.5).sum()), "hours_total": int(len(sun)),
         "min_battery_wh": round(float(min(rt["battery_wh"])), 1),
         "source": "website/assets/data/rover_route_site23.json (" + rt["meta"]["source"] + "; battery: " + rt.get("battery_source", "") + ")"}

# learned driver vs the rule drivers (physics backup)
ev = load("runs/rover/eval.json")
tl = [json.loads(l) for l in (REPO / "runs/rover/train_log.jsonl").read_text().splitlines() if l.strip()]
driver = {"learned": {"status": "not trained", "checkpoint": ev.get("checkpoint"), "smoke_updates_logged": len(tl),
                      "smoke_reached_ice_any": any(x.get("reached_ice_per_episode", 0) > 0 for x in tl)},
          "rule_drivers": ev.get("summary"), "rovers_per_world": ev.get("rovers_per_world"),
          "source": "runs/rover/eval.json (checkpoint rover_smoke lists the rule drivers only); runs/rover/train_log.jsonl (smoke runs)"}

OUT.write_text(json.dumps({"meta": {"provenance": "modelled", "created_by": "website/tools/export_rover_stats.py",
                                    "note": "Simulation on NASA LOLA terrain + SPICE light. Ice targets = PSR edges (dark and cold, NOT confirmed ice). Left out (no data): hazards avoided, comms windows, battery of runs other than cautious_15."},
                           "policies": POL, "policy_names": POL_NAME, "worlds": worlds, "distance": dist_block,
                           "range_fan": fan_block, "shown_run": shown, "driver": driver}, separators=(",", ":")))
print("wrote", OUT, OUT.stat().st_size // 1024, "kB")
for w in worlds: print(w["site"], w["date"], w["by_policy"])
print(dist_block["median_m"], dist_block["max_m"]); print({k: shown[k] for k in ("hours_dark", "hours_mostly_dark", "hours_total", "min_battery_wh", "ice_reach_h")})
