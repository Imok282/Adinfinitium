"""Export real training curves for the website (website/assets/data/training_curves.json). READ-ONLY on logs; no training.

    ~/venvs/adi/bin/python website/tools/export_training_curves.py

Curves (each with its own source; a curve without a log is left out):
  pilot   results_pc/extracted/runs/pilot.log          validation success (Site 11) vs update, pilot vs classic baseline
  hazard  runs/hazard/train_log.jsonl                  validation F1 (Site 11) vs epoch (last complete run in the log)
  light   runs/light/train_log.jsonl                   validation skyline MAE (Site 11) vs step, net vs coarse ray-cast (longest run)
  dream   results_pc/extracted/planner_dream/history.jsonl  Dream-RSI generations: accepted / rejected, val Brier before/after
  neat    website/assets/data/neat_training.json       only if placeholder is false (else marked pending)
The terrain log (runs/terrain/train_log.jsonl) has 2 points only, so it is left out.
"""
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website/assets/data/training_curves.json"


def jl(p):
    return [json.loads(l) for l in (REPO / p).read_text().splitlines() if l.strip()]


def runs_by_restart(rows, key):
    out, cur = [], []
    for r in rows:
        if cur and r[key] <= cur[-1][key]:
            out.append(cur); cur = []
        cur.append(r)
    if cur: out.append(cur)
    return out


curves = {}
# ---- lander pilot: parse the validation lines of the training log (restarts kept as separate segments)
log = (REPO / "results_pc/extracted/runs/pilot.log").read_text().splitlines()
segs, cur, start_upd = [], None, 0
for line in log:
    m = re.match(r"start: validation success pilot ([\d.]+) \| classic baseline ([\d.]+)", line)
    if m:
        if cur: segs.append(cur)
        cur = {"points": [], "start_pilot": float(m[1]), "classic": float(m[2])}
        continue
    m = re.match(r"update\s+(\d+)/(\d+) val success pilot ([\d.]+) vs classic ([\d.]+)", line)
    if m and cur is not None:
        cur["points"].append([int(m[1]), float(m[3]), float(m[4])])
if cur: segs.append(cur)
segs = [s for s in segs if s["points"]]
if segs:
    a, b = segs[-2] if len(segs) > 1 else None, segs[-1]
    pts = []
    if a:
        pts += [[0, a["start_pilot"], a["classic"]]] + [p for p in a["points"] if p[0] <= 40]
    pts += [p for p in b["points"] if p[0] > (pts[-1][0] if pts else -1)]
    curves["pilot"] = {
        "title": "Landing pilot", "x": "update", "y": "validation success (Site 11)",
        "series": [{"name": "AI pilot", "points": [[p[0], p[1]] for p in pts]},
                   {"name": "classic guidance", "points": [[p[0], p[2]] for p in pts], "baseline": True}],
        "note": "Run restarted from the update-40 best checkpoint (d53a162df9, the one that ships); later updates did not beat it.",
        "source": "results_pc/extracted/runs/pilot.log (validation lines of the last two runs)",
    }
# ---- hazard U-Net
h = runs_by_restart(jl("runs/hazard/train_log.jsonl"), "epoch")
h = max(h, key=len)
curves["hazard"] = {"title": "Hazard model", "x": "epoch", "y": "validation F1 (Site 11)",
                    "series": [{"name": "U-Net", "points": [[r["epoch"], round(r["val"]["f1"], 4)] for r in h]}],
                    "source": "runs/hazard/train_log.jsonl (longest run)"}
# ---- light skyline
L = runs_by_restart(jl("runs/light/train_log.jsonl"), "step")
L2 = []                                   # also split where the baseline changes (a new run with another validation sample)
for run in L:
    cur = []
    for r in run:
        if cur and r["val"]["mae_baseline_deg"] != cur[-1]["val"]["mae_baseline_deg"]:
            L2.append(cur); cur = []
        cur.append(r)
    L2.append(cur)
L = max(L2, key=len)
curves["light"] = {"title": "Light model", "x": "step", "y": "skyline error, deg (Site 11, lower is better)",
                   "series": [{"name": "skyline net", "points": [[r["step"], round(r["val"]["mae_net_deg"], 4)] for r in L]},
                              {"name": "coarse ray-cast", "points": [[r["step"], round(r["val"]["mae_baseline_deg"], 4)] for r in L], "baseline": True}],
                   "lower_is_better": True, "source": "runs/light/train_log.jsonl (longest run)"}
# ---- Dream-RSI generations
D = [r for r in jl("results_pc/extracted/planner_dream/history.jsonl") if "accepted" in r]
curves["dream"] = {"title": "Dream-RSI (planner)", "x": "generation", "y": "validation Brier (lower is better)",
                   "steps": [{"generation": r["generation"], "accepted": bool(r["accepted"]), "before": r["val_brier_before"], "after": r["val_brier_after"]} for r in D],
                   "lower_is_better": True, "source": "results_pc/extracted/planner_dream/history.jsonl"}
# ---- NEAT rover (agent B's file): only when real
neat = REPO / "website/assets/data/neat_training.json"
if neat.exists():
    n = json.loads(neat.read_text())
    if n.get("placeholder") is False:
        curves["neat"] = {"title": "Rover driver (NEAT)", "x": "generation", "y": "best fitness",
                          "series": [{"name": "best", "points": [[g["generation"], g["best_fitness"]] for g in n["generations"]]},
                                     {"name": "mean", "points": [[g["generation"], g["mean_fitness"]] for g in n["generations"]], "baseline": True}],
                          "source": n["meta"]["source"]}
    else:
        curves["neat"] = {"title": "Rover driver (NEAT)", "pending": True, "note": "Training run results coming."}

OUT.write_text(json.dumps({"meta": {"provenance": "modelled", "created_by": "website/tools/export_training_curves.py",
                                    "note": "Real training / validation logs only; validation = Site 11. Test Site 23 is never in these curves."},
                           "curves": curves}, indent=1))
print("wrote", OUT, list(curves))
