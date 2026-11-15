"""Benchmarks page: CSV exports + stats.json (effect sizes, Holm-adjusted lander tests, methodology facts).

    python3 website/tools/export_bench_csv.py          # from the repo root

Reads (never writes) the hand-off records and config, writes:
  website/assets/data/csv/<family>.csv      one CSV per benchmarks.json family (values, 95% CI, n, source, note)
  website/assets/data/csv/lander_tests.csv  the paired lander tests with raw and Holm-adjusted p
  website/assets/data/stats.json            every number the benchmarks page shows that is not in benchmarks.json /
                                            projections.json, each with its source file
Inputs:
  website/assets/data/benchmarks.json (copy of data/handoff_macmini_1115/benchmarks_final.json)
  website/assets/data/mission_bench.json (optional; its "tests" join the Holm family if present)
  data/handoff_macmini_1115/lander_mini/runs/lander/benchmark.json, benchmark_site23_final.json
  data/handoff_macmini_1115/macbook/records/weak_flags/{hazard,light}.json, rover_v31/eval.json
  data/handoff_macmini_1115/frozen_run/runs/light/eval.json, frozen_run/runs/FROZEN.json
  config/mission.yaml, data/models/lander_pilot.pt (sha256 only)
Stdlib only. Bootstraps use random.Random(0) and 10,000 resamples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "website"
HO = ROOT / "data" / "handoff_macmini_1115"
REC = HO / "macbook" / "records"
OUT_CSV = WEB / "assets" / "data" / "csv"
B = 10_000


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def write_atomic(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        f.write(text)
    os.replace(tmp, p)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


# ---------- statistics ----------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial sign test on the discordant pairs)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def boot(units: list, stat, rng: random.Random) -> list[float]:
    out = []
    for _ in range(B):
        sample = [units[rng.randrange(len(units))] for _ in units]
        v = stat(sample)
        if v is not None and math.isfinite(v):
            out.append(v)
    return [pct(out, 0.025), pct(out, 0.975)]


def f1_pooled(ws: list[dict]) -> float | None:
    tp = sum(w["tp"] for w in ws); fp = sum(w["fp"] for w in ws); fn = sum(w["fn"] for w in ws)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else None


def main() -> None:
    rng = random.Random(0)
    bench = load(WEB / "assets" / "data" / "benchmarks.json")
    mb_path = WEB / "assets" / "data" / "mission_bench.json"
    mission_bench = load(mb_path) if mb_path.exists() else None

    # ---------- CSV per family ----------
    for fam in bench["families"]:
        mets = [m["id"] for m in fam["metrics"]]
        head = ["model", "ours"] + [c for m in mets for c in (m, f"{m}_ci95_lo", f"{m}_ci95_hi")] + ["n", "source", "note"]
        lines = []
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(head)
        for r in fam["rows"]:
            row = [r["name"], int(bool(r.get("ours")))]
            for m in mets:
                ci = (r.get("ci95") or {}).get(m) or ["", ""]
                row += [(r.get("values") or {}).get(m, ""), ci[0], ci[1]]
            row += [r.get("n", ""), r.get("source", ""), r.get("note", "")]
            w.writerow(row)
        write_atomic(OUT_CSV / f"{fam['id']}.csv", buf.getvalue())
        del lines

    # ---------- lander tests + Holm ----------
    lb = load(HO / "lander_mini/runs/lander/benchmark.json")
    l23 = load(HO / "lander_mini/runs/lander/benchmark_site23_final.json")
    tests = []
    for src, data, label in ((rel(HO / "lander_mini/runs/lander/benchmark.json"), lb, "unseen maps (Site42, Shoemaker, DM2)"),
                             (rel(HO / "lander_mini/runs/lander/benchmark_site23_final.json"), l23, "Site23 final exam")):
        for cond in ("nominal", "harsh"):
            s = data["summary"][cond]
            b, c = s["only_pilot_landed"], s["only_classic_landed"]
            tests.append({
                "id": f"{'unseen' if data is lb else 'site23'}_{cond}",
                "label": f"Pilot vs classic, {label}, {'normal' if cond == 'nominal' else 'harsh (failure rates x3)'} rates",
                "n_missions": s["missions"], "pilot_only_landed": b, "classic_only_landed": c,
                "diff_pts": round(100 * s["diff"], 2),
                "diff_ci95_pts": [round(100 * x, 2) for x in s["diff_ci95"]] if s.get("diff_ci95") else None,
                "p_file": s["sign_test_p"], "p_recomputed": mcnemar_exact(b, c), "source": src,
                "registered": "pre-specified" , "kind": "paired, fixed seeds",
            })
    if mission_bench:
        for t in mission_bench.get("tests", []) or []:
            if isinstance(t, dict) and isinstance(t.get("p"), (int, float)):
                tests.append({"id": str(t.get("id", "mission_bench")), "label": str(t.get("label", "mission_bench test")),
                              "p_file": float(t["p"]), "p_recomputed": None, "source": "website/assets/data/mission_bench.json",
                              "registered": str(t.get("registered", "post hoc")), "kind": str(t.get("kind", ""))})
    ps = [t["p_recomputed"] if t["p_recomputed"] is not None else t["p_file"] for t in tests]
    for t, a in zip(tests, holm(ps)):
        t["p_holm"] = a
    per_map = []
    for r in lb["rows"]:
        per_map.append({"site": r["site"], "condition": r["condition"], "pilot_only_landed": r["only_pilot_landed"],
                        "classic_only_landed": r["only_classic_landed"], "diff_pts": round(100 * r["diff"], 2),
                        "p_file": r["sign_test_p"], "p_recomputed": mcnemar_exact(r["only_pilot_landed"], r["only_classic_landed"])})
    import io
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["id", "label", "n_missions", "pilot_only_landed", "classic_only_landed", "diff_pts", "diff_ci95_lo", "diff_ci95_hi", "p_exact_mcnemar", "p_holm", "source"])
    for t in tests:
        ci = t.get("diff_ci95_pts") or ["", ""]
        w.writerow([t["id"], t["label"], t.get("n_missions", ""), t.get("pilot_only_landed", ""), t.get("classic_only_landed", ""),
                    t.get("diff_pts", ""), ci[0], ci[1], t["p_recomputed"] if t["p_recomputed"] is not None else t["p_file"], t["p_holm"], t["source"]])
    write_atomic(OUT_CSV / "lander_tests.csv", buf.getvalue())

    # ---------- effect sizes ----------
    eff = []
    hz = load(REC / "weak_flags/hazard.json")
    for group, gid in (("fresh", "fresh"), ("held_out", "heldout")):
        ws = [w for w in hz["windows"].values() if w["group"] == group]
        net = [w["network_frozen"]["all"] for w in ws]; bk = [w["physics_backup"]["all"] for w in ws]
        pairs = list(zip(net, bk))
        ratio = lambda S: (lambda a, b: a / b if a and b else None)(f1_pooled([p[0] for p in S]), f1_pooled([p[1] for p in S]))
        eff.append({"id": f"hazard_f1_ratio_{gid}", "label": f"Hazard U-Net F1 / slope-rule F1, {'fresh sites' if group == 'fresh' else 'Site23+42'}",
                    "value": ratio(pairs), "unit": "x", "ci95": boot(pairs, ratio, rng),
                    "ci_method": f"bootstrap over {len(ws)} windows (pooled confusion counts re-summed per resample)",
                    "source": rel(REC / "weak_flags/hazard.json")})
    le = load(HO / "frozen_run/runs/light/eval.json")
    tw = [r for r in le["regions"].values() if r.get("role") == "TEST"]
    lp = [(r["skyline_model"]["mae_deg"], r["skyline_baseline"]["mae_deg"]) for r in tw]
    red = lambda S: 100 * (1 - sum(a for a, _ in S) / sum(b for _, b in S))
    eff.append({"id": "light_reduction_heldout", "label": "Light: skyline error reduction vs coarse ray-cast, Site23+42",
                "value": red(lp), "unit": "%", "ci95": boot(lp, red, rng), "ci_method": f"bootstrap over {len(tw)} windows",
                "windows_better": sum(a < b for a, b in lp), "windows": len(lp), "source": rel(HO / "frozen_run/runs/light/eval.json")})
    lf = load(REC / "weak_flags/light.json")["fresh_sites"]["sites"]
    fp = [(s["skyline_frozen_3258"]["mae_deg"], s["skyline_backup"]["mae_deg"]) for s in lf.values()]
    eff.append({"id": "light_reduction_fresh", "label": "Light: skyline error reduction vs coarse ray-cast, fresh sites",
                "value": red(fp), "unit": "%", "ci95": boot(fp, red, rng), "ci_method": f"bootstrap over {len(fp)} sites",
                "windows_better": sum(a < b for a, b in fp), "windows": len(fp), "source": rel(REC / "weak_flags/light.json")})
    rv = load(REC / "rover_v31/eval.json")["A_untouched"]
    by_site: dict[str, list] = {}
    for w in rv["per_world"]:
        by_site.setdefault(w["site"], []).append((w["v31"]["strict_returned"], w["naive"]["strict_returned"], w["v31"]["strict_lost"], w["naive"]["strict_lost"]))
    sites = [(sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v), sum(x[2] for x in v) / len(v), sum(x[3] for x in v) / len(v)) for v in by_site.values()]
    rr = lambda S: sum(s[0] for s in S) / sum(s[1] for s in S) if sum(s[1] for s in S) else None
    eff.append({"id": "rover_returned_ratio", "label": "Rover v3.1 / naive planner, strict returned, LM2-LM5",
                "value": rv["methods"]["v31"]["strict_returned"]["mean"] / rv["methods"]["naive"]["strict_returned"]["mean"], "unit": "x",
                "ci95": boot(sites, rr, rng), "ci_method": f"bootstrap over {len(sites)} sites (mean of each site's {len(rv['per_world']) // len(sites)} worlds); rough with 4 sites",
                "diff_pts": round(100 * rv["v31_minus_best_planner_strict_returned"]["mean_difference"], 1),
                "diff_ci95_pts": [round(100 * x, 1) for x in rv["v31_minus_best_planner_strict_returned"]["ci95_site_bootstrap"]],
                "sites_higher": rv["v31_minus_best_planner_strict_returned"]["sites_where_a_is_higher"], "sites": len(sites),
                "source": rel(REC / "rover_v31/eval.json")})
    pilot = next(f for f in bench["families"] if f["id"] == "pilot")
    pv = {r["name"]: r for r in pilot["rows"]}
    ours = next(r for r in pilot["rows"] if r.get("ours")); cl = next(r for r in pilot["rows"] if not r.get("ours"))
    fo, fc = ours["extra_values"]["fuel_kg_successful_unseen_normal"], cl["extra_values"]["fuel_kg_successful_unseen_normal"]
    eff.append({"id": "pilot_fuel_reduction", "label": "Pilot fuel vs classic, successful landings, unseen maps, normal rates",
                "value": 100 * (1 - fo / fc), "unit": "%", "ci95": None,
                "ci_method": "no CI computable: only mean fuel per map is stored, not per mission; the two successful sets differ slightly",
                "source": "website/assets/data/benchmarks.json pilot extra_values.fuel_kg_successful_unseen_normal"})
    del pv

    # ---------- methodology facts ----------
    yml = (ROOT / "config/mission.yaml").read_text()
    def yv(key):
        m = re.search(rf"^\s*{key}:\s*([0-9.]+)", yml, re.M)
        return float(m.group(1)) if m else None
    held = [w for w in hz["windows"].values() if w["group"] == "held_out"]
    fresh = [w for w in hz["windows"].values() if w["group"] == "fresh"]
    allw = list(hz["windows"].values())
    facts = {
        "touchdown_limits": {"max_vertical_speed_m_s": yv("max_vertical_speed_m_s"), "max_horizontal_speed_m_s": yv("max_horizontal_speed_m_s"),
                             "max_slope_deg": yv("max_slope_deg"), "source": "config/mission.yaml lander.touchdown_limits"},
        "laser_fraction_heldout_windows": {"min_pct": 100 * min(w["measured_fraction"] for w in held), "max_pct": 100 * max(w["measured_fraction"] for w in held),
                                           "windows": len(held), "source": rel(REC / "weak_flags/hazard.json") + " windows[*].measured_fraction"},
        "laser_fraction_fresh_windows": {"min_pct": 100 * min(w["measured_fraction"] for w in fresh), "max_pct": 100 * max(w["measured_fraction"] for w in fresh),
                                         "windows": len(fresh), "source": rel(REC / "weak_flags/hazard.json") + " windows[*].measured_fraction"},
        "hazard_label_interpolated_pct": {"value": 100 * (1 - sum(w["measured_pixels"] for w in allw) / sum(w["pixels"] for w in allw)),
                                          "windows": len(allw), "source": rel(REC / "weak_flags/hazard.json") + " (1 - measured_pixels / pixels, all 25 test windows)"},
    }
    ck = []
    fz = load(HO / "frozen_run/runs/FROZEN.json")
    for k, v in fz["sha256"].items():
        if k.startswith("checkpoint/"):
            ck.append({"name": k.split("/", 1)[1], "sha256": v, "source": rel(HO / "frozen_run/runs/FROZEN.json")})
    pt = ROOT / "data/models/lander_pilot.pt"
    if pt.exists():
        ck.append({"name": "lander_pilot", "id": lb["checkpoint_id"], "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
                   "source": "data/models/lander_pilot.pt (sha256 computed by this script); id from " + rel(HO / "lander_mini/runs/lander/benchmark.json")})
    ck.append({"name": "hazard_unet_big (id)", "id": hz["checkpoint_id"], "source": rel(REC / "weak_flags/hazard.json")})
    ck.append({"name": "light_skyline_big (id)", "id": le["checkpoint_id"], "source": rel(HO / "frozen_run/runs/light/eval.json")})

    stats = {
        "meta": {"provenance": "modelled", "frame": "n/a", "units": "per field",
                 "source": "computed by website/tools/export_bench_csv.py from the hand-off records listed in its docstring",
                 "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "bootstrap": {"resamples": B, "seed": 0, "interval": "percentile 2.5-97.5"}},
        "lander_tests": {"family": "every pre-specified paired pilot-vs-classic comparison that was reported" +
                         (" plus the tests in mission_bench.json" if mission_bench else ""),
                         "n_tests": len(tests), "test": "exact McNemar (two-sided binomial on discordant missions)",
                         "adjustment": "Holm-Bonferroni over the family", "items": tests},
        "lander_per_map_post_hoc": {"note": "per-map breakdown of the unseen-normal test, looked at after the pooled result; not part of the Holm family",
                                    "source": rel(HO / "lander_mini/runs/lander/benchmark.json"), "items": per_map},
        "optional_files": {"mission_bench": bool(mission_bench)},
        "effects": eff,
        "facts": facts,
        "checkpoints": ck,
        "reproduce": {
            "branch": "nimansh/KAN-45-rover-brain",
            "commands": [
                {"what": "Pilot vs classic, unseen maps (normal + harsh)", "cmd": "python -m worldmodel.lander.benchmark --n 256 --seeds 3 --harsh 3 --out benchmark",
                 "source": "ops/lander_mini/run_lander_job.sh step 5"},
                {"what": "Site23 final exam (flown once; re-running it would no longer be a blind exam)", "cmd": "python -m worldmodel.lander.benchmark --final --n 512 --seeds 2 --harsh 3 --out benchmark_site23_final",
                 "source": "ops/lander_mini/run_lander_job.sh step 6"},
                {"what": "Hazard (frozen run)", "cmd": "python -m worldmodel.hazard.evaluate --name hazard_unet_big --figure", "source": "ops/overnight/profiles.py macmini"},
                {"what": "Light (frozen run)", "cmd": "python -m worldmodel.light.evaluate --name light_skyline_big", "source": "ops/overnight/profiles.py macmini"},
                {"what": "Terrain network (frozen run)", "cmd": "python -m worldmodel.terrain.evaluate --name terrain_diffusion_mini --patches 96 --versions 6", "source": "ops/overnight/profiles.py macmini"},
                {"what": "Rover v1 driver (frozen run)", "cmd": "python -m worldmodel.rover.evaluate --name rover_driver_mini --n 512", "source": "ops/overnight/profiles.py macmini"},
                {"what": "Planner", "cmd": "python -m worldmodel.planner.evaluate", "source": "worldmodel/planner/README.md"},
                {"what": "Rover v3.1, one world", "cmd": "python -m worldmodel.rover3.run eval --sites LM4 --dates 2027-03-01 --n 512 --seed 1234 --threads 1 --out /tmp/x.json",
                 "source": "data/handoff_macmini_1115/macbook/records/SUBMISSION_SUMMARY.md section 7"},
                {"what": "Fresh-site and strict-rule re-scoring (addendum 4)", "cmd": "records/weak_flags/scripts/{terrain/run.sh, terrain/run_fresh.sh, hazard/run.sh, rover/run_a.sh}",
                 "source": "data/handoff_macmini_1115/macbook/records/SUBMISSION_SUMMARY.md section 7"},
                {"what": "This page's statistics and CSV files", "cmd": "python3 website/tools/export_bench_csv.py", "source": "website/tools/export_bench_csv.py"},
            ],
            "warning": "The frozen-run evaluate commands overwrite the frozen eval.json files; run them in a fresh copy (SUBMISSION_SUMMARY section 7). Training is timed by the clock, so retraining does not reproduce numbers digit for digit.",
        },
    }
    write_atomic(WEB / "assets" / "data" / "stats.json", json.dumps(stats, indent=1) + "\n")
    for t in tests:
        print(f"{t['id']:16s} p={t['p_recomputed'] if t['p_recomputed'] is not None else t['p_file']:.4f} file={t['p_file']:.4f} holm={t['p_holm']:.4f}")
    for e in eff:
        print(e["id"], round(e["value"], 3), e["ci95"])


if __name__ == "__main__":
    main()
