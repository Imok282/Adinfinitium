"""Rover v3: collect the stage files of the final evaluation into eval.json and the result tables, and apply the pre-registered verdict rules
(ROVER_V3_PLAN.md section 6 and Amendment 1). Reads only; computes means, site-level bootstrap intervals and the MET / NOT MET labels.

    python -m worldmodel.rover3.report --dir ~/adinfinitium-rover3/runs/final --val ~/adinfinitium-rover3/runs/val --out ~/adinfinitium-rover3/runs/final
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = ("naive", "cautious", "cautious_wait_light")
KEYS = ("reached_ice", "returned", "died", "mean_return")
BOOT_N, BOOT_SEED = 10000, 20261115


def load(d: Path, prefix: str) -> list[dict]:
    worlds = []
    for p in sorted(d.glob(f"{prefix}_*.json")):
        j = json.loads(p.read_text())
        worlds += [w | {"file_complete": j.get("complete", False)} for w in j.get("worlds", [])]
    return worlds


def site_boot(by_site: dict[str, list[float]]) -> tuple[float, float, float]:
    """Mean over worlds and a percentile interval that resamples SITES (each site brings all its dates)."""
    sites = sorted(by_site)
    vals = [np.asarray(by_site[s], dtype=np.float64) for s in sites]
    mean = float(np.concatenate(vals).mean())
    if len(sites) < 3:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(sites), size=(BOOT_N, len(sites)))
    sums = np.array([v.sum() for v in vals])
    cnts = np.array([len(v) for v in vals])
    stat = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return mean, float(lo), float(hi)


def summarise(worlds: list[dict], methods: list[str]) -> dict:
    out = {"worlds": len(worlds), "sites": sorted({w["site"] for w in worlds}), "methods": {}}
    for m in methods:
        row = {}
        for k in KEYS:
            by = defaultdict(list)
            for w in worlds:
                by[w["site"]].append(w["methods"][m][k])
            mean, lo, hi = site_boot(by)
            row[k] = {"mean": mean, "ci95_site_bootstrap": [lo, hi]}
        row["episodes"] = int(sum(w["methods"][m]["n"] for w in worlds))
        row["counts"] = {k: int(sum(w["methods"][m]["counts"][k] for w in worlds)) for k in ("reached_ice", "returned", "died")}
        out["methods"][m] = row
    return out


def paired(worlds: list[dict], a: str, b: str, key: str) -> dict:
    by = defaultdict(list)
    for w in worlds:
        by[w["site"]].append(w["methods"][a][key] - w["methods"][b][key])
    mean, lo, hi = site_boot(by)
    per_site = {s: float(np.mean(v)) for s, v in sorted(by.items())}
    return {"a": a, "b": b, "metric": key, "mean_difference": mean, "ci95_site_bootstrap": [lo, hi], "per_site": per_site,
            "sites_where_a_is_higher": int(sum(v > 0 for v in per_site.values())), "sites": len(per_site)}


def unchanged_block(worlds: list[dict], with_ci: bool) -> dict:
    if not worlds:
        return {"worlds": 0}
    s = summarise(worlds, ["v3", *BASE, "safe_harbour_only"])
    ms = s["methods"]
    best = max(BASE, key=lambda m: ms[m]["returned"]["mean"])
    d_ret = paired(worlds, "v3", best, "returned")
    low_died = min(ms[m]["died"]["mean"] for m in BASE)
    v3 = ms["v3"]
    lo = d_ret["ci95_site_bootstrap"][0]
    died_ok = v3["died"]["mean"] <= low_died
    if with_ci:
        verdict = "PASS" if (d_ret["mean_difference"] > 0 and lo > 0 and died_ok) else (
            "PARTIAL" if (d_ret["mean_difference"] >= 0 and all(v3["died"]["mean"] < ms[m]["died"]["mean"] for m in BASE)) else "FAIL")
    else:
        verdict = "MET (point values only, 2 sites, no interval)" if (d_ret["mean_difference"] > 0 and died_ok) else "NOT MET"
    deaths = {}
    for m in BASE:
        tot = sum(w["deaths"][m].get("died", 0) for w in worlds)
        deaths[m] = {"died": int(tot)}
        if tot:
            for k in ("share_before_reaching_ice", "share_on_return_leg", "share_in_too_little_light_to_drive", "share_in_full_dark", "share_where_no_v3_plan_could_return"):
                deaths[m][k] = float(sum(w["deaths"][m].get(k, 0.0) * w["deaths"][m].get("died", 0) for w in worlds) / tot)
            deaths[m]["median_distance_from_lander_m_mean_over_worlds"] = float(np.mean([w["deaths"][m]["median_distance_from_lander_m"] for w in worlds if w["deaths"][m].get("died", 0)]))
    n_all = sum(w["n"] for w in worlds)
    extras = {"kept_at_lander": float(np.mean([w["methods"]["v3"]["kept_at_lander"] for w in worlds])),
              "prediction_matched_min": float(min(w["methods"]["v3"]["prediction_matched"] for w in worlds)),
              "missions_a_plan_could_return": float(np.mean([w["methods"]["v3"]["missions_a_plan_could_return"] for w in worlds])),
              "any_baseline_returned": float(np.mean([w["any_baseline_returned"] for w in worlds])),
              "returned_left_at_once": int(sum(w["methods"]["v3"]["returned_left_at_once"] for w in worlds)),
              "returned_waited_first": int(sum(w["methods"]["v3"]["returned_waited_first"] for w in worlds)),
              "baseline_returned_but_v3_did_not": {m: int(sum(w["methods"][m]["returned_but_v3_did_not"] for w in worlds)) for m in BASE},
              "worlds_where_v3_returned_below_a_baseline": [w["world"] for w in worlds if any(w["methods"]["v3"]["returned"] < w["methods"][m]["returned"] for m in BASE)],
              "worlds_where_v3_died_above_a_baseline": [w["world"] for w in worlds if any(w["methods"]["v3"]["died"] > w["methods"][m]["died"] for m in BASE)],
              "worlds_without_psr_edge_targets": [w["world"] for w in worlds if not w["psr_edge_targets"]],
              "hazard_sources": sorted({w["hazard_source"] for w in worlds}), "episodes_per_method": n_all,
              "all_files_complete": all(w["file_complete"] for w in worlds)}
    return {"summary": s, "best_baseline_by_returned": best, "paired_returned_v3_minus_best": d_ret, "paired_died_v3_minus_best": paired(worlds, "v3", best, "died"),
            "lowest_baseline_died": low_died, "verdict": verdict, "deaths_under_baselines": deaths, "v3_extras": extras,
            "per_world": {w["world"]: {m: {k: w["methods"][m][k] for k in KEYS} for m in w["methods"]} for w in worlds}}


def changed_block(worlds: list[dict]) -> dict:
    out = {}
    for change in sorted({w["change_name"] for w in worlds}):
        ws = [w for w in worlds if w["change_name"] == change]
        methods = list(ws[0]["methods"])
        s = summarise(ws, methods)
        ms = s["methods"]
        stale = [m for m in methods if m.endswith("(stale belief)")]
        told = [m for m in methods if m.endswith("(told)")]
        bs, bt = max(stale, key=lambda m: ms[m]["returned"]["mean"]), max(told, key=lambda m: ms[m]["returned"]["mean"])
        v = ms["v3"]
        out[change] = {"summary": s, "what": ws[0]["change"], "best_stale_baseline": bs, "best_told_baseline": bt,
                       "S2": "MET" if (v["returned"]["mean"] >= ms[bs]["returned"]["mean"] and v["died"]["mean"] <= ms[bs]["died"]["mean"]) else "NOT MET",
                       "S2b": "MET" if v["returned"]["mean"] >= ms[bt]["returned"]["mean"] else "NOT MET",
                       "v3_replans_mean": float(np.mean([w["methods"]["v3"]["replans"] for w in ws])),
                       "per_world": {w["world"]: {m: {k: w["methods"][m][k] for k in KEYS} for m in methods} for w in ws}}
    return out


def dream_block(d: Path) -> dict:
    sits = [json.loads(p.read_text()) for p in sorted(d.glob("D_[0-9]*.json"))]
    sched = json.loads((d / "D_schedule.json").read_text()) if (d / "D_schedule.json").exists() else None
    rows = []
    for s in sits:
        name = s["change"]["name"]
        rows.append({"index": s["index"], "site": s["site"], "world": s["world"], "change": name, "finished": s.get("finished", False), "capped": s.get("capped"),
                     "rounds_run": len(s["rounds"]), "accepted": s.get("accepted"), "label": s.get("label", "NO CHANGE (unfinished)"),
                     "before": s.get("before"), "after": s.get("after"), "gain_mean_return": s.get("score_gain_mean_return"), "gain_se": s.get("score_gain_se"),
                     "final_brain": s.get("final_brain"), "minutes": s.get("minutes"),
                     "rounds": [{"round": r["round"], "sent_to_gate": r.get("sent_to_gate"), "gate": r.get("gate"),
                                 "proposals": r["proposals"], "imagined_current": r["imagined_current"]} for r in s["rounds"]]})
    blocked = [r for r in rows if r["change"] == "blocked_route"]
    none = [r for r in rows if r["change"] == "none"]
    fin_b = [r for r in blocked if r["finished"]]
    imp = sum(r["label"] == "IMPROVED" for r in blocked)
    worse_b = sum(r["label"] == "WORSE" for r in blocked)
    s3 = "NOT EVALUABLE (fewer than 4 blocked_route situations finished)" if len(fin_b) < 4 else ("MET" if (imp > 3 and worse_b == 0) else "NOT MET")
    fin_n = [r for r in none if r["finished"]]
    s4 = "NOT EVALUABLE (no unchanged situation finished)" if not fin_n else ("NOT MET" if any(r["label"] == "WORSE" for r in none) else "MET")
    return {"situations": rows, "schedule": sched, "S3_changed_world_self_learning": s3, "S3_detail": {"improved": imp, "worse": worse_b, "finished": len(fin_b), "of": 6},
            "S4_detail": {"finished": len(fin_n), "worse": sum(r["label"] == "WORSE" for r in none), "improved": sum(r["label"] == "IMPROVED" for r in none)},
            "S4_unchanged_world_no_harm": s4}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir).expanduser()
    rep = {"primary_fresh_heldout_unchanged": unchanged_block(load(d, "A"), with_ci=True),
           "previously_seen_heldout_unchanged": unchanged_block(load(d, "B"), with_ci=False),
           "changed_worlds_before_learning": changed_block(load(d, "C")),
           "self_learning": dream_block(d),
           "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "unit": "site (each drawn site brings all its dates); with 6 sites the interval is approximate"}}
    if a.val:
        v = unchanged_block(load(Path(a.val).expanduser(), "val"), with_ci=True)
        v["note"] = "VALIDATION sites (go / no-go check of the construction); not a score"
        rep["validation_go_no_go"] = v
    out = Path(a.out).expanduser()
    (out / "eval.json").write_text(json.dumps(rep, indent=1))

    def tab(block, methods):
        ms = block["summary"]["methods"]
        lines = ["| method | reached ice | returned (primary) | died | mean return | episodes |", "|---|---|---|---|---|---|"]
        for m in methods:
            if m in ms:
                r = ms[m]
                f = lambda k: (f"{r[k]['mean']:.3f} [{r[k]['ci95_site_bootstrap'][0]:.3f}, {r[k]['ci95_site_bootstrap'][1]:.3f}]"
                               if not np.isnan(r[k]['ci95_site_bootstrap'][0]) else f"{r[k]['mean']:.3f}")
                lines.append(f"| {m} | {f('reached_ice')} | {f('returned')} | {f('died')} | {r['mean_return']['mean']:+.1f} | {r['episodes']} |")
        return "\n".join(lines)

    txt = []
    for title, key in (("A. Fresh held-out sites, unchanged worlds (PRIMARY)", "primary_fresh_heldout_unchanged"),
                       ("B. Site23 + Site42, unchanged worlds (previously seen held-out)", "previously_seen_heldout_unchanged"),
                       ("Validation sites (go / no-go, not a score)", "validation_go_no_go")):
        b = rep.get(key)
        if b and b.get("worlds", 1):
            word = "construction check, same rule as the primary verdict (not a verdict, not a score)" if key == "validation_go_no_go" else "verdict"
            txt += [f"## {title}", "", tab(b, ["v3", *BASE, "safe_harbour_only"]), "", f"{word}: {b['verdict']}; best baseline by returned: {b['best_baseline_by_returned']}; "
                    f"paired returned difference v3 - best: {b['paired_returned_v3_minus_best']['mean_difference']:+.3f} {b['paired_returned_v3_minus_best']['ci95_site_bootstrap']}; "
                    f"v3 higher in {b['paired_returned_v3_minus_best']['sites_where_a_is_higher']} of {b['paired_returned_v3_minus_best']['sites']} sites", ""]
    for change, b in rep["changed_worlds_before_learning"].items():
        txt += [f"## C. Changed world before learning: {change}", "", tab(b, list(b["summary"]["methods"])), "", f"S2 {b['S2']} (best stale: {b['best_stale_baseline']}); S2b {b['S2b']} (best told: {b['best_told_baseline']})", ""]
    txt += ["## D. Self-learning", "", "| # | site | change | finished | rounds | accepted | mean return before -> after | returned before -> after | died before -> after | gain (SE) | label |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["self_learning"]["situations"]:
        bf, af = r["before"] or {}, r["after"] or {}
        g = f"{r['gain_mean_return']:+.2f} ({r['gain_se']:.2f})" if r["gain_mean_return"] is not None else "n/a"
        txt.append(f"| {r['index']} | {r['site']} | {r['change']} | {r['finished']} | {r['rounds_run']} | {r['accepted']} | {bf.get('mean_return', float('nan')):+.2f} -> {af.get('mean_return', float('nan')):+.2f} | "
                   f"{bf.get('returned', float('nan')):.3f} -> {af.get('returned', float('nan')):.3f} | {bf.get('died', float('nan')):.3f} -> {af.get('died', float('nan')):.3f} | {g} | {r['label']} |")
    txt += ["", f"S3 {rep['self_learning']['S3_changed_world_self_learning']} {rep['self_learning']['S3_detail']}; S4 {rep['self_learning']['S4_unchanged_world_no_harm']}"]
    (out / "tables.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
