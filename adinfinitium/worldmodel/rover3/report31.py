"""Rover v3.1: collect the stage files of the final evaluation into eval.json and tables.md and apply the pre-registered verdict rules
(ROVER_V31_PLAN.md section 6 and Amendment 1). Reads only.

    python -m worldmodel.rover3.report31 --dir ~/adinfinitium-rover31/runs/final --val ~/adinfinitium-rover31/runs/val --out ~/adinfinitium-rover31/runs/final
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = ("naive", "cautious", "cautious_wait_light")
KEYS = ("returned", "died", "strict_returned", "strict_lost", "reached_ice", "mean_return")
CKEYS = ("returned", "died", "strict_returned", "strict_lost", "reached_ice")
UNTOUCHED = ("LM2", "LM3", "LM4", "LM5")
FRESH6 = ("DM1", "LM7", "NPA", "NPB", "NPC", "NPD")
OLD2 = ("Site23", "Site42")
BOOT_N, BOOT_SEED = 10000, 0
CHANGE_LABEL = {"blocked_route": "blocked route", "dusty_panels": "dusty panels", "late_light": "light 12 h late", "early_light": "light 8 h early",
                "weak_battery": "weak battery", "none": "unchanged"}


def load(d: Path, prefix: str) -> list[dict]:
    worlds = []
    for p in sorted(d.glob(f"{prefix}_*.json")):
        j = json.loads(p.read_text())
        worlds += [w | {"file_complete": j.get("complete", False)} for w in j.get("worlds", [])]
    return worlds


def site_boot(by_site: dict) -> tuple[float, float, float]:
    """Mean over worlds and a percentile interval that resamples SITES (each site brings all its worlds)."""
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
    out = {}
    for m in methods:
        row = {}
        for k in KEYS:
            by = defaultdict(list)
            for w in worlds:
                by[w["site"]].append(w["methods"][m][k])
            mean, lo, hi = site_boot(by)
            row[k] = {"mean": mean, "ci95_site_bootstrap": [lo, hi]}
        row["episodes"] = int(sum(w["methods"][m]["n"] for w in worlds))
        row["counts"] = {k: int(sum(w["methods"][m]["counts"][k] for w in worlds)) for k in CKEYS}
        out[m] = row
    return out


def paired(worlds: list[dict], a: str, b: str, key: str) -> dict:
    by = defaultdict(list)
    for w in worlds:
        by[w["site"]].append(w["methods"][a][key] - w["methods"][b][key])
    mean, lo, hi = site_boot(by)
    per_site = {s: float(np.mean(v)) for s, v in sorted(by.items())}
    return {"a": a, "b": b, "metric": key, "mean_difference": mean, "ci95_site_bootstrap": [lo, hi], "per_site": per_site,
            "sites_where_a_is_higher": int(sum(v > 1e-12 for v in per_site.values())), "sites_where_a_is_lower": int(sum(v < -1e-12 for v in per_site.values())),
            "sites": len(per_site)}


def unchanged_block(worlds: list[dict], name: str) -> dict:
    if not worlds:
        return {"name": name, "worlds": 0}
    methods = ["v31", "v3", *BASE, "safe_harbour_only"]
    ms = summarise(worlds, methods)
    d_lost, d_sret = paired(worlds, "v31", "v3", "strict_lost"), paired(worlds, "v31", "v3", "strict_returned")
    d_ret, d_died = paired(worlds, "v31", "v3", "returned"), paired(worlds, "v31", "v3", "died")
    best = max(BASE, key=lambda m: ms[m]["strict_returned"]["mean"])
    nsite = d_lost["sites"]
    hi = d_lost["ci95_site_bootstrap"][1]
    need_sites = 3 if nsite >= 4 else nsite
    fell = d_lost["mean_difference"] < 0
    cond = {"strict_lost_falls_2_points": d_lost["mean_difference"] <= -0.02, "falls_at_enough_sites": d_lost["sites_where_a_is_lower"] >= need_sites,
            "interval_below_0": (hi < 0) if nsite >= 3 else None, "strict_returned_not_lower": d_sret["mean_difference"] >= 0,
            "lenient_died_not_higher": ms["v31"]["died"]["mean"] <= ms["v3"]["died"]["mean"] + 1e-12}
    hard = [v for v in cond.values() if v is not None]
    if all(hard):
        verdict = "PASS"
    elif fell and cond["strict_returned_not_lower"] and cond["lenient_died_not_higher"]:
        verdict = "PASS-SMALL (no material gain)"
    else:
        verdict = "FAIL"
    v31w = [w["methods"]["v31"] for w in worlds]
    split = {k: int(sum(m["vs_v3"][k] for m in v31w)) for k in v31w[0]["vs_v3"]}
    alive = int(sum(m["alive_no_trip"] for m in v31w))
    bat = [(m["alive_no_trip_mean_end_battery_wh"], m["alive_no_trip"]) for m in v31w if m["alive_no_trip"]]
    return {"name": name, "worlds": len(worlds), "sites": sorted({w["site"] for w in worlds}), "episodes_per_method": ms["v31"]["episodes"], "methods": ms,
            "v31_minus_v3": {"strict_lost": d_lost, "strict_returned": d_sret, "returned": d_ret, "died": d_died},
            "v31_minus_best_planner_strict_returned": paired(worlds, "v31", best, "strict_returned") | {"best_planner": best},
            "v3_minus_best_planner_strict_returned": paired(worlds, "v3", best, "strict_returned") | {"best_planner": best},
            "conditions": cond, "verdict": verdict, "split_of_change_vs_v3": split,
            "v31_alive_no_trip": alive, "v31_alive_no_trip_mean_end_battery_wh": (sum(b * n for b, n in bat) / sum(n for _, n in bat)) if bat else None,
            "v31_first_rank_counts_1_2_3": [int(sum(m["first_rank_counts"][i] for m in v31w)) for i in range(3)],
            "v31_shelter_only_first_choice": int(sum(m["first_choice_shelter_only"] for m in v31w)),
            "v31_shelter_then_go_first_choice": int(sum(m["first_choice_shelter_then_go"] for m in v31w)),
            "v31_prediction_matched_strict_min": min(m["prediction_matched_strict"] for m in v31w),
            "v3_prediction_matched_min": min(w["methods"]["v3"]["prediction_matched"] for w in worlds),
            "all_files_complete": all(w["file_complete"] for w in worlds),
            "per_world": [{"world": w["world"], "site": w["site"], "date": w["date"]} | {m: {k: w["methods"][m][k] for k in KEYS} | {"counts": w["methods"][m]["counts"]}
                                                                                         for m in methods} for w in worlds]}


def changed_block(worlds: list[dict]) -> dict:
    out = {}
    for ch in sorted({w["change_name"] for w in worlds}, key=list(CHANGE_LABEL).index):
        ws = [w for w in worlds if w["change_name"] == ch]
        methods = list(ws[0]["methods"])
        ms = summarise(ws, methods)
        stale = [m for m in methods if m.endswith("(stale belief)")]
        told = [m for m in methods if m.endswith("(told)")]
        bs = max(stale, key=lambda m: ms[m]["strict_returned"]["mean"])
        bt = max(told, key=lambda m: ms[m]["strict_returned"]["mean"])
        d_lost, d_sret = paired(ws, "v31", "v3", "strict_lost"), paired(ws, "v31", "v3", "strict_returned")
        out[ch] = {"worlds": len(ws), "sites": sorted({w["site"] for w in ws}), "what": ws[0]["change"]["what"], "methods": ms, "best_not_told_planner": bs,
                   "best_told_planner": bt, "v31_minus_v3": {"strict_lost": d_lost, "strict_returned": d_sret, "returned": paired(ws, "v31", "v3", "returned"),
                                                              "died": paired(ws, "v31", "v3", "died")},
                   "S3": "MET" if (d_lost["mean_difference"] <= 1e-12 and d_sret["mean_difference"] >= -1e-12) else "NOT MET",
                   "per_world": [{"world": w["world"], "site": w["site"]} | {m: {k: w["methods"][m][k] for k in KEYS} | {"counts": w["methods"][m]["counts"]} for m in methods}
                                 for w in ws]}
    return out


def dream_block(d: Path) -> dict:
    sits = [json.loads(p.read_text()) for p in sorted(d.glob("D_[0-9][0-9]_*.json"))]
    sched = json.loads((d / "D_schedule.json").read_text()) if (d / "D_schedule.json").exists() else None
    rows = []
    for s in sits:
        ch = s["change"]["name"]
        fin = bool(s.get("finished"))
        acc = [r for r in s["rounds"] if r.get("gate") and r["gate"]["accept"]]
        rows.append({"index": s["index"], "site": s["site"], "change": ch, "finished": fin, "rounds_run": len(s["rounds"]), "accepted": s.get("accepted"),
                     "label": s.get("label") if fin else "NOT FINISHED (counts as NO CHANGE)", "accepted_changes": [r["sent_to_gate"] for r in acc],
                     "accepted_rounds": [r["round"] for r in acc], "v3_gate_would_accept": [r["gate"]["v3_gate_would_accept"] for r in acc],
                     "sent_to_gate": [r.get("sent_to_gate") for r in s["rounds"]], "gate_accept": [(r["gate"]["accept"] if r.get("gate") else None) for r in s["rounds"]],
                     "measured_round1": s["rounds"][0]["measured"] if s["rounds"] else None,
                     "v3_no_learning": s.get("v3_no_learning_same_starts"), "before": s.get("before"), "after": s.get("after"),
                     "gain_strict_score": s.get("score_gain_strict"), "gain_se": s.get("score_gain_se"), "minutes": s.get("minutes")})
    by = defaultdict(list)
    for r in rows:
        by[r["change"]].append(r)
    verdicts = {}
    for ch, rs in by.items():
        lab = [r["label"] if r["finished"] else "NO CHANGE" for r in rs]
        imp, worse, fin = lab.count("IMPROVED"), lab.count("WORSE"), sum(r["finished"] for r in rs)
        if fin < 3:
            v = "NOT EVALUABLE"
        elif ch == "none":
            v = "MET" if worse == 0 else "NOT MET"
        else:
            v = "MET" if (imp > 4 / 2 and worse == 0) else "NOT MET"
        verdicts[ch] = {"situations": len(rs), "finished": fin, "improved": imp, "worse": worse, "no_change": len(rs) - imp - worse, "S4": v,
                        "rounds_run": sum(r["rounds_run"] for r in rs), "rounds_accepted": sum(r["accepted"] or 0 for r in rs),
                        "accepted_in_round_2": sum(1 for r in rs for x in r["accepted_rounds"] if x >= 2)}
    return {"situations": rows, "by_change": verdicts, "schedule": sched}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(row: dict, k: str) -> str:
    lo, hi = row[k]["ci95_site_bootstrap"]
    return pct(row[k]["mean"]) + ("" if np.isnan(lo) else f" [{100 * lo:.1f}, {100 * hi:.1f}]")


def tables(ev: dict) -> str:
    names = {"v31": "**v3.1**", "v3": "v3 (frozen)", "naive": "planner: naive", "cautious": "planner: cautious", "cautious_wait_light": "planner: cautious + wait for light",
             "safe_harbour_only": "reference: safe harbour only"}
    L = []
    for key in ("A_untouched", "B_fresh6", "B_site23_42"):
        b = ev.get(key)
        if not b or not b.get("worlds"):
            continue
        L += [f"### {b['name']}: {b['worlds']} worlds, {b['episodes_per_method']} episodes per method", "",
              "| method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient | reached target |", "|---|---|---|---|---|---|"]
        for m, r in b["methods"].items():
            L.append(f"| {names[m]} | {ci(r, 'strict_returned')} | {ci(r, 'strict_lost')} | {ci(r, 'returned')} | {ci(r, 'died')} | {pct(r['reached_ice']['mean'])} |")
        d = b["v31_minus_v3"]
        for k, lab in (("strict_lost", "strict lost"), ("strict_returned", "strict returned"), ("returned", "lenient returned")):
            x = d[k]
            lo, hi = x["ci95_site_bootstrap"]
            L.append("")
            L.append(f"- v3.1 minus v3, {lab}: {100 * x['mean_difference']:+.1f} points" + ("" if np.isnan(lo) else f" [{100 * lo:+.1f}, {100 * hi:+.1f}]")
                     + f"; per site: " + ", ".join(f"{s} {100 * v:+.1f}" for s, v in x["per_site"].items()))
        sp = b["split_of_change_vs_v3"]
        bp = b["v31_minus_best_planner_strict_returned"]
        L += ["", f"- Rovers v3 lost (strict) that v3.1 did not: {sp['strict_lost_by_v3_not_by_v31']} = {sp['of_those_now_strict_returned']} now strict returns + "
              f"{sp['of_those_now_alive_no_trip']} alive without a completed trip. Lost by v3.1 but not by v3: {sp['strict_lost_by_v31_not_by_v3']}. "
              f"Strict returns of v3 that v3.1 did not make: {sp['strict_returned_by_v3_not_by_v31']}. Lenient returns of v3 that v3.1 did not make: {sp['lenient_returned_by_v3_not_by_v31']}.",
              f"- v3.1 alive at hour 300 without a completed trip: {b['v31_alive_no_trip']}"
              + (f" (mean battery left {b['v31_alive_no_trip_mean_end_battery_wh']:.0f} Wh)" if b["v31_alive_no_trip_mean_end_battery_wh"] is not None else "")
              + f". First plan: rank 1 / 2 / 3 = {b['v31_first_rank_counts_1_2_3']}; shelter only {b['v31_shelter_only_first_choice']}, shelter then go {b['v31_shelter_then_go_first_choice']}.",
              f"- v3.1 minus best planner by strict returned ({bp['best_planner']}): {100 * bp['mean_difference']:+.1f} points, higher at {bp['sites_where_a_is_higher']} of {bp['sites']} sites.",
              f"- Prediction = outcome (strict): v3.1 at least {pct(b['v31_prediction_matched_strict_min'])} in every world. Rule result: **{b['verdict']}** {b['conditions']}", ""]
    if ev.get("C_changed"):
        L += ["### Changed worlds before any learning (untouched sites, 2027-06-01, 256 rovers per world; the rover is not told)", "",
              "| change | method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient |", "|---|---|---|---|---|---|"]
        for ch, b in ev["C_changed"].items():
            for m in ("v31", "v3", b["best_not_told_planner"], b["best_told_planner"]):
                r = b["methods"][m]
                lab = names.get(m, "best planner " + ("not told: " if "stale" in m else "told: ") + m.split(" (")[0])
                L.append(f"| {CHANGE_LABEL[ch]} | {lab} | {pct(r['strict_returned']['mean'])} | {pct(r['strict_lost']['mean'])} | {pct(r['returned']['mean'])} | {pct(r['died']['mean'])} |")
        L.append("")
        for ch, b in ev["C_changed"].items():
            d = b["v31_minus_v3"]
            L.append(f"- {CHANGE_LABEL[ch]}: v3.1 minus v3 strict lost {100 * d['strict_lost']['mean_difference']:+.1f}, strict returned {100 * d['strict_returned']['mean_difference']:+.1f}, "
                     f"lenient returned {100 * d['returned']['mean_difference']:+.1f}, lenient lost {100 * d['died']['mean_difference']:+.1f} points; S3 {b['S3']}")
        L.append("")
    if ev.get("D_dream"):
        L += ["### Self-learning (v3.1 layer on the v3.1 brain; untouched sites, 2027-06-01; 2 rounds; 256 score starts)", "",
              "| site | world | accepted (rounds) | accepted change | v3 no learning: strict ret / lost | v3.1 before -> after: strict returned | strict lost | lenient returned | lenient lost | gain strict score (SE) | label |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ev["D_dream"]["situations"]:
            if not r["before"] or not r["after"]:
                L.append(f"| {r['site']} | {CHANGE_LABEL[r['change']]} | - | - | - | - | - | - | - | - | {r['label']} |")
                continue
            b, a, v = r["before"], r["after"], r["v3_no_learning"]
            L.append(f"| {r['site']} | {CHANGE_LABEL[r['change']]} | {r['accepted']} ({r['accepted_rounds']}) | {'; '.join(r['accepted_changes']) or 'nothing'} | "
                     f"{pct(v['strict_returned'])} / {pct(v['strict_lost'])} | {pct(b['strict_returned'])} -> {pct(a['strict_returned'])} | {pct(b['strict_lost'])} -> {pct(a['strict_lost'])} | "
                     f"{pct(b['returned'])} -> {pct(a['returned'])} | {pct(b['died'])} -> {pct(a['died'])} | {r['gain_strict_score']:+.1f} ({r['gain_se']:.1f}) | {r['label']} |")
        L.append("")
        for ch, v in ev["D_dream"]["by_change"].items():
            L.append(f"- {CHANGE_LABEL[ch]}: {v['improved']} IMPROVED, {v['no_change']} NO CHANGE, {v['worse']} WORSE of {v['situations']} ({v['finished']} finished); "
                     f"rounds accepted {v['rounds_accepted']} of {v['rounds_run']} (in round 2: {v['accepted_in_round_2']}); S4 {v['S4']}")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    A, B = load(d, "A"), load(d, "B")
    ev = {"A_untouched": unchanged_block([w for w in A if w["site"] in UNTOUCHED], "A. Untouched sites LM2, LM3, LM4, LM5 (PRIMARY)"),
          "B_fresh6": unchanged_block([w for w in B if w["site"] in FRESH6], "B1. Six earlier fresh sites (previously seen)"),
          "B_site23_42": unchanged_block([w for w in B if w["site"] in OLD2], "B2. Site23 + Site42 (previously seen)"),
          "C_changed": changed_block(load(d, "C")), "D_dream": dream_block(d)}
    b1 = ev["B_fresh6"]
    if b1.get("worlds"):
        c = b1["methods"]["v3"]["counts"]
        ev["v3_row_reproduces_frozen_counts_on_fresh6"] = {"returned": c["returned"], "died": c["died"], "expected": [7327, 0], "strict_returned": c["strict_returned"],
                                                           "strict_lost": c["strict_lost"], "expected_strict": [5163, 6965],
                                                           "ok": c["returned"] == 7327 and c["died"] == 0 and c["strict_returned"] == 5163 and c["strict_lost"] == 6965}
    if a.val:
        V = load(Path(a.val), "V")
        if V:
            ev["validation"] = unchanged_block(V, "Validation (6 sites; go / no-go, not a score)")
            ev["validation"]["G3_every_world"] = all(w["methods"]["v31"]["counts"]["strict_returned"] >= w["methods"]["v3"]["counts"]["strict_returned"]
                                                     and w["methods"]["v31"]["counts"]["strict_lost"] <= w["methods"]["v3"]["counts"]["strict_lost"]
                                                     and w["methods"]["v31"]["counts"]["died"] <= w["methods"]["v3"]["counts"]["died"] for w in V)
    out = Path(a.out)
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    (out / "tables.md").write_text(tables(ev))
    print(tables(ev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
