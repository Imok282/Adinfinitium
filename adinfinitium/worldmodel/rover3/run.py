"""Rover v3 evaluation and self-learning (Dream-RSI layer). See ROVER_V3_PLAN.md (pre-registered) for what each stage means.

    python -m worldmodel.rover3.run eval    --sites DM1 --n 512 --seed 1234 --out runs/_rover3/A_dm1.json       # unchanged worlds: v3 + baselines, same starts
    python -m worldmodel.rover3.run changed --sites DM1 --date 2027-06-01 --n 256 --seed 1234 --out ...         # changed worlds, belief not told
    python -m worldmodel.rover3.run dream   --index 0 --site DM1 --change blocked_route --out ...               # one self-learning situation
    python -m worldmodel.rover3.run dream-all --out-dir ... --cap-minutes 40                                    # the pre-registered situation list, 3 at a time

Dream-RSI here (key-free, no LLM, no training data): record own episodes -> correct the belief and propose parameter changes -> practise every proposal in
imagination (episodes inside the corrected belief model) -> the best imagined proposal must pass a real-episode gate against the current brain -> keep or discard.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from contracts.mission import load_mission
from worldmodel.dream.changes import apply_change
from worldmodel.hazard.lola import regions_for_site
from worldmodel.rover.world import PSR_EDGE_ICE_SOURCE, build_world
from worldmodel.rover3.brain import (BASELINES, KIND_SAFE, KIND_SHELTER, KIND_STRATEGY, NEVER, SHIFT_GRID, Brain, FastEnv, Theta, act, candidate_table, new_exec_state,
                                      run_baseline, run_episode, strict_score)

DATES = ("2027-03-01", "2027-06-01", "2027-09-01", "2027-12-01")
FRESH = ("DM1", "LM7", "NPA", "NPB", "NPC", "NPD")
DREAM_DATE = "2027-06-01"
SITUATIONS = [(s, c) for s in FRESH for c in ("blocked_route", "none")] + [(s, "dusty_panels") for s in FRESH]     # the pre-registered order
DRIVE_SUN = 0.24                                                                                                   # panel 50 W x 0.24 = drive load 12 W


def reseed(env: FastEnv, n: int, seed: int) -> FastEnv:
    """Same starts as `RoverEnv(world, n, device, cfg, seed=seed)` followed by `env.reset()` (what evaluate_policy does), without rebuilding the route fields."""
    env.n = n
    env.gen = torch.Generator(device="cpu").manual_seed(seed)
    env.reset()
    env.reset()
    return env


def starts(env: FastEnv) -> tuple:
    return (env.pos.clone(), env.t.clone(), env.target_idx.clone())


def same_starts(a: tuple, b: tuple) -> bool:
    return all(bool(torch.equal(x, y)) for x, y in zip(a, b))


def rates(r: dict) -> dict:
    n = r["n"]
    out = {"n": n, "reached_ice": r["reached_ice"] / n, "returned": r["returned"] / n, "died": r["died"] / n, "mean_return": r["mean_return"],
           "counts": {"reached_ice": r["reached_ice"], "returned": r["returned"], "died": r["died"]}}
    if "strict_lost" in r:
        out |= {"strict_returned": r["strict_returned"] / n, "strict_lost": r["strict_lost"] / n}
        out["counts"] |= {"strict_returned": r["strict_returned"], "strict_lost": r["strict_lost"]}
    return out


def run_safe_only(real: FastEnv, belief: FastEnv) -> dict:
    """Reference: no planning at all, every rover drives to the lander and stays (the executor's safe harbour)."""
    n, dev = real.n, real.device
    from worldmodel.rover3.brain import get_state, set_state
    X = new_exec_state(n, dev)
    P = {"depart": torch.zeros(n, dtype=torch.long, device=dev), "lo": torch.full((n,), -1.0, device=dev), "hi": torch.ones(n, device=dev),
         "kind": torch.full((n,), KIND_SAFE, dtype=torch.long, device=dev)}
    ice = torch.zeros(n, dtype=torch.bool, device=dev)
    ret, dead, empty = ice.clone(), ice.clone(), ice.clone()
    for _ in range(real.rs.max_steps):
        set_state(belief, get_state(real))
        alive0 = ~real.done
        _, _, done, info = real.step(act(belief, P, X, real.slope))
        ice |= info["reached_ice"]; ret |= info["returned"]; dead |= info["dead"]
        empty |= alive0 & (real.battery <= 0)
        if bool(done.all()):
            break
    return {"n": n, "reached_ice": int(ice.sum()), "returned": int(ret.sum()), "died": int(dead.sum()), "mean_return": float(real.ep_return.mean()),
            "strict_lost": int(empty.sum()), "strict_returned": int((ret & ~empty).sum())}


def death_summary(b: dict, feasible: torch.Tensor) -> dict:
    d = b["dead_mask"]
    k = int(d.sum())
    if k == 0:
        return {"died": 0}
    return {"died": k, "share_before_reaching_ice": float((b["death_phase"][d] == 0).float().mean()),
            "share_on_return_leg": float((b["death_phase"][d] == 1).float().mean()),
            "share_in_too_little_light_to_drive": float((b["death_sun"][d] < DRIVE_SUN).float().mean()),
            "share_in_full_dark": float((b["death_sun"][d] < 0.02).float().mean()),
            "share_where_no_v3_plan_could_return": float((~feasible[d]).float().mean()),
            "median_distance_from_lander_m": float(b["death_dist_m"][d].median()), "median_hour_of_death": float(b["death_step"][d].float().median())}


def eval_world(region: str, date: str, cfg, device, n: int, seed: int) -> dict:
    w = build_world(region, date, cfg, device=device)
    real = FastEnv(w, n, device, cfg, seed=seed)
    brain = Brain(w, cfg, Theta(), env=copy.copy(real))                    # unchanged world: belief = the world itself (shares the route fields)
    t0 = time.time()
    reseed(real, n, seed)
    st = starts(real)
    v3 = run_episode(real, brain)
    table = candidate_table(brain.theta, "cpu")
    fc = v3["first_choice"]
    waited = (table["kind"][fc] == KIND_STRATEGY) & (table["wait"][fc] > 0)
    out = {"world": f"{region}@{date}", "n": n, "seed": seed, "hazard_source": w.hazard_source, "ice_source": w.ice_source,
           "psr_edge_targets": w.ice_source == PSR_EDGE_ICE_SOURCE, "methods": {}, "deaths": {}}
    out["methods"]["v3"] = rates(v3) | {"kept_at_lander": v3["kept_at_lander"] / n, "prediction_matched": v3["prediction_matched"] / n, "replans": v3["replans"],
                                        "returned_left_at_once": int((v3["returned_mask"] & ~waited).sum()), "returned_waited_first": int((v3["returned_mask"] & waited).sum()),
                                        "missions_a_plan_could_return": float(v3["mission_feasible"].float().mean())}
    out["v3_seconds"] = time.time() - t0
    out["methods"]["v3"]["prediction_matched_strict"] = v3["prediction_matched_strict"] / n
    t1 = time.time()
    reseed(real, n, seed)
    assert same_starts(st, starts(real)), "starts differ between methods"
    v31 = run_episode(real, Brain(w, cfg, Theta(v31=True), env=copy.copy(real)))
    t31 = candidate_table(Theta(v31=True), "cpu")
    k31 = t31["kind"][v31["first_choice"]]
    alive_no_trip = ~v31["strict_lost_mask"] & ~v31["returned_mask"]
    out["methods"]["v31"] = rates(v31) | {
        "kept_at_lander": v31["kept_at_lander"] / n, "prediction_matched": v31["prediction_matched"] / n, "prediction_matched_strict": v31["prediction_matched_strict"] / n,
        "replans": v31["replans"], "first_rank_counts": [int((v31["first_rank"] == k).sum()) for k in (1, 2, 3)],
        "first_choice_shelter_only": int(((k31 == KIND_SHELTER) & (t31["wait"][v31["first_choice"]] >= NEVER)).sum()),
        "first_choice_shelter_then_go": int(((k31 == KIND_SHELTER) & (t31["wait"][v31["first_choice"]] < NEVER)).sum()),
        "alive_no_trip": int(alive_no_trip.sum()),
        "alive_no_trip_mean_end_battery_wh": float(v31["end_battery"][alive_no_trip].mean()) if bool(alive_no_trip.any()) else None,
        "alive_no_trip_median_end_dist_m": float(v31["end_dist_m"][alive_no_trip].median()) if bool(alive_no_trip.any()) else None,
        "vs_v3": {"strict_lost_by_v3_not_by_v31": int((v3["strict_lost_mask"] & ~v31["strict_lost_mask"]).sum()),
                  "of_those_now_strict_returned": int((v3["strict_lost_mask"] & v31["strict_returned_mask"]).sum()),
                  "of_those_now_alive_no_trip": int((v3["strict_lost_mask"] & ~v31["strict_lost_mask"] & ~v31["strict_returned_mask"]).sum()),
                  "strict_lost_by_v31_not_by_v3": int((v31["strict_lost_mask"] & ~v3["strict_lost_mask"]).sum()),
                  "strict_returned_by_v3_not_by_v31": int((v3["strict_returned_mask"] & ~v31["strict_returned_mask"]).sum()),
                  "lenient_returned_by_v3_not_by_v31": int((v3["returned_mask"] & ~v31["returned_mask"]).sum())}}
    out["v31_seconds"] = time.time() - t1
    union = torch.zeros(n, dtype=torch.bool)
    for name in BASELINES:
        reseed(real, n, seed)
        assert same_starts(st, starts(real)), "starts differ between methods"
        b = run_baseline(real, name)
        out["methods"][name] = rates(b)
        out["deaths"][name] = death_summary(b, v3["mission_feasible"])
        union |= b["returned_mask"]
        out["methods"][name]["returned_but_v3_did_not"] = int((b["returned_mask"] & ~v3["returned_mask"]).sum())
    reseed(real, n, seed)
    assert same_starts(st, starts(real))
    out["methods"]["safe_harbour_only"] = rates(run_safe_only(real, brain.env))
    out["any_baseline_returned"] = float(union.float().mean())
    out["seconds"] = time.time() - t0
    return out


def changed_world(region: str, date: str, change: str, cfg, device, n: int, seed: int) -> dict:
    w = build_world(region, date, cfg, device=device)
    w2, cfg2, desc = apply_change(w, cfg, change, seed=0)
    real = FastEnv(w2, n, device, cfg2, seed=seed)
    belief = FastEnv(w, 1, device, cfg, seed=0)                            # the rover is NOT told: it believes the original world and physics
    brain = Brain(w, cfg, Theta(), env=belief)
    t0 = time.time()
    out = {"world": f"{region}@{date}", "change": desc, "n": n, "seed": seed, "methods": {}}
    reseed(real, n, seed)
    st = starts(real)
    v3 = run_episode(real, brain)
    out["methods"]["v3"] = rates(v3) | {"kept_at_lander": v3["kept_at_lander"] / n, "replans": v3["replans"], "replanned_rovers": v3["replanned_rovers"]}
    reseed(real, n, seed)
    assert same_starts(st, starts(real))
    v31 = run_episode(real, Brain(w, cfg, Theta(v31=True), env=copy.copy(belief)))
    out["methods"]["v31"] = rates(v31) | {"kept_at_lander": v31["kept_at_lander"] / n, "replans": v31["replans"], "replanned_rovers": v31["replanned_rovers"]}
    for name in BASELINES:
        reseed(real, n, seed)
        assert same_starts(st, starts(real))
        out["methods"][f"{name} (stale belief)"] = rates(run_baseline(real, name, belief_env=belief))
        reseed(real, n, seed)
        out["methods"][f"{name} (told)"] = rates(run_baseline(real, name))
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------ Dream-RSI layer
def gate_decision(inc_return: np.ndarray, cand_return: np.ndarray, inc_returned: int, cand_returned: int, inc_died: int, cand_died: int, k_se: float = 2.0) -> dict:
    """Real-episode gate, paired on the same starts. ACCEPT only if the mean-return gain is positive and above k_se standard errors AND the candidate
    returned at least as many rovers AND lost no more rovers. Anything else keeps the current brain."""
    d = np.asarray(cand_return, dtype=np.float64) - np.asarray(inc_return, dtype=np.float64)
    gain = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    accept = bool(gain > 0 and gain > k_se * se and cand_returned >= inc_returned and cand_died <= inc_died)
    return {"gain": gain, "se": se, "accept": accept, "returned": [int(inc_returned), int(cand_returned)], "died": [int(inc_died), int(cand_died)]}


def proposals(cur: Brain, seen: np.ndarray, true_slope: np.ndarray, solar_factor: float | None, solar_samples: int) -> tuple[list[Brain], Brain]:
    """Candidate brains in the pre-registered order, and the corrected belief used as the dream world."""
    out = []
    new_slope = np.where(seen, true_slope, cur.world.slope).astype(np.float32)
    map_world = dataclasses.replace(cur.world, slope=new_slope) if bool((np.abs(new_slope - cur.world.slope) > 0.5).any()) else None
    solar_cfg = None
    if solar_factor is not None and solar_samples >= 50 and abs(solar_factor - 1.0) > 0.05:
        solar_cfg = cur.cfg.model_copy(update={"rover": cur.cfg.rover.model_copy(update={"solar_panel_w": cur.cfg.rover.solar_panel_w * solar_factor})})
    if map_world is not None:
        out.append(Brain(map_world, cur.cfg, cur.theta, None, f"belief map <- ground seen ({int((np.abs(new_slope - cur.world.slope) > 0.5).sum())} cells corrected)"))
    if solar_cfg is not None:
        out.append(Brain(cur.world, solar_cfg, cur.theta, None, f"belief solar output x {solar_factor:.3f} (measured from battery telemetry)"))
    if map_world is not None and solar_cfg is not None:
        out.append(Brain(map_world, solar_cfg, cur.theta, None, f"belief map <- ground seen, and solar output x {solar_factor:.3f}"))
    for dm in (-0.05, 0.05, 0.10):
        m = round(cur.theta.margin + dm, 2)
        if 0.0 <= m <= 0.30:
            out.append(cur.with_theta(f"energy margin {cur.theta.margin:.2f} -> {m:.2f}", margin=m))
    for lo, hi in ((0.20, 0.80), (0.50, 1.00)):
        if (lo, hi) != (cur.theta.charge_lo, cur.theta.charge_hi):
            out.append(cur.with_theta(f"charge thresholds -> {lo:.2f} / {hi:.2f}", charge_lo=lo, charge_hi=hi))
    dream = Brain(map_world or cur.world, solar_cfg or cur.cfg, cur.theta, None if (map_world is not None or solar_cfg is not None) else cur.env, "dream world")
    return out, dream


def brain_summary(b: Brain, base_cfg) -> dict:
    return {"note": b.note, "margin": b.theta.margin, "charge_lo": b.theta.charge_lo, "charge_hi": b.theta.charge_hi,
            "belief_solar_panel_w": b.cfg.rover.solar_panel_w, "deployed_solar_panel_w": base_cfg.rover.solar_panel_w}


def dream_situation(index: int, site: str, change: str, cfg, device, rounds: int = 4, n_gate: int = 128, n_dream: int = 128, n_score: int = 256,
                    deadline: float | None = None, out_path: Path | None = None) -> dict:
    region = regions_for_site(site)[0]
    w = build_world(region, DREAM_DATE, cfg, device=device)
    w2, cfg2, desc = apply_change(w, cfg, change, seed=0)
    real = FastEnv(w2, n_gate, device, cfg2, seed=0)
    brain0 = Brain(w, cfg, Theta(), env=FastEnv(w, 1, device, cfg, seed=0))
    cur = brain0
    t0 = time.time()
    res = {"index": index, "site": site, "world": f"{region}@{DREAM_DATE}", "change": desc, "rounds_planned": rounds, "rounds": [], "finished": False,
           "sizes": {"gate": n_gate, "dream": n_dream, "score": n_score}}

    def save():
        res["minutes"] = (time.time() - t0) / 60
        if out_path is not None:
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(out_path)

    before = run_episode(reseed(real, n_score, 9000 + index), brain0)
    res["before"] = rates(before) | {"kept_at_lander": before["kept_at_lander"] / n_score}
    save()
    rec = run_episode(reseed(real, n_gate, 7000 + index), cur, record=True)["record"]
    seen = rec["seen"].copy()
    accepted = 0
    capped = False
    for r in range(rounds):
        if deadline is not None and time.time() > deadline:
            capped = True
            break
        props, dream = proposals(cur, seen, w2.slope, rec["solar_factor"], rec["solar_samples"])
        dream_env = FastEnv(dream.world, n_dream, device, dream.cfg, seed=0) if dream.env is None else copy.copy(dream.env)
        dseed = 8000 + 100 * index + r
        imag_cur = run_episode(reseed(dream_env, n_dream, dseed), cur)["mean_return"]
        imag = []
        for p in props:
            if p.env is None:
                p.env = FastEnv(p.world, 1, device, p.cfg, seed=0)
            imag.append(run_episode(reseed(dream_env, n_dream, dseed), p)["mean_return"])
        row = {"round": r + 1, "imagined_current": imag_cur, "proposals": [{"what": p.note, "imagined": v} for p, v in zip(props, imag)]}
        better = [i for i, v in enumerate(imag) if v > imag_cur + 1e-9]
        gseed = 5000 + 100 * index + r
        inc = run_episode(reseed(real, n_gate, gseed), cur, record=True)
        if better:
            best = max(better, key=lambda i: (imag[i], -i))
            cand = props[best]
            cnd = run_episode(reseed(real, n_gate, gseed), cand, record=True)
            g = gate_decision(inc["ep_return"].numpy(), cnd["ep_return"].numpy(), inc["returned"], cnd["returned"], inc["died"], cnd["died"])
            row.update(sent_to_gate=cand.note, gate=g, gate_seed=gseed, gate_current=rates(inc), gate_candidate=rates(cnd))
            seen |= cnd["record"]["seen"]
            if g["accept"]:
                cur, accepted, rec = cand, accepted + 1, cnd["record"]
            else:
                rec = inc["record"]
        else:
            row.update(sent_to_gate=None, gate=None, note="no proposal beat the current brain in imagination; current brain kept")
            rec = inc["record"]
        seen |= inc["record"]["seen"]
        row["brain_after_round"] = brain_summary(cur, cfg)
        res["rounds"].append(row)
        save()
    res["accepted"] = accepted
    res["capped"] = capped
    after = run_episode(reseed(real, n_score, 9000 + index), cur) if accepted else before
    res["after"] = rates(after) | {"kept_at_lander": after["kept_at_lander"] / n_score}
    d = (after["ep_return"] - before["ep_return"]).double().numpy()
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    gain = float(d.mean())
    res["score_gain_mean_return"] = gain
    res["score_gain_se"] = se
    res["label"] = "IMPROVED" if (accepted >= 1 and gain > 2 * se and gain > 0) else ("WORSE" if gain < -2 * se else "NO CHANGE")
    res["final_brain"] = brain_summary(cur, cfg)
    res["finished"] = not capped
    save()
    return res


# ------------------------------------------------------------------------------------------------ Dream-RSI layer, v3.1 (ROVER_V31_PLAN.md section 1.5)
CHANGE_ORDER = ("late_light", "dusty_panels", "blocked_route", "none")


def situations31(sites: list[str], early: bool = True, weak: bool = True) -> list[tuple[str, str]]:
    out = [(s, c) for s in sites for c in CHANGE_ORDER]
    if early:
        out += [(s, "early_light") for s in sites]
    if weak:
        out += [(s, "weak_battery") for s in sites]
    return out


def gate_decision31(inc: dict, cnd: dict, k_se: float = 2.0) -> dict:
    """v3.1 real-episode gate, paired on the same starts. Same form and thresholds as v3's gate; the gain is measured on the strict score. ACCEPT only if the
    mean gain is positive and above k_se standard errors AND strict returned is no lower AND strict lost is no higher AND the simulator's own (lenient) died
    count is no higher AND lenient returned is no lower. Also says what v3's original gate would have decided."""
    d = (cnd["strict_score"] - inc["strict_score"]).double().numpy()
    gain = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    accept = bool(gain > 0 and gain > k_se * se and cnd["strict_returned"] >= inc["strict_returned"] and cnd["strict_lost"] <= inc["strict_lost"]
                  and cnd["died"] <= inc["died"] and cnd["returned"] >= inc["returned"])
    old = gate_decision(inc["ep_return"].numpy(), cnd["ep_return"].numpy(), inc["returned"], cnd["returned"], inc["died"], cnd["died"], k_se)
    return {"gain": gain, "se": se, "accept": accept, "strict_returned": [inc["strict_returned"], cnd["strict_returned"]],
            "strict_lost": [inc["strict_lost"], cnd["strict_lost"]], "returned": [inc["returned"], cnd["returned"]], "died": [inc["died"], cnd["died"]],
            "v3_gate_would_accept": old["accept"], "v3_gate_gain": old["gain"], "v3_gate_se": old["se"]}


def shift_world(world, k: int):
    """Belief forecast moved by k hours: new[t] = old[t - k] (k > 0: the light comes k hours later than the old forecast said)."""
    sun = world.sun_frac
    if k > 0:
        new = np.concatenate([np.repeat(sun[:1], k, axis=0), sun[:-k]], axis=0)
    else:
        new = np.concatenate([sun[-k:], np.repeat(sun[-1:], -k, axis=0)], axis=0)
    return dataclasses.replace(world, sun_frac=new)


def derive_env(base: FastEnv, world, cfg, device) -> FastEnv:
    """Belief model for (world, cfg). The route fields depend only on slope, hazard and the slope limit: when those are unchanged the fields are shared."""
    bw = base.world
    if world.slope is bw.slope and world.hazard is bw.hazard and cfg.rover.max_slope_deg == base.cfg.rover.max_slope_deg:
        e = copy.copy(base)
        e.world, e.cfg, e.rs, e.rv = world, cfg, cfg.rover_sim, cfg.rover
        if world.sun_frac is not bw.sun_frac:
            e.sun = torch.from_numpy(world.sun_frac.astype(np.float32))[:, None].to(device)
        return e
    return FastEnv(world, 1, device, cfg, seed=0)


def fit_shift(rec: dict) -> tuple[int, float | None, bool]:
    """(shift in hours, solar factor at that shift, shift accepted as a proposal) from the telemetry sums of run_episode(record=True)."""
    sse = rec["shift_sse"]
    i0 = SHIFT_GRID.index(0)
    ib = int(np.argmin(sse))
    if rec["shift_samples"] >= 50 and SHIFT_GRID[ib] != 0 and sse[ib] < 0.5 * sse[i0]:
        return SHIFT_GRID[ib], float(rec["shift_factor"][ib]), True
    return 0, rec["solar_factor"], False


def proposals31(cur: Brain, seen: np.ndarray, true_slope: np.ndarray, rec: dict, device) -> tuple[list[tuple[str, Brain]], Brain, dict]:
    """(key, candidate brain) in a fixed order, the corrected belief used as the dream world, and what was measured."""
    new_slope = np.where(seen, true_slope, cur.world.slope).astype(np.float32)
    n_cells = int((np.abs(new_slope - cur.world.slope) > 0.5).sum())
    k, f, shift_ok = fit_shift(rec)
    samples = rec["shift_samples"] if shift_ok else rec["solar_samples"]
    solar_ok = f is not None and samples >= 50 and abs(f - 1.0) > 0.05
    cap_seen = float(rec["battery_seen_max"])
    bat_ok = abs(cap_seen - cur.cfg.rover.battery_wh) > 0.05 * cur.cfg.rover.battery_wh
    fixes = {"map": n_cells > 0, "shift": shift_ok, "solar": solar_ok, "battery": bat_ok}
    notes = {"map": f"belief map <- ground seen ({n_cells} cells corrected)", "shift": f"light forecast shifted by {k:+d} h (fitted from solar readings)",
             "solar": f"belief solar output x {f:.3f} (measured from battery telemetry)" if f is not None else "", "battery": f"battery capacity <- {cap_seen:.1f} Wh (measured)"}

    def build(which: tuple, note: str) -> Brain:
        w = cur.world
        if "map" in which:
            w = dataclasses.replace(w, slope=new_slope)
        if "shift" in which:
            w = shift_world(w, k)
        upd = {}
        if "solar" in which:
            upd["solar_panel_w"] = cur.cfg.rover.solar_panel_w * f
        if "battery" in which:
            upd["battery_wh"] = cap_seen
        cfg = cur.cfg.model_copy(update={"rover": cur.cfg.rover.model_copy(update=upd)}) if upd else cur.cfg
        return Brain(w, cfg, cur.theta, derive_env(cur.model(device), w, cfg, device), note)

    on = tuple(x for x in ("map", "shift", "solar", "battery") if fixes[x])
    out = [(f"{x}:{n_cells}" if x == "map" else x, build((x,), notes[x])) for x in on]
    if len(on) >= 2:
        out.append(("all:" + "+".join(on) + (f":{n_cells}" if "map" in on else ""), build(on, "; ".join(notes[x] for x in on))))
    for dm in (-0.05, 0.05, 0.10):
        m = round(cur.theta.margin + dm, 2)
        if 0.0 <= m <= 0.30:
            out.append((f"margin:{m:.2f}", cur.with_theta(f"energy margin {cur.theta.margin:.2f} -> {m:.2f}", margin=m)))
    for lo, hi in ((0.20, 0.80), (0.50, 1.00)):
        if (lo, hi) != (cur.theta.charge_lo, cur.theta.charge_hi):
            out.append((f"charge:{lo:.2f}/{hi:.2f}", cur.with_theta(f"charge thresholds -> {lo:.2f} / {hi:.2f}", charge_lo=lo, charge_hi=hi)))
    dream = build(on, "dream world") if on else cur
    measured = {"shift_h": k, "shift_proposed": shift_ok, "solar_factor": f, "battery_seen_max": cap_seen, "map_cells": n_cells,
                "sse_at_0": rec["shift_sse"][SHIFT_GRID.index(0)], "sse_at_best": float(min(rec["shift_sse"])), "samples": rec["shift_samples"]}
    return out, dream, measured


def brain_summary31(b: Brain, base_cfg) -> dict:
    return brain_summary(b, base_cfg) | {"belief_battery_wh": b.cfg.rover.battery_wh, "v31": b.theta.v31}


def score_rates(r: dict, n: int) -> dict:
    return rates(r) | {"kept_at_lander": r["kept_at_lander"] / n, "mean_strict_score": float(r["strict_score"].mean())}


def dream_situation31(index: int, site: str, change: str, cfg, device, rounds: int = 2, n_gate: int = 128, n_dream: int = 64, n_score: int = 256,
                      deadline: float | None = None, out_path: Path | None = None, date: str = DREAM_DATE) -> dict:
    region = regions_for_site(site)[0]
    w = build_world(region, date, cfg, device=device)
    w2, cfg2, desc = apply_change(w, cfg, change, seed=0)
    real = FastEnv(w2, n_gate, device, cfg2, seed=0)
    base_env = FastEnv(w, 1, device, cfg, seed=0)
    brain0 = Brain(w, cfg, Theta(v31=True), env=base_env)
    cur = brain0
    t0 = time.time()
    res = {"index": index, "site": site, "world": f"{region}@{date}", "change": desc, "rounds_planned": rounds, "rounds": [], "finished": False, "layer": "v3.1",
           "sizes": {"gate": n_gate, "dream": n_dream, "score": n_score}}

    def save():
        res["minutes"] = (time.time() - t0) / 60
        if out_path is not None:
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(out_path)

    v3_ref = run_episode(reseed(real, n_score, 9000 + index), Brain(w, cfg, Theta(), env=copy.copy(base_env)))
    res["v3_no_learning_same_starts"] = score_rates(v3_ref, n_score)
    before = run_episode(reseed(real, n_score, 9000 + index), brain0)
    res["before"] = score_rates(before, n_score)
    save()
    rec = run_episode(reseed(real, n_gate, 7000 + index), cur, record=True)["record"]
    seen = rec["seen"].copy()
    accepted = 0
    capped = False
    rejected: set = set()
    for r in range(rounds):
        if deadline is not None and time.time() > deadline:
            capped = True
            break
        props, dream, measured = proposals31(cur, seen, w2.slope, rec, device)
        denv = copy.copy(dream.model(device))
        dseed = 8000 + 100 * index + r
        imag_cur = float(run_episode(reseed(denv, n_dream, dseed), cur)["strict_score"].mean())
        imag = [float(run_episode(reseed(denv, n_dream, dseed), p)["strict_score"].mean()) for _, p in props]
        row = {"round": r + 1, "measured": measured, "imagined_current": imag_cur,
               "proposals": [{"key": k, "what": p.note, "imagined": v, "already_rejected": k in rejected} for (k, p), v in zip(props, imag)]}
        better = [i for i, v in enumerate(imag) if v > imag_cur + 1e-9 and props[i][0] not in rejected]
        gseed = 5000 + 100 * index + r
        inc = run_episode(reseed(real, n_gate, gseed), cur, record=True)
        if better:
            best = max(better, key=lambda i: (imag[i], -i))
            key, cand = props[best]
            cnd = run_episode(reseed(real, n_gate, gseed), cand, record=True)
            g = gate_decision31(inc, cnd)
            row.update(sent_to_gate=cand.note, sent_key=key, gate=g, gate_seed=gseed, gate_current=rates(inc), gate_candidate=rates(cnd))
            seen |= cnd["record"]["seen"]
            if g["accept"]:
                cur, accepted, rec = cand, accepted + 1, cnd["record"]
                rejected = set()
            else:
                rejected.add(key)
                rec = inc["record"]
        else:
            row.update(sent_to_gate=None, gate=None, note="no proposal that was not already rejected beat the current brain in imagination; current brain kept")
            rec = inc["record"]
        seen |= inc["record"]["seen"]
        row["brain_after_round"] = brain_summary31(cur, cfg)
        res["rounds"].append(row)
        save()
    res["accepted"] = accepted
    res["capped"] = capped
    after = run_episode(reseed(real, n_score, 9000 + index), cur) if accepted else before
    res["after"] = score_rates(after, n_score)
    d = (after["strict_score"] - before["strict_score"]).double().numpy()
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    gain = float(d.mean())
    res["score_gain_strict"] = gain
    res["score_gain_se"] = se
    worse = gain < -2 * se or after["strict_lost"] > before["strict_lost"]
    res["label"] = "WORSE" if worse else ("IMPROVED" if (accepted >= 1 and gain > 2 * se and gain > 0) else "NO CHANGE")
    res["final_brain"] = brain_summary31(cur, cfg)
    res["finished"] = not capped
    save()
    return res


# ------------------------------------------------------------------------------------------------ command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["eval", "changed", "dream", "dream-all", "dream31", "dream31-all"])
    ap.add_argument("--no-early", action="store_true")
    ap.add_argument("--no-weak", action="store_true")
    ap.add_argument("--sites", nargs="+", default=list(FRESH))
    ap.add_argument("--dates", nargs="+", default=list(DATES))
    ap.add_argument("--date", default=DREAM_DATE)
    ap.add_argument("--changes", nargs="+", default=["blocked_route", "dusty_panels"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--site", default="DM1")
    ap.add_argument("--change", default="blocked_route")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time after which a dream situation starts no new round")
    ap.add_argument("--cap-minutes", type=float, default=40.0)
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args(argv)
    import worldmodel
    print(f"module path: {worldmodel.__file__}", flush=True)
    torch.set_num_threads(a.threads)
    cfg = load_mission()
    device = torch.device(a.device)

    def write(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1))
        tmp.replace(p)

    with torch.no_grad():
        if a.stage == "eval":
            worlds = []
            for site in a.sites:
                for date in a.dates:
                    r = eval_world(regions_for_site(site)[0], date, cfg, device, a.n, a.seed) | {"site": site, "date": date}
                    worlds.append(r)
                    m = r["methods"]
                    print(f"{r['world']}: v31 strict home {m['v31']['strict_returned']:.3f} lost {m['v31']['strict_lost']:.3f} home {m['v31']['returned']:.3f} match {m['v31']['prediction_matched_strict']:.3f} "
                          f"{r['v31_seconds']:.0f}s | v3 strict {m['v3']['strict_returned']:.3f}/{m['v3']['strict_lost']:.3f}", flush=True)
                    print(f"{r['world']}: v3 ice {m['v3']['reached_ice']:.3f} home {m['v3']['returned']:.3f} died {m['v3']['died']:.3f} match {m['v3']['prediction_matched']:.3f} | "
                          + " | ".join(f"{k} {m[k]['reached_ice']:.3f}/{m[k]['returned']:.3f}/{m[k]['died']:.3f}" for k in BASELINES) + f" | {r['seconds']:.0f}s", flush=True)
                    write(a.out, {"stage": "eval", "worlds": worlds, "complete": False})
            write(a.out, {"stage": "eval", "worlds": worlds, "complete": True})
        elif a.stage == "changed":
            worlds = []
            for site in a.sites:
                for change in a.changes:
                    r = changed_world(regions_for_site(site)[0], a.date, change, cfg, device, a.n, a.seed) | {"site": site, "date": a.date, "change_name": change}
                    worlds.append(r)
                    print(f"{r['world']} {change}: " + " | ".join(f"{k} {v['reached_ice']:.3f}/{v['returned']:.3f}/{v['died']:.3f}" for k, v in r["methods"].items())
                          + f" | {r['seconds']:.0f}s", flush=True)
                    write(a.out, {"stage": "changed", "worlds": worlds, "complete": False})
            write(a.out, {"stage": "changed", "worlds": worlds, "complete": True})
        elif a.stage == "dream31":
            r = dream_situation31(a.index, a.site, a.change, cfg, device, rounds=a.rounds, deadline=a.deadline or None, out_path=Path(a.out), date=a.date)
            print(f"dream31 {a.site} {a.change}: accepted {r['accepted']} of {len(r['rounds'])} rounds, strict score gain {r['score_gain_strict']:.2f} "
                  f"(se {r['score_gain_se']:.2f}) {r['label']}, {r['minutes']:.1f} min", flush=True)
        elif a.stage == "dream":
            r = dream_situation(a.index, a.site, a.change, cfg, device, rounds=a.rounds, deadline=a.deadline or None, out_path=Path(a.out))
            print(f"dream {a.site} {a.change}: accepted {r['accepted']} of {len(r['rounds'])} rounds, mean return {r['before']['mean_return']:.2f} -> "
                  f"{r['after']['mean_return']:.2f} ({r['label']}), {r['minutes']:.1f} min", flush=True)
        else:
            out_dir = Path(a.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            deadline = t0 + a.cap_minutes * 60
            v31 = a.stage == "dream31-all"
            queue = list(enumerate(situations31(a.sites, not a.no_early, not a.no_weak) if v31 else SITUATIONS))
            running: list[tuple] = []
            log = []
            while queue or running:
                running = [(p, i, s, c) for p, i, s, c in running if p.poll() is None]
                while queue and len(running) < a.concurrency and time.time() < deadline:
                    i, (s, c) = queue.pop(0)
                    cmd = [sys.executable, "-m", "worldmodel.rover3.run", "dream31" if v31 else "dream", "--date", a.date, "--index", str(i), "--site", s, "--change", c, "--rounds", str(a.rounds),
                           "--deadline", str(deadline), "--threads", str(a.threads), "--device", a.device, "--out", str(out_dir / f"D_{i:02d}_{s}_{c}.json")]
                    p = subprocess.Popen(cmd, stdout=open(out_dir / f"D_{i:02d}_{s}_{c}.log", "w"), stderr=subprocess.STDOUT)
                    running.append((p, i, s, c))
                    log.append({"index": i, "site": s, "change": c, "started_minute": (time.time() - t0) / 60, "pid": p.pid})
                    print(f"started {i} {s} {c} pid {p.pid}", flush=True)
                if queue and time.time() >= deadline:
                    print(f"cap reached: not started: {[(s, c) for _, (s, c) in queue]}", flush=True)
                    log += [{"index": i, "site": s, "change": c, "started_minute": None} for i, (s, c) in queue]
                    queue = []
                time.sleep(2)
            write(out_dir / "D_schedule.json", {"cap_minutes": a.cap_minutes, "concurrency": a.concurrency, "situations": log, "minutes": (time.time() - t0) / 60})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
