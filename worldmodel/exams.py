"""Exam sets (KAN-100): the "intelligence score" is the success rate on these.

Classic guidance already lands ~99.6% on a clear target (docs/STATUS.md), so an exam where
classic also passes proves nothing. Each exam here stresses one thing classic can't handle,
using ONLY the public sim_jax API (EnvParams overrides + terrain + target choice). Nothing in
sim_jax is modified; this module is the single place the hard cases live.

    python -m worldmodel.exams                     # classic guidance on every exam (the baseline)
    python -m worldmodel.exams --n 100 --targets 3 # quicker

Exams (see EXAMS): control, engine_out, nav_drift, nav_drift_near_hazard, camera_blind_near_hazard,
tight_fuel, unseen_craters, combined_stress.

NOT covered yet: "surprise rocks" (a hazard that is absent from the orbital map). The sim has no
such mechanism (config `failures.surprise_hazard_prob` is defined but unused; KAN-104), so it
cannot be expressed as an exam without changing sim_jax.

Provenance: all exams run on GENERATED crater tiles until real LOLA tiles arrive (KAN-21).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from contracts.mission import MissionConfig, load_mission
from contracts.paths import run_dir
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile
from data_tools.make_demo_tile import make_demo_tile
from sim_jax.guidance import classic_policy, guidance_params_from_env
from sim_jax.lander_env import LanderEnv, make_params
from sim_jax.rollout import rollout, summarize
from sim_jax.terrain import terrain_from_tile


# ----------------------------------------------------------------------------- targets
def pick_targets(tile: Tile, k: int, near_hazard_m: float | None, seed: int = 0,
                 safe_margin_m: float = 15.0, max_radius_m: float = 300.0) -> list[tuple[float, float]]:
    """k target points (tile x, y in metres) that are safe to land on.

    near_hazard_m=None  -> any safe cell near the tile centre.
    near_hazard_m=d     -> safe cells that ALSO have a hazard cell within +-d m in x and y (a square
                           window, so up to 1.41*d metres away), i.e. a
                           target on a crater rim / next to rough ground, where a small error
                           lands you on a hazard.
    "Safe" = no hazard cell within `safe_margin_m` (the lander footprint + margin).
    Targets are >= 100 m apart.
    """
    mpp = tile.meters_per_pixel
    haz = tile.hazard >= 0.5
    h, w = haz.shape

    def any_within(r_m: float) -> np.ndarray:
        r = max(1, round(r_m / mpp))
        pad = np.pad(haz, r, constant_values=False)
        return sliding_window_view(pad, (2 * r + 1, 2 * r + 1)).any(axis=(-1, -2))

    ok = ~any_within(safe_margin_m)
    if near_hazard_m is not None:
        ok &= any_within(near_hazard_m)
    rows, cols = np.nonzero(ok)
    x = (cols - (w - 1) / 2) * mpp
    y = ((h - 1) / 2 - rows) * mpp
    inside = np.hypot(x, y) <= max_radius_m
    cand = np.stack([x[inside], y[inside]], axis=1)
    if len(cand) == 0:
        raise ValueError("no candidate targets (relax near_hazard_m / safe_margin_m)")
    rng = np.random.default_rng(seed)
    cand = cand[rng.permutation(len(cand))]
    chosen: list[np.ndarray] = []
    for c in cand:
        if all(np.hypot(*(c - o)) >= 100.0 for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return [(float(c[0]), float(c[1])) for c in chosen]


# ----------------------------------------------------------------------------- exams
@dataclass(frozen=True)
class Exam:
    name: str
    description: str
    overrides: dict = field(default_factory=dict)       # EnvParams fields forced for this exam
    near_hazard_m: float | None = None                  # target placement (see pick_targets)
    tile_seeds: tuple[int, ...] = (0,)                  # which generated crater fields (unseen = >= 100)
    start_mass_kg: float | None = None                  # tight fuel


EXAMS: tuple[Exam, ...] = (
    Exam("control", "Clear target, default failure rates. Classic should pass (sanity check)."),
    Exam("engine_out", "An engine fails in EVERY mission.", {"p_engine_out": 1.0}),
    # NB: nav drift is kept at the config value (0.3 m/s). At 0.6 m/s the velocity bias alone exceeds the
    # 0.5 m/s touchdown limit and a memoryless pilot cannot tell bias from real velocity: unlearnable.
    Exam("nav_drift", "Navigation velocity bias in EVERY mission (config magnitude), clear target.",
         {"p_nav_drift": 1.0}),
    Exam("nav_drift_near_hazard", "Nav drift in EVERY mission, target within 40 m of a hazard cell.",
         {"p_nav_drift": 1.0}, near_hazard_m=40.0),
    Exam("camera_blind_near_hazard", "Hazard/slope camera blind in EVERY mission, target near hazard.",
         {"p_camera_blind": 1.0}, near_hazard_m=40.0),
    Exam("tight_fuel", "Only 60 kg of propellant at start (config: 140 kg).", start_mass_kg=820.0),
    Exam("unseen_craters", "Four crater fields the pilot never trained on (generated), default failures.",
         tile_seeds=(101, 102, 103, 104)),
    Exam("combined_stress", "Engine-out + nav drift + camera blind together, target near a hazard.",
         {"p_engine_out": 1.0, "p_nav_drift": 1.0, "p_camera_blind": 1.0},
         near_hazard_m=40.0),
)
EXAM_BY_NAME = {e.name: e for e in EXAMS}


def make_exam_envs(exam: Exam, cfg: MissionConfig, targets_per_tile: int = 4,
                   env_overrides: dict | None = None) -> list[tuple[str, LanderEnv, object]]:
    """[(label, env, guidance-params-source EnvParams)] — one env per (tile, target)."""
    out = []
    for seed in exam.tile_seeds:
        tile = make_demo_tile(seed=seed, region=f"exam_{exam.name}_{seed}")
        terrain = terrain_from_tile(tile)
        for i, (x, y) in enumerate(pick_targets(tile, targets_per_tile, exam.near_hazard_m, seed=seed)):
            p = make_params(cfg, terrain, target_xy=(x, y), failures=True)
            p = p._replace(**exam.overrides, **(env_overrides or {}))   # e.g. {"lookahead": True}
            if exam.start_mass_kg is not None:
                p = p._replace(start_mass=exam.start_mass_kg,
                               start_fuel=exam.start_mass_kg - p.dry_mass)
            out.append((f"seed{seed}/t{i}@({x:.0f},{y:.0f})", LanderEnv(p), p))
    return out


# policy_factory(env, guidance_params) -> (policy_fn, policy_params)
PolicyFactory = Callable[[LanderEnv, object], tuple[Callable, object]]


def classic_factory(env, gp):
    return classic_policy, gp


def run_exam(exam: Exam, policy_factory: PolicyFactory, cfg: MissionConfig, n_per_target: int = 200,
             targets_per_tile: int = 4, seed: int = 0, env_overrides: dict | None = None) -> dict:
    """Fly `n_per_target` missions at every target. Returns success rate + failure counts."""
    cfgt = cfg.scenarios.terminal
    ok = total = 0
    failures: dict[str, int] = {}
    for j, (_label, env, p) in enumerate(make_exam_envs(exam, cfg, targets_per_tile, env_overrides)):
        gp = guidance_params_from_env(p, cfgt.hover_gate_altitude_m, cfgt.final_descent_speed_m_s)
        policy, params = policy_factory(env, gp)
        _, m, _ = rollout(env, policy, params, jax.random.PRNGKey(seed + 1000 * j), n_per_target)
        s = summarize(m)
        ok += round(s["success_rate"] * s["n"])
        total += s["n"]
        for k, v in s["failures"].items():
            if k != "ok":
                failures[k] = failures.get(k, 0) + v
    return {"exam": exam.name, "n": total, "success_rate": ok / total, "failures": failures}


def run_all(policy_factory: PolicyFactory, exams=EXAMS, n_per_target: int = 200,
            targets_per_tile: int = 4, seed: int = 0, env_overrides: dict | None = None) -> dict:
    cfg = load_mission()
    results = [run_exam(e, policy_factory, cfg, n_per_target, targets_per_tile, seed, env_overrides)
               for e in exams]
    return {"exams": results, "intelligence_score": float(np.mean([r["success_rate"] for r in results]))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="missions per target")
    ap.add_argument("--targets", type=int, default=4, help="targets per tile")
    ap.add_argument("--exam", action="append", choices=list(EXAM_BY_NAME), help="run only these")
    ap.add_argument("--out", default="exams", help="run name under runs/ for the JSON result")
    a = ap.parse_args(argv)
    exams = tuple(EXAM_BY_NAME[n] for n in a.exam) if a.exam else EXAMS

    t0 = time.time()
    res = run_all(classic_factory, exams, a.n, a.targets)
    print(f"{'exam':28s} {'n':>6s} {'classic':>9s}  failures")
    for r in res["exams"]:
        print(f"{r['exam']:28s} {r['n']:6d} {r['success_rate']:9.1%}  {r['failures']}")
    print(f"mean over exams: {res['intelligence_score']:.1%}   ({time.time() - t0:.0f} s)")
    meta = Meta(provenance=Provenance.GENERATED, source="worldmodel.exams on generated crater tiles, classic guidance",
                frame="tile ENU", created_by="worldmodel.exams")
    res["meta"] = json.loads(meta.model_dump_json())
    res["policy"] = "classic ZEM/ZEV"
    out = run_dir(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classic_baseline.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", out / "classic_baseline.json", "[GENERATED — NOT OBSERVED]")


if __name__ == "__main__":
    main()
