"""CHECK for the 4090 (KAN-20 / KAN-101): fly N landers with classic guidance and time it.

    python -m scripts.bench_swarm                       # 10,000 landers on the demo tile
    python -m scripts.bench_swarm --n 1000 --flat       # quick CPU check

Pass criteria on the 4090: runs, success_rate > 0.95, second (compiled) run < 10 s.
Writes runs/bench/generation_000.json so the brain window has something to read.
"""

from __future__ import annotations

import argparse
import time

import jax

from contracts.mission import load_mission
from contracts.paths import run_dir, tile_path
from contracts.provenance import Meta, Provenance
from contracts.runs import FailureCount, GenerationLog, gen_name, write_json
from contracts.tiles import load_tile
from sim_jax.guidance import classic_policy, guidance_params_from_env
from sim_jax.lander_env import LanderEnv, make_params
from sim_jax.rollout import rollout, summarize
from sim_jax.terrain import flat_terrain, terrain_from_tile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--region", default="demo_craters")
    ap.add_argument("--flat", action="store_true", help="use flat terrain (no tile needed)")
    ap.add_argument("--no-failures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("JAX devices:", jax.devices())
    cfg = load_mission()
    if a.flat:
        terrain, source = flat_terrain(101, 40.0), "flat synthetic"
    else:
        tile = load_tile(tile_path(a.region))
        terrain, source = terrain_from_tile(tile), f"{a.region} [{tile.meta.provenance.banner}]"
    p = make_params(cfg, terrain, failures=not a.no_failures)
    env = LanderEnv(p)
    gp = guidance_params_from_env(p, cfg.scenarios.terminal.hover_gate_altitude_m,
                                  cfg.scenarios.terminal.final_descent_speed_m_s)
    key = jax.random.PRNGKey(a.seed)

    t0 = time.time()
    _, m, _ = rollout(env, classic_policy, gp, key, a.n)
    m["success"].block_until_ready()
    t_first = time.time() - t0
    t0 = time.time()
    _, m, _ = rollout(env, classic_policy, gp, jax.random.fold_in(key, 1), a.n)
    m["success"].block_until_ready()
    t_second = time.time() - t0

    s = summarize(m)
    steps = a.n * p.n_steps
    print(f"terrain      : {source}")
    print(f"landers      : {a.n:,} x {p.n_steps} steps")
    print(f"first run    : {t_first:.2f} s (includes compile)")
    print(f"second run   : {t_second:.2f} s  ->  {steps / t_second / 1e6:.1f} M lander-steps/s")
    print(f"success rate : {s['success_rate']:.2%}")
    print(f"failures     : {s['failures']}")

    log = GenerationLog(
        run="bench", generation=0, population=a.n, best_score=float("nan"), mean_score=float("nan"),
        success_rate=s["success_rate"], classic_success_rate=s["success_rate"],
        failures=[FailureCount(reason=k, count=v) for k, v in s["failures"].items()],
        strategy={"policy": "classic ZEM/ZEV guidance"}, wall_time_s=t_second,
        meta=Meta(provenance=Provenance.MODELLED, source=f"sim_jax 3-DOF on {source}",
                  frame="tile ENU", created_by="scripts.bench_swarm"),
    )
    out = write_json(log, run_dir("bench") / gen_name("generation", 0, "json"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
