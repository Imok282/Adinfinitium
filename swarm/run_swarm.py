"""Evolve landing pilots with TensorNEAT on the JAX swarm sim (KAN-105).

    python -m swarm.run_swarm --pop 10000 --generations 300 --run overnight1     # 4090
    python -m swarm.run_swarm --pop 64 --generations 3 --missions 2 --flat --run smoke   # laptop

Per generation it writes, under runs/<run>/ (contracts/runs.py):
    generation_XXX.json   scores, success rates, failure reasons, champion genome, strategy
    champion_XXX.json     best pilot's mission (start state + thrust log) for the Basilisk re-fly
    genome_XXX.npz        exact genome of the generation's best individual (restart / audit)

How to read the numbers (all honest by construction):
  * champion                best of the top `--topk` by selection fitness, re-ranked on
                            `--val-missions` validation missions
  * `success_rate`          champion on `--eval-missions` HELD-OUT missions (fresh keys, used
                            neither for selection nor for the re-rank)
  * `classic_success_rate`  classic ZEM/ZEV guidance on the SAME held-out missions
  * `best_score`            champion's held-out mean *mission* score (unsafe landing = 0)
  * `mean_score`            population mean of the *shaped* selection fitness (see fitness.py)
  * `population_success_rate`  fraction of individuals with fitness > 0 on the selection missions
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from contracts.mission import load_mission
from contracts.paths import run_dir, tile_path
from contracts.provenance import Meta, Provenance
from contracts.runs import (
    Champion,
    ControlStep,
    FailureCount,
    GenerationLog,
    MissionGenes,
    gen_name,
    write_json,
)
from contracts.tiles import load_tile
from sim_jax.guidance import classic_policy, guidance_params_from_env
from sim_jax.lander_env import FLYING, LanderEnv, make_params
from sim_jax.rollout import _episode, episode_metrics, summarize
from sim_jax.terrain import flat_terrain, terrain_from_tile
from swarm.fitness import shaped_fitness
from swarm.pilot import PilotConfig, make_pilot_policy
from swarm.problem import LanderProblem, mission_keys, pilot_metrics


def build(args):
    """Everything needed to evolve: returns (cfg, env, gp, pilot_cfg, problem, pipeline, source)."""
    from tensorneat.algorithm.neat import NEAT
    from tensorneat.common import ACT
    from tensorneat.genome import DefaultConn, DefaultGenome, DefaultNode
    from tensorneat.pipeline import Pipeline

    cfg = load_mission()
    if args.flat:
        terrain, source = flat_terrain(101, 40.0), "flat synthetic"
    else:
        tile = load_tile(tile_path(args.region))
        terrain, source = terrain_from_tile(tile), f"{args.region} [{tile.meta.provenance.banner}]"
    p = make_params(cfg, terrain, failures=not args.no_failures)
    env = LanderEnv(p)
    gp = guidance_params_from_env(p, cfg.scenarios.terminal.hover_gate_altitude_m,
                                  cfg.scenarios.terminal.final_descent_speed_m_s)
    pilot = PilotConfig(mode=args.mode, residual_scale=args.residual_scale)
    problem = LanderProblem(env, gp, cfg, pilot, missions_per_ind=args.missions, seed=args.seed)
    genome = DefaultGenome(
        num_inputs=pilot.n_inputs, num_outputs=pilot.n_outputs,
        max_nodes=args.max_nodes, max_conns=args.max_conns,
        node_gene=DefaultNode(activation_options=[ACT.tanh], activation_default=ACT.tanh,
                              bias_init_std=args.init_weight_std),
        # Small initial weights AND biases: residual pilots then start close to classic guidance
        # instead of with a random +-0.3 thrust error on every axis.
        conn_gene=DefaultConn(weight_init_std=args.init_weight_std, weight_mutate_power=args.weight_mutate_power),
        output_transform=None,
    )
    algo = NEAT(genome=genome, pop_size=args.pop, species_size=args.species_size,
                survival_threshold=args.survival_threshold,
                compatibility_threshold=args.compatibility_threshold,
                pop_batch_size=args.pop_batch)
    pipe = Pipeline(algo, problem, seed=args.seed, generation_limit=args.generations,
                    fitness_target=float("inf"))
    return cfg, env, gp, pilot, problem, pipe, source


def make_eval_fns(cfg, env, gp, pilot, algo):
    """Jitted held-out evaluation of one genome, and classic guidance on the same missions."""

    @jax.jit
    def eval_genome(state, params, keys):
        fwd = lambda p, x: algo.forward(state, p, x)
        m = pilot_metrics(env, gp, pilot, fwd, params, keys)
        _, mission = shaped_fitness(m, cfg)
        return m, mission

    @jax.jit
    def rerank(state, genomes, keys):
        """Mean mission score of each of K candidate genomes on the same validation missions."""
        def one(g):
            tr_ = algo.transform(state, g)
            fwd = lambda p, x: algo.forward(state, p, x)
            m = pilot_metrics(env, gp, pilot, fwd, tr_, keys)
            return shaped_fitness(m, cfg)[0].mean()
        return jax.vmap(one)(genomes)

    @jax.jit
    def eval_classic(keys):
        def one(k):
            sf, m0, _ = _episode(env, classic_policy, gp, k, False)
            return sf, m0
        sf, m0 = jax.vmap(one)(keys)
        m = episode_metrics(sf, m0)
        _, mission = shaped_fitness(m, cfg)
        return m, mission

    @jax.jit
    def fly_one(state, params, key):
        fwd = lambda p, x: algo.forward(state, p, x)
        policy = make_pilot_policy(pilot, gp, fwd)
        k_reset, _ = jax.random.split(key)
        _, s0 = env.reset(k_reset)
        _, _, traj = _episode(env, policy, params, key, True)
        return s0, traj

    return eval_genome, eval_classic, fly_one, rerank


def champion_file(run, gen, cid, score, fly_out, env, region, outcome_name, meta):
    s0, (_pos, _vel, _mass, thrust, outcome) = jax.tree_util.tree_map(np.asarray, fly_out)
    done_idx = np.nonzero(outcome != FLYING)[0]
    n = int(done_idx[0]) + 1 if done_idx.size else len(outcome)
    dt = env.params.dt
    controls = [ControlStep(t_s=float((i + 1) * dt), thrust_n=[float(v) for v in thrust[i]])
                for i in range(n)]
    return Champion(
        run=run, generation=gen, champion_id=cid, score=float(score),
        genes=MissionGenes(site=region, target_x_m=float(s0.target[0]), target_y_m=float(s0.target[1])),
        initial_state={"pos_m": [float(v) for v in s0.pos], "vel_m_s": [float(v) for v in s0.vel],
                       "mass_kg": float(s0.mass), "engines_ok": int(s0.engines_ok),
                       "engine_out_t_s": None if not np.isfinite(s0.engine_out_t) else float(s0.engine_out_t)},
        controls=controls, outcome=outcome_name, meta=meta,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="swarm1")
    ap.add_argument("--pop", type=int, default=10_000)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--missions", type=int, default=4, help="missions per individual per generation")
    ap.add_argument("--eval-missions", type=int, default=512, help="held-out missions for champion + classic")
    ap.add_argument("--topk", type=int, default=16, help="champion candidates re-ranked on validation missions")
    ap.add_argument("--val-missions", type=int, default=128)
    ap.add_argument("--mode", choices=["residual", "scratch"], default="residual")
    ap.add_argument("--residual-scale", type=float, default=0.3)
    ap.add_argument("--region", default="demo_craters")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--no-failures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weight-std", type=float, default=0.05)
    ap.add_argument("--weight-mutate-power", type=float, default=0.15)
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-conns", type=int, default=120)
    ap.add_argument("--species-size", type=int, default=20)
    ap.add_argument("--survival-threshold", type=float, default=0.1)
    ap.add_argument("--compatibility-threshold", type=float, default=2.0)
    ap.add_argument("--pop-batch", type=int, default=None, help="evaluate the population in batches (GPU memory)")
    ap.add_argument("--champion-every", type=int, default=10)
    a = ap.parse_args(argv)

    print("JAX devices:", jax.devices())
    cfg, env, gp, pilot, _problem, pipe, source = build(a)
    algo = pipe.algorithm
    state = pipe.setup()
    eval_genome, eval_classic, fly_one, rerank = make_eval_fns(cfg, env, gp, pilot, algo)
    out = run_dir(a.run)
    out.mkdir(parents=True, exist_ok=True)
    meta = Meta(provenance=Provenance.MODELLED, source=f"sim_jax 3-DOF + TensorNEAT on {source}",
                frame="tile ENU", created_by="swarm.run_swarm",
                units={"thrust_n": "N", "pos_m": "m", "vel_m_s": "m/s"})
    strategy = {"algorithm": "TensorNEAT", "pilot_mode": a.mode, "residual_scale": a.residual_scale,
                "missions_per_ind": a.missions, "topk": a.topk, "val_missions": a.val_missions, "species_size": a.species_size,
                "survival_threshold": a.survival_threshold,
                "compatibility_threshold": a.compatibility_threshold, "seed": a.seed,
                "init_weight_std": a.init_weight_std, "weight_mutate_power": a.weight_mutate_power,
                "dream_rsi": "not enabled (KAN-106)"}

    step = jax.jit(pipe.step)
    val_base = jax.random.PRNGKey(a.seed + 777)           # validation: re-ranks the top-K champions
    eval_base = jax.random.PRNGKey(a.seed + 12345)        # held-out report set: never used for selection
    best_held_out = -1.0
    for gen in range(a.generations):
        t0 = time.time()
        state_before = state
        state, pop, fit = step(state)
        fit = np.asarray(jax.device_get(fit))
        valid = np.isfinite(fit)
        # Champion = best of the top-K by selection fitness, re-ranked on VALIDATION missions
        # (fresh keys, shared by the K candidates). Picking the single best of 10,000 on a few
        # selection missions mostly picks luck (winner's curse).
        top = np.argsort(-np.where(valid, fit, -np.inf))[:a.topk]
        top_genomes = (pop[0][top], pop[1][top])
        val_scores = np.asarray(rerank(state_before, top_genomes, mission_keys(val_base, gen, a.val_missions)))
        champ_i = int(top[int(np.argmax(val_scores))])
        genome = (pop[0][champ_i], pop[1][champ_i])
        transformed = algo.transform(state_before, genome)

        keys = mission_keys(eval_base, gen, a.eval_missions)
        m_champ, mission_champ = eval_genome(state_before, transformed, keys)
        m_classic, mission_classic = eval_classic(keys)
        sc, scl = summarize(m_champ), summarize(m_classic)
        wall = time.time() - t0

        nodes, conns = (np.asarray(x) for x in jax.device_get(genome))
        np.savez(out / f"genome_{gen:03d}.npz", nodes=nodes, conns=conns, fitness=fit[champ_i])
        cid = f"g{gen:03d}-best"
        log = GenerationLog(
            run=a.run, generation=gen, population=a.pop,
            best_score=float(mission_champ.mean()), mean_score=float(fit[valid].mean()),
            success_rate=sc["success_rate"], classic_success_rate=scl["success_rate"],
            failures=[FailureCount(reason=k, count=v) for k, v in sc["failures"].items()],
            champion_id=cid,
            champion_genome={"nodes": nodes.tolist(), "conns": conns.tolist(), "pilot_mode": a.mode},
            strategy=strategy, wall_time_s=wall,
            population_success_rate=float((fit[valid] > 0).mean()),
            best_fitness_shaped=float(fit[champ_i]), score_basis="best=mission(gated), mean=shaped",
            classic_best_score=float(mission_classic.mean()), n_eval_missions=a.eval_missions,
            meta=meta,
        )
        write_json(log, out / gen_name("generation", gen, "json"))

        held = sc["success_rate"]
        if gen % a.champion_every == 0 or gen == a.generations - 1 or held > best_held_out:
            ok = np.asarray(m_champ["success"])
            j = int(np.argmax(ok)) if ok.any() else 0     # first successful held-out mission
            fly = fly_one(state_before, transformed, keys[j])
            from contracts.runs import FAILURE_REASONS
            champ = champion_file(a.run, gen, cid, float(np.asarray(mission_champ)[j]), fly, env,
                                  a.region, FAILURE_REASONS[int(np.asarray(m_champ["outcome"])[j])], meta)
            write_json(champ, out / gen_name("champion", gen, "json"))
        best_held_out = max(best_held_out, held)

        print(f"gen {gen:3d} | {wall:6.1f}s | pop>0 {log.population_success_rate:6.1%} | "
              f"champion held-out {held:6.1%} vs classic {scl['success_rate']:6.1%} | "
              f"fails {sc['failures']}", flush=True)


if __name__ == "__main__":
    main()
