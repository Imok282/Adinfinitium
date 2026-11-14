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
import dataclasses
import os
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
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
from worldmodel.exams import EXAM_BY_NAME, make_exam_envs

# Arguments that define WHAT is being evolved. A resumed run must match them exactly; the rest
# (--generations, --champion-every, --eval-missions, --topk, --val-missions) may change.
IDENTITY = ("pop", "missions", "mode", "residual_scale", "region", "flat", "no_failures", "seed",
            "init_weight_std", "weight_mutate_power", "max_nodes", "max_conns", "species_size",
            "survival_threshold", "compatibility_threshold", "exams", "exam_targets", "exam_missions")
TRAIN_SEED_SHIFT = 1000   # training exams use tile seeds + 1000: same stress, different terrain/targets
CHECKPOINT = "checkpoint.pkl"


def save_checkpoint(out: Path, state, gen_done: int, best_held_out: float, args) -> None:
    """Whole TensorNEAT state (population, species, RNG, generation counter), written atomically
    AFTER the generation's json/npz files, so a crash never leaves a checkpoint ahead of the logs."""
    payload = {"state": jax.device_get(state), "gen_done": gen_done, "best_held_out": best_held_out,
               "identity": {k: getattr(args, k) for k in IDENTITY}}
    tmp = out / (CHECKPOINT + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out / CHECKPOINT)


def load_checkpoint(out: Path, args) -> dict:
    with open(out / CHECKPOINT, "rb") as f:
        ck = pickle.load(f)  # our own file, written by save_checkpoint
    diff = {k: (ck["identity"].get(k), getattr(args, k)) for k in IDENTITY
            if ck["identity"].get(k) != getattr(args, k)}
    if diff:
        raise SystemExit("--resume: arguments differ from the checkpointed run "
                         f"(checkpoint, now): {diff}. Use the original arguments or a new --run.")
    return ck


def exam_suites(cfg, names: list[str], n_targets: int):
    """(train, held_out). train = [(env, gp)] on tiles seeded +1000; held_out = [(exam_name, env, gp)]
    on the exam's own tiles. Same stress, different terrain and targets, so the exam score is not
    a memorised answer."""
    t = cfg.scenarios.terminal

    def gp_of(p):
        return guidance_params_from_env(p, t.hover_gate_altitude_m, t.final_descent_speed_m_s)

    train, held = [], []
    for name in names:
        exam = EXAM_BY_NAME[name]
        shifted = dataclasses.replace(exam, tile_seeds=tuple(x + TRAIN_SEED_SHIFT for x in exam.tile_seeds))
        train += [(e, gp_of(p)) for _, e, p in make_exam_envs(shifted, cfg, n_targets)]
        held += [(name, e, gp_of(p)) for _, e, p in make_exam_envs(exam, cfg, n_targets)]
    return train, held


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
    names = [n for n in args.exams.split(",") if n]
    unknown = [n for n in names if n not in EXAM_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown exam(s) {unknown}; choose from {sorted(EXAM_BY_NAME)}")
    train_suite, held_suite = exam_suites(cfg, names, args.exam_targets)
    problem = LanderProblem(env, gp, cfg, pilot, missions_per_ind=args.missions, seed=args.seed,
                            extra_suite=tuple(train_suite), missions_per_extra=args.exam_missions)
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
    return cfg, env, gp, pilot, problem, pipe, source, [(env, gp), *train_suite], held_suite


def make_eval_fns(cfg, env, gp, pilot, algo):
    """Jitted held-out evaluation of one genome, and classic guidance on the same missions."""

    @jax.jit
    def eval_genome(state, params, keys):
        fwd = lambda p, x: algo.forward(state, p, x)
        m = pilot_metrics(env, gp, pilot, fwd, params, keys)
        _, mission = shaped_fitness(m, cfg)
        return m, mission

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

    return eval_genome, eval_classic, fly_one


def make_suite_fns(cfg, pilot, algo, rerank_suite, held_suite):
    """rerank: K genomes scored on the training-style suite (validation keys).
    exam_champion / exam_classic: success matrix [E, n] on the HELD-OUT exam envs."""

    @partial(jax.jit, static_argnames="n")
    def rerank(state, genomes, base_key, gen, n):
        def one(g):
            tr_ = algo.transform(state, g)
            fwd = lambda p, x: algo.forward(state, p, x)
            fits = []
            for i, (env, gp) in enumerate(rerank_suite):
                ks = mission_keys(jax.random.fold_in(base_key, i), gen, n)
                fits.append(shaped_fitness(pilot_metrics(env, gp, pilot, fwd, tr_, ks), cfg)[0].mean())
            return jnp.mean(jnp.stack(fits))
        return jax.vmap(one)(genomes)

    @partial(jax.jit, static_argnames="n")
    def exam_champion(state, params, base_key, gen, n):
        fwd = lambda p, x: algo.forward(state, p, x)
        return jnp.stack([pilot_metrics(env, gp, pilot, fwd, params,
                                        mission_keys(jax.random.fold_in(base_key, i), gen, n))["success"]
                          for i, (_, env, gp) in enumerate(held_suite)])

    @partial(jax.jit, static_argnames="n")
    def exam_classic(base_key, gen, n):
        rows = []
        for i, (_, env, gp) in enumerate(held_suite):
            def one(k, env=env, gp=gp):
                sf, m0, _ = _episode(env, classic_policy, gp, k, False)
                return sf, m0
            sf, m0 = jax.vmap(one)(mission_keys(jax.random.fold_in(base_key, i), gen, n))
            rows.append(episode_metrics(sf, m0)["success"])
        return jnp.stack(rows)

    return rerank, exam_champion, exam_classic


def exam_report(held_suite, champ_ok, classic_ok) -> dict:
    """Pool per exam name, then average over exams (same convention as worldmodel.exams)."""
    names = sorted({n for n, _, _ in held_suite}, key=[n for n, _, _ in held_suite].index)
    idx = {n: [i for i, (m, _, _) in enumerate(held_suite) if m == n] for n in names}
    br = {n: {"champion": float(champ_ok[idx[n]].mean()), "classic": float(classic_ok[idx[n]].mean())}
          for n in names}
    return {"exam_success_rate": float(np.mean([v["champion"] for v in br.values()])),
            "classic_exam_success_rate": float(np.mean([v["classic"] for v in br.values()])),
            "exam_breakdown": br}


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
    ap.add_argument("--exams", default="",
                    help="comma list from worldmodel.exams (e.g. tight_fuel,nav_drift_near_hazard,combined_stress): "
                         "ALSO select on these stress scenarios (tiles seeded +1000) and report held-out "
                         "exam_success_rate on the exams' own tiles")
    ap.add_argument("--exam-targets", type=int, default=2, help="targets per exam tile")
    ap.add_argument("--exam-missions", type=int, default=2, help="missions per exam env per individual")
    ap.add_argument("--exam-eval-missions", type=int, default=64, help="held-out missions per exam env")
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<run>/checkpoint.pkl (starts fresh if there is none); "
                         "--generations is the TOTAL to reach")
    a = ap.parse_args(argv)

    print("JAX devices:", jax.devices())
    cfg, env, gp, pilot, _problem, pipe, source, rerank_suite, held_suite = build(a)
    algo = pipe.algorithm
    state = pipe.setup()
    eval_genome, eval_classic, fly_one = make_eval_fns(cfg, env, gp, pilot, algo)
    rerank, exam_champion, exam_classic = make_suite_fns(cfg, pilot, algo, rerank_suite, held_suite)
    out = run_dir(a.run)
    out.mkdir(parents=True, exist_ok=True)
    start_gen, best_held_out = 0, -1.0
    if (out / CHECKPOINT).exists():
        if not a.resume:
            raise SystemExit(f"{out / CHECKPOINT} exists: pass --resume to continue that run, "
                             "or choose a new --run (refusing to overwrite it)")
        ck = load_checkpoint(out, a)
        state, start_gen, best_held_out = ck["state"], ck["gen_done"] + 1, ck["best_held_out"]
        print(f"resuming run '{a.run}' from generation {start_gen} (target {a.generations})")
    elif a.resume:
        print(f"--resume: no checkpoint in {out}, starting fresh")
    meta = Meta(provenance=Provenance.MODELLED, source=f"sim_jax 3-DOF + TensorNEAT on {source}",
                frame="tile ENU", created_by="swarm.run_swarm",
                units={"thrust_n": "N", "pos_m": "m", "vel_m_s": "m/s"})
    strategy = {"algorithm": "TensorNEAT", "pilot_mode": a.mode, "residual_scale": a.residual_scale,
                "missions_per_ind": a.missions, "topk": a.topk, "val_missions": a.val_missions, "species_size": a.species_size,
                "survival_threshold": a.survival_threshold,
                "compatibility_threshold": a.compatibility_threshold, "seed": a.seed,
                "init_weight_std": a.init_weight_std, "weight_mutate_power": a.weight_mutate_power,
                "exams": a.exams or None, "exam_train_tile_seed_shift": TRAIN_SEED_SHIFT if a.exams else None,
                "dream_rsi": "not enabled (KAN-106)"}

    step = jax.jit(pipe.step)
    val_base = jax.random.PRNGKey(a.seed + 777)           # validation: re-ranks the top-K champions
    exam_base = jax.random.PRNGKey(a.seed + 424242)       # held-out exam missions
    eval_base = jax.random.PRNGKey(a.seed + 12345)        # held-out report set: never used for selection
    for gen in range(start_gen, a.generations):
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
        val_scores = np.asarray(rerank(state_before, top_genomes, val_base, gen, a.val_missions))
        champ_i = int(top[int(np.argmax(val_scores))])
        genome = (pop[0][champ_i], pop[1][champ_i])
        transformed = algo.transform(state_before, genome)

        keys = mission_keys(eval_base, gen, a.eval_missions)
        m_champ, mission_champ = eval_genome(state_before, transformed, keys)
        m_classic, mission_classic = eval_classic(keys)
        sc, scl = summarize(m_champ), summarize(m_classic)
        exam_extra = {}
        if held_suite:
            ex = exam_report(held_suite,
                             np.asarray(exam_champion(state_before, transformed, exam_base, gen, a.exam_eval_missions)),
                             np.asarray(exam_classic(exam_base, gen, a.exam_eval_missions)))
            exam_extra = {"exam_success_rate": ex["exam_success_rate"],
                          "classic_exam_success_rate": ex["classic_exam_success_rate"],
                          "exam_breakdown": ex["exam_breakdown"]}
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
            meta=meta, **exam_extra,
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
        save_checkpoint(out, state, gen, best_held_out, a)

        print(f"gen {gen:3d} | {wall:6.1f}s | pop>0 {log.population_success_rate:6.1%} | "
              f"champion held-out {held:6.1%} vs classic {scl['success_rate']:6.1%} | "
              f"fails {sc['failures']}"
              + (f" | EXAMS {exam_extra['exam_success_rate']:6.1%} vs classic "
                 f"{exam_extra['classic_exam_success_rate']:6.1%}" if exam_extra else ""), flush=True)


if __name__ == "__main__":
    main()
