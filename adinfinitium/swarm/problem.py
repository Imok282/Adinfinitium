"""TensorNEAT problem: one individual = one pilot network, flown through the JAX lander env.

TensorNEAT calls `evaluate(state, key, act_func, params)` once per individual and vmaps it over
the whole population, so a generation of 10,000 pilots is a single jitted call:

    population (vmap) x missions_per_ind (vmap) x n_steps (lax.scan)

Fairness: every individual in a generation flies the SAME missions (same start states, same
failure draws, same sensor noise), derived from (seed, generation). Selection then compares
pilots, not luck. Missions change every generation so the pilots can't memorise a start.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from tensorneat.problem import BaseProblem

from contracts.mission import MissionConfig
from sim_jax.guidance import GuidanceParams
from sim_jax.lander_env import ACT_DIM, LanderEnv
from sim_jax.rollout import _episode, episode_metrics
from swarm.fitness import shaped_fitness
from swarm.pilot import PilotConfig, make_pilot_policy


def mission_keys(base_key, generation, n: int):
    """The n mission keys every individual flies in this generation."""
    return jax.random.split(jax.random.fold_in(base_key, generation), n)


def pilot_metrics(env: LanderEnv, gp: GuidanceParams, pcfg: PilotConfig, net_forward, net_params, keys):
    """Fly one pilot through every mission key. Returns (metrics dict [n], start_mass [n])."""
    policy = make_pilot_policy(pcfg, gp, net_forward)

    def one(k):
        sf, m0, _ = _episode(env, policy, net_params, k, False)
        return sf, m0

    sf, m0 = jax.vmap(one)(keys)
    return episode_metrics(sf, m0)


class LanderProblem(BaseProblem):
    jitable = True

    def __init__(self, env: LanderEnv, gp: GuidanceParams, mission: MissionConfig,
                 pilot: PilotConfig, missions_per_ind: int = 4, seed: int = 0,
                 extra_suite: tuple = (), missions_per_extra: int = 1):
        """`extra_suite` = ((env, gp), ...): exam scenarios the pilot is ALSO selected on. Each env
        (the main one and every extra one) gets equal weight in the fitness."""
        super().__init__()
        self.env, self.gp, self.mission, self.pilot = env, gp, mission, pilot
        self.extra_suite, self.missions_per_extra = tuple(extra_suite), missions_per_extra
        self.missions_per_ind = missions_per_ind
        self.base_key = jax.random.PRNGKey(seed)

    @property
    def input_shape(self):
        return (self.pilot.n_inputs,)

    @property
    def output_shape(self):
        return (ACT_DIM,)

    def evaluate(self, state, randkey, act_func, params):
        del randkey  # shared missions, see module docstring
        keys = mission_keys(self.base_key, state.generation, self.missions_per_ind)
        metrics = pilot_metrics(self.env, self.gp, self.pilot,
                                lambda p, x: act_func(state, p, x), params, keys)
        fit, _ = shaped_fitness(metrics, self.mission)
        per_env = [jnp.mean(fit)]
        for i, (env, gp) in enumerate(self.extra_suite):
            ks = mission_keys(jax.random.fold_in(self.base_key, i + 1), state.generation,
                              self.missions_per_extra)
            m = pilot_metrics(env, gp, self.pilot, lambda p, x: act_func(state, p, x), params, ks)
            per_env.append(jnp.mean(shaped_fitness(m, self.mission)[0]))
        return jnp.mean(jnp.stack(per_env))
