"""Fitness used for evolution (shaped) vs the mission score we report (gated).

The mission score gates unsafe landings to exactly 0 (config scoring.landing_gate). If we
selected on that directly, a generation where nobody lands would have a flat landscape and
evolution would drift. So selection uses a *shaped* fitness that is:

    success  ->  landing_score  (100 + terms; always > 0 in practice)
    failure  ->  a NEGATIVE number that gets closer to 0 the gentler the failure was

Anything > 0 is therefore a safe landing, which also gives a free population success rate.
The reported `best_score` / `mean_score` in generation_XXX.json use the real gated score.
"""

from __future__ import annotations

import jax.numpy as jnp

from contracts.mission import MissionConfig
from sim_jax.lander_env import OUTCOME
from sim_jax.rollout import landing_score

FAIL_CAP = 20.0


def shaped_fitness(metrics: dict, cfg: MissionConfig):
    sc = cfg.scoring
    lim = cfg.lander.touchdown_limits
    mission = landing_score(metrics, sc.w_landed, sc.w_target_distance_per_m, sc.w_fuel_per_kg,
                            gate=True)
    # How far over the touchdown limits (in units of the limit) — 0 for a legal touchdown.
    over = (jnp.maximum(metrics["td_vz"] / lim.max_vertical_speed_m_s - 1.0, 0.0)
            + jnp.maximum(metrics["td_vh"] / lim.max_horizontal_speed_m_s - 1.0, 0.0)
            + jnp.maximum(metrics["td_slope"] / lim.max_slope_deg - 1.0, 0.0))
    landed_badly = (metrics["outcome"] >= OUTCOME["hard_vertical"]) & (
        metrics["outcome"] <= OUTCOME["out_of_fuel"])   # touched down, but illegally
    # Never touched down (timeout / out of bounds): the worst failure.
    failure = jnp.where(landed_badly, -jnp.minimum(1.0 + over, FAIL_CAP), -FAIL_CAP)
    return jnp.where(metrics["success"], mission, failure), mission
