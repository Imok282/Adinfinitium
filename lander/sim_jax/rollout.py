"""Fly many landers at once: vmap over missions, lax.scan over time.

    from sim_jax.rollout import rollout
    final, metrics = rollout(env, classic_policy, gp, key, n_envs=10_000)
    metrics["success"].mean()

`policy_fn(policy_params, obs, key) -> action` — classic guidance and evolved pilots share
this signature, so the same missions can be flown by both (AI vs physics split-screen).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from contracts.runs import FAILURE_REASONS
from sim_jax.lander_env import OUTCOME, EnvState, LanderEnv


def _episode(env: LanderEnv, policy_fn, policy_params, key, record: bool):
    k_reset, k_run = jax.random.split(key)
    obs0, s0 = env.reset(k_reset)
    start_mass = s0.mass

    def body(carry, k):
        obs, s = carry
        k_pol, k_step = jax.random.split(k)
        a = policy_fn(policy_params, obs, k_pol)
        obs2, s2, _, _, _ = env.step(k_step, s, a)
        out = (s2.pos, s2.vel, s2.mass, s2.thrust, s2.outcome) if record else None
        return (obs2, s2), out

    keys = jax.random.split(k_run, env.params.n_steps)
    (_, sf), traj = jax.lax.scan(body, (obs0, s0), keys)
    return sf, start_mass, traj


def episode_metrics(sf: EnvState, start_mass) -> dict:
    dist = jnp.linalg.norm(sf.pos[..., :2] - sf.target, axis=-1)
    return {
        "outcome": sf.outcome,
        "success": sf.outcome == OUTCOME["ok"],
        "target_dist_m": dist,
        "fuel_used_kg": start_mass - sf.mass,
        "time_s": sf.t,
        "td_vz": sf.td[..., 0],
        "td_vh": sf.td[..., 1],
        "td_slope": sf.td[..., 2],
        "engine_failed": jnp.isfinite(sf.engine_out_t),
    }


@partial(jax.jit, static_argnames=("env", "policy_fn", "n_envs", "record"))
def rollout(env: LanderEnv, policy_fn, policy_params, key, n_envs: int, record: bool = False):
    keys = jax.random.split(key, n_envs)
    sf, m0, traj = jax.vmap(lambda k: _episode(env, policy_fn, policy_params, k, record))(keys)
    return sf, episode_metrics(sf, m0), traj


def landing_score(metrics: dict, w_landed=100.0, w_dist=0.05, w_fuel=0.2, gate=True):
    """Landing-phase fitness. Surface/rover/night terms are added in KAN-29."""
    s = (w_landed * metrics["success"] - w_dist * metrics["target_dist_m"]
         - w_fuel * metrics["fuel_used_kg"])
    if gate:  # unsafe landing => score 0 (mission.yaml scoring.landing_gate)
        s = jnp.where(metrics["success"], s, 0.0)
    return s


def summarize(metrics: dict) -> dict:
    """Host-side summary: success rate + failure-reason counts (for generation_XXX.json)."""
    m = jax.tree_util.tree_map(np.asarray, metrics)
    counts = np.bincount(np.clip(m["outcome"], 0, None), minlength=len(FAILURE_REASONS))
    still_flying = int((m["outcome"] < 0).sum())
    ok = m["success"]
    return {
        "n": int(ok.size),
        "success_rate": float(ok.mean()),
        "failures": {FAILURE_REASONS[i]: int(c) for i, c in enumerate(counts) if c},
        "still_flying": still_flying,
        "mean_target_dist_m_success": float(m["target_dist_m"][ok].mean()) if ok.any() else None,
        "mean_fuel_kg_success": float(m["fuel_used_kg"][ok].mean()) if ok.any() else None,
        "max_td_vz_success": float(m["td_vz"][ok].max()) if ok.any() else None,
    }
