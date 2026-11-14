"""The evolved pilot: how a NEAT network's output becomes a thrust command.

Two modes (see `PilotConfig.mode`):

  "residual"  action = clip(classic_action + residual_scale * net(features), -1, 1)
              The net sees the sensor vector AND classic guidance's own command, and learns a
              *correction*. Evolution starts near the physics baseline instead of from random
              thrust, so 300 overnight generations are enough. This is what we report as the
              evolved pilot; the README/video must say "residual on classic guidance".
  "scratch"   action = net(features) in [-1, 1]. No physics prior. Much harder; kept so we can
              show (honestly) whether evolution alone can land.

Both flavours share the `policy(params, obs, key)` signature used by sim_jax.rollout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp

from sim_jax.guidance import GuidanceParams, classic_policy
from sim_jax.lander_env import ACT_DIM, LOOKAHEAD_NAMES, OBS_DIM


class PilotConfig(NamedTuple):
    mode: str = "residual"           # "residual" | "scratch"
    residual_scale: float = 0.3      # max correction, fraction of max total thrust per axis
    lookahead: bool = False          # env observes the 8-direction hazard ring (EnvParams.lookahead)

    @property
    def n_inputs(self) -> int:
        obs = OBS_DIM + (len(LOOKAHEAD_NAMES) if self.lookahead else 0)
        return obs + (ACT_DIM if self.mode == "residual" else 0)

    @property
    def n_outputs(self) -> int:
        return ACT_DIM


def features(cfg: PilotConfig, gp: GuidanceParams, obs):
    """Network input vector (all ~O(1) already: obs is normalised in lander_env.OBS_NAMES)."""
    if cfg.mode == "residual":
        return jnp.concatenate([obs, classic_policy(gp, obs)])
    return obs


def make_pilot_policy(cfg: PilotConfig, gp: GuidanceParams, net_forward: Callable) -> Callable:
    """Build policy(net_params, obs, key) -> action from a NEAT forward function.

    `net_forward(net_params, inputs) -> [3]` raw network output (tanh-squashed to [-1, 1] here).
    """
    if cfg.mode not in ("residual", "scratch"):
        raise ValueError(f"unknown pilot mode {cfg.mode!r}")

    def policy(net_params, obs, key=None):
        del key
        out = jnp.tanh(net_forward(net_params, features(cfg, gp, obs)))
        if cfg.mode == "residual":
            return jnp.clip(classic_policy(gp, obs) + cfg.residual_scale * out, -1.0, 1.0)
        return out

    return policy
