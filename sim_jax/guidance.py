"""Classic guidance — the physics backup AND the baseline the evolved pilot must beat.

Two phases, using ONLY the same noisy observation vector the AI pilot sees:
  1. ZEM/ZEV optimal-feedback guidance (zero-effort-miss / zero-effort-velocity) to a
     hover gate `gate_alt` metres above the target with vertical speed -v_final.
       a = 6*ZEM/tgo^2 - 2*ZEV/tgo,  ZEM = r_f - (r + v*tgo + g*tgo^2/2),  ZEV = v_f - (v + g*tgo)
     tgo is re-estimated every step from a constant-deceleration profile.
  2. From `switch_h` above the gate down: vertical descent at -v_final with a PD hold on horizontal position.
Thrust = m * a_control, returned as an action normalised by max_total_thrust
(phase 1: a_control = a_zem; phase 2: a_control = a_desired - g_vec).

Ref: Guo, Hawkins & Wie (2013), "Applications of generalized zero-effort-miss/zero-effort-
velocity feedback guidance algorithm", JGCD 36(3).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from sim_jax.lander_env import EnvParams


class GuidanceParams(NamedTuple):
    g: float
    start_mass: float
    max_total_thrust: float
    gate_alt: float = 30.0
    v_final: float = 1.0
    kp_h: float = 0.08     # terminal horizontal position gain (1/s^2)
    kv_h: float = 0.8      # terminal horizontal velocity gain (1/s)
    kv_z: float = 1.2      # terminal vertical speed gain (1/s)
    switch_h: float = 8.0  # hand over to terminal law this far above the gate (m); >> altimeter noise
    tgo_min: float = 2.0
    tgo_max: float = 240.0


def guidance_params_from_env(p: EnvParams, gate_alt: float = 30.0, v_final: float = 1.0) -> GuidanceParams:
    return GuidanceParams(g=p.g, start_mass=p.start_mass, max_total_thrust=p.max_total_thrust,
                          gate_alt=gate_alt, v_final=v_final)


def classic_policy(gp: GuidanceParams, obs, key=None):
    """policy(params, obs, key) -> action in [-1, 1]^3. Same signature as evolved pilots."""
    del key
    r_h = obs[0:2] * 1000.0               # m, estimated offset from target
    alt = obs[2] * 1000.0                 # m above terrain below
    v = obs[3:6] * 50.0                   # m/s
    mass = obs[8] * gp.start_mass
    g_vec = jnp.array([0.0, 0.0, -gp.g])

    # ---- phase 1: ZEM/ZEV to the hover gate
    h = alt - gp.gate_alt
    tgo = 2.0 * jnp.maximum(h, 0.0) / jnp.maximum(-v[2] + gp.v_final, 1.0)
    tgo = jnp.clip(tgo, gp.tgo_min, gp.tgo_max)
    r = jnp.array([r_h[0], r_h[1], h])
    r_f = jnp.zeros(3)
    v_f = jnp.array([0.0, 0.0, -gp.v_final])
    zem = r_f - (r + v * tgo + 0.5 * g_vec * tgo**2)
    zev = v_f - (v + g_vec * tgo)
    a_zem = 6.0 * zem / tgo**2 - 2.0 * zev / tgo

    # ---- phase 2: terminal vertical descent with horizontal hold
    a_h = -gp.kp_h * r_h - gp.kv_h * v[:2]
    a_z = gp.kv_z * (-gp.v_final - v[2])
    a_term = jnp.array([a_h[0], a_h[1], a_z])

    # ZEM/ZEV already accounts for gravity: a_zem IS the control (thrust) acceleration.
    # The terminal law commands total acceleration, so gravity is compensated there.
    a_ctrl = jnp.where(h > gp.switch_h, a_zem, a_term - g_vec)
    thrust = mass * a_ctrl
    return thrust / gp.max_total_thrust
