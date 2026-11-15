"""Classic ZEM/ZEV guidance in PyTorch (the physics backup AND the baseline the learned pilot must beat). Port of sim_jax/guidance.py: it sees
only the same noisy 13-number observation as the pilot and returns an action in [-1, 1]^3 (thrust / max total thrust)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldmodel.lander.env import EnvParams


@dataclass
class GuidanceParams:
    g: float
    start_mass: float
    max_total_thrust: float
    gate_alt: float = 30.0
    v_final: float = 1.0
    kp_h: float = 0.08
    kv_h: float = 0.8
    kv_z: float = 1.2
    switch_h: float = 8.0
    tgo_min: float = 2.0
    tgo_max: float = 240.0


def guidance_params(p: EnvParams, gate_alt: float = 30.0, v_final: float = 1.0) -> GuidanceParams:
    return GuidanceParams(p.g, p.start_mass, p.max_total_thrust, gate_alt, v_final)


def classic_action(gp: GuidanceParams, obs: torch.Tensor) -> torch.Tensor:
    r_h = obs[:, 0:2] * 1000.0
    alt = obs[:, 2] * 1000.0
    v = obs[:, 3:6] * 50.0
    mass = obs[:, 8] * gp.start_mass
    g_vec = torch.tensor([0.0, 0.0, -gp.g], device=obs.device)
    h = alt - gp.gate_alt
    tgo = (2.0 * h.clamp(min=0.0) / (-v[:, 2] + gp.v_final).clamp(min=1.0)).clamp(gp.tgo_min, gp.tgo_max)[:, None]
    r = torch.cat([r_h, h[:, None]], dim=1)
    v_f = torch.tensor([0.0, 0.0, -gp.v_final], device=obs.device)
    zem = -(r + v * tgo + 0.5 * g_vec * tgo**2)
    zev = v_f - (v + g_vec * tgo)
    a_zem = 6.0 * zem / tgo**2 - 2.0 * zev / tgo
    a_h = -gp.kp_h * r_h - gp.kv_h * v[:, :2]
    a_z = gp.kv_z * (-gp.v_final - v[:, 2])
    a_term = torch.cat([a_h, a_z[:, None]], dim=1)
    a_ctrl = torch.where((h > gp.switch_h)[:, None], a_zem, a_term - g_vec)
    return mass[:, None] * a_ctrl / gp.max_total_thrust
