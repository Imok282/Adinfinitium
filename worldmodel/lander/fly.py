"""Fly a batch of landers to the end with the classic guidance alone (model=None) or with a learned residual pilot (held for `repeat` sim steps)."""

from __future__ import annotations

import numpy as np
import torch

from contracts.runs import FAILURE_REASONS
from worldmodel.lander.env import EnvParams, LanderBatch
from worldmodel.lander.guidance import GuidanceParams, classic_action
from worldmodel.lander.pilot import authority, pilot_features, residual_action
from worldmodel.lander.scenes import Scene


@torch.no_grad()
def fly(scene: Scene, params: EnvParams, gp: GuidanceParams, model, n: int, seed: int, repeat: int = 5, crop: int = 32, device: torch.device | None = None,
        targets: torch.Tensor | None = None) -> dict:
    """Metrics for n landers on one scene (targets drawn from its safe cells, deterministic in `seed`). Same seed => same missions for any pilot."""
    device = device or scene.terrain.device
    rng = np.random.default_rng(seed)
    env = LanderBatch(params, scene.terrain, n, device, seed=seed, targets=targets if targets is not None else scene.sample_targets(n, rng, device))
    obs = env.observe()
    r = torch.zeros(n, 3, device=device)
    for i in range(params.n_steps):
        classic = classic_action(gp, obs)
        if model is not None:
            if i % repeat == 0:
                mu, _ = model(pilot_features(obs, classic), env.crop(crop))
                r = mu                                                    # deterministic: the mean residual
            act = residual_action(classic, r, authority(obs, getattr(model, "rule", None)))
        else:
            act = classic
        obs, _, _ = env.step(act)
        if bool(env.done.all()):
            break
    return env.metrics()


def summarize(m: dict) -> dict:
    ok = m["success"].cpu().numpy()
    out = m["outcome"].cpu().numpy()
    reasons = {FAILURE_REASONS[i]: int((out == i).sum()) for i in range(len(FAILURE_REASONS)) if (out == i).any()}
    if (out < 0).any():
        reasons["still_flying"] = int((out < 0).sum())
    dist = m["target_dist_m"].cpu().numpy()
    fuel = m["fuel_used_kg"].cpu().numpy()
    res = {"n": int(ok.size), "success_rate": float(ok.mean()), "failures": reasons,
           "mean_target_dist_m_success": float(dist[ok].mean()) if ok.any() else None, "mean_fuel_kg_success": float(fuel[ok].mean()) if ok.any() else None}
    for key in ("engine_failed", "nav_drift", "camera_blind"):
        f = m[key].cpu().numpy()
        res[f"success_when_{key}"] = float(ok[f].mean()) if f.any() else None
        res[f"n_{key}"] = int(f.sum())
    return res
