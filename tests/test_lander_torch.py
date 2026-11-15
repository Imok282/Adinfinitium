"""PyTorch lander twin: matches the JAX environment step for step, classic guidance lands on flat ground, injected failures behave."""

import numpy as np
import pytest
import torch

from contracts.mission import load_mission
from worldmodel.lander.env import OUTCOME, LanderBatch, make_params
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.terrain import TerrainT, flat_terrain

CFG = load_mission()


def _bumpy(size=96, mpp=40.0, seed=0) -> TerrainT:
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    h = gaussian_filter(rng.normal(size=(size, size)), 6) * 120.0
    gy, gx = np.gradient(h, mpp)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    haz = (slope >= 12.0).astype(np.float32)
    t = lambda a: torch.as_tensor(a.astype(np.float32))
    return TerrainT(t(h), t(slope), t(haz), t(haz), mpp, "bumpy")


def test_classic_lands_on_flat_ground():
    p = make_params(CFG, failures=False, noise=False, lookahead=False)
    env = LanderBatch(p, flat_terrain(), 32, seed=1)
    gp = guidance_params(p)
    obs = env.observe()
    for _ in range(p.n_steps):
        obs, _, _ = env.step(classic_action(gp, obs))
        if env.done.all():
            break
    ok = (env.outcome == OUTCOME["ok"]).float().mean().item()
    assert ok > 0.9, f"classic guidance landed only {ok:.0%} on flat ground"


def test_engine_out_and_blind_camera_flags_follow_probabilities():
    p = make_params(CFG, failures=True, noise=True)
    p.p_engine_out = 1.0
    p.p_camera_blind = 0.0
    env = LanderBatch(p, flat_terrain(), 64, seed=2)
    assert env.had_engine_out.all() and env.camera_ok.all() and torch.isfinite(env.engine_out_t).all()


def test_frozen_after_touchdown():
    p = make_params(CFG, failures=False, noise=False, lookahead=False)
    env = LanderBatch(p, flat_terrain(), 4, seed=3)
    env.pos[:, 2] = 0.05
    env.vel[:] = torch.tensor([0.0, 0.0, -0.5])
    _, _, jd = env.step(torch.zeros(4, 3))
    assert jd.all() and env.done.all()
    snap = env.pos.clone()
    env.step(torch.ones(4, 3))
    assert torch.equal(env.pos, snap)


def test_crop_shape_and_blind_camera():
    p = make_params(CFG, failures=False, noise=False)
    env = LanderBatch(p, _bumpy(), 5, seed=4)
    c = env.crop(32)
    assert c.shape == (5, 3, 32, 32) and torch.isfinite(c).all()
    env.camera_ok[:] = False
    assert env.crop(32).abs().sum() == 0


def test_parity_with_jax_environment():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from sim_jax import guidance as jg
    from sim_jax import lander_env as je
    from sim_jax import terrain as jt

    terr = _bumpy()
    jterr = jt.TerrainJ(jnp.asarray(terr.height.numpy()), jnp.asarray(terr.slope.numpy()), jnp.asarray(terr.hazard.numpy()), terr.mpp,
                        (terr.shape[1] - 1) / 2 * terr.mpp, (terr.shape[0] - 1) / 2 * terr.mpp)
    jp = je.make_params(CFG, jterr, failures=False, noise=False, lookahead=True)
    tp = make_params(CFG, failures=False, noise=False, lookahead=True)
    jenv = je.LanderEnv(jp)
    env = LanderBatch(tp, terr, 3, seed=5)
    gp_t, gp_j = guidance_params(tp), jg.guidance_params_from_env(jp)
    max_err = 0.0
    for i in range(3):
        key = jax.random.PRNGKey(i)
        _, s = jenv.reset(key)
        # start the torch twin from the same state
        env.pos[i], env.vel[i] = torch.tensor(np.asarray(s.pos)), torch.tensor(np.asarray(s.vel))
        env.mass[i], env.t[i], env.target[i] = float(s.mass), 0.0, torch.tensor(np.asarray(s.target))
        env.nav_bias[i], env.camera_ok[i], env.engine_out_t[i], env.outcome[i] = 0.0, True, float("inf"), -1
        env.start_mass[i] = float(s.mass)
        env.engines_ok[i] = tp.n_engines
        env.td[i] = 0.0
        env.thrust[i] = 0.0
    obs_t = env.observe()
    states = [jenv.reset(jax.random.PRNGKey(i))[1] for i in range(3)]
    for _ in range(300):
        a_t = classic_action(gp_t, obs_t)
        for i in range(3):
            obs_j_i = jenv.observe(jax.random.PRNGKey(0), states[i])
            a_j = jg.classic_policy(gp_j, obs_j_i)
            assert np.abs(np.asarray(a_j) - a_t[i].numpy()).max() < 5e-3 + 1e-3 * np.abs(np.asarray(a_j)).max()
            _, states[i], _, _, _ = jenv.step(jax.random.PRNGKey(1), states[i], jnp.asarray(a_t[i].numpy()))
        obs_t, _, _ = env.step(a_t)
    for i in range(3):
        max_err = max(max_err, float(np.abs(np.asarray(states[i].pos) - env.pos[i].numpy()).max()))
        assert int(states[i].outcome) == int(env.outcome[i]), "outcomes differ between JAX and PyTorch"
    assert max_err < 0.05, f"position drift between JAX and PyTorch: {max_err} m"
