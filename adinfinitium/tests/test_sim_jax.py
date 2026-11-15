import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contracts.mission import load_mission
from data_tools.make_demo_tile import make_demo_tile
from sim_jax.guidance import classic_policy, guidance_params_from_env
from sim_jax.lander_env import ACT_DIM, OBS_DIM, OUTCOME, LanderEnv, make_params
from sim_jax.rollout import landing_score, rollout, summarize
from sim_jax.terrain import flat_terrain, height_at, terrain_from_tile

CFG = load_mission()


@pytest.fixture(scope="module")
def flat_env():
    return LanderEnv(make_params(CFG, flat_terrain(101, 40.0), failures=False, noise=False))


def test_reset_and_step_shapes(flat_env):
    obs, s = flat_env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (OBS_DIM,)
    obs2, s2, _r, done, _ = flat_env.step(jax.random.PRNGKey(1), s, jnp.zeros(ACT_DIM))
    assert obs2.shape == (OBS_DIM,) and not bool(done)
    assert float(s2.mass) == pytest.approx(float(s.mass))  # zero command -> engines off, no fuel used


def test_free_fall_crashes(flat_env):
    zero = lambda p, o, k: jnp.zeros(ACT_DIM)
    _, m, _ = rollout(flat_env, zero, None, jax.random.PRNGKey(0), 16)
    assert not bool(m["success"].any())
    assert set(np.asarray(m["outcome"]).tolist()) == {OUTCOME["hard_vertical"]}


def test_thrust_limits(flat_env):
    _, s = flat_env.reset(jax.random.PRNGKey(0))
    p = flat_env.params
    big = flat_env.apply_limits(s, jnp.array([0.0, 0.0, 5.0]))
    assert float(jnp.linalg.norm(big)) == pytest.approx(p.max_total_thrust, rel=1e-5)
    sideways = flat_env.apply_limits(s, jnp.array([1.0, 0.0, 0.0]))
    tilt = np.degrees(np.arccos(float(sideways[2] / jnp.linalg.norm(sideways))))
    assert tilt == pytest.approx(np.degrees(p.max_tilt_rad), abs=1e-3)
    tiny = flat_env.apply_limits(s, jnp.array([0.0, 0.0, 1e-4]))
    assert float(jnp.linalg.norm(tiny)) == 0.0
    small = flat_env.apply_limits(s, jnp.array([0.0, 0.0, 0.05]))   # 200 N < min throttle floor 320 N
    assert float(jnp.linalg.norm(small)) == pytest.approx(p.min_throttle * p.engine_max_n, rel=1e-5)


def test_touchdown_limits_are_enforced(flat_env):
    _, s = flat_env.reset(jax.random.PRNGKey(0))
    p = flat_env.params
    hover = jnp.array([0.0, 0.0, float(s.mass) * p.g / p.max_total_thrust])
    cases = {
        (0.0, 0.0, -1.5): OUTCOME["ok"],
        (0.0, 0.0, -2.5): OUTCOME["hard_vertical"],
        (0.8, 0.0, -1.0): OUTCOME["hard_horizontal"],
    }
    for vel, expected in cases.items():
        st = s._replace(pos=jnp.array([0.0, 0.0, 0.05]), vel=jnp.array(vel, jnp.float32))
        _, st2, _, done, _ = flat_env.step(jax.random.PRNGKey(2), st, hover)
        assert bool(done) and int(st2.outcome) == expected, (vel, int(st2.outcome))


def test_steep_slope_is_rejected():
    t = make_demo_tile(size=64, mpp=5.0, seed=0, n_craters=0)
    t.slope[:] = 20.0
    env = LanderEnv(make_params(CFG, terrain_from_tile(t), failures=False, noise=False))
    _, s = env.reset(jax.random.PRNGKey(0))
    g = float(height_at(env.params.terrain, 0.0, 0.0))
    st = s._replace(pos=jnp.array([0.0, 0.0, g + 0.05]), vel=jnp.array([0.0, 0.0, -1.0]))
    _, st2, _, _, _ = env.step(jax.random.PRNGKey(1), st, jnp.zeros(3))
    assert int(st2.outcome) == OUTCOME["steep_slope"]


def test_classic_guidance_lands_on_flat_terrain():
    env = LanderEnv(make_params(CFG, flat_terrain(101, 40.0), failures=False, noise=True))
    gp = guidance_params_from_env(env.params)
    _, m, _ = rollout(env, classic_policy, gp, jax.random.PRNGKey(3), 256)
    s = summarize(m)
    assert s["success_rate"] >= 0.97, s
    assert s["max_td_vz_success"] <= CFG.lander.touchdown_limits.max_vertical_speed_m_s
    assert s["mean_fuel_kg_success"] < CFG.scenarios.terminal.start_propellant_kg


def test_classic_guidance_with_failures_on_crater_tile():
    env = LanderEnv(make_params(CFG, terrain_from_tile(make_demo_tile(seed=0)), failures=True))
    gp = guidance_params_from_env(env.params)
    _, m, _ = rollout(env, classic_policy, gp, jax.random.PRNGKey(4), 256)
    assert summarize(m)["success_rate"] >= 0.95


def test_landing_score_gate():
    m = {"success": jnp.array([True, False]), "target_dist_m": jnp.array([10.0, 0.0]),
         "fuel_used_kg": jnp.array([50.0, 10.0])}
    sc = landing_score(m)
    assert float(sc[0]) == pytest.approx(100 - 0.5 - 10)
    assert float(sc[1]) == 0.0


def test_same_seed_same_result():
    env = LanderEnv(make_params(CFG, flat_terrain(101, 40.0), failures=True, noise=True))
    gp = guidance_params_from_env(env.params)
    _, a, _ = rollout(env, classic_policy, gp, jax.random.PRNGKey(7), 32)
    _, b, _ = rollout(env, classic_policy, gp, jax.random.PRNGKey(7), 32)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_mass_flow_matches_rocket_equation(flat_env):
    _, s = flat_env.reset(jax.random.PRNGKey(0))
    p = flat_env.params
    s = s._replace(pos=jnp.array([0.0, 0.0, 5000.0]))
    full = jnp.array([0.0, 0.0, 1.0])
    _, s2, _, _, _ = flat_env.step(jax.random.PRNGKey(1), s, full)
    expected = p.max_total_thrust / (p.isp * p.g0) * p.dt
    assert float(s.mass - s2.mass) == pytest.approx(expected, rel=1e-3)  # float32 mass ~900 kg


# ---- opt-in hazard look-ahead (KAN-100) ------------------------------------------------------
def _ring_env(lookahead, camera_ok=True):
    import dataclasses  # noqa: F401
    t = flat_terrain(101, 10.0)
    haz = np.zeros((101, 101), np.float32)
    haz[50, 56:61] = 1.0        # hazard 60..100 m EAST of the centre (row 50 = y 0)
    t = t._replace(hazard=jnp.asarray(haz))
    p = make_params(CFG, t, failures=False, noise=False, lookahead=lookahead, lookahead_m=70.0)
    return LanderEnv(p)


def test_lookahead_off_keeps_observation_identical():
    off, on = _ring_env(False), _ring_env(True)
    key = jax.random.PRNGKey(3)
    (o0, _), (o1, _) = off.reset(key), on.reset(key)
    assert o0.shape == (OBS_DIM,) and o1.shape == (OBS_DIM + 8,) and on.params.obs_dim == OBS_DIM + 8
    np.testing.assert_array_equal(np.asarray(o1[:OBS_DIM]), np.asarray(o0))   # indices 0..12 unchanged


def test_lookahead_ring_points_at_the_hazard_and_respects_blind_camera():
    env = _ring_env(True)
    _, s = env.reset(jax.random.PRNGKey(0))
    s = s._replace(pos=s.pos.at[0].set(0.0).at[1].set(0.0))
    ring = np.asarray(env.lookahead_ring(s))
    assert ring[2] == 1.0 and ring.sum() == 1.0        # only "e" (index 2) sees the hazard
    blind = np.asarray(env.lookahead_ring(s._replace(camera_ok=jnp.asarray(False))))
    assert blind.sum() == 0.0


def test_classic_guidance_ignores_the_look_ahead():
    off, on = _ring_env(False), _ring_env(True)
    gp = guidance_params_from_env(off.params, 30.0, 1.0)
    key = jax.random.PRNGKey(5)
    _, m0, _ = rollout(off, classic_policy, gp, key, 16)
    _, m1, _ = rollout(on, classic_policy, gp, key, 16)
    np.testing.assert_array_equal(np.asarray(m0["outcome"]), np.asarray(m1["outcome"]))
