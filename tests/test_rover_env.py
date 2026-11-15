"""Rover environment and physics planner on a synthetic flat world (tests only; the real worlds are built from NASA data in world.py)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from contracts.mission import load_mission
from worldmodel.rover.env import MAP_CHANNELS, N_ACTIONS, VEC_DIM, RoverEnv
from worldmodel.rover.planner import build_graph, cost_grid, cost_to_go
from worldmodel.rover.world import N_HOURS, RoverWorld, choose_targets

CFG = load_mission()
S, SC = 128, 16


def flat_world(sun: float = 1.0, slope: np.ndarray | None = None, targets=((63.5, 100.0),)) -> RoverWorld:
    sl = np.zeros((S, S), np.float32) if slope is None else slope
    return RoverWorld("fixture", "2027-01-01", 5.0, S, (0, 0), np.zeros((S, S), np.float32), sl, np.zeros((S, S), np.float32), "test", np.zeros((S, S), bool),
                      np.full((N_HOURS, SC, SC), sun, np.float16), np.ones((N_HOURS, SC, SC), np.uint8), ["t"] * N_HOURS, (63.5, 63.5),
                      np.array(targets, dtype=np.float64), "test")


def run(env, policy, steps=200):
    ice = ret = dead = 0
    for _ in range(steps):
        _, _, done, info = env.step(policy())
        ice += int(info["reached_ice"].sum())
        ret += int(info["returned"].sum())
        dead += int(info["dead"].sum())
        if bool(done.all()):
            break
    return ice, ret, dead


def test_observation_shapes_and_finiteness():
    env = RoverEnv(flat_world(), 6, seed=0)
    o = env.reset()
    assert o["map"].shape == (6, MAP_CHANNELS, CFG.rover_sim.crop_px, CFG.rover_sim.crop_px) and o["vec"].shape == (6, VEC_DIM)
    assert torch.isfinite(o["map"]).all() and torch.isfinite(o["vec"]).all() and N_ACTIONS == 9


def test_a_move_covers_speed_times_step_and_stop_stays():
    env = RoverEnv(flat_world(), 4, seed=1)
    env.reset()
    p0 = env.pos.clone()
    env.step(torch.full((4,), 2))                                   # east
    assert torch.allclose(env.pos[:, 1] - p0[:, 1], torch.full((4,), env.dist_px), atol=1e-4) and torch.allclose(env.pos[:, 0], p0[:, 0], atol=1e-4)
    p1 = env.pos.clone()
    env.step(torch.full((4,), 8))                                   # stop
    assert torch.equal(env.pos, p1)


def test_energy_balance_sunny_and_dark():
    sunny = RoverEnv(flat_world(sun=1.0), 2, seed=0)
    sunny.reset()
    sunny.battery[:] = 50.0
    sunny.step(torch.full((2,), 8))
    assert torch.allclose(sunny.battery, torch.full((2,), 50.0 + CFG.rover.solar_panel_w - CFG.rover_sim.idle_power_w))     # clamped below capacity? 98 < 100
    dark = RoverEnv(flat_world(sun=0.0), 2, seed=0)
    dark.reset()
    dark.step(torch.full((2,), 2))
    assert torch.allclose(dark.battery, torch.full((2,), CFG.rover.battery_wh - CFG.rover_sim.drive_power_w))


def test_driving_in_the_dark_kills_the_rover_away_from_the_lander_but_not_at_it():
    env = RoverEnv(flat_world(sun=0.0), 3, seed=0)
    env.reset()
    dead = 0
    for _ in range(80):                                              # drive east to the world edge, then idle there at 2 W until the battery is empty
        _, _, _, info = env.step(torch.full((3,), 2))
        dead += int(info["dead"].sum())
    assert dead == 3 and bool(env.done.all())
    at_home = RoverEnv(flat_world(sun=0.0), 2, seed=0)
    at_home.reset()
    at_home.pos[:] = at_home.lander
    at_home.battery[:] = 0.5
    _, _, _, info = at_home.step(torch.full((2,), 8))
    assert int(info["dead"].sum()) == 0                              # sheltering at the lander is not death


def test_steep_slope_blocks_the_move_and_costs_reward():
    sl = np.zeros((S, S), np.float32)
    sl[:, 65:] = 35.0                                                # wall east of the lander, over the 20 deg rover limit
    env = RoverEnv(flat_world(slope=sl), 2, seed=0)
    env.reset()
    env.pos[:] = torch.tensor([63.5, 64.0])                          # a step east would end inside the wall (slowed to a quarter speed, still inside)
    p0 = env.pos.clone()
    _, r, _, info = env.step(torch.full((2,), 2))
    assert bool(info["blocked"].all()) and torch.equal(env.pos, p0) and bool((r < 0).all())


def test_target_counts_as_reached_when_the_path_passes_within_reach():
    env = RoverEnv(flat_world(targets=((63.5, 100.0),)), 1, seed=0)
    env.reset()
    env.pos[:] = torch.tensor([63.5, 94.0])                          # 6 px from the target; a 7.2 px step overshoots it
    _, r, _, info = env.step(torch.tensor([2]))
    assert bool(info["reached_ice"][0]) and int(env.phase[0]) == 1 and float(r[0]) > CFG.scoring.w_ice_reached - 5


def test_physics_planner_completes_the_mission_in_a_sunny_flat_world():
    env = RoverEnv(flat_world(sun=1.0), 16, seed=3)
    env.reset()
    ice, ret, dead = run(env, lambda: env.planner_action(turn_back=False))
    assert ice == 16 and ret == 16 and dead == 0


def test_finished_envs_stay_finished_until_reset():
    env = RoverEnv(flat_world(), 2, seed=0)
    env.reset()
    env.done[0] = True
    p0 = env.pos[0].clone()
    env.step(torch.full((2,), 2))
    assert torch.equal(env.pos[0], p0) and bool(env.done[0])
    env.reset(torch.tensor([True, False]))
    assert not bool(env.done[0])


def test_environment_is_deterministic_for_a_seed():
    a, b = RoverEnv(flat_world(), 8, seed=7), RoverEnv(flat_world(), 8, seed=7)
    a.reset(), b.reset()
    for _ in range(5):
        oa, ra, _, _ = a.step(a.planner_action())
        ob, rb, _, _ = b.step(b.planner_action())
    assert torch.equal(oa["vec"], ob["vec"]) and torch.equal(ra, rb)


def test_cost_to_go_is_a_distance_field_and_walls_block():
    cost = cost_grid(np.zeros((40, 40)), np.zeros((40, 40)), 20.0)
    f = cost_to_go(build_graph(cost, 5.0), (40, 40), (20, 20))
    assert f[20, 20] == 0 and f[20, 30] == pytest.approx(50.0, rel=1e-6) and f[30, 30] == pytest.approx(5.0 * 10 * 2**0.5, rel=1e-6)
    wall = np.zeros((40, 40))
    wall[:, 25] = 45.0
    f2 = cost_to_go(build_graph(cost_grid(wall, np.zeros((40, 40)), 20.0), 5.0), (40, 40), (20, 20))
    assert np.isinf(f2[20, 30])


def test_targets_are_psr_edges_and_unknown_fallback_is_labelled():
    psr = np.zeros((S, S), bool)
    psr[20:60, 90:120] = True
    rng = np.random.default_rng(0)
    t, src = choose_targets(psr, np.zeros((S, S)), (63.5, 63.5), 5.0, 20.0, 8, rng)
    assert len(t) == 8 and "PSR" in src and psr[t[:, 0].astype(int), t[:, 1].astype(int)].any()
    t2, src2 = choose_targets(np.zeros((S, S), bool), np.zeros((S, S)), (63.5, 63.5), 5.0, 20.0, 8, rng)
    assert "UNKNOWN" in src2 and len(t2) == 8
