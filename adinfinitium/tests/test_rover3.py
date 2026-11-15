"""Rover v3 (planner -> executor -> gated self-learning) on synthetic worlds. Only the new logic: plan = execution, never worse than the baselines,
safe harbour, getting round an obstacle the belief does not know, and the acceptance gate."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from contracts.mission import load_mission
from worldmodel.dream.changes import apply_change
from worldmodel.rover.world import N_HOURS, RoverWorld
from worldmodel.rover3.brain import BASELINES, Brain, FastEnv, Theta, run_baseline, run_episode
from worldmodel.rover3.run import gate_decision, reseed

CFG = load_mission()
S, SC, N = 128, 16, 48


def world(sun: np.ndarray) -> RoverWorld:
    z = np.zeros((S, S), np.float32)
    return RoverWorld("fixture", "2027-01-01", 5.0, S, (0, 0), z, z.copy(), z.copy(), "test", np.zeros((S, S), bool),
                      np.broadcast_to(sun[:, None, None], (N_HOURS, SC, SC)).astype(np.float16).copy(), np.ones((N_HOURS, SC, SC), np.uint8), ["t"] * N_HOURS,
                      (63.5, 63.5), np.array([(63.5, 110.0)]), "test")


def test_plan_equals_execution_and_v3_is_never_worse_than_any_baseline():
    sun = (np.arange(N_HOURS) % 200 >= 120).astype(np.float32)                 # 120 dark hours, then 80 lit, repeating; target 232 m away (dark range 300 m)
    w = world(sun)
    real = FastEnv(w, N, None, CFG, seed=3)
    brain = Brain(w, CFG, Theta(), env=copy.copy(real))
    v3 = run_episode(reseed(real, N, 3), brain)
    assert v3["prediction_matched"] == N and v3["died"] == 0 and v3["replans"] == 0
    base = {k: run_baseline(reseed(real, N, 3), k) for k in BASELINES}
    for b in base.values():
        assert not bool((b["returned_mask"] & ~v3["returned_mask"]).any())
    assert base["naive"]["died"] > 0 and v3["returned"] > base["naive"]["returned"]


def test_safe_harbour_when_no_plan_can_return():
    w = world(np.zeros(N_HOURS, np.float32))                                    # never any light: a 464 m round trip cannot be survived
    real = FastEnv(w, N, None, CFG, seed=5)
    v3 = run_episode(reseed(real, N, 5), Brain(w, CFG, Theta(), env=copy.copy(real)))
    assert v3["kept_at_lander"] == N and v3["died"] == 0 and v3["returned"] == 0 and v3["reached_ice"] == 0
    assert run_baseline(reseed(real, N, 5), "naive")["died"] == N


def test_executor_gets_round_an_obstacle_the_belief_does_not_know():
    w = world(np.ones(N_HOURS, np.float32))
    w2, cfg2, _ = apply_change(w, CFG, "blocked_route", seed=0)
    real = FastEnv(w2, N, None, cfg2, seed=7)
    belief = FastEnv(w, 1, None, CFG, seed=0)
    v3 = run_episode(reseed(real, N, 7), Brain(w, CFG, Theta(), env=belief))
    stale = run_baseline(reseed(real, N, 7), "naive", belief_env=belief)
    assert v3["returned"] > stale["returned"] and v3["returned"] > N // 2 and v3["died"] == 0


def test_gate_keeps_the_current_brain_unless_the_gain_is_clear_and_nothing_gets_worse():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 5, 128)
    assert gate_decision(base, base + 3 + rng.normal(0, 1, 128), 10, 12, 5, 5)["accept"]
    assert not gate_decision(base, base, 10, 10, 5, 5)["accept"]                                  # no gain
    assert not gate_decision(base, base - 3, 10, 10, 5, 5)["accept"]                              # worse
    assert not gate_decision(base, base + rng.normal(0.2, 5, 128), 10, 10, 5, 5)["accept"]        # inside the noise
    assert not gate_decision(base, base + 3, 10, 9, 5, 5)["accept"]                               # better return but fewer rovers home
    assert not gate_decision(base, base + 3, 10, 10, 5, 6)["accept"]                              # better return but more rovers lost


# ---------------------------------------------------------------------------------------------- rover v3.1
def split_world() -> RoverWorld:
    """Light alternates every 100 h between the west part (lander and target) and the east part; the west is dark first. The lit east starts about 80 m
    east of the lander; the target is 267 m west of it."""
    sun = np.zeros((N_HOURS, SC, SC), np.float32)
    phase = (np.arange(N_HOURS) // 100) % 2
    sun[phase == 1, :, : SC // 2 + 2] = 1.0
    sun[phase == 0, :, SC // 2 + 2:] = 1.0
    z = np.zeros((S, S), np.float32)
    return RoverWorld("fixture", "2027-01-01", 5.0, S, (0, 0), z, z.copy(), z.copy(), "test", np.zeros((S, S), bool), sun.astype(np.float16),
                      np.ones((N_HOURS, SC, SC), np.uint8), ["t"] * N_HOURS, (63.5, 63.5), np.array([(63.5, 10.0)]), "test")


def test_v31_survival_mode_shelters_on_lit_ground_where_v3_goes_flat_at_the_lander():
    w = split_world()
    real = FastEnv(w, N, None, CFG, seed=11)
    real.reset()
    real.t[:] = 0                                                               # every rover starts at hour 0: the lander is dark for the first 100 h
    st = {k: getattr(real, k).clone() for k in ("pos", "t", "target_idx")}

    def run(theta):
        real.reset()
        for k, v in st.items():
            setattr(real, k, v.clone())
        return run_episode(real, Brain(w, CFG, theta, env=copy.copy(real)))

    v3, v31 = run(Theta()), run(Theta(v31=True))
    assert v3["returned"] == N and v3["died"] == 0 and v3["strict_lost"] == N   # v3 waits at the lander with a flat battery, then makes the trip
    assert v31["strict_returned"] == N and v31["strict_lost"] == 0 and v31["died"] == 0    # v3.1 waits on the lit side, then makes the trip
    assert v31["prediction_matched_strict"] == N


def test_v31_is_never_worse_than_v3_under_the_strict_rule_in_an_unchanged_world():
    sun = (np.arange(N_HOURS) % 200 >= 120).astype(np.float32)
    w = world(sun)
    real = FastEnv(w, N, None, CFG, seed=3)
    v3 = run_episode(reseed(real, N, 3), Brain(w, CFG, Theta(), env=copy.copy(real)))
    v31 = run_episode(reseed(real, N, 3), Brain(w, CFG, Theta(v31=True), env=copy.copy(real)))
    assert not bool((v3["strict_returned_mask"] & ~v31["strict_returned_mask"]).any())
    assert not bool((v31["strict_lost_mask"] & ~v3["strict_lost_mask"]).any())
    assert v31["died"] == 0 and v31["prediction_matched_strict"] == N


def test_v31_gate_never_accepts_a_worse_change():
    from worldmodel.rover3.run import gate_decision31
    rng = np.random.default_rng(0)
    base = torch.tensor(rng.normal(0, 5, 128))

    def ep(score, sret, slost, ret, died):
        return {"strict_score": score, "ep_return": score, "strict_returned": sret, "strict_lost": slost, "returned": ret, "died": died}

    inc = ep(base, 10, 20, 12, 3)
    assert gate_decision31(inc, ep(base + 3 + torch.tensor(rng.normal(0, 1, 128)), 12, 18, 12, 3))["accept"]
    assert not gate_decision31(inc, ep(base, 10, 20, 12, 3))["accept"]                                       # no gain
    assert not gate_decision31(inc, ep(base - 3, 10, 20, 12, 3))["accept"]                                   # worse
    assert not gate_decision31(inc, ep(base + torch.tensor(rng.normal(0.2, 5, 128)), 10, 20, 12, 3))["accept"]   # inside the noise
    assert not gate_decision31(inc, ep(base + 3, 9, 20, 12, 3))["accept"]                                    # fewer strict returns
    assert not gate_decision31(inc, ep(base + 3, 10, 21, 12, 3))["accept"]                                   # more rovers lost (strict)
    assert not gate_decision31(inc, ep(base + 3, 10, 20, 12, 4))["accept"]                                   # more simulator deaths
    assert not gate_decision31(inc, ep(base + 3, 10, 20, 11, 3))["accept"]                                   # fewer lenient returns


def test_forecast_shift_is_fitted_from_telemetry():
    from worldmodel.rover3.run import fit_shift
    sun = (np.arange(N_HOURS) % 60 >= 30).astype(np.float32)                    # 30 h dark, 30 h lit
    w = world(sun)
    w2, cfg2, _ = apply_change(w, CFG, "late_light")
    real = FastEnv(w2, N, None, cfg2, seed=5)
    rec = run_episode(reseed(real, N, 5), Brain(w, CFG, Theta(v31=True), env=FastEnv(w, 1, None, CFG, seed=0)), record=True)["record"]
    k, f, ok = fit_shift(rec)
    assert ok and k == 12 and abs(f - 1.0) < 0.05
    rec0 = run_episode(reseed(FastEnv(w, N, None, CFG, seed=5), N, 5), Brain(w, CFG, Theta(v31=True), env=FastEnv(w, 1, None, CFG, seed=0)), record=True)["record"]
    assert fit_shift(rec0)[2] is False
