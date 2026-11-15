"""Rover v2 (imitation of the planners): teacher assignment, the test-site wall, the selection rule and the pre-registered verdict."""

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from tests.test_rover_env import CFG, S, flat_world
from worldmodel.rover2.report import verdict
from worldmodel.rover2.teach import STATE_KEYS, Buffer, TeachEnv, assert_not_test_sites, assign_teachers, better, collect, first_best, teacher_action


def test_first_best_picks_the_highest_return_and_the_first_on_ties():
    r = torch.tensor([[1.0, 5.0, 5.0], [2.0, 2.0, 2.0], [0.0, -1.0, 3.0]])
    assert first_best(r).tolist() == [1, 0, 2]


def test_teacher_assignment_restores_the_start_and_the_teacher_completes_a_sunny_flat_mission():
    env = TeachEnv(flat_world(), 8, cfg=CFG, seed=3)
    before = {k: getattr(env, k).clone() for k in STATE_KEYS}
    assign, stats = assign_teachers(env)
    assert all(torch.equal(before[k], getattr(env, k)) for k in STATE_KEYS) and not env.skip_obs
    assert stats["teacher_returned"] == 1.0 and assign.tolist() == [0] * 8          # every variant returns; ties go to the first (naive)
    assert torch.equal(teacher_action(env, assign), env.planner_action(turn_back=False))


def test_collect_stores_teacher_labels_only_for_live_rovers():
    env = TeachEnv(flat_world(), 8, cfg=CFG, seed=0)
    buf = Buffer(4096, CFG.rover_sim.crop_px)
    stats = collect(env, None, buf, torch.Generator().manual_seed(0), net_frac=0.0, keep_frac=1.0)
    assert stats["teacher_returned"] >= max(stats["variant_returned"].values()) > 0.5 and stats["net_rovers"] == 0
    assert 8 * 5 < buf.size < 8 * CFG.rover_sim.max_steps and int(buf.act[: buf.size].max()) <= 8


def test_stuck_planner_stops_are_not_stored_and_a_standing_rover_is_stored_rarely():
    slope = np.zeros((S, S), np.float32)
    slope[:, 80:90] = 60.0                                                         # a wall the rover cannot cross: the target behind it has no route
    env = TeachEnv(flat_world(slope=slope), 32, cfg=CFG, seed=0)
    buf = Buffer(8192, CFG.rover_sim.crop_px)
    stats = collect(env, None, buf, torch.Generator().manual_seed(0), net_frac=0.0, keep_frac=1.0)
    assert stats["stuck_states_not_stored"] > 32 * 20 and buf.size == 0            # every rover is stuck from the start: nothing to learn from
    env = TeachEnv(flat_world(sun=0.0, targets=((63.5, 120.0),)), 32, cfg=CFG, seed=0)
    assign = torch.full((32,), 2)                                                  # cautious + wait-for-light in the dark: waits at the lander, not stuck
    act, stuck = teacher_action(env, assign, with_stuck=True)
    assert (act == 8).all() and not stuck.any()
    buf = Buffer(8192, CFG.rover_sim.crop_px)                                     # cautious rovers turn back in the dark and stand at the lander:
    collect(env, None, buf, torch.Generator().manual_seed(0), net_frac=0.0, keep_frac=1.0)       # "stay at the lander" must remain a training example
    assert int((buf.act[: buf.size] == 8).sum()) > 0 and int((buf.act[: buf.size] < 8).sum()) > 0


@pytest.mark.parametrize("sites,regions", [(["Site01", "Site23"], []), (["Site01"], ["lola5m_site42_r0c1"]), (["Site11"], ["lola5m_site23"])])
def test_test_sites_are_refused_by_name_and_by_tile(sites, regions):
    with pytest.raises(SystemExit):
        assert_not_test_sites(sites, regions)
    assert_not_test_sites(["Site01", "Site11"], ["lola5m_site01", "lola5m_site11_r0c0"])


def test_selection_rule_orders_by_returned_then_ice_then_deaths_and_keeps_the_earlier_on_a_tie():
    a = {"returned": 0.10, "reached_ice": 0.2, "died": 0.5}
    assert better(a, None) and not better(dict(a), a)
    assert better({**a, "returned": 0.11, "died": 0.9}, a) and better({**a, "reached_ice": 0.3}, a) and better({**a, "died": 0.4}, a)
    assert not better({**a, "returned": 0.09, "reached_ice": 0.9, "died": 0.0}, a)


@pytest.mark.parametrize("returned,died,want", [(0.12, 0.80, "SUCCESS"), (0.12, 0.95, "PARTIAL"), (0.095, 0.88, "MATCH"), (0.085, 0.90, "PARTIAL"),
                                                 (0.05, 0.50, "PARTIAL"), (0.04, 0.10, "FAILURE"), (0.105, 0.50, "SUCCESS"), (0.1001, 0.5, "MATCH")])
def test_verdict_follows_the_pre_registered_thresholds(returned, died, want):
    planners = {"naive": {"returned": 0.093, "died": 0.868}, "cautious": {"returned": 0.088, "died": 0.543}, "cautious_wait_light": {"returned": 0.060, "died": 0.739}}
    assert verdict({**planners, "model": {"returned": returned, "died": died}})[0].startswith(want)


def test_preflight_refuses_a_live_stage_a_missing_freeze_a_missing_zip_and_data_inside_the_main_folder(tmp_path, monkeypatch):
    from worldmodel.rover2 import preflight

    main, here = tmp_path / "main", tmp_path / "v2"
    (main / "runs").mkdir(parents=True)
    (here / "runs").mkdir(parents=True)
    args = (main, here, here / "runs", here / "data")
    assert "still alive" in preflight.reason(*args, ps_output="77 /x/python -s -m worldmodel.terrain.train --name t\n")
    assert "still alive" in preflight.reason(*args, ps_output="78 /x/python -m worldmodel.rover2.teach --max-minutes 110\n")       # a second rover v2 run
    assert "freeze step" in preflight.reason(*args, ps_output="")
    assert "inside the main run folder" in preflight.reason(main, here, main / "runs" / "_rover2", here / "data", ps_output="")
    monkeypatch.setattr(preflight, "refusal_reason", lambda *a, **k: None)          # the dream guard's own checks pass: the two rover v2 checks remain
    assert "inside the main run folder" in preflight.reason(main, here, here / "runs", main / "data", ps_output="")
    assert preflight.RESULTS_ZIP in preflight.reason(*args, ps_output="")
    (main / preflight.RESULTS_ZIP).write_bytes(b"zip")
    assert preflight.reason(*args, ps_output="") is None


def test_report_gives_no_verdict_name_to_an_incomplete_evaluation():
    from worldmodel.rover2.report import build

    row = {"reached_ice": 0.2, "returned": 0.2, "died": 0.1, "mean_return": 1.0}
    ev = {"summary": {"naive": {**row, "returned": 0.09}, "model": row}, "worlds": {"a": {}}, "windows_planned": 8, "complete": False}
    text = build(ev, None, None, [], "none", [])
    assert "verdict: NO RESULT (held-out evaluation incomplete: 1 of 8 worlds)" in text and "verdict: SUCCESS" not in text
    assert "verdict: SUCCESS" in build({**ev, "complete": True}, None, None, [], "none", [])
