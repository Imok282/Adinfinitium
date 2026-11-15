"""Swarm tests. The pure-JAX parts run everywhere; the TensorNEAT end-to-end test is skipped
unless the "swarm" extra is installed (tensorneat is a GitHub-only package)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contracts.mission import load_mission
from sim_jax.guidance import classic_policy, guidance_params_from_env
from sim_jax.lander_env import OBS_DIM, OUTCOME, LanderEnv, make_params
from sim_jax.rollout import rollout
from sim_jax.terrain import flat_terrain
from swarm.fitness import shaped_fitness
from swarm.pilot import PilotConfig, make_pilot_policy

CFG = load_mission()


@pytest.fixture(scope="module")
def flat():
    p = make_params(CFG, flat_terrain(101, 40.0), failures=False, noise=False)
    return LanderEnv(p), guidance_params_from_env(p, 30.0, 1.0)


def test_residual_pilot_with_zero_net_equals_classic(flat):
    env, gp = flat
    obs, _ = env.reset(jax.random.PRNGKey(0))
    pol = make_pilot_policy(PilotConfig("residual", 0.3), gp, lambda p, x: jnp.zeros(3))
    np.testing.assert_allclose(pol(None, obs, None), jnp.clip(classic_policy(gp, obs), -1, 1), atol=1e-6)
    assert PilotConfig("residual").n_inputs == OBS_DIM + 3 and PilotConfig("scratch").n_inputs == OBS_DIM


def test_scratch_pilot_output_is_bounded(flat):
    env, gp = flat
    obs, _ = env.reset(jax.random.PRNGKey(0))
    pol = make_pilot_policy(PilotConfig("scratch"), gp, lambda p, x: 50.0 * jnp.ones(3))
    assert float(jnp.abs(pol(None, obs, None)).max()) <= 1.0


def test_shaped_fitness_orders_outcomes(flat):
    """success > gentle crash > hard crash > never landed; successes are > 0, failures <= 0."""
    env, gp = flat
    _, m, _ = rollout(env, classic_policy, gp, jax.random.PRNGKey(1), 32)
    fit, mission = shaped_fitness(m, CFG)
    ok = np.asarray(m["success"])
    assert ok.any() and (np.asarray(fit)[ok] > 0).all()
    np.testing.assert_allclose(np.asarray(fit)[ok], np.asarray(mission)[ok])

    zero = lambda p, o, k: jnp.zeros(3)  # free fall: hard_vertical
    _, m0, _ = rollout(env, zero, None, jax.random.PRNGKey(1), 8)
    f0, mission0 = shaped_fitness(m0, CFG)
    assert (np.asarray(f0) < 0).all() and (np.asarray(mission0) == 0).all()

    fake = {k: jnp.asarray(v) for k, v in {
        "outcome": [OUTCOME["timeout"], OUTCOME["hard_vertical"], OUTCOME["hazard_cell"]],
        "success": [False] * 3, "target_dist_m": [0.0] * 3, "fuel_used_kg": [0.0] * 3,
        "time_s": [240.0] * 3, "td_vz": [0.0, 10.0, 1.0], "td_vh": [0.0] * 3, "td_slope": [0.0] * 3,
    }.items()}
    f, _ = shaped_fitness(fake, CFG)
    f = np.asarray(f)
    assert f[2] > f[1] > f[0]     # gentle hazard landing > hard impact > never landed


def test_swarm_end_to_end_writes_valid_contract_files(tmp_path, monkeypatch):
    pytest.importorskip("tensorneat")
    monkeypatch.setenv("ADI_RUNS", str(tmp_path))
    from contracts.runs import read_champion, read_generation
    from swarm.run_swarm import main

    main(["--pop", "8", "--generations", "1", "--missions", "1", "--eval-missions", "8", "--topk", "2", "--val-missions", "4",
          "--flat", "--run", "t", "--max-nodes", "24", "--max-conns", "64"])
    g = read_generation(tmp_path / "t" / "generation_000.json")
    assert g.population == 8 and 0 <= g.success_rate <= 1 and g.classic_success_rate is not None
    assert g.meta.provenance.value == "modelled" and g.strategy["pilot_mode"] == "residual"
    c = read_champion(tmp_path / "t" / "champion_000.json")
    assert c.controls and c.controls[0].t_s == pytest.approx(0.1)
    assert len(c.initial_state["pos_m"]) == 3
    assert (tmp_path / "t" / "genome_000.npz").exists()


@pytest.mark.slow
def test_resume_continues_a_crashed_run_identically(tmp_path, monkeypatch):
    """1 generation, 'crash', --resume to 2  ==  2 generations straight through."""
    pytest.importorskip("tensorneat")
    import json

    from swarm.run_swarm import main

    base = ["--pop", "8", "--missions", "1", "--eval-missions", "8", "--topk", "2", "--val-missions", "4",
            "--flat", "--max-nodes", "24", "--max-conns", "64"]
    monkeypatch.setenv("ADI_RUNS", str(tmp_path))
    main([*base, "--generations", "2", "--run", "straight"])
    main([*base, "--generations", "1", "--run", "crashed"])
    first = (tmp_path / "crashed" / "generation_000.json").read_text()

    with pytest.raises(SystemExit, match="--resume"):                 # won't silently clobber a run
        main([*base, "--generations", "2", "--run", "crashed"])
    with pytest.raises(SystemExit, match="differ"):                   # can't resume with other settings
        main([*base[:1], "16", *base[2:], "--generations", "2", "--run", "crashed", "--resume"])

    main([*base, "--generations", "2", "--run", "crashed", "--resume"])
    assert (tmp_path / "crashed" / "generation_000.json").read_text() == first    # history untouched
    a = json.loads((tmp_path / "straight" / "generation_001.json").read_text())
    b = json.loads((tmp_path / "crashed" / "generation_001.json").read_text())
    for k in ("champion_genome", "success_rate", "classic_success_rate", "population_success_rate"):
        assert a[k] == b[k], k

    main([*base, "--generations", "2", "--run", "fresh", "--resume"])  # no checkpoint -> starts fresh
    assert (tmp_path / "fresh" / "generation_001.json").exists()


def test_exam_suites_train_on_different_terrain_than_held_out():
    """Training exams use the same stress on tiles seeded +1000, so the exam score isn't memorised."""
    pytest.importorskip("tensorneat")
    from swarm.run_swarm import TRAIN_SEED_SHIFT, exam_suites
    train, held = exam_suites(CFG, ["tight_fuel"], n_targets=1)
    assert TRAIN_SEED_SHIFT >= 1000 and len(train) == len(held) == 1
    (tr_env, _), (name, ho_env, _) = train[0], held[0]
    assert name == "tight_fuel"
    assert tr_env.params.start_mass == ho_env.params.start_mass == 820.0      # same stress
    assert not np.array_equal(np.asarray(tr_env.params.terrain.height),       # different terrain
                              np.asarray(ho_env.params.terrain.height))


def test_exam_report_pools_per_exam_then_averages():
    pytest.importorskip("tensorneat")
    from swarm.run_swarm import exam_report
    held = [("a", None, None), ("a", None, None), ("b", None, None)]
    champ = np.array([[1, 1], [0, 0], [1, 0]], bool)      # a: 2/4, b: 1/2
    classic = np.array([[0, 0], [0, 0], [1, 1]], bool)    # a: 0,   b: 1
    r = exam_report(held, champ, classic)
    assert r["exam_breakdown"]["a"] == {"champion": 0.5, "classic": 0.0}
    assert r["exam_breakdown"]["b"] == {"champion": 0.5, "classic": 1.0}
    assert r["exam_success_rate"] == 0.5 and r["classic_exam_success_rate"] == 0.5
    r2 = exam_report(held, champ, classic, tuned_ok=np.array([[1, 1], [1, 1], [0, 0]], bool))
    assert r2["exam_breakdown"]["a"]["classic_tuned"] == 1.0 and r2["tuned_classic_exam_success_rate"] == 0.5


def test_exam_envs_follow_the_pilot_lookahead_setting():
    pytest.importorskip("tensorneat")
    from swarm.run_swarm import exam_suites
    for flag, dim in ((False, OBS_DIM), (True, OBS_DIM + 8)):
        train, held = exam_suites(CFG, ["nav_drift_near_hazard"], 1, lookahead=flag)
        assert train[0][0].params.obs_dim == held[0][1].params.obs_dim == dim
