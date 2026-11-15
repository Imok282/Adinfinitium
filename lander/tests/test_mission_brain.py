"""Pilot brain + meta layer: untrained pilot == physics backup, sizes as documented, planner learns a real signal, metrics and the decision log are sane."""

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.plan import Plan
from contracts.provenance import Meta, Provenance
from worldmodel.lander.env import LanderBatch, make_params
from worldmodel.lander.fly import fly
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.pilot import build_model, pilot_features, residual_action
from worldmodel.lander.scenes import Scene
from worldmodel.lander.terrain import flat_terrain
from worldmodel.planner.data import Group
from worldmodel.planner.features import NF, navigator_score
from worldmodel.planner.metrics import brier, decision, ece
from worldmodel.planner.train import fit, predict

CFG = load_mission()


def test_untrained_pilot_flies_exactly_like_classic_guidance():
    p = make_params(CFG, failures=False, noise=False)
    gp = guidance_params(p)
    env = LanderBatch(p, flat_terrain(), 8, seed=0)
    obs = env.observe()
    for kind, kw in (("mlp", {}), ("former", {"d": 64, "layers": 2, "heads": 4})):
        m = build_model(kind, **kw)
        mu, v = m(pilot_features(obs, classic_action(gp, obs)), env.crop(32))
        assert torch.equal(mu, torch.zeros_like(mu)) and v.shape == (8,)
        assert torch.equal(residual_action(classic_action(gp, obs), mu), classic_action(gp, obs).clamp(-1, 1))
    scene = Scene(flat_terrain(), np.array([[32, 32]]), "flat")
    a = fly(scene, p, gp, None, 16, seed=3)
    b = fly(scene, p, gp, build_model("mlp"), 16, seed=3)
    assert torch.equal(a["success"], b["success"]) and torch.allclose(a["target_dist_m"], b["target_dist_m"], atol=1e-3)


def test_pilot_former_size_matches_the_docstring():
    with torch.device("meta"):
        n = sum(p.numel() for p in build_model("former", d=384, layers=8, heads=8).parameters())
    assert 12e6 < n < 20e6, n


def _groups(n_worlds=12, k=24, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_worlds):
        x = rng.normal(size=(k, NF)).astype(np.float32)
        y = (1 / (1 + np.exp(-(2.0 * x[:, 0] - 1.0 * x[:, 6])))).astype(np.float32)
        out.append(Group(f"lola5m_site{i:02d}", "2027-03-01", x, y, np.full(k, 32, np.float32), np.zeros(k, np.float32), {}))
    return out


def test_planner_learns_a_signal_and_beats_a_constant():
    g = _groups()
    for kind, kw in (("set", {"d": 32, "layers": 1, "heads": 4}), ("logistic", {})):
        model, _ = fit(kind, g[:9], g[9:], epochs=40, lr=3e-3, seed=0, **kw)
        p = np.concatenate([predict(model, x) for x in g[9:]])
        y = np.concatenate([x.y for x in g[9:]])
        assert brier(p, y) < 0.5 * brier(np.full_like(y, y.mean()), y), kind


def test_metrics_and_backup_score():
    y = np.array([0.1, 0.5, 0.9])
    d = decision(np.array([0.2, 0.4, 0.8]), y)
    assert d["regret"] == 0 and d["rank_corr"] > 0.99
    assert ece(y, y) == 0.0
    assert np.isfinite(navigator_score(np.zeros(NF, np.float32)))


def test_orchestrator_log_is_ordered_and_explains_itself():
    from worldmodel.planner.orchestrator import decision_log
    from worldmodel.rover.world import RoverWorld

    T, S = 709, 64
    sun = np.ones((T, 8, 8), np.float16)
    sun[200:400] = 0.0
    w = RoverWorld("r", "2027-03-01", 5.0, S, (0, 0), np.zeros((S, S)), np.zeros((S, S)), np.zeros((S, S)), "x", np.zeros((S, S), bool), sun,
                   np.ones((T, 8, 8), np.uint8), [f"2027-03-01T{h % 24:02d}:00:00Z" for h in range(T)], (32.0, 32.0), np.zeros((0, 2)), "x")
    plan = Plan(decision="go", region="r", landing_time_utc="2027-03-01T10:00:00Z", target_cell_rc=(32.0, 32.0), scatter_radius_m=20.0,
                return_deadline_utc="2027-03-08T00:00:00Z", p_success=0.5, backup_used=False, meta=Meta(provenance=Provenance.MODELLED, source="t"))
    rows = decision_log(plan, w, 8)
    phases = [r["phase"] for r in rows]
    assert phases[:3] == ["orbit", "descent", "touchdown"] and "night" in phases and "sunrise" in phases
    assert all(r["reason"] for r in rows)
    assert [r["t_utc"] for r in rows] == sorted(r["t_utc"] for r in rows)


def test_dream_gate_never_accepts_a_worse_planner():
    from worldmodel.planner.dream import accept

    assert accept(0.050, 0.040)
    assert not accept(0.050, 0.050) and not accept(0.050, 0.060) and not accept(0.05, float("nan"))


def test_dream_chooses_where_the_two_brains_disagree():
    from worldmodel.planner.dream import choose

    p_set = np.array([0.10, 0.50, 0.20, 0.90])
    p_log = np.array([0.10, 0.05, 0.20, 0.88])
    assert list(choose(p_set, p_log, 1, explore=0.0)) == [1]


def test_dream_validation_is_frozen_and_never_a_test_site():
    from worldmodel.planner.dream import split_val

    gs = _groups(4)
    gs[0].region, gs[1].region, gs[2].region, gs[3].region = "lola5m_site01", "lola5m_site06", "lola5m_site11", "lola5m_site23"
    train, val = split_val(gs)
    assert [g.region for g in val] == ["lola5m_site11"]
    assert "lola5m_site23" not in [g.region for g in train]


def test_pilot_dream_gate_needs_a_real_gain():
    from worldmodel.lander.dream import accept

    assert accept(0.950, 0.955, 0.002)
    assert not accept(0.950, 0.951, 0.002) and not accept(0.950, 0.940, 0.002) and not accept(0.95, float("nan"), 0.002)


def test_authority_rule_fades_the_residual_by_measured_altitude():
    from worldmodel.lander.pilot import authority

    obs = torch.zeros(4, 21)
    obs[:, 2] = torch.tensor([0.002, 0.020, 0.030, 0.500])                            # measured altitude 2, 20, 30, 500 m (obs is in km)
    assert authority(obs, None) == 1.0 and authority(obs, {}) == 1.0                  # no rule: full authority (the original behaviour)
    w = authority(obs, {"fade_lo": 10.0, "fade_hi": 30.0, "scale": 0.6})
    assert w.shape == (4, 1)
    assert torch.allclose(w[:, 0], torch.tensor([0.0, 0.3, 0.6, 0.6]), atol=1e-4)     # 0 below fade_lo, linear between, `scale` above fade_hi
    assert torch.allclose(authority(obs, {"scale": 0.6}), torch.full((4, 1), 0.6), atol=1e-4)
    classic, r = torch.rand(4, 3) * 2 - 1, torch.rand(4, 3) * 2 - 1
    assert torch.equal(residual_action(classic, r, 0.0), classic)                     # weight 0 == classic guidance
    act = residual_action(classic, r, authority(obs, {"fade_lo": 10.0, "fade_hi": 30.0}))
    assert torch.equal(act[0], classic[0]) and not torch.equal(act[3], classic[3])


def test_checkpoint_rule_is_applied_in_flight_and_the_solve_ladder_matches_it(tmp_path):
    from worldmodel.lander.analysis import fly_recorded
    from worldmodel.lander.pilot import load_checkpoint, save_checkpoint
    from worldmodel.lander.solve import fly_ladder

    p = make_params(CFG, failures=False, noise=False)                                  # no sensor noise: replays are exact
    gp = guidance_params(p)
    scene = Scene(flat_terrain(), np.array([[32, 32]]), "flat")
    torch.manual_seed(0)
    m = build_model("mlp", hidden=32)
    torch.nn.init.normal_(m.mu.weight, std=0.5)                                        # a pilot whose residual is NOT zero
    never = {"fade_lo": 1e6, "fade_hi": 2e6}                                           # hands over to classic everywhere
    full, _ = load_checkpoint(save_checkpoint(tmp_path / "a.pt", m, {}))
    off, info = load_checkpoint(save_checkpoint(tmp_path / "b.pt", m, {"residual_rule": never}))
    assert full.rule is None and off.rule == never and info["residual_rule"] == never
    n = 4
    c = fly_recorded(scene, p, gp, None, n, 5)
    a = fly_recorded(scene, p, gp, full, n, 5, scenarios=c["init"])
    b = fly_recorded(scene, p, gp, off, n, 5, scenarios=c["init"])
    assert np.array_equal(b["path"], c["path"])                                        # rule with weight 0 flies exactly like classic guidance
    assert not np.allclose(a["m"]["target_dist_m"], c["m"]["target_dist_m"], atol=1e-2)   # ... and without the rule the residual does act
    from worldmodel.lander.pilot import authority
    o = torch.zeros(3, 21)
    o[:, 2], o[1, 7], o[0, 7], o[2, 7] = 0.5, 0.75, 1.0, 1.0                           # 500 m up; lander 1 has lost an engine
    assert authority(o, {"engine_out_scale": 0.0}).flatten().tolist() == [1.0, 0.0, 1.0]   # classic flies the engine-out lander only
    assert authority(o, {"fade_lo": 5.0, "fade_hi": 15.0}).flatten().tolist() == [1.0, 1.0, 1.0]   # rules without the key are unchanged
    ladder = {"current pilot": None, "classic guidance only": "classic", "full authority": {"scale": 1.0}}
    lad = fly_ladder(scene, p, gp, off, c["init"], ladder, 5, torch.device("cpu"))    # `off` already carries a rule: "current pilot" must keep it
    assert lad["ok"].shape == (3, n)
    for row, ref in ((0, c), (1, c), (2, a)):                                          # row k of the ladder == the same missions flown one strategy at a time
        assert np.array_equal(lad["ok"][row], ref["m"]["success"].astype(bool))
        assert np.allclose(lad["miss"][row], ref["m"]["target_dist_m"], rtol=1e-3, atol=0.05) and np.allclose(lad["td_vh"][row], ref["m"]["td_vh"], atol=1e-2)
