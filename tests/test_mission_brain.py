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
