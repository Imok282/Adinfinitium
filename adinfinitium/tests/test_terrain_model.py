"""Terrain generator (KAN-26): data sampling, physics backup, diffusion pieces, GENERATED labelling, hide-and-check plumbing."""

import numpy as np
import pytest

from contracts.mission import load_mission
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, load_tile, save_tile
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.terrain.backup import backup_versions, gap_fill
from worldmodel.terrain.data import (
    MIN_MEASURED,
    PATCH,
    TEST_SITES,
    TRAIN_SITES,
    VAL_SITES,
    normalise,
    sample_patch_origins,
)
from worldmodel.terrain.evaluate import band_stats
from worldmodel.terrain.generate import generate_versions, to_tiles

CFG = load_mission()
SIGMAS = CFG.hazard_detector.gap_fill_sigma_px


def laser_tile(size: int = 160, coverage: float = 0.1, seed: int = 0) -> Tile:
    t = make_demo_tile(size=size, n_craters=8, seed=seed, region="fixture")
    t.real_mask = np.random.default_rng(seed).random(t.shape) < coverage
    t.meta = Meta(provenance=Provenance.OBSERVED, source="test laser tile")
    return t


def test_test_tile_never_in_training_or_validation():
    assert not set(TEST_SITES) & set(TRAIN_SITES + VAL_SITES) and "Site20v2" not in TRAIN_SITES


def test_patch_origins_always_contain_enough_laser_pixels():
    m = laser_tile(300, 0.08).real_mask
    o = sample_patch_origins(m, 50, np.random.default_rng(0))
    assert o.shape == (50, 2) and all(m[r:r + PATCH, c:c + PATCH].sum() >= MIN_MEASURED for r, c in o)


def test_normalise_uses_only_the_known_pixels():
    h = np.arange(100.0).reshape(10, 10)
    known = np.zeros((10, 10), bool)
    known[0, :5] = True
    mu, sd = normalise(h, known)
    assert mu == pytest.approx(2.0) and sd == pytest.approx(np.std([0, 1, 2, 3, 4]))     # 1.41 m, above the 1 m floor


def test_gap_fill_keeps_measurements_and_backup_noise_matches_the_error_map():
    t = laser_tile()
    g = gap_fill(t.height, t.real_mask, SIGMAS)
    assert np.allclose(g[t.real_mask], t.height[t.real_mask], atol=1e-3) and np.isfinite(g).all()
    v = backup_versions(t.height, t.real_mask, np.full(t.shape, 0.5), SIGMAS, 400, np.random.default_rng(0))
    assert v.shape == (400, *t.shape) and v.std(axis=0).mean() == pytest.approx(0.5, rel=0.05)


def test_band_stats_on_perfect_and_biased_versions():
    real = np.linspace(0, 10, 200)
    ok = band_stats(real, np.stack([real - 0.1, real, real + 0.1]))
    assert ok["rmse_m"] < 0.01 and ok["coverage90_pct"] == 100.0
    bad = band_stats(real, np.stack([real + 5, real + 5.1, real + 5.2]))
    assert bad["coverage90_pct"] == 0.0 and bad["rmse_m"] > 4.9


def test_backup_generation_is_generated_and_keeps_measured_pixels_exactly(tmp_path):
    t = laser_tile()
    v = generate_versions(t, CFG, 3, model=None)
    assert v.shape == (3, *t.shape) and np.array_equal(v[0][t.real_mask], t.height[t.real_mask].astype(np.float64))
    assert not np.allclose(v[0][~t.real_mask], v[1][~t.real_mask])                    # versions differ where nothing was measured
    out = to_tiles(t, v, CFG, "physics backup")
    assert all(o.meta.provenance == Provenance.GENERATED and "version" in o.extra for o in out)
    back = load_tile(save_tile(out[0], tmp_path / "tiles" / "fixture_gen0.npz"))
    assert back.meta.provenance == Provenance.GENERATED and back.region == "fixture_gen0"


torch = pytest.importorskip("torch")


def test_unet_shapes_and_schedule():
    from worldmodel.terrain.model import DenoiseUNet, cosine_alpha_bar, q_sample
    m = DenoiseUNet(base=8, depth=2, T=20)
    x = torch.randn(3, 1, 32, 32)
    assert m(x, x * 0, (x * 0), torch.tensor([1, 5, 20])).shape == (3, 1, 32, 32)
    ab = cosine_alpha_bar(20)
    assert ab[0] > 0.999 and ab[-1] < 0.01 and torch.all(ab[1:] < ab[:-1])
    assert torch.allclose(q_sample(x, torch.tensor([0, 0, 0]), torch.randn_like(x), ab), x, atol=0.02)


def test_sampler_never_changes_known_pixels():
    from worldmodel.terrain.model import DenoiseUNet, sample
    torch.manual_seed(0)
    m = DenoiseUNet(base=8, depth=2, T=20).eval()
    kv = torch.randn(2, 1, 32, 32)
    km = (torch.rand(2, 1, 32, 32) < 0.1).float()
    out = sample(m, kv * km, km, steps=4)
    assert torch.isfinite(out).all() and torch.equal(out * km, kv * km)


def test_training_step_lowers_hidden_pixel_loss_and_checkpoint_roundtrips(tmp_path):
    from worldmodel.terrain.model import (
        DenoiseUNet,
        cosine_alpha_bar,
        load_checkpoint,
        q_sample,
        save_checkpoint,
    )
    torch.manual_seed(0)
    m = DenoiseUNet(base=8, depth=2, T=20)
    ab = cosine_alpha_bar(20)
    x0 = torch.randn(16, 1, 32, 32).cumsum(2).cumsum(3) * 0.02          # smooth-ish terrain
    meas = (torch.rand_like(x0) < 0.3).float()
    C = meas * (torch.rand_like(x0) < 0.5).float()
    Hm = meas - C
    t = torch.randint(1, 21, (16,))
    xt = q_sample(x0, t, torch.randn_like(x0), ab)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss = lambda: (((m(xt, x0 * C, C, t) - x0) ** 2 * Hm).sum() / Hm.sum())
    first = float(loss().detach())
    for _ in range(40):
        opt.zero_grad()
        loss().backward()
        opt.step()
    assert float(loss().detach()) < first
    p = save_checkpoint(tmp_path / "m.pt", m, {"step": 40})
    m2, info = load_checkpoint(p)
    assert info["checkpoint_id"] and torch.allclose(m2(xt, x0 * C, C, t), m(xt, x0 * C, C, t), atol=1e-5)


def test_model_generation_and_hide_and_check_plumbing():
    from worldmodel.terrain.evaluate import hide_and_check
    from worldmodel.terrain.model import DenoiseUNet
    torch.manual_seed(0)
    t = laser_tile(192, 0.12)
    net = DenoiseUNet(base=8, depth=2, T=20).eval()
    v = generate_versions(t, CFG, 2, net, torch.device("cpu"), steps=2, stride=64)
    assert v.shape == (2, *t.shape) and np.isfinite(v).all() and np.array_equal(v[1][t.real_mask], t.height[t.real_mask].astype(np.float64))
    res = hide_and_check(net, t, CFG, torch.device("cpu"), n_patches=4, n_versions=3, steps=2, seed=0)
    assert set(res["model"]) == set(res["backup"]) == {"rmse_m", "mae_m", "coverage90_pct", "width90_m"} and res["n_hidden_pixels"] > 0
