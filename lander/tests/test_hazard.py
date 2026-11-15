"""Hazard detector (KAN-25): physics backup, features, labelling rules, and (if torch is installed) the U-Net."""

import math

import numpy as np
import pytest

from contracts.mission import load_mission
from contracts.paths import tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, load_tile, save_tile, slope_from_height
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.hazard.apply import apply_hazard
from worldmodel.hazard.backup import backup_hazard, label_hazard, slope_limit_deg
from worldmodel.hazard.features import N_CHANNELS, build_features, gap_fill
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, xy_to_latlon
from worldmodel.hazard.metrics import confusion

CFG = load_mission()
LIMIT = slope_limit_deg(CFG)
SIGMAS = CFG.hazard_detector.gap_fill_sigma_px


def ramp_tile(angle_deg: float, size: int = 48, mpp: float = 5.0, sparse: bool = False,
              provenance=Provenance.OBSERVED, seed: int = 0) -> Tile:
    xs = np.arange(size) * mpp
    height = np.tile(math.tan(math.radians(angle_deg)) * xs, (size, 1)).astype(np.float32)
    slope = slope_from_height(height, mpp)
    real = None
    if sparse:
        real = np.random.default_rng(seed).random((size, size)) < 0.12
        real[0, 0] = True
    return Tile(region="ramp", meters_per_pixel=mpp, height=height, slope=slope,
                hazard=np.zeros((size, size), np.float32), uncertainty=np.full((size, size), 0.4, np.float32),
                psr_mask=np.zeros((size, size), bool), real_mask=real,
                meta=Meta(provenance=provenance, source="test ramp"))


def test_holdout_region_is_never_trained_or_validated_on():
    from worldmodel.hazard.train import VAL_SITES

    assert not set(HOLDOUT_SITES) & set(TRAIN_SITES)
    assert not set(HOLDOUT_SITES) & set(VAL_SITES)
    assert "Site20v2" not in TRAIN_SITES  # extension of Site20: would leak terrain into training


def test_slope_limit_comes_from_config():
    assert LIMIT == CFG.lander.touchdown_limits.max_slope_deg == 12.0


def test_backup_follows_the_config_limit_on_a_known_ramp():
    assert backup_hazard(ramp_tile(LIMIT - 2), CFG).max() == 0.0
    assert backup_hazard(ramp_tile(LIMIT + 2), CFG).min() == 1.0
    assert label_hazard(ramp_tile(LIMIT + 2), CFG).min() == 1.0


def test_gap_fill_keeps_measured_pixels_and_recovers_a_plane():
    t = ramp_tile(8.0, size=64, sparse=True)
    filled = gap_fill(t.height, t.real_mask, SIGMAS)
    assert np.isfinite(filled).all()
    assert np.allclose(filled[t.real_mask], t.height[t.real_mask], atol=1e-3)
    assert np.median(np.abs(filled - t.height)) < 0.3  # metres, on a 8 deg ramp


def test_features_shape_range_and_hidden_interpolation():
    t = ramp_tile(8.0, sparse=True)
    f, slope = build_features(t, LIMIT, SIGMAS)
    assert f.shape == (N_CHANNELS, *t.shape) and f.dtype == np.float32 and np.isfinite(f).all()
    assert f[2].mean() == pytest.approx(t.real_mask.mean())  # laser-mask channel
    # Changing only INTERPOLATED pixels must not change the features: the model never sees them.
    t2 = ramp_tile(8.0, sparse=True)
    t2.height = np.where(t2.real_mask, t2.height, t2.height + 500.0).astype(np.float32)
    assert np.allclose(build_features(t2, LIMIT, SIGMAS)[0], f)
    assert slope.shape == t.shape


def test_confusion_metrics():
    pred = np.array([1, 1, 0, 0], bool)
    label = np.array([1, 0, 1, 0], bool)
    m = confusion(pred, label)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == 0.5 and m["missed_hazard_rate"] == 0.5


def test_polar_stereographic_to_latlon():
    lat, lon = xy_to_latlon(-500.0, 121500.0, CFG.moon.radius_m)  # centre of LOLA Site23 (Malapert)
    assert lat == pytest.approx(-86.0, abs=0.1) and (lon > 359.0 or lon < 1.0)
    assert xy_to_latlon(0.0, 0.0, CFG.moon.radius_m)[0] == pytest.approx(-90.0)
    assert xy_to_latlon(10000.0, 0.0, CFG.moon.radius_m)[1] == pytest.approx(90.0)


def test_apply_backup_keeps_generated_label_and_records_the_model(tmp_path):
    demo = make_demo_tile(size=48, n_craters=4, region="fixture")
    out = apply_hazard(demo, CFG, backup_only=True)
    assert out.meta.provenance == Provenance.GENERATED  # a fake tile is never relabelled "modelled"
    assert out.extra["hazard_model"] == "slope_rule_backup" and "backup" in out.meta.source
    assert 0.0 <= out.hazard.min() and out.hazard.max() <= 1.0
    back = load_tile(save_tile(out, tmp_path / "tiles" / "fixture.npz"))
    assert back.extra["hazard_slope_limit_deg"] == LIMIT


def test_apply_real_input_is_labelled_modelled():
    out = apply_hazard(ramp_tile(14.0, sparse=True), CFG, backup_only=True)
    assert out.meta.provenance == Provenance.MODELLED
    assert out.extra["hazard_input_provenance"] == "observed"


def test_official_slope_map_is_the_slope_of_the_heights_on_real_data():
    """Why the U-Net is trained on SPARSE input: with full-resolution input the slope rule is already ~perfect."""
    p = tile_path("lola5m_site23")
    if not p.exists():
        pytest.skip("real LOLA tile not on this machine (python -m worldmodel.hazard.lola all)")
    t = load_tile(p)
    full = slope_from_height(t.height, t.meters_per_pixel)
    assert np.corrcoef(full.ravel(), t.slope.ravel())[0, 1] > 0.99


# ------------------------------------------------------------------------------ U-Net (needs torch)
torch = pytest.importorskip("torch")


def test_unet_handles_odd_sizes_and_outputs_probabilities():
    from worldmodel.hazard.model import UNet, predict_probability

    model = UNet(base=8, depth=3)
    x = np.random.default_rng(0).normal(size=(N_CHANNELS, 50, 70)).astype(np.float32)
    p = predict_probability(model, x, torch.device("cpu"))
    assert p.shape == (50, 70) and 0.0 <= p.min() and p.max() <= 1.0


def test_unet_windowed_inference_matches_shape_on_large_input():
    from worldmodel.hazard.model import UNet, predict_probability

    model = UNet(base=8, depth=2)
    x = np.zeros((N_CHANNELS, 90, 100), np.float32)
    p = predict_probability(model, x, torch.device("cpu"), window=48, margin=8)
    assert p.shape == (90, 100) and np.isfinite(p).all()


def test_unet_learns_and_checkpoint_roundtrips(tmp_path):
    from worldmodel.hazard.model import UNet, load_checkpoint, predict_probability, save_checkpoint
    from worldmodel.hazard.train import loss_fn

    t = ramp_tile(14.0, size=32, sparse=True)
    f, _ = build_features(t, LIMIT, SIGMAS)
    x = torch.from_numpy(f)[None]
    y = torch.from_numpy(label_hazard(t, CFG))[None]
    torch.manual_seed(0)
    model = UNet(base=8, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = float(loss_fn(model(x), y).detach())
    for _ in range(40):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    assert float(loss_fn(model(x), y)) < first
    path = save_checkpoint(tmp_path / "m.pt", model, {"epoch": 1})
    model2, info = load_checkpoint(path)
    assert info["checkpoint_id"] and np.allclose(
        predict_probability(model, f, torch.device("cpu")), predict_probability(model2, f, torch.device("cpu")), atol=1e-5)
    out = apply_hazard(ramp_tile(14.0, size=32, sparse=True), CFG, checkpoint=path)
    assert out.extra["hazard_model"] == "hazard_unet" and out.extra["hazard_checkpoint_id"] == info["checkpoint_id"]
    assert out.meta.provenance == Provenance.MODELLED
