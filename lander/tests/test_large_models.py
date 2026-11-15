"""Large architectures (attention / transformer): presets, shapes, the 'untrained == physics baseline' guarantee, learning, kind-aware checkpoints.

Big presets are counted on the 'meta' device (no memory), tiny versions run for real so the tests stay fast on a laptop.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldmodel.hazard import model as hz
from worldmodel.light import model as lt
from worldmodel.nn_blocks import ResBlock, SelfAttention2d, TransformerStack, count_parameters, sincos_1d
from worldmodel.rover import policy as rv
from worldmodel.terrain import model as tr


def params_m(make) -> float:
    with torch.device("meta"):
        return count_parameters(make()) / 1e6


def test_large_presets_hit_their_target_sizes():
    assert 55 <= params_m(lambda: hz.AttnUNet(base=72)) <= 65            # hazard attention U-Net ~60 M
    assert 35 <= params_m(lambda: lt.SkylineFormer(64, 3, 512, 12, 8)) <= 42        # light skyline transformer ~38 M
    assert 105 <= params_m(lambda: tr.DenoiseAttnUNet(base=192)) <= 120             # terrain diffusion ~112 M
    assert 28 <= params_m(lambda: rv.DriverFormer(512, 10, 8)) <= 36                # rover vision transformer ~33 M


def test_building_blocks_attention_starts_as_identity_and_shapes():
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(SelfAttention2d(32, 4)(x), x)                              # zero-initialised projection: a safe residual
    assert ResBlock(32, 64, 16)(x, torch.randn(2, 16)).shape == (2, 64, 8, 8)
    assert TransformerStack(32, 2, 4)(torch.randn(2, 5, 32)).shape == (2, 5, 32)
    pe = sincos_1d(360, 64, period=360.0)
    assert pe.shape == (360, 64) and torch.allclose(pe[0], sincos_1d(360, 64, period=360.0)[0])


def test_hazard_attention_unet_forward_any_size_and_checkpoint_roundtrip(tmp_path):
    m = hz.AttnUNet(in_ch=5, base=16, depth=3, blocks=1, heads=4, attn_levels=1)
    f = np.random.default_rng(0).normal(size=(5, 50, 70)).astype(np.float32)
    p = hz.predict_probability(m, f, torch.device("cpu"))
    assert p.shape == (50, 70) and 0.0 <= p.min() and p.max() <= 1.0
    path = hz.save_checkpoint(tmp_path / "h.pt", m, {"epoch": 1})
    m2, info = hz.load_checkpoint(path)
    assert isinstance(m2, hz.AttnUNet) and info["checkpoint_id"] and np.allclose(p, hz.predict_probability(m2, f, torch.device("cpu")), atol=1e-5)
    old = hz.UNet(5, base=8, depth=2)                                                # an old-style checkpoint has no 'kind'
    path2 = hz.save_checkpoint(tmp_path / "u.pt", old, {})
    state = torch.load(path2, weights_only=False)
    state["arch"].pop("kind")
    torch.save(state, path2)
    assert isinstance(hz.load_checkpoint(path2)[0], hz.UNet)


def test_light_former_equals_the_physics_baseline_until_trained_and_then_learns(tmp_path):
    m = lt.SkylineFormer(width=8, n_down=3, d=32, layers=2, heads=4)
    x, base = torch.randn(4, 2, 64, 360), torch.randn(4, 360)
    assert torch.equal(m(x, base), base)                                             # the guarantee survives the big architecture
    target = base + 0.5 * torch.sin(torch.arange(360) * 0.1)[None]
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = float((m(x, base) - target).abs().mean())
    for _ in range(40):
        opt.zero_grad()
        (m(x, base) - target).abs().mean().backward()
        opt.step()
    assert float((m(x, base) - target).abs().mean()) < first
    m2, _ = lt.load_checkpoint(lt.save_checkpoint(tmp_path / "l.pt", m, {"step": 1}))
    assert isinstance(m2, lt.SkylineFormer) and torch.allclose(m2(x, base), m(x, base), atol=1e-5)


def test_light_checkpoints_without_kind_still_load_as_the_conv_net(tmp_path):
    net = lt.SkylineNet(8, 2)
    p = lt.save_checkpoint(tmp_path / "c.pt", net, {})
    state = torch.load(p, weights_only=False)
    state["arch"] = {"width": 8, "n_down": 2}
    torch.save(state, p)
    assert isinstance(lt.load_checkpoint(p)[0], lt.SkylineNet)


def test_terrain_attention_denoiser_runs_samples_and_keeps_known_pixels(tmp_path):
    m = tr.DenoiseAttnUNet(base=16, depth=2, T=20, blocks=1, heads=4, attn_levels=1).eval()
    kv = torch.randn(2, 1, 32, 32)
    km = (torch.rand(2, 1, 32, 32) < 0.1).float()
    out = tr.sample(m, kv * km, km, steps=3)
    assert torch.isfinite(out).all() and torch.equal(out * km, kv * km)
    m2, _ = tr.load_checkpoint(tr.save_checkpoint(tmp_path / "t.pt", m, {}))
    assert isinstance(m2, tr.DenoiseAttnUNet)


def test_rover_vision_transformer_policy_and_checkpoint(tmp_path):
    m = rv.DriverFormer(d=64, layers=2, heads=4, stem=16)
    obs = {"map": torch.randn(5, 6, 64, 64), "vec": torch.randn(5, 18)}
    logits, value = m(obs)
    assert logits.shape == (5, 9) and value.shape == (5,)
    assert float(logits.abs().max()) < 1.0                                           # near-uniform start (small policy-head init) for PPO
    m2, _ = rv.load_checkpoint(rv.save_checkpoint(tmp_path / "r.pt", m, {}))
    assert isinstance(m2, rv.DriverFormer) and torch.allclose(m2(obs)[0], logits, atol=1e-5)
