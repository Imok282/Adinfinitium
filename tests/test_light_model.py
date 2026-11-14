"""Neural light field (KAN-24): exact physics, ray-cast teacher, coarse/fine mapping, model guarantees (needs torch for the model part)."""

import numpy as np
import pytest

from contracts.mission import load_mission
from contracts.paths import repo_root
from data_tools.make_demo_light import SUN_ANGULAR_RADIUS_DEG, sun_visible_fraction
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.light.data import TEST_SITES, TRAIN_SITES, VAL_SITES, coarsen, fine_to_coarse
from worldmodel.light.physics import light_from_skyline, skyline_at
from worldmodel.light.physics import sun_visible_fraction as sun_fraction
from worldmodel.light.teacher import n_steps, raycast_skyline_numpy, raycast_skyline_torch

CFG = load_mission()
R = CFG.moon.radius_m


def ramp(angle_deg: float, n: int = 700, mpp: float = 5.0) -> np.ndarray:
    return np.tile(np.tan(np.radians(angle_deg)) * np.arange(n) * mpp, (n, 1))      # rising to the east (+col)


def test_test_tile_is_never_trained_or_validated_on():
    assert not set(TEST_SITES) & set(TRAIN_SITES + VAL_SITES) and "Site20v2" not in TRAIN_SITES


def test_light_model_config_block():
    lm = CFG.light_model
    assert lm.n_azimuth == 360 and lm.range_steps == n_steps(lm.max_ray_range_m, 5.0 * lm.coarse_factor)  # 64 x 20 m = 1280 m
    bad = (repo_root() / "config" / "mission.yaml").read_text().replace("coarse_factor: 4", "coarse_factor: 0")
    import pathlib
    import tempfile

    from pydantic import ValidationError
    f = pathlib.Path(tempfile.mkdtemp()) / "m.yaml"
    f.write_text(bad)
    with pytest.raises(ValidationError):
        load_mission(f)


def test_skyline_interpolation_wraps_around_north():
    sky = np.zeros((1, 360))
    sky[0, 359], sky[0, 0] = 4.0, 8.0
    assert skyline_at(sky, np.array([359.5]))[0, 0] == pytest.approx(6.0)
    assert skyline_at(sky, np.array([360.0]))[0, 0] == pytest.approx(8.0)


def test_sun_fraction_matches_the_standin_generator_and_endpoints():
    el = np.linspace(-0.6, 0.6, 25)
    mine = sun_fraction(el, np.zeros_like(el), SUN_ANGULAR_RADIUS_DEG)
    assert np.allclose(mine, sun_visible_fraction(el, np.zeros_like(el)))
    assert mine[0] == 0.0 and mine[-1] == 1.0 and np.all(np.diff(mine) >= -1e-12)


def test_earth_visibility_uses_elevation_relative_to_the_skyline():
    sky = np.full((1, 360), 5.0)
    out = light_from_skyline(sky, np.array([10.0, 10.0]), np.array([6.0, 4.0]), np.array([90.0, 90.0]), np.array([5.5, 4.5]), 0.2666)
    assert out["earth_visible"][:, 0].tolist() == [True, False]
    assert out["sun_elev_rel_deg"][:, 0] == pytest.approx([1.0, -1.0])


def test_raycast_on_a_ramp_and_flat_ground_numpy():
    cells = np.array([[350.0, 350.0]])
    sky = raycast_skyline_numpy(ramp(10.0), 5.0, cells, R, 1280.0)[0]
    assert sky[90] == pytest.approx(10.0, abs=0.15) and sky[270] < -9.5 and abs(sky[0]) < 0.6
    flat = raycast_skyline_numpy(np.zeros((700, 700)), 5.0, cells, R, 1280.0)[0]
    assert flat.max() < 0.0 and flat.min() > -0.05          # only Moon curvature: a hair below the horizontal


def test_edge_cells_are_rejected_instead_of_silently_biased():
    with pytest.raises(ValueError):
        raycast_skyline_numpy(ramp(5.0, n=400), 5.0, np.array([[100.0, 200.0]]), R, 1280.0)


torch = pytest.importorskip("torch")


def test_torch_raycast_matches_numpy():
    t = make_demo_tile(size=700, n_craters=40, seed=1)
    cells = np.array([[350.0, 350.0], [300.5, 400.25], [420.0, 330.0]])
    a = raycast_skyline_numpy(t.height, 5.0, cells, R, 1280.0)
    b = raycast_skyline_torch(t.height, 5.0, cells, R, 1280.0, device=torch.device("cpu"))
    assert np.abs(a - b).max() < 0.03


def test_coarse_fine_coordinate_mapping_is_exact_on_a_plane():
    f = 4
    fine = np.tile(0.7 * np.arange(64.0), (64, 1))             # height = 0.7 * col
    coarse = coarsen(fine, f)
    c = fine_to_coarse(np.array([[10.0, 33.0]]), f)[0]          # fine col 33 -> coarse coordinate
    from scipy.ndimage import map_coordinates
    assert map_coordinates(coarse, [[c[0]], [c[1]]], order=1)[0] == pytest.approx(0.7 * 33.0, abs=1e-6)


def _toy(n_cells=8):
    from worldmodel.light.model import ray_inputs
    t = make_demo_tile(size=700, n_craters=60, seed=2)
    coarse = torch.from_numpy(coarsen(t.height, 4).astype(np.float32))[None, None]
    cells_f = np.stack([np.linspace(300, 400, n_cells), np.linspace(320, 380, n_cells)], 1)
    cells_c = torch.from_numpy(fine_to_coarse(cells_f, 4).astype(np.float32))
    x, base = ray_inputs(coarse, cells_c, 20.0, 64, 360, R)
    return t, cells_f, x, base


def test_untrained_model_equals_the_physics_backup_exactly():
    from worldmodel.light.model import SkylineNet
    _, _, x, base = _toy()
    assert torch.equal(SkylineNet()(x, base), base)


def test_backup_baseline_equals_raycast_of_the_coarse_grid():
    t, cells_f, _, base = _toy()
    coarse = coarsen(t.height, 4)
    ref = raycast_skyline_numpy(coarse, 20.0, fine_to_coarse(cells_f, 4), R, 1280.0)
    assert np.abs(base.numpy() - ref).max() < 0.05


def test_network_is_compass_shift_equivariant():
    from worldmodel.light.model import SkylineNet
    _, _, x, base = _toy(4)
    torch.manual_seed(0)
    net = SkylineNet()
    for p in net.parameters():                                   # make the zero-init head non-trivial
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    s = 37
    a = net(torch.roll(x, s, -1), torch.roll(base, s, -1))
    b = torch.roll(net(x, base), s, -1)
    assert torch.allclose(a, b, atol=1e-4)


def test_model_learns_and_checkpoint_roundtrips(tmp_path):
    from worldmodel.light.model import (
        SkylineNet,
        SkylineStudent,
        checkpoint_id,
        load_checkpoint,
        save_checkpoint,
    )
    t, cells_f, x, base = _toy(8)
    truth = torch.from_numpy(raycast_skyline_numpy(t.height, 5.0, cells_f, R, 1280.0)).float()
    torch.manual_seed(0)
    net = SkylineNet(width=8, n_down=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = float((net(x, base) - truth).abs().mean().detach())
    for _ in range(60):
        opt.zero_grad()
        (net(x, base) - truth).abs().mean().backward()
        opt.step()
    assert float((net(x, base) - truth).abs().mean().detach()) < first
    p = save_checkpoint(tmp_path / "m.pt", net, {"step": 60})
    net2, info = load_checkpoint(p)
    assert info["checkpoint_id"] == checkpoint_id(net) and torch.allclose(net2(x, base), net(x, base), atol=1e-5)
    st = SkylineStudent(net2, CFG)
    out = st.skyline(t, cells_f)
    assert out.shape == (8, 360) and np.isfinite(out).all()
    assert np.allclose(st.skyline(t, cells_f, use_model=False), base.numpy(), atol=0.05)
