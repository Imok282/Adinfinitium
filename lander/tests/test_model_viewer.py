"""3D model viewer (trial harness): scene functions and a full app smoke test on a synthetic fixture tile (GENERATED, tests only)."""

import numpy as np
import pytest

from contracts.paths import repo_root, tile_path
from contracts.tiles import save_tile
from data_tools.make_demo_tile import make_demo_tile
from model_viewer import scene as sc


@pytest.fixture
def viewer_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    t = make_demo_tile(size=700, n_craters=40, seed=5, region="lola5m_site99")
    t.real_mask = np.random.default_rng(0).random(t.shape) < 0.1
    save_tile(t, tile_path("lola5m_site99"))
    return t


def test_available_regions_skips_generated_versions(viewer_data, tmp_path):
    save_tile(viewer_data, tmp_path / "tiles" / "lola5m_site99_gen0.npz")
    assert sc.available_regions() == ["lola5m_site99"]


def test_layers_marks_and_figures(viewer_data):
    cfg = sc.default_cfg()
    hz = sc.hazard_predictions(viewer_data, cfg)
    assert hz["unet"] is None and hz["backup"].shape == viewer_data.shape       # no checkpoint in the fixture: backup only
    for name in ("Height", "Slope (official LOLA)", "Hazard: slope-rule backup", "Hazard: official label", "Backup vs label (errors)"):
        arr, title, scale, _rng = sc.layer_array(name, viewer_data, hz)
        assert arr.shape == viewer_data.shape and title and scale
    with pytest.raises(ValueError):
        sc.layer_array("Hazard: U-Net probability", viewer_data, hz)
    sky = {"exact (5 m ray-cast)": np.zeros(360), "model": np.ones(360)}
    cell = np.array([350.0, 350.0])
    marks = sc.skyline_marks(viewer_data, cell, sky, 2.0, directions={"sun": (10.0, 1.0), "earth": (200.0, 3.0)})
    assert len(marks) == 1 + len(sky) + 2
    fig = sc.surface_figure(viewer_data, sc.layer_array("Height", viewer_data, hz), 2.0, 4, marks)
    assert len(fig.data) == 1 + len(marks) and len(sc.skyline_figure(sky, {"sun": (10.0, 1.0), "earth": (200.0, 3.0)}).data) == 4


def test_crop_tile_and_xy_mapping(viewer_data):
    c = sc.crop_tile(viewer_data, 256)
    assert c.shape == (256, 256) and c.real_mask.shape == (256, 256)
    xy = sc.to_xy(np.array([[349.5, 349.5]]), (512, 512), 5.0, 100, 100)[0]
    assert xy[0] == pytest.approx((349.5 - 100) * 5.0) and xy[1] == pytest.approx((511 - (349.5 - 100)) * 5.0)


def test_app_runs_and_labels_generated_terrain(viewer_data):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(repo_root() / "model_viewer" / "app.py"), default_timeout=300).run()
    assert not at.exception and len(at.tabs) == 4
    assert any("GENERATED — NOT OBSERVED" in e.value for e in at.error)
