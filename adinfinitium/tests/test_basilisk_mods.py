"""Basilisk starter tests. Skipped automatically where Basilisk (pip install bsk) isn't installed."""

import numpy as np
import pytest

from basilisk_mods.lunar_terrain import (
    Terrain,
    latlon_from_ps,
    moon_fixed_from_latlon,
    ps_from_latlon,
    ps_from_moon_fixed,
    tile_to_obj,
)
from data_tools.make_demo_tile import make_demo_tile


# ------------------------------------------------- terrain (numpy only, always runs)
def test_ps_roundtrip():
    lat = np.array([-89.9, -88.0, -85.0, -80.5])
    lon = np.array([0.0, 45.0, 200.0, 359.0])
    x, y = ps_from_latlon(lat, lon)
    lat2, lon2 = latlon_from_ps(x, y)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-6)


def test_moon_fixed_to_ps_altitude():
    r = moon_fixed_from_latlon(-89.0, 30.0, height_m=2000.0)
    x, y, alt = ps_from_moon_fixed(r)
    assert alt == pytest.approx(2000.0, abs=1e-3)
    x2, y2 = ps_from_latlon(-89.0, 30.0)
    assert x == pytest.approx(x2) and y == pytest.approx(y2)


def test_terrain_lookup_matches_grid():
    tile = make_demo_tile(size=41, mpp=5.0, n_craters=5, seed=3)
    terr = Terrain.from_tile(tile)
    # tile centre pixel is (20, 20) -> x = y = 0
    assert terr.ground_height(0.0, 0.0) == pytest.approx(float(tile.height[20, 20]), abs=1e-4)
    # one pixel east
    assert terr.ground_height(5.0, 0.0) == pytest.approx(float(tile.height[20, 21]), abs=1e-4)
    # one pixel north = one row UP
    assert terr.ground_height(0.0, 5.0) == pytest.approx(float(tile.height[19, 20]), abs=1e-4)


def test_obj_export(tmp_path):
    tile = make_demo_tile(size=21, mpp=5.0, n_craters=2)
    p = tile_to_obj(tile, tmp_path / "t.obj", step=2)
    lines = p.read_text().splitlines()
    assert "GENERATED" in lines[0]
    assert sum(1 for line in lines if line.startswith("v ")) == 11 * 11
    assert sum(1 for line in lines if line.startswith("f ")) == 2 * 10 * 10


# ------------------------------------------------- Basilisk scenarios
@pytest.fixture
def bsk_runs(tmp_path, monkeypatch):
    pytest.importorskip("Basilisk")
    monkeypatch.setenv("ADI_RUNS", str(tmp_path))


def test_orbit_release_reaches_pdi(bsk_runs):
    from basilisk_mods.scenario_orbit_release import run
    s = run(viz=False, plot=False)
    assert s["altitude_m"] == pytest.approx(15_000.0, abs=200.0)
    assert 1600 < s["speed_m_s"] < 1800


@pytest.mark.parametrize("engine_out_at", [None, 20.0])
def test_descent_lands_within_limits(bsk_runs, engine_out_at):
    from basilisk_mods.scenario_descent import run
    from contracts.paths import runs_root
    from contracts.runs import read_telemetry
    rep = run(run_id="t", engine_out_at=engine_out_at, viz=False)
    assert rep["outcome"] == "ok", rep
    assert rep["vertical_speed_m_s"] <= 2.0 and rep["horizontal_speed_m_s"] <= 0.5
    rows = read_telemetry(runs_root() / "basilisk" / "telemetry_t.csv")
    assert rows[-1]["phase"] == "touchdown"
