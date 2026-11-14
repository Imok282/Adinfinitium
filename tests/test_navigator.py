"""Navigator (KAN-35): exact-physics metrics on hand-made series with known answers, data layer, app smoke test."""

import numpy as np
import pytest

from contracts.light import PointSeries, save_points
from contracts.paths import points_path, repo_root, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import save_tile
from data_tools.make_demo_light import make_points
from data_tools.make_demo_tile import make_demo_tile
from navigator import figures
from navigator.data import display_name, list_entries
from navigator.metrics import (
    countdown,
    earth_up,
    format_hours,
    runs,
    step_hours,
    summarize,
    sun_countdown,
    sun_up,
)

META = Meta(provenance=Provenance.GENERATED, source="test")


def series(sun, earth_elev, horizon=0.0, earth_az=0.0, sun_elev=None) -> PointSeries:
    """T samples 1 h apart, one cell. sun = 0/1 per sample (sun_fraction)."""
    t = len(sun)
    sun = np.asarray(sun, float)
    return PointSeries(
        region="r", date="2027-01-01", cells=np.array([[5, 5]]), times_utc=[f"2027-01-01T{h:02d}:00:00Z" for h in range(t)],
        et_s=np.arange(t, dtype=float) * 3600.0, sun_fraction=sun[:, None].astype(np.float32),
        sun_elev_deg=np.asarray(sun_elev if sun_elev is not None else sun * 2.0 - 1.0, float)[:, None].astype(np.float32),
        earth_elev_deg=np.asarray(earth_elev, float)[:, None].astype(np.float32),
        sun_az_deg=np.zeros(t, np.float32), earth_az_deg=np.full(t, earth_az, np.float32),
        horizon_elev_deg=np.full((1, 360), horizon, np.float32), meta=META)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs(np.array([], bool)) == [] and runs(np.zeros(3, bool)) == []
    assert runs(np.ones(3, bool)) == [(0, 3)]


def test_earth_line_of_sight_follows_the_contract_elevation_is_already_relative_to_the_skyline():
    # earth_elev_deg > 0 <=> centre above the skyline. The horizon must NOT be subtracted a second time.
    assert earth_up(series([1, 1, 1], [3.0, 0.5, 3.0], horizon=5.0)).all()
    assert not earth_up(series([1, 1, 1], [-3.0, -0.5, 0.0], horizon=0.0)).any()


def test_mean_sun_elevation_is_reported_above_the_horizontal_plane():
    ps = series([1, 1, 0, 0], [1] * 4, horizon=5.0, sun_elev=[1.0, 3.0, -2.0, -4.0])
    s = summarize(ps)
    assert s.mean_sun_elev_deg == pytest.approx(7.0)   # (1+5 + 3+5) / 2: skyline-relative + 5 deg skyline
    assert s.max_sun_elev_deg == pytest.approx(8.0)


def test_summary_hours_and_blackout():
    #            sun:   1  1  0  0  0  1  1  0   (8 samples = 7 h window)
    #          earth:   +  -  +  -  -  -  +  +
    ps = series([1, 1, 0, 0, 0, 1, 1, 0], [2, -2, 2, -2, -2, -2, 2, 2])
    s = summarize(ps)
    assert step_hours(ps) == 1.0 and s.window_hours == 7.0
    assert s.sun_hours == 4 and s.earth_hours == 4 and s.both_hours == 2
    assert s.blackout_hours == 2                      # samples 3, 4: no sun, no earth
    assert s.longest_dark_hours == 3 and not s.longest_dark_is_lower_bound
    assert s.longest_blackout_hours == 2
    assert s.n_sunsets == 2 and s.n_earthsets == 2


def test_longest_darkness_touching_the_window_edge_is_a_lower_bound():
    s = summarize(series([1, 0, 0, 0], [1, 1, 1, 1]))
    assert s.longest_dark_hours == 3 and s.longest_dark_is_lower_bound


def test_never_sunlit_site():
    s = summarize(series([0, 0, 0], [1, 1, 1]))
    assert s.sun_hours == 0 and np.isnan(s.mean_sun_elev_deg) and s.n_sunsets == 0


def test_countdowns():
    ps = series([1, 1, 1, 0, 0, 1], [1] * 6)
    up = sun_countdown(ps, 0)
    assert (up.state, up.event, up.hours, up.at_utc) == ("up", "sunset", 3.0, "2027-01-01T03:00:00Z")
    dn = sun_countdown(ps, 3)
    assert (dn.state, dn.event, dn.hours) == ("down", "sunrise", 2.0)
    assert sun_countdown(ps, 5).event is None         # nothing changes before the window ends
    assert countdown(ps, np.array([True] * 6), 0, "a", "b").hours is None
    assert format_hours(None) == "no change in this window" and format_hours(5) == "5 h" and format_hours(50) == "2 d 2 h"


def test_data_layer_lists_files_and_names(tmp_path):
    for n in ("siteA_2027-03-01_points.npz", "siteA_2027-06-01_points.npz", "ignore_me.npz"):
        (tmp_path / n).write_bytes(b"x")
    es = list_entries(tmp_path)
    assert [(e.region, e.date) for e in es] == [("siteA", "2027-03-01"), ("siteA", "2027-06-01")]
    t = make_demo_tile(size=32, n_craters=1, region="siteA")
    t.meta = Meta(provenance=Provenance.OBSERVED, source="LOLA PGDA 5 m/pixel Site23 (Malapert massif), Barker")
    t.extra = {"site": "Site23"}
    assert display_name("siteA", t, {}) == "Malapert massif (Site23)"
    assert display_name("unknown", None, {}) == "unknown" and display_name("siteA", None, {"siteA": "Named"}) == "Named"


def test_figures_build_on_a_real_shaped_series():
    t = make_demo_tile(size=160, n_craters=3, region="f")
    ps = make_points(t, "2027-03-01")
    for f in (figures.timeline_figure(ps, 10), figures.panorama_figure(ps, 10), figures.dome_figure(ps, 10),
              figures.tile_map_figure(t, "hazard", (80, 80))):
        assert len(f.data) >= 2
    assert sun_up(ps).dtype == bool


@pytest.fixture
def demo_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    for region, seed in (("demo_a", 1), ("demo_b", 2)):
        tile = make_demo_tile(size=160, n_craters=4, seed=seed, region=region)
        save_tile(tile, tile_path(region))
        for date in ("2027-03-01", "2027-06-01"):
            save_points(make_points(tile, date), points_path(region, date))
    return tmp_path


def test_app_runs_shows_generated_banner_and_all_tabs(demo_data):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(repo_root() / "navigator" / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert any("GENERATED" in w.value for w in at.warning)
    assert len(at.tabs) == 5 and len(at.multiselect[0].value) == 2      # one default combination per region
    at.slider[0].set_value(100).run()
    assert not at.exception


def test_app_without_data_explains_how_to_get_some(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    at = AppTest.from_file(str(repo_root() / "navigator" / "app.py"), default_timeout=30).run()
    assert not at.exception and any("make_demo_light" in e.value for e in at.error)
