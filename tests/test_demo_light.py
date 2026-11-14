"""Stand-in light data (GENERATED): geometry, finite-disk Sun fraction, horizon ray-cast, contract compliance."""

import numpy as np
import pytest

from contracts.light import SYNODIC_SAMPLES, is_standard_axis, load_points, save_points
from contracts.provenance import Provenance
from data_tools.make_demo_light import (
    SUN_ANGULAR_RADIUS_DEG,
    elevation_azimuth,
    horizon_profile,
    make_points,
    ring_cells,
    sun_visible_fraction,
    tile_frame_azimuth,
    time_axis,
)
from data_tools.make_demo_tile import make_demo_tile


def test_overhead_body_is_at_90_degrees():
    el, _ = elevation_azimuth(10.0, 20.0, np.array([10.0]), np.array([20.0]))
    assert el[0] == pytest.approx(90.0, abs=1e-6)


def test_cardinal_directions_from_a_mid_latitude_site():
    # A body at the same longitude but further north must be due north; further east must be due east.
    _, az_n = elevation_azimuth(0.0, 0.0, np.array([30.0]), np.array([0.0]))
    el_e, az_e = elevation_azimuth(0.0, 0.0, np.array([0.0]), np.array([30.0]))
    assert az_n[0] == pytest.approx(0.0, abs=1e-6)
    assert az_e[0] == pytest.approx(90.0, abs=1e-6) and el_e[0] == pytest.approx(60.0, abs=1e-6)


def test_body_on_the_far_side_is_below_the_horizon():
    el, _ = elevation_azimuth(0.0, 0.0, np.array([0.0]), np.array([180.0]))
    assert el[0] == pytest.approx(-90.0, abs=1e-6)


def test_tile_frame_azimuth_rotates_by_east_longitude():
    assert tile_frame_azimuth(np.array([0.0]), 0.0)[0] == 0.0
    assert tile_frame_azimuth(np.array([350.0]), 20.0)[0] == pytest.approx(10.0)


def test_sun_fraction_finite_disk_endpoints():
    r = SUN_ANGULAR_RADIUS_DEG
    el = np.array([-2 * r, -r, 0.0, r, 2 * r])
    f = sun_visible_fraction(el, np.zeros(5))
    assert f[0] == 0.0 and f[1] == pytest.approx(0.0, abs=1e-9)
    assert f[2] == pytest.approx(0.5) and f[3] == pytest.approx(1.0) and f[4] == 1.0
    assert np.all(np.diff(f) >= 0)


def test_horizon_of_a_ramp_and_of_flat_ground():
    mpp, n = 5.0, 201
    flat = np.zeros((n, n))
    assert np.allclose(horizon_profile(flat, mpp, 100, 100, 1.7374e6, max_px=90), 0.0, atol=0.01)
    # Ground rising to the east (+col) at 10 degrees: horizon to the east (az 90) is ~10 deg, to the west ~ -10.
    ramp = np.tile(np.tan(np.radians(10.0)) * np.arange(n) * mpp, (n, 1))
    h = horizon_profile(ramp, mpp, 100, 100, 1.7374e6, max_px=90)
    assert h[90] == pytest.approx(10.0, abs=0.1)
    assert h[270] < -9.0  # ground falls away to the west
    assert abs(h[0]) < 0.5  # north runs along the contour: flat


def test_ring_cells_are_candidate_plus_eight_inside_the_tile():
    t = make_demo_tile(size=128, n_craters=3)
    c = ring_cells(t)
    assert c.shape == (9, 2) and c.min() >= 0 and c.max() < 128
    assert tuple(c[0]) == (63, 63)


def test_time_axis_is_the_agreed_standard_window():
    times, et = time_axis("2027-03-01")
    assert len(times) == SYNODIC_SAMPLES and is_standard_axis(et) and times[0] == "2027-03-01T00:00:00Z"


def test_stand_in_follows_the_terrain_relative_elevation_contract():
    t = make_demo_tile(size=160, n_craters=12, seed=3, region="f")
    ps = make_points(t, "2027-03-01")
    r = SUN_ANGULAR_RADIUS_DEG
    # elev is relative to the skyline: a whole disk above it means fraction 1, a whole disk below means 0.
    assert np.all(ps.sun_fraction[ps.sun_elev_deg >= r] == pytest.approx(1.0))
    assert np.all(ps.sun_fraction[ps.sun_elev_deg <= -r] == pytest.approx(0.0))
    mid = np.abs(ps.sun_elev_deg) < 0.05 * r
    if mid.any():
        assert np.allclose(ps.sun_fraction[mid], 0.5, atol=0.1)
    # Absolute elevation (for drawing) is one direction in the sky: the same for every cell up to 1 deg skyline binning.
    absolute = ps.absolute_elev_deg("earth")
    assert np.allclose(absolute, absolute[:, :1], atol=1.5)
    assert np.array_equal(ps.earth_visible(), ps.earth_elev_deg > 0)


def test_make_points_is_generated_and_contract_valid(tmp_path):
    t = make_demo_tile(size=160, n_craters=5, region="fixture")
    ps = make_points(t, "2027-03-01")
    assert ps.meta.provenance == Provenance.GENERATED and "NOT SPICE" in ps.meta.source
    assert ps.sun_fraction.shape == (SYNODIC_SAMPLES, 9) and ps.horizon_elev_deg.shape == (9, 360)
    # The Sun at a polar site stays within a few degrees of the horizon (axial tilt 1.5 deg) and sweeps all azimuths.
    assert np.abs(ps.sun_elev_deg).max() < 12.0
    assert ps.sun_az_deg.max() - ps.sun_az_deg.min() > 300.0
    back = load_points(save_points(ps, tmp_path / "fixture_2027-03-01_points.npz"))
    assert np.allclose(back.sun_fraction, ps.sun_fraction)
