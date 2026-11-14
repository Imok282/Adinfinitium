"""Real SPICE ephemeris (NASA NAIF DE440) pinned to JPL Horizons, and the real light point-series generator. Skipped if kernels are missing."""

import numpy as np
import pytest

from contracts.provenance import Provenance
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.light.ephemeris import KERNELS, grid_azimuth, kernel_dir, local_basis, to_az_el

pytest.importorskip("spiceypy")
needs_kernels = pytest.mark.skipif(not all((kernel_dir() / k).exists() for k in KERNELS),
                                   reason="SPICE kernels not in <ADI_DATA>/kernels (download from naif.jpl.nasa.gov)")

# JPL Horizons (NASA's own web service), observer on the Moon at lon 359.8 E, lat -85.99, alt 0, 'apparent' azimuth/elevation, fetched 2026-11-14.
HORIZONS_SUN = [("2027-03-01T00:00:00Z", 265.762455, -0.868102), ("2027-03-01T06:00:00Z", 262.725548, -1.084936),
                ("2027-03-01T12:00:00Z", 259.687915, -1.300289), ("2027-03-01T18:00:00Z", 256.649479, -1.513576)]


def test_local_basis_is_orthonormal_and_directions_are_right():
    for lat, lon in ((0.0, 0.0), (-86.0, 120.0), (45.0, -70.0)):
        e, n, u = local_basis(lat, lon)
        assert np.allclose([e @ e, n @ n, u @ u, e @ n, e @ u, n @ u], [1, 1, 1, 0, 0, 0], atol=1e-12)
        assert np.allclose(np.cross(e, n), u, atol=1e-12)
    e, n, u = local_basis(0.0, 0.0)
    assert to_az_el(u, 0.0, 0.0)[1] == pytest.approx(90.0) and to_az_el(n, 0.0, 0.0) == pytest.approx((0.0, 0.0))
    assert to_az_el(e, 0.0, 0.0) == pytest.approx((90.0, 0.0))


def test_grid_azimuth_wraps():
    assert grid_azimuth(np.array([350.0]), 20.0)[0] == pytest.approx(10.0)


@needs_kernels
def test_spice_directions_match_jpl_horizons_to_a_thousandth_of_a_degree():
    from worldmodel.light.ephemeris import sun_earth_directions
    d = sun_earth_directions(-85.99, 359.8, [r[0] for r in HORIZONS_SUN])
    for i, (_, az, el) in enumerate(HORIZONS_SUN):
        assert abs(d["sun_el"][i] - el) < 1e-3 and abs((d["sun_az"][i] - az + 180) % 360 - 180) < 1e-3


@needs_kernels
def test_earth_is_mostly_overhead_from_the_near_side_centre():
    from worldmodel.light.ephemeris import sun_earth_directions
    d = sun_earth_directions(0.0, 0.0, ["2027-03-01T00:00:00Z", "2027-03-08T00:00:00Z"])
    assert np.all(d["earth_el"] > 75.0)          # Earth hangs near the zenith at the sub-Earth point (libration <= ~8 deg)


@needs_kernels
def test_real_points_file_is_modelled_nasa_data_and_contract_valid(tmp_path):
    from contracts.light import load_points, save_points
    from worldmodel.light.make_points import make_points
    t = make_demo_tile(size=700, n_craters=30, seed=4, region="fixture")
    ps = make_points(t, "2027-03-01")
    assert ps.meta.provenance == Provenance.MODELLED and "SPICE" in ps.meta.source and ps.sun_fraction.shape == (709, 9)
    assert np.array_equal(ps.earth_visible(), ps.earth_elev_deg > 0)
    sun_abs = ps.absolute_elev_deg("sun")[:, 0]
    assert np.abs(sun_abs).max() < 8.0           # polar Sun stays within a few degrees of the horizontal
    back = load_points(save_points(ps, tmp_path / "fixture_2027-03-01_points.npz"))
    assert np.allclose(back.horizon_elev_deg, ps.horizon_elev_deg)
