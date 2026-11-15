"""Multi-window data support and the kernel fetcher (offline)."""

import numpy as np
import pytest

from contracts.paths import tile_path
from contracts.tiles import save_tile
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.hazard.lola import ALL_SITES, HOLDOUT_SITES, regions_for_site
from worldmodel.light import fetch_kernels
from worldmodel.light.ephemeris import KERNELS


def make_site_tiles(names):
    for n in names:
        t = make_demo_tile(size=96, n_craters=3, seed=hash(n) % 1000, region=n)
        t.real_mask = np.random.default_rng(1).random(t.shape) < 0.1
        t.extra = {"site": "Site01"}
        save_tile(t, tile_path(n))


def test_every_split_module_shares_one_source_of_truth():
    from worldmodel.light import data as light_data
    from worldmodel.terrain import data as terrain_data
    assert light_data.TEST_SITES == terrain_data.TEST_SITES == HOLDOUT_SITES == ["Site23", "Site42"]
    assert light_data.TRAIN_SITES == terrain_data.TRAIN_SITES and light_data.VAL_SITES == terrain_data.VAL_SITES
    assert not set(HOLDOUT_SITES) & set(light_data.TRAIN_SITES + light_data.VAL_SITES)


def test_rover_world_list_takes_windows_per_site_and_all_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    make_site_tiles(["lola5m_site01", "lola5m_site01_r0c0", "lola5m_site01_r2c3"])
    pytest.importorskip("torch")
    from worldmodel.rover.train import world_list
    assert world_list(["Site01"], ["d1", "d2"], 1) == [("lola5m_site01", "d1"), ("lola5m_site01", "d2")]
    assert [r for r, _ in world_list(["Site01"], ["d1"], 2)] == ["lola5m_site01", "lola5m_site01_r0c0"]
    assert world_list(["Site04"], ["d1"], 3) == []                                  # a site that is not on this machine is skipped


def test_psr_and_points_default_regions(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    make_site_tiles(["lola5m_site01", "lola5m_site01_r0c0", "lola5m_haworth", "lola5m_site01_gen0"])
    from worldmodel.light.make_points import default_regions as points_regions
    from worldmodel.rover.psr import default_regions as psr_regions
    assert psr_regions() == ["lola5m_haworth", "lola5m_site01", "lola5m_site01_r0c0"]          # every window of every site, no generated versions
    assert points_regions() == ["lola5m_site23"][:0] + [r for r in ["lola5m_site01", "lola5m_haworth"] if r in points_regions()]
    assert set(points_regions()) == {"lola5m_site01", "lola5m_haworth"} and "lola5m_site01_r0c0" not in points_regions()


def test_light_and_terrain_use_every_window(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    make_site_tiles(["lola5m_site01", "lola5m_site01_r0c0"])
    assert regions_for_site("Site01") == ["lola5m_site01", "lola5m_site01_r0c0"]
    from worldmodel.terrain.data import load_region
    assert load_region("lola5m_site01_r0c0").real_mask is not None


def test_kernel_fetcher_knows_every_kernel_and_checks_sizes():
    assert set(fetch_kernels.PATHS) == set(fetch_kernels.MIN_BYTES) == set(KERNELS)
    for name in KERNELS:
        assert fetch_kernels.url_for(name).startswith("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/")


def test_kernel_fetcher_skips_complete_files_and_rejects_short_downloads(tmp_path, monkeypatch):
    done = tmp_path / "naif0012.tls"
    done.write_bytes(b"x" * 5000)
    assert fetch_kernels.fetch("naif0012.tls", tmp_path) == done          # already complete: no network

    class Fake:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, n):
            return b""

    monkeypatch.setattr(fetch_kernels.urllib.request, "urlopen", lambda *a, **k: Fake())
    monkeypatch.setattr(fetch_kernels.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        fetch_kernels.fetch("pck00011.tpc", tmp_path, retries=2)
    assert "ALL_SITES" and ALL_SITES[0] == "Site23"
