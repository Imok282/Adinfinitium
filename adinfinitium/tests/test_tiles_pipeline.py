"""Tiles and PSR stages survive a failing site or tile (offline: NASA reads are monkeypatched)."""

import threading

import numpy as np
import pytest

from contracts.paths import tile_path
from contracts.tiles import load_tile
from worldmodel.hazard import lola
from worldmodel.rover import psr

N = 8


@pytest.fixture
def nasa(tmp_path, monkeypatch):
    """Fake NASA: SiteBad is unreachable; every other site is 3x3 windows of N px. Records the fetch order and thread use."""
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    monkeypatch.setattr(lola, "WINDOW", N)
    log = {"fetch": [], "tiles_seen": [], "threads": set()}

    def dims(site):
        if site == "SiteBad":
            raise RuntimeError("503")
        return 3 * N, 3 * N

    def read(site, layer, size, retries=6, origin=None):
        if site == "SiteBad":
            raise RuntimeError("503")
        log["threads"].add(threading.get_ident())
        if layer == "surf":
            log["fetch"].append((site, origin))
            log["tiles_seen"].append(len(list((tmp_path / "tiles").glob("*.npz"))))
        return layer, np.full((N, N), 1.0, np.float32), (100.0, -200.0), lola.MPP

    monkeypatch.setattr(lola, "site_dims", dims)
    monkeypatch.setattr(lola, "_read_window", read)
    real_grid = lola.grid_origins
    monkeypatch.setattr(lola, "grid_origins", lambda s: real_grid(s, N))
    return log


def test_unreachable_site_does_not_stop_the_others_and_tiles_are_built_as_fetched(nasa, capsys):
    rc = lola.main(["all", "--grid", "--limit", "2", "--workers", "3", "--sites", "Site01", "SiteBad", "Site42"])
    out = capsys.readouterr().out
    assert rc == 0 and "FAILED site SiteBad" in out and "summary Site42: wanted 3 fetched 3 built 3 failed 0" in out
    # Priority site first, centres before grid, grid round-robin across the reachable sites.
    assert [s for s, _ in nasa["fetch"]] == ["Site42", "Site01", "Site42", "Site01", "Site42", "Site01"]
    assert [o is None for _, o in nasa["fetch"]] == [True, True, False, False, False, False]
    assert nasa["tiles_seen"] == [0, 1, 2, 3, 4, 5]                      # each tile is on disk before the next fetch starts
    assert len(nasa["threads"]) <= 3
    assert load_tile(tile_path("lola5m_site42")).extra["centre_xy_m"] == [100.0, -200.0]
    assert lola.main(["all", "--sites", "SiteBad"]) == 1                 # nothing built and nothing on disk


def test_rebuild_keeps_an_attached_psr_mask_and_resume_leaves_built_tiles_alone(nasa, monkeypatch, capsys):
    assert lola.main(["all", "--sites", "Site23"]) == 0
    monkeypatch.setattr(psr, "read_psr_window", lambda cx, cy, half: np.ones((N // 4, N // 4), np.uint8))
    assert psr.main(["--regions", "lola5m_site23"]) == 0
    capsys.readouterr()
    assert lola.main(["all", "--sites", "Site23"]) == 0 and "wrote" not in capsys.readouterr().out
    assert lola.main(["build", "--sites", "Site23"]) == 0 and "wrote" in capsys.readouterr().out
    t = load_tile(tile_path("lola5m_site23"))
    assert t.psr_mask.all() and t.extra["psr_mask"] == psr.SOURCE and t.extra["psr_fraction"] == 1.0


def test_psr_main_continues_after_a_failing_tile(nasa, monkeypatch, capsys):
    assert lola.main(["all", "--sites", "Site23", "Site42"]) == 0

    def flaky(cx, cy, half):
        flaky.calls += 1
        if flaky.calls == 1:
            raise RuntimeError("503")
        return np.ones((N // 4, N // 4), np.uint8)

    flaky.calls = 0
    monkeypatch.setattr(psr, "read_psr_window", flaky)
    assert psr.main(["--regions", "lola5m_site23", "lola5m_site42"]) == 0
    assert "FAILED lola5m_site23" in capsys.readouterr().out and load_tile(tile_path("lola5m_site42")).psr_mask.all()
    monkeypatch.setattr(psr, "read_psr_window", lambda cx, cy, half: (_ for _ in ()).throw(RuntimeError("503")))
    assert psr.main(["--regions", "lola5m_site23"]) == 1
