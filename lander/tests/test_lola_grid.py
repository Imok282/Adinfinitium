"""LOLA window grid and tile naming (offline: no network)."""

import numpy as np
import pytest

from worldmodel.hazard import lola
from worldmodel.hazard.lola import (
    ALL_SITES,
    HOLDOUT_SITES,
    TRAIN_SITES,
    VAL_SITES,
    WINDOW,
    grid_origins,
    raw_window_path,
    region_name,
)


def test_split_is_disjoint_and_keeps_the_test_sites_out_of_training():
    assert not set(HOLDOUT_SITES) & set(TRAIN_SITES + VAL_SITES) and not set(VAL_SITES) & set(TRAIN_SITES)
    assert {"Site23", "Site42"} == set(HOLDOUT_SITES) and "Site20v2" not in ALL_SITES      # Site20v2 extends Site20: would leak


def test_region_and_raw_names():
    assert region_name("Site23") == "lola5m_site23" and region_name("Haworth", (1024, 3072)) == "lola5m_haworth_r1c3"
    assert raw_window_path("Site01").name == "window.npz" and raw_window_path("Site01", (0, 2048)).name == "window_r0c2.npz"


@pytest.mark.parametrize("hw", [(3200, 3200), (4000, 4000), (4200, 4200), (5960, 5960)])
def test_grid_windows_never_overlap_each_other_or_the_centre_window(monkeypatch, hw):
    monkeypatch.setattr(lola, "site_dims", lambda site: hw)
    origins = grid_origins("SiteX")
    rc, cc = (hw[0] - WINDOW) // 2, (hw[1] - WINDOW) // 2
    assert origins and len(set(origins)) == len(origins)
    boxes = [(r, c) for r, c in origins]
    assert all(r + WINDOW <= hw[0] and c + WINDOW <= hw[1] for r, c in boxes)
    assert all(abs(r - rc) >= WINDOW or abs(c - cc) >= WINDOW for r, c in boxes)           # no overlap with the centre window
    for i, (r1, c1) in enumerate(boxes):
        for r2, c2 in boxes[i + 1:]:
            assert abs(r1 - r2) >= WINDOW or abs(c1 - c2) >= WINDOW                       # no overlap between grid windows


def test_regions_for_site_lists_centre_and_grid_tiles_but_not_generated_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    d = tmp_path / "tiles"
    d.mkdir()
    for n in ("lola5m_site01", "lola5m_site01_r0c0", "lola5m_site01_r2c3", "lola5m_site01_gen0", "lola5m_site011", "lola5m_site02"):
        np.savez(d / f"{n}.npz", x=np.zeros(1))
    assert lola.regions_for_site("Site01") == ["lola5m_site01", "lola5m_site01_r0c0", "lola5m_site01_r2c3"]
