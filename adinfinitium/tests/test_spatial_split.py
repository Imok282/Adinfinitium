"""Spatial-leakage guard: train/val windows next to a test tile are excluded, the test tile never is (offline, fake tiles)."""

import json

import numpy as np
import pytest

from contracts.mission import load_mission
from contracts.paths import tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, save_tile
from ops import spatial_split
from worldmodel.hazard import lola

N, MPP = 8, 640.0          # 8 px at 640 m = a 5120 m tile, the real tile size
TEST_XY = (-500.0, 121500.0)


def fake_tile(region: str, xy: tuple[float, float], as_extra: bool) -> None:
    """A tiny tile centred at stereographic xy; position stored as extra.centre_xy_m or only as lat/lon."""
    lat, lon = lola.xy_to_latlon(*xy, load_mission().moon.radius_m)
    z = np.zeros((N, N), dtype=np.float32)
    meta = Meta(provenance=Provenance.MODELLED, source="test", frame="test", created_by="tests")
    tile = Tile(region=region, meters_per_pixel=MPP, height=z, slope=z, hazard=z, uncertainty=z, psr_mask=z.astype(bool), meta=meta,
                center_lat_deg=lat, center_lon_deg=lon, extra={"centre_xy_m": list(xy)} if as_extra else {})
    save_tile(tile, tile_path(region))


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    monkeypatch.delenv("ADI_NO_SPATIAL_EXCLUDE", raising=False)
    x, y = TEST_XY
    layout = {
        "lola5m_site23": (TEST_XY, True),                                # test tile
        "lola5m_haworth": ((x + 40000.0, y), False),                     # train, far
        "lola5m_haworth_r0c1": ((x + 5120.0, y + 5120.0), False),        # train, touching corner: inside the margin (lat/lon only)
        "lola5m_haworth_r0c2": ((x + 6300.0, y), True),                  # train, 6300 m < 5120 + 1280
        "lola5m_haworth_r0c3": ((x + 6500.0, y), True),                  # train, 6500 m > 6400: kept
        "lola5m_haworth_r0c4": ((x + 1000.0, y + 7000.0), True),         # train, close in x only: kept
        "lola5m_site11_r1c1": ((x - 3000.0, y - 3000.0), True),          # val, overlapping
        "lola5m_site11": ((x, y - 30000.0), True),                       # val, far
    }
    for region, (xy, as_extra) in layout.items():
        fake_tile(region, xy, as_extra)
    return tmp_path


def test_overlapping_train_and_val_windows_are_excluded_and_the_test_tile_never_is(tiles):
    rec = spatial_split.check()
    assert set(rec["excluded"]) == {"lola5m_haworth_r0c1", "lola5m_haworth_r0c2", "lola5m_site11_r1c1"}
    assert {r: e["role"] for r, e in rec["excluded"].items()} == {
        "lola5m_haworth_r0c1": "train", "lola5m_haworth_r0c2": "train", "lola5m_site11_r1c1": "val"}
    assert all(e["near"] == "lola5m_site23" for e in rec["excluded"].values())
    assert rec["excluded"]["lola5m_haworth_r0c1"]["dx_m"] == pytest.approx(5120.0, abs=1.0)    # lat/lon inversion is metre-accurate
    assert rec["checked"] == {"train": 5, "val": 2, "test": 1} and rec["min_train_test_dist_m"] == pytest.approx(6300.0)
    assert json.loads(spatial_split.excluded_path().read_text())["excluded"].keys() == rec["excluded"].keys()

    assert lola.regions_for_site("Haworth") == ["lola5m_haworth", "lola5m_haworth_r0c3", "lola5m_haworth_r0c4"]
    assert lola.regions_for_site("Site11") == ["lola5m_site11"]
    assert lola.regions_for_site("Site23") == ["lola5m_site23"]

    assert spatial_split.trained_overlaps(["lola5m_haworth", "lola5m_haworth_r0c2", "lola5m_site23"]) == ["lola5m_haworth_r0c2"]
    assert spatial_split.main(["--verify"]) == 0


def test_filter_can_be_switched_off_and_verify_fails_on_a_stale_file(tiles, monkeypatch):
    assert "lola5m_haworth_r0c2" in lola.regions_for_site("Haworth")                           # no file yet: nothing filtered
    spatial_split.check()
    monkeypatch.setenv("ADI_NO_SPATIAL_EXCLUDE", "1")
    assert "lola5m_haworth_r0c2" in lola.regions_for_site("Haworth")
    fake_tile("lola5m_site01_r0c0", (TEST_XY[0], TEST_XY[1] + 2000.0), True)                   # built after the check ran
    assert spatial_split.main(["--verify"]) == 1
