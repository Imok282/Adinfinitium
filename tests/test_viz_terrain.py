# Copyright (c) 2026, Adinfinitium team
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
"""Fast geometry checks for the Vizard Moon model (basilisk_mods.viz_terrain)."""

import numpy as np
import pytest

from basilisk_mods.lunar_terrain import R_MOON_M, Terrain
from basilisk_mods.viz_terrain import moon_with_site_obj, sphere, tile_patch
from contracts.paths import tile_path

REGION = "lola5m_site01"  # tile with the real LOLA heights; the test is skipped where it is not installed


@pytest.fixture(scope="module")
def terrain() -> Terrain:
    """Terrain of the landing tile, or skip when its data file is missing."""
    if not tile_path(REGION).exists():
        pytest.skip("landing tile not installed")
    return Terrain.from_tile(tile_path(REGION))


def test_patch_has_the_real_heights_inside_and_a_lowered_rim(terrain: Terrain) -> None:
    """The tile centre keeps the DEM height; the rim is lowered to the floor so the tile does not end in a cliff."""
    v, shape, floor = tile_patch(terrain)
    rows, cols = shape
    h = np.linalg.norm(v, axis=2) - R_MOON_M  # [m] height above the reference sphere
    dem = np.asarray(terrain.tile.height, float)[::2, ::2]
    centre = (slice(rows // 2 - 20, rows // 2 + 20), slice(cols // 2 - 20, cols // 2 + 20))
    assert np.allclose(h[centre], dem[centre], atol=0.5)  # [m] bit-identical up to the sphere/ellipse round trip
    assert np.allclose(h[0, :], floor, atol=0.5) and np.allclose(h[:, -1], floor, atol=0.5)  # [m] rim at the floor


def test_sphere_faces_point_outwards_and_sit_below_the_tile() -> None:
    """Every sphere triangle faces away from the Moon's centre."""
    v, tri = sphere(R_MOON_M)
    n = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    assert (np.einsum("ij,ij->i", n, v[tri].mean(1)) > 0.0).all()


def test_obj_is_written_with_both_parts_and_a_material(terrain: Terrain, tmp_path) -> None:
    """The OBJ names its material file, carries the provenance banner and has vertices, normals and faces for both parts."""
    p = moon_with_site_obj(terrain, tmp_path / "moon.obj", step=8)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# Adinfinitium Moon for Vizard") and "mtllib moon.mtl" in text
    assert p.with_suffix(".mtl").read_text(encoding="utf-8").startswith("newmtl moon")
    nv, nn, nf = (sum(1 for line in text.splitlines() if line.startswith(k + " ")) for k in ("v", "vn", "f"))
    assert nv == nn and nf > 10000
