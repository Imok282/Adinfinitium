# Copyright (c) 2026, Adinfinitium team
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
"""The Moon with the real landing-site terrain, as one OBJ model for Vizard.

Vizard draws the Moon as a smooth sphere, so a lander touching down on the LOLA terrain (1.5 to 2.0 km above the reference
sphere at Site01) would appear to hover or sink. This module writes a replacement Moon model in the Moon-fixed frame
(metres, +z towards the north pole, origin at the Moon's centre), so that it can be attached to the Moon with
``vizSupport.createCustomModel(viz, modelPath=..., simBodiesToModify=["moon"])`` (``MissionSetup.viz_models={"moon": path}``).

The model has two parts:

* the landing tile (LOLA, observed) at its real heights, rim lowered smoothly to the floor height so it does not end in a cliff;
* a smooth sphere just below the tile's lowest point, standing for the rest of the Moon (MODELLED: no relief, no texture).

Status: the geometry is checked by tests/test_viz_terrain.py. How Vizard places and rotates a custom model on a celestial body
has NOT been checked on a machine with Vizard installed; run ``python -m scripts.live_sim --live-viz --viz-terrain`` with Vizard
connected and compare with the 3D view before relying on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from basilisk_mods.lunar_terrain import R_MOON_M, Terrain, latlon_from_ps, moon_fixed_from_latlon

FADE_M = 400.0  # [m] width of the tile rim that is lowered to the floor height
SPHERE_STEP_DEG = 2.0  # [deg] latitude and longitude spacing of the smooth sphere
SPHERE_BELOW_TILE_M = 2.0  # [m] sphere radius below the lowest tile height (sphere stays hidden under the tile)
MOON_GREY = (0.62, 0.61, 0.60)  # [-] diffuse reflectance (RGB) of the model


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tile_patch(terrain: Terrain, step: int = 2, fade_m: float = FADE_M) -> tuple[np.ndarray, np.ndarray, float]:
    """Vertices of the landing tile in the Moon-fixed frame.

    :param terrain: terrain of the landing tile
    :param step: keep every ``step``-th tile pixel
    :param fade_m: width [m] of the rim lowered to the floor height
    :return: ``(vertices, grid_shape, floor_m)``: vertices ``(rows, cols, 3)`` [m], the ``(rows, cols)`` shape, and the floor height [m]
        above the reference sphere that the rim is lowered to
    """
    tile = terrain.tile
    hgt = np.asarray(tile.height, float)[::step, ::step]
    rows, cols = hgt.shape
    mpp = tile.meters_per_pixel * step  # [m] vertex spacing
    xs = (np.arange(cols) - (cols - 1) / 2.0) * mpp  # [m] tile east
    ys = ((rows - 1) / 2.0 - np.arange(rows)) * mpp  # [m] tile north
    X, Y = np.meshgrid(xs, ys)
    floor = float(hgt.min())  # [m]
    edge = np.minimum.reduce([X - xs[0], xs[-1] - X, Y - ys[-1], ys[0] - Y])  # [m] distance to the nearest tile edge
    s = _smoothstep(edge / fade_m)
    h = floor + (hgt - floor) * s  # [m] height above the reference sphere
    lat, lon = latlon_from_ps(terrain.center_ps_x + X, terrain.center_ps_y + Y)
    return moon_fixed_from_latlon(lat, lon, h), np.array([rows, cols]), floor


def sphere(radius_m: float, step_deg: float = SPHERE_STEP_DEG) -> tuple[np.ndarray, np.ndarray]:
    """A latitude/longitude sphere with outward-facing triangles.

    :param radius_m: sphere radius [m]
    :param step_deg: latitude and longitude spacing [deg]
    :return: vertices ``(n, 3)`` [m] and triangles ``(m, 3)`` (zero-based vertex indices)
    """
    lats = np.arange(-90.0, 90.0 + 1e-9, step_deg)
    lons = np.arange(0.0, 360.0, step_deg)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    v = moon_fixed_from_latlon(LA, LO, 0.0, radius_m).reshape(-1, 3)
    nl, no = len(lats), len(lons)
    idx = np.arange(nl * no).reshape(nl, no)
    a, b = idx[:-1, :].ravel(), idx[:-1, (np.arange(no) + 1) % no].ravel()
    c, d = idx[1:, :].ravel(), idx[1:, (np.arange(no) + 1) % no].ravel()
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    cen = v[tri].mean(1)
    flip = np.einsum("ij,ij->i", np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), cen) < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    keep = np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), axis=1) > 0.0  # drop the pole slivers
    return v, tri[keep]


def moon_with_site_obj(terrain: Terrain, path: str | Path, step: int = 2, fade_m: float = FADE_M) -> Path:
    """Write the Moon model (smooth sphere plus the landing tile) as ``path`` (+ ``.mtl``).

    :param terrain: terrain of the landing tile
    :param path: output ``.obj`` path
    :param step: keep every ``step``-th tile pixel (2 gives about 65 000 vertices for a 512 x 512 tile)
    :param fade_m: width [m] of the tile rim lowered to the floor height
    :return: the written ``.obj`` path
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    patch, shape, floor = tile_patch(terrain, step, fade_m)
    rows, cols = int(shape[0]), int(shape[1])
    sv, st = sphere(R_MOON_M + floor - SPHERE_BELOW_TILE_M)
    pv = patch.reshape(-1, 3)
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    pt = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])  # counter-clockwise seen from above
    nrm = np.zeros_like(pv)  # smooth normals: sum of the face normals around each vertex
    fn = np.cross(pv[pt[:, 1]] - pv[pt[:, 0]], pv[pt[:, 2]] - pv[pt[:, 0]])
    for k in range(3):
        np.add.at(nrm, pt[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    snrm = sv / np.linalg.norm(sv, axis=1, keepdims=True)
    banner = terrain.tile.meta.provenance.banner
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Adinfinitium Moon for Vizard. Tile {terrain.tile.region} [{banner}] (observed LOLA); the rest is a smooth sphere (MODELLED).\n")
        f.write("# Moon-fixed frame, metres, +z north; rim of the tile lowered to the floor height over %.0f m\n" % fade_m)
        f.write(f"mtllib {p.stem}.mtl\nusemtl moon\n")
        for arr in (pv, sv):
            f.write("\n".join(f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in arr) + "\n")
        for arr in (nrm, snrm):
            f.write("\n".join(f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in arr) + "\n")
        npv = len(pv)
        f.write("\n".join(f"f {i + 1}//{i + 1} {j + 1}//{j + 1} {k + 1}//{k + 1}" for i, j, k in pt) + "\n")
        f.write("\n".join(f"f {i + npv + 1}//{i + npv + 1} {j + npv + 1}//{j + npv + 1} {k + npv + 1}//{k + npv + 1}" for i, j, k in st) + "\n")
    p.with_suffix(".mtl").write_text("newmtl moon\nKa 0 0 0\nKd %.2f %.2f %.2f\nKs 0 0 0\nNs 1\n" % MOON_GREY, encoding="utf-8")
    return p
