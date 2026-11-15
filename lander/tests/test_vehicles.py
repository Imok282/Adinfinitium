"""3D vehicle models: closed outward-facing meshes, config-driven counts, plausible envelopes, valid OBJ/GLB files."""

import json
import struct

import numpy as np
import pytest

from contracts.mission import load_mission
from vizard.vehicles import (
    ENGINE_CANT_DEG,
    ENGINE_EXIT_Z_M,
    ENGINE_RING_RADIUS_M,
    bbox,
    build_all,
    build_lander,
    build_rover,
    volume,
)


def test_every_part_is_closed_and_outward():
    for mm in build_lander() + build_rover():
        assert volume(mm) > 0, f"{mm.part} has inward faces or is not closed"
        assert mm.faces.max() < len(mm.verts)


def test_lander_counts_follow_config():
    cfg = load_mission()
    parts = [m.part for m in build_lander(cfg)]
    assert sum(p.startswith("engine_") for p in parts) == cfg.lander.engines.count
    assert sum(p.startswith("rcs_") for p in parts) == cfg.lander.attitude_thrusters.count
    assert sum(p.startswith("leg_") and "brace" not in p for p in parts) == 4
    panel = next(m for m in build_lander(cfg) if m.part == "solar_panel_top")
    side = panel.verts.max(0) - panel.verts.min(0)
    assert side[0] * side[1] == pytest.approx(cfg.power.panel_area_m2, rel=1e-6)


def test_envelopes_are_plausible():
    lander, rover = bbox(build_lander())["size_m"], bbox(build_rover())["size_m"]
    assert 2.0 < max(lander[0], lander[1]) < 4.0 and 1.5 < lander[2] < 2.8      # Chandrayaan-3 class, ~2 m body + legs
    assert 0.8 < rover[0] < 1.1 and 0.6 < rover[1] < 0.9 and 0.3 < rover[2] < 0.5   # Pragyan class 0.92 x 0.75 x 0.40 m
    assert sum(m.part.startswith("wheel_") for m in build_rover()) == 6


def test_engine_nozzles_end_on_the_basilisk_ring():
    exits = [m for m in build_lander() if m.part.startswith("engine_")]
    lows = sorted(round(float(m.verts[:, 2].min()), 2) for m in exits)
    assert lows[0] <= ENGINE_EXIT_Z_M + 0.02 and ENGINE_RING_RADIUS_M == 0.8 and ENGINE_CANT_DEG == 5.0


def test_engine_layout_matches_basilisk_module_when_available():
    ld = pytest.importorskip("basilisk_mods.lander_descent")
    lay = ld.engine_layout()
    assert lay[1][0][:2] == [lay[1][0][0], 0.0] and abs(lay[1][0][0]) == ENGINE_RING_RADIUS_M and lay[1][0][2] == ENGINE_EXIT_Z_M
    assert ld.ENGINE_CANT_DEG == ENGINE_CANT_DEG


def test_files_written_and_valid(tmp_path):
    man = build_all(tmp_path)
    for name in ("lander", "rover"):
        obj = (tmp_path / f"{name}.obj").read_text()
        nv = obj.count("\nv ")
        assert f"mtllib {name}.mtl" in obj and nv > 100
        nums = [int(t) for line in obj.splitlines() if line.startswith("f ") for t in line.split()[1:]]
        assert max(nums) <= nv and min(nums) >= 1
        glb = (tmp_path / f"{name}.glb").read_bytes()
        magic, ver, length = struct.unpack("<III", glb[:12])
        assert magic == 0x46546C67 and ver == 2 and length == len(glb)
        jlen = struct.unpack("<I", glb[12:16])[0]
        gltf = json.loads(glb[20:20 + jlen])
        assert gltf["asset"]["version"] == "2.0" and gltf["meshes"][0]["primitives"]
        assert np.isfinite(man["models"][name]["size_m"]).all()
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["provenance"] == "modelled"
