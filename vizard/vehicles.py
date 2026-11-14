"""3D models of the lander and the rover, built in code (numpy only) from config/mission.yaml. Output: OBJ+MTL (Vizard), GLB (web Navigator), preview PNG, MANIFEST.json.

    python -m vizard.vehicles                 # writes vizard/assets/{lander,rover}.{obj,mtl,glb}, preview PNGs and MANIFEST.json

Honest labelling: these are ILLUSTRATIVE, to-scale models of a Chandrayaan-3-class lander (4 legs, 5 engines) and a Pragyan-class rover (6 wheels),
sized from published envelopes, NOT CAD of flight hardware. Provenance: MODELLED. No agency logos.

Frames (metres, right-handed, the same as the Basilisk body frame B used in basilisk_mods/lander_descent.py):
  lander: origin = body centre, +z = up = thrust direction, engines at z = -1.0 m on a 0.8 m ring (5 deg cant), feet at z ~ -1.4 m
  rover : origin = body centre, +x = forward, +y = left, +z = up, ground plane at z = -0.2 m
The OBJ files keep this frame (Z up). The GLB files are converted to glTF's Y-up convention (x, z, -y).
Numbers taken from config: engine count, attitude-thruster count, lander solar-panel area (the deck panel side is sqrt(area)).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.mission import load_mission

ASSETS = Path(__file__).resolve().parent / "assets"

# visual-only constants (not mission numbers). The engine ring mirrors basilisk_mods/lander_descent.engine_layout (checked by a skip-able test).
ENGINE_RING_RADIUS_M = 0.8
ENGINE_EXIT_Z_M = -1.0
ENGINE_CANT_DEG = 5.0

MATERIALS = {          # name: (rgb, metallic, roughness)
    "gold_foil": ((0.86, 0.66, 0.22), 0.9, 0.35),
    "metal": ((0.72, 0.73, 0.76), 0.8, 0.4),
    "solar": ((0.08, 0.14, 0.38), 0.3, 0.3),
    "dark": ((0.10, 0.10, 0.11), 0.2, 0.7),
    "white": ((0.92, 0.92, 0.9), 0.1, 0.6),
    "rubber": ((0.16, 0.16, 0.17), 0.0, 0.9),
}


@dataclass
class Mesh:
    verts: np.ndarray       # [N,3] float64
    faces: np.ndarray       # [M,3] int, counter-clockwise seen from outside
    material: str
    part: str


def _rot_to(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix taking +z to `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    v = np.cross(z, d)
    c = float(z @ d)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def box(center, size, material: str, part: str, rot: np.ndarray | None = None) -> Mesh:
    sx, sy, sz = (np.asarray(size, dtype=float) / 2.0)
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz], [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]])
    if rot is not None:
        v = v @ rot.T
    return Mesh(v + np.asarray(center, dtype=float), f, material, part)


def frustum(base, direction, length: float, r0: float, r1: float, material: str, part: str, seg: int = 20) -> Mesh:
    """Closed cone frustum / cylinder starting at `base` (radius r0), pointing along `direction`, ending after `length` with radius r1."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(seg)], axis=1)
    v = np.vstack([ring * r0, ring * r1 + [0, 0, length], [[0, 0, 0]], [[0, 0, length]]])
    c0, c1 = 2 * seg, 2 * seg + 1
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f += [[i, j, seg + j], [i, seg + j, seg + i], [c0, j, i], [c1, seg + i, seg + j]]
    v = v @ _rot_to(direction).T + np.asarray(base, dtype=float)
    return Mesh(v, np.array(f), material, part)


def strut(p0, p1, radius: float, material: str, part: str, seg: int = 10) -> Mesh:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    return frustum(p0, p1 - p0, float(np.linalg.norm(p1 - p0)), radius, radius, material, part, seg)


def sphere(center, radius: float, material: str, part: str, n: int = 10) -> Mesh:
    th = np.linspace(0, np.pi, n + 1)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    v, f = [[0, 0, 1]], []
    for t in th[1:-1]:
        for p in ph:
            v.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    v.append([0, 0, -1])
    m = len(ph)
    for p in range(m):
        f.append([0, 1 + p, 1 + (p + 1) % m])
    for r in range(n - 2):
        for p in range(m):
            a, b = 1 + r * m + p, 1 + r * m + (p + 1) % m
            f += [[a, a + m, b + m], [a, b + m, b]]
    last, base = len(v) - 1, 1 + (n - 2) * m
    for p in range(m):
        f.append([last, base + (p + 1) % m, base + p])
    return Mesh(np.array(v) * radius + np.asarray(center, dtype=float), np.array(f), material, part)


def volume(mesh: Mesh) -> float:
    """Signed volume (positive = outward-facing triangles, closed surface)."""
    a, b, c = (mesh.verts[mesh.faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def build_lander(cfg=None) -> list[Mesh]:
    cfg = cfg or load_mission()
    n_eng = cfg.lander.engines.count
    n_rcs = cfg.lander.attitude_thrusters.count
    panel_side = float(np.sqrt(cfg.power.panel_area_m2))
    m = [box((0, 0, -0.25), (1.9, 1.9, 1.2), "gold_foil", "deck"),
         box((0, 0, 0.37), (panel_side, panel_side, 0.04), "solar", "solar_panel_top"),
         box((0, 0, -0.9), (1.5, 1.5, 0.1), "metal", "base_plate")]
    c, s = np.cos(np.radians(ENGINE_CANT_DEG)), np.sin(np.radians(ENGINE_CANT_DEG))
    layout = [((0.0, 0.0, ENGINE_EXIT_Z_M), (0.0, 0.0, 1.0))]
    for ex, ey in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        layout.append(((ENGINE_RING_RADIUS_M * ex, ENGINE_RING_RADIUS_M * ey, ENGINE_EXIT_Z_M), (-s * ex, -s * ey, c)))
    for k, (r, t) in enumerate(layout[:n_eng]):
        plume = -np.asarray(t)                                   # nozzle points opposite to the force
        top = np.asarray(r) - plume * 0.28                       # nozzle throat sits 0.28 m up the plume axis from the exit
        m.append(frustum(top, plume, 0.28, 0.05, 0.17, "metal", f"engine_{k}"))
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for k, (sx, sy) in enumerate(corners):
        m.append(strut((0.9 * sx, 0.9 * sy, -0.75), (1.55 * sx, 1.55 * sy, -1.4), 0.04, "metal", f"leg_{k}"))
        m.append(strut((0.55 * sx, 0.55 * sy, -0.85), (1.4 * sx, 1.4 * sy, -1.34), 0.025, "metal", f"leg_brace_{k}"))
        m.append(frustum((1.55 * sx, 1.55 * sy, -1.45), (0, 0, 1), 0.07, 0.24, 0.2, "white", f"foot_{k}"))
    for k in range(n_rcs):                                       # attitude thrusters: two per upper corner, pointing out
        sx, sy = corners[k % 4]
        side = 0 if k < 4 else 1
        d = np.array([sx, 0.0, 0.0]) if side == 0 else np.array([0.0, sy, 0.0])
        pos = np.array([0.95 * sx, 0.95 * sy, 0.25])
        m.append(frustum(pos, d, 0.12, 0.03, 0.05, "dark", f"rcs_{k}", 10))
    m += [strut((0.7, -0.7, 0.39), (0.7, -0.7, 0.75), 0.02, "metal", "antenna_mast"),
          frustum((0.7, -0.7, 0.75), (0.0, 0.3, 1.0), 0.06, 0.02, 0.28, "white", "antenna_dish", 24),
          box((-0.6, 0.95, 0.1), (0.25, 0.1, 0.18), "dark", "camera_block"),
          box((0.0, 1.02, -0.2), (0.9, 0.12, 0.6), "white", "ramp")]
    return m


def build_rover(cfg=None) -> list[Mesh]:
    cfg = cfg or load_mission()
    m = [box((0, 0, 0.0), (0.74, 0.54, 0.22), "metal", "chassis"),
         box((0, 0, 0.12), (0.8, 0.6, 0.02), "solar", "solar_panel_top")]
    for sx in (-1, 0, 1):
        for sy in (-1, 1):
            x, y = 0.37 * sx, 0.34 * sy
            m.append(frustum((x, y - 0.04, -0.12), (0, 1, 0), 0.08, 0.08, 0.08, "rubber", f"wheel_{'FMR'[sx + 1]}{'L' if sy > 0 else 'R'}", 24))
    for sy in (-1, 1):
        m.append(strut((-0.37, 0.34 * sy, -0.12), (0.37, 0.34 * sy, -0.12), 0.012, "metal", "rocker_bogie_axle"))
        m.append(strut((-0.1, 0.30 * sy, -0.02), (0.0, 0.34 * sy, -0.12), 0.012, "metal", "rocker_link"))
        m.append(strut((0.0, 0.30 * sy, 0.0), (0.37, 0.34 * sy, -0.12), 0.012, "metal", "rocker_link_front"))
    m += [strut((0.25, 0.0, 0.13), (0.25, 0.0, 0.2), 0.012, "metal", "mast"),
          box((0.25, 0.0, 0.205), (0.08, 0.14, 0.04), "dark", "camera_head"),
          sphere((-0.3, 0.0, 0.18), 0.03, "white", "antenna")]
    return m


def bbox(meshes: list[Mesh]) -> dict:
    v = np.vstack([mm.verts for mm in meshes])
    lo, hi = v.min(0), v.max(0)
    return {"min_m": [round(float(x), 3) for x in lo], "max_m": [round(float(x), 3) for x in hi], "size_m": [round(float(x), 3) for x in hi - lo]}


def write_obj(meshes: list[Mesh], path: Path, name: str) -> None:
    path = Path(path)
    lines = [f"# {name}: ILLUSTRATIVE to-scale model, provenance MODELLED, not flight CAD. metres, Z up.", f"mtllib {path.stem}.mtl"]
    off = 1
    for mm in meshes:
        lines += [f"o {mm.part}", f"usemtl {mm.material}"]
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mm.verts]
        lines += [f"f {a + off} {b + off} {c + off}" for a, b, c in mm.faces]
        off += len(mm.verts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mtl = []
    for mat in sorted({mm.material for mm in meshes}):
        (r, g, b), metallic, rough = MATERIALS[mat]
        mtl += [f"newmtl {mat}", f"Kd {r:.3f} {g:.3f} {b:.3f}", f"Ks {metallic * 0.5:.3f} {metallic * 0.5:.3f} {metallic * 0.5:.3f}", f"Ns {(1 - rough) * 200:.1f}", ""]
    path.with_suffix(".mtl").write_text("\n".join(mtl), encoding="utf-8")


def write_glb(meshes: list[Mesh], path: Path, name: str) -> None:
    """Binary glTF 2.0, one flat-shaded primitive per material, converted to Y-up."""
    conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    blobs, views, accs, prims, mats, idx = bytearray(), [], [], [], [], {}
    for mat in sorted({mm.material for mm in meshes}):
        tris = np.vstack([mm.verts[mm.faces] for mm in meshes if mm.material == mat])          # [T,3,3]
        tris = tris @ conv.T
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        pos = tris.reshape(-1, 3).astype("<f4")
        nor = np.repeat(n, 3, axis=0).astype("<f4")
        base = []
        for arr, ty in ((pos, "POSITION"), (nor, "NORMAL")):
            views.append({"buffer": 0, "byteOffset": len(blobs), "byteLength": arr.nbytes, "target": 34962})
            acc = {"bufferView": len(views) - 1, "componentType": 5126, "count": len(arr), "type": "VEC3"}
            if ty == "POSITION":
                acc["min"], acc["max"] = pos.min(0).tolist(), pos.max(0).tolist()
            accs.append(acc)
            base.append(len(accs) - 1)
            blobs += arr.tobytes()
        (r, g, b), metallic, rough = MATERIALS[mat]
        mats.append({"name": mat, "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0], "metallicFactor": metallic, "roughnessFactor": rough}})
        prims.append({"attributes": {"POSITION": base[0], "NORMAL": base[1]}, "material": len(mats) - 1, "mode": 4})
        idx[mat] = len(prims) - 1
    gltf = {"asset": {"version": "2.0", "generator": "adinfinitium vizard.vehicles", "extras": {"provenance": "MODELLED illustrative model, not flight CAD"}},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": name, "mesh": 0}], "meshes": [{"name": name, "primitives": prims}],
            "materials": mats, "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(blobs)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blobs += b"\0" * ((-len(blobs)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blobs)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blobs), 0x004E4942) + bytes(blobs))


def preview(meshes: list[Mesh], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 4.4))
    box_ = bbox(meshes)
    lo, hi = np.array(box_["min_m"]), np.array(box_["max_m"])
    mid, span = (lo + hi) / 2, float((hi - lo).max()) / 2
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    for k, (elev, azim, label) in enumerate([(22, -55, "perspective"), (90, -90, "top"), (0, -90, "front")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for mm in meshes:
            tri = mm.verts[mm.faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
            col = np.array(MATERIALS[mm.material][0])[None, :] * shade[:, None]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(col, 0, 1), edgecolors="none"))
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{title}  -  size {box_['size_m']} m  -  MODELLED, illustrative, not flight CAD", fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_all(out: Path = ASSETS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_mission()
    manifest = {"provenance": "modelled", "note": "illustrative to-scale models, not flight CAD; no agency logos", "units": "metres", "models": {}}
    for name, builder in (("lander", build_lander), ("rover", build_rover)):
        meshes = builder(cfg)
        write_obj(meshes, out / f"{name}.obj", name)
        write_glb(meshes, out / f"{name}.glb", name)
        preview(meshes, out / f"{name}_preview.png", name)
        manifest["models"][name] = {"parts": len(meshes), "triangles": int(sum(len(m.faces) for m in meshes)), **bbox(meshes),
                                    "frame": "Z up, body frame (lander thrust +z)" if name == "lander" else "x forward, y left, z up, ground at z=-0.2",
                                    "files": [f"{name}.obj", f"{name}.mtl", f"{name}.glb", f"{name}_preview.png"]}
    manifest["models"]["lander"]["from_config"] = {"engines": cfg.lander.engines.count, "attitude_thrusters": cfg.lander.attitude_thrusters.count,
                                                  "deck_panel_area_m2": cfg.power.panel_area_m2}
    manifest["models"]["rover"]["from_config"] = {"mass_kg": cfg.rover.mass_kg}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
