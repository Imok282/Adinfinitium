"""Blender build of the chosen designs (Lander B, Rover B, Separation B): detailed, to-scale, Z-up models + hero renders + GLB/OBJ export.

    Blender -b -P vizard/blender/build_vehicles.py -- --out vizard/assets/blender --terrain <patch.npz> [--samples 64] [--no-render]

Frames (metres, same as the simulator / Basilisk body frame):
  lander : origin = body centre, +z up = thrust, 5 nozzle exits at z = -1.0 m (4 on a 0.8 m ring, 5 deg cant), feet at z = -1.45 m
  rover  : origin = body centre, +x forward, +z up, wheels touch z = -0.205 m
  propulsion module (PM): origin = PM top separation plane; in the stack the lander origin sits 1.25 m above it (attached) or 2.45 m (separating)
Illustrative, to scale, not flight CAD; no agency logos. Surface renders use real NASA LOLA 5 m terrain (Site23) for the ground shape; the regolith
texture is artistic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


OUT = Path(arg("--out", "vizard/assets/blender")).resolve()
TERRAIN = arg("--terrain")
SAMPLES = int(arg("--samples", "64"))
RENDER = "--no-render" not in ARGS
RES = (int(arg("--width", "1600")), int(arg("--height", "1000")))

# ---------------------------------------------------------------------------------------------------------------- materials


def _principled(name, color, metallic=0.0, roughness=0.5, bump=0.0, bump_scale=40.0, emission=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*emission, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 2.0
    if bump > 0:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = bump_scale
        tex.inputs["Detail"].default_value = 8.0
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(tex.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    m.diffuse_color = (*color, 1.0)          # what the viewport and some exporters use
    m.metallic, m.roughness = metallic, roughness
    return m


MAT = {}


def materials():
    MAT.update({
        "gold_foil": _principled("gold_foil_MLI", (0.83, 0.6, 0.2), 1.0, 0.32, bump=0.35, bump_scale=25.0),
        "silver_foil": _principled("silver_foil_MLI", (0.82, 0.83, 0.86), 1.0, 0.28, bump=0.3, bump_scale=30.0),
        "black_kapton": _principled("black_kapton", (0.03, 0.03, 0.035), 0.2, 0.45),
        "white_paint": _principled("white_paint", (0.86, 0.86, 0.84), 0.0, 0.55),
        "aluminium": _principled("aluminium", (0.72, 0.73, 0.75), 1.0, 0.35),
        "titanium": _principled("titanium_nozzle", (0.3, 0.29, 0.28), 1.0, 0.42),
        "nozzle_inner": _principled("nozzle_inner_niobium", (0.12, 0.1, 0.09), 0.9, 0.5),
        "solar_cell": _principled("solar_cell", (0.025, 0.04, 0.12), 0.35, 0.18),
        "panel_back": _principled("panel_substrate", (0.18, 0.18, 0.2), 0.3, 0.6),
        "glass": _principled("optics_glass", (0.02, 0.03, 0.05), 0.0, 0.05),
        "orange": _principled("instrument_orange", (0.85, 0.42, 0.1), 0.1, 0.5),
        "red_tag": _principled("remove_before_flight", (0.75, 0.05, 0.05), 0.0, 0.6),
        "wheel": _principled("wheel_aluminium", (0.6, 0.6, 0.62), 0.9, 0.45),
        "regolith": _principled("regolith", (0.36, 0.35, 0.34), 0.0, 0.95, bump=0.6, bump_scale=6.0),
        "moon": _principled("moon_far", (0.42, 0.41, 0.4), 0.0, 0.95, bump=0.5, bump_scale=3.0),
    })

# ---------------------------------------------------------------------------------------------------------------- geometry helpers


def _obj(name, bm, mat, parent=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(MAT[mat])
    bpy.context.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def _align(direction) -> Matrix:
    d = Vector(direction).normalized()
    return d.to_track_quat("Z", "Y").to_matrix().to_4x4()


def box(name, center, size, mat, parent=None, rot: Matrix | None = None, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    if rot is not None:
        bmesh.ops.transform(bm, matrix=rot, verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(center), verts=bm.verts)
    ob = _obj(name, bm, mat, parent)
    if bevel:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments = bevel, 2
    return ob


def cyl(name, base, direction, length, r0, r1, mat, parent=None, seg=32, smooth=True):
    """Closed frustum from `base` along `direction`."""
    bm = bmesh.new()
    m = Matrix.Translation(Vector(base)) @ _align(direction) @ Matrix.Translation((0, 0, length / 2))
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r0, radius2=r1, depth=length, matrix=m)
    ob = _obj(name, bm, mat, parent)
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = len(p.vertices) == 4
    return ob


def rod(name, p0, p1, r, mat, parent=None, seg=12):
    p0, p1 = Vector(p0), Vector(p1)
    return cyl(name, p0, p1 - p0, (p1 - p0).length, r, r, mat, parent, seg)


def sphere(name, center, r, mat, parent=None, seg=24, scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(center), verts=bm.verts)
    ob = _obj(name, bm, mat, parent)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def lathe(name, profile, mat, matrix: Matrix, parent=None, seg=48, cap=False):
    """Surface of revolution about local +z from [(r, z), ...]; open shell (nozzles, dishes) unless cap."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z)) for k in range(seg)])
    for a, b in zip(rings[:-1], rings[1:], strict=True):
        for k in range(seg):
            bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    if cap:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
    ob = _obj(name, bm, mat, parent)
    for p in ob.data.polygons:
        p.use_smooth = True
    sol = ob.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness = 0.008
    return ob


def panel(name, center, w, h, rot: Matrix, parent=None, cells=(6, 4), gap=0.012):
    """Solar panel: substrate + frame + a cell grid (one mesh for the cells)."""
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    root.parent = parent
    root.matrix_local = Matrix.Translation(Vector(center)) @ rot
    box(name + "_substrate", (0, 0, -0.012), (w, h, 0.02), "panel_back", root)
    for i, (cx, cy, sx, sy) in enumerate([(0, h / 2, w + 0.03, 0.03), (0, -h / 2, w + 0.03, 0.03), (w / 2, 0, 0.03, h), (-w / 2, 0, 0.03, h)]):
        box(f"{name}_frame{i}", (cx, cy, 0.0), (sx, sy, 0.035), "aluminium", root)
    bm = bmesh.new()
    cw, ch = (w - gap) / cells[0], (h - gap) / cells[1]
    for i in range(cells[0]):
        for j in range(cells[1]):
            c = Vector((-w / 2 + gap / 2 + cw * (i + 0.5), -h / 2 + gap / 2 + ch * (j + 0.5), 0.002))
            g = bmesh.new()
            bmesh.ops.create_cube(g, size=1.0)
            bmesh.ops.scale(g, vec=Vector((cw - gap, ch - gap, 0.006)), verts=g.verts)
            bmesh.ops.translate(g, vec=c, verts=g.verts)
            tmp = bpy.data.meshes.new("tmp")
            g.to_mesh(tmp)
            g.free()
            bm.from_mesh(tmp)
            bpy.data.meshes.remove(tmp)
    _obj(name + "_cells", bm, "solar_cell", root)
    return root


def empty(name, parent=None, loc=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(e)
    e.parent = parent
    e.location = loc
    return e


# solar wing orientation: panel width along the boom (+y), height vertical (+z), cells facing +x (the vehicle front)
WING_ROT = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def Rz(deg):
    return Matrix.Rotation(math.radians(deg), 4, "Z")


def Rx(deg):
    return Matrix.Rotation(math.radians(deg), 4, "X")


def Ry(deg):
    return Matrix.Rotation(math.radians(deg), 4, "Y")

# ---------------------------------------------------------------------------------------------------------------- lander B


BELL = [(0.055, 0.30), (0.05, 0.26), (0.065, 0.2), (0.1, 0.12), (0.14, 0.05), (0.18, 0.0)]     # (r, z) throat at top, exit at z=0


def engine(name, exit_pos, thrust_dir, parent):
    """Main engine: bell nozzle (exit plane at exit_pos), combustion chamber, valve block, heat-shield collar."""
    plume = -Vector(thrust_dir).normalized()
    m = Matrix.Translation(Vector(exit_pos)) @ _align(-plume)              # local +z points up the nozzle (towards the lander)
    e = empty(name, parent)
    e["pivot"] = tuple(exit_pos)                                                 # plume attach point (nozzle exit centre)
    lathe(name + "_bell", BELL, "titanium", m, e)
    lathe(name + "_bell_inner", [(r - 0.006, z) for r, z in BELL], "nozzle_inner", m, e)
    top = Vector(exit_pos) - plume * 0.30
    cyl(name + "_chamber", top, -plume, 0.14, 0.06, 0.06, "aluminium", e, 24)
    box(name + "_valves", top - plume * 0.17, (0.11, 0.09, 0.07), "aluminium", e)
    cyl(name + "_collar", top - plume * 0.02, -plume, 0.03, 0.12, 0.12, "silver_foil", e, 32)
    return e


def leg(name, sx, sy, parent):
    """Landing leg: primary strut with telescoping crushable damper, two secondary struts (A-frame), footpad with ribs."""
    e = empty(name, parent)
    foot = Vector((1.55 * sx, 1.55 * sy, -1.38))
    hip = Vector((0.82 * sx, 0.82 * sy, -0.55))
    mid = hip.lerp(foot, 0.5)
    rod(name + "_primary_upper", hip, mid, 0.055, "aluminium", e, 20)
    rod(name + "_primary_lower", mid, foot, 0.038, "aluminium", e, 20)          # telescoping damper: thinner lower tube
    cyl(name + "_damper_ring", mid, foot - hip, 0.06, 0.065, 0.065, "black_kapton", e, 20)
    for k, (ax, ay) in enumerate([(0.95 * sx, 0.35 * sy), (0.35 * sx, 0.95 * sy)]):
        rod(f"{name}_secondary{k}", (ax, ay, -0.85), foot + Vector((0, 0, 0.08)), 0.026, "aluminium", e, 12)
    lathe(name + "_footpad", [(0.03, 0.0), (0.22, 0.0), (0.26, 0.03), (0.24, 0.07), (0.06, 0.1)], "white_paint",
          Matrix.Translation(foot - Vector((0, 0, 0.07))), e, 40, cap=True)
    rod(name + "_ball_joint", foot + Vector((0, 0, 0.02)), foot + Vector((0, 0, 0.09)), 0.045, "aluminium", e, 16)
    for k in range(6):                                                           # contact probe / ribs
        a = 2 * math.pi * k / 6
        box(f"{name}_rib{k}", foot + Vector((0.14 * math.cos(a), 0.14 * math.sin(a), -0.06)), (0.16, 0.012, 0.02), "aluminium", e, Rz(math.degrees(a)))
    return e


def build_lander(root_name="lander"):
    root = empty(root_name)
    # body: octagonal prism, MLI on the sides, radiator top, structural ring and bottom deck
    cyl("deck_body", (0, 0, -0.85), (0, 0, 1), 1.2, 1.05, 1.05, "gold_foil", root, seg=8, smooth=False).rotation_euler.z = math.radians(22.5)
    cyl("top_radiator", (0, 0, 0.35), (0, 0, 1), 0.05, 1.04, 1.0, "silver_foil", root, seg=8, smooth=False).rotation_euler.z = math.radians(22.5)
    cyl("bottom_deck", (0, 0, -0.92), (0, 0, 1), 0.07, 1.0, 1.05, "aluminium", root, seg=8, smooth=False).rotation_euler.z = math.radians(22.5)
    cyl("interface_ring", (0, 0, -1.0), (0, 0, 1), 0.08, 0.62, 0.62, "aluminium", root, seg=48)
    for k in range(4):                                                           # propellant tank domes showing through the bottom
        a = math.radians(45 + 90 * k)
        sphere(f"tank_dome{k}", (0.45 * math.cos(a), 0.45 * math.sin(a), -0.93), 0.24, "silver_foil", root, scale=(1, 1, 0.45))
    # main engines (match the simulator: exits at z=-1.0, 0.8 m ring, 5 deg cant)
    c, s = math.cos(math.radians(5)), math.sin(math.radians(5))
    engine("engine_centre", (0, 0, -1.0), (0, 0, 1), root)
    for k, (ex, ey) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        engine(f"engine_outer{k}", (0.8 * ex, 0.8 * ey, -1.0), (-s * ex, -s * ey, c), root)
    for k, (sx, sy) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        leg(f"leg{k}", sx, sy, root)
    # attitude thrusters: 8 around the upper rim, each a small bell on a bracket
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        d = Vector((math.cos(a), math.sin(a), 0))
        base = d * 0.98 + Vector((0, 0, 0.22))
        box(f"rcs{k}_bracket", base, (0.08, 0.08, 0.06), "aluminium", root, Rz(math.degrees(a)))
        lathe(f"rcs{k}_nozzle", [(0.015, 0.0), (0.02, 0.03), (0.035, 0.08)], "titanium", Matrix.Translation(base + d * 0.03) @ _align(d), root, 20)
    # solar wings: yoke, hinge, two-panel wings either side (+y / -y)
    for sgn, tag in ((1, "port"), (-1, "starboard")):
        rod(f"wing_{tag}_yoke_a", (0.0, sgn * 0.98, 0.25), (0.0, sgn * 1.45, 0.2), 0.022, "aluminium", root)
        rod(f"wing_{tag}_yoke_b", (0.0, sgn * 0.98, -0.35), (0.0, sgn * 1.45, -0.3), 0.022, "aluminium", root)
        cyl(f"wing_{tag}_hinge", (0.0, sgn * 1.46, -0.35), (0, 0, 1), 0.6, 0.03, 0.03, "black_kapton", root, 16)
        panel(f"wing_{tag}_panel1", (0, sgn * 2.0, -0.05), 1.05, 1.0, WING_ROT, root, cells=(6, 6))
        panel(f"wing_{tag}_panel2", (0, sgn * 3.04, -0.05), 1.05, 1.0, WING_ROT, root, cells=(6, 6))
    # top deck equipment
    rod("antenna_mast", (0.45, 0.45, 0.4), (0.45, 0.45, 1.05), 0.03, "aluminium", root)
    box("antenna_gimbal", (0.45, 0.45, 1.07), (0.1, 0.1, 0.08), "black_kapton", root)
    dish_m = Matrix.Translation((0.45, 0.45, 1.1)) @ _align((0.35, 0.25, 1.0))
    lathe("antenna_dish", [(0.0, 0.0), (0.1, 0.012), (0.2, 0.045), (0.3, 0.1), (0.34, 0.13)], "white_paint", dish_m, root, 48)
    rod("antenna_feed", dish_m @ Vector((0, 0, 0)), dish_m @ Vector((0, 0, 0.3)), 0.012, "aluminium", root)
    for k, (x, y, d) in enumerate([(0.65, -0.55, (0.5, -0.4, 0.8)), (-0.65, -0.55, (-0.5, -0.4, 0.8))]):
        box(f"star_tracker{k}_body", (x, y, 0.47), (0.14, 0.14, 0.18), "black_kapton", root)
        cyl(f"star_tracker{k}_baffle", (x, y, 0.55), d, 0.2, 0.05, 0.08, "black_kapton", root, 24)
    for k, (x, y) in enumerate([(-0.4, 0.6), (0.0, -0.75), (-0.7, 0.1)]):
        box(f"sun_sensor{k}", (x, y, 0.41), (0.07, 0.07, 0.03), "black_kapton", root)
        box(f"sun_sensor{k}_window", (x, y, 0.428), (0.045, 0.045, 0.004), "glass", root)
    rod("whip_antenna", (-0.55, -0.35, 0.38), (-0.55, -0.35, 0.95), 0.007, "aluminium", root)
    box("radiator_fins", (-0.2, 0.15, 0.4), (0.7, 0.5, 0.04), "silver_foil", root)
    # side face instruments: hazard detection cameras, landing position camera, laser altimeter + doppler velocimeter
    for k, z in enumerate((-0.05, -0.35)):
        box(f"hazard_cam{k}", (0.98, -0.25 + 0.5 * k, z), (0.08, 0.16, 0.12), "black_kapton", root)
        cyl(f"hazard_cam{k}_lens", (1.02, -0.25 + 0.5 * k, z), (1, 0, -0.4), 0.05, 0.035, 0.03, "glass", root, 24)
    box("position_camera", (0.7, 0.72, -0.55), (0.1, 0.1, 0.1), "black_kapton", root, Rz(45))
    cyl("laser_altimeter", (0.3, -0.5, -0.99), (0, 0, -1), 0.06, 0.06, 0.06, "black_kapton", root, 24)
    cyl("laser_altimeter_window", (0.3, -0.5, -1.05), (0, 0, -1), 0.005, 0.045, 0.045, "glass", root, 24)
    cyl("doppler_velocimeter", (-0.3, 0.5, -0.99), (0, 0, -1), 0.07, 0.08, 0.08, "black_kapton", root, 24)
    cyl("doppler_velocimeter_window", (-0.3, 0.5, -1.06), (0, 0, -1), 0.005, 0.06, 0.06, "glass", root, 24)
    # rover ramp stowed on the -x face, with hinges and latches; remove-before-flight tags
    box("ramp_panel", (-1.0, 0.0, -0.25), (0.05, 0.8, 1.0), "silver_foil", root, bevel=0.01)
    for k, y in enumerate((-0.3, 0.3)):
        cyl(f"ramp_hinge{k}", (-1.0, y - 0.06, -0.78), (0, 1, 0), 0.12, 0.025, 0.025, "aluminium", root, 16)
        box(f"ramp_latch{k}", (-1.03, y, 0.22), (0.04, 0.08, 0.05), "aluminium", root)
        box(f"rbf_tag{k}", (-1.06, y, 0.12), (0.005, 0.03, 0.12), "red_tag", root)
    for k in range(8):                                                           # harness runs on the corners
        a = math.radians(22.5 + 45 * k)
        rod(f"harness{k}", (1.0 * math.cos(a), 1.0 * math.sin(a), 0.3), (1.0 * math.cos(a), 1.0 * math.sin(a), -0.8), 0.01, "black_kapton", root, 8)
    return root

# ---------------------------------------------------------------------------------------------------------------- rover B


def wheel(name, center, parent):
    e = empty(name, parent)
    c = Vector(center)
    cyl(name + "_rim", c - Vector((0, 0.045, 0)), (0, 1, 0), 0.09, 0.085, 0.085, "wheel", e, 40)
    cyl(name + "_hub", c - Vector((0, 0.06, 0)), (0, 1, 0), 0.12, 0.025, 0.025, "aluminium", e, 16)
    for k in range(6):                                                           # spokes on the outer face
        a = 2 * math.pi * k / 6
        rod(f"{name}_spoke{k}", c + Vector((0, 0.05 * (1 if center[1] > 0 else -1), 0)),
            c + Vector((0.075 * math.cos(a), 0.05 * (1 if center[1] > 0 else -1), 0.075 * math.sin(a))), 0.006, "aluminium", e, 6)
    for k in range(14):                                                          # chevron grousers
        a = 2 * math.pi * k / 14
        box(f"{name}_grouser{k}", c + Vector((0.089 * math.cos(a), 0, 0.089 * math.sin(a))), (0.012, 0.088, 0.014), "aluminium", e,
            Ry(-math.degrees(a)))
    return e


def build_rover(root_name="rover"):
    root = empty(root_name)
    box("chassis", (0, 0, 0.0), (0.74, 0.54, 0.22), "gold_foil", root, bevel=0.01)
    box("chassis_top_deck", (0, 0, 0.115), (0.7, 0.5, 0.01), "aluminium", root)
    # Sun-tilted solar panel on two rear hinges with a support rod
    panel("tilt_panel", (-0.06, 0, 0.27), 0.68, 0.6, Ry(-24), root, cells=(6, 5))
    for k, y in enumerate((0.22, -0.22)):
        cyl(f"panel_hinge{k}", (-0.36, y - 0.03, 0.17), (0, 1, 0), 0.06, 0.015, 0.015, "aluminium", root, 12)
        rod(f"panel_hinge_arm{k}", (-0.36, y, 0.12), (-0.36, y, 0.18), 0.012, "aluminium", root)
    rod("panel_support_rod", (0.18, 0.0, 0.12), (0.2, 0.0, 0.39), 0.01, "aluminium", root)
    # stereo navigation camera mast with pan-tilt head
    rod("nav_mast", (0.27, 0.0, 0.12), (0.27, 0.0, 0.36), 0.016, "aluminium", root)
    box("pan_tilt", (0.27, 0.0, 0.38), (0.05, 0.06, 0.04), "black_kapton", root)
    box("navcam_bar", (0.29, 0.0, 0.42), (0.05, 0.22, 0.05), "black_kapton", root)
    for k, y in enumerate((0.075, -0.075)):
        cyl(f"navcam{k}_lens", (0.315, y, 0.42), (1, 0, 0), 0.025, 0.018, 0.016, "glass", root, 20)
    # hazard cameras front and rear
    for k, (x, d) in enumerate(((0.375, 1), (-0.375, -1))):
        box(f"hazcam{k}", (x, 0, -0.04), (0.02, 0.18, 0.05), "black_kapton", root)
        for j, y in enumerate((0.05, -0.05)):
            cyl(f"hazcam{k}_lens{j}", (x + 0.01 * d, y, -0.04), (d, 0, -0.3), 0.015, 0.013, 0.012, "glass", root, 16)
    # folded science arm with spectrometer head (APXS-like), LIBS window on the front
    rod("arm_shoulder", (0.37, -0.18, 0.02), (0.42, -0.05, 0.02), 0.012, "aluminium", root)
    rod("arm_forearm", (0.42, -0.05, 0.02), (0.44, 0.12, -0.04), 0.011, "aluminium", root)
    box("spectrometer_head", (0.45, 0.15, -0.055), (0.06, 0.06, 0.06), "orange", root, bevel=0.005)
    cyl("libs_window", (0.372, 0.18, 0.05), (1, 0, 0), 0.01, 0.025, 0.025, "glass", root, 20)
    # antennas, radiator, harness
    rod("whip_antenna", (-0.33, 0.24, 0.12), (-0.33, 0.24, 0.36), 0.006, "aluminium", root)
    box("patch_antenna", (-0.28, -0.2, 0.125), (0.08, 0.08, 0.01), "white_paint", root)
    box("radiator", (0.0, -0.272, 0.0), (0.4, 0.004, 0.12), "silver_foil", root)
    # rocker-bogie suspension + differential bar
    for sgn in (1, -1):
        y = 0.285 * sgn
        rod(f"rocker_{sgn}", (-0.37, y, -0.12), (0.0, y, 0.0), 0.014, "aluminium", root)
        rod(f"bogie_front_{sgn}", (0.0, y, 0.0), (0.37, y, -0.12), 0.014, "aluminium", root)
        rod(f"bogie_mid_{sgn}", (0.05, y, -0.02), (0.0, y, -0.12), 0.012, "aluminium", root)
        cyl(f"rocker_pivot_{sgn}", (0.0, y - 0.03, 0.0), (0, 1, 0), 0.06, 0.022, 0.022, "black_kapton", root, 16)
    rod("differential_bar", (0.0, -0.27, 0.13), (0.0, 0.27, 0.13), 0.009, "aluminium", root)
    for sx in (-1, 0, 1):
        for sy in (-1, 1):
            wheel(f"wheel_{'RMF'[sx + 1]}{'L' if sy > 0 else 'R'}", (0.37 * sx, 0.32 * sy, -0.12), root)   # 0.75 m track (Pragyan width)
    return root

# ---------------------------------------------------------------------------------------------------------------- propulsion module B


def build_pm(root_name="propulsion_module"):
    """Cylindrical propulsion module: origin at its top separation plane. Carries the lander to lunar orbit, then separates."""
    root = empty(root_name)
    cyl("pm_body", (0, 0, -2.1), (0, 0, 1), 2.0, 1.1, 1.1, "gold_foil", root, 32)
    cyl("pm_top_deck", (0, 0, -0.1), (0, 0, 1), 0.06, 1.1, 1.08, "silver_foil", root, 32)
    cyl("pm_bottom_deck", (0, 0, -2.16), (0, 0, 1), 0.06, 1.05, 1.1, "aluminium", root, 32)
    for k in range(3):
        cyl(f"pm_band{k}", (0, 0, -1.7 + 0.6 * k), (0, 0, 1), 0.035, 1.115, 1.115, "aluminium", root, 32)
    # separation system: ring, clamp band, 4 spring pushers, pyro bolts
    cyl("separation_ring_lower", (0, 0, -0.04), (0, 0, 1), 0.04, 0.66, 0.66, "aluminium", root, 48)
    cyl("clamp_band", (0, 0, -0.02), (0, 0, 1), 0.03, 0.68, 0.68, "black_kapton", root, 48)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        p = Vector((0.55 * math.cos(a), 0.55 * math.sin(a), 0.0))
        cyl(f"spring_pusher{k}_housing", p - Vector((0, 0, 0.12)), (0, 0, 1), 0.12, 0.05, 0.05, "aluminium", root, 16)
        cyl(f"spring_pusher{k}_plunger", p, (0, 0, 1), 0.05, 0.025, 0.025, "orange", root, 12)
        a2 = math.radians(90 * k)
        box(f"pyro_bolt{k}", (0.66 * math.cos(a2), 0.66 * math.sin(a2), -0.01), (0.05, 0.05, 0.05), "red_tag", root)
    # main engine (440 N class) with gimbal-less bell, tanks, plumbing
    lathe("pm_main_engine_bell", [(0.07, 0.4), (0.065, 0.34), (0.1, 0.25), (0.17, 0.12), (0.23, 0.0)], "titanium",
          Matrix.Translation((0, 0, -2.75)), root, 48)
    cyl("pm_engine_chamber", (0, 0, -2.36), (0, 0, -1), 0.2, 0.09, 0.09, "aluminium", root, 24)
    for k in range(2):
        sphere(f"pm_tank_dome{k}", (0.45 * (1 if k else -1), 0, -2.17), 0.35, "silver_foil", root, scale=(1, 1, 0.35))
    # RCS clusters on the PM
    for k in range(4):
        a = math.radians(90 * k)
        d = Vector((math.cos(a), math.sin(a), 0))
        box(f"pm_rcs{k}_block", d * 1.12 + Vector((0, 0, -1.95)), (0.1, 0.1, 0.1), "aluminium", root, Rz(math.degrees(a)))
        for j, dd in enumerate((d, Vector((0, 0, -1)))):
            lathe(f"pm_rcs{k}_nozzle{j}", [(0.012, 0.0), (0.018, 0.025), (0.03, 0.06)], "titanium",
                  Matrix.Translation(d * 1.12 + Vector((0, 0, -1.95)) + dd * 0.05) @ _align(dd), root, 16)
    # twin solar wings (deployed), yokes and drives
    for sgn, tag in ((1, "port"), (-1, "starboard")):
        box(f"pm_wing_{tag}_drive", (0, sgn * 1.15, -1.1), (0.16, 0.1, 0.16), "black_kapton", root)
        rod(f"pm_wing_{tag}_yoke", (0, sgn * 1.2, -1.1), (0, sgn * 1.55, -1.1), 0.03, "aluminium", root)
        for i in range(2):
            panel(f"pm_wing_{tag}_panel{i}", (0, sgn * (2.2 + 1.3 * i), -1.1), 1.25, 1.2, WING_ROT, root, cells=(7, 7))
    # star tracker, antenna, harness
    box("pm_star_tracker", (0.6, -0.75, -0.05), (0.14, 0.14, 0.16), "black_kapton", root)
    cyl("pm_star_tracker_baffle", (0.6, -0.75, 0.02), (0.4, -0.4, 0.8), 0.18, 0.05, 0.08, "black_kapton", root, 24)
    lathe("pm_antenna", [(0.0, 0.0), (0.12, 0.02), (0.22, 0.06)], "white_paint", Matrix.Translation((-0.7, 0.7, -0.05)) @ _align((0, 0, 1)), root, 32)
    for k in range(6):
        a = math.radians(30 + 60 * k)
        rod(f"pm_harness{k}", (1.105 * math.cos(a), 1.105 * math.sin(a), -0.2), (1.105 * math.cos(a), 1.105 * math.sin(a), -2.0), 0.01, "black_kapton", root, 8)
    return root

# ---------------------------------------------------------------------------------------------------------------- detail pass (batched meshes)


class Batch:
    """Many small parts in ONE mesh object (fast, few objects): boxes, frusta, rods, spheres with full transforms."""

    def __init__(self, name, mat, parent=None):
        self.name, self.mat, self.parent, self.bm = name, mat, parent, bmesh.new()

    def box(self, center, size, rot=None):
        m = Matrix.Translation(Vector(center)) @ (rot if rot is not None else Matrix.Identity(4)) @ Matrix.Diagonal((*size, 1.0))
        bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        return self

    def cyl(self, base, direction, length, r0, r1, seg=16):
        m = Matrix.Translation(Vector(base)) @ _align(direction) @ Matrix.Translation((0, 0, length / 2))
        bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r0, radius2=r1, depth=length, matrix=m)
        return self

    def rod(self, p0, p1, r, seg=10):
        p0, p1 = Vector(p0), Vector(p1)
        return self.cyl(p0, p1 - p0, (p1 - p0).length, r, r, seg)

    def sphere(self, c, r, seg=12, scale=(1, 1, 1)):
        m = Matrix.Translation(Vector(c)) @ Matrix.Diagonal((*scale, 1.0))
        bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=max(seg // 2, 4), radius=r, matrix=m)
        return self

    def pipe(self, pts, r, seg=10):
        pts = [Vector(p) for p in pts]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            self.rod(a, b, r, seg)
        for p in pts[1:-1]:
            self.sphere(p, r * 1.15, seg)
        return self

    def done(self):
        if not self.bm.verts:
            self.bm.free()
            return None
        ob = _obj(self.name, self.bm, self.mat, self.parent)
        for p in ob.data.polygons:
            p.use_smooth = len(p.vertices) == 4
        return ob


def panel(name, center, w, h, rot: Matrix, parent=None, cells=(6, 4), gap=0.012):  # noqa: F811 - detailed version replaces the first one
    """Solar panel: substrate, frame, cell grid with bus bars, back ribs, corner fittings."""
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    root.parent = parent
    root.matrix_local = Matrix.Translation(Vector(center)) @ rot
    box(name + "_substrate", (0, 0, -0.012), (w, h, 0.02), "panel_back", root)
    fr = Batch(name + "_frame", "aluminium", root)
    for cx, cy, sx, sy in [(0, h / 2, w + 0.03, 0.03), (0, -h / 2, w + 0.03, 0.03), (w / 2, 0, 0.03, h), (-w / 2, 0, 0.03, h)]:
        fr.box((cx, cy, 0.0), (sx, sy, 0.035))
    for sx in (-1, 1):
        for sy in (-1, 1):
            fr.box((sx * w / 2, sy * h / 2, -0.01), (0.06, 0.06, 0.05))           # corner fittings / hold-down points
    fr.done()
    ribs = Batch(name + "_back_ribs", "aluminium", root)
    for k in range(3):
        ribs.box((-w / 3 + k * w / 3, 0, -0.035), (0.02, h * 0.96, 0.03))
    ribs.box((0, 0, -0.035), (w * 0.96, 0.02, 0.03))
    ribs.done()
    cb = Batch(name + "_cells", "solar_cell", root)
    bus = Batch(name + "_busbars", "aluminium", root)
    cw, ch = (w - gap) / cells[0], (h - gap) / cells[1]
    for i in range(cells[0]):
        for j in range(cells[1]):
            c = Vector((-w / 2 + gap / 2 + cw * (i + 0.5), -h / 2 + gap / 2 + ch * (j + 0.5), 0.002))
            cb.box(c, (cw - gap, ch - gap, 0.006))
            for t in (-0.25, 0.25):
                bus.box(c + Vector((t * (cw - gap), 0, 0.0035)), (0.003, ch - gap, 0.001))
    cb.done()
    bus.done()
    return root


def lander_details(root):
    R = 1.05
    AP = R * math.cos(math.radians(22.5))
    seams = Batch("mli_seams_tape", "silver_foil", root)
    for k in range(8):
        a = math.radians(45 * k)
        if k == 4:                                                               # -x face = rover bay door
            continue
        n, t = Vector((math.cos(a), math.sin(a), 0)), Vector((-math.sin(a), math.cos(a), 0))
        for off in (-0.2, 0.2):
            seams.box(n * (AP + 0.004) + t * off + Vector((0, 0, -0.25)), (0.006, 0.018, 1.18), Rz(45 * k))
        for z in (-0.6, 0.1):
            seams.box(n * (AP + 0.004) + Vector((0, 0, z)), (0.006, 0.78, 0.018), Rz(45 * k))
    seams.done()
    lon = Batch("corner_longerons", "aluminium", root)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        p = Vector((R * math.cos(a), R * math.sin(a), 0))
        lon.rod(p + Vector((0, 0, -0.85)), p + Vector((0, 0, 0.37)), 0.024, 8)
        lon.box(p * 0.97 + Vector((0, 0, 0.42)), (0.09, 0.05, 0.06), Rz(math.degrees(a)))      # lifting lugs
    lon.done()
    bolts = Batch("top_bolt_ring", "aluminium", root)
    for k in range(40):
        a = 2 * math.pi * k / 40
        bolts.cyl((0.93 * math.cos(a), 0.93 * math.sin(a), 0.4), (0, 0, 1), 0.012, 0.012, 0.012, 6)
    bolts.done()
    # optical solar reflector radiators on two side faces (normals at 135 and 225 deg)
    base, tiles = Batch("radiator_backing", "black_kapton", root), Batch("radiator_osr_tiles", "silver_foil", root)
    for deg in (135, 225):
        a = math.radians(deg)
        n, t = Vector((math.cos(a), math.sin(a), 0)), Vector((-math.sin(a), math.cos(a), 0))
        base.box(n * (AP + 0.006) + Vector((0, 0, -0.2)), (0.004, 0.62, 0.74), Rz(deg))
        for i in range(4):
            for j in range(5):
                tiles.box(n * (AP + 0.01) + t * (-0.225 + 0.15 * i) + Vector((0, 0, -0.5 + 0.145 * j)), (0.004, 0.14, 0.135), Rz(deg))
    base.done()
    tiles.done()
    # underside: nozzle heat shields, pressurant tanks, feed lines, radar altimeter, landing camera
    under = Batch("underside_heat_shields", "black_kapton", root)
    for x, y in [(0, 0), (0.8, 0), (0, 0.8), (-0.8, 0), (0, -0.8)]:
        under.cyl((x, y, -0.945), (0, 0, 1), 0.012, 0.25, 0.25, 32)
    under.done()
    copv = Batch("pressurant_tanks_COPV", "white_paint", root)
    pipes = Batch("feed_lines", "aluminium", root)
    for sx in (1, -1):
        c = Vector((0.55 * sx, 0.32 * sx, -0.98))
        copv.sphere(c, 0.11, 20)
        copv.cyl(c, (0, 0, 1), 0.04, 0.12, 0.12, 20)
        pipes.pipe([c + Vector((0, 0, 0.05)), Vector((0.25 * sx, 0.12 * sx, -0.95)), Vector((0.06 * sx, 0.06 * sx, -0.95))], 0.012)
        for ex, ey in [(0.8, 0), (0, 0.8), (-0.8, 0), (0, -0.8)]:
            if ex * sx > 0 or ey * sx > 0:
                pipes.pipe([Vector((0.06 * sx, 0.06 * sx, -0.95)), Vector((ex * 0.55, ey * 0.55, -0.95)), Vector((ex * 0.78, ey * 0.78, -0.95))], 0.009)
    copv.done()
    pipes.done()
    box("radar_altimeter_antenna", (-0.5, -0.25, -0.95), (0.22, 0.16, 0.035), "white_paint", root)
    grid = Batch("radar_altimeter_patches", "aluminium", root)
    for i in range(4):
        for j in range(3):
            grid.box((-0.5 - 0.075 + 0.05 * i, -0.25 - 0.05 + 0.05 * j, -0.97), (0.03, 0.03, 0.004))
    grid.done()
    box("landing_camera_down", (0.5, 0.5, -0.95), (0.1, 0.1, 0.06), "black_kapton", root)
    cyl("landing_camera_down_lens", (0.5, 0.5, -0.98), (0, 0, -1), 0.03, 0.03, 0.025, "glass", root, 20)
    # legs: hip clevis, crush cartridge rings, strut end fittings
    fit = Batch("leg_fittings", "aluminium", root)
    rings = Batch("leg_crush_rings", "black_kapton", root)
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        foot, hip = Vector((1.55 * sx, 1.55 * sy, -1.38)), Vector((0.82 * sx, 0.82 * sy, -0.55))
        d = (foot - hip).normalized()
        fit.box(hip - d * 0.02, (0.13, 0.13, 0.1), Rz(45 * (1 if sx * sy > 0 else -1)))
        for t in (0.56, 0.62, 0.68, 0.74):
            rings.cyl(hip.lerp(foot, t), d, 0.014, 0.046, 0.046, 16)
        for ax, ay in [(0.95 * sx, 0.35 * sy), (0.35 * sx, 0.95 * sy)]:
            fit.sphere((ax, ay, -0.85), 0.035, 10)
        fit.sphere(foot + Vector((0, 0, 0.08)), 0.04, 10)
    fit.done()
    rings.done()
    # attitude thrusters: second (downward) nozzle + valve per cluster
    rcs = Batch("rcs_down_nozzles", "titanium", root)
    valves = Batch("rcs_valves", "aluminium", root)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        d = Vector((math.cos(a), math.sin(a), 0))
        b = d * 0.98 + Vector((0, 0, 0.22))
        rcs.cyl(b + Vector((0, 0, -0.03)), (0, 0, -1), 0.07, 0.015, 0.032, 14)
        valves.box(b - d * 0.05 + Vector((0, 0, 0.04)), (0.04, 0.04, 0.03), Rz(math.degrees(a)))
    rcs.done()
    valves.done()
    # wings: inter-panel hinge brackets and harness along the yokes
    hw = Batch("wing_hinges", "aluminium", root)
    hn = Batch("wing_harness", "black_kapton", root)
    for sgn in (1, -1):
        for dz in (-0.38, 0.38):
            hw.box((0.0, sgn * 2.52, -0.05 + dz), (0.05, 0.06, 0.08))
        hn.rod((0.05, sgn * 0.98, -0.1), (0.05, sgn * 1.5, -0.1), 0.009, 8)
    hw.done()
    hn.done()
    # dish: three feed struts + subreflector
    dm = Matrix.Translation((0.45, 0.45, 1.1)) @ _align((0.35, 0.25, 1.0))
    st = Batch("dish_feed_struts", "aluminium", root)
    for k in range(3):
        a = 2 * math.pi * k / 3
        st.rod(dm @ Vector((0.3 * math.cos(a), 0.3 * math.sin(a), 0.1)), dm @ Vector((0, 0, 0.3)), 0.006, 6)
    st.cyl(dm @ Vector((0, 0, 0.3)), dm.to_3x3() @ Vector((0, 0, 1)), 0.02, 0.06, 0.05, 24)
    st.done()
    # top-deck science: retroreflector, seismometer, stowed thermal probe, plasma probe booms, low-gain antennas
    ins = Batch("science_instruments", "white_paint", root)
    ins.box((0.1, -0.45, 0.42), (0.14, 0.14, 0.09))
    for k in range(3):
        a = 2 * math.pi * k / 3
        ins.rod((0.1 + 0.07 * math.cos(a), -0.45 + 0.07 * math.sin(a), 0.38), (0.1 + 0.08 * math.cos(a), -0.45 + 0.08 * math.sin(a), 0.4), 0.008, 6)
    ins.box((0.8, 0.15, 0.42), (0.1, 0.08, 0.06))                                  # thermal probe drive
    ins.done()
    cyl("retroreflector_base", (-0.55, 0.45, 0.38), (0, 0, 1), 0.035, 0.07, 0.07, "aluminium", root, 24)
    sphere("retroreflector_dome", (-0.55, 0.45, 0.415), 0.06, "glass", root, 24, scale=(1, 1, 0.5))
    rod("thermal_probe_rod", (0.8, 0.15, 0.45), (0.98, 0.42, 0.47), 0.012, "aluminium", root)
    cyl("thermal_probe_tip", (0.98, 0.42, 0.47), (0.66, 1.0, 0.07), 0.07, 0.014, 0.0, "orange", root, 12)
    booms = Batch("plasma_probe_booms", "aluminium", root)
    tips = Batch("plasma_probe_tips", "silver_foil", root)
    for deg in (112.5, 292.5):
        a = math.radians(deg)
        d = Vector((math.cos(a), math.sin(a), 0))
        booms.rod(d * 1.0 + Vector((0, 0, 0.3)), d * 1.9 + Vector((0, 0, 0.55)), 0.008, 8)
        tips.sphere(d * 1.92 + Vector((0, 0, 0.56)), 0.035, 16)
    booms.done()
    tips.done()
    lga = Batch("low_gain_antennas", "white_paint", root)
    for x, y in ((-0.2, -0.7), (-0.75, -0.2)):
        lga.cyl((x, y, 0.38), (0, 0, 1), 0.08, 0.025, 0.025, 12).sphere((x, y, 0.47), 0.03, 12)
    lga.done()


def rover_details(root):
    s = Batch("chassis_mli_seams", "silver_foil", root)
    for x in (-0.18, 0.18):
        for sy in (1, -1):
            s.box((x, 0.272 * sy, 0.0), (0.012, 0.004, 0.21))
    for y in (-0.12, 0.12):
        for sx in (1, -1):
            s.box((0.372 * sx, y, 0.0), (0.004, 0.012, 0.21))
    s.done()
    td = Batch("tie_down_fittings", "aluminium", root)
    for sx in (1, -1):
        for sy in (1, -1):
            td.box((0.34 * sx, 0.25 * sy, -0.115), (0.05, 0.05, 0.02)).rod((0.34 * sx, 0.25 * sy, -0.125), (0.34 * sx, 0.25 * sy, -0.16), 0.008, 8)
    td.done()
    bf = Batch("navcam_baffles", "black_kapton", root)
    for y in (0.075, -0.075):
        bf.cyl((0.335, y, 0.42), (1, 0, 0), 0.03, 0.022, 0.026, 16)
    bf.done()
    rod("mast_cable", (0.25, 0.012, 0.12), (0.25, 0.012, 0.36), 0.005, "black_kapton", root)
    j = Batch("arm_joints", "black_kapton", root)
    for p in ((0.37, -0.18, 0.02), (0.42, -0.05, 0.02)):
        j.cyl(Vector(p) - Vector((0, 0, 0.02)), (0, 0, 1), 0.04, 0.018, 0.018, 12)
    j.box((0.35, -0.2, 0.03), (0.04, 0.05, 0.04))
    j.done()
    ant = Batch("rover_antennas", "aluminium", root)
    ant.rod((-0.33, -0.24, 0.12), (-0.33, -0.24, 0.3), 0.005, 6)
    for k in range(6):
        ant.cyl((-0.33, -0.24, 0.3 + 0.012 * k), (0, 0, 1), 0.004, 0.02, 0.02, 12)
    ant.done()
    box("sun_sensor", (0.3, -0.22, 0.125), (0.04, 0.04, 0.015), "black_kapton", root)
    box("sun_sensor_window", (0.3, -0.22, 0.133), (0.025, 0.025, 0.002), "glass", root)


def wheel(name, center, parent):  # noqa: F811 - detailed version replaces the first one
    """Rover wheel: rim with side flanges, hub, spokes, 14 chevron grousers, hub motor (inboard)."""
    e = empty(name, parent)
    c = Vector(center)
    side = 1 if center[1] > 0 else -1
    e["pivot"] = tuple(center)                                                   # spin axis = body y through the wheel centre
    lathe(name + "_tyre", [(0.083, -0.045), (0.086, -0.04), (0.086, 0.04), (0.083, 0.045)], "wheel", Matrix.Translation(c) @ _align((0, 1, 0)), e, 48, cap=False)
    fl = Batch(name + "_flanges", "aluminium", e)
    for dy in (-0.043, 0.043):
        fl.cyl(c + Vector((0, dy - 0.002, 0)), (0, 1, 0), 0.004, 0.078, 0.078, 32)
    fl.cyl(c - Vector((0, 0.06, 0)), (0, 1, 0), 0.12, 0.022, 0.022, 16)
    for k in range(6):
        a = 2 * math.pi * k / 6
        fl.rod(c + Vector((0, 0.044 * side, 0)), c + Vector((0.074 * math.cos(a), 0.044 * side, 0.074 * math.sin(a))), 0.006, 6)
    fl.done()
    gr = Batch(name + "_grousers", "aluminium", e)
    for k in range(14):
        a = 2 * math.pi * k / 14
        rad = Vector((math.cos(a), 0, math.sin(a)))
        for sgn in (1, -1):
            gr.box(c + rad * 0.089 + Vector((0, 0.021 * sgn, 0)), (0.009, 0.046, 0.012), Ry(-math.degrees(a)) @ Rz(18 * sgn))
    gr.done()
    cyl(name + "_hub_motor", c, (0, -side, 0), 0.075, 0.032, 0.032, "black_kapton", e, 20)
    if abs(center[0]) > 0.1:                                                      # corner wheels steer
        cyl(name + "_steering_motor", c + Vector((0, -0.04 * side, 0.06)), (0, 0, 1), 0.06, 0.024, 0.024, "black_kapton", e, 16)
    return e


def pm_details(root):
    st = Batch("pm_stringers", "aluminium", root)
    for k in range(16):
        a = 2 * math.pi * k / 16 + 0.1
        st.box((1.106 * math.cos(a), 1.106 * math.sin(a), -1.1), (0.012, 0.03, 1.95), Rz(math.degrees(a)))
    st.done()
    b = Batch("separation_bolts", "aluminium", root)
    for k in range(24):
        a = 2 * math.pi * k / 24
        b.cyl((0.6 * math.cos(a), 0.6 * math.sin(a), 0.0), (0, 0, 1), 0.02, 0.018, 0.018, 6)
    b.done()
    cl = Batch("vband_clamp_segments", "aluminium", root)
    for k in range(12):
        a = 2 * math.pi * (k + 0.5) / 12
        cl.box((0.695 * math.cos(a), 0.695 * math.sin(a), -0.02), (0.03, 0.26, 0.05), Rz(math.degrees(a)))
    cl.done()
    for k, y in enumerate((0.42, -0.42)):
        box(f"umbilical_plate{k}", (0.3, y, 0.0), (0.12, 0.08, 0.04), "black_kapton", root)
    box("earth_spectrometer", (-0.45, -0.3, -0.0), (0.26, 0.2, 0.16), "white_paint", root)
    cyl("earth_spectrometer_aperture", (-0.45, -0.4, -0.0), (0, -1, 0), 0.08, 0.05, 0.05, "black_kapton", root, 24)
    eq = Batch("pm_equipment_boxes", "silver_foil", root)
    for x, y, sx, sy in ((0.6, 0.3, 0.2, 0.16), (-0.2, 0.62, 0.22, 0.12), (0.25, -0.62, 0.18, 0.14)):
        eq.box((x, y, -0.02), (sx, sy, 0.1))
    eq.done()
    hs = Batch("pm_engine_heat_shield", "black_kapton", root)
    hs.cyl((0, 0, -2.2), (0, 0, 1), 0.012, 0.55, 0.55, 48)
    hs.done()
    fb = Batch("pm_nozzle_flange_bolts", "aluminium", root)
    fb.cyl((0, 0, -2.4), (0, 0, 1), 0.03, 0.13, 0.13, 32)
    for k in range(12):
        a = 2 * math.pi * k / 12
        fb.cyl((0.115 * math.cos(a), 0.115 * math.sin(a), -2.37), (0, 0, 1), 0.01, 0.008, 0.008, 6)
    fb.done()
    cp = Batch("pm_pressurant_tanks", "white_paint", root)
    pp = Batch("pm_feed_lines", "aluminium", root)
    for sy in (1, -1):
        c = Vector((0.0, 0.62 * sy, -2.22))
        cp.sphere(c, 0.13, 20)
        pp.pipe([c + Vector((0, -0.08 * sy, 0.02)), Vector((0.0, 0.25 * sy, -2.2)), Vector((0.0, 0.1 * sy, -2.25))], 0.012)
    for sx in (1, -1):
        pp.pipe([Vector((0.45 * sx, 0, -2.25)), Vector((0.2 * sx, 0.05, -2.28)), Vector((0.08 * sx, 0.0, -2.3))], 0.012)
    cp.done()
    pp.done()
    for k, (x, y) in enumerate(((0.0, 1.106), (0.0, -1.106))):
        box(f"pm_sband_patch{k}", (x + 0.5, y, -0.5), (0.12, 0.01, 0.12), "white_paint", root)
    hw = Batch("pm_wing_hinges", "aluminium", root)
    hn = Batch("pm_wing_harness", "black_kapton", root)
    for sgn in (1, -1):
        for dz in (-0.45, 0.45):
            hw.box((0.0, sgn * 2.85, -1.1 + dz), (0.05, 0.06, 0.08))
        hn.rod((0.05, sgn * 1.2, -1.15), (0.05, sgn * 1.58, -1.15), 0.01, 8)
    hw.done()
    hn.done()


_build_lander_base, _build_rover_base, _build_pm_base = build_lander, build_rover, build_pm


def build_lander(root_name="lander"):  # noqa: F811
    root = _build_lander_base(root_name)
    lander_details(root)
    return root


def build_rover(root_name="rover"):  # noqa: F811
    root = _build_rover_base(root_name)
    rover_details(root)
    return root


def build_pm(root_name="propulsion_module"):  # noqa: F811
    root = _build_pm_base(root_name)
    pm_details(root)
    return root


# ---------------------------------------------------------------------------------------------------------------- scene, export, render


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def children(ob):
    out = [ob]
    for c in ob.children:
        out += children(c)
    return out


def export(root, stem):
    bpy.ops.object.select_all(action="DESELECT")
    for o in children(root):
        if "cutter" not in o.name:
            o.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.gltf(filepath=str(OUT / f"{stem}.glb"), export_format="GLB", use_selection=True, export_apply=True)
    bpy.ops.wm.obj_export(filepath=str(OUT / f"{stem}.obj"), export_selected_objects=True, apply_modifiers=True, forward_axis="Y", up_axis="Z",
                          export_materials=True)


def stats(root):
    deps = bpy.context.evaluated_depsgraph_get()
    tris = 0
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in children(root):
        if o.type != "MESH" or "cutter" in o.name:
            continue
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        tris += len(me.loop_triangles)
        for v in me.vertices:
            w = o.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
        ev.to_mesh_clear()
    return {"objects": len(children(root)), "triangles": tris, "size_m": [round(x, 3) for x in (hi - lo)], "min_m": [round(x, 3) for x in lo]}


def terrain(path, name="lola_terrain_site23"):
    import numpy as np

    z = np.load(path, allow_pickle=False)
    h, mpp = z["height"], float(z["mpp"])
    n = h.shape[0]
    bm = bmesh.new()
    vs = [[bm.verts.new(((j - n // 2) * mpp, (n // 2 - i) * mpp, float(h[i, j]))) for j in range(n)] for i in range(n)]
    for i in range(n - 1):
        for j in range(n - 1):
            bm.faces.new((vs[i][j], vs[i + 1][j], vs[i + 1][j + 1], vs[i][j + 1]))
    ob = _obj(name, bm, "regolith")
    for p in ob.data.polygons:
        p.use_smooth = True
    sub = ob.modifiers.new("subdiv", "SUBSURF")
    sub.levels = sub.render_levels = 2
    return ob, h, mpp


def ground_z(h, mpp, x, y):
    n = h.shape[0]
    j, i = x / mpp + n // 2, n // 2 - y / mpp
    j0, i0 = int(j), int(i)
    fj, fi = j - j0, i - i0
    return float(h[i0, j0] * (1 - fj) * (1 - fi) + h[i0, j0 + 1] * fj * (1 - fi) + h[i0 + 1, j0] * (1 - fj) * fi + h[i0 + 1, j0 + 1] * fj * fi)


def world_and_lights(sun_elev=7.0, sun_az=-35.0, strength=6.0, stars=True):
    world = bpy.data.worlds.new("space")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = strength
    sun.angle = math.radians(0.53)                                                # real solar disk size
    so = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(so)
    so.rotation_euler = (math.radians(90 - sun_elev), 0, math.radians(sun_az))
    fill = bpy.data.lights.new("EarthshineFill", "SUN")
    fill.energy = 0.15
    fo = bpy.data.objects.new("EarthshineFill", fill)
    bpy.context.collection.objects.link(fo)
    fo.rotation_euler = (math.radians(50), 0, math.radians(sun_az + 160))
    if stars:
        import random

        rnd = random.Random(3)
        bm = bmesh.new()
        for _ in range(500):
            v = Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), abs(rnd.gauss(0, 1)) + 0.05)).normalized() * 3000
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rnd.uniform(1.5, 3.5), matrix=Matrix.Translation(v))
        st = _principled("star", (1, 1, 1), emission=(1, 1, 1))
        MAT["star"] = st
        o = _obj("stars", bm, "star")
        o.visible_shadow = False


def camera(target, dist, elev, az, lens=50.0):
    cam = bpy.data.cameras.new("cam")
    cam.lens = lens
    cam.clip_end = 20000
    co = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(co)
    t = Vector(target)
    e, a = math.radians(elev), math.radians(az)
    co.location = t + Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))) * dist
    co.rotation_euler = (t - co.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = co
    return co


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:  # noqa: BLE001
        print("GPU not available, CPU render:", e)
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX" if "AgX" in [i.name for i in type(sc.view_settings).bl_rna.properties["view_transform"].enum_items] else "Filmic"
    sc.view_settings.look = "None"


def render(path):
    if not RENDER:
        return
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)



def repivot_all(root):
    """Put every group with a 'pivot' (wheels, engines) at its own pivot point without moving anything in the world."""
    bpy.context.view_layer.update()
    for ob in children(root):
        if "pivot" not in ob.keys():
            continue
        kids = [(c, c.matrix_world.copy()) for c in ob.children]
        ob.location = Vector(ob["pivot"])
        bpy.context.view_layer.update()
        for c, mw in kids:
            c.matrix_parent_inverse = ob.matrix_world.inverted()
            c.matrix_basis = Matrix.Identity(4)
            bpy.context.view_layer.update()
            c.matrix_world = mw
    bpy.context.view_layer.update()


def export_glb_ui(root, path):
    """UI convention: glTF Y up, vehicle front facing -Z. Blender +Y becomes glTF -Z, so the front (+x in the body frame) is turned to +Y."""
    old = root.rotation_euler.copy()
    root.rotation_euler = (0, 0, math.radians(90))
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    for o in children(root):
        if "cutter" not in o.name:
            o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_apply=True, export_yup=True)
    root.rotation_euler = old
    bpy.context.view_layer.update()


def export_obj_sim(root, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in children(root):
        if "cutter" not in o.name:
            o.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True, apply_modifiers=True, forward_axis="Y", up_axis="Z", export_materials=True)



def showcase(path):
    """One scene for looking at the models: lander on a regolith disc, rover in front of its ramp, the separation stack beside them, low Sun."""
    clear()
    materials()
    ld = build_lander()
    repivot_all(ld)
    ld.location = (0, 0, 1.45)
    rv = build_rover()
    repivot_all(rv)
    rv.location = (-2.2, 0.0, 0.205)
    rv.rotation_euler = (0, 0, math.radians(180))
    pm = build_pm()
    l2 = build_lander("lander_separating")
    for r in (pm, l2):
        repivot_all(r)
    pm.location = (9.0, 0.0, 3.0)
    l2.location = (9.0, 0.0, 3.0 + 2.45)
    cyl("regolith_disc", (0, 0, -0.05), (0, 0, 1), 0.05, 7.0, 7.0, "regolith", None, 96)
    world_and_lights(sun_elev=12, sun_az=-40, strength=5.0, stars=False)
    w = bpy.context.scene.world
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.022, 0.03, 1.0)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    camera((3.0, 0.0, 1.5), 16.0, 16, -55, 35)
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


# ---------------------------------------------------------------------------------------------------------------- cinematic animation (keyframes only)
FPS = 24


def _key(ob, path, frame, value, index=-1):
    setattr(ob, path, value) if index < 0 else getattr(ob, path).__setitem__(index, value)
    ob.keyframe_insert(data_path=path, frame=frame, index=index)


def _ease_all(ob, mode="BEZIER"):
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    try:
        curves = ad.action.fcurves
    except AttributeError:                                                       # Blender 5: layered actions
        curves = [fc for layer in ad.action.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = mode
    return curves


def _fcurves(ob):
    return _ease_all(ob) or []


def _set_collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    lc = bpy.context.view_layer.layer_collection.children[name]
    bpy.context.view_layer.active_layer_collection = lc
    return coll


def _glow_material(name, color, strength, alpha):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    for key in ("Emission Color", "Emission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = (*color, 1.0)
            break
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Alpha"].default_value = alpha
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        try:
            setattr(m, attr, val)
        except (AttributeError, TypeError):
            pass
    MAT[name] = m
    return m, bsdf


def _cone_object(name, length, r0, r1, mat, seg=32):
    """Cone with its origin at the wide-open base (nozzle exit), pointing down -z."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=seg, radius1=r1, radius2=r0, depth=length, matrix=Matrix.Translation((0, 0, -length / 2)))
    return _obj(name, bm, mat)


def _camera(name, lens=35.0):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start, cam.clip_end = 0.05, 4.0e6
    co = bpy.data.objects.new(name, cam)
    bpy.context.collection.objects.link(co)
    return co


def _track(cam, target):
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    return c


def _plumes(lander_root, frames_on, frames_off, flicker=True):
    """Faint engine plumes (artistic: real hypergolic plumes in vacuum are nearly invisible) under the 5 nozzles, keyed on/off."""
    _glow_material("plume", (0.55, 0.7, 1.0), 1.5, 0.05)
    c, s = math.cos(math.radians(5)), math.sin(math.radians(5))
    exits = [((0, 0, -1.0), (0, 0, 1))] + [((0.8 * ex, 0.8 * ey, -1.0), (-s * ex, -s * ey, c)) for ex, ey in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
    out = []
    for k, (p, t) in enumerate(exits):
        pl = _cone_object(f"plume{k}", 4.0, 0.16, 0.42, "plume")
        pl.parent = lander_root
        pl.location = p
        pl.rotation_euler = _align(Vector(t)).to_euler()                         # cone points along -t (down the nozzle)
        for f, on in ((frames_on - 1, 0.0), (frames_on, 1.0), (frames_off, 1.0), (frames_off + 2, 0.0)):
            _key(pl, "scale", f, (max(on, 0.001),) * 3)
        if flicker:
            for fc in _fcurves(pl):
                mod = fc.modifiers.new("NOISE")
                mod.scale, mod.strength = 2.0, 0.12
                mod.use_restricted_range = True
                mod.frame_start, mod.frame_end = frames_on, frames_off
        out.append(pl)
    return out


def shot_separation(scene_name="Shot1_Separation_Orbit"):
    scene = bpy.context.scene
    scene.name = scene_name
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, 180
    pm = build_pm()
    ld = build_lander("lander")
    for r in (pm, ld):
        repivot_all(r)
    ld.location = (0, 0, 1.25)
    # spring release at frame 40: slow push for 20 frames, then drift; small tumble on the PM
    for f, z in ((1, 1.25), (40, 1.25), (60, 1.6), (180, 5.6)):
        _key(ld, "location", f, (0, 0, z))
    for f, rz in ((1, 0.0), (180, math.radians(4))):
        _key(ld, "rotation_euler", f, (0, 0, rz))
    for f, rx in ((40, 0.0), (180, math.radians(-2.5))):
        _key(pm, "rotation_euler", f, (rx, 0, 0))
    _ease_all(ld)
    # RCS puffs on the lander (frames 110-140)
    _glow_material("rcs_puff", (0.8, 0.9, 1.0), 1.5, 0.08)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        puff = _cone_object(f"rcs_puff{k}", 0.6, 0.02, 0.12, "rcs_puff", 16)
        puff.parent = ld
        puff.location = (0.98 * math.cos(a), 0.98 * math.sin(a), 0.22 - 0.06)
        for f, on in ((109, 0.0), (112, 1.0), (124, 1.0), (127, 0.0)) if k % 2 == 0 else ((119, 0.0), (122, 1.0), (134, 1.0), (137, 0.0)):
            _key(puff, "scale", f, (max(on, 0.001),) * 3)
    world_and_lights(sun_elev=20, sun_az=-30, strength=5.0)
    moon = sphere("moon", (0, 0, 0), 1737400.0, "moon", seg=256)
    moon.location = (0, 0, -1737400.0 - 100000.0)                                # 100 km circular orbit
    focus = empty("cam_focus")
    for f, p in ((1, (0, 0, 0.1)), (60, (0, 0, 0.6)), (180, (0, 0, 2.0))):
        _key(focus, "location", f, p)
    cam = _camera("cam_separation", 40)
    for f, p in ((1, (2.6, -2.0, 0.4)), (90, (6.0, -6.5, 1.8)), (180, (10.0, -13.0, 4.5))):
        _key(cam, "location", f, p)
    _ease_all(cam)
    _ease_all(focus)
    _track(cam, focus)
    scene.camera = cam
    return scene


def shot_surface(terrain_path, scene_name="Shot2_Landing_Rover"):
    scene = bpy.context.scene
    scene.name = scene_name
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, 620
    ter, h, mpp = terrain(terrain_path)
    gz = ground_z(h, mpp, 0.0, 0.0)
    ld = build_lander("lander")
    repivot_all(ld)
    plumes = _plumes(ld, 1, 300)
    # ramp on a hinge pivot (hinge line at body x=-1.0, z=-0.78)
    hinge = empty("ramp_hinge_pivot", ld, (-1.0, 0.0, -0.78))
    bpy.context.view_layer.update()
    for name in ("ramp_panel", "ramp_latch0", "ramp_latch1", "rbf_tag0", "rbf_tag1"):
        ob = bpy.data.objects[name]
        mw = ob.matrix_world.copy()
        ob.parent = hinge
        ob.matrix_parent_inverse = hinge.matrix_world.inverted()
        ob.matrix_world = mw
    for f, a in ((330, 0.0), (390, math.radians(-132.0))):
        _key(hinge, "rotation_euler", f, (0, a, 0))
    _ease_all(hinge)
    # descent: 120 m up, 15 m divert (hazard avoidance), tilt into the divert and level, touchdown at frame 300
    path = ((1, (-15.0, -6.0, 120.0)), (120, (-6.0, -2.0, 48.0)), (200, (-0.6, -0.2, 14.0)), (260, (0.0, 0.0, 4.0)), (300, (0.0, 0.0, 0.0)))
    for f, (x, y, dz) in path:
        _key(ld, "location", f, (x, y, gz + 1.45 + dz))
    for f, (rx, ry) in ((1, (0.0, 0.0)), (70, (math.radians(3), math.radians(8))), (170, (0.0, 0.0)), (300, (0.0, 0.0))):
        _key(ld, "rotation_euler", f, (rx, ry, 0.0))
    _ease_all(ld)
    # dust sheet blown out by the plumes in the last metres
    m, bsdf = _glow_material("dust", (0.45, 0.43, 0.4), 0.0, 0.0)
    dust = cyl("dust_sheet", (0, 0, gz + 0.05), (0, 0, 1), 0.35, 0.8, 1.6, "dust", None, 64)
    for f, sc, al in ((245, 0.3, 0.0), (270, 1.0, 0.45), (300, 3.0, 0.35), (340, 6.0, 0.0)):
        _key(dust, "scale", f, (sc, sc, 1.0))
        bsdf.inputs["Alpha"].default_value = al
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=f)
    # rover: hidden until the ramp is down, then down the ramp and away across real terrain
    rv = build_rover("rover")
    repivot_all(rv)
    for f, hide in ((1, True), (391, True), (392, False)):
        rv.hide_viewport = rv.hide_render = hide
        rv.keyframe_insert("hide_viewport", frame=f)
        rv.keyframe_insert("hide_render", frame=f)
    ramp_deg = 42.0
    top = Vector((-1.05, 0.0, gz + 0.67 + 0.2))
    bottom = Vector((-1.74, 0.0, gz + 0.205))
    wps = [(392, top, ramp_deg), (440, bottom, ramp_deg * 0.3)]
    pts = [(-2.4, -0.1), (-3.4, -0.7), (-4.6, -1.8), (-5.8, -2.6), (-7.0, -3.0)]
    for i, (x, y) in enumerate(pts):
        wps.append((450 + 20 * i, Vector((x, y, ground_z(h, mpp, x, y) + 0.205)), 0.0))
    dist, prev = 0.0, None
    for f, p, pitch in wps:
        if prev is not None:
            d = p - prev[1]
            dist += d.length
            heading = math.atan2(d.y, d.x)
            while heading - prev[2] > math.pi:
                heading -= 2 * math.pi
            while heading - prev[2] < -math.pi:
                heading += 2 * math.pi
        else:
            heading = math.pi
        _key(rv, "location", f, p)
        _key(rv, "rotation_euler", f, (0.0, math.radians(pitch), heading))
        for wn in ("wheel_FL", "wheel_ML", "wheel_RL", "wheel_FR", "wheel_MR", "wheel_RR"):
            _key(bpy.data.objects[wn], "rotation_euler", f, (0.0, dist / 0.086, 0.0))
        prev = (f, p, heading)
    _ease_all(rv)
    # solar panel tilts up toward the low Sun once on the ground (stowed flat before)
    tp = bpy.data.objects["tilt_panel"]
    base_rot = tp.rotation_euler.copy()
    for f, k in ((392, 0.0), (470, 0.0), (505, 1.0)):
        _key(tp, "rotation_euler", f, (base_rot.x * k, base_rot.y * k, base_rot.z * k))
    # sky: low polar Sun, Earth just above the horizon (apparent size about 2 deg)
    world_and_lights(sun_elev=6, sun_az=-20, strength=7.0)
    earth_m = _principled("earth", (0.12, 0.3, 0.75), 0.0, 0.6, emission=(0.1, 0.25, 0.6))
    MAT["earth"] = earth_m
    sphere("earth", (-2600.0, 1500.0, gz + 280.0), 52.0, "earth", seg=48)
    # cameras + markers (cuts)
    tgt_l = empty("target_lander")
    tgt_l.parent = ld
    tgt_l.location = (0, 0, 0.3)
    tgt_r = empty("target_rover")
    tgt_r.parent = rv
    cams = {}
    c1 = _camera("cam_descent_lowangle", 55)
    for f, p in ((1, (40.0, -60.0, gz + 3.0)), (240, (16.0, -22.0, gz + 1.4))):
        _key(c1, "location", f, p)
    _track(c1, tgt_l)
    cams[1] = c1
    c2 = _camera("cam_touchdown_wide", 32)
    for f, p in ((241, (-9.0, -14.0, gz + 1.5)), (330, (-7.5, -11.5, gz + 1.3))):
        _key(c2, "location", f, p)
    _track(c2, tgt_l)
    cams[241] = c2
    c3 = _camera("cam_ramp_close", 40)
    for f, p in ((331, (-4.5, -3.2, gz + 1.1)), (440, (-4.2, -2.4, gz + 0.8))):
        _key(c3, "location", f, p)
    _track(c3, empty("ramp_focus", None, (-1.4, 0.0, gz + 0.5)))
    cams[331] = c3
    c4 = _camera("cam_rover_follow", 35)
    for f, p in ((441, (-2.0, -3.8, gz + 0.8)), (540, (-6.0, -6.5, gz + 0.9))):
        _key(c4, "location", f, p)
    _track(c4, tgt_r)
    cams[441] = c4
    c5 = _camera("cam_hero_orbit", 30)
    centre = Vector((-3.0, -1.2, gz + 0.9))
    for i, f in enumerate(range(541, 621, 8)):
        a = math.radians(-120 + 6 * i)
        _key(c5, "location", f, centre + Vector((12 * math.cos(a), 12 * math.sin(a), 2.2)))
    hero = empty("hero_focus", None, tuple(centre))
    _track(c5, hero)
    cams[541] = c5
    for c in cams.values():
        _ease_all(c)
    for f, c in cams.items():
        mk = scene.timeline_markers.new(c.name, frame=f)
        mk.camera = c
    scene.camera = c1
    return scene


def cinematic(path, terrain_path):
    """Two scenes in one file: Shot1_Separation_Orbit and Shot2_Landing_Rover (keyframes and camera cuts only; nothing is rendered)."""
    tmp = Path(path).with_name("_shot1_tmp.blend")
    clear()
    materials()
    shot_separation()
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp))
    clear()
    materials()
    shot_surface(terrain_path)
    bpy.ops.wm.append(filepath=str(tmp / "Scene" / "Shot1_Separation_Orbit"), directory=str(tmp) + "/Scene/", filename="Shot1_Separation_Orbit")
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    tmp.unlink(missing_ok=True)



# ================================================================================================================ v3: rover bay, better materials, story
R_MOON = 1737400.0
ORBIT_ALT = 100000.0
ROVER_STOW = (-0.48, 0.0, -0.595)                 # rover centre inside the bay, lander body frame (wheels on the bay floor at z = -0.80)


def _crinkle(name, color, metallic, roughness, scale, strength):
    """Foil with a crinkled look: Voronoi cells + fine noise drive the bump."""
    m = _principled(name, color, metallic, roughness)
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = scale
    noi = nt.nodes.new("ShaderNodeTexNoise")
    noi.inputs["Scale"].default_value = scale * 3
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(vor.outputs["Distance"], add.inputs[0])
    nt.links.new(noi.outputs["Fac"], add.inputs[1])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    nt.links.new(add.outputs["Value"], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _cratered(name, color, scale, strength):
    m = _principled(name, color, 0.0, 0.95)
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "SMOOTH_F1" if "SMOOTH_F1" in [i.identifier for i in vor.bl_rna.properties["feature"].enum_items] else "F1"
    vor.inputs["Scale"].default_value = scale
    noi = nt.nodes.new("ShaderNodeTexNoise")
    noi.inputs["Scale"].default_value = scale * 4
    noi.inputs["Detail"].default_value = 10.0
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(vor.outputs["Distance"], add.inputs[0])
    nt.links.new(noi.outputs["Fac"], add.inputs[1])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    nt.links.new(add.outputs["Value"], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, 1)
    ramp.color_ramp.elements[1].color = (color[0] * 1.2, color[1] * 1.2, color[2] * 1.2, 1)
    nt.links.new(noi.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


_materials_v2 = materials


def materials():  # noqa: F811 - realistic spacecraft palette: crinkled foils, blue cells, cratered regolith and Moon
    _materials_v2()
    MAT["gold_foil"] = _crinkle("gold_foil_MLI_v3", (1.0, 0.76, 0.32), 0.85, 0.3, 18.0, 0.45)
    MAT["silver_foil"] = _crinkle("silver_foil_MLI_v3", (0.9, 0.9, 0.93), 0.8, 0.28, 22.0, 0.35)
    MAT["black_kapton"] = _principled("black_kapton_v3", (0.02, 0.02, 0.025), 0.3, 0.35)
    MAT["white_paint"] = _principled("white_paint_v3", (0.9, 0.9, 0.88), 0.0, 0.45)
    MAT["solar_cell"] = _principled("solar_cell_v3", (0.05, 0.09, 0.28), 0.6, 0.12)
    MAT["panel_back"] = _principled("panel_back_v3", (0.55, 0.55, 0.58), 0.4, 0.5)
    MAT["aluminium"] = _principled("aluminium_v3", (0.85, 0.86, 0.88), 0.9, 0.32)
    MAT["titanium"] = _principled("nozzle_niobium_v3", (0.42, 0.4, 0.42), 1.0, 0.35)
    MAT["nozzle_inner"] = _principled("nozzle_inner_v3", (0.08, 0.07, 0.07), 0.9, 0.5)
    MAT["regolith"] = _cratered("regolith_v3", (0.4, 0.39, 0.37), 1.5, 0.7)
    MAT["moon"] = _cratered("moon_v3", (0.45, 0.44, 0.42), 30.0, 0.8)


def lander_bay(root):
    """Rover bay behind the -x door: boolean cut in the body, black lining, floor rails, tie-downs, door frame."""
    body = next(o for o in children(root) if o.name.startswith("deck_body"))
    cutter = box("rover_bay_cutter", (-0.6, 0.0, -0.45), (1.2, 0.8, 0.70), "black_kapton", root)
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    mod = body.modifiers.new("rover_bay", "BOOLEAN")
    mod.operation, mod.object = "DIFFERENCE", cutter
    lin = Batch("rover_bay_lining", "black_kapton", root)
    lin.box((-0.5, 0.0, -0.81), (0.94, 0.8, 0.02)).box((-0.5, 0.0, -0.09), (0.94, 0.8, 0.02))
    for sy in (1, -1):
        lin.box((-0.5, 0.405 * sy, -0.45), (0.94, 0.01, 0.72))
    lin.box((-0.025, 0.0, -0.45), (0.01, 0.8, 0.72))
    lin.done()
    rails = Batch("rover_bay_rails_tiedowns", "aluminium", root)
    for sy in (1, -1):
        rails.box((-0.5, 0.32 * sy, -0.795), (0.94, 0.03, 0.012))
        for x in (-0.15, -0.8):
            rails.box((x, 0.25 * sy, -0.77), (0.05, 0.05, 0.05))
    for sy in (1, -1):
        rails.box((-0.978, 0.41 * sy, -0.45), (0.03, 0.03, 0.72))
    rails.box((-0.978, 0.0, -0.085), (0.03, 0.85, 0.03))
    rails.done()
    return cutter


_build_lander_v2 = build_lander


def build_lander(root_name="lander"):  # noqa: F811
    root = _build_lander_v2(root_name)
    lander_bay(root)
    return root


def rover_head(root):
    """Group the camera head on a pan pivot at the mast top (named navcam_head for animation / the UI)."""
    bpy.context.view_layer.update()
    head = empty("navcam_head", root, (0.27, 0.0, 0.38))
    bpy.context.view_layer.update()
    for o in list(children(root)):
        if o.name.startswith(("pan_tilt", "navcam_bar", "navcam0_lens", "navcam1_lens", "navcam_baffles")):
            mw = o.matrix_world.copy()
            o.parent = head
            o.matrix_parent_inverse = head.matrix_world.inverted()
            o.matrix_world = mw
    return head


_build_rover_v2 = build_rover


def build_rover(root_name="rover"):  # noqa: F811
    root = _build_rover_v2(root_name)
    rover_head(root)
    return root


def find(root, prefix):
    return next(o for o in children(root) if o.name.startswith(prefix))


def ramp_pivot(ld):
    """Door/ramp on a hinge pivot (hinge line at body x=-1.0, z=-0.78). Rotation y = -132 deg puts the ramp end on level ground."""
    hinge = empty("ramp_hinge_pivot", ld, (-1.0, 0.0, -0.78))
    bpy.context.view_layer.update()
    for pre in ("ramp_panel", "ramp_latch0", "ramp_latch1", "rbf_tag0", "rbf_tag1"):
        ob = find(ld, pre)
        mw = ob.matrix_world.copy()
        ob.parent = hinge
        ob.matrix_parent_inverse = hinge.matrix_world.inverted()
        ob.matrix_world = mw
    return hinge


def stow_rover(ld, rv):
    """Rover inside the lander bay, panel flat, facing the door (-x)."""
    rv.parent = ld
    rv.location = ROVER_STOW
    rv.rotation_euler = (0.0, 0.0, math.pi)
    tp = find(rv, "tilt_panel")
    tp["deployed_rot_y"] = tp.rotation_euler.y
    tp.rotation_euler.y = 0.0
    return rv


def world_stars(scene, strength=1.0, ambient=0.07):
    """Camera rays see a black sky with sparse stars; reflections and fill light see a dim grey (sunlit regolith all around)."""
    w = bpy.data.worlds.new("space_stars")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 900.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position = 0.045
    ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    stars = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(ramp.outputs["Color"], stars.inputs["Color"])
    stars.inputs["Strength"].default_value = strength
    amb = nt.nodes.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = (0.5, 0.5, 0.52, 1)
    amb.inputs["Strength"].default_value = ambient
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(amb.outputs["Background"], mix.inputs[1])
    nt.links.new(stars.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return w


def sun(scene_coll_name, elev, az, energy):
    li = bpy.data.lights.new("Sun", "SUN")
    li.energy = energy
    li.angle = math.radians(0.53)
    so = bpy.data.objects.new("Sun", li)
    bpy.context.collection.objects.link(so)
    so.rotation_euler = (math.radians(90 - elev), 0, math.radians(az))
    return so


def _dof(cam, target, fstop=2.8):
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = target
    cam.data.dof.aperture_fstop = fstop


def _shake(cam, f0, f1, strength=0.06):
    for fc in _fcurves(cam):
        if fc.data_path == "location":
            m = fc.modifiers.new("NOISE")
            m.scale, m.strength = 1.5, strength
            m.use_restricted_range = True
            m.frame_start, m.frame_end = f0, f1
            m.blend_in, m.blend_out = 2, 6


def look(scene):
    """AgX with a bit more contrast, EEVEE for the viewport."""
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    vs = scene.view_settings
    names = [i.identifier for i in vs.bl_rna.properties["view_transform"].enum_items]
    vs.view_transform = "AgX" if "AgX" in names else "Filmic"
    for lk in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            vs.look = lk
            break
        except TypeError:
            continue



def moon_textures_dir() -> Path:
    import os

    return Path(os.environ.get("ADI_DATA", Path(__file__).resolve().parents[2] / "data")) / "textures" / "moon"


def moon_sphere(name="moon", seg=512):
    """The Moon with REAL NASA imagery: LRO LROC WAC colour mosaic (CGI Moon Kit, poles filled) + LOLA elevation (ldem_16) as relief.
    South pole turned to face +z, because the landing sites are south-polar. Source: NASA SVS 4720 (CGI Moon Kit)."""
    tdir = moon_textures_dir()
    col_p, dem_p = tdir / "lroc_color_poles_4k.tif", tdir / "ldem_16_uint.tif"
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=R_MOON, calc_uvs=True)
    m = bpy.data.materials.new("moon_LROC_LOLA")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.95
    uv = nt.nodes.new("ShaderNodeTexCoord")
    if col_p.exists():
        img = nt.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(str(col_p))
        img.interpolation = "Cubic"
        nt.links.new(uv.outputs["UV"], img.inputs["Vector"])
        nt.links.new(img.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        print("WARNING: no LROC colour map at", col_p, "- run the download in vizard/blender/README.md", flush=True)
    if dem_p.exists():
        dem = nt.nodes.new("ShaderNodeTexImage")
        dem.image = bpy.data.images.load(str(dem_p))
        dem.image.colorspace_settings.name = "Non-Color"
        dem.interpolation = "Cubic"
        nt.links.new(uv.outputs["UV"], dem.inputs["Vector"])
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = 0.6
        nt.links.new(dem.outputs["Color"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    MAT["moon_lroc"] = m
    ob = _obj(name, bm, "moon_lroc")
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.rotation_euler = (math.pi, 0.0, 0.0)                                       # south pole up
    return ob


def shot_separation(scene_name="Shot1_Orbit_Separation_Deorbit"):  # noqa: F811
    """Wide orbit (time compressed) -> separation at real orbital speed -> lander turns retrograde and fires its deorbit burn."""
    scene = bpy.context.scene
    scene.name = scene_name
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, 420
    look(scene)
    world_stars(scene)
    sun("Shot1", 15, -35, 5.0)
    rr = R_MOON + ORBIT_ALT
    moon = moon_sphere("moon", seg=256)
    moon.location = (0, 0, -rr)
    # (a) wide: the orbit ring and a marker going round (time compressed: one full orbit is 118 min)
    ring_m, _ = _glow_material("orbit_ring", (0.4, 0.75, 1.0), 3.0, 0.6)
    curve = bpy.data.curves.new("orbit_curve", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 2500.0
    sp = curve.splines.new("POLY")
    n = 256
    sp.points.add(n)
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        sp.points[i].co = (0.0, rr * math.sin(a), rr * math.cos(a) - rr, 1.0)
    orbit = bpy.data.objects.new("orbit_path_100km", curve)
    orbit.data.materials.append(ring_m)
    bpy.context.collection.objects.link(orbit)
    mk_m, _ = _glow_material("craft_marker", (1.0, 0.85, 0.4), 30.0, 1.0)
    pivot_wide = empty("marker_pivot", None, (0, 0, -rr))
    marker = sphere("craft_marker", (0, 0, rr), 25000.0, "craft_marker", seg=16)
    marker.parent = pivot_wide
    for f, a in ((1, math.radians(-150)), (120, math.radians(-10))):
        _key(pivot_wide, "rotation_euler", f, (a, 0, 0))
    for ob in (orbit, marker):
        for f, hide in ((1, False), (120, False), (121, True)):
            ob.hide_viewport = ob.hide_render = hide
            ob.keyframe_insert("hide_viewport", frame=f)
            ob.keyframe_insert("hide_render", frame=f)
    camw = _camera("cam_orbit_wide", 35)
    camw.data.clip_start, camw.data.clip_end = 1000.0, 3.0e7                      # the Moon is ~6000 km away in the wide shot
    for f, p in ((1, (-5.2e6, -3.2e6, 2.4e6)), (120, (-2.6e6, -1.4e6, 0.9e6))):
        _key(camw, "location", f, p)
    _track(camw, empty("moon_centre_focus", None, (0, 0, -rr * 0.6)))
    # (b) close: the stack rides a carrier that turns about the Moon centre at the real orbital rate (v = 1.63 km/s)
    carrier = empty("orbit_carrier", None, (0, 0, -rr))
    w = 1633.0 / rr / FPS
    for f in (1, 420):
        _key(carrier, "rotation_euler", f, (-w * (f - 121), 0, 0))
    for fc in _fcurves(carrier):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    frame = empty("orbit_frame", carrier, (0, 0, rr))
    pm = build_pm()
    ld = build_lander("lander")
    rv = build_rover("rover")
    for r in (pm, ld, rv):
        repivot_all(r)
    stow_rover(ld, rv)
    pm.parent = frame
    ld.parent = frame
    ld.location = (0, 0, 1.25)
    for f, z in ((121, 1.25), (160, 1.25), (180, 1.6), (300, 5.6), (420, 8.0)):
        _key(ld, "location", f, (0, 0, z))
    for f, rx, rz in ((121, 0.0, 0.0), (300, 0.0, math.radians(4)), (345, math.radians(-90), math.radians(4)), (420, math.radians(-90), math.radians(4))):
        _key(ld, "rotation_euler", f, (rx, 0, rz))
    for f, rx in ((160, 0.0), (420, math.radians(-4))):
        _key(pm, "rotation_euler", f, (rx, 0, 0))
    _ease_all(ld)
    # RCS puffs: settle after separation (230-255) and the retrograde turn (300-345)
    _glow_material("rcs_puff", (0.8, 0.9, 1.0), 1.5, 0.08)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        puff = _cone_object(f"rcs_puff{k}", 0.6, 0.02, 0.12, "rcs_puff", 16)
        puff.parent = ld
        puff.location = (0.98 * math.cos(a), 0.98 * math.sin(a), 0.16)
        keys = [(229, 0.0), (232, 1.0), (244, 1.0), (247, 0.0), (299, 0.0), (302, 1.0), (312, 1.0), (315, 0.0), (334, 0.0), (337, 1.0), (345, 1.0), (348, 0.0)]
        for f, on in keys:
            _key(puff, "scale", f, (max(on, 0.001),) * 3)
    _plumes(ld, 355, 420)                                                         # deorbit burn (continues off screen)
    focus = empty("cam_focus", frame)
    for f, p in ((121, (0, 0, 0.1)), (180, (0, 0, 0.6)), (300, (0, 0, 2.0)), (420, (0, 0, 6.0))):
        _key(focus, "location", f, p)
    cams = _camera("cam_separation", 40)
    cams.parent = frame
    for f, p in ((121, (2.6, -2.0, 0.4)), (210, (6.0, -6.5, 1.8)), (300, (10.0, -13.0, 4.5))):
        _key(cams, "location", f, p)
    _track(cams, focus)
    _dof(cams, focus, 4.0)
    camd = _camera("cam_deorbit", 32)
    camd.parent = frame
    for f, p in ((301, (-7.0, 9.0, 10.0)), (420, (-4.0, 13.0, 7.5))):
        _key(camd, "location", f, p)
    _track(camd, ld)
    for c in (camw, cams, camd, focus):
        _ease_all(c)
    for f, c in ((1, camw), (121, cams), (301, camd)):
        mk = scene.timeline_markers.new(c.name, frame=f)
        mk.camera = c
    scene.camera = camw
    return scene


def shot_surface(terrain_path, scene_name="Shot2_Landing_Rover"):  # noqa: F811
    scene = bpy.context.scene
    scene.name = scene_name
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, 660
    look(scene)
    world_stars(scene, 0.6)
    sun("Shot2", 6, -20, 7.0)
    ter, h, mpp = terrain(terrain_path)
    gz = ground_z(h, mpp, 0.0, 0.0)
    ld = build_lander("lander")
    rv = build_rover("rover")
    for r in (ld, rv):
        repivot_all(r)
    stow_rover(ld, rv)
    hinge = ramp_pivot(ld)
    _plumes(ld, 1, 298)
    glow = bpy.data.lights.new("engine_glow", "POINT")
    glow.color = (0.75, 0.82, 1.0)
    glow.shadow_soft_size = 0.5
    go = bpy.data.objects.new("engine_glow", glow)
    bpy.context.collection.objects.link(go)
    go.parent = ld
    go.location = (0, 0, -1.6)
    for f, e in ((1, 1500.0), (298, 1500.0), (300, 0.0)):
        glow.energy = e
        glow.keyframe_insert("energy", frame=f)
    path = ((1, (-15.0, -6.0, 120.0)), (120, (-6.0, -2.0, 48.0)), (200, (-0.6, -0.2, 14.0)), (260, (0.0, 0.0, 4.0)), (300, (0.0, 0.0, 0.0)))
    for f, (x, y, dz) in path:
        _key(ld, "location", f, (x, y, gz + 1.45 + dz))
    for f, (rx, ry) in ((1, (0.0, 0.0)), (70, (math.radians(3), math.radians(8))), (170, (0.0, 0.0)), (300, (0.0, 0.0))):
        _key(ld, "rotation_euler", f, (rx, ry, 0.0))
    _ease_all(ld)
    m, bsdf = _glow_material("dust", (0.45, 0.43, 0.4), 0.0, 0.0)
    dust = cyl("dust_sheet", (0, 0, gz + 0.05), (0, 0, 1), 0.35, 0.8, 1.6, "dust", None, 64)
    for f, sc, al in ((245, 0.3, 0.0), (270, 1.0, 0.45), (300, 3.0, 0.35), (340, 6.0, 0.0)):
        _key(dust, "scale", f, (sc, sc, 1.0))
        bsdf.inputs["Alpha"].default_value = al
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=f)
    # door opens into a ramp (330-390), rover rolls out of the bay (395-470), drives (480-580)
    for f, a in ((330, 0.0), (390, math.radians(-132.0))):
        _key(hinge, "rotation_euler", f, (0, a, 0))
    _ease_all(hinge)
    d = Vector((-math.cos(math.radians(42)), 0, -math.sin(math.radians(42))))
    nrm = Vector((-math.sin(math.radians(42)), 0, math.cos(math.radians(42))))
    hinge_pt = Vector((-1.0, 0.0, -0.78))
    local = [(395, Vector(ROVER_STOW), 0.0), (420, Vector((-0.98, 0.0, -0.595)), 0.0), (445, hinge_pt + d * 0.5 + nrm * 0.205, 42.0),
             (468, hinge_pt + d * 1.0 + nrm * 0.205, 20.0)]
    lz = gz + 1.45
    for i, (x, y) in enumerate([(-2.5, -0.1), (-3.5, -0.7), (-4.7, -1.8), (-5.9, -2.6), (-7.1, -3.0)]):
        local.append((480 + 25 * i, Vector((x, y, ground_z(h, mpp, x, y) + 0.205 - lz)), 0.0))
    dist, prev = 0.0, None
    for f, p, pitch in local:
        if prev is not None:
            dd = p - prev[1]
            dist += dd.length
            heading = math.atan2(dd.y, dd.x) if dd.xy.length > 1e-6 else prev[2]
            while heading - prev[2] > math.pi:
                heading -= 2 * math.pi
            while heading - prev[2] < -math.pi:
                heading += 2 * math.pi
        else:
            heading = math.pi
        _key(rv, "location", f, tuple(p))
        _key(rv, "rotation_euler", f, (0.0, math.radians(pitch), heading))
        for wn in ("wheel_FL", "wheel_ML", "wheel_RL", "wheel_FR", "wheel_MR", "wheel_RR"):
            _key(find(rv, wn), "rotation_euler", f, (0.0, dist / 0.086, 0.0))
        prev = (f, p, heading)
    _key(rv, "location", 1, tuple(ROVER_STOW))
    _ease_all(rv)
    tp = find(rv, "tilt_panel")
    for f, k in ((1, 0.0), (485, 0.0), (515, 1.0)):
        _key(tp, "rotation_euler", f, (0.0, tp["deployed_rot_y"] * k, 0.0))
    head = find(rv, "navcam_head")
    for f, a in ((1, 0.0), (500, 0.0), (520, math.radians(45)), (545, math.radians(-40)), (565, 0.0)):
        _key(head, "rotation_euler", f, (0.0, 0.0, a))
    earth_m = _principled("earth", (0.12, 0.3, 0.75), 0.0, 0.6, emission=(0.1, 0.25, 0.6))
    MAT["earth"] = earth_m
    sphere("earth", (-2600.0, 1500.0, gz + 280.0), 52.0, "earth", seg=48)
    tgt_l = empty("target_lander", ld, (0, 0, 0.3))
    tgt_r = empty("target_rover", rv, (0, 0, 0.1))
    cams = {}
    c1 = _camera("cam_descent_lowangle", 55)
    for f, p in ((1, (40.0, -60.0, gz + 3.0)), (240, (16.0, -22.0, gz + 1.4))):
        _key(c1, "location", f, p)
    _track(c1, tgt_l)
    cams[1] = c1
    c2 = _camera("cam_touchdown_wide", 32)
    for f, p in ((241, (-9.0, -14.0, gz + 1.5)), (330, (-7.5, -11.5, gz + 1.3))):
        _key(c2, "location", f, p)
    _track(c2, tgt_l)
    cams[241] = c2
    c3 = _camera("cam_bay_door", 40)
    for f, p in ((331, (-4.2, -2.6, gz + 1.2)), (470, (-4.0, -1.6, gz + 0.9))):
        _key(c3, "location", f, p)
    bay_focus = empty("bay_focus", None, (-1.2, 0.0, gz + 0.6))
    _track(c3, bay_focus)
    _dof(c3, bay_focus, 3.5)
    cams[331] = c3
    c4 = _camera("cam_rover_follow", 35)
    for f, p in ((471, (-2.2, -3.6, gz + 0.8)), (580, (-6.0, -6.8, gz + 0.9))):
        _key(c4, "location", f, p)
    _track(c4, tgt_r)
    _dof(c4, tgt_r, 2.8)
    cams[471] = c4
    c5 = _camera("cam_hero_orbit", 30)
    centre = Vector((-3.4, -1.4, gz + 0.9))
    for i, f in enumerate(range(581, 661, 8)):
        a = math.radians(-120 + 6 * i)
        _key(c5, "location", f, centre + Vector((12 * math.cos(a), 12 * math.sin(a), 2.2)))
    _track(c5, empty("hero_focus", None, tuple(centre)))
    cams[581] = c5
    for c in cams.values():
        _ease_all(c)
    _shake(c2, 292, 318, 0.05)
    for f, c in cams.items():
        mk = scene.timeline_markers.new(c.name, frame=f)
        mk.camera = c
    scene.camera = c1
    return scene


def cinematic(path, terrain_path):  # noqa: F811
    tmp = Path(path).with_name("_shot1_tmp.blend")
    clear()
    materials()
    shot_separation()
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp))
    clear()
    materials()
    shot_surface(terrain_path)
    bpy.ops.wm.append(filepath=str(tmp / "Scene" / "Shot1_Orbit_Separation_Deorbit"), directory=str(tmp) + "/Scene/", filename="Shot1_Orbit_Separation_Deorbit")
    try:
        bpy.ops.file.pack_all()                                                    # NASA textures travel inside the .blend
    except RuntimeError as e:
        print("pack_all:", e, flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    tmp.unlink(missing_ok=True)


def showcase(path):  # noqa: F811
    """Lander with the door open and the rover in its bay; the separation stack (rover inside, door closed) beside it."""
    clear()
    materials()
    ld = build_lander()
    rv = build_rover()
    for r in (ld, rv):
        repivot_all(r)
    stow_rover(ld, rv)
    ramp_pivot(ld).rotation_euler = (0, math.radians(-132.0), 0)
    ld.location = (0, 0, 1.45)
    pm = build_pm()
    l2 = build_lander("lander_separating")
    r2 = build_rover("rover_stowed")
    for r in (pm, l2, r2):
        repivot_all(r)
    stow_rover(l2, r2)
    pm.location = (9.0, 0.0, 3.0)
    l2.location = (9.0, 0.0, 3.0 + 2.45)
    cyl("regolith_disc", (0, 0, -0.05), (0, 0, 1), 0.05, 7.0, 7.0, "regolith", None, 96)
    look(bpy.context.scene)
    world_stars(bpy.context.scene, 0.5)
    sun("show", 12, -40, 5.0)
    camera((3.0, 0.0, 1.5), 16.0, 16, -55, 35)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))



def action_fcurves(act):
    try:
        return list(act.fcurves)
    except AttributeError:
        return [fc for layer in act.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]


def _actions_of(objs):
    acts = set()
    for ob in objs:
        for owner in (ob, getattr(ob, "data", None)):
            ad = getattr(owner, "animation_data", None) if owner is not None else None
            if ad and ad.action:
                acts.add(ad.action)
        for slot in getattr(ob, "material_slots", []):
            m = slot.material
            if m and m.node_tree and m.node_tree.animation_data and m.node_tree.animation_data.action:
                acts.add(m.node_tree.animation_data.action)
    return acts


def _shift_action(act, dt):
    for fc in action_fcurves(act):
        for kp in fc.keyframe_points:
            kp.co.x += dt
            kp.handle_left.x += dt
            kp.handle_right.x += dt
        for m in fc.modifiers:
            if m.use_restricted_range:
                m.frame_end += dt
                m.frame_start += dt
        fc.update()


def _has_hide_keys(ob):
    ad = ob.animation_data
    return bool(ad and ad.action and any(fc.data_path == "hide_viewport" for fc in action_fcurves(ad.action)))


def _key_hide(ob, frame, hidden):
    ob.hide_viewport = ob.hide_render = hidden
    ob.keyframe_insert("hide_viewport", frame=frame)
    ob.keyframe_insert("hide_render", frame=frame)


def cinematic(path, terrain_path):  # noqa: F811 - one continuous film: orbit -> separation -> deorbit -> landing -> rover release
    out = Path(path)
    t1, t2 = out.with_name("_shot1_tmp.blend"), out.with_name("_shot2_tmp.blend")
    clear()
    materials()
    shot_separation()
    bpy.ops.wm.save_as_mainfile(filepath=str(t1))
    clear()
    materials()
    shot_surface(terrain_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(t2))
    bpy.ops.wm.open_mainfile(filepath=str(t1))
    a = bpy.data.scenes["Shot1_Orbit_Separation_Deorbit"]
    bpy.ops.wm.append(filepath=str(t2 / "Scene" / "Shot2_Landing_Rover"), directory=str(t2) + "/Scene/", filename="Shot2_Landing_Rover")
    b = bpy.data.scenes["Shot2_Landing_Rover"]
    dt = a.frame_end                                                              # surface part starts right after the deorbit burn
    objs_a, objs_b = list(a.collection.all_objects), list(b.collection.all_objects)
    for act in _actions_of(objs_b):
        _shift_action(act, dt)
    surf = bpy.data.collections.new("Surface_Landing_Rover")
    a.collection.children.link(surf)
    for ob in objs_b:
        if ob.name not in surf.objects:
            surf.objects.link(ob)
    for ob in objs_a:
        if "cutter" in ob.name:
            continue
        if not _has_hide_keys(ob):
            _key_hide(ob, dt, False)
        _key_hide(ob, dt + 1, True)
    for ob in objs_b:
        if "cutter" in ob.name:
            continue
        _key_hide(ob, dt, True)
        _key_hide(ob, dt + 1, False)
    for ob in objs_a + objs_b:                                                    # boolean helpers: never rendered, wire in the viewport
        if "cutter" in ob.name:
            ob.hide_render = True
            ob.display_type = "WIRE"
    for m in b.timeline_markers:
        nm = a.timeline_markers.new(m.name, frame=m.frame + dt)
        nm.camera = m.camera
    a.frame_end = dt + b.frame_end
    a.name = "Mission_Orbit_Land_Rover"
    bpy.data.scenes.remove(b)
    # playback: drop frames instead of slowing down; light viewport settings (final renders keep full quality)
    a.sync_mode = "FRAME_DROP"
    try:
        a.eevee.taa_samples = 4
    except AttributeError:
        pass
    for ob in a.collection.all_objects:
        for mod in getattr(ob, "modifiers", []):
            if mod.type == "SUBSURF":
                mod.levels = 1
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:
        print("pack_all:", e, flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    t1.unlink(missing_ok=True)
    t2.unlink(missing_ok=True)
    print("MERGED frames", a.frame_start, a.frame_end, "markers", [(m.frame, m.name) for m in a.timeline_markers], flush=True)


def main():
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"provenance": "modelled (illustrative, to scale, not flight CAD; no agency logos)", "units": "metres",
                "glb_frame": "glTF Y up, vehicle front facing -Z (lander front = hazard-camera face, rover front = camera mast end)",
                "obj_frame": "simulator body frame: Z up, front +x (lander thrust +z, nozzle exits z = -1.0 m)",
                "named_parts": "wheel_FL/ML/RL/FR/MR/RR (pivot = wheel centre, spin about the body y axis); engine_centre, engine_outer0..3 (pivot = nozzle exit centre); "
                               "tilt_panel, wing_port_panel1/2, wing_starboard_panel1/2, pm_wing_* (solar panels)", "models": {}}
    # 1) each vehicle alone -> export
    if "--cinematic" in ARGS:
        cinematic(OUT / "cinematic_B.blend", TERRAIN)
        print("DONE cinematic", flush=True)
        return
    if "--showcase" in ARGS:
        showcase(OUT / "showcase_B.blend")
        print("DONE showcase", flush=True)
        return
    ui = Path(arg("--ui-dir")).resolve() if arg("--ui-dir") else None
    if ui:
        ui.mkdir(parents=True, exist_ok=True)
    for stem, ui_name, builder in (("lander_B", "lander", build_lander), ("rover_B", "rover", build_rover), ("propulsion_module_B", "propulsion_module", build_pm)):
        clear()
        materials()
        root = builder()
        repivot_all(root)
        export_obj_sim(root, OUT / f"{stem}.obj")                                 # simulator body frame (Z up, front +x) for Basilisk/Vizard
        export_glb_ui(root, OUT / f"{stem}.glb")                                  # UI convention (Y up, front -Z)
        if ui:
            export_glb_ui(root, ui / f"{ui_name}.glb")
        manifest["models"][stem] = stats(root)
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f"{stem}.blend"))
        print(stem, manifest["models"][stem], flush=True)
    # 2) attached stack (lander on the PM) -> export
    clear()
    materials()
    pm, ld = build_pm(), build_lander()
    ld.location = (0, 0, 1.25)                                                   # lander interface ring (z=-1.08 local) sits on the PM top plane
    bpy.context.view_layer.update()
    stack_root = empty("stack_attached")
    pm.parent, ld.parent = stack_root, stack_root
    repivot_all(stack_root)
    export_obj_sim(stack_root, OUT / "stack_attached_B.obj")
    export_glb_ui(stack_root, OUT / "stack_attached_B.glb")
    if ui:
        export_glb_ui(stack_root, ui / "stack_attached.glb")
    manifest["models"]["stack_attached_B"] = stats(stack_root)
    # 3) separation render: lander pushed off on the springs, Moon below
    if RENDER:
        ld.location = (0, 0, 2.45)
        world_and_lights(sun_elev=25, sun_az=-30, strength=5.0)
        moon = sphere("moon_limb", (0, 0, -1737400 - 100000), 1737400, "moon", seg=128)
        moon.location = (0, 2.0e5, -1737400 - 3.0e4)
        setup_render()
        camera((0, 0, -0.2), 13.0, 12, -38, 45)
        render(OUT / "render_separation_B.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene_separation_B.blend"))
    # 4) surface hero: lander + rover on real LOLA terrain (Site23)
    if RENDER and TERRAIN:
        clear()
        materials()
        ter, h, mpp = terrain(TERRAIN)
        ld = build_lander()
        gz = ground_z(h, mpp, 0.0, 0.0)
        ld.location = (0, 0, gz + 1.45)
        rv = build_rover()
        rx, ry = -3.2, -1.2
        rv.location = (rx, ry, ground_z(h, mpp, rx, ry) + 0.205)
        rv.rotation_euler = (0, 0, math.radians(-160))
        world_and_lights(sun_elev=6, sun_az=-20, strength=7.0)
        setup_render()
        camera((-1.0, -0.3, gz + 0.6), 11.0, 9, -60, 35)
        render(OUT / "render_surface_lander_rover_B.png")
        camera((rx, ry, ground_z(h, mpp, rx, ry) + 0.15), 2.6, 14, -25, 40)
        render(OUT / "render_rover_B.png")
        camera((0, 0, gz + 1.0), 9.5, 18, 30, 40)
        render(OUT / "render_lander_B.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene_surface_B.blend"))
        manifest["surface_scene"] = "ground shape: NASA LOLA 5 m, Site23 (observed, mostly interpolated); regolith texture artistic; Sun 6 deg high"
    (OUT / "MANIFEST_blender.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", OUT, flush=True)


main()
