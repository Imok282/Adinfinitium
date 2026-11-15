# vizard/

**Owner:** Arnav · **Jira:** KAN-47

Vizard 3D: LOLA tile → OBJ mesh, replay settings.

| | |
|---|---|
| Reads | tiles, Basilisk Vizard binaries |
| Writes | OBJ meshes, recordings |

Unity Addressables upgrade only if time allows on Sunday.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

## Vehicle models (lander + rover) — ready to use

`python -m vizard.vehicles` regenerates everything in `vizard/assets/` (numpy only, no Blender, < 2 s). The files are committed, so you do not have to run it.

| File | Use |
|---|---|
| `lander.obj` + `lander.mtl` | Vizard (Basilisk custom model), Z up, metres, origin = body centre, +z = thrust direction |
| `rover.obj` + `rover.mtl` | Vizard, x forward, y left, z up, ground at z = -0.2 m |
| `lander.glb`, `rover.glb` | web Navigator / any glTF viewer (converted to Y-up) |
| `*_preview.png`, `MANIFEST.json` | what they look like, sizes, counts, provenance |

What they are: **illustrative, to-scale** models of a Chandrayaan-3-class lander (2 m deck, 2 x 2 m solar panel = `power.panel_area_m2`, 4 legs, 5 engines = `lander.engines.count` on the 0.8 m / 5 deg ring of `basilisk_mods/lander_descent.engine_layout`, 8 attitude thrusters = `lander.attitude_thrusters.count`, ramp, dish) and a Pragyan-class 6-wheel rocker-bogie rover (0.90 x 0.76 x 0.43 m). They are **not** flight CAD; provenance MODELLED; no agency logos.

### Attach to a Basilisk run (UNTESTED here: Basilisk is not installed on the machine that built these; check the call against your bsk 2.12 install)
```python
viz = vizSupport.enableUnityVisualization(sim, "dynTask", h["sc"], saveFile=..., thrEffectorList=[h["engines"]])
vizSupport.createCustomModel(viz, simBodiesToModify=[h["sc"].ModelTag],
                             modelPath=str(Path("vizard/assets/lander.obj").resolve()), scale=[1, 1, 1])
```
If the model appears sideways in Vizard, pass `rotation=[...]` (radians) to the same call: the OBJ is in the body frame, Vizard (Unity) may want a different up axis.
Rover: same call on the rover spacecraft, `rover.obj`. Engine plumes are drawn by Vizard from the thruster list; they should sit on the five nozzle exits at z = -1.0 m.
