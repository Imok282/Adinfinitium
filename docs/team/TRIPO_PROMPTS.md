# Tripo prompts: lander and rover (for Vizard / Navigator)

Rules for the hackathon: **no NASA or ISRO logos, flags, emblems or readable text** on the models; the files are labelled AI-generated when shown. Tripo limits and options change, so check the current prompt length and export options in the app; if a prompt is cut, use the SHORT version.

Best results: generate a **reference image first** (any image tool), then use Tripo **Image to 3D** (or multi-view if offered), and add the text prompt for detail. `vizard/assets/lander_preview.png` and `rover_preview.png` are to-scale line-ups you can use as the layout reference (proportions, leg positions, engine ring).

## LANDER (Chandrayaan-3 / Vikram class, 1.5 t, 5 engines)

### Prompt (long)
Professional lunar lander spacecraft, hard-surface sci-fi realism, Chandrayaan-3 Vikram class, photoreal PBR. Boxy octagonal-square body about 2 m wide and 1.2 m tall wrapped in gold and silver multi-layer insulation foil with visible seams and stitched corners. Top deck fully covered by a flat dark-blue solar panel with cell grid lines. Underside: FIVE bell-shaped throttleable main engines, four on an outer ring and one in the centre, each with a gimbal-less metal nozzle and heat-shield collar, nozzles pointing down. FOUR landing legs splayed outward on A-frame struts with telescoping crushable shock-absorber honeycomb cylinders and large round footpads with ribbed underside. EIGHT small attitude-control thruster nozzles, two at each upper corner, pointing outward. Visible components: propellant tank bulges under the foil, avionics panel with connectors and harness clips, laser altimeter and laser doppler velocimeter windows on the underside, two hazard-detection and avoidance camera lenses on a side face, a landing-position camera, two star-tracker baffles and sun sensors on the top rim, an S-band parabolic dish antenna on a short mast, a whip antenna, a deployable rover ramp folded flat on one side face with hinges and latches, radiator patches, ground-strap cables, red remove-before-flight tags, bolted propulsion-module interface ring around the bottom edge. Matte and metallic mix, slight dust on feet. Static pose, three-quarter view, no logos, no text, no flags, plain neutral background.

### Prompt (short, if the long one is cut)
Chandrayaan-3 class lunar lander, gold foil boxy body 2 m, blue solar panel top, 5 rocket engine nozzles underneath (4 outer + 1 centre), 4 splayed legs with crushable shock absorbers and round footpads, 8 small thrusters, cameras, laser altimeter, dish antenna, rover ramp on one side. Photoreal PBR, no logos, no text.

### Negative prompt
cartoon, toy, low poly, floating parts, extra legs, six legs, wheels, tracks, human, astronaut, flag, logo, text, letters, ground, terrain, background clutter, blurry

### Component checklist (what must be visible; what professional landers carry)
- [ ] 5 main engines (4 outer + 1 centre), downward, on a ring about 0.8 m from the centre
- [ ] 4 landing legs with crushable shock absorbers + footpads
- [ ] 8 attitude thrusters (RCS), pointing outward at the upper corners
- [ ] top solar panel (about 2 x 2 m = 4 m2)
- [ ] propellant tanks (under the foil), structural deck, MLI thermal blankets
- [ ] laser altimeter / velocimeter (underside), hazard-detection cameras, position camera
- [ ] star trackers + sun sensors
- [ ] S-band dish + whip antenna
- [ ] rover ramp with hinges
- [ ] harness, connectors, radiators, tags, interface ring

## ROVER (Pragyan class, 25 kg, 6 wheels)

### Prompt (long)
Small lunar rover, Pragyan class, 6-wheel rocker-bogie suspension, photoreal PBR, hard-surface realism. Low rectangular aluminium chassis about 0.75 m long, 0.55 m wide, with a flat dark-blue solar panel across the whole top and a gold foil thermal blanket around the sides. SIX small ribbed metal wheels with chevron grousers, three per side, joined by visible rocker and bogie arms with pivot joints and differential bar across the top. Front: a stereo pair of navigation cameras on a short mast, two hazard-avoidance cameras low on the front face, and a short deployable arm carrying an alpha-particle X-ray spectrometer and a laser-induced breakdown spectroscopy payload head, folded. Rear: a small antenna on a stub, a rear hazard camera, a deployment bracket and tie-down fittings. Visible details: bolts, cable harness, connectors, small radiator, a slightly dusty lunar-regolith finish on wheels. Static pose, three-quarter view, no logos, no text, no flags, plain neutral background.

### Prompt (short)
Small lunar rover, 6 wheels with rocker-bogie arms, blue solar panel top, gold foil sides, stereo navigation cameras on a mast, hazard cameras front and rear, folded science arm, small antenna, dusty wheels. Photoreal PBR, no logos, no text.

### Negative prompt
cartoon, toy, low poly, four wheels, tracks, tank, human, astronaut, flag, logo, text, letters, ground, terrain, blurry

### Component checklist
- [ ] 6 wheels (3 per side) with grousers, rocker-bogie arms and differential bar
- [ ] top solar panel (whole top face)
- [ ] stereo nav cameras on a mast, hazard cameras front and rear
- [ ] science arm / spectrometer head (folded)
- [ ] antenna, harness, thermal foil, deployment brackets

## After Tripo: make the file usable by the simulator (do this for both)
1. Export **GLB** (and OBJ or FBX if Vizard needs it) with PBR textures. Keep polygons reasonable (lander under about 300k triangles, rover under about 150k; ask Tripo for "low/medium poly" if offered).
2. **Scale to real metres.** Lander: body deck 2.0 m wide, overall footprint (foot to foot) about 3.6 m, height about 2.3 m. Rover: 0.90 x 0.76 x 0.43 m. Use Blender, Windows 3D Builder or any viewer's "scale" tool; check against `vizard/assets/MANIFEST.json`.
3. **Orientation and origin.** Lander: origin at the body centre, +Z up (thrust direction), engines at the bottom. Rover: origin at the body centre, +X forward, +Z up. If your tool is Y-up, rotate so the model is upright in Z-up before exporting OBJ for Vizard (the GLB stays Y-up).
4. The **five nozzle exits must sit near z = -1.0 m on a 0.8 m ring** or the Basilisk plumes will not line up with the nozzles. If Tripo's engines are elsewhere, keep the procedural model (`vizard/assets/lander.obj`) as the physics-aligned fallback and use the Tripo model for the cinematic shots.
5. Save into `vizard/assets/tripo/` as `lander_tripo.glb`, `rover_tripo.glb` (plus OBJ/MTL/textures), and write in the commit message that they are AI-generated (Tripo) so the submission can disclose it.
6. The procedural models in `vizard/assets/` stay as the always-working fallback (same rule as the rest of the project: physics backup first).
