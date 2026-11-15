// 3D figures for research/hazard-model.html (loaded by js/fig3d.js).
import { THREE, terrainMesh, loadJSON, loadHeights, loadTex } from '../fig3d.js';

const M3 = 'assets/models3d/';

// Site 23 lit by the real Sun (SPICE DE440 direction, our ray-cast shadows on the observed LOLA DEM), 16 instants over
// September 2027, as a time slider. Data: assets/models3d/ (tools/export_models3d.py), same as the home-page view.
export async function light(ctx) {
  const meta = await loadJSON(M3 + 'models3d.json'), m = meta.a, c = meta.c;
  const [h, tex] = await Promise.all([loadHeights(M3 + 'site23_height.bin', m), loadTex(M3 + c.atlas)]);
  if (ctx.dead) return;
  tex.generateMipmaps = false; tex.minFilter = THREE.LinearFilter;   // no bleeding between atlas frames
  const [gx, gy] = c.grid;
  tex.repeat.set(1 / gx, 1 / gy);
  ctx.add(terrainMesh(h, m, 1.5, new THREE.MeshBasicMaterial({ map: tex })));
  const top = (m.max_m - (m.min_m + m.max_m) / 2) * 1.5 / 1000;
  const sun = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 12), new THREE.MeshBasicMaterial({ color: 0xffd27a }));
  sun.add(new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 12), new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true, opacity: 0.25 })));
  const rayGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const ray = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: 0xffb547, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.7 }));
  // horizon ring: the Sun stays within ~5 degrees of it all month
  const ring = new THREE.Mesh(new THREE.RingGeometry(4.17, 4.2, 96), new THREE.MeshBasicMaterial({ color: 0xa8a49a, transparent: true, opacity: 0.25, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = top * 0.4;
  ctx.add(sun); ctx.add(ray); ctx.add(ring);
  ctx.legend([['#f4e2b0', 'sunlit'], ['#2c3a66', 'in shadow'], ['#ffd27a', 'Sun (true direction)']]);
  ctx.slider('t', 'Time, September 2027', {
    min: 0, max: c.frames.length - 1, value: 8, fps: 0.9,
    format: i => {
      const f = c.frames[i];
      return `${f.utc.slice(0, 16).replace('T', ' ')} UTC · Sun ${f.sun_elev_deg >= 0 ? '+' : ''}${f.sun_elev_deg.toFixed(1)}° · ${Math.round(f.lit_fraction_tile * 100)}% of tile lit`;
    },
    onInput: i => {
      const f = c.frames[i];
      tex.offset.set((i % gx) / gx, 1 - (Math.floor(i / gx) + 1) / gy);
      const az = THREE.MathUtils.degToRad(f.sun_az_deg), el = THREE.MathUtils.degToRad(f.sun_elev_deg), Rr = 4.2;
      const p = new THREE.Vector3(Math.sin(az) * Math.cos(el) * Rr, top * 0.4 + Math.sin(el) * Rr, -Math.cos(az) * Math.cos(el) * Rr);
      sun.position.copy(p);
      rayGeo.setFromPoints([p, new THREE.Vector3(0, top * 0.4, 0)]); ray.computeLineDistances();
    },
  });
  ctx.frame({ dist: 9.4, elev: 0.5, az: 0.9, targetY: 0.05 });
}

// Held-out Site 23 (the scored 5.12 km window) with three hazard maps draped on the observed LOLA terrain:
// our U-Net, the slope-rule backup and the official label; missed hazards (label unsafe, map safe) in yellow.
// Data: assets/fig3d/hazard/ (tools/export_fig3d_hazard.py), counts identical to runs/hazard/eval.json.
export async function drape(ctx) {
  const D = 'assets/fig3d/hazard/';
  const [meta, dm] = await Promise.all([loadJSON(M3 + 'models3d.json'), loadJSON(D + 'drape.json')]);
  const m = meta.a;
  const h = await loadHeights(M3 + 'site23_height.bin', m);
  if (ctx.dead) return;
  const texCache = {};
  const tex = name => (texCache[name] ||= loadTex(D + name));
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });   // hillshade is baked into the textures
  mat.map = await tex(dm.maps.unet.texture);
  if (ctx.dead) return;
  ctx.add(terrainMesh(h, m, 1.5, mat));
  ctx.onDispose(() => Object.values(texCache).forEach(p => p.then(t => t.dispose())));
  const read = ctx.readout();
  const NAMES = { unet: 'Our U-Net', backup: 'Slope rule (backup)', label: 'Official label' };
  let token = 0;
  const show = async () => {
    const k = ctx.state.map, fn = ctx.state.fn && k !== 'label';
    const s = dm.maps[k], my = ++token;
    const t = await tex(fn ? s.fn_texture : s.texture);
    if (ctx.dead || my !== token) return;
    mat.map = t; mat.needsUpdate = true; ctx.dirty();
    read.textContent = k === 'label'
      ? `Official LOLA slope ≥ ${dm.slope_limit_deg}°: ${(dm.prevalence * 100).toFixed(1)}% of the window is unsafe.`
      : `${NAMES[k]}: missed ${(s.missed_hazard_rate * 100).toFixed(1)}% of unsafe pixels (${s.missed_area_km2.toFixed(2)} km²) · F1 ${s.f1.toFixed(3)}`;
    if (fnBtn) { fnBtn.disabled = k === 'label'; fnBtn.style.opacity = k === 'label' ? '.45' : ''; }
  };
  let fnBtn = null;
  ctx.state.map ??= 'unet';
  ctx.toggle('map', 'Map', [['unet', 'U-Net'], ['backup', 'Slope rule'], ['label', 'Official label']], () => show());
  ctx.check('fn', 'Highlight missed hazards', () => show(), true);
  fnBtn = ctx.fig.querySelector('.f3d-chk');
  ctx.legend([['rgb(230,107,97)', 'unsafe (≥ 12°)'], ['rgb(92,158,128)', 'safe'], ['rgb(255,219,26)', 'missed hazard']]);
  await show();
  ctx.frame({ dist: 8.6, elev: 0.82, az: 0.35, targetY: -0.12 });
}
